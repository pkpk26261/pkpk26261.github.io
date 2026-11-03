'use strict';
const { randomBytes, randomInt } = require('node:crypto');
const { EventEmitter } = require('node:events');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
// Use the same seat/role rules as the existing lobby and official engine.
const context = { window: {} };
vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../assets/game/match-roster.js'), 'utf8'), context);
const rosterAPI = context.window.DroneMatchRoster;
const clone = v => JSON.parse(JSON.stringify(v));
const token = () => randomBytes(24).toString('base64url');
const alphabet = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
function code(existing) {
  let value;
  do { value = Array.from({ length: 6 }, () => alphabet[randomInt(alphabet.length)]).join(''); } while (existing.has(value));
  return value;
}
const neutral = input => ['lx', 'ly', 'rx', 'ry'].every(k => Math.abs(input[k]) <= .12) && !input.buttons.some(v => v > .35);
function cleanInput(v) {
  if (!v || !['lx', 'ly', 'rx', 'ry'].every(k => typeof v[k] === 'number' && Number.isFinite(v[k]) && Math.abs(v[k]) <= 1) ||
      !Array.isArray(v.buttons) || v.buttons.length > 64 || !v.buttons.every(n => typeof n === 'number' && Number.isFinite(n) && n >= 0 && n <= 1)) throw Error('無效的搖桿資料');
  return { lx:v.lx, ly:v.ly, rx:v.rx, ry:v.ry, buttons:v.buttons.slice(), gear:Number.isInteger(v.gear) && v.gear >= 0 && v.gear < 3 ? v.gear : 0 };
}
const idle = () => ({ connected:false, lx:0, ly:0, rx:0, ry:0 });
class Room extends EventEmitter {
  constructor(id, roster, options = {}) {
    super();
    this.code = id; this.hostToken = token(); this.status = 'lobby'; this.reason = '等待玩家準備';
    this.epoch = 1; this.engineRevision = 1; this.engineReady = false; this.snapshot = null; this.lastSnapshot = 0;
    this.locked = false; this.players = new Map(); this.tickets = new Map(); this.now = options.now || Date.now;
    this.camera = { revision:1, mode:'center', fov:60, distance:2.6, height:1.65, rotation:'team-end', bob:false, aspect:16/9 };
    this.broadcastCamera = { mode:'all', target:'' };
    this.breakSec = 120; this.difficulty = 'medium'; this.command = 0;
    this.name='正式賽房間';this.arenaId='';this.arenaName='';this.createdAt=this.now();this.endedAt=null;
    this.setRoster(roster);
  }
  setRoster(roster) {
    if (this.status !== 'lobby') throw Error('編成只能在開賽前修改；暫停後可將個別真人席位替換為 AI');
    if (!rosterAPI.valid(roster)) throw Error('每隊合計必須為 1～3 台');
    const defs = clone(rosterAPI.definitions(roster));
    for (const p of [...this.players.values()]) if (!defs.some(d => d.id === p.seat && !d.ai)) this.remove(p.id, false);
    this.roster = clone(roster); this.seats = defs;
    this.engineRevision++; this.engineReady = false; this.snapshot = null;
    for (const p of this.players.values()) { p.ready = false; p.viewAck = 0; }
    this.changed();
  }
  changed() { this.emit('change'); }
  view() {
    return {
      code:this.code, name:this.name, arenaId:this.arenaId, arenaName:this.arenaName, createdAt:this.createdAt, endedAt:this.endedAt, status:this.status, reason:this.reason, locked:this.locked, epoch:this.epoch,
      engineRevision:this.engineRevision, engineReady:this.engineReady, recoverable:!this.engineReady&&this.status==='paused'&&!!this.checkpoint, camera:clone(this.camera), roster:clone(this.roster),
      breakSec:this.breakSec, difficulty:this.difficulty, broadcastCamera:clone(this.broadcastCamera), canStart:this.conditions(),
      seats:this.seats.map(s => ({ id:s.id, team:s.team, role:s.role, ai:s.ai, label:this.label(s),
        player: (() => { const p = [...this.players.values()].find(p => p.seat === s.id); return p ? {
          id:p.id, name:p.name, connected:p.connected, pad:p.pad, ready:p.ready,
          cameraSynced:p.cameraAck === this.camera.revision, viewSynced:p.viewAck === this.engineRevision
        } : null; })() }))
    };
  }
  label(s) { return `${s.team === 'red' ? '紅隊' : '藍隊'} ${s.id.endsWith('atk') ? 1 : Number(s.id.slice(-1))+1} 號・${s.role === 'attacker' ? '前鋒' : '防守'}`; }
  join({ name, seat, resume, device }, ws) {
    if (this.status === 'closed') throw Error('房間已關閉，原代碼與 QR Code 已失效');
    if (resume) {
      const p = [...this.players.values()].find(p => p.resume === resume);
      if (!p) throw Error('原連線已失效或已被主持人移除，請重新選擇席位');
      this.pause('原玩家正在恢復連線，請回中並重新準備');
      if (p.ws && p.ws !== ws) { p.ws.send(JSON.stringify({ type:'replaced', message:'此席位已由原玩家重新連線' })); p.ws.close(); }
      p.ws = ws; p.connected = true; p.ready = false; p.pad = false; p.seq = -1; p.lastInput = 0; p.input = null;
      p.cameraAck = 0; p.viewAck = 0; p.session = token(); this.changed(); return p;
    }
    if (this.locked || !['lobby','paused'].includes(this.status)) throw Error('房間已鎖定加入；原玩家可使用原瀏覽器恢復席位');
    if (typeof device !== 'string' || !/^[a-zA-Z0-9_-]{16,80}$/.test(device)) throw Error('平板識別資料無效，請重新整理頁面');
    if ([...this.players.values()].some(p => p.device === device)) throw Error('此瀏覽器已綁定一個真人席位，請使用原玩家頁面或請主持人釋放席位');
    if (typeof name !== 'string' || !name.trim() || name.trim().length > 24) throw Error('暱稱需為 1～24 個字');
    if (!this.seats.some(s => s.id === seat && !s.ai)) throw Error('請選擇真人席位，AI 席位無法加入');
    if ([...this.players.values()].some(p => p.seat === seat)) throw Error('席位已被占用，請選擇另一個席位');
    const p = { id:token(), resume:token(), session:token(), device, name:name.trim(), seat, ws, connected:true,
      pad:false, ready:false, cameraAck:0, viewAck:0, input:null, seq:-1, lastInput:0 };
    this.players.set(p.id, p); this.changed(); return p;
  }
  disconnect(p, ws) {
    if (p.ws !== ws || !this.players.has(p.id)) return;
    p.connected = false; p.pad = false; p.ready = false; p.input = null; p.ws = null;
    this.pause(`${this.label(this.seats.find(s => s.id === p.seat))}連線中斷`); this.changed();
  }
  remove(id, publish = true) {
    const p = this.players.get(id); if (!p) throw Error('玩家不存在');
    this.players.delete(id); p.input = null;
    if (p.ws) { p.ws.send(JSON.stringify({ type:'removed', message:'主持人已釋放此席位，原連線憑證失效' })); p.ws.close(); }
    this.pause('主持人移除玩家，請補足席位後重新準備'); if (publish) this.changed();
  }
  leave(p,ws) {
    if(!this.players.has(p.id)||p.ws!==ws)throw Error('席位已失效');
    this.players.delete(p.id);p.ws=null;p.connected=false;p.ready=false;p.input=null;
    this.pause(`${p.name}已離開房間，席位已釋放`);this.changed();
  }
  replaceAI(seat) {
    if (this.status !== 'paused') throw Error('請先暫停比賽，再明確替換 AI');
    const s = this.seats.find(s => s.id === seat && !s.ai); if (!s) throw Error('真人席位不存在');
    const p = [...this.players.values()].find(p => p.seat === seat); if (p) this.remove(p.id, false);
    s.ai = true; this.roster[s.team].humans--; this.roster[s.team].ai++; this.changed();
  }
  conditions() {
    return this.engineReady && this.seats.filter(s => !s.ai).every(s => {
      const p = [...this.players.values()].find(p => p.seat === s.id);
      return p && p.connected && p.pad && p.ready && p.cameraAck === this.camera.revision && p.viewAck === this.engineRevision && this.now() - p.lastInput < 600;
    });
  }
  newEpoch() { this.epoch++; for (const p of this.players.values()) { p.input = null; p.ready = false; } }
  start() {
    if (this.status !== 'lobby') throw Error('比賽已開始');
    if (!this.conditions()) throw Error('所有真人必須加入、搖桿正常、視角同步並準備完成，且比賽引擎已就緒');
    this.newEpoch(); this.status = 'running'; this.reason = ''; this.command++;
    // Start/resume invalidates queued inputs; healthy players remain ready.
    for (const p of this.players.values()) p.ready = true;
    this.changed();
  }
  pause(reason = '主持人暫停') {
    if (this.status !== 'running') return;
    this.newEpoch(); this.status = 'paused'; this.reason = reason; this.changed();
  }
  resume() {
    if (this.status !== 'paused') throw Error('比賽尚未暫停');
    if (!this.conditions()) throw Error('請等待全部玩家回中、重新準備並同步視角');
    this.newEpoch(); this.status = 'running'; this.reason = ''; for (const p of this.players.values()) p.ready = true; this.changed();
  }
  setCamera(values) {
    if (!['lobby','paused'].includes(this.status)) throw Error('比賽中鏡頭已鎖定，必須先暫停');
    if (!['center','left','right'].includes(values.mode) || values.bob===true ||
      ![['fov',40,80],['distance',.6,3],['height',1.2,2.4]].every(([k,min,max]) => typeof values[k] === 'number' && Number.isFinite(values[k]) && values[k] >= min && values[k] <= max)) throw Error('請選擇己方短邊的固定中間、左邊或右邊站位，並確認鏡頭參數');
    this.camera = { ...this.camera, ...Object.fromEntries(['mode','fov','distance','height'].map(k => [k,values[k]])),rotation:'team-end',bob:false,aspect:16/9, revision:this.camera.revision+1 };
    for (const p of this.players.values()) { p.cameraAck = 0; p.ready = false; } this.changed();
  }
  issueTicket() {
    const value = token(), now = this.now(); this.tickets.set(value,now);
    for (const [k,t] of this.tickets) if (now-t > 450) this.tickets.delete(k);
    return { type:'tick', ticket:value, epoch:this.epoch, at:now };
  }
  input(p, ws, msg) {
    if(['closed','ended'].includes(this.status))return false;
    if (!this.players.has(p.id) || p.ws !== ws || msg.session !== p.session) throw Error('玩家連線已失效');
    const at = this.tickets.get(msg.ticket);
    if (at === undefined || this.now()-at > 400 || msg.epoch !== this.epoch || !Number.isSafeInteger(msg.seq) || msg.seq <= p.seq) return false;
    const input = cleanInput(msg.input); p.seq = msg.seq; p.lastInput = this.now();
    const healthy = msg.pad === true && msg.visible === true;
    p.pad = healthy; p.input = healthy ? input : null;
    if (!healthy) { const changed=p.ready||this.status==='running';p.ready = false; this.pause('玩家搖桿中斷或平板進入背景'); if(changed)this.changed(); }
    return true;
  }
  ready(p, msg) {
    if(!['lobby','paused'].includes(this.status))throw Error('目前不能準備，請等待主持人暫停或建立下一場比賽');
    if (p.ws !== msg.ws || !this.players.has(p.id)) throw Error('席位已失效');
    if (!p.connected || !p.pad || !p.input || !neutral(p.input) || this.now()-p.lastInput >= 400 ||
      p.cameraAck !== this.camera.revision || p.viewAck !== this.engineRevision) throw Error('請放開全部按鈕、雙桿回中，等待 3D 視角同步後再準備');
    p.ready = true; this.changed();
  }
  tick() {
    for (const p of this.players.values()) if (p.ready && this.now()-p.lastInput >= 600) {
      p.ready = false; p.pad = false; p.input = null; this.pause('玩家輸入逾時；已清除舊操作'); this.changed();
    }
    if (!['ended','closed'].includes(this.status)&&this.engineReady && this.now()-this.lastSnapshot > 1000) this.failEngine('權威運算中斷；請由主持人復原引擎');
  }
  failEngine(reason) { this.engineReady=false; this.pause(reason); this.reason=reason; this.changed(); }
  prepareRecovery() {
    if(this.engineReady||this.status!=='paused'||!this.checkpoint)throw Error('目前沒有可復原的引擎中斷狀態');
    const checkpoint=clone(this.checkpoint);this.newEpoch();this.engineRevision++;this.snapshot=null;
    for(const p of this.players.values()){p.viewAck=0;p.cameraAck=0;}
    this.reason='正在從最後同步狀態復原；接觸與過門暫存會重置，請等待全員重新準備';this.changed();return checkpoint;
  }
  controls() {
    return { epoch:this.epoch, engineRevision:this.engineRevision, status:this.status, command:this.command, breakSec:this.breakSec, difficulty:this.difficulty,
      seats:this.seats.map(s => ({ id:s.id, ai:s.ai })), inputs:Object.fromEntries([...this.players.values()].map(p => [p.seat,
        this.status === 'running' && p.ready && p.pad && this.now()-p.lastInput < 600 && p.input ? { connected:true,...p.input } : idle()])) };
  }
  receiveSnapshot(snapshot) {
    if (snapshot.epoch !== this.epoch || snapshot.engineRevision !== this.engineRevision || this.status === 'closed') return;
    this.lastSnapshot = this.now(); const wasReady = this.engineReady; this.engineReady = true;
    const { checkpoint, ...presentation } = snapshot;
    if(checkpoint)this.checkpoint=checkpoint;
    this.snapshot = { ...presentation, serverAt:this.now() };
    if (snapshot.failure && this.status === 'running') this.pause(snapshot.failure);
    if (snapshot.match.phase === 'final' && this.status === 'running') { this.status = 'ended';this.endedAt=this.now(); this.reason = snapshot.match.winReason || '比賽結束'; this.newEpoch();this.snapshot.epoch=this.epoch; this.changed(); }
    this.emit('snapshot', this.snapshot); if (!wasReady) this.changed();
  }
  close() { this.status = 'closed'; this.reason = '房間已關閉，原代碼與 QR Code 已失效'; this.newEpoch(); this.changed(); }
  finish(reason='主持人結束本場比賽') {
    if(!['running','paused'].includes(this.status))throw Error('比賽尚未開始或已經結束');
    this.status='ended';this.reason=reason;this.endedAt=this.now();this.newEpoch();
    if(this.snapshot){this.snapshot={...this.snapshot,epoch:this.epoch,match:{...this.snapshot.match,phase:'final',winReason:reason}};this.emit('snapshot',this.snapshot);}
    this.changed();
  }
  directory() {
    const human=this.seats.filter(s=>!s.ai),joined=human.filter(s=>[...this.players.values()].some(p=>p.seat===s.id&&p.connected)).length;
    return {code:this.code,name:this.name,arenaId:this.arenaId,arenaName:this.arenaName,status:this.status,reason:this.reason,
      locked:this.locked,humans:human.length,joined,ready:[...this.players.values()].filter(p=>p.ready).length,
      available:human.length-this.players.size,ai:this.seats.filter(s=>s.ai).length,roster:clone(this.roster),
      scores:this.snapshot?.scores||{red:0,blue:0},match:this.snapshot?.match||null,engineReady:this.engineReady,
      createdAt:this.createdAt,endedAt:this.endedAt,canJoin:['lobby','paused'].includes(this.status)&&!this.locked&&human.length>this.players.size};
  }
  serialize() {
    return {code:this.code,name:this.name,arenaId:this.arenaId,arenaName:this.arenaName,createdAt:this.createdAt,endedAt:this.endedAt,
      hostToken:this.hostToken,roster:this.roster,seats:this.seats,status:this.status,reason:this.reason,locked:this.locked,
      camera:this.camera,broadcastCamera:this.broadcastCamera,breakSec:this.breakSec,difficulty:this.difficulty,
      epoch:this.epoch,engineRevision:this.engineRevision,command:this.command,checkpoint:this.checkpoint,snapshot:this.snapshot,
      players:[...this.players.values()].map(p=>({id:p.id,resume:p.resume,device:p.device,name:p.name,seat:p.seat}))};
  }
  static restore(data) {
    const room=new Room(data.code,data.roster);
    for(const key of ['name','arenaId','arenaName','createdAt','endedAt','hostToken','locked','camera','broadcastCamera','breakSec','difficulty','command','checkpoint','snapshot'])if(data[key]!==undefined)room[key]=clone(data[key]);
    if(!['center','left','right'].includes(room.camera.mode))room.camera={...room.camera,mode:'center',distance:2.6,height:1.65,rotation:'team-end',bob:false,revision:room.camera.revision+1};
    if(Array.isArray(data.seats))room.seats.forEach(s=>{const saved=data.seats.find(v=>v.id===s.id);if(saved)s.ai=!!saved.ai;});
    room.epoch=(data.epoch||1)+1;room.engineRevision=(data.engineRevision||1)+1;
    room.status=data.status==='ended'?'ended':data.command>0?'paused':'lobby';
    room.reason=room.status==='paused'?'伺服器已重新啟動，請等待玩家重連、同步並重新準備後恢復':data.reason||'等待玩家準備';
    for(const saved of data.players||[])if(room.seats.some(s=>s.id===saved.seat&&!s.ai))room.players.set(saved.id,{...saved,ws:null,connected:false,pad:false,ready:false,cameraAck:0,viewAck:0,input:null,seq:-1,lastInput:0,session:token()});
    if(room.snapshot){room.snapshot.epoch=room.epoch;room.snapshot.engineRevision=room.engineRevision;}
    if(room.status==='ended')room.engineReady=true;
    return room;
  }
}
module.exports = { Room, code, cleanInput, neutral, rosterAPI };
