'use strict';
const https=require('node:https'),http=require('node:http'),fs=require('node:fs'),path=require('node:path'),os=require('node:os'),net=require('node:net');
const {X509Certificate,randomBytes,timingSafeEqual}=require('node:crypto');
const {WebSocketServer}=require('ws'),QRCode=require('qrcode');
const {Room,code}=require('./room.cjs'),{EnginePool}=require('./engine-runner.cjs'),{ServiceStore}=require('./service-store.cjs');
const {renderCertificateSetup}=require('./certificate-setup.cjs');
const ROOT=path.resolve(__dirname,'../../..'),PUBLIC=path.join(__dirname,'public');
const active=r=>!['ended','closed'].includes(r.status);
const secretEqual=(a,b)=>typeof a==='string'&&typeof b==='string'&&Buffer.byteLength(a)===Buffer.byteLength(b)&&timingSafeEqual(Buffer.from(a),Buffer.from(b));
function addresses(){return Object.entries(os.networkInterfaces()).flatMap(([name,rows])=>rows.filter(r=>r.family==='IPv4'&&!r.internal&&!r.address.startsWith('169.254.')).map(r=>({name,address:r.address})));}
function args(argv){const options={};for(let i=0;i<argv.length;i++){const k=argv[i];if(k==='--list-addresses')options.list=true;else if(k==='--online')options.online=true;else if(k==='--cloud')options.cloud=true;else if(['--address','--port','--cert-dir','--channel','--setup-port','--data-dir','--max-arenas'].includes(k))options[k.slice(2)]=argv[++i];else throw Error('未知參數 '+k);}return options;}
function cleanPublicOrigin(value){
 if(typeof value!=='string'||!value||value.trim()!==value||/[\x00-\x20\x7f]/.test(value)||!/^https:\/\/[^/?#]+\/?$/i.test(value))throw Error('公開入口需為完整 HTTPS 來源');
 let url;try{url=new URL(value);}catch{throw Error('公開入口需為完整 HTTPS 來源');}
 if(url.protocol!=='https:'||url.username||url.password||url.pathname!=='/'||url.search||url.hash)throw Error('公開入口只接受 HTTPS 網域與連接埠，不可包含帳密、路徑、查詢或片段');
 return url.origin;
}
function cleanName(value,fallback,max=36){if(value===undefined||value==='')return fallback;if(typeof value!=='string'||!value.trim()||value.trim().length>max||/[\x00-\x1f]/.test(value))throw Error('名稱需為 1～'+max+' 個字');return value.trim();}
async function start(options={}){
 const available=addresses();if(options.list){console.log(JSON.stringify(available,null,2));return;}
 const cloud=options.cloud===true,online=cloud||options.online===true;
 if(cloud&&options.online)throw Error('--cloud 與 --online 不可同時使用');
 const address=online?'127.0.0.1':options.address||(available.length===1?available[0].address:null);
 if(!address)throw Error('請用 --list-addresses 查看網卡，再用 --address 指定平板可連線的 IPv4 位址');
 if(!online&&(!net.isIPv4(address)||address.startsWith('127.')||address==='0.0.0.0'||!available.some(a=>a.address===address)))throw Error('加入網址必須使用本機實際網卡位址');
 const port=Number(options.port??(cloud?process.env.PORT||8080:8443)),maxArenas=Number(options['max-arenas']??(cloud?process.env.MAX_ARENAS||2:8));
 if(!Number.isInteger(port)||port<1024||port>65535||!Number.isInteger(maxArenas)||maxArenas<1||maxArenas>16)throw Error('連接埠需為 1024～65535，賽場上限需為 1～16');
 const setupPort=options['setup-port']?Number(options['setup-port']):0;
 if(setupPort&&(!Number.isInteger(setupPort)||setupPort<1024||setupPort>65535||setupPort===port))throw Error('憑證設定連接埠需為 1024～65535，且不可與 HTTPS 相同');
 if(options['setup-port']&&!setupPort)throw Error('憑證設定連接埠無效');
 if(online&&setupPort)throw Error('免憑證線上模式不提供 HTTP 憑證安裝入口');
 const internalOrigin=`${cloud?'http':'https'}://${address}:${port}`,tlsDir=cloud?null:path.resolve(options['cert-dir']||path.join(os.homedir(),'.webduino-lan/tls'));
 let publicOrigin=cloud?cleanPublicOrigin(options['public-origin']??(process.env.PUBLIC_ORIGIN||process.env.RENDER_EXTERNAL_URL)):online?null:internalOrigin;
 const allowedOrigin=value=>value===internalOrigin||!!publicOrigin&&value===publicOrigin;
 const setupOrigin=setupPort?`http://${address}:${setupPort}`:null;
 let cert,key;
 if(!cloud){cert=fs.readFileSync(path.join(tlsDir,'server.pem'));key=fs.readFileSync(path.join(tlsDir,'server-key.pem'));const x509=new X509Certificate(cert);if(!x509.checkIP(address)||Date.parse(x509.validTo)<Date.now())throw Error('憑證未涵蓋所選 IP 或已過期，請執行 setup-certs.py');}
 const configuredAdminToken=options['admin-token']??process.env.ADMIN_TOKEN;
 if(cloud&&!configuredAdminToken)throw Error('雲端模式需要設定至少 32 字元的 ADMIN_TOKEN');
 const store=new ServiceStore({persist:options.persist,dataDir:options.dataDir||options['data-dir']||(cloud?process.env.DATA_DIR||path.join(os.homedir(),'.webduino-lan/cloud'):undefined),maxArenas,...(cloud?{adminToken:configuredAdminToken,siteName:'Flight Club 線上賽事服務'}:{})}),adminToken=store.state.adminToken;
 if(store.state.arenas.length>maxArenas)throw Error('已保存的賽場數多於設定上限；請先在管理中心移除空閒賽場，或提高 --max-arenas');
 const arenas=store.state.arenas,roomMap=new Map(),closedCodes=new Set(store.state.closedCodes||[]),history=store.state.history||[],peers=new Set(),sessions=new Map();
 const engines=new EnginePool(internalOrigin,{channel:options.channel}),startedAt=Date.now();let siteName=store.state.siteName,closing=false,engineStarted=false,saveTimer,dirty=false;
 const send=(ws,data)=>{if(ws.readyState!==1)return;if(ws.bufferedAmount>1024*1024){ws.terminate();return;}ws.send(JSON.stringify(data));};
 const broadcast=(room,data)=>{for(const ws of peers)if(ws.room===room)send(ws,data);};
 const occupied=id=>[...roomMap.values()].find(r=>r.arenaId===id&&active(r));
 function directory(){return {siteName,origin:publicOrigin,accessMode:online?'online':'lan',publicReady:!!publicOrigin,startedAt,maxArenas,rooms:[...roomMap.values()].map(r=>r.directory()),history:history.slice(-20).reverse(),arenas:arenas.map(a=>({...a,roomCode:occupied(a.id)?.code||null})),counts:{rooms:roomMap.size,running:[...roomMap.values()].filter(r=>r.status==='running').length,players:[...roomMap.values()].reduce((n,r)=>n+[...r.players.values()].filter(p=>p.connected).length,0)}};}
 function directoryChanged(){const data={type:'directory',service:directory()};for(const ws of peers)if(ws.directory)send(ws,data);}
 function setPublicOrigin(value){
  if(!online)throw Error('只有免憑證線上模式可設定公開入口');
  if(closing)throw Error('服務正在停止，無法設定公開入口');
  const next=value===null?null:cleanPublicOrigin(value);
  if(next!==publicOrigin){publicOrigin=next;directoryChanged();}
  return publicOrigin;
 }
 function flush(){clearTimeout(saveTimer);saveTimer=null;dirty=false;store.write({version:1,adminToken,siteName,arenas,closedCodes:[...closedCodes],history:history.slice(-100),rooms:[...roomMap.values()].map(r=>r.serialize())});}
 function persist(){dirty=true;if(!saveTimer&&!closing)saveTimer=setTimeout(()=>{try{flush();}catch(e){console.error('STATE SAVE:',e.message);}},300);}
 function bind(room){
  room.on('change',()=>{broadcast(room,{type:'room',room:room.view()});persist();directoryChanged();if(room.status==='ended'&&engines.runners.has(room.code))engines.stop(room.code).catch(e=>console.error(e.message));});
  room.on('snapshot',snapshot=>broadcast(room,{type:'snapshot',value:snapshot}));
 }
 for(const data of store.state.rooms){const room=Room.restore(data);if(room.status!=='closed'){roomMap.set(room.code,room);bind(room);}}
 function lookup(value){const c=String(value||'').trim().toUpperCase();if(!/^[A-HJ-NP-Z2-9]{6}$/.test(c))throw Error('房間代碼需為 6 位英數字，請確認輸入');if(closedCodes.has(c))throw Error('房間已關閉，原代碼與 QR Code 已失效');const room=roomMap.get(c);if(!room)throw Error('房間不存在，請確認代碼與主機網址');return room;}
 const sessionValid=id=>!!id&&sessions.has(id)&&sessions.get(id)>Date.now();
 const admin=ws=>ws.adminKey===adminToken||sessionValid(ws.sessionId);
 const requireAdmin=ws=>{if(!admin(ws))throw Error('需要服務管理權限，請先登入管理中心');};
 const host=ws=>{if(ws.role!=='host'||!ws.room||!ws.roomCapability&&!admin(ws))throw Error('需要此房主持人權限');return ws.room;};
 function detach(ws){if(ws.player)throw Error('請先離開目前席位，再切換房間');ws.room=null;ws.role='visitor';ws.roomCapability=false;}
 async function closeRoom(room){
  const result={...room.directory(),status:'closed',closedAt:Date.now()};history.push(result);if(history.length>100)history.shift();
  room.close();closedCodes.add(room.code);roomMap.delete(room.code);
  for(const ws of peers)if(ws.room===room){send(ws,{type:'roomClosed',code:room.code,message:'房間已關閉，席位與加入資訊已失效'});ws.player=null;ws.room=null;ws.role='visitor';ws.roomCapability=false;}
  room.players.clear();await engines.stop(room.code);persist();directoryChanged();
 }
 async function onMessage(ws,msg){
  if(closing)throw Error('服務正在停止，請稍後重新連線');if(!msg||typeof msg.type!=='string')throw Error('無效訊息');
  if(msg.type==='directory'){ws.directory=true;send(ws,{type:'directory',service:directory()});return;}
  if(msg.type==='admin'){
   if(secretEqual(msg.token,adminToken))ws.adminKey=adminToken;requireAdmin(ws);detach(ws);ws.role='admin';ws.directory=true;send(ws,{type:'admin',authenticated:true,service:directory()});return;
  }
  if(msg.type==='service.settings'){requireAdmin(ws);siteName=cleanName(msg.name,siteName,60);persist();directoryChanged();return;}
  if(msg.type==='arena.add'||msg.type==='arena.update'||msg.type==='arena.remove'){
   requireAdmin(ws);
   if(msg.type==='arena.add'){if(arenas.length>=maxArenas)throw Error('已達此主機的賽場上限 '+maxArenas);arenas.push({id:'arena-'+randomBytes(4).toString('hex'),name:cleanName(msg.name,'新賽場'),note:cleanName(msg.note,'正式賽・獨立運算',80),enabled:true});}
   else{const arena=arenas.find(a=>a.id===msg.id);if(!arena)throw Error('賽場不存在');if(occupied(arena.id)&&(msg.type==='arena.remove'||msg.enabled===false))throw Error('此賽場仍有房間，請先結束或關閉房間');
    if(msg.type==='arena.remove')arenas.splice(arenas.indexOf(arena),1);
    else{arena.name=cleanName(msg.name,arena.name);arena.note=cleanName(msg.note,arena.note,80);if(typeof msg.enabled==='boolean')arena.enabled=msg.enabled;for(const r of roomMap.values())if(r.arenaId===arena.id){r.arenaName=arena.name;r.changed();}}
   }persist();directoryChanged();return;
  }
  if(msg.type==='admin.closeRoom'){requireAdmin(ws);await closeRoom(lookup(msg.code));return;}
  if(msg.type==='create'){
   if(secretEqual(msg.hostToken,adminToken))ws.adminKey=adminToken;requireAdmin(ws);detach(ws);
   const arena=msg.arenaId?arenas.find(a=>a.id===msg.arenaId):arenas.find(a=>a.enabled&&!occupied(a.id));
   if(!arena||!arena.enabled)throw Error('沒有可用賽場，請等待場地釋放或在管理中心新增賽場');if(occupied(arena.id))throw Error('此賽場已被其他房間占用');
   const room=new Room(code(new Set([...roomMap.keys(),...closedCodes])),msg.roster);room.name=cleanName(msg.name,'正式賽 '+room.code);room.arenaId=arena.id;room.arenaName=arena.name;
   room.difficulty=['easy','medium','hard'].includes(msg.difficulty)?msg.difficulty:'medium';room.breakSec=msg.breakSec===180?180:120;
   roomMap.set(room.code,room);bind(room);ws.room=room;ws.role='host';send(ws,{type:'created',room:room.view(),hostToken:room.hostToken,authKind:ws.sessionId?'session':'key'});persist();directoryChanged();await engines.load(room);return;
  }
  if(msg.type==='host'){
   const room=lookup(msg.code);if(secretEqual(msg.hostToken,adminToken))ws.adminKey=adminToken;
   const capability=secretEqual(msg.hostToken,room.hostToken);if(!capability&&!admin(ws))throw Error('主持人管理權限無效，請重新登入');detach(ws);ws.room=room;ws.role='host';ws.roomCapability=capability;send(ws,{type:'room',room:room.view()});if(room.snapshot)send(ws,{type:'snapshot',value:room.snapshot});return;
  }
  if(msg.type==='inspect'||msg.type==='watch'){const room=lookup(msg.code);detach(ws);ws.room=room;ws.role='viewer';send(ws,{type:'room',room:room.view()});if(room.snapshot)send(ws,{type:'snapshot',value:room.snapshot});return;}
  if(msg.type==='join'){
   if(ws.player)throw Error('一台連線只能綁定一個席位，請先離房');const room=lookup(msg.code);
   if(!msg.resume&&[...roomMap.values()].some(r=>r!==room&&[...r.players.values()].some(p=>p.device===msg.device)))throw Error('此平板已綁定另一個房間的席位，請先離開原房間');
   const p=room.join(msg,ws);ws.room=room;ws.role='player';ws.player=p;send(ws,{type:'joined',id:p.id,seat:p.seat,resume:p.resume,session:p.session,room:room.view()});if(room.snapshot)send(ws,{type:'snapshot',value:room.snapshot});return;
  }
  if(msg.type==='leave'){
   const room=ws.room;if(ws.player)room.leave(ws.player,ws);ws.player=null;detach(ws);send(ws,{type:'left',code:room?.code||null});return;
  }
  if(msg.type==='input'){if(ws.role!=='player'||!ws.room)throw Error('請先加入真人席位');ws.room.input(ws.player,ws,msg);return;}
  if(msg.type==='ack'){if(ws.role!=='player')return;const p=ws.player,room=ws.room;if(p.ws!==ws||!room.players.has(p.id))throw Error('席位已失效');if(msg.camera===room.camera.revision)p.cameraAck=msg.camera;if(msg.view===room.engineRevision)p.viewAck=msg.view;room.changed();return;}
  if(msg.type==='ready'){if(ws.role!=='player')throw Error('請先加入真人席位');ws.room.ready(ws.player,{ws});return;}
  const room=host(ws);
  if(msg.type==='start')room.start();else if(msg.type==='pause')room.pause();else if(msg.type==='resume')room.resume();else if(msg.type==='end')room.finish();
  else if(msg.type==='camera')room.setCamera(msg.camera);
  else if(msg.type==='broadcast'){if(!['all','left','right','center','follow'].includes(msg.mode)||!room.seats.some(s=>s.id===msg.target))throw Error('轉播鏡頭無效');room.broadcastCamera={mode:msg.mode,target:msg.target};room.changed();}
  else if(msg.type==='lock'){room.locked=!!msg.locked;room.changed();}
  else if(msg.type==='remove')room.remove(msg.id);else if(msg.type==='replaceAI')room.replaceAI(msg.seat);
  else if(msg.type==='recover'){const checkpoint=room.prepareRecovery();await engines.load(room,checkpoint);}
  else if(msg.type==='roster'){room.setRoster(msg.roster);await engines.load(room);}
  else if(msg.type==='rename'){room.name=cleanName(msg.name,room.name);room.changed();}
  else if(msg.type==='close')await closeRoom(room);else throw Error('不支援的操作');
 }
 const cookieId=req=>{const entry=String(req.headers.cookie||'').split(';').map(v=>v.trim()).find(v=>v.startsWith('lan_session='));return entry?.slice(12)||'';};
 const json=(res,status,data)=>{res.writeHead(status,{'Content-Type':'application/json; charset=utf-8'});res.end(JSON.stringify(data));};
 const readJSON=req=>new Promise((resolve,reject)=>{let data='';req.on('data',d=>{data+=d;if(data.length>8192){reject(Error('資料過大'));req.destroy();}});req.on('end',()=>{try{resolve(JSON.parse(data||'{}'));}catch(e){reject(e);}});req.on('error',reject);});
 const mime={'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8','.svg':'image/svg+xml','.png':'image/png','.jpg':'image/jpeg','.woff2':'font/woff2','.json':'application/json','.md':'text/plain; charset=utf-8'};
 const handler=async(req,res)=>{
  try{
   const u=new URL(req.url,internalOrigin);res.setHeader('Cache-Control','no-store');res.setHeader('X-Content-Type-Options','nosniff');res.setHeader('Referrer-Policy','no-referrer');res.setHeader('Permissions-Policy','gamepad=(self), fullscreen=(self)');
   res.setHeader('Content-Security-Policy',"default-src 'self' data: blob:; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; font-src 'self' data:; frame-ancestors 'self'");
   if(u.pathname==='/api/login'&&req.method==='POST'){
    if(!allowedOrigin(req.headers.origin))return json(res,403,{error:'不接受跨來源登入'});const data=await readJSON(req);if(!secretEqual(data.key,adminToken))return json(res,401,{error:'管理金鑰不正確'});
    const id=randomBytes(32).toString('base64url');sessions.set(id,Date.now()+12*60*60*1000);res.setHeader('Set-Cookie',`lan_session=${id}; HttpOnly; Secure; SameSite=Strict; Path=/; Max-Age=43200`);return json(res,200,{authenticated:true});
   }
   if(u.pathname==='/api/session')return json(res,200,{authenticated:sessionValid(cookieId(req))});
   if(u.pathname==='/api/logout'&&req.method==='POST'){
    if(!allowedOrigin(req.headers.origin))return json(res,403,{error:'不接受跨來源登出'});const id=cookieId(req);sessions.delete(id);res.setHeader('Set-Cookie','lan_session=; HttpOnly; Secure; SameSite=Strict; Path=/; Max-Age=0');
    for(const ws of peers)if(ws.sessionId===id&&['host','admin'].includes(ws.role)&&!ws.adminKey&&!ws.roomCapability){send(ws,{type:'signedOut'});ws.room=null;ws.role='visitor';ws.directory=false;}return json(res,200,{authenticated:false});
   }
   if(u.pathname==='/api/config')return json(res,200,{origin:publicOrigin,joinURL:publicOrigin?publicOrigin+'/play':null,setupURL:setupOrigin?setupOrigin+'/':null,invitationURL:publicOrigin?(setupOrigin?setupOrigin+'/':publicOrigin+'/play'):null,...(!online?{addresses:available}:{}),accessMode:online?'online':'lan',publicReady:!!publicOrigin,directJoin:online&&!!publicOrigin,certificateTrustRequired:!online,secure:true,siteName,serviceURL:publicOrigin?publicOrigin+'/':null,adminURL:publicOrigin?publicOrigin+'/admin':null});
   if(u.pathname==='/api/rooms')return json(res,200,directory());
   if(u.pathname==='/api/health'){const status=closing?'stopping':engineStarted&&engines.browser?.isConnected()?'ok':'starting';return json(res,status==='ok'?200:503,{status,startedAt,rooms:roomMap.size,running:directory().counts.running});}
   if(u.pathname==='/api/qr'){if(!publicOrigin)return json(res,503,{error:'免憑證公開入口尚未就緒，請等待主機完成連線'});const room=lookup(u.searchParams.get('code')),invitation=setupOrigin?`${setupOrigin}/?code=${room.code}`:`${publicOrigin}/play?code=${room.code}`;res.writeHead(200,{'Content-Type':'image/svg+xml'});res.end(await QRCode.toString(invitation,{type:'svg',errorCorrectionLevel:'M',margin:3,width:280}));return;}
   const aliases={'/':'service.html','/admin':'admin.html','/host':'host.html','/play':'player.html','/watch':'watch.html','/lan/app.js':'app.js','/lan/style.css':'style.css','/lan/service.js':'service.js','/lan/service.css':'service.css','/lan/pro.css':'pro.css','/lan/match-theme.css':'match-theme.css','/lan/admin.js':'admin.js','/lan/panels.js':'panels.js','/lan/panel-system.css':'panel-system.css'};
   let file;if(aliases[u.pathname])file=path.join(PUBLIC,aliases[u.pathname]);
   else if(cloud&&u.pathname==='/apps/drone-soccer/lan/setup.html')file=path.join(__dirname,'../cloud/help.html');
   else if(['/apps/drone-soccer/lan/setup.html','/apps/drone-soccer/lan/README.md'].includes(u.pathname))file=path.join(__dirname,path.basename(u.pathname));
   else if(u.pathname.startsWith('/apps/drone-soccer/assets/')||u.pathname.startsWith('/vendor/')||/^\/apps\/drone-soccer\/games\/(keyboard|radio)\/[a-z0-9_]+\.html$/.test(u.pathname)||['/apps/drone-soccer/','/apps/drone-soccer/index.html'].includes(u.pathname)){
    const relative=decodeURIComponent(u.pathname).replace(/^\//,'');if(relative.split(/[\\/]/).some(p=>p==='..'||p.startsWith('.')))throw Error('路徑無效');file=path.resolve(ROOT,relative);if(u.pathname.endsWith('/'))file=path.join(file,'index.html');if(!file.startsWith(ROOT+path.sep)||file.includes(`${path.sep}node_modules${path.sep}`))throw Error('路徑無效');
   }else{res.writeHead(404);res.end('Not found');return;}
   if(!fs.existsSync(file)||!fs.statSync(file).isFile()){res.writeHead(404);res.end('Not found');return;}res.writeHead(200,{'Content-Type':mime[path.extname(file)]||'application/octet-stream'});fs.createReadStream(file).pipe(res);
  }catch(error){if(!res.headersSent)json(res,400,{error:error.message});else res.end();}
 };
 // Only --cloud accepts plain HTTP, behind an external HTTPS/WSS proxy. Public
 // URLs and Origin checks come from configuration, never forwarding headers.
 const server=cloud?http.createServer(handler):https.createServer({cert,key,minVersion:'TLSv1.2'},handler);
 const wss=new WebSocketServer({noServer:true,maxPayload:16384,perMessageDeflate:false});
 server.on('upgrade',(req,socket,head)=>{if(req.url!=='/socket'||!allowedOrigin(req.headers.origin)){socket.destroy();return;}wss.handleUpgrade(req,socket,head,ws=>{ws.sessionId=cookieId(req);wss.emit('connection',ws,req);});});
 wss.on('connection',ws=>{peers.add(ws);ws.messages=0;ws.window=Date.now();ws.alive=true;ws.on('pong',()=>ws.alive=true);
  ws.on('message',raw=>{if(Date.now()-ws.window>1000){ws.messages=0;ws.window=Date.now();}if(++ws.messages>100){ws.close(1008,'Rate limit');return;}let msg;try{msg=JSON.parse(raw.toString());}catch{send(ws,{type:'error',message:'無效 JSON'});return;}onMessage(ws,msg).catch(e=>send(ws,{type:'error',message:e.message,request:msg.type}));});
  ws.on('close',()=>{peers.delete(ws);if(ws.player&&ws.room)ws.room.disconnect(ws.player,ws);});ws.on('error',()=>{});
 });
 await new Promise((resolve,reject)=>{server.once('error',reject);server.listen(port,cloud?'0.0.0.0':online?'127.0.0.1':'0.0.0.0',resolve);});
 try{await engines.launch();engineStarted=true;}catch(e){for(const ws of peers)ws.terminate();await engines.close().catch(()=>{});await new Promise(r=>server.close(r));throw Error('無法啟動 Chromium，請安裝瀏覽器或指定 --channel chrome。'+e.message);}
 for(const room of roomMap.values())if(active(room))engines.load(room,room.command>0?room.checkpoint:null);
 const ticker=setInterval(()=>{for(const room of roomMap.values()){room.tick();broadcast(room,room.issueTicket());}},100);
 const stateTicker=setInterval(()=>{for(const room of roomMap.values())broadcast(room,{type:'room',room:room.view()});},500);
 const directoryTicker=setInterval(directoryChanged,1000);
 const checkpointTicker=setInterval(()=>{if(dirty||[...roomMap.values()].some(r=>r.status==='running'))try{flush();}catch(e){console.error(e.message);}},2000);
 const pinger=setInterval(()=>{for(const ws of peers){if(!ws.alive)ws.terminate();else{ws.alive=false;ws.ping();}}for(const [id,expires]of sessions)if(expires<Date.now())sessions.delete(id);},5000);
 let setupServer=null;
 const close=async()=>{if(closing)return;closing=true;clearTimeout(saveTimer);for(const timer of [ticker,stateTicker,directoryTicker,checkpointTicker,pinger])clearInterval(timer);for(const room of roomMap.values())room.pause('伺服器正在重新啟動，請稍後重連');flush();for(const ws of peers){send(ws,{type:'serverClosing'});ws.terminate();}await engines.close();await Promise.all([new Promise(r=>server.close(r)),setupServer?new Promise(r=>setupServer.close(r)):Promise.resolve()]);};
 if(setupPort){
  const fingerprint=new X509Certificate(fs.readFileSync(path.join(tlsDir,'ca.pem'))).fingerprint256;
  setupServer=http.createServer((req,res)=>{
   res.setHeader('Cache-Control','no-store');res.setHeader('X-Content-Type-Options','nosniff');res.setHeader('Referrer-Policy','no-referrer');
   if(req.method!=='GET'){res.writeHead(405,{'Allow':'GET'});res.end('Method not allowed');return;}
   let u;try{u=new URL(req.url,setupOrigin);}catch{res.writeHead(400);res.end('Invalid URL');return;}
   if(u.pathname==='/ca.crt'){res.writeHead(200,{'Content-Type':'application/x-x509-ca-cert','Content-Disposition':'attachment; filename="webduino-lan-ca.crt"'});fs.createReadStream(path.join(tlsDir,'ca.pem')).pipe(res);return;}
   if(!['/','/setup'].includes(u.pathname)){res.writeHead(404);res.end('Not found');return;}
   const candidate=(u.searchParams.get('code')||'').toUpperCase(),match=/^[A-HJ-NP-Z2-9]{6}$/.test(candidate)?roomMap.get(candidate):null;
   res.setHeader('Content-Security-Policy',`default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src ${internalOrigin}; base-uri 'none'; form-action 'none'; frame-ancestors 'none'`);
   res.writeHead(200,{'Content-Type':'text/html; charset=utf-8'});res.end(renderCertificateSetup({origin:internalOrigin,code:match&&match.status!=='closed'?match.code:'',fingerprint}));
  });
  try{await new Promise((resolve,reject)=>{setupServer.once('error',reject);setupServer.listen(setupPort,'0.0.0.0',resolve);});}catch(e){await close();throw e;}
 }
 console.log(`\nFlight Club 多賽場服務\n網卡：${cloud?'雲端代理服務':online?'本機私有連線':available.find(a=>a.address===address).name}\n服務大廳：${publicOrigin?publicOrigin+'/':'正在等待免憑證公開入口'}\n服務管理（勿分享）：${publicOrigin?publicOrigin+'/admin'+(cloud?'':'#admin='+adminToken):'由啟動器在公開入口就緒後提供'}\n資料目錄：${store.file?path.dirname(store.file):'測試用記憶體'}\nCtrl+C 安全停止，進行中的比賽將保存為暫停狀態。`);
 persist();
 return {internalOrigin,get origin(){return publicOrigin;},setPublicOrigin,adminToken,engines,roomMap,arenas,directory,store,close,server,get engine(){return engines.runners.values().next().value;}};
}
if(require.main===module)start(args(process.argv.slice(2))).then(service=>{if(service)for(const sig of ['SIGINT','SIGTERM'])process.once(sig,()=>service.close().then(()=>process.exit(0)));}).catch(e=>{console.error(e.message);process.exitCode=1;});
module.exports={start,addresses};
