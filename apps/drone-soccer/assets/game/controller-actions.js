/* Shared gamepad actions. Standard Gamepad API button order is the default. */
(() => {
  'use strict';
  const definitions = Object.freeze([
    {id:'gear',label:'LB · 切換檔位',hint:'按一下切換低速／中速／高速',button:4},
    {id:'backflip',label:'RB · 後空翻',hint:'空中按一下，完成一次後空翻',button:5},
    {id:'cameraLeft',label:'LT · 視角往左移',hint:'按住持續將視角往左移',button:6},
    {id:'cameraRight',label:'RT · 視角往右移',hint:'按住持續將視角往右移',button:7},
    {id:'headless',label:'A · 無頭模式',hint:'按一下開關，以開啟時的機頭方向為前進基準',button:0},
    {id:'brake',label:'B · 煞車／懸停',hint:'按住停止方向指令並煞車，放開恢復操控',button:1}
  ]);
  const extraIds = definitions.filter(d=>d.id!=='gear').map(d=>d.id);
  const reserved = new Set([2,3,8,9,12,13,14,15]);
  const states = new WeakMap();
  let adapter, rows = [], storageKey, learning = null, notice;
  const pilots = () => window.DroneControlSettings?.pilots || [];
  const official = () => !!adapter?.controller?.official;
  const practice = () => !!adapter?.controller?.practice;
  const validButton = value => Number.isInteger(value) && value >= 0 && value < 64;
  const copy = value => JSON.parse(JSON.stringify(value));
  const starting = () => Array.from(document.querySelectorAll('.fc-start-overlay')).some(el=>el.getClientRects().length&&!el.classList.contains('opacity-0'));
  const buttonValue = (pad,index) => {
    if (!validButton(index)) return 0;
    const b=pad?.buttons?.[index], v=typeof b==='number'?b:b?.pressed?1:b?.value;
    return Number.isFinite(v)?Math.max(0,Math.min(1,v)):0;
  };
  function binding(index,action) {
    if(action==='gear')return window.DroneControlSettings?.preferences[index]?.pad || {type:'none',index:0};
    return validButton(rows[index]?.[action])?{type:'button',index:rows[index][action]}:{type:'none',index:0};
  }
  function used(index,button,except='') {
    return reserved.has(button)||definitions.some(d=>d.id!==except && binding(index,d.id).type==='button' && binding(index,d.id).index===button);
  }
  function extraUsed(index,button,except='') {
    return reserved.has(button)||extraIds.some(id=>id!==except && rows[index]?.[id]===button);
  }
  function label(index,action) {
    const b=binding(index,action),standard=definitions.find(d=>d.id===action);
    return b.type==='button'?b.index===standard?.button?standard.label.split(' · ')[0]:`按鈕 ${b.index+1}（B${b.index}）`:b.type==='axis'?`Axis ${b.index} ${b.direction>0?'+':'−'}`:'未綁定';
  }
  function tell(index,text) {
    if(!notice){notice=document.createElement('div');notice.className='fc-controller-notice fc-ui-region';notice.setAttribute('role','status');document.body.append(notice);}
    notice.textContent=`P${index+1} · ${text}`;notice.hidden=false;
    clearTimeout(tell.timer);tell.timer=setTimeout(()=>{notice.hidden=true;},3000);
  }
  function state(actor) {
    if(!states.has(actor))states.set(actor,{identity:'',ready:false,previous:{},holds:{},mode:'blocked',landing:false,revision:0,waiting:''});
    return states.get(actor);
  }
  function release() {
    if(learning){learning.status.textContent='已取消偵測，請回到遊戲後重新設定';learning=null;}
    for(const p of pilots()){
      const s=state(p.actor);s.ready=false;s.previous={};s.holds={};s.mode='blocked';s.waiting='';s.at=0;p.actor.controllerBrakeHeld=false;
    }
  }
  function onGround(actor) {
    return actor.position.y <= (window.DroneFlightDynamics?.groundHeight(actor) ?? .1)+.015;
  }
  function motorStopped(actor) {
    return window.DroneFlightSafety?.isStopped(actor) || (!actor.input?.throttle && onGround(actor));
  }
  function allNeutral(sticks) {
    return Array.isArray(sticks) && sticks.length===2 && sticks.every(s=>Number.isFinite(s.x)&&Number.isFinite(s.y)&&Math.max(Math.abs(s.x),Math.abs(s.y))<=.12);
  }
  function sync(actor,pad,setupOpen) {
    const s=state(actor), index=pilots().findIndex(p=>p.actor===actor);
    const identity=pad?`${pad.bridged?'bridge':'browser'}:${pad.index}:${pad.id}:${pad.axes.length}`:'';
    if(s.identity!==identity){s.identity=identity;s.ready=false;s.previous={};s.holds={};s.mode='blocked';s.waiting='';s.at=0;actor.controllerBrakeHeld=false;}
    if(!pad || setupOpen || window.DroneFlightInput?.paused || starting()){
      s.ready=false;s.waiting='';actor.controllerBrakeHeld=false;return false;
    }
    const sticks=adapter.controller.sticks(actor,pad);
    const released=definitions.every(d=>{const b=binding(index,d.id);return b.type!=='button'||buttonValue(pad,b.index)<.35;})&&[2,3,8,9,12,13,14,15].every(i=>buttonValue(pad,i)<.35);
    if(!s.ready){
      if(allNeutral(sticks)&&released){
        s.ready=true;
        if(s.waiting)tell(index,'已接手搖桿，可開始操控');
        s.waiting='';
      }else{
        const waiting=released?'請將左右雙桿回中，接手飛行控制':'請放開遙控器按鈕，再將雙桿回中';
        if(waiting!==s.waiting){s.waiting=waiting;tell(index,waiting);}
      }
    }
    return s.ready;
  }
  function updateBinding(index,action,button) {
    if(!definitions.some(d=>d.id===action)||!pilots()[index])return '玩家或操作無效';
    if(button!==null&&!validButton(button))return 'USB 按鈕索引無效';
    if(button!==null&&used(index,button,action))return '此按鈕已用於其他操作或選單，請選擇另一顆';
    let error='';
    if(action==='gear'){
      const prefs=window.DroneControlSettings.preferences;
      prefs[index].pad=button===null?{type:'none',index:0}:{type:'button',index:button};
      error=window.DroneControlSettings.save(prefs);
    }else{
      rows[index][action]=button;
      try{localStorage.setItem(storageKey,JSON.stringify(rows));}catch(_){error='已套用；瀏覽器無法儲存此綁定';}
    }
    if(!error || error.startsWith('已套用')){release();window.DroneControlSettings?.refreshReferences?.();}
    return error;
  }
  function perform(index,action) {
    const actor=pilots()[index]?.actor;if(!actor)return;
    if(action==='headless'){
      actor.controllerHeadless=!actor.controllerHeadless;
      if(actor.controllerHeadless)actor.controllerHeadlessYaw=actor.rotation?.y||0;
      tell(index,actor.controllerHeadless?'無頭模式已開啟，以目前機頭方向為前進基準':'無頭模式已關閉');
    }else if(action==='backflip'){
      if(window.DroneFlightDynamics?.beginBackflip?.(actor))tell(index,'後空翻');
      else tell(index,'請先起飛，保留足夠高度與空間再後空翻');
    }
  }
  function transformCommand(actor,command) {
    if(actor.controllerBrakeHeld)return {...command,pitch:0,roll:0};
    if(!actor.controllerHeadless)return command;
    const angle=(actor.controllerHeadlessYaw||0)-(actor.rotation?.y||0),c=Math.cos(angle),s=Math.sin(angle);
    return {...command,pitch:command.pitch*c-command.roll*s,roll:command.roll*c+command.pitch*s};
  }
  function poll(actor,pad,enabled) {
    const index=pilots().findIndex(p=>p.actor===actor);if(index<0)return false;
    const s=state(actor);s.pad=pad;actor.controllerBrakeHeld=false;
    if(learning?.index===index){learn(pad);return false;}
    const blocked=!enabled || document.hidden || !pad || window.DroneControlSettings.open || adapter.setupOpen?.() || window.DroneFlightInput?.paused || starting();
    const mode=!blocked&&s.ready?'active':'blocked';
    const now=performance.now(),continuous=mode==='active'&&s.mode==='active',delta=Math.max(0,Math.min(.05,(now-(s.at||now))/1000));
    for(const action of extraIds){
      const b=binding(index,action),down=b.type==='button'&&buttonValue(pad,b.index)>(s.previous[action]?.down ? .35 : .65);
      const edge=continuous&&down&&!s.previous[action]?.down;
      if(action==='brake'){
        actor.controllerBrakeHeld=continuous&&down;
        if(edge){
          window.DroneFlightInput?.reset(actor);
          if(!motorStopped(actor)){actor.autoFlightState='none';actor.targetAltitude=Math.max(.1,actor.position.y);}
          tell(index,'煞車／懸停中；放開 B 恢復操控');
        }
      }else if(edge&&!['cameraLeft','cameraRight'].includes(action))perform(index,action);
      s.previous[action]={down};
    }
    if(continuous){
      const direction=Number(!!s.previous.cameraRight?.down)-Number(!!s.previous.cameraLeft?.down);
      if(direction){
        const id=actor.id?`${actor.id}-btnCam`:actor.prefix?`${actor.prefix}btnCam`:'btnCam';
        window.DroneCameraControls?.get(id)?.shiftView?.(direction,delta);
      }
    }
    s.mode=mode;s.at=now;
    return mode==='active';
  }
  function learn(pad) {
    if(!learning)return;
    const key=pad?`${pad.bridged?'bridge':'browser'}:${pad.index}:${pad.id}`:'';
    if(!pad || learning.identity&&key!==learning.identity){learning.status.textContent='裝置已中斷或變更，請重新偵測';learning=null;return;}
    const values=Array.from(pad.buttons||[],(_,i)=>buttonValue(pad,i));
    if(!learning.identity){learning.identity=key;learning.released=false;}
    if(!learning.released){learning.released=values.every(v=>v<.35);return;}
    const button=values.findIndex(v=>v>.65);
    if(button<0)return;
    const pending=learning;learning=null;
    const error=updateBinding(pending.index,pending.action,button);
    pending.status.textContent=error||`已儲存：${definitions.find(d=>d.id===pending.action).label} → ${label(pending.index,pending.action)}`;
    pending.refresh();
  }
  function editor(host,index) {
    learning=null;
    const section=document.createElement('section');section.className='fc-controller-editor';
    const title=document.createElement('h3');title.textContent='搖桿按鈕配置';
    const description=document.createElement('p');description.textContent='預設採用 Xbox 標準配置。方向鍵與左搖桿控制升降／旋轉，右搖桿控制前後／平移。非標準手把可在此偵測按鈕。';
    const status=document.createElement('p');status.className='fc-control-status';status.setAttribute('role','status');
    const list=document.createElement('div');list.className='fc-controller-bindings';
    const refresh=()=>{
      for(const d of definitions){const text=list.querySelector(`[data-controller-current="${d.id}"]`);text.textContent=label(index,d.id);}
      for(const button of list.querySelectorAll('[data-controller-learn]'))button.setAttribute('aria-pressed',String(learning?.index===index&&learning.action===button.dataset.controllerLearn));
    };
    for(const d of definitions){
      const row=document.createElement('div');row.className='fc-controller-binding';
      const name=document.createElement('strong');name.textContent=d.label;
      const hint=document.createElement('span');hint.textContent=d.hint;
      const current=document.createElement('span');current.dataset.controllerCurrent=d.id;
      const learnButton=document.createElement('button');learnButton.type='button';learnButton.textContent='偵測按鈕';learnButton.dataset.controllerLearn=d.id;learnButton.setAttribute('aria-label',`偵測${d.label}按鈕`);
      learnButton.onclick=()=>{
        if(!motorStopped(pilots()[index].actor)){status.textContent='請先降落停機，再設定按鈕';return;}
        window.DroneControlSettings.cancelLearning();
        learning={index,action:d.id,status,refresh};status.textContent=`請放開所有按鈕，再按下「${d.label}」的實體按鈕；無訊號可取消。`;refresh();
      };
      const clear=document.createElement('button');clear.type='button';clear.textContent='取消綁定';clear.setAttribute('aria-label',`取消${d.label}綁定`);
      clear.onclick=()=>{learning=null;status.textContent=updateBinding(index,d.id,null)||'已取消綁定';refresh();};
      row.append(name,hint,current,learnButton,clear);list.append(row);
    }
    const cancel=document.createElement('button');cancel.type='button';cancel.textContent='取消按鈕偵測';cancel.onclick=()=>{learning=null;status.textContent='已取消偵測，保留設定';refresh();};
    const note=document.createElement('p');note.textContent='X 確認、Y 取消、START 返回選單、BACK 返回上一步。這些按鈕與方向鍵保留給選單操作。斷線或關閉設定後，請放開按鈕並將雙桿回中。';
    const restore=document.createElement('button');restore.type='button';restore.textContent='還原標準按鈕';restore.onclick=()=>{
      learning=null;const prefs=window.DroneControlSettings.preferences;prefs[index].pad={type:'button',index:4};
      rows[index]=Object.fromEntries(definitions.filter(d=>d.id!=='gear').map(d=>[d.id,d.button]));
      const error=window.DroneControlSettings.save(prefs);
      try{localStorage.setItem(storageKey,JSON.stringify(rows));}catch(_){}
      release();status.textContent=error||'已還原標準搖桿按鈕';refresh();window.DroneControlSettings.refreshReferences();
    };
    section.append(title,description,list,restore,cancel,status,note);host.append(section);refresh();
  }
  function reference(host,index) {
    const section=document.createElement('section');section.className='fc-controller-reference';
    const title=document.createElement('h3');title.textContent='搖桿操作';section.append(title);
    for(const text of ['左搖桿／方向鍵：上升、下降、左旋、右旋','右搖桿：前進、後退、左移、右移']){const row=document.createElement('p');row.textContent=text;section.append(row);}
    for(const d of definitions){const row=document.createElement('p');row.textContent=`${d.label}：${label(index,d.id)} · ${d.hint}`;section.append(row);}
    const hint=document.createElement('p');hint.textContent='X 確認 · Y 取消 · START 返回選單 · BACK 返回上一步。上推左桿啟槳，回中後再次上推才離地；低空下拉約 2 秒可降落停槳。';section.append(hint);host.append(section);
  }
  function register(config) {
    if(!config.radio||!config.controller)return;
    adapter=config;storageKey=`droneSoccer.controllerActions.v2.${pilots().length}`;
    rows=pilots().map(()=>Object.fromEntries(definitions.filter(d=>d.id!=='gear').map(d=>[d.id,d.button])));
    try{
      const saved=JSON.parse(localStorage.getItem(storageKey));
      if(Array.isArray(saved)&&saved.length===rows.length)saved.forEach((row,i)=>{
        rows[i]=Object.fromEntries(extraIds.map(id=>[id,null]));
        for(const id of extraIds){const button=row?.[id];if(validButton(button)&&!used(i,button,id))rows[i][id]=button;}
      });
    }catch(_){}
    for(const p of pilots()){p.actor.controllerHeadless=false;p.actor.controllerBrakeHeld=false;}
    window.addEventListener('blur',release);document.addEventListener('visibilitychange',()=>{if(document.hidden)release();});
    document.addEventListener('close',()=>{learning=null;release();},true);
  }
  window.DroneControllerActions=Object.freeze({register,sync,poll,editor,reference,binding,label,extraUsed,updateBinding,release,transformCommand,
    navigationSources:()=>pilots().map(p=>({pad:state(p.actor).pad,actor:p.actor})).filter(s=>s.pad).map(s=>({pad:s.pad,sticks:adapter.controller.sticks(s.actor,s.pad)})),
    cancelLearning:()=>{learning=null;},
    get learning(){return !!learning;},get preferences(){return copy(rows);},get official(){return official();},get practice(){return practice();}});
})();
