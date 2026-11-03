(() => {
  'use strict';
  const $=id=>document.getElementById(id), role=document.body.dataset.role;
  const params=new URLSearchParams(location.search);
  if(role==='watch'&&params.get('embedded')==='1')document.body.dataset.embedded='true';
  const send=data=>{if(socket?.readyState===WebSocket.OPEN)socket.send(JSON.stringify(data));else tell('網路未連線，請等待恢復');};
  const post=data=>{if(sceneLoaded)$('scene').contentWindow.postMessage({source:'drone-lan-app',...data},location.origin);};
  const text=(id,value)=>{if($(id))$(id).textContent=value;};
  let messageTimer;
  const tell=message=>{clearTimeout(messageTimer);text('message',message);if(role==='player')text('joinError',message);if(message==='已複製')messageTimer=setTimeout(()=>text('message',''),1800);};
  const options=(select,rows,value)=>{select.replaceChildren(...rows.map(([id,label,disabled])=>{const o=new Option(label,id);o.disabled=!!disabled;return o;}));if(rows.some(r=>r[0]===value&&!r[2]))select.value=value;};
  let socket, room, mySeat='', myId='', session='', ticket='', epoch=0, seq=0, lastSnapshot=null, lastSnapshotAt=0;
  let sceneLoaded=false, sceneRendered=false,sceneKey='', cameraKey='', retry, stopped=false, audio=null, sound=false, lastPhase='';
  const icon=name=>{const svg=document.createElementNS('http://www.w3.org/2000/svg','svg'),use=document.createElementNS(svg.namespaceURI,'use');svg.classList.add('lan-icon');svg.setAttribute('aria-hidden','true');use.setAttribute('href','/apps/drone-soccer/assets/game/lan-icons.svg#'+name);svg.append(use);return svg;};
  let currentCode=(params.get('code')||(role==='host'?sessionStorage.getItem('lan.current.host'):'')||'').toUpperCase(), renderedRoster='', renderedCamera='';
  let authenticated=false,loginPromise=Promise.resolve();
  let playerExitReason='',readyRequested=false,playerUiSample={present:false,healthy:false,channelsReady:false,center:false,synced:false,sticks:[{x:0,y:0},{x:0,y:0}]},lastUiPublish=0,lastUiSignature='';
  let device;
  const newDevice=()=>crypto.randomUUID?.()||Array.from(crypto.getRandomValues(new Uint8Array(18)),n=>n.toString(16).padStart(2,'0')).join('');
  try{device=localStorage.getItem('lan.device');if(!device){device=newDevice();localStorage.setItem('lan.device',device);}}catch{device=newDevice();}
  async function login(key){
    if(role!=='host')return;
    try{
      if(key){const result=await fetch('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({key})});if(!result.ok)throw Error('管理金鑰無效，請重新登入主持人控制台');}
      authenticated=(await (await fetch('/api/session')).json()).authenticated;
      if(authenticated){sessionStorage.removeItem('lan.admin');for(const item of Object.keys(sessionStorage))if(/^lan\.host\.[A-HJ-NP-Z2-9]{6}$/.test(item))sessionStorage.removeItem(item);tell('');}
    }catch(e){authenticated=false;tell(e.message);}
  }
  function consumeKey(){const key=new URLSearchParams(location.hash.slice(1)).get('admin')||sessionStorage.getItem('lan.admin');if(location.hash)history.replaceState(null,'',location.pathname+location.search);return login(key);}
  window.addEventListener('hashchange',()=>{if(role!=='host')return;loginPromise=consumeKey().then(()=>{clearTimeout(retry);const old=socket;socket=null;old?.close();stopped=false;connect();});});
  function rememberRoom(){const query=new URLSearchParams(location.search);query.set('code',currentCode);history.replaceState(null,'',location.pathname+'?'+query);}
  const credentials=()=>{try{return JSON.parse(sessionStorage.getItem('lan.player.'+currentCode)||'null');}catch{return null;}};
  const readRoster=()=>Object.fromEntries(['red','blue'].map(t=>[t,{humans:Number($(t+'Humans').value),ai:Number($(t+'AI').value)}]));
  const seatLabel=s=>`${s.label}${s.ai?'・AI':''}`;
  const cameraName=()=>({center:'己方短邊・固定中間',left:'己方短邊・固定左邊',right:'己方短邊・固定右邊'})[room?.camera.mode]||'己方短邊・固定中間';
  let hostPanel=role==='host'&&['roster','room','camera','invite'].includes(params.get('panel'))?params.get('panel'):role==='host'&&currentCode?(sessionStorage.getItem('lan.host.panel')||'room'):'roster',hostRoomCode=currentCode,serverJoinURL='',serverSetupURL='',serverInvitationURL='';
  let accessMode='online',invitationReady=false,joinConfiguration='',configRetry,configStopped=false;
  function roomInvitation(){
    if(!invitationReady||!serverInvitationURL)return '';
    const url=new URL(serverInvitationURL);
    if(room&&room.status!=='closed')url.searchParams.set('code',room.code);
    return url.href;
  }
  function applyConnectionConfig(value){
    const online=value.accessMode==='online';
    accessMode=online?'online':'legacy';
    let join='',setup='',invitation='',ready=!online||value.publicReady===true&&value.directJoin===true;
    if(ready){
      const joinURL=new URL(value.joinURL);
      if(joinURL.protocol!=='https:'||joinURL.username||joinURL.password||joinURL.pathname!=='/play'||joinURL.search||joinURL.hash)throw Error('玩家加入網址尚未就緒');
      if(online){
        const publicOrigin=new URL(value.origin);
        if(publicOrigin.protocol!=='https:'||publicOrigin.origin!==joinURL.origin||publicOrigin.pathname!=='/'||publicOrigin.search||publicOrigin.hash||publicOrigin.username||publicOrigin.password)throw Error('公開遊戲網址尚未就緒');
      }else if(value.setupURL){
        const setupURL=new URL(value.setupURL);
        if(setupURL.protocol!=='http:'||setupURL.hostname!==joinURL.hostname||setupURL.username||setupURL.password||setupURL.pathname!=='/'||setupURL.search||setupURL.hash)throw Error('區網設定網址無效');
        setup=setupURL.href;
      }
      join=joinURL.href;invitation=online?join:(setup||join);
    }
    const signature=[accessMode,ready,join,setup].join('|');
    if(signature!==joinConfiguration&&role==='host'){for(const id of ['joinQR','qr'])$(id).removeAttribute('src');joinConfiguration=signature;}
    invitationReady=ready;serverJoinURL=join;serverSetupURL=setup;serverInvitationURL=invitation;
    if(role==='host'){
      const note=document.querySelector('#invite .invite-note'),methods=document.querySelector('#invite .invite-methods');
      note.textContent=online?(ready?'玩家掃碼直接進入本房間，免安裝或信任憑證。主機與玩家皆需網際網路。':'正在建立公開安全連線，完成後即可顯示 QR。'):(setup?'區網模式：首次加入需先完成裝置信任。':'區網模式：僅供已完成裝置信任的裝置加入。');
      methods.textContent=online?'掃碼或開啟此網址，即會帶入本場房碼。':'掃碼進入本房間，或使用相同網址與代碼加入。';
      text('interfaceInfo',ready?(online?'公開遊戲入口：':'區網入口：')+new URL(invitation).origin:'公開遊戲入口準備中；尚未產生可分享網址。');
      renderJoinInformation();
    }
    if(socket?.readyState===WebSocket.OPEN)text('connection',online?'遊戲連線已建立':'區網已連線');
  }
  async function loadConnectionConfig(){
    clearTimeout(configRetry);if(configStopped)return;
    try{
      const response=await fetch('/api/config',{cache:'no-store'});
      if(!response.ok)throw Error('連線資訊暫時無法取得');
      applyConnectionConfig(await response.json());
    }catch(_){
      invitationReady=false;serverJoinURL=serverSetupURL=serverInvitationURL='';
      if(role==='host'){text('interfaceInfo','正在確認公開遊戲入口，請稍候。');renderJoinInformation();}
    }
    if(!configStopped)configRetry=setTimeout(loadConnectionConfig,invitationReady?5000:2500);
  }
  window.addEventListener('pagehide',()=>{configStopped=true;clearTimeout(configRetry);});
  window.addEventListener('pageshow',event=>{if(event.persisted){configStopped=false;void loadConnectionConfig();}});
  function selectHostPanel(panel){
    if(!['roster','room','camera','invite'].includes(panel))panel='roster';
    hostPanel=panel;document.body.dataset.hostPanel=panel;sessionStorage.setItem('lan.host.panel',panel);
    if(role==='host'&&currentCode){const query=new URLSearchParams(location.search);query.set('code',currentCode);query.set('panel',panel);history.replaceState(null,'',location.pathname+'?'+query+location.hash);}
    for(const button of document.querySelectorAll('[data-host-panel-button]')){const selected=button.dataset.hostPanelButton===panel;button.setAttribute('aria-selected',String(selected));button.tabIndex=selected?0:-1;}
    for(const pane of document.querySelectorAll('.host-panel[data-host-panel]'))pane.hidden=pane.dataset.hostPanel!==panel;
  }
  function connect(){
    clearTimeout(retry);if(stopped)return;
    const transport=new WebSocket(`wss://${location.host}/socket`);socket=transport;text('connection','正在連線');
    transport.onopen=()=>{
      if(socket!==transport)return;
      text('connection',accessMode==='online'?'遊戲連線已建立':'區網已連線');
      if(role==='host'&&currentCode)send({type:'host',code:currentCode,...(!authenticated?{hostToken:sessionStorage.getItem('lan.host.'+currentCode)}:{})});
      else if(role==='player'&&currentCode){const c=credentials();if(c)send({type:'join',code:currentCode,resume:c.resume});else send({type:'inspect',code:currentCode});}
      else if(role==='watch'&&currentCode)send({type:'watch',code:currentCode});
      else if(role==='watch')showCover('請從服務大廳選擇賽事','返回大廳即可加入或觀看其他房間');
    };
    transport.onmessage=event=>{
      if(socket!==transport)return;
      const msg=JSON.parse(event.data);
      if(msg.type==='left'){sessionStorage.removeItem('lan.player.'+currentCode);sessionStorage.removeItem('lan.current.player');stopped=true;location.href='/';return;}
      if(msg.type==='signedOut'){authenticated=false;stopped=true;location.href='/admin';return;}
      if(msg.type==='serverClosing'){ticket='';armed=false;showCover('伺服器重新啟動中','請等待連線恢復，原席位與比賽進度會保留');return;}
      if(msg.type==='roomClosed'){
        if(role==='player'){playerExitReason=msg.message||'房間已關閉';readyRequested=false;}
        stopped=true;ticket='';armed=false;mySeat='';session='';sessionStorage.removeItem('lan.player.'+currentCode);sessionStorage.removeItem('lan.current.'+role);
        if(room){room.status='closed';room.reason=msg.message;}if(role==='player'){$('joinPanel').hidden=true;$('setupPanel').hidden=true;$('setupOpen').hidden=true;}
        showCover('房間已關閉','此代碼與 QR Code 已失效，請返回服務大廳');return;
      }
      if(msg.type==='error'){
        if(role==='player'){readyRequested=false;if(mySeat&&['lobby','paused'].includes(room?.status))$('setupPanel').hidden=false;}
        tell(msg.message);
        if(role==='player'&&/原連線已失效/.test(msg.message)){sessionStorage.removeItem('lan.player.'+currentCode);mySeat='';myId='';$('joinPanel').hidden=false;send({type:'inspect',code:currentCode});}
        return;
      }
      if(msg.type==='removed'||msg.type==='replaced'){
        if(role==='player'){playerExitReason=msg.message;readyRequested=false;}
        tell(msg.message);stopped=true;mySeat='';myId='';sessionStorage.removeItem('lan.player.'+currentCode);
        $('setupPanel').hidden=true;$('joinPanel').hidden=false;showCover(msg.message,'重新整理頁面後可選擇可用席位');return;
      }
      if(msg.type==='tick'){ticket=msg.ticket;epoch=msg.epoch;return;}
      if(msg.type==='created'){
        stopped=false;
        currentCode=msg.room.code;sessionStorage.setItem('lan.current.host',currentCode);if(msg.authKind!=='session')sessionStorage.setItem('lan.host.'+currentCode,msg.hostToken);rememberRoom();
        updateRoom(msg.room);if(room.seats.some(s=>!s.ai))openInvite();return;
      }
      if(msg.type==='joined'){
        mySeat=msg.seat;myId=msg.id;session=msg.session;seq=0;armed=false;neutralAt=0;readyRequested=false;playerExitReason='';
        // A resumed session needs a fresh player-view render/ack, even for the same room and camera.
        sceneKey='';cameraKey='';sceneRendered=false;
        tell('');
        sessionStorage.setItem('lan.player.'+currentCode,JSON.stringify({resume:msg.resume}));sessionStorage.setItem('lan.current.player',currentCode);
        $('joinPanel').hidden=true;$('setupPanel').hidden=false;$('setupOpen').hidden=false;updateRoom(msg.room);return;
      }
      if(msg.type==='room')updateRoom(msg.room);
      if(msg.type==='snapshot'){
        if(lastSnapshot&&msg.value.seq<=lastSnapshot.seq&&msg.value.epoch===lastSnapshot.epoch)return;
        lastSnapshot=msg.value;lastSnapshotAt=performance.now();post({type:'snapshot',value:msg.value});drawScore(msg.value);
      }
    };
    transport.onclose=()=>{
      if(socket!==transport||stopped)return;
      text('connection','網路已中斷');ticket='';armed=false;neutralAt=0;readyRequested=false;showCover('網路中斷・操作已停止','恢復連線後請將搖桿回中並重新準備');
      if(!stopped)retry=setTimeout(connect,1500);
    };
    transport.onerror=()=>{if(socket===transport)tell(accessMode==='online'?'無法連線：請確認主機仍開啟、主機與玩家都能連上網際網路，並使用最新的房間 QR 或網址。':'無法連線：請確認區網主機仍開啟、裝置信任已完成，並使用目前的區網 HTTPS 網址。');};
  }
  function showCover(title,detail='',live=false){
    const el=$('cover'),status=$('broadcastStatus');el.replaceChildren();const strong=document.createElement('strong');strong.textContent=title;el.append(strong);if(detail){const span=document.createElement('span');span.textContent=detail;el.append(span);}
    if(status){status.replaceChildren();status.hidden=!(title&&live);if(!status.hidden){status.append(...el.childNodes);status.title=[title,detail].filter(Boolean).join('・');el.hidden=true;return;}}
    el.hidden=!title;el.classList.toggle('live',live);
  }
  function syncCover(){
    if(!room)return;
    if(room.status==='closed'){showCover('房間已關閉','此代碼與 QR Code 已失效');return;}
    if(room.status==='ended'){showCover('比賽結束',room.reason+'・可返回大廳選擇下一場');return;}
    if(!room.engineReady){showCover('比賽引擎正在同步',room.reason);return;}
    if(!sceneLoaded||!sceneRendered){showCover('正在接回即時賽況','同步此房間的比分、時間與無人機位置');return;}
    if(performance.now()-lastSnapshotAt>1500){showCover('賽況訊號中斷','場上畫面已停止，請等待主持人處理');return;}
    if(room.status==='paused'){showCover('比賽暫停',room.reason+'・回中並重新準備，等待主持人恢復');return;}
    if(room.status==='lobby'){showCover('等待主持人開賽',role==='player'?(room.seats.find(s=>s.id===mySeat)?.player?.ready?'已準備完成':'請完成搖桿偵測與準備'):room.seats.every(s=>s.ai)?'全 AI 已就緒，可直接開賽':'請等待真人席位全部準備完成',true);return;}
    const phase=lastSnapshot?.match.phase;
    if(room.status==='ended'){showCover('比賽結束',room.reason);return;}
    const names={ready:'開賽倒數・等哨音後上推起飛',break:'局間休息',setEnd:'本局結束',penReady:'點球準備',penIntro:'點球大戰',penFoul:'罰點球',penShot:'點球進行中',penResult:'點球結果',goldenIntro:'準備驟死賽'};
    const detail=phase==='ready'?`${Math.max(0,Math.ceil(lastSnapshot.match.phaseLeft))} 秒`:phase==='break'?`無人機已返回起飛區・休息後進入第 ${lastSnapshot.match.set+1} 局`:'';
    showCover(names[phase]||'',detail,phase!=='ready');
  }
  function drawScore(s){
    text('redScore',s.scores.red);text('blueScore',s.scores.blue);
    const m=s.match, phases=['ready','break','setEnd','penReady','penShot','penResult','goldenIntro','penIntro','penFoul'];
    if(role==='player'){const safety=s.rigs.find(r=>r.id===mySeat)?.safety;if(safety?.locked)tell(`馬達已停機：${({landed:'降落完成',trapped:'受困',failure:'故障'})[safety.reason]||safety.reason}。先將雙桿回中；機體直立停穩後再上推啟槳。`);}
    const time=phases.includes(m.phase)?m.phaseLeft:m.phase==='golden'?m.goldenElapsed:m.setRemaining;
    const n=Math.max(0,Math.ceil(time));text('timer',`${String(Math.floor(n/60)).padStart(2,'0')}:${String(n%60).padStart(2,'0')}`);
    text('round',`第 ${Math.max(1,m.set)} 局・勝局 ${m.setWins.red} : ${m.setWins.blue}・${room?.status==='paused'?'暫停':({play:'比賽中',golden:'驟死賽',final:'終場',ready:'倒數',break:'局間休息'}[m.phase]||'正式賽')}`);
    if(lastPhase&&lastPhase!==m.phase&&['play','golden','setEnd','final'].includes(m.phase))whistle(['setEnd','final'].includes(m.phase));lastPhase=m.phase;syncCover();
  }
  function updateRoom(value){
    const oldEpoch=epoch,oldStatus=room?.status,oldHumans=room?.seats.filter(s=>!s.ai).length||0;room=value;currentCode=room.code;epoch=room.epoch;if(room.status!=='closed')rememberRoom();
    if(room.status==='running'&&oldStatus!=='running')tell('');
    if(oldEpoch&&oldEpoch!==epoch&&role==='player'&&room.status==='paused') {armed=false;neutralAt=0;}
    if(room.status==='closed'){if(role==='host'){sessionStorage.removeItem('lan.current.host');history.replaceState(null,'','/host');}stopped=true;}
    syncScene();syncCover();
    if(role==='watch')text('watchRoomName',room.name+' · '+room.arenaName);
    if(role==='host'){renderHost();if(room.status==='lobby'&&room.seats.some(s=>!s.ai)&&((oldStatus==='lobby'&&room.seats.filter(s=>!s.ai).length>oldHumans)||(!oldStatus&&params.get('panel')==='invite')))openInvite();}
    if(role==='player'){
      const preparedByServer=!!room.seats.find(s=>s.id===mySeat)?.player?.ready;
      if(preparedByServer&&readyRequested)tell('');
      if(preparedByServer||['running','ended','closed'].includes(room.status))readyRequested=false;
      text('cameraLabel',`${cameraName()}・LT／RT 可在己方短邊左右移動站位`);
      if(mySeat){const s=room.seats.find(s=>s.id===mySeat);text('identity',s?seatLabel(s):'席位已釋放');}
      else if(!['ended','closed'].includes(room.status)){ $('codeForm').hidden=true;$('joinForm').hidden=false;text('joinTitle',`${room.name}・${room.code}`);const previous=$('seat').value;
        options($('seat'),room.seats.map(s=>[s.id,`${seatLabel(s)}${s.player?'・已占用':''}`,s.ai||!!s.player]),previous);confirmSeat();
        if(room.locked)tell('房間已鎖定新玩家加入；原玩家可在原瀏覽器重連'); }
      else $('joinPanel').hidden=true;
      if(['ended','closed'].includes(room.status)){$('setupPanel').hidden=true;$('setupOpen').hidden=true;}
    }
    if(role!=='player'&&$('target')){const v=$('target').value;options($('target'),room.seats.map(s=>[s.id,seatLabel(s)]),v);
      if(room.broadcastCamera){$('observer').value=role==='watch'&&params.get('embedded')==='1'?'all':room.broadcastCamera.mode;$('target').value=room.broadcastCamera.target||room.seats[0].id;observer(false);}}
  }
  function syncScene(){
    if(!room||role==='player'&&!mySeat)return;
    const key=room.code+'|'+room.engineRevision+'|'+JSON.stringify(room.roster);
    if(sceneKey!==key){
      sceneKey=key;sceneLoaded=false;sceneRendered=false;cameraKey='';lastSnapshot=null;
      const q=new URLSearchParams({lan:'viewer'});for(const t of ['red','blue']){q.set(t+'Humans',room.roster[t].humans);q.set(t+'AI',room.roster[t].ai);}
      $('scene').src='/apps/drone-soccer/games/radio/drone_soccer_competition_add_ai_realgame.html?'+q;$('scene').hidden=false;
    }
    configureScene();
  }
  function configureScene(){
    if(!sceneLoaded||!room)return;
    const key=sceneKey+'|'+room.camera.revision+'|'+mySeat;
    if(cameraKey!==key){cameraKey=key;post({type:'config',role:role==='player'?'player':'watch',seat:mySeat,camera:room.camera,engineRevision:room.engineRevision});if(lastSnapshot)post({type:'snapshot',value:lastSnapshot});}
  }
  window.addEventListener('message',event=>{
    if(event.source!==$('scene').contentWindow||event.origin!==location.origin||event.data?.source!=='drone-lan-scene')return;
    if(event.data.type==='loaded'){sceneLoaded=true;cameraKey='';configureScene();observer(false);}
    if(event.data.type==='rendered'&&event.data.revision===room?.camera.revision&&event.data.engineRevision===room?.engineRevision){sceneRendered=true;syncCover();if(role==='player'&&mySeat)send({type:'ack',camera:event.data.revision,view:event.data.engineRevision});}
  });
  function observer(publish=true){if(!$('observer'))return;const data={type:'observer',mode:$('observer').value,target:$('target').value};post(data);if(role==='host'&&room&&publish)send({...data,type:'broadcast'});}
  $('observer')?.addEventListener('change',()=>observer());$('target')?.addEventListener('change',()=>observer());
  function renderHost(){
    text('hostTitle',room.name);text('hostArena',room.arenaName);$('hostTitle').title=room.name;
    renderJoinInformation();
    if(room.status==='closed'){
      if(hostRoomCode){hostRoomCode='';selectHostPanel('roster');}$('tabRoom').disabled=true;$('tabCamera').disabled=true;
      $('roomControls').hidden=true;$('create').hidden=false;text('create','建立新房間');$('updateRoster').hidden=true;
      for(const id of ['redHumans','redAI','blueHumans','blueAI','difficulty','breakSec'])$(id).disabled=false;
      currentCode='';return;
    }
    $('roomControls').hidden=false;$('create').hidden=true;$('updateRoster').hidden=room.status!=='lobby';
    $('tabRoom').disabled=false;$('tabCamera').disabled=false;
    if(hostRoomCode!==room.code){hostRoomCode=room.code;hostPanel='room';}
    selectHostPanel(hostPanel);
    if(renderedRoster!==JSON.stringify(room.roster)){
      renderedRoster=JSON.stringify(room.roster);for(const t of ['red','blue']){$(t+'Humans').value=room.roster[t].humans;$(t+'AI').value=room.roster[t].ai;}
      $('difficulty').value=room.difficulty;$('breakSec').value=String(room.breakSec);
    }
    if(renderedCamera!==room.code+'|'+room.camera.revision){
      renderedCamera=room.code+'|'+room.camera.revision;$('cameraMode').value=room.camera.mode;
      for(const key of ['fov','distance','height'])$(key).value=room.camera[key];
      window.dispatchEvent(new CustomEvent('flight-camera-applied',{detail:{revision:room.camera.revision}}));
    }
    const statusReason=room.status==='lobby'&&room.seats.every(s=>s.ai)?(room.engineReady?'全 AI 已就緒，可直接開賽':'比賽引擎準備中'):room.reason;
    text('roomCode',room.code);text('roomStatus',`${({lobby:'等待開賽',running:'比賽進行中',paused:'比賽暫停',ended:'比賽結束',closed:'房間關閉'})[room.status]}${statusReason?'・'+statusReason:''}`);
    $('roomStatus').dataset.state=room.status;
    $('joinLock').checked=room.locked;
    const canEdit=room.status==='lobby';for(const id of ['redHumans','redAI','blueHumans','blueAI','difficulty','breakSec'])$(id).disabled=!canEdit;
    $('start').hidden=room.status!=='lobby';$('start').disabled=!room.canStart;
    $('pause').hidden=room.status!=='running';$('resume').hidden=room.status!=='paused';$('resume').disabled=!room.canStart;
    $('end').hidden=!['running','paused'].includes(room.status);$('joinLock').disabled=room.status==='ended';
    $('recover').hidden=!room.recoverable;
    for(const el of $('cameraForm').elements)el.disabled=!['lobby','paused'].includes(room.status);
    const seatRoot=$('seats');seatRoot.replaceChildren();
    for(const team of ['red','blue']){const seats=room.seats.filter(s=>s.team===team);text(team+'TeamCount',`${seats.filter(s=>s.ai||s.player?.ready).length} / ${seats.length} 就緒`);}
    for(const s of room.seats){const row=document.createElement('div');row.className='seat-row';row.dataset.team=s.team;row.style.gridColumn=s.team==='red'?'1':'2';row.style.gridRow=String(s.id.endsWith('atk')?1:Number(s.id.slice(-1))+1);const info=document.createElement('div'),copy=document.createElement('div'),symbol=document.createElement('span'),title=document.createElement('strong'),detail=document.createElement('small');
      info.className='seat-info';copy.className='seat-copy';symbol.className='seat-symbol '+s.team;symbol.append(icon('drone'));title.textContent=`${s.id.endsWith('atk')?1:Number(s.id.slice(-1))+1} 號・${s.role==='attacker'?'前鋒':'防守'}`;title.title=seatLabel(s);row.setAttribute('aria-label',seatLabel(s));const p=s.player;row.dataset.state=s.ai?'ai':!p?'empty':!p.connected?'offline':p.ready?'ready':'connected';
      detail.textContent=s.ai?'AI 已配置':!p?'等待加入':`${p.name}・${!p.connected?'已斷線':p.ready?'準備完成':p.pad?'搖桿已連線':'已加入，等待搖桿'}${p&&(!p.cameraSynced||!p.viewSynced)?'・等待視角同步':''}`;
      detail.title=detail.textContent;copy.append(title,detail);info.append(symbol,copy);row.append(info);
      const actions=document.createElement('div');actions.className='seat-actions';
      if(p){const remove=document.createElement('button');remove.textContent='移除';remove.setAttribute('aria-label','移除'+seatLabel(s)+'玩家');remove.onclick=async()=>{if(await FlightPanels.confirm({title:'移除這位玩家？',description:p.name+' 將離開'+seatLabel(s)+'席位。比賽進行中會自動暫停，待席位準備後才能恢復。',acceptLabel:'移除玩家',danger:true}))send({type:'remove',id:p.id});};actions.append(remove);}
      if(!s.ai&&room.status==='paused'){const ai=document.createElement('button');ai.textContent='替換 AI';ai.setAttribute('aria-label',seatLabel(s)+'替換 AI');ai.onclick=async()=>{if(await FlightPanels.confirm({title:'由 AI 接手席位？',description:seatLabel(s)+'將改由 AI 操作。原玩家會離房，仍需由主持人恢復比賽。',acceptLabel:'替換為 AI'}))send({type:'replaceAI',seat:s.id});};actions.append(ai);}
      if(actions.children.length)row.append(actions);
      seatRoot.append(row);
    }
    $('watchLink').hidden=false;$('watchLink').href='/watch?code='+room.code;
    text('inviteCode',room.code);const humans=room.seats.filter(s=>!s.ai),joined=humans.filter(s=>s.player?.connected).length;
    text('inviteCount',`已加入 ${joined} / ${humans.length}・等待加入 ${humans.length-joined}`);
  }
  function renderJoinInformation(){
    if(!$('joinDetails'))return;
    const active=!!room&&room.status!=='closed';$('joinEmpty').hidden=active;$('joinDetails').hidden=!active;
    const invitation=roomInvitation();
    for(const id of ['preJoinAddress','joinAddress']){text(id,invitation||'正在建立公開遊戲入口…');if(invitation)$(id).href=invitation;else $(id).removeAttribute('href');}
    $('joinURL').value=invitation;
    for(const id of ['copyJoinURL','copyURL'])$(id).disabled=!invitation;
    for(const id of ['copyJoinCode','copyCode','enlargeInvite'])$(id).disabled=!invitation||!active;
    for(const id of ['joinQR','qr'])$(id).hidden=!invitation;
    if($('directJoinURL')){$('directJoinURL').hidden=accessMode!=='legacy'||!serverSetupURL||!invitationReady;if(invitationReady){const direct=new URL(serverJoinURL);if(active)direct.searchParams.set('code',room.code);$('directJoinURL').href=direct.href;}else $('directJoinURL').removeAttribute('href');}
    if(!active||!invitation){for(const id of ['joinQR','qr'])$(id).removeAttribute('src');text('joinRoomCode',active?room.code:'');text('joinPlayerCount',invitationReady?'':'公開連線準備完成後，即可邀請玩家。');text('joinQrStatus',invitationReady?'':'公開連線準備中，尚未產生 QR。');$('joinQrStatus').hidden=invitationReady;return;}
    text('joinQrStatus','QR 載入失敗，請重新開啟本分頁，或使用下方網址加入。');
    const src='/api/qr?code='+room.code;if($('joinQR').getAttribute('src')!==src){$('joinQrStatus').hidden=true;$('joinQR').src=src;}
    if($('invite').open&&$('qr').getAttribute('src')!==src)$('qr').src=src;
    text('joinRoomCode',room.code);const humans=room.seats.filter(s=>!s.ai),joined=humans.filter(s=>s.player?.connected).length;
    text('joinPlayerCount',`已加入 ${joined} / ${humans.length}・等待 ${humans.length-joined}`);
  }
  function openInvite(){if(!room||room.status==='closed')return;selectHostPanel('invite');if(!$('invite').open)$('invite').showModal();renderJoinInformation();}
  if(role==='host'){
    for(const [id,name] of Object.entries({tabRoster:'grid',tabRoom:'users',tabCamera:'camera',tabInvite:'qr',start:'play',pause:'pause',resume:'play',end:'stop',closeRoom:'close',inviteOpen:'qr'}))$(id)?.prepend(icon(name));
    for(const [selector,name] of [['[data-host-view-button="broadcast"]','broadcast'],['[data-host-view-button="controls"]','settings']])document.querySelector(selector)?.prepend(icon(name));
    if(params.get('view')==='broadcast'){document.body.dataset.hostView='broadcast';for(const b of document.querySelectorAll('[data-host-view-button]'))b.setAttribute('aria-pressed',String(b.dataset.hostViewButton==='broadcast'));}
    selectHostPanel(hostPanel);
    for(const button of document.querySelectorAll('[data-host-panel-button]')){
      button.onclick=()=>selectHostPanel(button.dataset.hostPanelButton);
      button.onkeydown=event=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;event.preventDefault();const enabled=[...document.querySelectorAll('[data-host-panel-button]:not(:disabled)')];const i=enabled.indexOf(button);const next=event.key==='Home'?enabled[0]:event.key==='End'?enabled.at(-1):enabled[(i+(event.key==='ArrowRight'?1:-1)+enabled.length)%enabled.length];selectHostPanel(next.dataset.hostPanelButton);next.focus();};
    }
    for(const button of document.querySelectorAll('[data-host-view-button]'))button.onclick=()=>{document.body.dataset.hostView=button.dataset.hostViewButton;for(const item of document.querySelectorAll('[data-host-view-button]'))item.setAttribute('aria-pressed',String(item===button));};
    for(const t of ['red','blue'])for(const k of ['Humans','AI'])if(params.has(t+k))$(t+k).value=params.get(t+k);
    if(params.has('difficulty'))$('difficulty').value=params.get('difficulty');
    $('createForm').onsubmit=async e=>{e.preventDefault();await loginPromise;if(!authenticated){tell('請先從服務管理中心登入，或開啟主機終端機的管理網址');return;}const data={type:'create',roster:readRoster(),difficulty:$('difficulty').value,breakSec:Number($('breakSec').value)};if(socket?.readyState!==1){tell('管理連線正在恢復，請稍候再建立');return;}send(data);};
    $('updateRoster').onclick=()=>send({type:'roster',roster:readRoster()});
    for(const action of ['start','pause','resume'])$(action).onclick=()=>send({type:action});
    $('end').onclick=async()=>{if(await FlightPanels.confirm({title:'結束這場比賽？',description:'目前結果會保存到比賽紀錄，賽場釋放給下一個房間。結束後無法恢復本場。',acceptLabel:'結束並保存結果',danger:true}))send({type:'end'});};
    $('recover').onclick=async()=>{if(await FlightPanels.confirm({title:'從最後同步狀態復原？',description:'比分、時間與位置保留；碰撞接觸與未完成過門會重置。復原後仍需全員準備，再由主持人恢復。',acceptLabel:'復原比賽引擎'}))send({type:'recover'});};
    $('joinLock').onchange=()=>send({type:'lock',locked:$('joinLock').checked});
    $('cameraForm').onsubmit=e=>{e.preventDefault();send({type:'camera',camera:{mode:$('cameraMode').value,fov:Number($('fov').value),distance:Number($('distance').value),height:Number($('height').value),bob:false}});};
    $('inviteOpen').onclick=openInvite;$('inviteClose').onclick=()=>$('invite').close();
    $('createInvite').onclick=()=>{if(!$('createForm').checkValidity()){selectHostPanel('roster');$('createForm').reportValidity();return;}$('createForm').requestSubmit();};
    $('enlargeInvite').onclick=openInvite;
    $('copyJoinURL').onclick=()=>{const invitation=roomInvitation();if(invitation)void copy(invitation);};$('copyJoinCode').onclick=()=>room&&invitationReady&&copy(room.code);
    $('joinQR').onload=()=>{$('joinQrStatus').hidden=true;};$('joinQR').onerror=()=>{if(room&&room.status!=='closed')$('joinQrStatus').hidden=false;};
    $('copyURL').onclick=()=>copy($('joinURL').value);$('copyCode').onclick=()=>copy(room.code);
    $('closeRoom').onclick=async()=>{if(await FlightPanels.confirm({title:'關閉這個房間？',description:'本場將結束，所有玩家離房，原代碼與 QR 失效。比賽紀錄仍會保存。',acceptLabel:'關閉房間',danger:true}))send({type:'close'});};
    renderJoinInformation();
  }
  async function copy(value){try{await navigator.clipboard.writeText(value);tell('已複製');}catch{tell('無法存取剪貼簿，請長按或選取文字複製');}}
  function confirmSeat(){
    const option=$('seat').selectedOptions[0],s=option&&!option.disabled?room?.seats.find(s=>s.id===$('seat').value):null;
    const blocked=!room||room.locked||!['lobby','paused'].includes(room.status)||!s||socket?.readyState!==1;
    if($('joinSubmit'))$('joinSubmit').disabled=blocked;
    text('seatConfirm',s?`你將加入 ${seatLabel(s)}，只控制這台無人機。`:'目前沒有可選的真人席位。');
  }
  // Read-only, sanitized presentation state. No tickets or session credentials.
  function playerStatusSnapshot(){
    const current=room?.code===currentCode?room:null,seat=current?.seats.find(s=>s.id===mySeat);
    return {code:currentCode,connected:socket?.readyState===1,stopped,exitReason:playerExitReason,joined:!!mySeat,
      room:current?{code:current.code,name:current.name,arenaName:current.arenaName,status:current.status,locked:current.locked,engineReady:current.engineReady,reason:current.reason}:null,
      seats:current?current.seats.map(s=>({id:s.id,team:s.team,role:s.role,label:s.label,ai:s.ai,taken:!!s.player})):[],
      seat:seat?{id:seat.id,label:seat.label,team:seat.team,name:seat.player?.name||'',prepared:!!seat.player?.ready,synced:!!seat.player?.cameraSynced&&!!seat.player?.viewSynced}:null,
      sample:{...playerUiSample,sticks:playerUiSample.sticks.map(s=>({...s}))},readyAllowed:!$('ready').disabled,readyRequested,
      setupOpen:!$('setupPanel').hidden,sceneRendered,controllerReason:$('padReason').textContent,controllerLabel:$('padStatus').textContent,
      notice:$('message').textContent};
  }
  function publishPlayerStatus(now){
    if(role!=='player'||now-lastUiPublish<100)return;lastUiPublish=now;
    const detail=playerStatusSnapshot(),signature=JSON.stringify(detail);if(signature===lastUiSignature)return;lastUiSignature=signature;
    window.dispatchEvent(new CustomEvent('flight-player-status',{detail}));
  }
  let padId='',armed=false,neutralAt=0, selectedIndex=null, gear=0, previousGear=false, previousPad='',lastSent=0,lastMonitor=0;
  let cfg=null, rawPad=null, sweeping=null, preset='auto';
  let playerPollTimer=null, playerPageSuspended=false;
  const reservedGearButtons=new Set([0,1,2,3,5,6,7,8,9,12,13,14,15]);
  let navIdentity='',navPrimed=false,navPrevious={},navFocus=null,navDirection='',navAt=0,navConfirm=null,lastViewShift=0;
  const navVisible=el=>{
    if(!el||el.hidden||el.closest('[hidden]')||!el.getClientRects().length||getComputedStyle(el).visibility==='hidden'||getComputedStyle(el).display==='none')return false;
    for(let ancestor=el.parentElement;ancestor;ancestor=ancestor.parentElement)if(ancestor.matches('details:not([open])')&&!ancestor.querySelector(':scope > summary')?.contains(el))return false;
    return true;
  };
  function playerNavScope(){return document.querySelector('dialog[open]')||($('playerOptions').open?$('playerOptions'):navVisible($('setupPanel'))?$('setupPanel'):document.body.dataset.playerMode==='workspace'?$('playerWorkspace'):null);}
  function playerNavFocus(el){navFocus?.removeAttribute('data-gamepad-focus');navFocus=el;if(el){el.dataset.gamepadFocus='';el.focus({preventScroll:true});el.scrollIntoView({block:'nearest',inline:'nearest'});}}
  function playerNavBack(){
    if(window.FlightPanels?.busy){$('panelDecisionCancel')?.click();return;}
    const advanced=$('setupPanel').querySelector('details[open]');if(advanced){advanced.open=false;return;}
    if($('playerOptions').open){$('playerOptions').open=false;playerNavFocus($('playerOptions').querySelector('summary'));return;}
    if(navVisible($('setupPanel'))){$('setupClose').click();return;}
    if(!mySeat&&!$('joinForm').hidden){$('changeRoom').click();return;}
    $('playerOptions').open=true;stopControl();
  }
  function playerNavigation(p,now){
    const key=p?`${p.index}:${p.id}`:'',value=i=>{const b=p?.buttons?.[i];return typeof b==='number'?b:b?.value||Number(!!b?.pressed)||0;};
    if(key!==navIdentity){navIdentity=key;navPrimed=false;navPrevious={};navConfirm=null;playerNavFocus(null);}
    if(!p||document.hidden)return;
    const buttons={x:value(2)>.6,y:value(3)>.6,back:value(8)>.6,menu:value(9)>.6};
    const sticks=cfg?window.DroneUSBInput.readSticks(p,cfg):null,stickX=sticks?.[0].x??p.axes[0]??0,stickY=sticks? -sticks[0].y:p.axes[1]||0;
    const dx=Number(value(15)>.5)-Number(value(14)>.5)||(Math.abs(stickX)>.6?Math.sign(stickX):0);
    const dy=Number(value(13)>.5)-Number(value(12)>.5)||(Math.abs(stickY)>.6?Math.sign(stickY):0);
    if(!navPrimed){navPrimed=!Object.values(buttons).some(Boolean)&&!dx&&!dy;navPrevious=buttons;return;}
    if(buttons.menu&&!navPrevious.menu){$('playerOptions').open=true;stopControl();navConfirm=null;}
    else if(buttons.y&&!navPrevious.y||buttons.back&&!navPrevious.back){playerNavBack();navConfirm=null;}
    else{
      const root=playerNavScope();
      if(root){
        // Ready becomes temporarily disabled while a navigation button is
        // held. Keep its cursor position; confirmation still requires release
        // and the regular enabled/neutral checks before it can submit.
        const all=[...root.querySelectorAll('button,a[href],input:not([type=hidden]),select,summary')].filter(el=>navVisible(el)&&(!el.disabled||el.id==='ready')&&el.getAttribute('aria-hidden')!=='true');
        if(!all.includes(navFocus))playerNavFocus(all.includes(document.activeElement)?document.activeElement:all.find(el=>el.matches('#ready,#joinSubmit'))||all[0]);
        const direction=Math.abs(dx)>Math.abs(dy)?dx>0?'right':'left':dy>0?'down':dy<0?'up':'';
        if(direction&&(direction!==navDirection||now-navAt>220)&&navFocus){
          if((direction==='left'||direction==='right')&&navFocus.matches('select,input[type=number]')){
            const step=direction==='right'?1:-1;
            if(navFocus.matches('select')){const options=[...navFocus.options].filter(o=>!o.disabled),i=options.indexOf(navFocus.selectedOptions[0]);navFocus.value=options[Math.max(0,Math.min(options.length-1,i+step))]?.value||navFocus.value;}
            else step>0?navFocus.stepUp():navFocus.stepDown();
            navFocus.dispatchEvent(new Event('change',{bubbles:true}));
          }else{
            const r=navFocus.getBoundingClientRect(),v={left:[-1,0],right:[1,0],up:[0,-1],down:[0,1]}[direction];let best,score=Infinity;
            for(const el of all){if(el===navFocus)continue;const b=el.getBoundingClientRect(),x=b.x+b.width/2-r.x-r.width/2,y=b.y+b.height/2-r.y-r.height/2,along=x*v[0]+y*v[1],cross=Math.abs(x*v[1]-y*v[0]);if(along>4&&along+cross*2.5<score){best=el;score=along+cross*2.5;}}
            if(best)playerNavFocus(best);
          }
          navAt=now;
        }
        navDirection=direction;
        if(buttons.x&&!navPrevious.x)navConfirm=navFocus;
        // Ready requires every button released, so confirm on X release.
        if(!buttons.x&&navPrevious.x&&navConfirm){const target=navConfirm;navConfirm=null;if(all.includes(target)&&!target.disabled)target.click();}
      }else{navDirection='';playerNavFocus(null);navConfirm=null;}
    }
    navPrevious=buttons;
  }
  function resetArm(){armed=false;neutralAt=0;previousGear=false;if(role==='player')$('ready').disabled=true;}
  function saveConfig(){if(padId&&cfg)try{localStorage.setItem('lan.pad.'+padId,JSON.stringify({cfg,preset,version:2,gearButton:Number($('gearButton').value)}));}catch{tell('設定已套用；瀏覽器限制儲存，關閉頁面後需重新校正。');}resetArm();}
  function fillMap(){
    const root=$('axisMap');root.replaceChildren();
    const count=Math.max(6,rawPad?.axes.length||0);
    for(const [i,k,title] of [[0,'x','左桿左右'],[0,'y','左桿上下'],[1,'x','右桿左右'],[1,'y','右桿上下']]){
      const box=document.createElement('div'),label=document.createElement('label'),select=document.createElement('select');label.textContent=title;
      options(select,Array.from({length:count},(_,n)=>[String(n),'Axis '+n]),String(cfg.map[i][k]));select.onchange=()=>{cfg.map[i][k]=Number(select.value);saveConfig();};
      const invert=document.createElement('input');invert.type='checkbox';invert.checked=cfg.map[i][k+'i'];invert.setAttribute('aria-label',title+'反向');invert.onchange=()=>{cfg.map[i][k+'i']=invert.checked;saveConfig();};
      label.append(select);box.append(label,invert,document.createTextNode('反向'));root.append(box);
    }
  }
  function initializePad(p){
    rawPad=p;padId=p.id;resetArm();gear=0;let saved;
    try{saved=JSON.parse(localStorage.getItem('lan.pad.'+padId)||'null');}catch{}
    preset=saved?.preset||'auto';$('preset').value=preset;
    cfg=window.DroneUSBInput.normalizeConfig(saved?.cfg||{map:window.DroneUSBInput.presets[window.DroneUSBInput.resolvePreset(p,preset)],dz:.03,cal:{}});
    $('deadzone').value=cfg.dz;$('gearButton').value=saved?.version===2&&Number.isInteger(saved.gearButton)&&saved.gearButton>=0&&saved.gearButton<64&&!reservedGearButtons.has(saved.gearButton)?saved.gearButton:4;fillMap();
  }
  function stopControl(){resetArm();if(mySeat&&ticket&&session)sendInput(false,{lx:0,ly:0,rx:0,ry:0,buttons:[],gear});}
  function sendInput(pad,input){if(socket?.readyState===1&&ticket)socket.send(JSON.stringify({type:'input',session,seq:seq++,epoch,ticket,pad,visible:!document.hidden,input}));}
  function controlSample(p){
    const missing=[...new Set(cfg.map.flatMap(m=>[m.x,m.y]).filter(index=>!Number.isFinite(p.axes[index])))],channelsReady=!missing.length;
    const sticks=window.DroneUSBInput.readSticks(p,cfg),buttons=Array.from(p.buttons,b=>Math.max(0,Math.min(1,b.value)));
    const input={lx:sticks[0].x,ly:sticks[0].y,rx:sticks[1].x,ry:sticks[1].y,buttons,gear};
    const center=channelsReady&&[input.lx,input.ly,input.rx,input.ry].every(v=>Math.abs(v)<=.12)&&buttons.every(v=>v<.35);
    return {missing,channelsReady,buttons,input,center};
  }
  function poll(now){
    if(role!=='player'||playerPageSuspended)return;
    // Control packets must remain fresh even when the 3D view renders slowly.
    playerPollTimer=setTimeout(()=>{playerPollTimer=null;poll(performance.now());},32);
    let pads=[],reason='';
    if(!isSecureContext)reason=accessMode==='online'?'此頁面不是安全遊戲網址。請使用主持人目前顯示的 HTTPS 房間 QR 重新開啟；若網址失效，請主持人重新建立公開連線。':'此區網頁面不是安全來源。請由主持人確認區網憑證與 HTTPS 網址。';
    else if(!navigator.getGamepads)reason='此瀏覽器未提供 Gamepad API。請改用支援該平板搖桿的瀏覽器，並確認作業系統已完成藍牙配對。';
    else try{pads=window.DroneUSBInput.controllerPads(Array.from(navigator.getGamepads()).filter(Boolean));}catch(e){reason=`Gamepad API 受到限制（${e.name}）。請直接開啟此頁面、允許控制器存取，按鍵啟用偵測。`;}
    const listKey=pads.map(p=>`${p.index}:${p.id}`).join('|');
    if($('padSelect').dataset.list!==listKey){$('padSelect').dataset.list=listKey;options($('padSelect'),pads.length?pads.map(p=>[String(p.index),p.id]):[['','尚未偵測搖桿，請先配對並按鍵'] ],String(selectedIndex));}
    if(selectedIndex===null&&pads.length)selectedIndex=pads[0].index;
    const p=pads.find(p=>p.index===selectedIndex);
    const identity=p?`${p.index}:${p.id}`:'';
    if(identity!==previousPad){previousPad=identity;if(p)initializePad(p);else resetArm();}
    rawPad=p;
    let input={lx:0,ly:0,rx:0,ry:0,buttons:[],gear},healthy=false;
    if(p&&cfg){
      if(sweeping)for(let i=0;i<p.axes.length;i++){const c=sweeping[i]||(sweeping[i]={min:0,max:0,center:cfg.cal[i]?.center??p.axes[i]});c.min=Math.min(c.min,p.axes[i]);c.max=Math.max(c.max,p.axes[i]);}
      const sample=controlSample(p),{missing,channelsReady,buttons,center}=sample;input=sample.input;
      if(!armed){if(center&&!document.hidden&&!window.FlightPanels?.busy&&!$('playerOptions').open){if(!neutralAt)neutralAt=now;if(now-neutralAt>500)armed=true;}else neutralAt=0;}
      const button=buttons[Number($('gearButton').value)]>.65;
      if(armed&&button&&!previousGear){gear=(gear+1)%3;input.gear=gear;}previousGear=button;
      healthy=channelsReady&&armed&&!document.hidden&&!sweeping&&!window.FlightPanels?.busy&&!$('playerOptions').open;
      const viewDirection=Number((buttons[7]||0)>.35)-Number((buttons[6]||0)>.35);
      if(healthy&&room?.status==='running'&&viewDirection)post({type:'player-view-shift',direction:viewDirection,delta:Math.min(.1,(now-(lastViewShift||now))/1000)});
      lastViewShift=now;
      reason=!channelsReady?`控制器沒有配置所需的 Axis ${missing.join('、')}。請調整四軸配置，或改用支援四軸的搖桿。`:healthy?'搖桿正常。確認方向與按鍵後，回中並按「準備完成」。':'請放開全部按鈕，左右雙桿回中維持半秒，確認恢復操作。';
      if(now-lastMonitor>100){text('padName',p.id);text('padTest',`左桿 左右 ${input.lx.toFixed(2)} / 升降 ${input.ly.toFixed(2)}\n右桿 左右 ${input.rx.toFixed(2)} / 前後 ${input.ry.toFixed(2)}\n原始軸 ${Array.from(p.axes,v=>v.toFixed(2)).join(' ')}\n按鈕 ${buttons.map((b,i)=>b>.35?'B'+i+':'+b.toFixed(1):'').filter(Boolean).join(' ')||'全部放開'}\n目前檔位 ${['低速 33.3%','中速 66.6%','高速 100%'][gear]}`);lastMonitor=now;}
      const seat=room?.seats.find(s=>s.id===mySeat),synced=seat?.player?.cameraSynced&&seat?.player?.viewSynced;
      const transportReady=socket?.readyState===1&&!!ticket&&!!session;
      $('ready').disabled=!transportReady||!healthy||!center||!mySeat||!synced||!['lobby','paused'].includes(room?.status);
      playerUiSample={present:true,healthy,channelsReady,center,synced:!!synced,sticks:[{x:input.lx,y:input.ly},{x:input.rx,y:input.ry}]};
      if(healthy&&center&&!synced)reason='搖桿已回中，正在同步 3D 視角；完成後即可準備。';
    }else{playerUiSample={present:false,healthy:false,channelsReady:false,center:false,synced:false,sticks:[{x:0,y:0},{x:0,y:0}]};$('ready').disabled=true;reason=reason||'尚未偵測搖桿：請在系統設定配對，回到本頁並按任一搖桿按鍵。若仍無訊號，需確認此平板／瀏覽器／搖桿組合是否相容。';}
    text('padReason',reason);
    const prepared=room?.seats.find(s=>s.id===mySeat)?.player?.ready;
    const flightState=lastSnapshot?.rigs?.find(r=>r.id===mySeat),flightMode=flightState?.braking?'・煞車／懸停':flightState?.backflip?'・後空翻':flightState?.headless?'・無頭模式':'';
    text('padStatus',healthy?`搖桿正常・${prepared?'已準備': '未準備'}・${['低','中','高'][gear]}速${flightMode}`:'搖桿未就緒');
    playerNavigation(p,now);
    healthy=healthy&&armed&&!window.FlightPanels?.busy&&!$('playerOptions').open;
    if(mySeat&&now-lastSent>=32){sendInput(healthy,input);lastSent=now;}
    publishPlayerStatus(now);
  }
  if(role==='player'){
    window.FlightPlayerStatus=Object.freeze({snapshot:playerStatusSnapshot});
    $('leaveRoom').onclick=async()=>{if(!mySeat){location.href='/';return;}if(await FlightPanels.confirm({title:'離開這個房間？',description:'你的席位將釋放。若比賽正在進行，會自動暫停並等待主持人處理。',acceptLabel:'離開並釋放席位',danger:true})){stopControl();send({type:'leave'});}};
    $('code').value=currentCode;
    $('codeForm').onsubmit=e=>{e.preventDefault();const value=$('code').value.trim().toUpperCase();if(!/^[A-HJ-NP-Z2-9]{6}$/.test(value)){$('code').setAttribute('aria-invalid','true');tell('請輸入 6 位房碼，不含 I、O、0、1。');return;}currentCode=value;$('code').value=value;$('code').removeAttribute('aria-invalid');tell('');send({type:'inspect',code:currentCode});};
    $('seat').onchange=confirmSeat;
    $('joinForm').onsubmit=e=>{e.preventDefault();confirmSeat();if($('joinSubmit')?.disabled){tell('目前無法加入此席位，請選擇其他空位或聯絡主持人。');return;}tell('');send({type:'join',code:currentCode,device,name:$('nickname').value,seat:$('seat').value});};
    $('setupOpen').onclick=()=>{$('setupPanel').hidden=false;stopControl();};$('setupClose').onclick=()=>{$('setupPanel').hidden=true;};
    $('ready').onclick=()=>{let live;try{live=window.DroneUSBInput.controllerPads(Array.from(navigator.getGamepads()).filter(Boolean)).find(p=>p.index===selectedIndex);}catch{stopControl();tell('無法讀取控制器，請重新偵測後再準備。');return;}const sample=live&&cfg&&`${live.index}:${live.id}`===previousPad?controlSample(live):null,seat=room?.seats.find(s=>s.id===mySeat);if(socket?.readyState!==1||!ticket||!session||!['lobby','paused'].includes(room?.status)||!isSecureContext||!sample?.channelsReady||!sample.center||!armed||document.hidden||sweeping||window.FlightPanels?.busy||!seat?.player?.cameraSynced||!seat?.player?.viewSynced){stopControl();tell('請雙桿回中，等控制器與視角就緒後再準備。');return;}sendInput(true,sample.input);send({type:'ready'});readyRequested=true;$('setupPanel').hidden=true;tell('已送出準備，等待主持人開賽／恢復。');};
    $('padSelect').onchange=()=>{selectedIndex=Number($('padSelect').value);previousPad='';stopControl();};
    $('preset').onchange=()=>{if(!rawPad)return;preset=$('preset').value;cfg=window.DroneUSBInput.normalizeConfig({map:window.DroneUSBInput.presets[window.DroneUSBInput.resolvePreset(rawPad,preset)],dz:Number($('deadzone').value),cal:{}});fillMap();saveConfig();};
    $('deadzone').onchange=()=>{if(cfg){cfg.dz=Math.max(0,Math.min(.2,Number($('deadzone').value)||0));saveConfig();}};
    $('gearButton').onchange=()=>{const index=Number($('gearButton').value);if(!Number.isInteger(index)||index<0||index>63||reservedGearButtons.has(index)){$('gearButton').value=4;tell('此按鈕已用於飛行或選單，換檔已還原為 LB（B4）。');}saveConfig();};
    $('calCenter').onclick=()=>{if(!rawPad||!cfg)return;cfg.cal={};rawPad.axes.forEach((v,i)=>{if(v>-.94&&v<.94)cfg.cal[i]={min:-1,center:v,max:1};});saveConfig();tell('已記錄中立點；請確認四個方向訊號皆接近 0。');};
    $('calSweep').onclick=()=>{if(!rawPad)return;sweeping={};$('calFinish').disabled=false;stopControl();tell('請將所有搖桿推至各方向最大行程，完成後回中，再按「完成校正」。');};
    $('calFinish').onclick=()=>{if(!sweeping||!cfg)return;const valid=Object.fromEntries(Object.entries(sweeping).filter(([,c])=>c.center-c.min>.05&&c.max-c.center>.05));cfg.cal={...cfg.cal,...valid};sweeping=null;$('calFinish').disabled=true;saveConfig();tell('校正已儲存；未走完正負行程的軸保留原設定。請再次測試方向。');};
    if($('calCancel'))$('calCancel').onclick=()=>{if(!sweeping)return;sweeping=null;$('calFinish').disabled=true;stopControl();tell('已取消本次校正，保留原本已儲存的設定。');};
    window.addEventListener('flight-panel-modal',event=>{if(event.detail.open)stopControl();});
    const releasePlayer=()=>{navPrimed=false;navPrevious={};navConfirm=null;stopControl();};
    document.addEventListener('visibilitychange',()=>{if(document.hidden)releasePlayer();});window.addEventListener('blur',releasePlayer);
    const navStyle=document.createElement('style');navStyle.textContent='[data-gamepad-focus]{outline:3px solid #f4c75b!important;outline-offset:3px!important}';document.head.append(navStyle);
    window.addEventListener('pagehide',()=>{playerPageSuspended=true;clearTimeout(playerPollTimer);playerPollTimer=null;stopControl();});
    window.addEventListener('pageshow',event=>{if(event.persisted){playerPageSuspended=false;if(playerPollTimer===null)poll(performance.now());}});
    poll(performance.now());
  }
  $('fullscreen').onclick=()=>{if(document.fullscreenElement)document.exitFullscreen();else document.documentElement.requestFullscreen?.().catch(()=>tell('此瀏覽器不支援網頁全螢幕，請使用瀏覽器的全螢幕功能。'));};
  $('sound').onclick=()=>{audio ||= new (window.AudioContext||window.webkitAudioContext)();audio.resume();sound=!sound;text('sound',sound?'關閉哨音':'開啟哨音');};
  function whistle(long){if(!sound||!audio)return;const osc=audio.createOscillator(),gain=audio.createGain(),time=audio.currentTime;osc.frequency.value=2650;gain.gain.setValueAtTime(.05,time);gain.gain.exponentialRampToValueAtTime(.0001,time+(long?.9:.4));osc.connect(gain);gain.connect(audio.destination);osc.start();osc.stop(time+1);}
  setInterval(syncCover,250);void loadConnectionConfig();loginPromise=consumeKey().then(connect);
})();
