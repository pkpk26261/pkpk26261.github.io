/* Player presentation consumes sanitized owner state; it never samples or sends input. */
(() => {
 'use strict';if(document.body.dataset.role!=='player')return;
 const $=id=>document.getElementById(id),api=window.FlightPlayerStatus;
 if(!api)return;
 document.body.classList.add('player-ui-ready');
 const compactContext=matchMedia('(max-width:850px)');const syncContext=()=>{$('roomContextDetails').open=!compactContext.matches;};compactContext.addEventListener('change',syncContext);syncContext();
 const nativeSeat=$('seat');nativeSeat.tabIndex=-1;nativeSeat.setAttribute('aria-hidden','true');
 let latest=api.snapshot(),seatSignature='',preview=false,editingCode=false,lastStage='';
 const assign=(id,value)=>{const e=$(id);if(e.textContent!==String(value))e.textContent=String(value);};
 function condition(id,ok,label){const e=$(id);e.dataset.state=ok?'complete':'pending';e.querySelector('small').textContent=label;}
 function seatChoices(s){
  const locked=s.room?.locked||!['lobby','paused'].includes(s.room?.status),selected=nativeSeat.value;
  const key=JSON.stringify([s.seats,locked,selected,s.connected]);
  if(key===seatSignature)return;seatSignature=key;
  const focus=document.activeElement?.dataset.seatId;
  const picker=$('seatPicker');picker.replaceChildren();
  for(const [team,label] of [['red','紅隊'],['blue','藍隊']]){
   const column=document.createElement('section');column.className='seat-team '+team;column.dataset.team=team;
   const title=document.createElement('h3');title.textContent=label;column.append(title);
   for(const seat of s.seats.filter(seat=>seat.team===team)){
    const b=document.createElement('button');b.type='button';b.className='seat-choice';b.dataset.team=team;b.dataset.seatId=seat.id;b.setAttribute('role','radio');
    const unavailable=seat.ai||seat.taken||locked||!s.connected;b.disabled=unavailable;b.setAttribute('aria-checked',String(!unavailable&&seat.id===selected));b.tabIndex=!unavailable&&seat.id===selected?0:-1;
    const name=document.createElement('strong');name.textContent=seat.label.replace(/^紅隊\s*|^藍隊\s*/,'');const state=document.createElement('small');state.textContent=seat.ai?'AI':seat.taken?'已占用':locked?'暫不開放':'可加入';b.append(name,state);
    b.onclick=()=>{nativeSeat.value=seat.id;nativeSeat.dispatchEvent(new Event('change',{bubbles:true}));seatSignature='';seatChoices(api.snapshot());};
    b.onkeydown=event=>{if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home','End'].includes(event.key))return;event.preventDefault();const all=[...picker.querySelectorAll('button:not(:disabled)')],i=all.findIndex(v=>v.dataset.seatId===seat.id),next=event.key==='Home'?all[0]:event.key==='End'?all.at(-1):all[(i+(['ArrowRight','ArrowDown'].includes(event.key)?1:all.length-1))%all.length];if(next){const id=next.dataset.seatId;next.click();picker.querySelector(`[data-seat-id="${id}"]`)?.focus();}};
    column.append(b);
   }picker.append(column);
  }
  const enabled=[...picker.querySelectorAll('button:not(:disabled)')];if(enabled.length&&!enabled.some(b=>b.tabIndex===0))enabled[0].tabIndex=0;
  if(focus){const target=picker.querySelector(`[data-seat-id="${focus}"]:not(:disabled)`)||enabled[0]||$('playerEntryTitle');if(!target.matches('button'))target.tabIndex=-1;target.focus({preventScroll:true});}
 }
 function render(s){
  latest=s;
  const room=s.room,joined=s.joined&&s.seat,prepared=!!s.seat?.prepared;
  const terminal=!!s.exitReason||['ended','closed'].includes(room?.status);
  let stage='join';
  if(terminal)stage='terminal';else if(joined){if(s.setupOpen)stage='setup';else if(!s.connected)stage='waiting';else if(s.readyRequested)stage='pending';else if(room.status==='running')stage='play';else if(room.status==='paused'&&!prepared)stage='paused';else stage='waiting';}
  if(!joined)preview=false;if(stage==='play')preview=false;
  const mode=stage==='play'||preview&&joined&&!terminal?'game':'workspace';
  document.body.dataset.playerStage=stage;document.body.dataset.playerMode=mode;document.body.dataset.roomKnown=String(!!room);document.body.dataset.playerJoined=String(!!joined);document.body.dataset.playerPrepared=String(prepared);
  const feedbackHost=mode==='game'?document.body:$('workspaceFeedback');if($('message').parentElement!==feedbackHost)feedbackHost.append($('message'));
  $('playerWorkspace').hidden=mode==='game';$('waitingPanel').hidden=!['waiting','pending','paused'].includes(stage);$('playerEndPanel').hidden=!terminal;
  $('joinPanel').hidden=stage!=='join';if(terminal)$('setupPanel').hidden=true;
  $('preparationOpen').hidden=mode!=='game'||stage==='play';
  $('playerGame').setAttribute('aria-hidden',String(mode!=='game'));
  $('scene').tabIndex=mode==='game'?0:-1;
  assign('playerPageTitle',({join:'加入比賽',setup:'確認操控',pending:'確認準備中',waiting:prepared?'準備上場':'準備進度',paused:'重新準備',terminal:'比賽狀態',play:'比賽進行中'})[stage]);
  const currentStep=stage==='join'?'join':stage==='setup'||stage==='paused'||stage==='waiting'&&!prepared?'setup':'waiting';
  document.querySelector('.player-progress').hidden=terminal;
  for(const li of document.querySelectorAll('[data-player-step]')){const current=currentStep===li.dataset.playerStep;if(current)li.setAttribute('aria-current','step');else li.removeAttribute('aria-current');}
  assign('playerRoomName',room?.name||'尚未選擇房間');assign('playerRoomCode',room?.code||s.code||'—');assign('playerArena',room?.arenaName||'確認房間後顯示');
  assign('playerRoomState',({lobby:'等待開賽',running:'比賽中',paused:'已暫停',ended:'已結束',closed:'已關閉'})[room?.status]||'等待確認');
  $('playerRoomState').dataset.state=room?.status||'unknown';
  if(!joined)assign('identity','尚未加入');
  $('leaveRoom').textContent=joined?'離開房間':'返回賽事大廳';
  $('entryBack').hidden=!!joined;
  $('connectionNotice').hidden=s.connected;
  assign('connectionNotice',joined?'連線中斷，操作已停止。恢復後請重新確認控制器與準備。':'正在連線至主機，請保持網路與此頁面開啟。');
  if(stage==='join'){
   const selecting=room&&!editingCode&&!$('joinForm').hidden;
   assign('playerEntryTitle',selecting?'選擇你的席位':'加入房間');assign('joinIntro',selecting?'填寫暱稱，再選擇紅隊或藍隊空位。':'輸入房碼，找到這一場比賽。');
   seatChoices(s);
   const seats=s.seats.filter(seat=>!seat.ai&&!seat.taken),locked=room?.locked||room&&!['lobby','paused'].includes(room.status),allAI=room&&s.seats.every(seat=>seat.ai);
   assign('availableSeatCount',room?locked?'暫不開放':allAI?'AI 對戰':seats.length+' 個空位':'');
   const reason=locked?'本房間暫停接受新玩家，請聯絡主持人。':allAI?'本場為全 AI 對戰，可以直接觀看轉播。':!seats.length&&room?'真人席位已滿，可先觀看轉播或選擇其他房間。':'';
   $('joinAvailability').hidden=!reason;assign('joinAvailability',reason);
   $('joinSubmit').disabled=!s.connected||!room||!!locked||!seats.length||!nativeSeat.selectedOptions[0]||nativeSeat.selectedOptions[0].disabled;
   $('watchInstead').hidden=!reason||!room||!s.connected;if(room&&s.connected)$('watchInstead').href='/watch?code='+encodeURIComponent(room.code);else $('watchInstead').removeAttribute('href');
  }
  condition('checkController',s.sample.present&&s.sample.channelsReady,s.sample.present?s.sample.channelsReady?'已連線':'請調整配置':'等待連線');
  condition('checkCenter',s.sample.healthy&&s.sample.center,s.sample.healthy&&s.sample.center?'已確認':'放開雙桿與按鈕');
  condition('checkView',!!s.seat?.synced,s.seat?.synced?'已同步':'等待主機同步');
  const readyText=!s.connected?'等待連線恢復後，再重新準備。':s.readyAllowed?'操控與視角已就緒，可以準備上場。':!s.sample.present?'先接上控制器，並按一下按鈕喚醒。':!s.sample.channelsReady?'控制器配置尚未完整，請展開下方設定。':!s.sample.healthy||!s.sample.center?'請放開按鈕與雙桿，回中保持半秒。':!s.seat?.synced?'正在同步比賽視角，完成後即可準備。':'等待主機確認目前的準備條件。';
  assign('readyHint',readyText);$('padReason').hidden=s.readyAllowed||!s.connected;
  assign('controllerReadiness',s.readyAllowed?'已就緒':s.sample.present?'確認方向與回中':'等待搖桿');$('controllerReadiness').dataset.state=s.readyAllowed?'ready':'check';
  s.sample.sticks.forEach((stick,i)=>{$(i?'rightStickDot':'leftStickDot').style.transform=`translate(${Math.max(-1,Math.min(1,stick.x))*24}px,${Math.max(-1,Math.min(1,stick.y))*24}px)`;});
  if(['waiting','pending','paused'].includes(stage)){
   const offline=!s.connected;
   assign('waitingKicker',offline?'RECONNECTING':stage==='pending'?'CONFIRMING':prepared?'READY TO PLAY':'PREFLIGHT');
   assign('waitingTitle',offline?'正在恢復連線':stage==='pending'?'正在確認準備':prepared?room.status==='paused'?'已準備，等待恢復':'已準備，等待開賽':stage==='paused'?'比賽暫停，請重新準備':'尚未完成準備');
   assign('waitingDescription',offline?'恢復連線後會保留原席位，請再次確認控制器。':stage==='pending'?'準備要求已送出，正在等待主機回覆。':prepared?'請保持頁面開啟，等待主持人開始或恢復比賽。':'請先檢查控制器、雙桿回中與比賽視角。');
   assign('waitingSummary',s.seat?.label||'');
   $('reviewController').disabled=!s.connected||stage==='pending';$('showMatchView').hidden=offline||stage==='pending';
   assign('waitingNote',prepared?'等哨音後上推起飛；重新檢查控制器會取消目前準備。':'確認操控後，再按「準備完成」。');
  }
  if(terminal){assign('playerEndTitle',s.exitReason?'席位或房間已結束':room?.status==='ended'?'本場比賽已結束':'房間已關閉');assign('playerEndDescription',s.exitReason||room?.reason||'請返回賽事大廳，選擇下一場比賽。');$('retryPlayerPage').hidden=room?.status==='closed';}
  if(stage!==lastStage){lastStage=stage;if(mode==='workspace'&&['setup','waiting','pending','terminal','paused'].includes(stage)){const heading=stage==='setup'?$('controllerReadyTitle'):terminal?$('playerEndTitle'):$('waitingTitle');heading.tabIndex=-1;heading.focus({preventScroll:true});window.scrollTo({top:0,behavior:'instant'});}}
 }
 $('changeRoom').onclick=()=>{editingCode=true;$('codeForm').hidden=false;$('joinForm').hidden=true;$('joinError').textContent='';render(api.snapshot());$('code').focus();};
 $('codeForm').addEventListener('submit',()=>{editingCode=false;});
 $('code').addEventListener('input',()=>{$('code').removeAttribute('aria-invalid');$('joinError').textContent='';});
 $('reviewController').onclick=()=>{preview=false;$('setupOpen').click();render(api.snapshot());};
 $('showMatchView').onclick=event=>{event.preventDefault();preview=true;render(api.snapshot());window.scrollTo(0,0);};
 $('preparationOpen').onclick=()=>{preview=false;render(api.snapshot());};
 $('retryPlayerPage').onclick=()=>location.reload();
 $('playerBrand').onclick=event=>{if(api.snapshot().joined){event.preventDefault();$('leaveRoom').click();}};
 document.addEventListener('keydown',event=>{if(event.key==='Escape'&&$('playerOptions').open&&!document.querySelector('dialog[open]')){$('playerOptions').open=false;$('playerOptions').querySelector('summary').focus();}});
 window.addEventListener('flight-player-status',event=>render(event.detail));
 render(latest);
})();
