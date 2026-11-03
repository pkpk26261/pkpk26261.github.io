(() => {
 'use strict';
 const $=id=>document.getElementById(id),el=(tag,text,className)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(className)n.className=className;return n;};
 let joinAvailability={ready:false,mode:'online',origin:''};
 const names={lobby:'等待開賽',running:'比賽中',paused:'已暫停',ended:'已結束',closed:'已關閉'};
 const icon=name=>{const svg=document.createElementNS('http://www.w3.org/2000/svg','svg'),use=document.createElementNS(svg.namespaceURI,'use');svg.classList.add('lan-icon');svg.setAttribute('aria-hidden','true');use.setAttribute('href','/apps/drone-soccer/assets/game/lan-icons.svg#'+name);svg.append(use);return svg;};
 const phaseName=r=>r.status==='running'?({ready:'開賽倒數',play:'比賽中',break:'局間休息',setEnd:'本局結束',final:'終場',golden:'驟死賽'}[r.match?.phase]||'正式賽'):names[r.status];
 const link=(text,url,primary=false)=>{const a=el('a',text,'action-link'+(primary?' primary':''));a.href=url;return a;};
 function replaceLiveContent(container,children,fallback){
  const active=document.activeElement,inside=container.contains(active),key=active?.dataset.liveKey,code=active?.closest('[data-room]')?.dataset.room;
  container.replaceChildren(...children);
  if(!inside)return;
  const actions=[...container.querySelectorAll('[data-live-key]')].filter(a=>!a.matches(':disabled'));
  const next=actions.find(a=>a.dataset.liveKey===key)||actions.find(a=>code&&a.closest('[data-room]')?.dataset.room===code)||fallback;
  next?.focus({preventScroll:true});
 }
 const request=async(url,options)=>{const response=await fetch(url,options);const data=await response.json();if(!response.ok)throw Error(data.error||'服務暫時無法處理要求');return data;};
 function counts(data){if(!$('serviceCounts'))return;
  if(document.body.dataset.serviceRole==='admin'){const free=data.arenas.filter(a=>a.enabled&&!a.roomCode).length,rows=[['grid','賽場',data.arenas.length,free+' 個空閒'],['users','房間',data.counts.rooms,data.rooms.filter(r=>r.status==='lobby').length+' 間等待開賽'],['broadcast','進行中',data.counts.running,'隨時切換即時觀賽'],['drone','在線玩家',data.counts.players,'真人席位連線人數']];$('serviceCounts').replaceChildren(...rows.map(([name,label,n,hint])=>{const card=el('div',undefined,'admin-stat'),mark=el('span',undefined,'stat-icon'),copy=el('div',undefined,'stat-copy');mark.append(icon(name));copy.append(el('span',label),el('strong',n));card.append(mark,copy,el('small',hint));return card;}));return;}
  const rows=[['場可加入',data.rooms.filter(r=>r.canJoin).length],['場進行中',data.counts.running],['位玩家在線',data.counts.players]];$('serviceCounts').replaceChildren(...rows.map(([label,n])=>{const card=el('div');card.append(el('strong',n),document.createTextNode(label));return card;}));}
 function publicRoomCard(room){
  const card=el('article',undefined,'room-card');card.dataset.room=room.code;card.dataset.status=room.status;
  const main=el('div',undefined,'room-card-main'),header=el('div',undefined,'room-card-header');
  header.append(el('span',phaseName(room)||room.status,'badge '+room.status));
  if(room.canJoin)header.append(el('span','開放加入','availability-tag'));
  main.append(header,el('h3',room.name));
  const meta=el('div',undefined,'room-meta');meta.append(el('span',room.code,'room-code'),el('span','·','room-meta-separator'),el('span',room.arenaName,'room-arena'));main.append(meta);
  const roster=el('div',undefined,'room-roster');
  for(const [team,label] of [['red','紅隊'],['blue','藍隊']]){const row=el('div',undefined,'room-team '+team),composition=el('strong',room.roster[team].humans+' 人');composition.append(el('small',' ＋ '+room.roster[team].ai+' AI'));row.append(el('i',undefined,'team-dot'),el('span',label),composition);roster.append(row);}
  const state=el('div',undefined,'room-state-summary');
  if(['running','paused','ended'].includes(room.status)){const score=el('div',undefined,'room-score');score.setAttribute('aria-label',`比分：紅隊 ${room.scores.red}，藍隊 ${room.scores.blue}`);score.append(el('strong',room.scores.red,'red'),el('span',':'),el('strong',room.scores.blue,'blue'));state.append(score,el('small',room.status==='ended'?'最終比分':'即時比分'));}
  else if(room.humans){state.append(el('strong',room.joined+' / '+room.humans),el('small','玩家已加入'));}
  else state.append(el('strong','AI 對戰','ai-view'),el('small','開放觀賽'));
  const actions=el('div',undefined,'card-actions');
  if(!joinAvailability.ready){const pending=el('button','連線準備中');pending.disabled=true;actions.append(pending);}
  else{
   if(room.canJoin)actions.append(link('加入比賽',joinAvailability.origin+'/play?code='+room.code,true));
   else if(room.status!=='ended')actions.append(el('span',room.locked?'加入已鎖定':room.humans?'席位已滿':'全 AI 對戰','join-unavailable'));
   actions.append(link(room.status==='ended'?'查看結果':'觀看轉播',joinAvailability.origin+'/watch?code='+room.code));
  }
  for(const action of actions.querySelectorAll('a,button')){action.dataset.liveKey='room:'+room.code+':'+(action.tagName==='A'?new URL(action.href).pathname:'unavailable');action.setAttribute('aria-label',action.textContent+'：'+room.name+'（'+room.code+'）');}
  card.append(main,roster,state,actions);return card;
 }
 function roomCard(room,manage=false,close){
  if(!manage)return publicRoomCard(room);
  const card=el('article',undefined,'room-card');card.dataset.room=room.code;
  if(!manage){const art=el('div',undefined,'room-keyart'),image=el('img'),label=el('span',room.arenaName,'room-arena-tag');image.src='/apps/drone-soccer/assets/lobby/arena-scene.png';image.alt='';image.loading='lazy';art.append(image,label,el('span','RED  /  BLUE','room-versus'));card.append(art);}
  const header=el('div',undefined,'room-card-header'),title=el('div');title.append(el('h3',room.name),el('span',room.code,'room-code'));header.append(title,el('span',phaseName(room)||room.status,'badge '+room.status));card.append(header,el('p',room.arenaName,'arena-label'));
  const r=room.roster;card.append(el('p',`紅隊 ${r.red.humans} 真人＋${r.red.ai} AI ／ 藍隊 ${r.blue.humans} 真人＋${r.blue.ai} AI`,'lineup'));
  const score=el('div',undefined,'room-score');score.append(el('span','紅隊','red'),el('strong',room.scores.red,'red'),el('span',':'),el('strong',room.scores.blue,'blue'),el('span','藍隊','blue'));card.append(score);
  card.append(el('p',room.status==='ended'?(room.match?.winReason||room.reason):`${room.joined} / ${room.humans} 真人已加入・${room.ready} 已準備${room.locked?'・已鎖定新加入':''}`,'room-progress'));
  const actions=el('div',undefined,'card-actions');
  if(manage){const watch=link(room.status==='ended'?'查看結果':'即時觀賽','/host?code='+room.code+'&panel=room&view=broadcast',true);watch.prepend(icon('broadcast'));actions.append(watch,link('主持人管理','/host?code='+room.code),link('QR／邀請','/host?code='+room.code+'&panel=invite'));const b=el('button','關閉','danger');b.onclick=()=>close(room);actions.append(b);}
  else{if(!joinAvailability.ready){const waiting=el('button','公開連線準備中');waiting.disabled=true;actions.append(waiting);}else{if(room.canJoin)actions.append(link('加入比賽',joinAvailability.origin+'/play?code='+room.code,true));else if(room.status!=='ended'){const b=el('button',room.locked?'加入已鎖定':room.humans?'席位已滿':'全 AI 觀戰');b.disabled=true;actions.append(b);}actions.append(link(room.status==='ended'?'查看結果':'觀看轉播',joinAvailability.origin+'/watch?code='+room.code));}}
  for(const [i,action] of [...actions.children].entries()){action.dataset.liveKey='room:'+room.code+':'+(manage?['watch','manage','invite','close'][i]:action.tagName==='A'?new URL(action.href).pathname:'unavailable');action.setAttribute('aria-label',action.textContent+'：'+room.name+'（'+room.code+'）');}
  card.append(actions);return card;
 }
 function resultRows(rows){return rows.map(room=>{const row=el('div',undefined,'result-row');row.append(el('strong',room.name),el('span',room.arenaName),el('span',`紅 ${room.scores.red} : ${room.scores.blue} 藍`),el('span',names[room.status]||'已結束'));if(room.status==='ended'){const action=link('查看結果','/watch?code='+room.code);action.dataset.liveKey='result:'+room.code;action.setAttribute('aria-label','查看結果：'+room.name+'（'+room.code+'）');row.append(action);}return row;});}
 window.FlightService={$,el,names,link,request,counts,roomCard,resultRows,icon,phaseName,replaceLiveContent};
 if(document.body.dataset.serviceRole!=='public')return;
 let service,ws,retry,configRetry,configurationStopped=false;
 async function loadJoinAvailability(){
  clearTimeout(configRetry);if(configurationStopped)return;
  try{
   const config=await request('/api/config',{cache:'no-store'}),online=config.accessMode==='online';
   const ready=online?config.publicReady===true&&config.directJoin===true:true;
   let origin='';if(ready){const url=new URL(config.origin);if(url.protocol!=='https:'||url.username||url.password||url.pathname!=='/'||url.search||url.hash)throw Error('公開遊戲入口尚未就緒');origin=url.origin;}
   joinAvailability={ready,mode:online?'online':'legacy',origin};
  }catch{joinAvailability={ready:false,mode:'online',origin:''};}
  $('quickJoin').querySelector('button[type=submit]').disabled=!joinAvailability.ready;
  $('entryHint').textContent=joinAvailability.ready?(joinAvailability.mode==='online'?'需保持網際網路連線，免安裝憑證。':'區網模式：與主機連上同一網路，並完成裝置信任。'):'正在建立連線，完成後即可加入。';
  $('arrivalNetworkTitle').textContent=joinAvailability.mode==='online'?'保持網際網路連線':'連上現場網路';
  $('arrivalNetworkDetail').textContent=joinAvailability.mode==='online'?'主機與玩家都需連網':'使用已完成信任的區網裝置';
  if(service)renderSecondaryContents();renderRooms();
  if(!configurationStopped)configRetry=setTimeout(loadJoinAvailability,joinAvailability.ready?5000:2500);
 }
 window.addEventListener('pagehide',()=>{configurationStopped=true;clearTimeout(configRetry);});
 window.addEventListener('pageshow',event=>{if(event.persisted){configurationStopped=false;void loadJoinAvailability();}});
 function render(data){service=data;$('siteName').textContent=data.siteName;counts(data);$('updatedAt').textContent='即時更新 '+new Date().toLocaleTimeString('zh-TW');renderSecondaryContents();renderRooms();}
 function renderSecondaryContents(){
  const data=service;
  replaceLiveContent($('arenaList'),data.arenas.map(a=>{const room=data.rooms.find(r=>r.code===a.roomCode),card=el('article',undefined,'arena-card '+(!a.enabled?'disabled':room?'busy':'free'));const head=el('div');head.append(el('h3',a.name),el('span',!a.enabled?'停用':room?'使用中':'空閒','badge'));card.append(head,el('p',a.note||'正式賽'));if(room&&joinAvailability.ready){const action=link(room.name,joinAvailability.origin+'/watch?code='+room.code);action.dataset.liveKey='arena:'+a.id;action.setAttribute('aria-label','觀看'+a.name+'：'+room.name);card.append(action);}else card.append(el('span',room?'連線準備中':a.enabled?'等待主持人安排比賽':'暫不開放','muted'));return card;}),$('roomSearch'));
  const seen=new Set(),rows=[...data.rooms.filter(r=>r.status==='ended'),...data.history].sort((a,b)=>(b.endedAt||b.closedAt||0)-(a.endedAt||a.closedAt||0)).filter(r=>!seen.has(r.code)&&seen.add(r.code)).slice(0,10);
  $('recentResults').hidden=!rows.length;$('resultCount').textContent=rows.length+' 場';
  const results=resultRows(rows);results.forEach((row,index)=>{const action=row.querySelector('a');if(!action)return;if(joinAvailability.ready)action.href=joinAvailability.origin+'/watch?code='+rows[index].code;else action.replaceWith(el('span','連線準備中','muted'));});
  replaceLiveContent($('resultList'),results,$('roomSearch'));
 }
 function renderRooms(){
  if(!service)return;
  const search=$('roomSearch').value.trim().toLowerCase(),filter=$('roomFilter').value;
  const rows=service.rooms.filter(r=>(filter==='all'||(filter==='active'?!['ended','closed'].includes(r.status):r.status===filter))&&[r.name,r.code,r.arenaName].join(' ').toLowerCase().includes(search));
  const priority=r=>r.canJoin?0:r.status==='running'?1:r.status==='paused'?2:r.status==='lobby'?3:4;
  rows.sort((a,b)=>priority(a)-priority(b)||(b.createdAt||0)-(a.createdAt||0));
  replaceLiveContent($('roomList'),rows.map(r=>roomCard(r)),$('roomSearch'));$('roomEmpty').hidden=!!rows.length;
  const filtered=!!search||filter!=='active';const summary=filtered?'找到 '+rows.length+' 場符合條件的比賽':rows.length+' 場目前賽事';
  if($('roomSearchStatus').textContent!==summary)$('roomSearchStatus').textContent=summary;
  $('roomEmptyTitle').textContent=filtered?'沒有符合條件的比賽':'目前沒有開放中的比賽';
  const hasResults=service.rooms.some(r=>r.status==='ended')||service.history.length;
  $('roomEmptyDetail').textContent=filtered?'換個房名、房碼或狀態試試看。':hasResults?'目前賽事已結束，可展開下方「最近賽果」查看結果。':'主持人建立房間後，就會顯示在這裡。';
  $('clearRoomSearch').hidden=!filtered;
 }
 $('roomSearch').oninput=renderRooms;$('roomFilter').onchange=renderRooms;
 $('clearRoomSearch').onclick=()=>{$('roomSearch').value='';$('roomFilter').value='active';renderRooms();$('roomSearch').focus();};
 for(const role of document.querySelectorAll('[name=entryRole]'))role.addEventListener('change',()=>{$('joinActionLabel').textContent=role.checked&&role.value==='watch'?'觀看轉播':'加入比賽';});
 $('entryCode').oninput=()=>{$('entryFeedback').textContent='';$('entryCode').removeAttribute('aria-invalid');};
 $('quickJoin').onsubmit=event=>{event.preventDefault();if(!joinAvailability.ready){$('entryFeedback').textContent='公開遊戲入口尚未就緒，請稍候再試。';return;}const code=$('entryCode').value.trim().toUpperCase();if(!/^[A-HJ-NP-Z2-9]{6}$/.test(code)){$('entryFeedback').textContent='請輸入 6 位房間代碼，不含 I、O、0、1。';$('entryCode').setAttribute('aria-invalid','true');$('entryCode').focus();return;}const role=document.querySelector('[name=entryRole]:checked').value;location.href=joinAvailability.origin+'/'+role+'?code='+code;};
 function connect(){ws=new WebSocket('wss://'+location.host+'/socket');ws.onopen=()=>{$('serviceConnection').textContent='服務已連線';$('serviceConnection').dataset.state='connected';$('serviceMessage').textContent='';ws.send(JSON.stringify({type:'directory'}));};ws.onmessage=event=>{const msg=JSON.parse(event.data);if(msg.type==='directory')render(msg.service);};ws.onclose=()=>{$('serviceConnection').textContent='重新連線中';$('serviceConnection').dataset.state='reconnecting';clearTimeout(retry);retry=setTimeout(connect,1500);};ws.onerror=()=>{$('serviceMessage').textContent='無法連線至主機，請確認服務仍在執行。';};}
 void loadJoinAvailability();request('/api/rooms').then(render).catch(e=>$('serviceMessage').textContent=e.message);connect();
})();
