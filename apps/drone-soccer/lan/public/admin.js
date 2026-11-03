(() => {
 'use strict';
 const {$,el,request,resultRows,phaseName,icon,replaceLiveContent}=window.FlightService;
 const entryParams=new URLSearchParams(location.search),arenaDrafts=new Map();
 let service,ws,retry,loggedIn=false,liveCode='',livePinned=false,roomSignature='',siteDirty=false,sitePending='',creating=false;
 let openOnEntry=entryParams.get('create')==='1';
 const tell=(message,state='info')=>{$('serviceMessage').textContent=message;$('serviceMessage').dataset.state=state;};
 function send(data){if(ws?.readyState!==1){tell('管理連線尚未就緒，請稍候再試。','error');return false;}ws.send(JSON.stringify(data));return true;}
 function connection(text,state){$('serviceConnection').textContent=text;$('serviceConnection').dataset.state=state;}
 function showLogin(){loggedIn=false;creating=false;$('adminLogin').hidden=false;$('adminDashboard').hidden=true;$('adminLogout').hidden=true;connection('尚未登入','offline');$('liveScene').removeAttribute('src');$('liveScene').hidden=true;liveCode='';}
 function renderCounts(data){
  const free=data.arenas.filter(a=>a.enabled&&!a.roomCode).length;
  const rows=[['等待開賽',data.rooms.filter(r=>r.status==='lobby').length],['進行中',data.counts.running],['在線玩家',data.counts.players],['可用賽場',free]];
  $('serviceCounts').replaceChildren(...rows.map(([label,n])=>{const item=el('div',undefined,'admin-summary-stat');item.append(el('strong',n),el('span',label));return item;}));
  $('arenaCapacity').textContent=free+' 個賽場可用';$('arenaCapacityDetail').textContent=`${data.arenas.length} 個賽場，${data.arenas.filter(a=>a.roomCode).length} 個使用中。使用中的賽場無法停用或移除。`;
 }
 function renderLive(data){
  const current=data.rooms.find(r=>r.code===liveCode),room=(livePinned&&current)||data.rooms.find(r=>r.status==='running')||current||data.rooms.find(r=>r.status==='paused')||data.rooms[0];
  $('liveMonitor').hidden=!room;
  if(!room){$('liveScene').removeAttribute('src');$('liveScene').hidden=true;liveCode='';return;}
  const list=data.rooms.map(r=>r.code+':'+r.name+':'+r.status+':'+r.match?.phase).join('|');
  if($('liveRoomSelect').dataset.list!==list){$('liveRoomSelect').dataset.list=list;$('liveRoomSelect').replaceChildren(...data.rooms.map(r=>new Option(r.name+' · '+phaseName(r),r.code)));}
  $('liveRoomSelect').value=room.code;
  if(liveCode!==room.code){liveCode=room.code;$('liveScene').src='/watch?code='+room.code+'&embedded=1';$('liveScene').hidden=false;}
  $('liveRoomName').textContent=room.name;$('liveMatchPhase').textContent=phaseName(room);$('liveMatchPhase').className='badge '+room.status;
  $('liveHostLink').href='/host?code='+room.code+'&panel=room&view=broadcast';
  $('liveIndicator').textContent=({running:'進行中',paused:'已暫停',ended:'已結束',lobby:'等待開賽'})[room.status]||'即時預覽';
  $('liveRoomDetail').textContent=room.match?.phase==='break'?'局間休息中，下一局即將繼續。':room.status==='running'?'此為同場即時預覽。管理其他房間時，比賽會繼續。':room.status==='lobby'?room.engineReady?'等待玩家完成準備，請從「管理比賽」開賽。':'正在準備賽場，完成後即可開賽。':room.reason||'選擇房間，查看目前賽況。';
 }
 function action(text,url,key,room,primary=false){const a=el('a',text,'action-link'+(primary?' primary':''));a.href=url;a.dataset.liveKey='room:'+room.code+':'+key;a.setAttribute('aria-label',text+'：'+room.name+'（'+room.code+'）');return a;}
 function adminRoomCard(room,menuOpen){
  const card=el('article',undefined,'room-card admin-room-row');card.dataset.room=room.code;card.dataset.preview=String(room.code===liveCode);
  const main=el('div',undefined,'admin-room-main'),heading=el('div',undefined,'admin-room-heading');heading.append(el('h3',room.name),el('span',phaseName(room)||room.status,'badge '+room.status));main.append(heading);
  const meta=el('div',undefined,'admin-room-meta');meta.append(el('span',room.code,'room-code'),el('span',room.arenaName,'arena-name'));main.append(meta);
  const r=room.roster;main.append(el('p',`紅隊 ${r.red.humans} 真人＋${r.red.ai} AI　·　藍隊 ${r.blue.humans} 真人＋${r.blue.ai} AI`,'admin-room-roster'));
  const state=el('div',undefined,'admin-room-state');
  if(['running','paused','ended'].includes(room.status)){const score=el('div',undefined,'room-score');score.setAttribute('aria-label',`紅隊 ${room.scores.red}，藍隊 ${room.scores.blue}`);score.append(el('strong',room.scores.red,'red'),el('span',':'),el('strong',room.scores.blue,'blue'));state.append(score,el('small',room.status==='ended'?'最終比分':`${room.ready} / ${room.humans} 已準備`));}
  else state.append(el('strong',room.humans?`${room.joined} / ${room.humans}`:'全 AI'),el('small',room.engineReady?(room.humans?`${room.ready} 位已準備`:'可直接開賽'):'賽場準備中'));
  const actions=el('div',undefined,'admin-room-actions');actions.append(action(room.status==='ended'?'查看結果':'管理比賽','/host?code='+room.code+'&panel=room','manage',room,true),action('邀請玩家','/host?code='+room.code+'&panel=invite','invite',room));
  const menu=el('details',undefined,'room-actions-menu'),summary=el('summary','更多');menu.open=menuOpen;summary.dataset.liveKey='room:'+room.code+':menu';summary.setAttribute('aria-label','更多房間操作：'+room.name);const items=el('div',undefined,'room-menu-items');
  const preview=el('button','在此預覽');preview.type='button';preview.dataset.liveKey='room:'+room.code+':preview';preview.onclick=()=>{livePinned=true;liveCode=room.code;$('liveScene').src='/watch?code='+room.code+'&embedded=1';$('liveScene').hidden=false;menu.open=false;renderLive(service);roomSignature='';renderRooms();$('liveRoomSelect').focus({preventScroll:true});};
  items.append(preview,action('開啟賽事轉播','/host?code='+room.code+'&panel=room&view=broadcast','watch',room));
  const close=el('button','關閉房間','danger');close.type='button';close.dataset.liveKey='room:'+room.code+':close';close.setAttribute('aria-label','關閉房間：'+room.name);close.onclick=async()=>{if(await FlightPanels.confirm({title:'關閉這個房間？',description:'「'+room.name+'」的玩家將離房，原房碼與 QR 失效。比賽結果會保留，其他房間不受影響。',acceptLabel:'關閉房間',danger:true}))send({type:'admin.closeRoom',code:room.code});};items.append(close);menu.append(summary,items);actions.append(menu);card.append(main,state,actions);return card;
 }
 function renderRooms(){
  if(!service)return;const search=$('adminRoomSearch').value.trim().toLowerCase(),filter=$('adminRoomFilter').value;
  const rank={running:0,paused:1,lobby:2,ended:3};
  const rows=service.rooms.filter(r=>(filter==='all'||r.status===filter)&&[r.name,r.code,r.arenaName].join(' ').toLowerCase().includes(search)).sort((a,b)=>(rank[a.status]??4)-(rank[b.status]??4)||(b.createdAt||0)-(a.createdAt||0));
  const signature=JSON.stringify({liveCode,rows:rows.map(({match,...r})=>({...r,phase:match?.phase,reason:match?.winReason||r.reason}))});
  if(signature!==roomSignature){roomSignature=signature;const openMenus=new Set([...$('roomList').querySelectorAll('details[open]')].map(d=>d.closest('[data-room]').dataset.room));replaceLiveContent($('roomList'),rows.map(r=>adminRoomCard(r,openMenus.has(r.code))),$('adminRoomSearch'));}
  const summary=rows.length+' / '+service.rooms.length+' 間';if($('roomTotal').textContent!==summary)$('roomTotal').textContent=summary;
  $('roomEmpty').hidden=!!rows.length;const filtered=!!service.rooms.length;
  $('roomEmptyTitle').textContent=filtered?'沒有符合條件的比賽':'建立第一場比賽';$('roomEmptyDetail').textContent=filtered?'換個房名、房碼或狀態試試看。':'選擇賽場與隊伍，即可邀請玩家上場。';$('firstRoom').hidden=filtered;$('clearRoomSearch').hidden=!filtered;
 }
 function arenaEditor(arena){
  const form=el('form',undefined,'arena-card arena-editor');form.dataset.arena=arena.id;const title=el('div',undefined,'arena-editor-heading'),nameTitle=el('h3'),state=el('span',undefined,'badge');title.append(nameTitle,state);form.append(title);
  const name=el('input'),note=el('input'),enabled=el('input');name.required=true;name.maxLength=36;note.maxLength=80;note.placeholder='例如：體育館東側';enabled.type='checkbox';
  for(const [key,text,input] of [['name','賽場名稱',name],['note','場地說明',note],['enabled','開放建立比賽',enabled]]){input.dataset.arenaField=key;input.dataset.liveKey='arena:'+arena.id+':'+key;const label=el('label',text);label.append(input);if(key==='enabled')label.className='arena-enabled';form.append(label);}
  const status=el('p',undefined,'arena-save-status');status.setAttribute('role','status');
  const buttons=el('div',undefined,'card-actions'),save=el('button','儲存變更'),remove=el('button','移除賽場','danger');save.type='submit';remove.type='button';remove.dataset.arenaRemove='';buttons.append(save,remove);form.append(status,buttons);
  form.oninput=()=>{arenaDrafts.set(arena.id,{name:name.value,note:note.value,enabled:enabled.checked,pending:false});form.dataset.dirty='true';status.textContent='有變更尚未儲存';};
  form.onsubmit=e=>{e.preventDefault();const draft={name:name.value,note:note.value,enabled:enabled.checked,pending:true};if(send({type:'arena.update',id:arena.id,name:draft.name,note:draft.note,enabled:draft.enabled})){arenaDrafts.set(arena.id,draft);status.textContent='已送出，等待主機確認';}};
  remove.onclick=async()=>{const current=service.arenas.find(a=>a.id===arena.id);if(current?.roomCode)return;if(await FlightPanels.confirm({title:'移除這個賽場？',description:'「'+(current?.name||arena.name)+'」會從可分配賽場中移除，已保存的比賽紀錄仍會保留。',acceptLabel:'移除賽場',danger:true}))send({type:'arena.remove',id:arena.id});};return form;
 }
 function renderArenas(data){
  const valid=new Set(data.arenas.map(a=>a.id));for(const form of [...$('arenaList').children])if(!valid.has(form.dataset.arena)){if(form.contains(document.activeElement))$('addArena').focus();form.remove();arenaDrafts.delete(form.dataset.arena);}
  for(const arena of data.arenas){let form=[...$('arenaList').children].find(f=>f.dataset.arena===arena.id);if(!form){form=arenaEditor(arena);$('arenaList').append(form);}const name=form.querySelector('[data-arena-field=name]'),note=form.querySelector('[data-arena-field=note]'),enabled=form.querySelector('[data-arena-field=enabled]');let draft=arenaDrafts.get(arena.id);
   if(draft?.pending&&draft.name.trim()===arena.name&&draft.note.trim()===arena.note&&draft.enabled===arena.enabled){arenaDrafts.delete(arena.id);draft=null;form.dataset.dirty='false';form.querySelector('.arena-save-status').textContent='已儲存';}
   if(!draft){if(document.activeElement!==name)name.value=arena.name;if(document.activeElement!==note)note.value=arena.note;enabled.checked=arena.enabled;}
   enabled.disabled=!!arena.roomCode;form.querySelector('[data-arena-remove]').disabled=!!arena.roomCode;form.querySelector('h3').textContent=arena.name;const badge=form.querySelector('.badge');badge.textContent=arena.roomCode?'使用中':arena.enabled?'空閒':'已停用';badge.className='badge '+(arena.roomCode?'running':arena.enabled?'lobby':'ended');
  }
 }
 function fillArenas(){if(!service)return;const previous=$('newArena').value,free=service.arenas.filter(a=>a.enabled&&!a.roomCode),signature=JSON.stringify(free.map(a=>[a.id,a.name]));if($('newArena').dataset.options!==signature){$('newArena').dataset.options=signature;$('newArena').replaceChildren(new Option(free.length?'自動分配空閒賽場':'目前沒有空閒賽場',''));for(const a of free)$('newArena').append(new Option(a.name,a.id));if(free.some(a=>a.id===previous))$('newArena').value=previous;}
  $('submitRoom').disabled=creating||!free.length;if($('roomDialog').open&&!free.length)$('newRoomError').textContent='目前沒有空閒賽場，請關閉已結束的房間或新增賽場。';
 }
 function render(data){
  service=data;$('siteName').textContent=data.siteName;$('siteName').title=data.siteName;if(sitePending&&data.siteName===sitePending){siteDirty=false;sitePending='';tell('已儲存大廳名稱','saved');}if(!siteDirty&&document.activeElement!==$('siteTitle'))$('siteTitle').value=data.siteName;
  renderCounts(data);renderLive(data);renderRooms();renderArenas(data);
  const seen=new Set(),rows=[...data.rooms.filter(r=>r.status==='ended'),...data.history].sort((a,b)=>(b.endedAt||b.closedAt||0)-(a.endedAt||a.closedAt||0)).filter(r=>!seen.has(r.code)&&seen.add(r.code));replaceLiveContent($('resultList'),resultRows(rows),document.querySelector('[data-admin-panel=history]'));$('historyEmpty').hidden=!!rows.length;$('historyTotal').textContent=rows.length+' 場比賽';fillArenas();$('addArena').disabled=data.arenas.length>=data.maxArenas;
 }
 function connect(){clearTimeout(retry);if(!loggedIn)return;const previous=ws,transport=new WebSocket('wss://'+location.host+'/socket');ws=transport;previous?.close();connection('正在連線','connecting');transport.onopen=()=>send({type:'admin'});transport.onmessage=event=>{if(ws!==transport)return;const msg=JSON.parse(event.data);if(msg.type==='admin'){connection('管理已連線','connected');render(msg.service);if(openOnEntry){openOnEntry=false;open();history.replaceState(null,'','/admin');}}if(msg.type==='directory')render(msg.service);if(msg.type==='created'){location.href='/host?code='+msg.room.code+'&panel='+(msg.room.seats.some(s=>!s.ai)?'invite':'room');}if(msg.type==='error'){creating=false;fillArenas();tell(msg.message,'error');if(msg.request==='arena.update')for(const [id,draft] of arenaDrafts){if(!draft.pending)continue;draft.pending=false;const form=[...$('arenaList').children].find(f=>f.dataset.arena===id);if(form)form.querySelector('.arena-save-status').textContent='尚未儲存，請依上方提示調整後重試';}$('newRoomError').textContent=msg.request==='create'?msg.message:'';if(/管理權限|登入/.test(msg.message))showLogin();}if(msg.type==='signedOut'){showLogin();tell('管理工作階段已登出，進行中的比賽仍繼續運作。');}};transport.onclose=()=>{if(ws===transport&&loggedIn){creating=false;fillArenas();connection('正在重新連線','connecting');retry=setTimeout(connect,1500);}};}
 async function enter(key){try{if(key)await request('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({key})});const result=await request('/api/session');if(!result.authenticated){showLogin();return;}loggedIn=true;if(entryParams.get('returnTo')==='game'){location.replace('/apps/drone-soccer/?screen=home');return;}$('adminLogin').hidden=true;$('adminDashboard').hidden=false;$('adminLogout').hidden=false;$('adminKey').value='';tell('');connect();}catch(e){showLogin();tell(e.message,'error');}}
 function consumeKey(){const key=new URLSearchParams(location.hash.slice(1)).get('admin');if(key){history.replaceState(null,'',location.pathname+location.search);enter(key);}else enter();}
 $('loginForm').onsubmit=e=>{e.preventDefault();enter($('adminKey').value);};window.addEventListener('hashchange',consumeKey);
 $('liveRoomSelect').onchange=()=>{livePinned=true;liveCode=$('liveRoomSelect').value;$('liveScene').src='/watch?code='+liveCode+'&embedded=1';$('liveScene').hidden=false;renderLive(service);roomSignature='';renderRooms();};
 $('adminRoomSearch').oninput=renderRooms;$('adminRoomFilter').onchange=renderRooms;$('clearRoomSearch').onclick=()=>{$('adminRoomSearch').value='';$('adminRoomFilter').value='all';renderRooms();$('adminRoomSearch').focus();};
 $('adminLogout').onclick=async()=>{try{await request('/api/logout',{method:'POST'});loggedIn=false;clearTimeout(retry);ws?.close();for(const key of Object.keys(sessionStorage))if(key==='lan.admin'||key.startsWith('lan.host.'))sessionStorage.removeItem(key);showLogin();tell('已登出，進行中的比賽仍繼續運作。');}catch(e){tell(e.message,'error');}};
 const tabs=[...document.querySelectorAll('[data-admin-panel]')].filter(e=>e.matches('button'));
 function selectPanel(next,focus=false){document.body.dataset.adminPanel=next;for(const tab of tabs){const active=tab.dataset.adminPanel===next;tab.setAttribute('aria-selected',String(active));tab.setAttribute('aria-pressed',String(active));tab.tabIndex=active?0:-1;if(active&&focus)tab.focus();}for(const name of ['rooms','arenas','history'])$('admin'+name[0].toUpperCase()+name.slice(1)).hidden=name!==next;$('adminPageTitle').textContent=({rooms:'賽事管理',arenas:'賽場設定',history:'歷史賽果'})[next];}
 tabs.forEach((tab,i)=>{tab.prepend(icon(({rooms:'users',arenas:'grid',history:'shield'})[tab.dataset.adminPanel]));tab.onclick=()=>selectPanel(tab.dataset.adminPanel);tab.onkeydown=e=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;e.preventDefault();const n=e.key==='Home'?0:e.key==='End'?tabs.length-1:(i+(e.key==='ArrowRight'?1:tabs.length-1))%tabs.length;selectPanel(tabs[n].dataset.adminPanel,true);};});
 document.addEventListener('keydown',e=>{if(e.key==='Escape'){const menu=document.activeElement?.closest('.room-actions-menu[open]');if(menu){e.preventDefault();menu.open=false;menu.querySelector('summary').focus();}}});
 document.addEventListener('click',e=>{if(e.target.closest('#panelDecision'))return;for(const menu of document.querySelectorAll('.room-actions-menu[open]'))if(!menu.contains(e.target)){menu.open=false;if(menu.contains(document.activeElement))menu.querySelector('summary').focus();}});
 const open=()=>{creating=false;fillArenas();$('newRoomError').textContent='';for(const team of ['Red','Blue'])for(const kind of ['Humans','AI']){const key=team.toLowerCase()+kind;if(entryParams.has(key))$('new'+team+kind).value=entryParams.get(key);}if(['easy','medium','hard'].includes(entryParams.get('difficulty')))$('newDifficulty').value=entryParams.get('difficulty');$('newRoomName').value='正式賽 '+new Date().toLocaleTimeString('zh-TW',{hour:'2-digit',minute:'2-digit'});$('newRoomForm').dispatchEvent(new Event('input'));$('roomDialog').showModal();fillArenas();};$('newRoom').onclick=open;$('firstRoom').onclick=open;$('roomDialogClose').onclick=()=>$('roomDialog').close();
 $('newRoomForm').onsubmit=e=>{e.preventDefault();if(creating)return;const roster={red:{humans:Number($('newRedHumans').value),ai:Number($('newRedAI').value)},blue:{humans:Number($('newBlueHumans').value),ai:Number($('newBlueAI').value)}};const invalidTeam=Object.keys(roster).find(team=>{const r=roster[team];return !Number.isInteger(r.humans)||!Number.isInteger(r.ai)||r.humans<0||r.ai<0||r.humans+r.ai<1||r.humans+r.ai>3;});if(invalidTeam){$('newRoomError').textContent='每隊合計必須為 1–3 台。';$('new'+(invalidTeam==='red'?'Red':'Blue')+'Humans').focus();return;}if(!service.arenas.some(a=>a.enabled&&!a.roomCode)){$('newRoomError').textContent='目前沒有空閒賽場。';return;}if(send({type:'create',name:$('newRoomName').value,arenaId:$('newArena').value||undefined,roster,difficulty:$('newDifficulty').value,breakSec:Number($('newBreak').value)})){creating=true;fillArenas();}};
 $('addArena').onclick=async()=>{const name=await FlightPanels.prompt({title:'新增賽場',description:'設定一個容易辨認的名稱，建立比賽時就能分配。',label:'賽場名稱',value:'新賽場',acceptLabel:'建立賽場',maxLength:36});if(name)send({type:'arena.add',name});};
 $('siteTitle').oninput=()=>{siteDirty=true;sitePending='';};$('siteForm').onsubmit=e=>{e.preventDefault();const name=$('siteTitle').value.trim();if(send({type:'service.settings',name}))sitePending=name;};consumeKey();
})();
