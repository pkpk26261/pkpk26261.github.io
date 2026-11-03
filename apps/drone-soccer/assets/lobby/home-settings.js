(() => {
  'use strict';
  const $=id=>document.getElementById(id), dialog=$('home-settings-dialog');
  const tabs=[...dialog.querySelectorAll('[data-settings-tab]')];
  const mobileNavigation=matchMedia('(max-width:700px)');
  function updateNavigation(){dialog.querySelector('.preferences-nav').setAttribute('aria-orientation',mobileNavigation.matches?'horizontal':'vertical');}
  mobileNavigation.addEventListener('change',updateNavigation);updateNavigation();
  let category='general', returnToSettings=null, opener=null;
  function feedback(text){$('settings-feedback').textContent=text;}
  function storageAvailable(){try{const key='flight-club-storage-check';localStorage.setItem(key,'1');localStorage.removeItem(key);return true;}catch(_){return false;}}
  function saved(key,value,also=[]){
    let durable=false;try{durable=[[key,value],...also].every(([name,expected])=>localStorage.getItem(name)===String(expected));}catch(_){}
    feedback(durable?'已套用，並保存至此瀏覽器':'已套用至本次遊戲；此瀏覽器無法保存設定');
  }
  function selectTab(next,focus=false){
    if(!tabs.some(tab=>tab.dataset.settingsTab===next))return;
    category=next;
    tabs.forEach(tab=>{const active=tab.dataset.settingsTab===next;tab.setAttribute('aria-selected',String(active));tab.tabIndex=active?0:-1;if(active&&focus)tab.focus();});
    dialog.querySelectorAll('[data-settings-panel]').forEach(panel=>{panel.hidden=panel.dataset.settingsPanel!==next;});
    $('settings-reset').hidden=!['general','display'].includes(next);
    dialog.querySelector('.preferences-content').scrollTop=0;
    feedback('變更即時套用並自動保存');refresh();
  }
  function updateConnection(){
    const connection=window.FlightConnection?.snapshot || {status:'offline'};
    const live=connection.status==='connected', checking=connection.status==='checking', waiting=connection.status==='pending',online=connection.accessMode==='online';
    $('settings-connection-state').textContent=live?(online?'掃碼連線已就緒':'區網連線已就緒'):checking?'正在檢查連線…':waiting?'正在建立公開遊戲入口':'單機遊戲已就緒';
    $('settings-connection-detail').textContent=live?(online?'玩家掃 QR 即可加入，免安裝或信任憑證。主機與玩家皆需網際網路。':connection.invitationURL?.startsWith('http:')?'目前為區網模式，首次加入需完成裝置信任。':'目前為區網模式，僅供已完成裝置信任的裝置加入。'):waiting?'連線就緒後會自動顯示玩家網址；請保持應用程式與網際網路連線。':'從 Windows 應用程式啟動掃碼連線，即可建立房間並邀請玩家。';
    $('settings-network-refresh').disabled=checking;
    $('settings-public-url-row').hidden=!live;
    $('settings-public-url').value=live?(connection.invitationURL||connection.origin+'/'):'';
    $('settings-public-link').href=live?connection.origin+'/':'lan/setup.html';
    $('home-host-link').href=live?connection.origin+'/admin':'lan/setup.html';
    dialog.querySelector('.connection-context').dataset.connected=String(live);
  }
  function refresh(){
    const state=window.FlightLobby.snapshot();
    $('settings-mode-context').textContent=`${state.title} · ${state.players} 位本機玩家`;
    $('settings-controller').value=state.controller;
    $('settings-controller').disabled=true;
    $('settings-controller-note').textContent='所有模式統一使用搖桿。';
    $('settings-open-controls').disabled=state.players===0;
    $('settings-open-pad').disabled=state.players===0;
    $('settings-controls-note').textContent=state.players===0?'目前為全 AI 觀戰。請先在大廳選擇有真人玩家的模式。':'開啟目前模式的設定畫面，BACK 可返回這裡。';
    const fps=String(window.DroneFrameTiming.preference), options=$('settings-fps').options;
    if(![...options].some(option=>option.value===fps))$('settings-fps').add(new Option(`${fps} FPS · 自訂`,fps));
    $('settings-fps').value=fps;
    $('settings-follow-pitch').checked=window.DroneCameraSettings.followPitch;
    $('settings-storage-state').textContent=storageAvailable()?'此瀏覽器的使用者資料':'僅於本次開啟期間保存';
    updateConnection();
  }
  function open(tab=category){opener=document.activeElement;selectTab(tab);if(!dialog.open)dialog.showModal();}
  function close(){dialog.close();}
  window.FlightSettings=Object.freeze({open,close});
  tabs.forEach((tab,index)=>{
    tab.addEventListener('click',()=>selectTab(tab.dataset.settingsTab));
    tab.addEventListener('keydown',event=>{
      let next=index;
      if(['ArrowDown','ArrowRight'].includes(event.key))next=(index+1)%tabs.length;
      else if(['ArrowUp','ArrowLeft'].includes(event.key))next=(index+tabs.length-1)%tabs.length;
      else if(event.key==='Home')next=0;else if(event.key==='End')next=tabs.length-1;else return;
      event.preventDefault();selectTab(tabs[next].dataset.settingsTab,true);
    });
  });
  for(const id of ['setup-settings','game-settings-button'])$(id).addEventListener('click',()=>open());
  for(const id of ['home-settings-close','settings-done'])$(id).addEventListener('click',close);
  dialog.addEventListener('close',()=>opener?.isConnected&&opener.focus({preventScroll:true}));
  dialog.addEventListener('click',event=>{if(event.target!==dialog)return;const b=dialog.getBoundingClientRect();if(event.clientX<b.left||event.clientX>b.right||event.clientY<b.top||event.clientY>b.bottom)close();});
  $('settings-fps').addEventListener('change',event=>{window.DroneFrameTiming.setTarget(event.target.value);saved('droneSoccer.render.targetFPS',event.target.value);});
  $('settings-follow-pitch').addEventListener('change',event=>{window.DroneCameraSettings.setFollowPitch(event.target.checked);saved('droneSoccer.camera.followPitch',String(event.target.checked));});
  window.addEventListener('flight-menu-preferences',event=>{if(dialog.open)feedback(event.detail.saved?'已套用，並保存至此瀏覽器':'已套用至本次遊戲；此瀏覽器無法保存設定');});
  $('settings-reset').addEventListener('click',()=>{
    if(category==='general'){const durable=window.FlightMenuPreferences.reset();feedback(durable?'已還原聲音與介面預設':'已還原；此瀏覽器無法保存設定');}
    if(category==='display'){window.DroneFrameTiming.setTarget('120');window.DroneCameraSettings.setFollowPitch(false);saved('droneSoccer.render.targetFPS','120',[['droneSoccer.camera.followPitch','false']]);refresh();}
  });
  async function fullscreen(){try{if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen();feedback('已更新顯示方式');}catch(_){feedback('此瀏覽器無法開啟全螢幕，可使用視窗最大化。');}}
  $('home-fullscreen').addEventListener('click',fullscreen);
  document.addEventListener('fullscreenchange',()=>{$('home-fullscreen').querySelector('span').textContent=document.fullscreenElement?'離開全螢幕':'全螢幕顯示';});
  $('settings-network-refresh').addEventListener('click',async()=>{await window.FlightConnection.refresh();feedback(window.FlightConnection.snapshot.status==='connected'?'賽事連線檢查完成':window.FlightConnection.snapshot.status==='pending'?'公開連線準備中，就緒後會自動顯示網址':'目前使用單機入口');});
  window.addEventListener('flight-connection-change',updateConnection);
  $('settings-copy-url').addEventListener('click',async()=>{try{await navigator.clipboard.writeText($('settings-public-url').value);feedback('已複製玩家公開入口');}catch(_){$('settings-public-url').focus();$('settings-public-url').select();feedback('請複製已選取的公開網址');}});
  function openModeSettings(kind){
    const state=window.FlightLobby.snapshot();if(state.players===0)return;
    if(!state.playing){returnToSettings=document.documentElement.dataset.screen;}
    close();
    if(returnToSettings)$('back-button').querySelector('span').textContent='返回遊戲設定';
    if(!window.FlightLobby.openSettings(kind)){returnToSettings=null;open('controls');}
  }
  $('settings-open-controls').addEventListener('click',()=>openModeSettings('controls'));
  $('settings-open-pad').addEventListener('click',()=>openModeSettings('pad'));
  window.addEventListener('flight-lobby-menu',()=>{returnToSettings=null;$('back-button').querySelector('span').textContent='回到大廳';});
  window.addEventListener('flight-lobby-view',event=>{
    if(event.detail.screen==='game'||!returnToSettings)return;
    const previous=returnToSettings;returnToSettings=null;
    $('back-button').querySelector('span').textContent='回到大廳';
    queueMicrotask(()=>{if(previous==='home')window.FlightLobby.home();open('controls');});
  });
  for(const name of ['drone-frame-settings-change','drone-camera-settings-change','flight-lobby-change'])window.addEventListener(name,()=>{if(dialog.open)refresh();});
  refresh();
})();
