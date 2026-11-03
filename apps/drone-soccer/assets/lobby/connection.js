(() => {
  'use strict';
  let snapshot = {status:'checking',origin:null}, pending, retry, stopped=false,knownOnline=false;
  function publish(next) {
    snapshot=Object.freeze(next);
    const connected=next.status==='connected', waiting=next.status==='pending', online=next.accessMode==='online', $=id=>document.getElementById(id);
    $('lobby-connection').textContent=connected?(online?'掃碼連線就緒':'已連上區網主機'):waiting?'公開連線準備中':'單機訓練';
    $('lobby-connection').classList.toggle('is-connected',connected);
    for(const id of ['service-link','footer-service-link','home-service-link']) $(id).href=connected?next.origin+'/':'lan/setup.html';
    $('service-link').querySelector('span').textContent=connected?'賽事大廳':'多人連線';
    $('home-host-link').href=connected?next.origin+'/admin':'lan/setup.html';
    $('home-network-status').lastChild.textContent=connected?(online?'掃碼連線已就緒':'已連上區網賽事'):waiting?'正在建立公開遊戲入口':'單機遊戲就緒';
    window.dispatchEvent(new CustomEvent('flight-connection-change',{detail:snapshot}));
    return snapshot;
  }
  function refresh(silent=false){
    if(pending)return pending;
    clearTimeout(retry);
    if(!/^https?:$/.test(location.protocol))return Promise.resolve(publish({status:'offline',origin:null}));
    if(!silent||snapshot.status!=='connected')publish({status:'checking',origin:null});
    const controller=new AbortController(), timeout=setTimeout(()=>controller.abort(),2500);
    pending=(async()=>{
      try{
        const response=await fetch('/api/config',{signal:controller.signal,credentials:'same-origin',cache:'no-store'});
        if(response.ok && response.headers.get('content-type')?.includes('application/json')){
          const config=await response.json();
          if(config.secure&&config.accessMode==='online'){
            knownOnline=true;
            if(config.publicReady!==true||config.directJoin!==true)return publish({status:'pending',origin:null,invitationURL:null,accessMode:'online',publicReady:false,directJoin:true});
            const publicOrigin=new URL(config.origin),invitation=new URL(config.invitationURL||config.joinURL);
            if(publicOrigin.protocol!=='https:'||publicOrigin.username||publicOrigin.password||publicOrigin.pathname!=='/'||publicOrigin.search||publicOrigin.hash||invitation.origin!==publicOrigin.origin||invitation.pathname!=='/play'||invitation.username||invitation.password||invitation.search||invitation.hash)throw Error('Invalid public join URL');
            return publish({status:'connected',origin:publicOrigin.origin,invitationURL:invitation.href,accessMode:'online',publicReady:true,directJoin:true});
          }
          if(config.secure && new URL(config.origin).origin===location.origin){
            let invitationURL=location.origin+'/';
            try{
              const setup=new URL(config.setupURL);
              if(setup.protocol==='http:'&&setup.hostname===location.hostname&&!setup.username&&!setup.password&&setup.pathname==='/'&&!setup.search&&!setup.hash)invitationURL=setup.href;
            }catch(_){ /* HTTPS-only hosts have no onboarding endpoint. */ }
            return publish({status:'connected',origin:location.origin,invitationURL,accessMode:'legacy',directJoin:false});
          }
        }
      }catch(_){ /* Static/offline play stays available. */ }
      return publish(knownOnline?{status:'pending',origin:null,invitationURL:null,accessMode:'online',publicReady:false,directJoin:true}:{status:'offline',origin:null});
    })().finally(()=>{clearTimeout(timeout);pending=null;if(!stopped&&knownOnline)retry=setTimeout(()=>refresh(true),snapshot.status==='connected'?5000:2500);});
    return pending;
  }
  window.FlightConnection=Object.freeze({refresh,get snapshot(){return snapshot;}});
  window.addEventListener('pagehide',()=>{stopped=true;clearTimeout(retry);});
  window.addEventListener('pageshow',event=>{if(event.persisted){stopped=false;void refresh();}});
  void refresh();
})();
