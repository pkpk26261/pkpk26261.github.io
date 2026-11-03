/* Controller navigation shared by the lobby, setup screens and in-game panels. */
(() => {
  'use strict';
  // Network player screens own navigation and relay buttons to the authority.
  if(['authority','viewer'].includes(new URLSearchParams(location.search).get('lan')))return;
  if (window.DroneGamepadUI) return;
  const selector='button,a[href],input:not([type="hidden"]),select,summary,[role="button"],[role="tab"]';
  const visible=e=>{
    if(!e?.isConnected||e.hidden||e.closest('[hidden],[inert],.opacity-0')||!e.getClientRects().length||getComputedStyle(e).visibility==='hidden'||getComputedStyle(e).display==='none')return false;
    for(let p=e.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS'&&!p.open&&!p.querySelector(':scope>summary')?.contains(e))return false;
    return true;
  };
  const activeOverlay=()=>['dialog[open]',':popover-open','.fc-pad-overlay','.fc-start-overlay','.fc-result-overlay'].map(s=>Array.from(document.querySelectorAll(s)).filter(visible).at(-1)).find(Boolean);
  const lobby=()=>!!window.FlightLobby;
  function scope() { return activeOverlay() || (lobby()?document.body:null); }
  const candidates=root=>Array.from(root?.querySelectorAll(selector)||[]).filter(e=>visible(e)&&!e.disabled&&e.getAttribute('aria-disabled')!=='true');
  let focused=null,axisAt=0,lastDirection='',lastScope=null,previousTime=0,unfocused=false;
  const padStates=new Map();
  function focus(element) {
    focused?.removeAttribute('data-gamepad-focus');focused=element;
    if (!element) return;
    element.dataset.gamepadFocus='';element.focus({preventScroll:true});
    element.scrollIntoView({block:'nearest',inline:'nearest'});
  }
  function ensure(root) {
    const all=candidates(root);
    if (root!==lastScope||!all.includes(focused)) {
      lastScope=root;
      const current=all.includes(document.activeElement)?document.activeElement:null;
      // Setup completion and starting play are the primary controller actions.
      focus(current || all.find(e=>e.matches('#psClose,.fc-primary-action,#home-play,#launch-button,[autofocus]')) || all[0]);
    }
    return all;
  }
  function changeField(element,direction) {
    if (element?.matches('select')) {
      const options=Array.from(element.options).filter(o=>!o.disabled&&!o.hidden),i=options.indexOf(element.selectedOptions[0]);
      const next=options[Math.max(0,Math.min(options.length-1,i+direction))];
      if(next&&next.value!==element.value){element.value=next.value;element.dispatchEvent(new Event('change',{bubbles:true}));}
      return true;
    }
    if(element?.matches('input[type="range"],input[type="number"]')){
      direction>0?element.stepUp():element.stepDown();
      element.dispatchEvent(new Event('input',{bubbles:true}));element.dispatchEvent(new Event('change',{bubbles:true}));return true;
    }
    return false;
  }
  function navigate(root,direction) {
    const all=ensure(root);if(!all.length)return;
    if ((direction==='left'||direction==='right')&&changeField(focused,direction==='right'?1:-1))return;
    const r=focused.getBoundingClientRect(),x=r.x+r.width/2,y=r.y+r.height/2;
    const vector={left:[-1,0],right:[1,0],up:[0,-1],down:[0,1]}[direction];
    let best=null,score=Infinity;
    for(const e of all){if(e===focused)continue;const b=e.getBoundingClientRect(),dx=b.x+b.width/2-x,dy=b.y+b.height/2-y;
      const along=dx*vector[0]+dy*vector[1],cross=Math.abs(dx*vector[1]-dy*vector[0]);
      if(along<=4)continue;const value=along+cross*2.5;if(value<score){best=e;score=value;}
    }
    if(best)focus(best);
  }
  function confirm(root) {
    ensure(root);if(!focused)return;
    if(focused.matches('select'))changeField(focused,1);
    else if(focused.matches('input[type="range"],input[type="number"]'))changeField(focused,1);
    else focused.click();
    // Legacy game listeners should not treat a menu button as live flight focus.
    if(!scope()){document.activeElement?.blur();focus(null);}
  }
  function closePanel(root) {
    if(!root)return false;
    if(root.matches(':popover-open')){root.hidePopover();return true;}
    const close=root.querySelector('.fc-guide-close,.fc-close,.fc-restart-cancel,[data-close],#psClose,.fc-control-dialog>header button,button[aria-label*="關閉"],button[aria-label*="收合"]');
    if(close){close.click();return true;}
    if(root.matches('dialog[open]')){
      const event=new Event('cancel',{cancelable:true});
      if(root.dispatchEvent(event))root.close();
      return true;
    }
    return false;
  }
  function command(action) {
    if(action==='back'&&closePanel(activeOverlay()))return;
    const event=new Event(`drone-gamepad-${action}`,{cancelable:true});
    if(!window.dispatchEvent(event))return;
    window.DroneFlightInput?.release?.();
    if(window.parent!==window)window.parent.postMessage({type:`flight-club:gamepad-${action}`},location.protocol==='file:'?'*':location.origin);
    else if(!lobby())location.href=new URL('../../index.html',location.href).href;
  }
  const value=(pad,i)=>{const b=pad?.buttons?.[i];return typeof b==='number'?b:b?.value??Number(!!b?.pressed);};
  function tick(now) {
    requestAnimationFrame(tick);
    const delta=Math.max(0,Math.min(.05,(now-(previousTime||now))/1000));previousTime=now;
    // The embedded game owns navigation while playing; avoid handling a press
    // once in the game and again in its parent after an iframe is removed.
    if(unfocused||document.hidden||(lobby()&&window.FlightLobby.snapshot().playing&&!activeOverlay())){padStates.clear();return;}
    const pads=window.DroneUSBInput?.browserPads?.()||Array.from(navigator.getGamepads?.()||[]);
    const candidatesPads=window.DroneUSBInput?.controllerPads?.(pads)||pads.filter(Boolean);
    const sources=new Map(),keyFor=p=>`${p.bridged?'bridge':'native'}:${p.index}:${p.id}`;
    for(const source of window.DroneControllerActions?.navigationSources?.()||[])sources.set(keyFor(source.pad),source);
    for(const pad of candidatesPads)if(!sources.has(keyFor(pad)))sources.set(keyFor(pad),{pad});
    for(const key of padStates.keys())if(!sources.has(key))padStates.delete(key);
    const samples=[];
    for(const [key,{pad,sticks}]of sources){
      if(pad.connected===false)continue;
      const state=padStates.get(key)||{ready:false,previous:{}};padStates.set(key,state);
      const buttons={x:value(pad,2)>.6,y:value(pad,3)>.6,back:value(pad,8)>.6,menu:value(pad,9)>.6};
      const mapped=sticks||window.DroneUSBInput?.readSticks(pad,{map:window.DroneUSBInput.presets[window.DroneUSBInput.resolvePreset(pad,'auto')]});
      const stickX=mapped?.[0].x??pad.axes?.[0]??0,stickY=-(mapped?.[0].y??-(pad.axes?.[1]??0));
      const dx=Number(value(pad,15)>.5)-Number(value(pad,14)>.5)||(Math.abs(stickX)>.6?Math.sign(stickX):0);
      const dy=Number(value(pad,13)>.5)-Number(value(pad,12)>.5)||(Math.abs(stickY)>.6?Math.sign(stickY):0);
      const edges=Object.fromEntries(Object.entries(buttons).map(([name,down])=>[name,state.ready&&down&&!state.previous[name]]));
      if(!state.ready&&!Object.values(buttons).some(Boolean)&&!dx&&!dy)state.ready=true;
      state.previous=buttons;samples.push({pad,key,ready:state.ready,edges,dx,dy});
    }
    // Every assigned player may open/cancel menus. Update all edge states so
    // a held button never fires later merely because a different pad was active.
    if(samples.some(s=>s.edges.menu)){command('menu');return;}
    if(samples.some(s=>s.edges.back||s.edges.y)){command('back');return;}
    const root=scope(),input=samples.find(s=>s.ready&&(s.dx||s.dy));
    if(root){ensure(root);
      const direction=input?(Math.abs(input.dx)>Math.abs(input.dy)?(input.dx>0?'right':'left'):(input.dy>0?'down':'up')):'';
      const repeatKey=direction?`${input.key}:${direction}`:'';
      if(direction&&(repeatKey!==lastDirection||now-axisAt>220)){navigate(root,direction);axisAt=now;}
      lastDirection=repeatKey;
      if(samples.some(s=>s.edges.x))confirm(root);
    }else{lastDirection='';focus(null);
      if(samples.some(s=>s.edges.x))document.querySelector('.fc-action-more')?.click();
      if(window.DroneControlSettings?.pilots.length===0){
        const spectator=samples.find(s=>s.ready),pad=spectator?.pad;
        const direction=Number(value(pad,7)>.5)-Number(value(pad,6)>.5);
        if(direction)window.DroneCameraControls?.shiftStanding(direction,delta);
      }
    }
  }
  const style=document.createElement('style');
  style.textContent='[data-gamepad-focus]{outline:3px solid #f4c75b!important;outline-offset:3px!important}';document.head.append(style);
  window.DroneGamepadUI=Object.freeze({back:()=>command('back'),menu:()=>command('menu'),get active(){return !!scope();}});
  window.addEventListener('blur',()=>{unfocused=true;padStates.clear();});
  window.addEventListener('focus',()=>{unfocused=false;padStates.clear();});
  requestAnimationFrame(tick);
})();
