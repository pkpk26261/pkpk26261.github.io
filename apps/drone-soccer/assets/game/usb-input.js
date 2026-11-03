/* Shared RC/Gamepad signal handling. No device-specific axis order is assumed. */
(function () {
  'use strict';
  const presets = {
    rc: [{x:4,y:2,xi:false,yi:true},{x:0,y:1,xi:false,yi:true}],
    standard: [{x:0,y:1,xi:false,yi:false},{x:2,y:3,xi:false,yi:false}],
    usbJoystick: [{x:0,y:1,xi:false,yi:false},{x:2,y:5,xi:false,yi:false}]
  };
  const flightModes=Object.freeze({
    hold:'左桿控制升降，回中後維持高度；右桿控制前後及平移，回桿後逐步煞停。',
    manual:'油門直接調整馬達推力，中間約為懸停推力，仍需自行微調高度。KESTREL 操作請選定高模式。'
  });
  function resolvePreset(pad, choice) {
    if (Object.prototype.hasOwnProperty.call(presets,choice)) return choice;
    if (pad?.mapping === 'standard' || pad?.axes?.length === 4) return 'standard';
    // Chrome's raw HID axes retain usage offsets: X/Y/Z/Rz => 0/1/2/5.
    // Pygame bridge frames use compact axes instead, normally four channels.
    if (!pad?.bridged && pad?.axes?.length >= 6 && /0079.*0006/i.test(String(pad.id))) return 'usbJoystick';
    return 'rc';
  }
  const clone = x => JSON.parse(JSON.stringify(x));
  const finite = x => typeof x === 'number' && Number.isFinite(x);
  // Some gaming mice expose a joystick HID collection with many idle axes.
  // Exclude those collections from both explicit selection and auto-assignment.
  function isPeripheral(pad) {
    return /pugio|\bmouse\b|\bkeyboard\b|滑鼠|鍵盤/i.test(String(pad?.id || '')) ||
      /0b05.*(?:1906|1908)/i.test(String(pad?.id || ''));
  }
  function controllerPads(pads) {
    return pads.filter(p => p && p.connected !== false && !isPeripheral(p));
  }
  let automaticPadIdentity = '';
  const padIdentity = pad => `${pad.bridged?'bridge':'browser'}:${pad.index}:${pad.id}`;
  function selectPad(pads, selected = null) {
    // An explicit selection stays disconnected when unplugged; never hand its
    // controls to another device. Automatic choices retain the assigned seat.
    const candidates = controllerPads(pads);
    if (selected !== null) { automaticPadIdentity=''; return candidates.find(p => p.index === selected) || null; }
    if (automaticPadIdentity) return candidates.find(p=>padIdentity(p)===automaticPadIdentity)||null;
    const pad=candidates.find(p => /0079.*0006/i.test(String(p.id))) || candidates[0] || null;
    if(pad)automaticPadIdentity=padIdentity(pad);
    return pad;
  }
  const assignments = new WeakMap();
  function resolvePads(pads, slots) {
    const candidates = controllerPads(pads), taken = new Set();
    const identity = pad => `${pad.bridged ? 'bridge' : 'browser'}:${pad.index}:${pad.id}`;
    const result = slots.map(slot => {
      if (slot.padIndex === null) return null;
      // Explicit choices keep their slot even while unplugged.
      assignments.delete(slot);
      const pad = candidates.find(p => p.index === slot.padIndex && !taken.has(p.index)) || null;
      if (pad) taken.add(pad.index);
      return pad;
    });
    // Retain every connected automatic assignment before filling empty slots.
    // A missing P1 must not pull P2's controller into P1's aircraft.
    slots.forEach((slot, i) => {
      if (slot.padIndex !== null) return;
      const previous = assignments.get(slot);
      const pad = candidates.find(p => identity(p) === previous && !taken.has(p.index));
      if (pad) { result[i] = pad; taken.add(pad.index); }
    });
    slots.forEach((slot, i) => {
      if (slot.padIndex !== null || result[i]) return;
      // A previously assigned controller keeps its empty seat while unplugged.
      // Explicitly choose a device in setup to replace that assignment.
      if (assignments.has(slot)) return;
      const pad = candidates.find(p => !taken.has(p.index));
      if (pad) { result[i] = pad; taken.add(pad.index); assignments.set(slot, identity(pad)); }
    });
    return result;
  }
  const axisIndex = x => Number.isInteger(x) && x >= 0 && x < 64;
  function validCal(c) {
    return !!c && [c.min,c.center,c.max].every(finite) && c.min >= -1 && c.max <= 1 &&
      c.center - c.min > .05 && c.max - c.center > .05;
  }
  function normalizeConfig(cfg) {
    const map = presets.rc.map((fallback,i) => {
      const m = cfg?.map?.[i];
      return {x:axisIndex(m?.x)?m.x:fallback.x, y:axisIndex(m?.y)?m.y:fallback.y,
        xi:typeof m?.xi === 'boolean'?m.xi:fallback.xi, yi:typeof m?.yi === 'boolean'?m.yi:fallback.yi};
    });
    const cal = {};
    for (const [k,c] of Object.entries(cfg?.cal && typeof cfg.cal === 'object' ? cfg.cal : {})) {
      if (axisIndex(Number(k)) && validCal(c)) cal[k] = {min:c.min,center:c.center,max:c.max};
    }
    return {map,cal,dz:finite(cfg?.dz)?Math.max(0,Math.min(.2,cfg.dz)):.03};
  }
  const initializedConfigs = new WeakSet();
  function initializeConfig(pad, cfg, mapKey) {
    if (!pad || pad.connected === false || isPeripheral(pad) || !cfg ||
        pad.axes?.length < 4 || initializedConfigs.has(cfg)) return false;
    let saved;
    try { saved = JSON.parse(localStorage.getItem(mapKey)); } catch (_) {}
    initializedConfigs.add(cfg);
    // A saved/custom map belongs to the pilot. Auto-detection only replaces
    // the unsaved RC fallback, independently for every multiplayer seat.
    const savedMap = Array.isArray(saved) && saved.length === 2 &&
      saved.every(m => axisIndex(m?.x) && axisIndex(m?.y));
    if (savedMap || JSON.stringify(cfg.map) !== JSON.stringify(presets.rc)) return false;
    const map = clone(presets[resolvePreset(pad, 'auto')]);
    if (JSON.stringify(map) === JSON.stringify(cfg.map)) return false;
    cfg.map = map;
    try { localStorage.setItem(mapKey, JSON.stringify(map)); } catch (_) {}
    return true;
  }
  function applyCal(axis, value, cfg) {
    // A missing channel must stay neutral even with an off-centre calibration.
    if (!finite(value)) return 0;
    let out = Math.max(-1,Math.min(1,value));
    const c = cfg?.cal?.[axis];
    if (validCal(c)) out = (out-c.center)/(out<c.center?c.center-c.min:c.max-c.center);
    out = Math.max(-1,Math.min(1,out));
    const dz = finite(cfg?.dz)?Math.max(0,Math.min(.2,cfg.dz)):.03;
    return Math.abs(out) <= dz ? 0 : Math.sign(out)*(Math.abs(out)-dz)/(1-dz);
  }
  function readSticks(pad, cfg) {
    const sticks = presets.rc.map((_,i) => {
      const m = cfg?.map?.[i];
      const read = k => axisIndex(m?.[k]) ? applyCal(m[k],pad?.connected === false ? undefined : pad?.axes?.[m[k]],cfg) : 0;
      return {x:read('x')*(m?.xi?-1:1), y:read('y')*(m?.yi?1:-1)};
    });
    // Standard Gamepad D-pad is a second left stick. Opposing directions
    // cancel; an idle D-pad leaves proportional stick control untouched.
    if (pad?.mapping === 'standard') {
      const down = index => { const b=pad.buttons?.[index]; return (typeof b==='number'?b:b?.value ?? Number(!!b?.pressed)) > .5; };
      const x=Number(down(15))-Number(down(14)), y=Number(down(12))-Number(down(13));
      if (down(14)||down(15)) sticks[0].x=x;
      if (down(12)||down(13)) sticks[0].y=y;
    }
    return sticks;
  }
  let error = '';
  const relayMessage = 'flight-club:gamepads';
  let relayed = null, retryNativeAt = 0;
  // file:// frames can have an opaque origin even when the lobby may read USB.
  // Accept snapshots only from our immediate parent, never another frame/tab.
  if (window.addEventListener) window.addEventListener('message', event => {
    if (window.parent === window || event.source !== window.parent ||
        (event.origin !== window.location.origin &&
          !(window.location.protocol === 'file:' && event.origin === 'null')) ||
        event.data?.type !== relayMessage) return;
    const {pads, error:message} = event.data;
    if (!Array.isArray(pads) || pads.length > 64 || typeof message !== 'string') return;
    if (!pads.every(p => p === null || (p && Number.isInteger(p.index) && p.index >= 0 &&
        typeof p.id === 'string' && typeof p.mapping === 'string' && p.connected === true &&
        Array.isArray(p.axes) && p.axes.length <= 64 && p.axes.every(v => finite(v) && Math.abs(v) <= 1) &&
        Array.isArray(p.buttons) && p.buttons.length <= 128 &&
        p.buttons.every(b => b && finite(b.value) && b.value >= 0 && b.value <= 1 && typeof b.pressed === 'boolean')))) return;
    relayed = {pads,error:message,received:performance.now()};
  });
  function relayPads() {
    if (!relayed || performance.now() - relayed.received > 250) return null;
    error = relayed.error;
    return relayed.pads;
  }
  function browserPads() {
    if (retryNativeAt > performance.now()) {
      const pads = relayPads();
      if (pads) return pads;
    }
    try {
      if (!navigator.getGamepads) throw Error('Gamepad API unavailable');
      const pads = Array.from(navigator.getGamepads()).map(p => p && p.connected !== false ? p : null);
      // A frame awaiting its own activation may see no devices yet.
      if (!pads.some(Boolean)) {
        const forwarded = relayPads();
        if (forwarded?.some(Boolean)) return forwarded;
      }
      // A frame can see the mouse before the controller is activated in the
      // lobby. Do not discard a controller relayed by the parent in that case.
      const forwarded = relayPads();
      if (forwarded) for (const p of forwarded) {
        if (p && !pads.some(native => native?.index === p.index)) pads.push(p);
      }
      retryNativeAt=0; error=''; return pads;
    } catch (_) {
      retryNativeAt=performance.now()+1000;
      const pads = relayPads();
      if (pads) return pads;
      error=window.top !== window
        ? '正在由大廳連接手把，請按一下手把按鈕或撥動搖桿。若仍無訊號，請確認瀏覽器的手把權限。'
        : '無法讀取手把，請確認瀏覽器允許手把存取，並使用 Chrome 或 Edge。';
      return [];
    }
  }
  function relayTo(frame) {
    const destination = new URL(frame.src,window.location.href);
    const target = destination.protocol === 'file:' || destination.origin === 'null' ? '*' : destination.origin;
    let stopped = false, tick;
    function send() {
      if (stopped || !frame.isConnected) return;
      const pads = browserPads().map(p => p ? {
        index:p.index,id:p.id,mapping:p.mapping || '',connected:true,timestamp:p.timestamp,
        axes:Array.from(p.axes),buttons:Array.from(p.buttons,b=>({pressed:b.pressed,touched:b.touched,value:b.value}))
      } : null);
      frame.contentWindow.postMessage({type:relayMessage,pads,error},target);
      tick = window.requestAnimationFrame(send);
    }
    tick = window.requestAnimationFrame(send);
    return () => { stopped=true; window.cancelAnimationFrame(tick); };
  }
  function bridgePads(data) {
    if (!data || !Array.isArray(data.pads) || data.pads.length > 64) throw Error('無效橋接資料');
    const used = new Set();
    return data.pads.map(p => {
      if (!p || !Number.isInteger(p.id) || p.id<0 || p.id>1000000 || used.has(p.id) ||
          !Array.isArray(p.axes) || p.axes.length>64 || !p.axes.every(v=>finite(v)&&v>=-1&&v<=1) ||
          !Array.isArray(p.buttons) || p.buttons.length>128 || !p.buttons.every(v=>finite(v)&&v>=0&&v<=1)) throw Error('無效橋接通道');
      used.add(p.id);
      return {index:100+p.id,id:String(p.name||'手把'),bridged:true,connected:true,simulated:data.simulated===true,
        axes:p.axes.slice(),buttons:p.buttons.map(value=>({pressed:value>.5,value}))};
    });
  }
  function warnings(pad,cfg) {
    if (!pad) return [error || '尚未偵測到手把。接上後請按一下按鈕或撥動搖桿。'];
    const messages=[], used=[];
    if (isPeripheral(pad)) messages.push('目前選到滑鼠／鍵盤的遊戲控制器介面，請改選 SR-012／USB Joystick。');
    if (pad.simulated) messages.push('目前為橋接模擬訊號，沒有讀取實體 USB。');
    for (const [i,m] of (cfg?.map||[]).entries()) for (const k of ['x','y']) {
      const a=m[k]; used.push(a);
      if (!axisIndex(a) || !finite(pad.axes?.[a])) messages.push(`${i?'右':'左'}搖桿 ${k.toUpperCase()} 指向不存在或無效的軸 ${a}，此方向已歸零。`);
    }
    if (new Set(used).size<used.length) messages.push('多個方向共用同一通道，請確認是否為預期設定。');
    if (pad.mapping==='standard' && used.includes(4) && pad.axes.length===4) messages.push('此裝置可先試用「一般雙搖桿」對應。');
    return messages;
  }
  function mount(adapter) {
    // Mount after the shared interface and engine-owned setup fields are ready.
    window.addEventListener('load', () => {
      const root=document.getElementById('padSetup'), heading=root?.querySelector('.fc-card-heading');
      if (!heading) return;
      // A fully automated match has no controller seats to configure.
      if (adapter.profiles?.().length===0 && window.DroneControlSettings?.pilots.length===0) {
        return;
      }
      const box=document.createElement('div'); box.className='usb-health'; box.id='usbHealth';
      const status=document.createElement('p'); status.setAttribute('role','status');
      const actions=document.createElement('div'); actions.className='usb-actions';
      const label=document.createElement('label'); label.textContent='手把類型';
      const select=document.createElement('select'); select.id='usbPreset'; select.className='ps-sel'; select.setAttribute('aria-label','遙控器快速對應');
      for (const [key,text] of [['detect','重新自動辨識'],['custom','已儲存的方向設定'],['standard','一般雙搖桿手把'],['usbJoystick','USB Joystick／SR-012'],['rc','RC 遙控器模擬線']]) {
        const option=document.createElement('option');option.value=key;option.textContent=text;select.append(option);
        if(key==='custom')option.disabled=true;
      }
      select.addEventListener('change',()=>{const choice=select.value==='detect'?resolvePreset(adapter.pad(),'auto'):select.value;if(presets[choice]){
        adapter.cancelCalibration();const cfg=adapter.config();cfg.map=clone(presets[choice]);adapter.changed();update();
      }});
      label.append(select);root.querySelector('.usb-preset-field').append(label);
      let deviceSelect, deviceKey='';
      if (adapter.selectPad) {
        const deviceLabel=document.createElement('label'); deviceLabel.textContent='使用的手把';
        deviceSelect=document.createElement('select'); deviceSelect.id='usbDevice';
        deviceSelect.className='ps-sel'; deviceSelect.setAttribute('aria-label','USB 輸入裝置');
        deviceSelect.addEventListener('change',()=>{
          adapter.cancelCalibration();adapter.selectPad(deviceSelect.value==='auto'?null:Number(deviceSelect.value)); update();
        });
        deviceLabel.append(deviceSelect);actions.prepend(deviceLabel);
      }
      box.append(status,actions); root.querySelector('.usb-setup-content').prepend(box);
      const stage=document.createElement('ol');stage.className='usb-setup-steps';stage.setAttribute('aria-label','遙控器準備步驟');
      for(const [id,title,hint]of [['connect','連接控制器','USB 插入並按一下按鈕'],['direction','確認方向','推動雙桿，查看回應'],['ready','回中完成','放開按鈕，雙桿置中']]){
        const item=document.createElement('li');item.dataset.usbStep=id;const number=document.createElement('b');number.textContent=String(stage.children.length+1).padStart(2,'0');const copy=document.createElement('span'),strong=document.createElement('strong'),small=document.createElement('small');strong.textContent=title;small.textContent=hint;copy.append(strong,small);item.append(number,copy);stage.append(item);
      }
      box.before(stage);
      const healthHeading=document.createElement('div');healthHeading.className='usb-health-heading';const dot=document.createElement('i'),healthTitle=document.createElement('strong');healthTitle.textContent='控制器連線';healthHeading.append(dot,healthTitle);box.prepend(healthHeading);
      const headerTitle=heading.querySelector('h2');if(headerTitle)headerTitle.textContent='遙控器設定';
      const summary=root.querySelector('#usbHelp>summary');if(summary){summary.textContent='進階對應與校正';const hint=document.createElement('small');hint.textContent='方向反向・裝置類型・行程校正';summary.append(hint);}
      const raw=document.createElement('details');raw.className='usb-raw-inspector';const rawSummary=document.createElement('summary');rawSummary.textContent='手動通道與即時原始值';const rawDescription=document.createElement('p');rawDescription.textContent='只有自動對應不符合裝置時才需調整。X 為左右、Y 為上下；更改通道會取消進行中的校正。';rawDescription.className='usb-field-hint';const rawGrid=document.createElement('div');rawGrid.className='usb-raw-grid';
      const rawMaps=Array.from(root.querySelectorAll('.usb-legacy .ps-map')),rawValues=Array.from(root.querySelectorAll('.usb-legacy .ps-vx'),node=>node.parentElement.parentElement);
      rawMaps.forEach((map,index)=>{const unit=document.createElement('section');unit.className='usb-raw-unit';const title=document.createElement('h3');title.textContent=index?'右搖桿通道':'左搖桿通道';unit.append(title);if(rawValues[index])unit.append(rawValues[index]);unit.append(map);rawGrid.append(unit);});
      raw.append(rawSummary,rawDescription,rawGrid);root.querySelector('.usb-help-body').append(raw);
      const footerHint=root.querySelector('.fc-pad-footer .fc-footer-hint');if(footerHint)footerHint.textContent='設定即時套用 · 回中後繼續';
      const guide=root.querySelector('.usb-flight-guide');
      const calButton=document.getElementById('psCal'), cancel=document.getElementById('psCalCancel');
      const calibrationFeedback=message=>{const feedback=document.getElementById('psCalFileMsg');feedback.hidden=false;feedback.textContent=message;};
      cancel?.addEventListener('click',()=>{adapter.cancelCalibration();calibrationFeedback('已取消校正，保留原設定。');});
      root.querySelector('#usbHelp').addEventListener('toggle',event=>{if(!event.currentTarget.open)adapter.cancelCalibration();});
      let calibrationIdentity='';
      calButton.addEventListener('click',event=>{
        const pad=adapter.pad();
        if(!pad){event.stopImmediatePropagation();status.textContent='請先接上並選擇手把，再開始校正。';return;}
        calibrationIdentity=`${pad.index}:${pad.id}`;
      },true);
      document.getElementById('psCalDone').addEventListener('click',event=>{
        const pad=adapter.pad();
        if(!pad||`${pad.index}:${pad.id}`!==calibrationIdentity){event.stopImmediatePropagation();adapter.cancelCalibration();calibrationFeedback('連線或手把已變更，請重新校正；原校正已保留。');}
      },true);
      let previousIdentity='';
      function update() {
        if (!adapter.open()) return;
        let mappingChanged=false;
        for (const profile of adapter.profiles?.() || []) {
          if (initializeConfig(profile.pad,profile.config,profile.mapKey)) mappingChanged=true;
        }
        if (mappingChanged) { adapter.cancelCalibration();adapter.changed(); }
        const pads=controllerPads(browserPads()), p=adapter.pad();
        if (deviceSelect) {
          const selected=adapter.selectedPad();
          deviceSelect.parentElement.hidden=!(pads.length>1||selected!==null||(!p&&pads.length>0));
          const key=JSON.stringify([selected,pads.map(p=>p&&[p.index,p.id])]);
          if (key!==deviceKey && document.activeElement!==deviceSelect) {
            deviceKey=key;
            const options=[['auto','自動選擇手把（建議）'],
              ...pads.map(p=>[String(p.index),`#${p.index} ${p.id}`])];
            if (selected!==null && !pads.some(p=>p?.index===selected)) options.push([String(selected),'指定的搖桿（未接上）']);
            deviceSelect.replaceChildren(...options.map(([value,text])=>{
              const option=document.createElement('option'); option.value=value;option.textContent=text;return option;
            }));
            deviceSelect.value=selected===null?'auto':String(selected);
          }
        }
        const issues=warnings(p,adapter.config());
        const identity=p?`${p.index}:${p.id}`:'';
        if(identity!==previousIdentity){adapter.cancelCalibration();previousIdentity=identity;}
        calButton.disabled=!p;
        const calBox=document.getElementById('psCalBox');
        const calibrating=!calBox.hidden;
        calButton.disabled=!p||calibrating;
        document.getElementById('psCalDone').disabled=!p;
        // Changing channels invalidates an in-progress sweep.
        document.getElementById('psCalClear').disabled=calibrating;
        if (!p) issues.splice(0,issues.length,error || (pads.length?'請選擇要使用的手把。':'尚未找到手把：接上 USB 後，按一下手把按鈕或撥動搖桿。'));
        const centered=p&&readSticks(p,adapter.config()).every(s=>Math.max(Math.abs(s.x),Math.abs(s.y))<=.12);
        const currentStep=!p?'connect':calibrating||issues.length||!centered?'direction':'ready';
        root.dataset.usbStage=currentStep;root.dataset.usbCalibrating=String(calibrating);
        stage.querySelectorAll('[data-usb-step]').forEach((item,index)=>{const active=item.dataset.usbStep===currentStep;item.setAttribute('aria-current',active?'step':'false');item.dataset.complete=String(index<['connect','direction','ready'].indexOf(currentStep));});
        healthTitle.textContent=calibrating?'正在校正行程':!p?'等待控制器':issues.length?'需要確認對應':centered?'控制器已就緒':'請將雙桿回中';
        const message=!p?issues[0]:issues.length?(p.simulated?'目前使用模擬訊號。':'搖桿設定需要調整，請展開下方選項重新辨識手把。'):`已連接 · ${p.id}。${centered?'雙桿已回中，可繼續。':'正在讀取搖桿；請將雙桿回中。'}`;
        if (status.textContent!==message) status.textContent=message;
        const manual=document.getElementById('psModeManual').getAttribute('aria-pressed')==='true';
        guide.textContent=`左桿：升降／轉向；右桿：前後／平移。${manual?'目前為手動油門，需自行維持高度。':'雙桿回中後維持高度並逐步煞停。'}`;
        if (document.activeElement!==select) select.value=Object.keys(presets).find(k=>JSON.stringify(presets[k])===JSON.stringify(adapter.config().map))||'custom';
        box.dataset.state=issues.length?'warn':'ok';
      }
      root.querySelectorAll('.ps-map select,.usb-direction-grid input').forEach(field=>field.addEventListener('change',()=>adapter.cancelCalibration(),true));
      root.querySelectorAll('#psSwap,#psFlipY').forEach(field=>field.addEventListener('click',()=>adapter.cancelCalibration(),true));
      update(); setInterval(update,250);
    });
  }
  window.DroneUSBInput=Object.freeze({presets,flightModes,resolvePreset,isPeripheral,controllerPads,selectPad,resolvePads,forgetAssignment:slot=>assignments.delete(slot),normalizeConfig,initializeConfig,validCal,applyCal,readSticks,browserPads,relayTo,bridgePads,warnings,mount});
})();
