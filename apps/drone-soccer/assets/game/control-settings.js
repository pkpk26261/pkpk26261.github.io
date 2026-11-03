/* Per-pilot controls and gears. The game owns device assignment and physics. */
(() => {
  'use strict';
  const radioGears = Object.freeze([
    Object.freeze({ label: '低速', powerLimit: 0.333, drive: 1, brake: 1, yaw: 1, climb: 1.65, descent: 1.50, liftRamp: 0.3, hint: '最大動力 33.3%・適合練習，仍會互相碰撞' }),
    Object.freeze({ label: '中速', powerLimit: 0.666, drive: 1, brake: 1, yaw: 1, climb: 1.65, descent: 1.50, liftRamp: 0.3, hint: '最大動力 66.6%・一般飛行，留意煞停距離' }),
    Object.freeze({ label: '高速', powerLimit: 1, drive: 1, brake: 1, yaw: 1, climb: 1.65, descent: 1.50, liftRamp: 0.3, hint: '最大動力 100%・速度越快越容易碰撞，請提早煞車' })
  ]);
  const keyboardGears = Object.freeze(radioGears.map(gear => Object.freeze({ ...gear,
    climb: 3.5 * gear.powerLimit, descent: 3 * gear.powerLimit,
    hint: `飛行油門 ${Number((gear.powerLimit * 100).toFixed(1))}%・平移 ${Number((3 * gear.powerLimit).toFixed(2))} m/s，上升 ${Number((3.5 * gear.powerLimit).toFixed(2))} m/s，請提早煞車`
  })));
  let gears = radioGears;
  const outputLabel = () => adapter?.radio ? '最大動力' : '飛行油門上限';
  const actions = [['w', '上升'], ['s', '下降'], ['a', '左轉'], ['d', '右轉'],
    ['arrowup', '前進'], ['arrowdown', '後退'], ['arrowleft', '向左平移'], ['arrowright', '向右平移'], ['gear', '切換檔位']];
  const defaultMaps = [
    ['w','s','a','d','arrowup','arrowdown','arrowleft','arrowright'],
    ['g','v','f','b','u','h','y','j'], ['1','2','3','4','5','6','7','8'],
    ['q','e','z','x','i','k','o','p'], ['[',']',';',"'",'.',',','/','\\'],
    ['9','0','n','`','-','=',' ','enter']
  ];
  const reserved = new Set(['t', 'l', 'c', 'r', 'm', 'escape', 'tab', 'enter']);
  const allowed = new Set([...Array.from('abcdefghijklmnopqrstuvwxyz0123456789[];\'.,/\\`-='),
    'arrowup','arrowdown','arrowleft','arrowright',' ','control','shift', ...Array.from({length:12}, (_,i)=>`f${i+1}`)]);
  const padStates = new WeakMap(), held = new Set(), references = new Set(), notices = new Map();
  let noticeStack, noticeFrame = 0;
  let adapter, pilots = [], preferences = [], storageKey, dialog, selected = 0, learning = null, status, returnTarget;
  let selectedSection = 'keys', gearExplanation;
  const copy = value => JSON.parse(JSON.stringify(value));
  const keyLabel = key => ({arrowup:'↑',arrowdown:'↓',arrowleft:'←',arrowright:'→',' ':'Space',control:'Ctrl',shift:'Shift',enter:'Enter'})[key] || key.toUpperCase();
  const make = (tag, text, className) => { const el = document.createElement(tag); if (text) el.textContent = text; if (className) el.className = className; return el; };
  const blocked = () => !!document.querySelector('dialog[open], :popover-open') || adapter?.setupOpen?.() || document.hidden || window.DroneFlightInput?.paused;
  const editing = event => event.target?.closest?.('input,textarea,select,button,[role="button"],[role="tab"],[contenteditable="true"],dialog[open]');
  function engineKeys(index) {
    const map = pilots[index].map || Object.fromEntries(defaultMaps[index].map((key,i)=>[key,actions[i][0]]));
    return Object.fromEntries(actions.slice(0,8).map(([action],i) => [action, Object.keys(map).find(key => map[key] === action) || defaultMaps[index][i]]));
  }
  function defaults(index) {
    // Q belongs to P1's gear. Move the legacy P4 ascent key to the freed F1.
    const keys = engineKeys(index);
    Object.keys(keys).forEach(action => { if (keys[action] === 'q') keys[action] = 'f1'; });
    return {keys:{...keys,gear:index === 0 ? 'q' : `f${index+1}`},pad:{type:'button',index:4},initial:adapter?.radio ? 0 : 1};
  }
  function validate(rows) {
    if (!Array.isArray(rows) || rows.length !== pilots.length) return '設定的玩家數量不符';
    const used = new Set();
    for (const [i,row] of rows.entries()) {
      if (!row || !row.keys || !Number.isInteger(row.initial) || row.initial < 0 || row.initial > 2) return '檔位設定無效';
      for (const [action] of actions) {
        if (adapter.radio) continue;
        const key = row.keys[action];
        // Existing six-player layouts use Enter for lateral movement.
        if (!allowed.has(key) && key !== 'enter' || reserved.has(key) && !(key === 'enter' && defaults(i).keys[action] === key)) return '此按鍵保留給系統，請選擇其他按鍵';
        if (used.has(key)) return `${keyLabel(key)} 已指派給其他操作或玩家`;
        used.add(key);
      }
      const p = row.pad;
      if (p?.type === 'button' && window.DroneControllerActions?.extraUsed(i,p.index)) return '此按鈕已用於其他遙控器操作';
      if (!p || !['button','axis','none'].includes(p.type) || !Number.isInteger(p.index) || p.index < 0 || p.index > 63 || p.type === 'axis' && ![-1,1].includes(p.direction)) return '遙控器按鈕設定無效';
    }
    return '';
  }
  function release() {
    held.clear();
    pilots.forEach(p => { if (p.keys) Object.keys(p.keys).forEach(k => { p.keys[k] = false; }); });
  }
  function save(rows) {
    const error = validate(rows); if (error) { if(status)status.textContent=error; return error; }
    release(); preferences = copy(rows); refreshReferences();
    pilots.forEach(p => padStates.delete(p.actor));
    try { localStorage.setItem(storageKey, JSON.stringify(preferences)); }
    catch (_) { if (status) status.textContent = '設定已套用；瀏覽器無法儲存，關閉頁面後需重新設定。'; return ''; }
    if (status) status.textContent = '設定已儲存在此瀏覽器。';
    return '';
  }
  const pilotIndex = actor => pilots.findIndex(p => p.actor === actor);
  function gearFor(actor) {
    const activeGears = !adapter?.radio && (actor.touchActive || actor.touchActiveLeft || actor.ai || actor.aiAutoGear) ? radioGears : gears;
    if (!actor.aiAutoGear && pilotIndex(actor) < 0 && !Number.isInteger(actor.controlGear)) return activeGears[2];
    return activeGears[Number.isInteger(actor.controlGear) && actor.controlGear >= 0 && actor.controlGear < activeGears.length ? actor.controlGear : 1];
  }
  function cycle(actor) {
    const index = pilotIndex(actor);
    if (index < 0 || actor.ai) return;
    actor.controlGear = ((actor.controlGear ?? 1) + 1) % 3;
    showGearNotice(index);
  }
  function showGearNotice(index) {
    if (!noticeStack) {
      noticeStack=make('div',null,'fc-gear-notices');
      noticeStack.setAttribute('aria-live','polite');noticeStack.setAttribute('aria-atomic','false');
      document.body.append(noticeStack);
    }
    let notice=notices.get(index);
    if (!notice) {
      const card=make('div',null,'fc-gear-notice');card.setAttribute('role','status');
      notice={card};notices.set(index,notice);noticeStack.append(card);
    }
    clearTimeout(notice.timer);
    const gear=gears[pilots[index].actor.controlGear];
    notice.card.dataset.gear=pilots[index].actor.controlGear;
    notice.card.replaceChildren(make('span',`P${index+1} · ${pilots[index].label}`,'fc-gear-notice-pilot'),make('strong',`已切換${gear.label} · ${adapter.radio ? '動力上限' : '飛行油門'} ${Number((gear.powerLimit*100).toFixed(1))}%`),make('span',gear.hint));
    notice.card.hidden=false;
    positionGearNotices();
    notice.timer=setTimeout(()=>{notice.card.hidden=true;},2600);
  }
  // Each human pilot owns a view, even when all views share one WebGL canvas.
  // Run only while feedback is visible so moving cameras cannot cover the drone.
  function positionGearNotices() {
    if (noticeFrame) cancelAnimationFrame(noticeFrame);
    noticeFrame=0;
    let active=false;
    for (const [index,notice] of notices) {
      if (notice.card.hidden) continue;
      active=true;
      const actor=pilots[index].actor, view=adapter.view?.(actor);
      const bounds=view?.element?.getBoundingClientRect();
      if (!bounds || bounds.width<=0 || bounds.height<=0) { notice.card.style.visibility='hidden'; continue; }
      const card=notice.card, tiny=bounds.height<90, short=bounds.height<200, margin=tiny?2:short?4:12, gap=tiny?2:short?8:20;
      card.dataset.compact=String(bounds.width<600 || bounds.height<500);
      card.dataset.short=String(short);
      card.dataset.tiny=String(tiny);
      card.style.width=`${Math.max(0,Math.min(320,bounds.width-margin*2))}px`;
      card.style.maxWidth=`${Math.max(0,bounds.width-margin*2)}px`;
      let bottom=bounds.top+bounds.height/2;
      const camera=view.camera, THREE=window.THREE;
      if (camera && actor.mesh && THREE) {
        // A conservative camera-aligned cube encloses the 20 cm cage, rotors
        // and beacon at every tilt. Project it into this pilot's own viewport.
        const center=actor.mesh.getWorldPosition(new THREE.Vector3()).applyMatrix4(camera.matrixWorldInverse);
        const radius=0.15;
        if (center.z < -radius-camera.near) {
          let droneTop=Infinity;
          for (const x of [-radius,radius]) for (const y of [-radius,radius]) for (const z of [-radius,radius]) {
            const point=new THREE.Vector3(center.x+x,center.y+y,center.z+z).applyMatrix4(camera.projectionMatrix);
            droneTop=Math.min(droneTop,bounds.top+(1-point.y)*bounds.height/2);
          }
          bottom=Math.min(bottom,droneTop-gap);
        }
      }
      // Never clamp downward onto the aircraft when it approaches the top edge.
      const height=card.getBoundingClientRect().height;
      const fits=bottom-height>=bounds.top+margin;
      card.style.visibility=fits?'':'hidden';
      card.style.left=`${bounds.left+bounds.width/2}px`;
      card.style.top=`${bottom-height}px`;
    }
    if (active) noticeFrame=requestAnimationFrame(positionGearNotices);
  }
  function referenceRow(section,label,keys) {
    const row=make('div',null,'fc-guide-row'),caps=make('span',null,'fc-guide-keys');
    keys.forEach((key,i)=>{if(i)caps.append(document.createTextNode(' / '));caps.append(make('kbd',keyLabel(key),'fc-keycap'));});
    row.append(make('span',label,'fc-guide-label'),caps);section.append(row);
  }
  function drawReference(view) {
    const {host,options}=view,index=Math.min(view.index,pilots.length-1);
    if(index<0)return;
    const row=preferences[index],parts=[];
    if(pilots.length>1) {
      const label=make('label','查看玩家操作','fc-reference-picker'),picker=make('select');picker.setAttribute('aria-label','查看玩家操作');
      pilots.forEach((p,i)=>{const option=make('option',`P${i+1} · ${p.label}`);option.value=i;picker.append(option);});picker.value=index;
      picker.onchange=event=>{event.stopPropagation();release();view.index=Number(picker.value);drawReference(view);};label.append(picker);parts.push(label);
    }
    const gear=make('section',null,'fc-reference-gear');
    if(!adapter.radio)referenceRow(gear,'切換檔位（按一下）',[row.keys.gear]);
    if(adapter.radio)gear.append(make('p',`切換檔位：${window.DroneControllerActions?.label(index,'gear')||'LB'}`));
    gear.append(make('p','低速 → 中速 → 高速 → 低速'));
    gear.append(make('p','降檔仍有慣性，請提早煞車。'));
    const levels=make('div',null,'fc-reference-gears');
    gears.forEach((g,i)=>{const item=make('div');item.dataset.gear=i;item.append(make('strong',g.label),make('span',`${outputLabel()} ${Number((g.powerLimit*100).toFixed(1))}%`));levels.append(item);});gear.append(levels);
    parts.push(gear);
    if(!adapter.radio) {
      const movement=make('section',null,'fc-guide-section');movement.append(make('h3','方向與高度 · 長按操作'));
      for(const [label,a,b] of [['上升 / 下降','w','s'],['前進 / 後退','arrowup','arrowdown'],['向左 / 向右平移','arrowleft','arrowright'],['向左 / 向右旋轉','a','d']])referenceRow(movement,label,[row.keys[a],row.keys[b]]);
      if(options.backup && index===0 && pilots.length<4)referenceRow(movement,'升降備用鍵',[' ','control']);
      if(Object.values(engineKeys(index)).includes('q')&&row.keys.w==='f1')movement.append(make('p','原 Q 上升鍵改為 F1，避免與 P1 換檔重複。','fc-reference-note'));
      parts.push(movement);
    }
    if(!adapter.radio && (options.system || options.takeoff)) {
      const system=make('section',null,'fc-guide-section');system.append(make('h3',options.system?'起降與系統 · 按一下':'起飛與降落 · 按一下'));
      referenceRow(system,pilots.length>1?'全體啟槳 / 降落':'啟槳 / 降落',['t','l']);
      system.append(make('p','先按起飛或上升鍵啟槳，機身保持在地面；放開後再按上升鍵才離地。','fc-reference-note'));
      system.append(make('p','降到最低高度後，持續往下約 2 秒可降落停槳；往上可中止。','fc-reference-note'));
      system.append(make('p','受困時強下拉約 2 秒可停機；回中並直立落地後，再按重新啟槳。','fc-reference-note'));
      if(options.system) {
        if(document.querySelector('#btnCam,[id$="-btnCam"]')&&pilots.length<4)referenceRow(system,pilots.length>1?'P1 鏡頭視角':'切換鏡頭視角',['c']);
        if(options.zoom && pilots.length<4)referenceRow(system,'定中視角縮放 / 網後距離',['[',']']);
        referenceRow(system,'重設 / 靜音',['r','m']);
      }
      parts.push(system);
    }
    const configure=make('button',adapter.radio?'修改搖桿與檔位設定':'修改按鍵與檔位設定','fc-control-entry');configure.type='button';
    configure.onclick=event=>{event.stopPropagation();selected=index;showSettings(configure);};parts.push(configure);
    host.replaceChildren(...parts);
    if(adapter.radio)window.DroneControllerActions?.reference(host,index);
  }
  function refreshReferences() { references.forEach(drawReference); }
  function mountReference(host,index=0,options={}) {
    host.classList.add('fc-control-reference');host.classList.add('fc-ui-region');
    const view={host,index,options};references.add(view);drawReference(view);
  }
  function padValue(pad, binding) {
    if (binding.type === 'button') { const b = pad.buttons?.[binding.index]; return typeof b === 'number' ? b : b?.pressed ? 1 : b?.value || 0; }
    if (binding.type === 'axis') return (pad.axes?.[binding.index] || 0) * binding.direction;
    return 0;
  }
  function pollPad(actor, pad, enabled) {
    if(adapter?.radio && window.DroneControllerActions)enabled=window.DroneControllerActions.poll(actor,pad,enabled);
    const index = pilotIndex(actor); if (index < 0) return;
    const previous = padStates.get(actor), identity = pad ? `${pad.bridged?'bridge':'browser'}:${pad.index}:${pad.id}` : '';
    const binding = preferences[index].pad;
    const value = pad ? padValue(pad, binding) : 0;
    const down = value > (previous?.down ? 0.35 : 0.65);
    // Prime every reconnect, reassignment and settings transition. A held switch
    // must return to neutral before it can shift; repeats never advance a gear.
    if (!enabled || blocked() || !pad || previous?.identity !== identity || !previous?.enabled) {
      padStates.set(actor, {identity,enabled:enabled && !blocked(),down});
      if (learning?.kind === 'pad' && learning.index === index && pad) learnPad(pad, previous);
      return;
    }
    if (down && !previous.down) cycle(actor);
    padStates.set(actor, {identity,enabled:true,down});
  }
  function learnPad(pad) {
    if (!learning?.baseline || learning.identity !== `${pad.index}:${pad.id}`) {
      learning.identity = `${pad.index}:${pad.id}`;
      learning.baseline = {buttons:Array.from(pad.buttons || [], (_,i)=>padValue(pad,{type:'button',index:i})),axes:Array.from(pad.axes || [])};
      return;
    }
    let binding;
    for (let i=0;i<(pad.buttons?.length || 0);i++) if (padValue(pad,{type:'button',index:i})>.65 && learning.baseline.buttons[i]<.35) { binding={type:'button',index:i}; break; }
    if (!binding) for (let i=4;i<(pad.axes?.length || 0);i++) if (Math.abs(pad.axes[i]-learning.baseline.axes[i])>1.2) { binding={type:'axis',index:i,direction:pad.axes[i]>0?1:-1}; break; }
    if (binding) { const rows=copy(preferences); rows[learning.index].pad=binding; learning=null; save(rows); renderEditor(); }
  }
  function keyboard(event) {
    if(adapter?.radio)return;
    if (!adapter || event.metaKey || event.altKey || event.ctrlKey && !event.code?.startsWith('Control')) { if (adapter) release(); return; }
    if (learning?.kind === 'key') {
      event.preventDefault(); event.stopImmediatePropagation();
      if (event.type !== 'keydown' || event.repeat) return;
      if (event.key === 'Escape') { learning=null;status.textContent='已取消按鍵變更。';renderEditor(); return; }
      const key = window.DroneFlightInput.key({...event, type:event.type, code:event.code, key:event.key, preventDefault(){}});
      const rows=copy(preferences); rows[learning.index].keys[learning.action]=key;
      const error=save(rows);
      if (error) status.textContent=error; else { learning=null; renderEditor(); }
      return;
    }
    const resume=event.target?.closest?.('.fc-flight-toolbar') && !blocked() && !['Tab','Enter',' ','Escape'].includes(event.key);
    if (editing(event) && !resume) return;
    if (resume) event.target.blur();
    const raw=window.DroneFlightInput.key(event);
    const down=event.type === 'keydown';
    let handled=false;
    for (const [i,pilot] of pilots.entries()) {
      const row=preferences[i];
      if (row.keys.gear === raw) {
        handled=true;
        if (down && !event.repeat && !held.has(raw) && !blocked()) cycle(pilot.actor);
      }
    }
    // Preserve original startup and shortcuts. Only replace movement dispatch
    // after a user changes that layout; the existing engines keep canonical keys.
    const customized=!adapter.radio && pilots.some((_,i)=>actions.slice(0,8).some(([a])=>preferences[i].keys[a]!==engineKeys(i)[a]));
    const starting=Array.from(document.querySelectorAll('.fc-start-overlay')).some(el=>el.getClientRects().length && !el.classList.contains('opacity-0'));
    if (customized && !starting && !blocked()) {
      for (const [i,pilot] of pilots.entries()) {
        for (const [action] of actions.slice(0,8)) {
          if (preferences[i].keys[action] === raw) {
            handled=true; pilot.keys[action]=down;
            if (down && ['w','s'].includes(action)) pilot.actor.autoFlightState='none';
          }
          if (engineKeys(i)[action] === raw) handled=true;
        }
      }
      if (down && handled) adapter.wake?.();
    }
    if (handled) { event.preventDefault(); event.stopImmediatePropagation(); if(down)held.add(raw);else held.delete(raw); }
  }
  function renderEditor() {
    // Keep the shared explanation alive while rebuilding this player's controls.
    gearExplanation?.remove();
    const content=dialog.querySelector('.fc-control-editor'); content.replaceChildren();
    const p=pilots[selected], row=preferences[selected];
    const sectionTitle=adapter.radio?'檔位設定':'按鍵配置';
    const title=make('h3',pilots.length>1?`${p.label} · ${sectionTitle}`:sectionTitle); content.append(title);
    const grid=make('div',null,'fc-control-grid');
    for(const [action,label] of actions) {
      if(adapter.radio)continue;
      const item=make('div',null,'fc-control-row'), name=make('span',label), button=make('button',keyLabel(row.keys[action]));
      button.type='button';button.dataset.controlAction=action;button.setAttribute('aria-label',`設定${label}按鍵`);
      button.onclick=()=>{window.DroneControllerActions?.cancelLearning();learning={kind:'key',index:selected,action};content.querySelectorAll('[data-control-action]').forEach(b=>{b.textContent=keyLabel(row.keys[b.dataset.controlAction]);b.removeAttribute('aria-pressed');});button.textContent='請按新按鍵…';button.setAttribute('aria-pressed','true');status.textContent='正在設定「'+label+'」。請按新按鍵；Esc 取消。重複按鍵會提示衝突。';};
      item.append(name,button);grid.append(item);
    }
    if(!adapter.radio)content.append(grid);
    const initialLabel=make('label','下次開始時的速度'), initial=make('select'); initial.setAttribute('aria-label','預設檔位');
    gears.forEach((g,i)=>{const option=make('option',`${g.label} · ${outputLabel()} ${Number((g.powerLimit*100).toFixed(1))}%`);option.value=i;initial.append(option);});initial.value=row.initial;
    initial.onchange=()=>{const rows=copy(preferences);rows[selected].initial=Number(initial.value);save(rows);};initialLabel.append(initial);content.append(initialLabel);
    if(adapter.radio) {
      const label=make('label','遙控器「切換檔位」'), select=make('select');select.setAttribute('aria-label','遙控器切換檔位');
      const none=make('option','不使用遙控器換檔');none.value='none';select.append(none);
      for(let i=0;i<32;i++){const option=make('option',`按鈕 ${i+1}（B${i}）`);option.value=`button:${i}`;select.append(option);}
      for(let i=4;i<16;i++)for(const direction of [-1,1]){const option=make('option',`開關通道 ${i+1}（Axis ${i}，${direction>0?'正向':'負向'}）`);option.value=`axis:${i}:${direction}`;select.append(option);}
      select.value=row.pad.type==='none'?'none':`${row.pad.type}:${row.pad.index}${row.pad.type==='axis'?`:${row.pad.direction}`:''}`;
      select.onchange=()=>{const [type,index,direction]=select.value.split(':');const rows=copy(preferences);rows[selected].pad={type,index:Number(index)||0,...(type==='axis'?{direction:Number(direction)}:{})};learning=null;const error=save(rows);if(error)renderEditor();};
      const manual=make('details',null,'fc-binding-advanced');manual.append(make('summary','手動選擇換檔按鈕／開關'),label);label.append(select);
      const learn=make('button','按一下，偵測換檔按鈕');learn.type='button';learn.onclick=()=>{window.DroneControllerActions?.cancelLearning();learning={kind:'pad',index:selected};status.textContent='請先放開按鈕，再按一下想用的換檔按鈕。也可撥動額外開關；按「取消偵測」即可結束。';learn.textContent='重新偵測換檔按鈕';cancel.hidden=false;};
      const cancel=make('button','取消偵測');cancel.type='button';cancel.hidden=true;cancel.onclick=()=>{learning=null;status.textContent='已取消偵測，保留原設定。';renderEditor();};
      content.append(make('p',row.pad.type==='none'?'目前：尚未設定搖桿換檔':row.pad.type==='button'?`目前：按鈕 ${row.pad.index+1} 換檔`:`目前：通道 ${row.pad.index+1} 換檔`,'fc-binding-current'),learn,cancel,manual);
      const axes=make('button','搖桿對應與範圍校正');axes.type='button';axes.onclick=()=>{
        dialog.close();
        const parentDialog=returnTarget?.closest('dialog');if(parentDialog?.open)parentDialog.close();
        adapter.openPad?.();
      };content.append(axes);
      content.append(make('p','同一顆按鈕可用在不同玩家各自的搖桿；必須先完成裝置指派。標準手把預設使用 LB 換檔。'));
    } else content.append(make('p','點選右側按鍵，再按想換的新按鍵。共用快捷鍵與其他玩家已使用的按鍵不能重複。'));
    const reset=make('button',pilots.length>1?'恢復此玩家預設':'恢復預設設定');reset.type='button';reset.onclick=()=>{const rows=copy(preferences);rows[selected]=defaults(selected);const error=save(rows);status.textContent=error || '已恢復預設設定；目前檔位維持不變。';learning=null;renderEditor();};content.append(reset);
    if(adapter.radio)window.DroneControllerActions?.editor(content,selected);
    organizeEditor(content);
  }
  function organizeEditor(content) {
    const nodes=Array.from(content.children), grid=content.querySelector('.fc-control-grid');
    const reset=nodes.find(node=>node.matches('button')&&/^恢復/.test(node.textContent));
    const axes=nodes.find(node=>node.matches('button')&&/範圍校正/.test(node.textContent));
    const controller=content.querySelector('.fc-controller-editor');
    const sections=adapter.radio
      ? [['gear','檔位與換檔','設定開始速度與換檔方式。'],['actions','遙控器按鈕','將實體按鈕對應到遊戲動作。'],['device','裝置與校正','確認搖桿方向、中立點與行程。']]
      : [['keys','飛行按鍵','點選目前的按鍵，再按下新的按鍵。'],['speed','起始檔位','設定下次進場時的預設速度。']];
    if(!sections.some(([id])=>id===selectedSection))selectedSection=sections[0][0];
    const nav=dialog.querySelector('.fc-controls-navigation');nav.replaceChildren();
    const panels=new Map();
    for(const [id,label,description]of sections){
      const panel=make('section',null,'fc-controls-panel');panel.dataset.controlsPanel=id;panel.id='fc-controls-panel-'+id;panel.setAttribute('role','tabpanel');panel.setAttribute('aria-labelledby','fc-controls-tab-'+id);
      const heading=make('div',null,'fc-controls-section-heading');heading.append(make('p',id==='keys'?'KEY BINDINGS':id==='speed'?'STARTING GEAR':id==='actions'?'CONTROLLER ACTIONS':id==='device'?'DEVICE SETUP':'GEAR CONTROL','fc-controls-section-kicker'),make('h3',label),make('p',description));panel.append(heading);panels.set(id,panel);
      const tab=make('button',null,'fc-controls-tab');tab.type='button';tab.id='fc-controls-tab-'+id;tab.dataset.controlsSection=id;tab.setAttribute('role','tab');tab.setAttribute('aria-controls',panel.id);tab.append(make('span',label),make('span','›','fc-controls-nav-arrow'));tab.onclick=()=>{const cancelled=learning||window.DroneControllerActions?.learning;selectedSection=id;learning=null;window.DroneControllerActions?.cancelLearning();if(cancelled)status.textContent='已取消偵測，保留原設定。';renderEditor();dialog.querySelector('[data-controls-section="'+id+'"]').focus({preventScroll:true});};nav.append(tab);
    }
    const primary=panels.get(adapter.radio?'gear':'keys'), speed=panels.get(adapter.radio?'gear':'speed');
    for(const node of nodes){
      if(node===reset||node.matches('h3'))continue;
      if(node===controller){panels.get('actions').append(node);continue;}
      if(node===axes){panels.get('device').append(make('div','控制器的方向與校正會沿用目前裝置，調整前請先放開全部按鈕並將雙桿回中。','fc-controls-device-note'),node);continue;}
      if(node.matches('label')){speed.append(node);node.classList.add('fc-controls-option');continue;}
      if(node===grid||node.matches('.fc-binding-current,.fc-binding-advanced')||node.matches('button'))primary.append(node);
      else if(adapter.radio&&node.matches('p'))panels.get('device').append(node);
      else primary.append(node);
    }
    if(gearExplanation)speed.append(gearExplanation);
    content.replaceChildren(...panels.values());
    if(reset){reset.classList.add('fc-controls-reset');reset.textContent=adapter.radio?'還原檔位與換檔':pilots.length>1?'還原此玩家按鍵與檔位':'還原按鍵與檔位';dialog.querySelector('.fc-controls-reset-slot').replaceChildren(reset);}
    selectControlSection();
  }
  function selectControlSection(){
    dialog.querySelectorAll('[data-controls-panel]').forEach(panel=>{panel.hidden=panel.dataset.controlsPanel!==selectedSection;});
    dialog.querySelectorAll('[data-controls-section]').forEach(button=>{const active=button.dataset.controlsSection===selectedSection;button.setAttribute('aria-selected',String(active));button.tabIndex=active?0:-1;});
    const body=dialog.querySelector('.fc-control-body');if(body)body.scrollTop=0;
  }
  function showSettings(target) {
    if (!dialog || !pilots.length) return;
    release();learning=null;returnTarget=target;
    const picker=dialog.querySelector('[aria-label="選擇操作玩家"]');
    if(picker)picker.value=selected;
    renderEditor();dialog.showModal();
  }
  function attachSettingsEntries() {
    if (!pilots.length) return;
    const sections=new Set([...document.querySelectorAll('.fc-camera-dialog,.fc-more-actions'),
      ...Array.from(document.querySelectorAll('.fc-start-overlay .fc-pages'), pages=>pages.firstElementChild)]);
    sections.forEach(section=>{
      if (!section || section.querySelector('[data-control-settings]')) return;
      const entry=make('button',adapter.radio?'搖桿與檔位設定':'按鍵與檔位設定','fc-control-entry fc-ui-region');entry.type='button';entry.dataset.controlSettings='';
      entry.onclick=event=>{event.stopPropagation();section.closest('[popover]')?.hidePopover();showSettings(entry);};(section.querySelector('.fc-settings-related')||section).append(entry);
    });
    document.querySelectorAll('.fc-start-overlay .fc-pages').forEach((pages,i)=>{
      const guide=pages.children[1];if(!guide || guide.hasAttribute('data-control-reference'))return;
      guide.dataset.controlReference='';
      const backup=/Space|Ctrl/.test(guide.textContent),zoom=/定中視野/.test(guide.textContent);
      const host=make('div');
      {
        // Keep engine-owned tables in the DOM; some modes rebuild them on reset.
        const touch=Array.from(guide.querySelectorAll('.touch-only'));
        const legacy=make('div');legacy.hidden=true;legacy.append(...guide.childNodes);guide.append(legacy,...touch,host);
      }
      mountReference(host,Math.min(i,pilots.length-1),{system:true,backup,zoom});
    });
  }
  function mount() {
    if (!adapter || dialog) return;
    dialog=make('dialog',null,'fc-control-dialog fc-controls-window fc-ui-region');dialog.setAttribute('aria-labelledby','fc-controls-title');
    const restoreFocus=()=>{
      learning=null;release();
      if(returnTarget?.isConnected && returnTarget.getClientRects().length)returnTarget.focus({preventScroll:true});
      else document.activeElement?.blur();
    };
    const closeSettings=()=>{dialog.close();restoreFocus();};
    const header=make('header'),heading=make('div',null,'fc-controls-heading'),title=make('h2','操控設定');title.id='fc-controls-title';const close=make('button','關閉 ×','fc-controls-close');close.type='button';close.setAttribute('aria-label','關閉操控設定');close.onclick=closeSettings;heading.append(make('p','FLIGHT CLUB / CONTROLS','fc-controls-kicker'),title,make('p','為每一次操作，找到適合你的手感。','fc-controls-subtitle'));header.append(heading,close);dialog.append(header);
    const layout=make('div',null,'fc-control-layout'),sidebar=make('aside',null,'fc-controls-sidebar'),context=make('div',null,'fc-controls-context');
    context.append(make('p','目前設定','fc-controls-context-label'),make('strong',adapter.radio?'搖桿控制':'鍵盤操控'));
    if(pilots.length>1) {
      const label=make('label','設定玩家','fc-control-picker');
      const picker=make('select');picker.setAttribute('aria-label','選擇操作玩家');pilots.forEach((p,i)=>{const option=make('option',`P${i+1} · ${p.label}`);option.value=i;picker.append(option);});picker.onchange=()=>{selected=Number(picker.value);learning=null;status.textContent='';renderEditor();};label.append(picker);context.append(label);
    }
    const nav=make('nav',null,'fc-controls-navigation');nav.setAttribute('role','tablist');nav.setAttribute('aria-label','操控設定分類');nav.addEventListener('keydown',event=>{const tabs=Array.from(nav.querySelectorAll('[role=tab]'));let index=tabs.indexOf(document.activeElement);if(index<0)return;if(['ArrowRight','ArrowDown'].includes(event.key))index=(index+1)%tabs.length;else if(['ArrowLeft','ArrowUp'].includes(event.key))index=(index+tabs.length-1)%tabs.length;else if(event.key==='Home')index=0;else if(event.key==='End')index=tabs.length-1;else return;event.preventDefault();tabs[index].click();});sidebar.append(context,nav,make('p','設定只影響目前選擇的玩家。變更會立即套用並保存在此瀏覽器。','fc-controls-local-note'));
    const body=make('div',null,'fc-control-body');body.append(make('div',null,'fc-control-editor'));layout.append(sidebar,body);dialog.append(layout);
    const explanation=make('details',null,'fc-settings-details');explanation.append(make('summary',adapter.radio ? '了解檔位與最大動力' : '了解鍵盤油門與速度'));
    const cards=make('div',null,'fc-gear-explainer');gears.forEach((gear,i)=>{const card=make('div');card.append(make('strong',gear.label),make('span',gear.hint));cards.append(card);});explanation.append(cards,make('p',adapter.radio ? '按一下換檔：低速 33.3% → 中速 66.6% → 高速 100%。微撥追加少量動力，撥越多追加越多，推到底才用滿該檔操作量。回中定高、逐步煞車；換檔限制馬達最大動力，保留慣性，降檔後仍需煞車。' : '鍵盤飛行油門：低速 33.3% → 中速 66.6% → 高速 100%，與目標速度成正比。前進／後退最高 3 m/s，上升最高 3.5 m/s；切換時保留慣性，降檔與反向時仍需煞車。馬達推力與飛行油門分開計算。觸控搖桿則依撥動幅度追加動力，檔位限制馬達動力上限。'));gearExplanation=explanation;
    status=make('p',null,'fc-control-status');status.setAttribute('role','status');dialog.append(status);
    const footer=make('footer',null,'fc-controls-footer'),resetSlot=make('div',null,'fc-controls-reset-slot'),saveHint=make('span','變更自動儲存','fc-controls-save-hint'),done=make('button','完成設定','fc-controls-done');done.type='button';done.onclick=closeSettings;footer.append(resetSlot,saveHint,done);dialog.append(footer);document.body.append(dialog);
    dialog.addEventListener('close',restoreFocus);
    dialog.addEventListener('click',event=>{
      if(event.target!==dialog)return;
      const bounds=dialog.getBoundingClientRect();
      if(event.clientX<bounds.left||event.clientX>bounds.right||event.clientY<bounds.top||event.clientY>bounds.bottom)closeSettings();
    });
    attachSettingsEntries();
  }
  function register(config) {
    gears = config.radio ? radioGears : keyboardGears;
    adapter=config;pilots=config.players();
    storageKey=`droneSoccer.controls.${config.radio?'v3.radio':'v2.keyboard'}.${pilots.length}`;
    preferences=pilots.map((_,i)=>defaults(i));
    try {
      let saved=JSON.parse(localStorage.getItem(storageKey));
      if(!saved && config.radio){
        const legacy=JSON.parse(localStorage.getItem(storageKey.replace('.v3.','.v2.')));
        if(Array.isArray(legacy)&&legacy.length===preferences.length)legacy.forEach((row,i)=>{if(Number.isInteger(row?.initial)&&row.initial>=0&&row.initial<=2)preferences[i].initial=row.initial;});
      }
      if(!saved && !config.radio) {
        saved=JSON.parse(localStorage.getItem(storageKey.replace('.v2.','.v1.')));
        if(saved && !validate(saved)) {
          const migrated=copy(saved);
          // Only migrate legacy defaults; retain custom bindings and pad choices.
          if(migrated[0]?.keys.gear==='f1') {
            migrated[0].keys.gear='q';
            migrated.forEach((row,i)=>{for(const [action,key] of Object.entries(row.keys))if(action!=='gear'&&key==='q'&&engineKeys(i)[action]==='q')row.keys[action]='f1';});
            if(!validate(migrated))saved=migrated;
          }
        }
      }
      if(saved && !validate(saved))preferences=saved;
    } catch (_) { /* Use defaults if storage is corrupt/unavailable. */ }
    pilots.forEach((p,i)=>{p.actor.controlGear=preferences[i].initial;});
    window.DroneFlightSafety?.mount(config);
    window.DroneControllerActions?.register(config);
    if(document.readyState!=='loading')mount();
  }
  window.DroneControlSettings=Object.freeze({register,get gears(){return gears;},get radio(){return !!adapter?.radio;},gearFor,cycle,pollPad,release,mountReference,show:showSettings,
    get open(){return !!dialog?.open;}, get pilots(){return pilots.slice();}, get preferences(){return copy(preferences);}, save, refreshReferences, cancelLearning:()=>{learning=null;}});
  window.addEventListener('keydown',keyboard,true);window.addEventListener('keyup',keyboard,true);
  window.addEventListener('blur',release);document.addEventListener('visibilitychange',()=>{if(document.hidden)release();});
  document.addEventListener('DOMContentLoaded',mount);
  window.addEventListener('load',attachSettingsEntries,{once:true});
})();
