/* Shared presentation layer. Existing game inputs, event handlers and IDs stay intact. */
(() => {
  'use strict';
  const presentationSource = document.currentScript?.src;
  const controllerOnly = location.pathname.includes('/radio/');
  if (controllerOnly) {
    document.documentElement.dataset.fcController = 'gamepad';
    const controllerStyle = document.createElement('style');
    controllerStyle.dataset.fcControllerStyle = '';
    controllerStyle.textContent = 'html[data-fc-controller="gamepad"] .kb-only,html[data-fc-controller="gamepad"] .touch-only{display:none!important}';
    document.head.append(controllerStyle);
  }
  if (presentationSource && !document.querySelector('[data-fc-panels-style]')) {
    const style = document.createElement('link');
    style.rel = 'stylesheet'; style.href = new URL('panels.css', presentationSource).href; style.dataset.fcPanelsStyle = '';
    document.head.append(style);
  }
  if (presentationSource && !document.querySelector('[data-fc-settings-bridge]')) {
    const bridge = document.createElement('script');
    bridge.src = new URL('settings-bridge.js', presentationSource).href; bridge.dataset.fcSettingsBridge = '';
    document.head.append(bridge);
  }
  const $ = selector => document.querySelector(selector);
  const create = (tag, className, text) => {
    const element = document.createElement(tag);
    element.className = className;
    if (text) element.textContent = text;
    return element;
  };
  let sequence = 0;
  let restartDialog = null;
  let restartAction = null;
  let restartAudio = null;
  let restartReturnTarget = null;
  function dismissRestart(restoreFocus = false) {
    if (!restartDialog?.open) return;
    restartAction = null;
    restartDialog.close();
    if (restartAudio) restartAudio.resume().catch(() => {});
    restartAudio = null;
    if (restoreFocus) {
      const target = restartReturnTarget?.checkVisibility() ? restartReturnTarget : $('.fc-action-more');
      target?.focus({ preventScroll: true });
    }
  }
  function confirmRestart(action, audioContext) {
    if (restartDialog?.open) return;
    if (!restartDialog) {
      restartDialog = create('dialog', 'fc-restart-dialog fc-ui-region');
      restartDialog.setAttribute('aria-labelledby', 'fc-restart-title');
      restartDialog.setAttribute('aria-describedby', 'fc-restart-description');
      const content = create('div', 'fc-restart-content');
      const symbol = create('div', 'fc-restart-symbol');
      symbol.append(icon('reset'));
      const title = create('h2', '', '要重新開始這場比賽嗎？');
      title.id = 'fc-restart-title';
      const description = create('p', '', '目前的比分與勝局將全部清除，從第 1 局重新開始。');
      description.id = 'fc-restart-description';
      content.append(symbol, create('span', 'fc-restart-status', '比賽已暫停'), title, description);
      const actions = create('footer', 'fc-restart-actions');
      const cancel = create('button', 'fc-restart-cancel', '返回比賽');
      cancel.type = 'button';
      cancel.autofocus = true;
      cancel.addEventListener('click', () => dismissRestart(true));
      const confirm = create('button', 'fc-restart-confirm', '確認重賽');
      confirm.type = 'button';
      confirm.addEventListener('click', () => {
        const run = restartAction;
        dismissRestart();
        run?.();
        document.activeElement?.blur();
      });
      actions.append(cancel, confirm);
      restartDialog.append(content, actions);
      restartDialog.addEventListener('cancel', event => { event.preventDefault(); dismissRestart(true); });
      restartDialog.addEventListener('click', event => {
        if (isDialogBackdrop(event, restartDialog)) dismissRestart(true);
      });
      document.body.append(restartDialog);
    }
    restartReturnTarget = document.activeElement;
    restartAction = action;
    restartAudio = audioContext?.state === 'running' ? audioContext : null;
    if (restartAudio) restartAudio.suspend().catch(() => {});
    restartDialog.showModal();
  }
  function showGroundFailure(titleText, reason, retry) {
    dismissRestart();
    document.querySelector('.fc-ground-result')?.remove();
    const dialog = create('dialog', 'fc-restart-dialog fc-ui-region fc-ground-result');
    dialog.setAttribute('aria-labelledby', 'fc-ground-result-title');
    const content = create('div', 'fc-restart-content');
    const title = create('h2', '', titleText);
    title.id = 'fc-ground-result-title';
    const actions = create('footer', 'fc-restart-actions');
    const restart = create('button', 'fc-restart-confirm', '重新比賽');
    restart.addEventListener('click', () => { dialog.close(); dialog.remove(); retry(); document.activeElement?.blur(); });
    const menu = create('a', 'fc-restart-cancel', '回主選單');
    menu.href = '../../index.html'; menu.target = '_top';
    actions.append(menu, restart);
    content.append(create('span', 'fc-restart-status', '比賽結束'), title, create('p', '', reason), actions);
    dialog.append(content);
    dialog.addEventListener('cancel', event => event.preventDefault());
    document.body.append(dialog);
    dialog.showModal();
    restart.focus();
  }
  window.DroneMatchActions = Object.freeze({ confirmRestart, dismissRestart, showGroundFailure, get paused() { return !!restartDialog?.open; } });
  function isDialogBackdrop(event, dialog) {
    if (event.target !== dialog) return false;
    const bounds = dialog.getBoundingClientRect();
    return event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom;
  }
  // This interface always uses dark colors, independent of OS and legacy preferences.
  document.documentElement.dataset.fcTheme = 'dark';
  if (window.parent !== window) document.documentElement.dataset.fcHosted = '';
  try { localStorage.removeItem('flight-club-color-theme-v1'); } catch (_) { /* Storage is optional. */ }

  function enhanceAppearance() {
    document.querySelectorAll('.fc-card-heading').forEach(header => {
      const copy = create('div', 'fc-heading-copy');
      copy.append(...header.childNodes);
      header.append(copy);
    });
    // Engine-owned selections use inline colors. Mirror their state without changing handlers.
    document.querySelectorAll('.role-btn,.difficulty-option,.attack-difficulty-option,.red-ai-difficulty-option,.break-option').forEach(option => {
      const sync = () => {
        const selected = option.matches('.role-btn') ? option.style.boxShadow !== 'none' && !!option.style.boxShadow : !!option.style.background;
        option.setAttribute('aria-pressed', String(selected));
      };
      new MutationObserver(sync).observe(option, { attributes: true, attributeFilter: ['style'] });
      sync();
    });
  }

  function cameraSetting() {
    const row = create('label', 'fc-camera-setting fc-ui-region');
    const copy = create('span', 'fc-camera-setting-copy');
    const title = create('strong', '', '第一人稱移動晃動');
    const hint = create('span', '', '開啟時，視線隨機身前傾、後仰；關閉時保持水平。');
    hint.id = `fc-camera-hint-${++sequence}`;
    copy.append(title, hint);
    const input = create('input', 'fc-camera-toggle');
    input.type = 'checkbox';
    input.dataset.fcCockpitPitch = '';
    input.setAttribute('aria-label', '第一人稱移動晃動');
    input.setAttribute('aria-describedby', hint.id);
    input.checked = window.DroneCameraSettings.followPitch;
    input.addEventListener('change', () => window.DroneCameraSettings.setFollowPitch(input.checked));
    // Practice starts when its overlay is clicked; changing a preference must not start it.
    row.addEventListener('click', event => event.stopPropagation());
    row.append(copy, input);
    return row;
  }

  function enhanceCameraSettings() {
    const dialog = create('dialog', 'fc-guide-dialog fc-camera-dialog fc-settings-window fc-ui-region');
    dialog.setAttribute('aria-labelledby', 'fc-camera-settings-title');
    const header = create('header', 'fc-settings-header');
    const heading = create('div', '');
    heading.append(create('span', 'fc-settings-kicker', 'FLIGHT CLUB / SETTINGS'));
    const title = create('h2', '', '畫面與視角'); title.id = 'fc-camera-settings-title';
    heading.append(title, create('p', '', '調整流暢度與舒適度，找到最適合你的飛行畫面。'));
    const close = create('button', 'fc-settings-close', ''); close.type = 'button'; close.setAttribute('aria-label', '關閉畫面設定'); close.append(icon('close'));
    close.onclick = () => dialog.close(); header.append(heading, close);
    const layout = create('div', 'fc-settings-layout'), nav = create('nav', 'fc-settings-nav'), body = create('div', 'fc-settings-body');
    nav.setAttribute('role', 'tablist'); nav.setAttribute('aria-label', '畫面設定分類'); nav.setAttribute('aria-orientation', 'vertical');
    const narrowSettings = matchMedia('(max-width:650px) and (min-height:481px)');
    const orientSettings = () => nav.setAttribute('aria-orientation', narrowSettings.matches ? 'horizontal' : 'vertical');
    narrowSettings.addEventListener('change', orientSettings); orientSettings();
    const sections = [
      ['display', '流暢度', 'DISPLAY', '畫面流暢度', '遊戲會依實際效能調整渲染品質，讓操控維持順暢。'],
      ['comfort', '鏡頭舒適度', 'CAMERA', '鏡頭舒適度', '依自己的習慣選擇視線感受，不影響無人機的飛行能力。'],
      ['status', '運行狀態', 'LIVE STATUS', '目前遊戲的運行狀態', '以下是這個遊戲畫面的即時觀測值，不是螢幕硬體規格。']
    ];
    const pages = [], tabs = [];
    const selectPage = index => {
      tabs.forEach((tab,i)=>{tab.setAttribute('aria-selected',String(i===index));tab.tabIndex=i===index?0:-1;});
      pages.forEach((page,i)=>page.hidden=i!==index);
      body.scrollTop = 0;
    };
    sections.forEach(([id,label,english,name,intro],index)=>{
      const tab=create('button','fc-settings-nav-item');tab.type='button';tab.id='fc-settings-tab-'+id;tab.setAttribute('role','tab');tab.setAttribute('aria-controls','fc-settings-page-'+id);
      tab.append(icon(id==='display'?'settings':id==='comfort'?'camera':'mode'),create('span','',label),create('small','',english));
      const page=create('section','fc-settings-page');page.id='fc-settings-page-'+id;page.setAttribute('role','tabpanel');page.setAttribute('aria-labelledby',tab.id);page.tabIndex=0;
      page.append(create('p','fc-settings-section-kicker',english),create('h3','',name),create('p','fc-settings-description',intro));
      tab.addEventListener('click',()=>selectPage(index));nav.append(tab);body.append(page);tabs.push(tab);pages.push(page);
    });
    const frameRow = create('label', 'fc-camera-setting fc-frame-setting');
    const frameCopy = create('span', 'fc-camera-setting-copy');
    const frameHint = create('span', '', '以自動或常用更新率開始。較高目標需要足夠效能；選擇不會限制瀏覽器的顯示更新。');
    frameHint.id = 'fc-frame-target-hint'; frameCopy.append(create('strong', '', '流暢度目標'), frameHint);
    const frameSelect = create('select', 'fc-frame-target'); frameSelect.setAttribute('aria-label', '流暢度目標'); frameSelect.setAttribute('aria-describedby',frameHint.id);
    const targets = ['auto','60','90','120','144','165','180','240','360','480'];
    if (!targets.includes(window.DroneFrameTiming.preference)) targets.push(window.DroneFrameTiming.preference);
    targets.forEach(value=>{const option=create('option','',value==='auto'?'自動（建議）':value+' FPS');option.value=value;frameSelect.append(option);});
    frameSelect.value=window.DroneFrameTiming.preference;frameRow.append(frameCopy,frameSelect);
    const automatic=create('div','fc-settings-information');automatic.append(icon('check'),create('div',''));
    automatic.lastElementChild.append(create('strong','','自動維持畫面品質'),create('p','','遊戲會依目前渲染狀態調整像素比例與陰影品質，無需手動切換畫質段位。'));
    pages[0].append(frameRow,automatic);
    const comfort=cameraSetting(); pages[1].append(comfort);
    const comfortTip=create('div','fc-settings-information');comfortTip.append(icon('cockpit'),create('div',''));
    comfortTip.lastElementChild.append(create('strong','','只有第一人稱使用這個選項'),create('p','','保持水平可減少視線晃動。其他跟隨或固定視角仍沿用各自的鏡頭方式；正式連線賽的玩家視角由主持人設定。'));
    pages[1].append(comfortTip);
    const metrics=create('div','fc-runtime-metrics');
    const values={};[['fps','目前幀率','FPS'],['pixelRatio','渲染比例','DPR'],['quality','動態品質','LEVEL']].forEach(([id,label,unit])=>{const metric=create('div','fc-runtime-metric');const value=create('strong','','—');values[id]=value;metric.append(create('span','',label),value,create('small','',unit));metrics.append(metric);});
    const runtimeStatus=create('p','fc-runtime-status','等待遊戲畫面初始化…');runtimeStatus.setAttribute('role','status');
    const detail=create('details','fc-runtime-details');detail.append(create('summary','','查看渲染明細'));const table=create('dl','fc-runtime-data');const runtimeValues={};
    [['target','目前目標'],['calls','繪製批次'],['triangles','場景三角形'],['shadows','動態陰影光源']].forEach(([id,label])=>{const row=create('div',''),value=create('dd','','—');row.append(create('dt','',label),value);table.append(row);runtimeValues[id]=value;});detail.append(table);
    pages[2].append(metrics,runtimeStatus,detail);
    const footer=create('footer','fc-settings-footer'), displayStatus=create('p','fc-camera-note fc-settings-save-status','變更立即套用；在這個瀏覽器保留你的選擇。');displayStatus.setAttribute('role','status');
    const resetDisplay=create('button','fc-settings-reset','恢復建議畫面設定');resetDisplay.type='button';
    const done=create('button','fc-settings-done','完成');done.type='button';done.onclick=()=>dialog.close();
    const related=create('div','fc-settings-related');const footerActions=create('div','fc-settings-footer-actions');footerActions.append(resetDisplay,done);footer.append(displayStatus,related,footerActions);
    layout.append(nav,body);dialog.append(header,layout,footer);selectPage(0);
    const savedStatus = message => {
      let stored=false;
      try { stored=localStorage.getItem('droneSoccer.render.targetFPS')===window.DroneFrameTiming.preference && localStorage.getItem('droneSoccer.camera.followPitch')===String(window.DroneCameraSettings.followPitch); } catch (_) {}
      displayStatus.textContent=message || (stored?'已套用，並儲存在這個瀏覽器。':'已套用到目前遊戲；關閉頁面前請保留此分頁。');
    };
    const updateRuntime=()=>{
      if(!dialog.open)return;
      const state=window.__perf?.();
      if(!state){runtimeStatus.textContent='遊戲開始後會顯示即時渲染資料。';return;}
      values.fps.textContent=state.fps>0?Math.round(state.fps):'—';values.pixelRatio.textContent=Number.isFinite(state.pixelRatio)?state.pixelRatio.toFixed(2):'—';
      values.quality.textContent=Number.isFinite(state.qualityStep)?state.qualityStep+' / '+state.steps:'—';
      runtimeValues.target.textContent=state.targetFPS+' FPS';runtimeValues.calls.textContent=Number(state.calls).toLocaleString();runtimeValues.triangles.textContent=Number(state.triangles).toLocaleString();runtimeValues.shadows.textContent=String(state.dynamicShadowLights??'—');
      runtimeStatus.textContent=state.fps>0?'觀測值會隨場景與電腦負載變化。':'畫面已就緒，正在觀測幀率…';
    };
    let returnTarget, runtimeTimer;
    const openDisplay=target=>{returnTarget=target||document.activeElement;dialog.showModal();updateRuntime();clearInterval(runtimeTimer);runtimeTimer=setInterval(updateRuntime,600);};
    frameSelect.addEventListener('change',()=>{window.DroneFrameTiming.setTarget(frameSelect.value);try{localStorage.setItem('droneSoccer.camera.followPitch',String(window.DroneCameraSettings.followPitch));}catch(_){}savedStatus();updateRuntime();});
    comfort.querySelector('input').addEventListener('change',()=>{try{localStorage.setItem('droneSoccer.render.targetFPS',window.DroneFrameTiming.preference);}catch(_){}savedStatus();});
    resetDisplay.onclick=()=>{window.DroneCameraSettings.setFollowPitch(false);window.DroneFrameTiming.setTarget('auto');savedStatus('已恢復建議值：自動流暢度，視線保持水平。');updateRuntime();};
    window.addEventListener('drone-frame-settings-change',()=>{frameSelect.value=window.DroneFrameTiming.preference;updateRuntime();});
    window.addEventListener('drone-camera-settings-change',()=>{document.querySelectorAll('[data-fc-cockpit-pitch]').forEach(input=>input.checked=window.DroneCameraSettings.followPitch);});
    dialog.addEventListener('close',()=>{clearInterval(runtimeTimer);if(returnTarget?.isConnected&&returnTarget.getClientRects().length)returnTarget.focus({preventScroll:true});});
    dialog.addEventListener('click',event=>{if(isDialogBackdrop(event,dialog))dialog.close();});document.body.append(dialog);
    const toolbars=new Set(Array.from(document.querySelectorAll('[id$="btnReset"],#btnResetAll'),button=>button.parentElement));
    toolbars.forEach(toolbar=>{const button=create('button','fc-hud-button','畫面設定');button.type='button';button.dataset.fcIcon='settings';if(toolbar.closest('#topBar'))button.prepend(icon('settings'));button.addEventListener('click',()=>openDisplay(button));toolbar.append(button);});
    document.querySelectorAll('.fc-start-overlay .fc-pages').forEach(pages=>{const button=create('button','fc-control-entry fc-ui-region','畫面與流暢度設定');button.type='button';button.onclick=event=>{event.stopPropagation();openDisplay(button);};pages.firstElementChild?.append(button);});
  }

  // Inline SVGs keep controls consistent even when the icon font is unavailable.
  const iconPaths = {
    takeoff: 'M4 18h16M12 14V3m-4 4 4-4 4 4M5 13v2m14-2v2',
    land: 'M4 20h16M12 3v11m-4-4 4 4 4-4M5 15v2m14-2v2',
    menu: 'm12 3 9 5-9 5-9-5 9-5Zm-9 9 9 5 9-5M3 16l9 5 9-5',
    sound: 'm11 5-6 4H2v6h3l6 4V5Zm4 3a6 6 0 0 1 0 8m3-11a10 10 0 0 1 0 14',
    muted: 'm11 5-6 4H2v6h3l6 4V5Zm5 4 5 6m0-6-5 6',
    camera: 'M4 7h4l2-3h4l2 3h4v13H4V7Zm12 6a4 4 0 1 1-8 0 4 4 0 0 1 8 0',
    cockpit: 'M3 18V7l4-3h10l4 3v11M3 13l5-3h8l5 3M12 5v5m-4 8h8m-4-3v6',
    chase: 'm4 5 8-3 8 3v9M8 13l4-2 4 2v5l-4 2-4-2v-5Zm-4 8 4-3m12 3-4-3M12 11V7',
    free: 'M8 4a9 9 0 1 1-5 8M3 4h5v5M8 12h8m-4-4v8',
    center: 'M4 4h16v11H4V4Zm8 11v6m-4-2 4 2 4-2M8 8h8m-4-2v5',
    left: 'M4 4h16v11H4V4Zm1 11v6m-3-3 3 3 3-3M7 8h6m-3-2L7 8l3 2',
    right: 'M4 4h16v11H4V4Zm15 11v6m-3-3 3 3 3-3M11 8h6m-3-2 3 2-3 2',
    check: 'm5 12 4 4L19 6',
    chevron: 'm8 10 4 4 4-4',
    reset: 'M20 7v5h-5M20 12a8 8 0 1 0-2 6M20 7l-3-2',
    guide: 'M3 5h18v14H3V5Zm4 4h1m4 0h1m4 0h1M7 12h1m4 0h1m4 0h1M8 16h8',
    close: 'm6 6 12 12M6 18 18 6',
    pad: 'M8 7h8c3 0 5 3 5 8s-2 5-4 2l-1-1H8l-1 1c-2 3-4 3-4-2s2-8 5-8Zm-2 4h4m-2-2v4m7-2h.01m3 2h.01',
    mode: 'M5 4v16M12 4v16M19 4v16M2 8h6m1 8h6m1-6h6',
    settings: 'M4 6h16M4 12h16M4 18h16M8 3v6m8 0v6M10 15v6',
    more: 'M5 11h.01M12 11h.01M19 11h.01M4 11a1 1 0 1 0 2 0 1 1 0 0 0-2 0m7 0a1 1 0 1 0 2 0 1 1 0 0 0-2 0m7 0a1 1 0 1 0 2 0 1 1 0 0 0-2 0'
  };
  function icon(name) {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('fill', 'none');
    svg.setAttribute('stroke', 'currentColor');
    svg.setAttribute('stroke-width', '1.6');
    svg.setAttribute('stroke-linecap', 'round');
    svg.setAttribute('stroke-linejoin', 'round');
    svg.setAttribute('aria-hidden', 'true');
    svg.classList.add('fc-icon');
    const path = document.createElementNS(svg.namespaceURI, 'path');
    path.setAttribute('d', iconPaths[name]);
    svg.append(path);
    return svg;
  }

  function enhanceToolbar(controls) {
    controls.classList.add('fc-flight-toolbar', 'fc-ui-region');
    controls.setAttribute('role', 'group');
    controls.setAttribute('aria-label', '飛行控制');
    const flight = create('div', 'fc-flight-actions');
    const system = create('div', 'fc-system-actions');
    const secondary = [];
    flight.setAttribute('role', 'group');
    flight.setAttribute('aria-label', '起飛與降落');
    system.setAttribute('role', 'group');
    system.setAttribute('aria-label', '視角與設定');
    const definitions = [
      ['btnTakeoffAll', 'takeoff', '全體起飛'], ['btnLandAll', 'land', '全體降落'],
      ['btnResetAll', 'reset', '重新比賽'], ['btnMuteAll', 'sound', '切換聲音'],
      ['btnRedAIDifficulty', 'mode', 'AI 難度'],
      ['btnTakeoff', 'takeoff', '起飛'], ['btnLand', 'land', '降落'],
      ['btnMenu', 'menu', '主選單'], ['btnLevel', 'menu', '主選單'], ['btnSound', 'sound', '切換聲音'],
      ['btnCam', 'camera', '切換鏡頭視角'], ['btnReset', 'reset', '重設無人機'],
      ['btnPad', 'pad', '手把設定'], ['btnFlightMode', 'mode', '切換飛行模式'],
      ['btnDifficulty', 'menu', '切換難度'], ['btnAtkDifficulty', 'menu', '前鋒難度'],
      ['btnRole', 'mode', '切換角色']
    ];
    Array.from(controls.children).forEach(button => {
      if (!button.matches('button')) return;
      const definition = definitions.find(([suffix]) => button.id.endsWith(suffix)) ||
        (button.matches('.js-btn-pad') ? ['', 'pad', '手把設定'] :
          button.matches('.js-btn-mode') ? ['', 'mode', '切換飛行模式'] : null);
      const name = definition?.[1] || button.dataset.fcIcon;
      if (!name) return;
      const label = definition?.[2] || button.textContent.trim();
      button.classList.add('fc-action', `fc-action-${name}`);
      button.type = 'button';
      button.title ||= label;
      button.setAttribute('aria-label', label);
      const shortcut = button.querySelector('.kb-only');
      if (shortcut) {
        const key = shortcut.textContent.replace(/[()\s]/g, '');
        const badge = create('kbd', 'fc-shortcut kb-only', key);
        shortcut.remove();
        // Keep shortcuts beside the label so menu rows can align them at the right edge.
        button.append(badge);
      }
      // Keep original label IDs and <i> nodes; the game updates them at runtime.
      for (const child of button.children) {
        if (child.matches('span:not(.kb-only)') && child.textContent.trim()) child.classList.add('fc-action-label');
      }
      Array.from(button.childNodes).filter(node => node.nodeType === Node.TEXT_NODE && node.textContent.trim()).forEach(node => {
        const label = create('span', 'fc-action-label', node.textContent.trim());
        node.replaceWith(label);
      });
      if (!button.querySelector('.fc-action-label')) button.append(create('span', 'fc-action-label', label));
      if (name === 'reset') button.querySelector('.fc-action-label').textContent = /All$/.test(button.id) ? '重新比賽' : '重設無人機';
      if (/Difficulty$/.test(button.id)) {
        button.prepend(create('span', 'fc-action-context', /RedAI/.test(button.id) ? 'AI 難度' : /Atk/.test(button.id) ? '前鋒難度' : '防守難度'));
      }
      if (!button.querySelector('.fc-icon')) button.prepend(icon(name));
      if (name === 'sound') {
        const originalIcon = button.querySelector('i');
        const syncSound = () => {
          const muted = originalIcon?.classList.contains('fa-volume-xmark');
          button.setAttribute('aria-pressed', String(!muted));
          button.setAttribute('aria-label', muted ? '開啟聲音' : '靜音');
          button.title = controllerOnly ? (muted ? '開啟聲音' : '靜音') : (muted ? '開啟聲音（M）' : '靜音（M）');
          button.querySelector('.fc-icon path').setAttribute('d', iconPaths[muted ? 'muted' : 'sound']);
          button.querySelector('.fc-action-label').textContent = muted ? '開啟聲音' : '靜音';
        };
        if (originalIcon) new MutationObserver(syncSound).observe(originalIcon, { attributes: true, attributeFilter: ['class'] });
        syncSound();
      }
      if (name === 'takeoff' || name === 'land') flight.append(button);
      else if (name === 'camera' || name === 'guide') system.append(button);
      else secondary.push(button);
      // Mouse/touch commands return focus to flight. Keyboard users retain their place.
      button.addEventListener('click', event => {
        if (event.detail > 0 && !button.closest(':popover-open') && !document.querySelector('dialog[open]')) button.blur();
      });
    });
    const more = create('button', 'fc-action fc-action-more');
    more.type = 'button';
    more.append(icon('more'), create('span', 'fc-action-label', '更多'));
    more.setAttribute('aria-label', '更多操作');
    more.setAttribute('aria-expanded', 'false');
    more.setAttribute('aria-haspopup', 'dialog');
    const panel = create('div', 'fc-toolbar-popover fc-ui-region');
    panel.id = `fc-toolbar-more-${++sequence}`;
    panel.setAttribute('popover', 'auto');
    panel.setAttribute('role', 'dialog');
    const heading = create('header', 'fc-more-heading');
    const title = create('strong', '', '更多操作');
    title.id = `${panel.id}-title`;
    panel.setAttribute('aria-labelledby', title.id);
    const close = create('button', 'fc-more-close');
    close.type = 'button';
    close.setAttribute('aria-label', '關閉更多操作');
    close.autofocus = true;
    close.append(icon('close'));
    close.addEventListener('click', () => { panel.hidePopover(); more.focus({preventScroll:true}); });
    heading.append(title, close);
    const actions = create('div', 'fc-more-actions');
    if (controllerOnly) {
      // X opens this menu during flight. Mirror HUD-only actions here so a
      // controller can reach them while the original buttons retain handlers.
      system.querySelectorAll('.fc-action-camera,.fc-action-guide').forEach(original => {
        const name = original.classList.contains('fc-action-camera') ? 'camera' : 'guide';
        const proxy = create('button', `fc-action fc-action-${name}`);
        proxy.type = 'button';
        proxy.dataset.fcControllerProxy = name;
        const label = name === 'camera' ? '鏡頭視角' : '操作指南';
        proxy.setAttribute('aria-label', label);
        proxy.append(icon(name), create('span', 'fc-action-label', label));
        proxy.addEventListener('click', () => {
          panel.hidePopover();
          original.click();
        });
        actions.append(proxy);
      });
    }
    secondary.sort((a, b) => Number(a.matches('.fc-action-reset')) - Number(b.matches('.fc-action-reset')));
    actions.append(...secondary);
    panel.append(heading, actions);
    document.body.append(panel);
    more.setAttribute('popovertarget', panel.id);
    more.setAttribute('aria-controls', panel.id);
    const position = () => {
      if (!panel.matches(':popover-open')) return;
      const anchor = more.getBoundingClientRect();
      panel.style.left = `${Math.max(8, Math.min(innerWidth - panel.offsetWidth - 8, anchor.right - panel.offsetWidth))}px`;
      panel.style.top = `${Math.max(8, Math.min(innerHeight - panel.offsetHeight - 8, anchor.bottom + 8))}px`;
    };
    panel.addEventListener('beforetoggle', event => {
      more.setAttribute('aria-expanded', String(event.newState === 'open'));
      if (event.newState === 'open') requestAnimationFrame(position);
    });
    window.addEventListener('resize', position);
    panel.addEventListener('click', event => {
      const button = event.target.closest('button');
      if (button && (button.matches('.fc-action-reset,.fc-action-pad,.fc-action-settings,[id$="btnMenu"],[id$="btnLevel"]') || document.querySelector('dialog[open]'))) {
        panel.hidePopover();
        if (event.detail > 0 && !document.querySelector('dialog[open]')) button.blur();
      }
    });
    // Escape dismisses the panel before the game's own Escape shortcut sees it.
    window.addEventListener('keydown', event => {
      if (!panel.matches(':popover-open') || event.key !== 'Escape' || document.querySelector('dialog[open]')) return;
      event.preventDefault(); event.stopImmediatePropagation();
      panel.hidePopover(); more.focus({preventScroll:true});
    }, true);
    system.append(more);
    controls.append(flight, system);
  }

  function enhanceCameraSelectors() {
    const api = window.DroneCameraControls;
    document.querySelectorAll('[id$="btnCam"]').forEach(button => {
      if (button.dataset.fcCameraPicker !== undefined || !api.get(button.id)) return;
      button.dataset.fcCameraPicker = '';
      button.type = 'button';
      button.classList.add('fc-camera-trigger', 'fc-ui-region');
      if (!button.querySelector('.fc-icon')) button.prepend(icon('camera'));
      const arrow = icon('chevron');
      arrow.classList.add('fc-camera-chevron');
      button.append(arrow);
      const label = button.querySelector('[id$="camText"]');
      const panel = create('div', 'fc-camera-picker fc-ui-region');
      panel.id = `fc-camera-picker-${++sequence}`;
      panel.setAttribute('popover', 'auto');
      panel.setAttribute('role', 'dialog');
      const heading = create('header', 'fc-camera-picker-heading');
      const copy = create('div', '');
      const title = create('strong', '', '鏡頭視角');
      title.id = `${panel.id}-title`;
      panel.setAttribute('aria-labelledby', title.id);
      const status = create('span', 'fc-camera-current');
      status.setAttribute('aria-live', 'polite');
      copy.append(title, status);
      const close = create('button', 'fc-more-close');
      close.type = 'button';
      close.setAttribute('aria-label', '關閉視角選單');
      close.append(icon('close'));
      heading.append(copy, close);
      panel.append(heading);
      const options = [];
      const dismiss = keyboard => {
        panel.hidePopover();
        if (keyboard) button.focus({preventScroll: true});
        else if (document.activeElement?.closest('.fc-camera-picker') || document.activeElement === button) document.activeElement.blur();
      };
      close.addEventListener('click', event => dismiss(event.detail === 0));
      const descriptions = {
        cockpit: '以機身朝向飛行，適合練習沉浸式操控。',
        chase: '鏡頭跟隨機身，方便掌握位置與飛行方向。',
        free: '滑鼠拖曳旋轉、滾輪縮放；觸控可拖曳或雙指縮放。',
        center: '站在己方中間，從固定位置觀察飛行。',
        left: '面向場地，站在己方左邊觀察飛行。',
        right: '面向場地，站在己方右邊觀察飛行。'
      };
      [['飛行視角', ['cockpit', 'chase', 'free']], ['固定觀察', ['left', 'center', 'right']]].forEach(([name, ids]) => {
        const section = create('section', 'fc-camera-section');
        const sectionTitle = create('h3', '', name);
        sectionTitle.id = `${panel.id}-${ids[0]}`;
        const grid = create('div', 'fc-camera-options');
        grid.setAttribute('role', 'group');
        grid.setAttribute('aria-labelledby', sectionTitle.id);
        ids.forEach(id => {
          const mode = api.modes.find(mode => mode.id === id);
          const option = create('button', 'fc-camera-option');
          option.type = 'button';
          option.dataset.fcCameraMode = id;
          option.title = mode.hint;
          const selected = icon('check');
          selected.classList.add('fc-camera-check');
          option.append(icon(mode.icon), create('span', '', mode.label), selected);
          option.addEventListener('click', event => {
            api.get(button.id).setMode(id);
            sync();
            dismiss(event.detail === 0);
          });
          options.push(option);
          grid.append(option);
        });
        section.append(sectionTitle, grid);
        panel.append(section);
      });
      const hint = create('p', 'fc-camera-description');
      const comfort = cameraSetting();
      comfort.classList.add('fc-camera-comfort');
      panel.append(hint, comfort);
      document.body.append(panel);
      button.setAttribute('aria-haspopup', 'dialog');
      button.setAttribute('aria-controls', panel.id);
      button.setAttribute('aria-expanded', 'false');
      function sync() {
        const mode = api.modes.find(mode => mode.id === api.get(button.id).getMode());
        if (!mode) return;
        if (label && label.textContent !== mode.label) label.textContent = mode.label;
        status.textContent = `目前：${mode.label}`;
        button.title = `選擇鏡頭視角（目前：${mode.label}）`;
        button.setAttribute('aria-label', button.title);
        button.dataset.fcCameraCurrent = mode.id;
        button.querySelector('.fc-icon path').setAttribute('d', iconPaths[mode.icon]);
        options.forEach(option => option.setAttribute('aria-pressed', String(option.dataset.fcCameraMode === mode.id)));
        hint.textContent = descriptions[mode.id];
        comfort.hidden = mode.id !== 'cockpit';
      }
      function position() {
        if (!panel.matches(':popover-open')) return;
        const rect = button.getBoundingClientRect();
        panel.style.left = `${Math.max(8, Math.min(innerWidth - panel.offsetWidth - 8, rect.right - panel.offsetWidth))}px`;
        panel.style.top = `${Math.max(8, Math.min(innerHeight - panel.offsetHeight - 8, rect.bottom + 8))}px`;
      }
      // Capture replaces the old cycle-on-click command. Shortcuts still call the engine.
      button.addEventListener('click', event => {
        event.preventDefault(); event.stopImmediatePropagation();
        if (panel.matches(':popover-open')) dismiss(event.detail === 0);
        else {
          sync();
          panel.showPopover();
          position();
          if (event.detail === 0) options.find(option => option.getAttribute('aria-pressed') === 'true')?.focus();
        }
      }, true);
      panel.addEventListener('beforetoggle', event => {
        button.setAttribute('aria-expanded', String(event.newState === 'open'));
      });
      window.addEventListener('keydown', event => {
        if (!panel.matches(':popover-open') || event.key !== 'Escape') return;
        event.preventDefault(); event.stopImmediatePropagation(); dismiss(true);
      }, true);
      window.addEventListener('resize', position);
      window.addEventListener('drone-camera-change', event => { if (event.detail.id === button.id) { sync(); position(); } });
      // Role selection and resets may refresh the engine-owned label independently.
      if (label) new MutationObserver(sync).observe(label, {childList: true, characterData: true, subtree: true});
      sync();
    });
  }

  function enhanceGuide(panel, controls) {
    const isRadio = controllerOnly || /搖桿/.test(panel.textContent);
    const rows = Array.from(panel.children).filter(node => node.matches('div') && node.firstElementChild?.matches('span'));
    if (!rows.length) return;
    const header = create('header', 'fc-guide-heading');
    const heading = create('div', 'fc-guide-title');
    const titles = create('div', '');
    const headingId = `fc-guide-title-${++sequence}`;
    const title = create('h2', '', '飛行操作指南');
    title.id = headingId;
    titles.append(create('span', 'fc-guide-eyebrow', isRadio ? 'CONTROLLER REFERENCE' : 'KEYBOARD REFERENCE'), title);
    heading.append(icon(isRadio ? 'pad' : 'guide'), titles);
    const close = create('button', 'fc-guide-close');
    close.type = 'button';
    close.setAttribute('aria-label', '收合操作指南');
    close.title = '收合操作指南';
    close.append(icon('close'));
    header.append(heading, close);
    const body = create('div', 'fc-guide-body');
    const sections = new Map();
    rows.forEach(row => {
      const label = row.firstElementChild;
      const text = label.textContent.trim().replace(/[：:]$/, '');
      const sectionName = /鏡頭|視野|視角|重[設置]|更換|靜音|設定|模式|起飛.*降落/.test(text) && !/一鍵/.test(text) ? '視角與系統' : /前進|後退|俯仰|滾|偏航/.test(text) ? '方向控制' : '起降與高度';
      if (!sections.has(sectionName)) {
        const section = create('section', 'fc-guide-section');
        section.append(create('h3', '', sectionName));
        sections.set(sectionName, section);
        body.append(section);
      }
      const keys = create('span', 'fc-guide-keys');
      Array.from(row.childNodes).slice(1).forEach(node => keys.append(node));
      // Replace only keycaps, retaining USB descriptions and each mode's wording.
      keys.querySelectorAll('span').forEach(span => {
        if (!span.className.includes('bg-slate-8')) return;
        const cap = create('kbd', 'fc-keycap', span.textContent.trim());
        if (!cap.textContent && span.querySelector('i')) cap.textContent = 'Esc';
        if (/^[TL]$/.test(cap.textContent)) cap.classList.add(cap.textContent === 'T' ? 'fc-key-takeoff' : 'fc-key-land');
        span.replaceWith(cap);
      });
      row.className = 'fc-guide-row';
      label.className = 'fc-guide-label';
      label.textContent = text;
      row.replaceChildren(label, keys);
      sections.get(sectionName).append(row);
    });
    const reference = create('div', '');
    const pilotIndex = Array.from(document.querySelectorAll('[id$="gameUI"]')).indexOf(controls.closest('[id$="gameUI"]'));
    const backup = /Space|Ctrl/.test(body.textContent);
    if (!isRadio) {
      for (const [name, section] of sections) if (name !== '視角與系統') section.remove();
    }
    body.prepend(reference);
    window.DroneControlSettings?.mountReference(reference, Math.max(0, pilotIndex), {backup, takeoff: !isRadio});
    if (isRadio && reference.childElementCount) {
      // The shared reference owns every controller action. Discard old static
      // descriptions and keyboard shortcuts rather than showing two schemes.
      sections.forEach(section => section.remove());
      panel.classList.remove('kb-only', 'touch-only');
    }
    body.querySelectorAll('.fc-guide-label').forEach(label => {
      if (/定中視野/.test(label.textContent)) label.textContent = '定中視角縮放 / 網後距離';
    });
    panel.classList.add('fc-flight-guide', 'fc-ui-region');
    panel.setAttribute('role', 'region');
    panel.setAttribute('aria-labelledby', headingId);
    panel.replaceChildren(header, body, create('footer', 'fc-guide-footer', isRadio ? '搖桿對應與範圍校正可於「更多 → 手把設定」調整' : '方向鍵長按；換檔與系統鍵按一下。按鍵依目前設定顯示。'));
    const home = document.createComment('Flight guide original position');
    panel.before(home);
    const dialog = create('dialog', 'fc-flight-guide-dialog fc-ui-region');
    dialog.id = `fc-flight-guide-dialog-${++sequence}`;
    dialog.dataset.controllerOwner=String(Math.max(0,pilotIndex));
    dialog.setAttribute('aria-labelledby', headingId);
    document.body.append(dialog);
    const toggle = create('button', 'fc-hud-button', '操作指南');
    toggle.type = 'button';
    toggle.dataset.controllerGuide=String(Math.max(0,pilotIndex));
    toggle.dataset.fcIcon = 'guide';
    panel.id ||= `fc-flight-guide-${++sequence}`;
    toggle.setAttribute('aria-controls', dialog.id);
    controls.append(toggle);
    let expanded = false;
    toggle.setAttribute('aria-haspopup', 'dialog');
    const sync = () => {
      panel.classList.toggle('fc-guide-collapsed', !expanded);
      toggle.setAttribute('aria-expanded', String(expanded));
    };
    const closeGuide = () => {
      if (dialog.open) dialog.close();
      expanded = false;
      sync();
      toggle.focus({ preventScroll: true });
    };
    close.addEventListener('click', closeGuide);
    dialog.addEventListener('close', () => {
      home.after(panel);
      expanded = false;
      sync();
      if (toggle.isConnected && toggle.getClientRects().length) toggle.focus({ preventScroll: true });
    });
    toggle.addEventListener('click', () => {
      expanded = true;
      sync();
      dialog.append(panel);
      dialog.showModal();
    });
    dialog.addEventListener('click', event => { if (isDialogBackdrop(event, dialog)) closeGuide(); });
    window.addEventListener('keydown', event => {
      if (event.key !== 'Escape' || !expanded || dialog.open) return;
      event.preventDefault(); event.stopImmediatePropagation(); closeGuide();
    }, true);
    sync();
  }

  function tabbedCard(card, labels, title = '設定分頁') {
    card.classList.add('fc-card', 'fc-ui-region');
    const tabs = create('div', 'fc-tabs');
    tabs.setAttribute('role', 'tablist');
    tabs.setAttribute('aria-label', title);
    const body = create('div', 'fc-pages');
    const pages = labels.map((label, index) => {
      const id = `fc-tab-${++sequence}`;
      const tab = create('button', 'fc-tab', label);
      tab.type = 'button';
      tab.id = id;
      tab.setAttribute('role', 'tab');
      tab.setAttribute('aria-controls', `${id}-panel`);
      const page = create('section', 'fc-page');
      page.id = `${id}-panel`;
      page.setAttribute('role', 'tabpanel');
      page.setAttribute('aria-labelledby', id);
      page.tabIndex = 0;
      tabs.append(tab);
      body.append(page);
      return { tab, page };
    });
    function select(index, focus = false) {
      pages.forEach((entry, i) => {
        entry.tab.setAttribute('aria-selected', String(i === index));
        entry.tab.tabIndex = i === index ? 0 : -1;
        entry.page.hidden = i !== index;
      });
      if (focus) pages[index].tab.focus();
    }
    pages.forEach(({ tab }, index) => {
      tab.addEventListener('click', event => { event.stopPropagation(); select(index); });
      tab.addEventListener('keydown', event => {
        let next;
        if (event.key === 'ArrowRight') next = (index + 1) % pages.length;
        if (event.key === 'ArrowLeft') next = (index + pages.length - 1) % pages.length;
        if (event.key === 'Home') next = 0;
        if (event.key === 'End') next = pages.length - 1;
        if (next !== undefined) { event.preventDefault(); event.stopPropagation(); select(next, true); }
      });
    });
    select(0);
    return { tabs, body, pages: pages.map(entry => entry.page), select };
  }

  function enhanceStart(overlay) {
    const card = overlay.firstElementChild;
    if (!card || card.classList.contains('fc-card')) return;
    overlay.classList.add('fc-start-overlay');
    const original = Array.from(card.children);
    const heading = original.find(node => node.matches('h1,h2'));
    if (!heading) return;
    const ui = tabbedCard(card, ['01 準備', '02 操控', '03 規則'], '賽前準備');
    const header = create('header', 'fc-card-heading');
    header.append(create('p', 'fc-kicker', 'FLIGHT CLUB / PRE-FLIGHT'), heading);
    const footer = create('footer', 'fc-start-footer');
    const settings = ui.pages[0], guide = ui.pages[1], rules = ui.pages[2];
    settings.append(create('p', 'fc-page-intro', '確認下方設定，準備好就開始飛行。'));
    let subtitleDone = false;
    let hasAction = false;
    for (const node of original) {
      if (node === heading) continue;
      const text = node.textContent.trim();
      if (node.matches('button[id$="btnStart"], #btnLaunch')) {
        node.classList.add('fc-primary-action');
        if (node.id === 'btnLaunch') node.textContent = '開始飛行';
        footer.append(node);
        hasAction = true;
      } else if (node.matches('span') && text.startsWith('PLAYER')) {
        header.prepend(node);
      } else if (node.matches('p') && !subtitleDone && text.length > 45 && !/按任意鍵|選擇|難度|局間休息/.test(text)) {
        node.classList.add('fc-subtitle');
        header.append(node);
        subtitleDone = true;
      } else if (node.matches('.kb-only,.touch-only') || node.querySelector('table') || /鍵盤|飛行控制|操控指南|操控說明/.test(text.slice(0, 18))) {
        guide.append(node);
      } else if (node.querySelector('button,.glass-button,.role-btn,.level-btn,[id^="level"],.difficulty-option,.break-option,[id*="btnDiff"],[id*="btnTeam"],input,select') || node.matches('.level-btn,.difficulty-option') || /選擇|難度|局間休息|按任意鍵|點選上方|點擊/.test(text.slice(0, 28))) {
        settings.append(node);
      } else if (text) {
        rules.append(node);
      }
    }
    // Practice's original click-to-start handler remains on its overlay.
    if (!hasAction && settings.querySelector('#btnTeamRed,#btnTeamBlue')) hasAction = true;
    if (!hasAction && !settings.querySelector('.level-btn, [id^="level"]')) {
      const start = create('button', 'fc-primary-action', '開始飛行 →');
      start.type = 'button';
      footer.append(start);
    }
    if (!guide.children.length) guide.append(create('p', 'fc-page-intro', '此模式使用畫面上的操控按鈕；USB 遙控器請先完成「手把」設定。'));
    if (!rules.children.length) rules.append(create('p', 'fc-page-intro', '依畫面提示完成飛行任務。想換個玩法時，可從上方返回遊戲大廳。'));
    if (!footer.children.length) footer.append(create('span', 'fc-footer-hint', settings.querySelector('.level-btn') ? '選擇難度即可開始挑戰' : '選擇隊伍即可進場'));
    if (footer.querySelector('.fc-primary-action')) footer.prepend(create('span', 'fc-footer-hint', '準備完成，即可進入賽場'));
    card.replaceChildren(header, ui.tabs, ui.body, footer);
  }

  // One pre-flight flow for every legacy keyboard and USB entrypoint.
  function redesignMainMenus() {
    const catalog = {
      practise: ['practice', '自由飛行練習', '飛行練習', '先穩住飛行，再練習攻防。', '前鋒挑戰兩台 AI 防守；防守與一台 AI 隊友合作，抵擋 AI 對手。不限時，依自己的步調熟悉操控。'],
      competition: ['attack', '單人進攻', '自由對戰', '找到空檔，突破防線。', '駕駛前鋒穿越對方球門，挑戰 AI 防守。'],
      defense: ['defense', '單人防守', '自由對戰', '守住球門，掌握節奏。', '觀察來襲路線，調整站位，攔下 AI 前鋒。'],
      competition_ad: ['attack-defense', '攻守對決', '自由對戰', '一攻一守，正面交鋒。', '兩位玩家分別進攻與防守，AI 隊友協助防線。'],
      competition_dd: ['double-defense', '雙人聯防', '自由對戰', '一起補位，守住每一球。', '兩位玩家合作防守，迎戰 AI 前鋒。'],
      competition_add: ['one-vs-two', '一攻二守', '自由對戰', '三人同場，攻守較量。', '一位玩家進攻，兩位玩家聯手守門。'],
      competition_add_ai: ['team-ai', '組隊挑戰 AI', '自由對戰', '集結隊友，迎接挑戰。', '確認紅藍隊陣容，分工進攻與防守。'],
      competition_addadd: ['six-player', '紅藍全員對戰', '自由對戰', '紅藍集結，默契開戰。', '確認隊伍與操控席位，與朋友一起上場。'],
      competition_add_ai_realgame: ['official-ai', '正式賽・挑戰 AI', '正式賽', '每一局，都全力以赴。', '三局兩勝；先確認 AI 強度與局間休息，再開始比賽。'],
      competition_addadd_realgame: ['official-six', '正式賽・紅藍對抗', '正式賽', '準備就緒，正式開賽。', '三局兩勝；確認隊伍與局間休息，一起爭取勝利。'],
      racing_1: ['rings', '高低環', '競速挑戰', '掌握高度，流暢穿環。', '依序通過環門，練習升降控制與精準飛行。'],
      racing_2: ['maze', '立體迷宮', '競速挑戰', '看準通道，找到出口。', '綠框通行、金框升降，穿越迷宮抵達藍色終點。'],
      racing_3: ['grid', '九宮格', '競速挑戰', '看準目標，精準穿越。', '依任務提示通過指定格位，挑戰操控精度。'],
      racing_4: ['tunnel', '單軌隧道', '競速挑戰', '穩住航線，一路向前。', '保持穩定，沿隧道完成精準穿越。'],
      racing_5: ['slalom', '蛇形繞標', '競速挑戰', '左右交錯，飛出節奏。', '依序穿過最後一個環門，即可過關並查看完成秒數。'],
      racing_6: ['spiral', '螺旋爬升', '競速挑戰', '盤旋向上，穩定爬升。', '沿螺旋航線穿過最後一個環門，即可過關並查看完成秒數。']
    };
    const key = location.pathname.split('/').pop().replace(/^drone_soccer_|\.html$/g, '');
    let mode = catalog[key];
    if (!mode) return;
    const roster = window.DroneMatchRoster?.read(new URLSearchParams(location.search));
    const humanCount = roster ? roster.red.humans + roster.blue.humans : null;
    if (roster) {
      mode = [...mode];
      mode[1] = mode[2] === '正式賽' ? '正式賽' : '自由編成對戰';
      mode[3] = humanCount === 0 ? '紅藍交鋒，觀戰開始。' : '自由編隊，一起上場。';
      mode[4] = `紅隊 ${roster.red.humans} 真人＋${roster.red.ai} AI；藍隊 ${roster.blue.humans} 真人＋${roster.blue.ai} AI。${humanCount === 0 ? '按開始後自動起飛，觀察雙方攻防。' : '確認操控席位，再一起進場。'}`;
    }
    const [art, title, category, slogan, description] = mode;
    const radio = location.pathname.includes('/radio/');
    document.querySelectorAll('.fc-start-overlay').forEach(overlay => {
      const card = overlay.querySelector('.fc-card');
      if (!card) return;
      card.classList.add('fc-main-menu');
      overlay.setAttribute('role', 'dialog');
      overlay.setAttribute('aria-label', `${title}主選單`);
      const heading = card.querySelector('h1,h2');
      const playerTitle = heading.textContent.trim();
      heading.textContent = /^P\d/.test(playerTitle) ? playerTitle : title;
      card.querySelector('.fc-kicker').textContent = 'FLIGHT CLUB / PRE-FLIGHT';
      const originalSubtitle = card.querySelector('.fc-subtitle');
      originalSubtitle?.remove();
      const aside = create('aside', 'fc-menu-overview');
      card.dataset.releaseMenu = category === '競速挑戰' ? 'race' : category === '正式賽' ? 'match' : category === '飛行練習' ? 'practice' : 'battle';
      aside.append(create('span', 'fc-menu-wordmark', 'FLIGHT\nCLUB'));
      aside.append(create('span', 'fc-menu-category', category));
      const image = create('img', 'fc-menu-art');
      const venueArt = category === '飛行練習' ? 'practice' : category === '正式賽' ? 'official' : category === '自由對戰' ? 'battle' : null;
      image.src = venueArt ? `../../assets/lobby/mode-${venueArt}.png` : `../../assets/lobby/previews/${art}.svg`;
      image.alt = venueArt ? `${title}遊戲場館實景` : `${title}場地示意`;
      image.width = 1200; image.height = 720;
      aside.append(image, create('h3', 'fc-menu-slogan', slogan), create('p', 'fc-menu-description', description));
      const badges = create('div', 'fc-menu-badges');
      const device = create('span', ''); device.append(icon(radio ? 'pad' : 'guide'), radio ? '搖桿控制' : '鍵盤操控');
      badges.append(device, create('span', '', roster ? (humanCount === 0 ? '全 AI 觀戰' : `${humanCount} 位真人`) : category === '競速挑戰' ? '單人計時挑戰' : category === '飛行練習' ? '不限時陪練' : '攻防對戰'));
      aside.append(badges);
      const tip = create('p', 'fc-menu-tip', radio ? '方向鍵／左桿選擇，X 確認、Y 取消；START 主選單、BACK 返回。' : '確認設定 → 熟悉操控 → 開始飛行');
      aside.append(tip);
      const routeTag = create('div', 'fc-menu-route-tag');
      routeTag.append(create('span', '', category === '競速挑戰' ? 'TIME ATTACK' : category === '正式賽' ? 'OFFICIAL MATCH' : category === '飛行練習' ? 'FLIGHT TRAINING' : 'ARENA BATTLE'), create('strong', '', '準備上場'));
      aside.append(routeTag);
      const layout = create('div', 'fc-menu-layout');
      const content = create('div', 'fc-menu-content');
      const header = card.querySelector('.fc-card-heading');
      const footer = card.querySelector('.fc-start-footer');
      const tabs = card.querySelector('.fc-tabs');
      const pages = card.querySelector('.fc-pages');
      content.append(tabs, pages);
      layout.append(aside, content);
      header.after(layout);
      const playerNavigation = card.querySelector('.fc-player-navigation');
      if (playerNavigation) layout.before(playerNavigation);
      const [settings, guide, rules] = pages.children;
      if (originalSubtitle) { originalSubtitle.classList.remove('fc-subtitle'); rules.prepend(originalSubtitle); }
      settings.classList.add('fc-menu-settings');
      settings.querySelector('.fc-page-intro').textContent = '確認本場設定，準備好就開始。';
      if (category === '正式賽') {
        const row = create('label', 'fc-pregame-camera');
        const copy = create('span', 'fc-camera-setting-copy');
        const title = create('strong', '', '統一視角');
        const hint = create('span', '', '套用到所有玩家；開賽後仍可各自切換。');
        hint.id = `fc-pregame-camera-hint-${++sequence}`;
        copy.append(title, hint);
        const select = create('select', '');
        select.dataset.fcPregameCamera = '';
        select.setAttribute('aria-label', '賽前統一視角');
        select.setAttribute('aria-describedby', hint.id);
        window.DroneCameraControls.modes.forEach(mode => {
          const option = create('option', '', mode.label);
          option.value = mode.id;
          select.append(option);
        });
        select.value = window.DroneCameraControls.preflightMode;
        select.addEventListener('change', () => window.DroneCameraControls.setPreflightMode(select.value));
        row.append(copy, select);
        settings.querySelector('.fc-page-intro').after(row);
      }
      // Separate mixed legacy control/rule grids without copying IDs or inputs.
      Array.from(rules.children).forEach(grid => {
        if (!grid.matches('.grid')) return;
        Array.from(grid.children).forEach(section => {
          if (/觸控|飛行控制|鍵盤|USB.*操控/.test(section.querySelector('p')?.textContent || '')) guide.append(section);
        });
      });
      card.querySelectorAll('p').forEach(p => {
        if (/按任意鍵|點選上方難度|點擊畫面/.test(p.textContent) && !p.querySelector('button,input,select')) p.hidden = true;
      });
      if (settings.children.length === 1) {
        const ready = create('div', 'fc-menu-ready');
        ready.append(icon('check'), create('strong', '', '設定已就緒'), create('p', '', playerNavigation ? '可切換玩家查看操作說明，準備好後一起進場。' : '可先查看操作說明與任務規則，準備好後進場。'));
        settings.append(ready);
      }
      // Native team/level actions launch immediately. Stage the choice, then
      // forward the explicit start to the SAME engine handler exactly once.
      const choices = Array.from(settings.querySelectorAll('.level-btn,#btnTeamRed,#btnTeamBlue'));
      if (choices.length) {
        let selected = choices.find(c => /Medium|Red/.test(c.id) || c.dataset.level === '2') || choices[0];
        let launching = false;
        const action = create('button', 'fc-primary-action fc-menu-confirm');
        action.type = 'button';
        const sync = () => {
          choices.forEach(c => { c.classList.add('fc-menu-choice'); c.setAttribute('aria-pressed', String(c === selected)); });
          const label = selected.querySelector('p')?.textContent.trim() || selected.textContent.trim();
          action.textContent = category === '競速挑戰' ? '開始挑戰 →' : '開始比賽 →';
          footer.querySelector('.fc-footer-hint').textContent = `已選擇：${label}`;
        };
        choices.forEach(choice => {
          choice.setAttribute('role', 'button'); choice.tabIndex = 0;
          choice.addEventListener('click', event => {
            if (launching) return;
            event.preventDefault(); event.stopImmediatePropagation();
            selected = choice; sync();
          }, true);
        });
        action.addEventListener('click', event => {
          event.stopPropagation();
          launching = true;
          try { selected.click(); } finally { launching = false; }
        });
        footer.append(action); sync();
      }
      footer.querySelectorAll('.fc-primary-action').forEach(action => {
        if (!choices.length) action.textContent = category === '飛行練習' ? '開始飛行 →' : '開始比賽 →';
      });
      // The briefing is a game screen: cinematic mode presentation on the
      // left, a focused preparation panel on the right. Move original nodes
      // intact so all engine-owned IDs and handlers keep their behaviour.
      const playerNav = card.querySelector('.fc-player-navigation');
      content.prepend(header);
      if (playerNav) header.after(playerNav);
      content.append(footer);
      card.replaceChildren(layout);
      const options = create('details', 'fc-more-options');
      const summary = create('summary', '', '進階設定');
      summary.append(create('span', '', '畫面・按鍵'));
      const optionBody = create('div', 'fc-more-options-body');
      options.append(summary, optionBody);
      const groupAdvanced = () => {
        const entries = Array.from(settings.children).filter(node => node.matches('.fc-control-entry'));
        if (!entries.length) return;
        entries.forEach(node => optionBody.append(node));
        if (!options.isConnected) settings.append(options);
      };
      groupAdvanced();
      // Control settings may register after the shared interface is mounted.
      new MutationObserver(groupAdvanced).observe(settings, { childList: true });
      footer.addEventListener('click', event => {
        if (!event.target.closest('.fc-primary-action')) return;
        showLaunchCue();
      }, true);
      // Practice's engine starts on an overlay click; browsing the menu must
      // not start it. Only the explicit primary action may reach that handler.
      card.addEventListener('click', event => {
        if (!event.target.closest('.fc-primary-action') && !event.target.closest('.fc-menu-choice')) event.stopPropagation();
      });
      overlay.addEventListener('click', event => {
        if (event.target === overlay) event.stopImmediatePropagation();
      }, true);
    });
    // The old "any key starts" listeners also accept ordinary navigation.
    // Keep all pre-flight keyboard input inside the menu until explicit start.
    for (const type of ['keydown', 'keyup']) window.addEventListener(type, event => {
      const menu = Array.from(document.querySelectorAll('.fc-start-overlay')).find(e => e.getClientRects().length && !e.classList.contains('opacity-0') && e.dataset.launched !== 'true');
      const padOpen = Array.from(document.querySelectorAll('.fc-pad-overlay')).some(e => e.getClientRects().length);
      if (menu && !padOpen && !document.querySelector('dialog[open],:popover-open') && type === 'keydown' && event.key === 'Tab') {
        const focusable = Array.from(menu.querySelectorAll('button,[role="button"],a[href],input,select,textarea,[tabindex]'))
          .filter(e => e.tabIndex >= 0 && !e.disabled && e.getClientRects().length);
        const index = focusable.indexOf(document.activeElement);
        if (focusable.length && (index < 0 || (event.shiftKey ? index === 0 : index === focusable.length - 1))) {
          event.preventDefault(); event.stopImmediatePropagation();
          focusable[event.shiftKey ? focusable.length - 1 : 0].focus();
          return;
        }
      }
      if (menu && !event.target.closest?.('.fc-ui-region')) {
        event.stopImmediatePropagation();
        if (type === 'keydown' && event.key === 'Tab') {
          event.preventDefault();
          menu.querySelector('button,[role="button"],select')?.focus();
        }
      }
    }, true);
  }

  function showLaunchCue() {
    if (document.querySelector('.fc-launch-cue') || matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const cue = create('div', 'fc-launch-cue');
    cue.setAttribute('aria-hidden', 'true');
    cue.append(create('span', '', 'FLIGHT CLUB'), create('strong', '', 'LET’S FLY'), create('i', ''));
    document.body.append(cue);
    setTimeout(() => cue.remove(), 720);
  }

  function enhancePadSetup() {
    const overlay = $('#padSetup'), card = overlay?.querySelector('.ps-card');
    if (!card) return;
    overlay.classList.add('fc-pad-overlay');
    card.classList.add('fc-card', 'fc-ui-region', 'usb-simple-card');
    const original = Array.from(card.children), header = original.shift();
    header.classList.add('fc-card-heading');
    header.prepend(create('p', 'fc-kicker', 'USB CONTROLLER'));
    header.querySelector('h2').textContent = '確認 USB 手把';
    const body = create('div', 'fc-pages');
    const content = create('section', 'fc-page usb-setup-content');
    const legacy = create('div', 'usb-legacy');
    legacy.hidden = true;
    // The engine still updates these fields; show a single connection status below.
    const oldStatus = header.querySelector('#psStatus')?.parentElement;
    if (oldStatus) legacy.append(oldStatus);
    const footer = create('footer', 'fc-pad-footer');
    footer.append(create('span', 'fc-footer-hint', '設定自動儲存'));
    const intro=create('p','fc-settings-intro','接上手把，推動左右搖桿確認方向，即可繼續。');
    content.append(intro);
    const actions = create('div', 'usb-cal-actions');
    const cal = $('#psCal'), clear = $('#psCalClear');
    cal.textContent='校正搖桿';clear.textContent='重設校正';
    actions.append(cal, clear);
    const calBox = $('#psCalBox'), calMessage = $('#psCalFileMsg');
    for (const node of original) {
      if (node.querySelector('#psClose')) footer.append(node);
      else if (node.querySelector('#psPlayers')) {
        node.classList.add('fc-player-assignments');
        node.firstElementChild.classList.add('fc-player-assignments-heading');
        content.append(node);
      }
      else if (node.querySelector('.ps-canvas')) {
        node.classList.add('fc-sticks');
        content.append(node);
      } else legacy.append(node);
    }
    // Reattach engine-owned fields before resolving IDs from their new homes.
    card.replaceChildren(header, body, footer, legacy);
    body.append(content);
    const guide=create('p','usb-flight-guide usb-field-hint');
    intro.after(guide);
    const help=create('details','usb-help');help.id='usbHelp';
    help.append(create('summary','','搖桿偏移或方向不對？'));
    const helpBody=create('div','usb-help-body');
    const mapping=create('div','usb-direction-grid');
    overlay.querySelectorAll('.ps-map').forEach((map,i)=>{
      // Preserve existing saved mappings and event handlers without exposing axis numbers.
      map.querySelectorAll('.ps-inv').forEach((label,j)=>{
        const input=label.querySelector('input');
        label.replaceChildren(input,document.createTextNode(`${i?'右':'左'}桿${j?'上下':'左右'}反向`));
        mapping.append(label);
      });
      legacy.append(map);
    });
    overlay.querySelectorAll('.fc-sticks .ps-vx').forEach(value=>legacy.append(value.parentElement.parentElement));
    const swap=$('#psSwap');swap.textContent='左右搖桿互換';
    helpBody.append(create('div','usb-preset-field'),mapping,swap,
      create('p','usb-field-hint','中心偏移或推不到底時，再將兩支搖桿完整校正。'),actions,calBox,calMessage);
    help.append(helpBody);
    const cancel=create('button','ps-btn','取消校正');cancel.type='button';cancel.id='psCalCancel';calBox.append(cancel);
    const calInstruction=calBox.querySelector('span');
    if(calInstruction)calInstruction.textContent='校正中：將兩支搖桿推到上下左右盡頭，再放回中央；油門桿也請置中，最後按「完成」。';
    calBox.querySelector('#psCalDone').textContent='已回中，完成校正';
    content.append(help);
    // Keep engine-owned fields for saved settings and live calibration updates.
    // They are no longer part of the user-facing setup flow.
    const editName = $('#psEditName');
    if (editName) {
      const note = editName.parentElement;
      note.classList.add('fc-player-edit-note');
      note.replaceChildren('目前檢查：', editName, create('span', '', '點選玩家名稱，檢查各自的搖桿。'));
    }
    // Keep the engine-owned controls and listeners, giving each field its own row.
    document.querySelectorAll('#psPlayers .ps-padsel').forEach(select => {
      const field = create('label', 'fc-player-device');
      field.htmlFor = select.id;
      field.append(create('span', '', '指派手把'));
      select.before(field);
      field.append(select);
    });
  }

  function enhanceSplitStart() {
    const overlays = Array.from(document.querySelectorAll('.fc-start-overlay'));
    if (overlays.length < 2) return;
    const navigations = [];
    const selectPlayer = index => {
      overlays.forEach((overlay, i) => overlay.classList.toggle('fc-start-inactive', i !== index));
      navigations.forEach(navigation => Array.from(navigation.children).forEach((button, i) => button.setAttribute('aria-pressed', String(i === index))));
    };
    overlays.forEach(overlay => {
      overlay.classList.add('fc-shared-start');
      const navigation = create('div', 'fc-player-navigation');
      navigation.setAttribute('role', 'group');
      navigation.setAttribute('aria-label', '選擇玩家設定與操作說明');
      overlays.forEach((player, index) => {
        const label = player.querySelector('h1,h2').textContent.trim().replace(/\s*[（(].*$/, '');
        const button = create('button', '', label);
        button.type = 'button';
        button.addEventListener('click', event => { event.stopPropagation(); selectPlayer(index); });
        navigation.append(button);
      });
      navigations.push(navigation);
      overlay.querySelector('.fc-card-heading').after(navigation);
    });
    selectPlayer(0);
  }

  function enhanceSetupGuide(controls, guide) {
    if (!guide) return;
    const home = document.createComment('Controls guide original page');
    guide.before(home);
    const dialog = create('dialog', 'fc-guide-dialog fc-team-guide fc-ui-region');
    const owner=Array.from(document.querySelectorAll('[id$="gameUI"]')).indexOf(controls.closest('[id$="gameUI"]'));
    dialog.dataset.controllerOwner=owner<0?'shared':String(owner);
    dialog.setAttribute('aria-label', '操作指南');
    const heading = create('header', 'fc-dialog-heading');
    const close = create('button', 'fc-close', '關閉 ×');
    close.type = 'button';
    close.addEventListener('click', () => dialog.close());
    heading.append(create('strong', '', '操作指南'), close);
    dialog.append(heading);
    document.body.append(dialog);
    const toggle = create('button', 'fc-hud-button', '操作指南');
    toggle.type = 'button';
    toggle.dataset.controllerGuide=owner<0?'shared':String(owner);
    toggle.dataset.fcIcon = 'guide';
    toggle.setAttribute('aria-haspopup', 'dialog');
    toggle.setAttribute('aria-expanded', 'false');
    dialog.id = `fc-team-guide-${++sequence}`;
    toggle.setAttribute('aria-controls', dialog.id);
    let wasHidden;
    toggle.addEventListener('click', () => {
      wasHidden = guide.hidden;
      guide.hidden = false;
      dialog.append(guide);
      dialog.showModal();
      toggle.setAttribute('aria-expanded', 'true');
    });
    dialog.addEventListener('close', () => {
      home.after(guide);
      guide.hidden = wasHidden;
      toggle.setAttribute('aria-expanded', 'false');
      if (toggle.isConnected && toggle.getClientRects().length) toggle.focus({ preventScroll: true });
    });
    dialog.addEventListener('click', event => { if (isDialogBackdrop(event, dialog)) dialog.close(); });
    controls.append(toggle);
  }

  function enhanceHUD() {
    const huds = document.querySelectorAll('[id$="gameUI"]');
    huds.forEach(hud => {
      hud.classList.add('fc-hud');
      if (huds.length > 1) hud.classList.add('fc-split-hud');
      hud.querySelector('header > .glassmorphism')?.classList.add('fc-score-card');
      hud.querySelector('[id$="compassVal"]')?.closest('.glassmorphism')?.classList.add('fc-compass-card');
      hud.querySelector('[id$="telemetrySpeed"]')?.closest('.glassmorphism')?.classList.add('fc-telemetry');
      const attitude = hud.querySelector('[id$="horizonLine"]')?.closest('.glassmorphism');
      if (attitude) {
        // Original engines still write to these IDs every frame.
        attitude.classList.add('fc-attitude');
        attitude.hidden = true;
        attitude.setAttribute('aria-hidden', 'true');
      }
      const controls = hud.querySelector('[id$="btnReset"]')?.parentElement;
      if (controls) {
        const guide = Array.from(hud.querySelectorAll('footer .glassmorphism')).find(panel => {
          const title = panel.querySelector(':scope > span');
          return title && /指南/.test(title.textContent);
        });
        if (guide) enhanceGuide(guide, controls);
        else {
          const prefix = hud.id.replace(/gameUI$/, '');
          enhanceSetupGuide(controls, document.getElementById(`${prefix}startScreen`)?.querySelector('.fc-pages')?.children[1]);
        }
        enhanceToolbar(controls);
      }
    });
    const sharedControls = $('#btnResetAll')?.parentElement;
    if (sharedControls) {
      // Reuse the live instructions, including tables filled in by the game on load.
      enhanceSetupGuide(sharedControls, $('#startOverlay .fc-pages')?.children[1]);
      enhanceToolbar(sharedControls);
      sharedControls.classList.add('fc-shared-toolbar');
      $('#topBar').append(sharedControls);
      $('#topBar').classList.add('fc-unified-scoreboard');
    }
    // Give icon-only actions accessible names, retaining each game's own handlers.
    document.querySelectorAll('button[id$="btnSound"],button[id$="btnReset"],#btnMuteAll').forEach(button => {
      if (button.textContent.trim() || button.hasAttribute('aria-label')) return;
      const label = /Sound|Mute/.test(button.id) ? '切換聲音' : '重設無人機';
      button.setAttribute('aria-label', label);
      button.title ||= label;
    });
  }

  function enhanceMiniTelemetry() {
    document.querySelectorAll('.panel-root [id$="-spd"]').forEach(value => {
      const panel = value.closest('.glassmorphism');
      if (!panel || panel.classList.contains('fc-mini-telemetry')) return;
      panel.classList.add('fc-mini-telemetry');
      [['alt', '高度', 'm'], ['spd', '速度', 'm/s'], ['thr', location.pathname.includes('/radio/') ? '油門' : '飛行油門', '']].forEach(([suffix, label, unit]) => {
        const number = panel.querySelector(`[id$="-${suffix}"]`);
        if (!number) return;
        const metric = number.parentElement;
        const title = create('span', 'fc-mini-label', label);
        metric.replaceChildren(title, number);
        if (unit) metric.append(create('small', '', unit));
      });
    });
  }

  function enhanceMatchClock() {
    const clocks = Array.from(document.querySelectorAll('[id$="gameClockPanel"]'));
    let clock = clocks[0];
    if (clock) {
      // Split-player clocks all show the same match time. Keep their live IDs,
      // but display a single shared clock above the entire playing area.
      clocks.slice(1).forEach(duplicate => duplicate.classList.add('fc-clock-duplicate'));
    } else {
      const race = $('#raceTimer');
      const mission = $('#bigTimerVal');
      if (race) {
        clock = create('div', 'glassmorphism');
        clock.append(create('span', 'fc-clock-caption', '競速計時'), race.parentElement);
      } else if (mission) {
        clock = mission.closest('.glassmorphism');
      }
    }
    if (clock) {
      clock.classList.add('fc-match-clock');
      clock.dataset.fcMainClock = '';
      clock.setAttribute('role', 'timer');
      clock.setAttribute('aria-label', $('#raceTimer') || $('#bigTimerVal') ? '競速計時' : '比賽計時');
      // Escape each HUD's size container so fixed positioning is viewport-based.
      document.body.append(clock);
      const arena = $('#dualArena');
      if (arena) arena.classList.add('fc-clock-arena');
      else document.querySelectorAll('.fc-hud').forEach(hud => {
        hud.classList.add('fc-clock-hud');
        const header = hud.querySelector('header');
        const toolbar = header.querySelector('.fc-flight-toolbar');
        const summary = create('div', 'fc-match-summary');
        Array.from(header.children).filter(child => child !== toolbar).forEach(child => summary.append(child));
        header.prepend(summary);
      });
    }
    const scoreboard = $('#topBar');
    if (scoreboard && $('#clockText')) {
      scoreboard.classList.add('fc-centered-scoreboard');
      $('#clockText').dataset.fcMainClock = '';
      if ($('#setLabel')) {
        $('#clockText').parentElement.classList.add('fc-match-status');
        scoreboard.classList.add('fc-official-scoreboard');
        const blue = $('#scoreBlue').parentElement;
        const red = $('#scoreRed').parentElement;
        const status = $('#clockText').parentElement.parentElement;
        blue.classList.add('fc-team-score', 'fc-team-blue');
        red.classList.add('fc-team-score', 'fc-team-red');
        status.classList.add('fc-official-clock');
        // Keep DOM and visual order consistent: blue, match time, red.
        scoreboard.insertBefore(blue, status);
        scoreboard.insertBefore(red, status.nextSibling);
        // User request (2026-10-29): hide official-match takeoff/landing in this bar.
        // Keep live IDs/handlers; keyboard and USB throttle remain engine-controlled.
        scoreboard.querySelector('.fc-flight-actions').hidden = true;
        // Keep the scoreboard reserved for scores. Settings and guides remain reachable
        // in a separate compact dock; the engine-owned takeoff/landing buttons stay hidden.
        const tools = scoreboard.querySelector('.fc-shared-toolbar');
        tools.hidden = false; tools.classList.add('fc-official-tools');
        document.body.append(tools);
      } else {
        scoreboard.classList.add('fc-inline-scoreboard');
        $('#scoreRed').parentElement.classList.add('fc-inline-red');
        $('#scoreBlue').parentElement.classList.add('fc-inline-blue');
        $('#clockText').parentElement.classList.add('fc-inline-clock');
      }
    }
  }

  function centerFlightStatus() {
    const compasses = Array.from(document.querySelectorAll('.fc-compass-card'));
    if (!compasses.length) return;
    const status = create('div', 'fc-flight-status');
    const headings = create('div', 'fc-heading-row');
    compasses.forEach((compass, index) => {
      // Move the live elements so each engine keeps updating its original IDs.
      if (compasses.length > 1) compass.firstElementChild.textContent = `P${index + 1} 航向`;
      headings.append(compass);
    });
    status.append(headings);
    const clock = document.querySelector('.fc-match-clock');
    if (clock) status.append(clock);
    document.body.append(status);
    document.documentElement.classList.add('fc-centered-flight-status');
    // Untimed practice needs the same reserved center column as timed modes.
    document.querySelectorAll('.fc-hud:not(.fc-clock-hud):not(.fc-split-hud)').forEach(hud => {
      hud.classList.add('fc-clock-hud');
      const header = hud.querySelector('header');
      const summary = create('div', 'fc-match-summary');
      Array.from(header.children).filter(child => !child.classList.contains('fc-flight-toolbar')).forEach(child => summary.append(child));
      header.prepend(summary);
    });
    const headers = Array.from(document.querySelectorAll('.fc-clock-hud:not(.fc-split-hud) > header'));
    const reserveSpace = () => {
      document.documentElement.style.setProperty('--fc-flight-status-width', `${status.getBoundingClientRect().width}px`);
      document.documentElement.style.setProperty('--fc-flight-status-height', `${status.getBoundingClientRect().height}px`);
      const bottom = Math.max(status.getBoundingClientRect().bottom, ...headers.map(header => header.getBoundingClientRect().bottom));
      document.documentElement.style.setProperty('--fc-flight-notification-top', `${bottom + 12}px`);
    };
    reserveSpace();
    const observer = new ResizeObserver(reserveSpace);
    [status, ...headers].forEach(element => observer.observe(element));
    window.addEventListener('resize', reserveSpace);
  }

  function enhanceOfficialPanels() {
    const overlay = document.getElementById('faiOverlay');
    if (!overlay) return;
    overlay.setAttribute('role', 'region'); overlay.setAttribute('aria-label', '正式賽事狀態');
    let sizing;
    const enhance = () => {
      const card = overlay.firstElementChild;
      if (!card || card.classList.contains('fc-official-panel')) return;
      card.classList.add('fc-official-panel', 'fc-ui-region');
      card.tabIndex = 0;
      card.prepend(create('span', 'fc-official-kicker', 'FLIGHT CLUB / OFFICIAL MATCH'));
      sizing?.disconnect();
      sizing = new ResizeObserver(() => card.classList.toggle('fc-official-scroll', card.scrollHeight > card.clientHeight + 1));
      sizing.observe(card);
      const action = card.querySelector('[data-fai-action]');
      if (action) requestAnimationFrame(() => { if (!document.querySelector('dialog[open]')) action.focus({ preventScroll: true }); });
    };
    new MutationObserver(enhance).observe(overlay, { childList: true }); enhance();
  }

  try {
  document.documentElement.classList.add('fc-game');
  document.querySelectorAll('#victoryScreen,#finishScreen,#endScreen').forEach(overlay => {
    overlay.classList.add('fc-result-overlay');
    const card = overlay.firstElementChild;
    if (!card) return;
    card.classList.add('fc-result-card', 'fc-ui-region');
    card.prepend(create('span', 'fc-result-kicker', 'FLIGHT CLUB / FLIGHT REPORT'));
    const title = card.querySelector('h2');
    if (title) { title.id ||= `fc-result-title-${++sequence}`; overlay.setAttribute('aria-labelledby', title.id); }
    overlay.setAttribute('role', 'region');
    // Native engines reveal these results; move keyboard focus to their existing next action.
    let visible = false;
    const syncResult = () => {
      const next = !overlay.classList.contains('hidden') && getComputedStyle(overlay).display !== 'none';
      if (next && !visible) requestAnimationFrame(() => card.querySelector('button:not(.hidden)')?.focus({ preventScroll: true }));
      visible = next;
    };
    new MutationObserver(syncResult).observe(overlay, { attributes: true, attributeFilter: ['class', 'style'] });
  });
  document.querySelectorAll('[id$="startScreen"],#startOverlay').forEach(enhanceStart);
  enhanceSplitStart();
  enhancePadSetup();
  enhanceCameraSettings();
  enhanceAppearance();
  redesignMainMenus();
  enhanceHUD();
  enhanceOfficialPanels();
  enhanceMatchClock();
  centerFlightStatus();
  enhanceMiniTelemetry();
  enhanceCameraSelectors();
  window.addEventListener('drone-camera-ready', enhanceCameraSelectors);
  // Some six-player games build their panels in window.onload.
  window.addEventListener('load', () => { enhanceMiniTelemetry(); enhanceCameraSelectors(); }, { once: true });
  document.querySelectorAll('.difficulty-option,.attack-difficulty-option,.red-ai-difficulty-option,.break-option,[id*="btnTeam"],#btnLaunch').forEach(element => {
    if (element.matches('button')) return;
    element.setAttribute('role', 'button');
    element.tabIndex = 0;
    element.addEventListener('keydown', event => {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); event.stopPropagation(); element.click(); }
    });
  });
  // Editing parameters and navigating tabs must never fly a drone or start a match.
  for (const type of ['keydown', 'keyup']) window.addEventListener(type, event => {
    // Tab from the document body enters setup instead of triggering "any key to start".
    if (event.key === 'Tab' && event.target === document.body) {
      const overlay = Array.from(document.querySelectorAll('.fc-pad-overlay,.fc-start-overlay')).find(element => element.getClientRects().length && !element.classList.contains('opacity-0'));
      if (overlay) {
        event.stopImmediatePropagation();
        if (type === 'keydown') {
          event.preventDefault();
          overlay.querySelector('button,[role="button"],input,select')?.focus();
        }
        return;
      }
    }
    if (!event.target.closest?.('.fc-ui-region')) return;
    if (event.target.closest('.fc-flight-toolbar') && !document.querySelector(':popover-open,dialog[open]') &&
        !Array.from(document.querySelectorAll('.fc-start-overlay,.fc-pad-overlay')).some(e => e.getClientRects().length && !e.classList.contains('opacity-0')) &&
        !['Tab', 'Enter', ' ', 'Escape'].includes(event.key) && !event.ctrlKey && !event.altKey && !event.metaKey) {
      // After closing an auxiliary panel, the next flight key resumes control.
      event.target.blur();
      return;
    }
    if (event.target.matches('.fc-page')) { event.stopPropagation(); return; }
    // A modal can itself hold focus after its blank inset is clicked. Keep
    // Escape and other keys out of legacy "any key starts" game listeners.
    if (event.target.closest('dialog[open]')) event.stopImmediatePropagation();
    if (event.target.closest('input,textarea,select,button,[role="button"],summary')) {
      // Stop the game's global shortcuts while retaining native field editing.
      // Tab arrow navigation is handled here because capture stops target listeners.
      event.stopImmediatePropagation();
      if (type === 'keydown' && !event.repeat && (event.key === 'Enter' || event.key === ' ') && event.target.matches('summary')) {
        event.preventDefault(); event.target.click();
      }
      if (type === 'keydown' && !event.repeat && (event.key === 'Enter' || event.key === ' ') && event.target.matches('[role="button"]:not(button)')) {
        event.preventDefault(); event.target.click();
      }
      if (type === 'keydown' && event.target.getAttribute('role') === 'tab') {
        const tabs = Array.from(event.target.parentElement.children);
        const index = tabs.indexOf(event.target);
        const vertical = event.target.parentElement.getAttribute('aria-orientation') === 'vertical';
        const next = event.key === (vertical ? 'ArrowDown' : 'ArrowRight') ? (index + 1) % tabs.length : event.key === (vertical ? 'ArrowUp' : 'ArrowLeft') ? (index + tabs.length - 1) % tabs.length : event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : -1;
        if (next >= 0) { event.preventDefault(); tabs[next].click(); tabs[next].focus(); }
      }
      if (type === 'keydown' && event.target.matches('[data-fc-camera-mode]')) {
        const options = Array.from(event.target.closest('.fc-camera-picker').querySelectorAll('[data-fc-camera-mode]'));
        const index = options.indexOf(event.target);
        const step = {ArrowRight: 1, ArrowLeft: -1, ArrowDown: 3, ArrowUp: -3}[event.key];
        const next = event.key === 'Home' ? 0 : event.key === 'End' ? options.length - 1 : step === undefined ? -1 : (index + step + options.length) % options.length;
        if (next >= 0) { event.preventDefault(); options[next].focus(); }
      }
    }
  }, true);
  } finally {
    // Reveal only after setup cards, menus and the HUD share their final layout.
    document.documentElement.classList.add('fc-ui-ready');
  }
})();
