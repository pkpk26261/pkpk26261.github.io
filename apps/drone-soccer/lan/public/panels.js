/* Shared settings presentation. Engine state and network permissions remain owned by app.js. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const node = (tag, className, text) => { const el=document.createElement(tag); if(className)el.className=className;if(text!==undefined)el.textContent=text;return el; };
  let modal=null, pending=null, returnTarget=null, returnKey='';
  function finish(value) {
    const resolve=pending;pending=null;
    if(modal?.open)modal.close();
    window.dispatchEvent(new CustomEvent('flight-panel-modal',{detail:{open:false}}));
    const target=returnTarget?.isConnected?returnTarget:[...document.querySelectorAll('[data-live-key]')].find(el=>el.dataset.liveKey===returnKey)||($('adminRoomSearch')?.getClientRects().length?$('adminRoomSearch'):document.querySelector('[data-admin-panel][aria-pressed=true]'));
    target?.focus({preventScroll:true});resolve?.(value);
  }
  function ask(options, input) {
    if(pending)return Promise.resolve(input?null:false);
    if(!modal){
      modal=node('dialog','fc-panel-dialog');modal.id='panelDecision';modal.setAttribute('aria-labelledby','panelDecisionTitle');modal.setAttribute('aria-describedby','panelDecisionDescription');
      modal.innerHTML='<form method="dialog"><header><p class="panel-eyebrow">MATCH CONTROL</p><h2 id="panelDecisionTitle"></h2><button type="button" class="panel-dialog-close" aria-label="關閉確認視窗">×</button></header><div class="panel-decision-body"><p id="panelDecisionDescription"></p><label class="panel-prompt-field" hidden><span id="panelPromptLabel"></span><input id="panelPromptInput" autocomplete="off"></label></div><footer><button type="button" id="panelDecisionCancel">返回</button><button type="submit" id="panelDecisionAccept" class="primary">確認</button></footer></form>';
      document.body.append(modal);
      modal.querySelector('.panel-dialog-close').onclick=()=>finish(modal.dataset.input==='true'?null:false);
      $('panelDecisionCancel').onclick=()=>finish(modal.dataset.input==='true'?null:false);
      modal.addEventListener('cancel',event=>{event.preventDefault();finish(modal.dataset.input==='true'?null:false);});
      modal.querySelector('form').onsubmit=event=>{event.preventDefault();if(modal.dataset.input==='true'){const value=$('panelPromptInput').value.trim();if(!value){$('panelPromptInput').setCustomValidity('請輸入名稱');$('panelPromptInput').reportValidity();return;}finish(value);}else finish(true);};
      $('panelPromptInput').oninput=()=>{ $('panelPromptInput').setCustomValidity(''); };
    }
    returnTarget=document.activeElement;returnKey=returnTarget?.dataset.liveKey||'';
    modal.dataset.input=String(!!input);modal.dataset.danger=String(!!options.danger);
    $('panelDecisionTitle').textContent=options.title;
    $('panelDecisionDescription').textContent=options.description||'';
    $('panelDecisionAccept').textContent=options.acceptLabel||'確認';
    $('panelDecisionCancel').textContent=options.cancelLabel||'返回';
    const field=modal.querySelector('.panel-prompt-field');field.hidden=!input;
    $('panelPromptInput').required=!!input;
    if(input){$('panelPromptLabel').textContent=options.label||'名稱';$('panelPromptInput').value=options.value||'';$('panelPromptInput').maxLength=options.maxLength||36;$('panelPromptInput').setCustomValidity('');}
    modal.showModal();window.dispatchEvent(new CustomEvent('flight-panel-modal',{detail:{open:true}}));
    (input?$('panelPromptInput'):$('panelDecisionCancel')).focus();if(input)$('panelPromptInput').select();
    return new Promise(resolve=>{pending=resolve;});
  }
  window.FlightPanels=Object.freeze({confirm:options=>ask(options,false),prompt:options=>ask(options,true),get busy(){return !!modal?.open;}});
  document.body.classList.add('panel-system');
  const sound=$('sound'),fullscreen=$('fullscreen');
  if(sound){sound.setAttribute('aria-pressed','false');document.addEventListener('click',event=>{if(event.target.closest('#sound'))sound.setAttribute('aria-pressed',String(sound.textContent.includes('關閉')));});}
  if(fullscreen){fullscreen.setAttribute('aria-pressed','false');document.addEventListener('fullscreenchange',()=>{const active=!!document.fullscreenElement;fullscreen.setAttribute('aria-pressed',String(active));fullscreen.textContent=active?'退出全螢幕':'全螢幕';});}
  function heading(title,subtitle,eyebrow='SETTINGS') {
    const el=node('div','settings-section-heading');el.append(node('p','panel-eyebrow',eyebrow),node('h3','',title));if(subtitle)el.append(node('p','settings-section-note',subtitle));return el;
  }
  function group(title,subtitle,items,extra='') {
    const el=node('section','settings-group '+extra);el.append(heading(title,subtitle));for(const item of items)if(item)el.append(item);return el;
  }
  function field(id) { const element=$(id);return element?.closest('label')||element; }

  function admin() {
    const total=$('roomTotal');
    if(total){total.setAttribute('role','status');total.setAttribute('aria-live','polite');total.setAttribute('aria-atomic','true');total.setAttribute('aria-label','符合篩選的房間數與全部房間數');}
    const form=$('newRoomForm');if(!form)return;
    const summary=$('creationCount');
    const updateCount=()=>{
      const red=Number($('newRedHumans').value)+Number($('newRedAI').value),blue=Number($('newBlueHumans').value)+Number($('newBlueAI').value),humans=Number($('newRedHumans').value)+Number($('newBlueHumans').value);
      const valid=red>=1&&red<=3&&blue>=1&&blue<=3;
      summary.dataset.valid=String(valid);
      summary.textContent=valid?'紅隊 '+red+' 台 · 藍隊 '+blue+' 台 ／ '+humans+' 位真人、'+(red+blue-humans)+' 台 AI':'每隊合計需要 1–3 台，請調整真人或 AI 數量。';
      $('newRoomError').textContent='';
    };
    form.addEventListener('input',updateCount);form.addEventListener('change',updateCount);updateCount();
  }
  function host() {
    const camera=$('cameraForm');if(!camera)return;
    const title=camera.querySelector('h2'),firstNote=camera.querySelector('p');title.before(node('p','panel-eyebrow','PLAYER VIEW / 統一視角'));
    firstNote.textContent='所有玩家使用同一組視野規則，從己方短邊觀看賽場。';
    const mode=field('cameraMode'),numbers=camera.querySelector('.camera-numbers'),tail=camera.querySelector('p:last-of-type'),button=$('applyCamera');button.classList.add('primary');
    const preview=node('div','camera-layout-preview');preview.setAttribute('aria-hidden','true');preview.innerHTML='<span class="preview-team red">紅隊</span><span class="preview-field"><i></i><i></i><i></i></span><span class="preview-team blue">藍隊</span><small>同隊共用固定機位 · 統一 16:9</small>';mode.after(preview);
    mode.classList.add('settings-option-row');
    for(const [id,name,help,unit] of [['fov','視野範圍','數值越大，可看到的範圍越寬。','°'],['distance','網外距離','從己方短邊往後退的距離。','m'],['height','觀看高度','固定站位的眼睛離地高度。','m']]){
      const label=field(id),input=$(id);label.replaceChildren();const copy=node('span','settings-option-copy');copy.append(node('strong','',name),node('small','',help));const value=node('span','settings-number');value.append(input,node('span','',unit));label.append(copy,value);label.classList.add('settings-option-row');
    }
    numbers.classList.add('camera-setting-rows');tail.classList.add('settings-callout');tail.textContent='比賽進行中無法變更。請先暫停，再套用；所有真人重新準備後才能恢復。';
    const state=node('span','settings-apply-state','變更後需統一套用');state.id='cameraApplyState';state.setAttribute('role','status');const footer=node('div','camera-apply-footer');footer.append(state,button);camera.append(footer);
    const syncPreview=()=>{preview.dataset.station=$('cameraMode').value;};camera.addEventListener('input',()=>{state.textContent='有變更尚未套用';state.dataset.state='pending';syncPreview();});syncPreview();
    camera.addEventListener('submit',()=>{state.textContent='已送出，等待主機確認';state.dataset.state='pending';});
    window.addEventListener('flight-camera-applied',event=>{state.textContent='主機已套用 · 版本 '+event.detail.revision;state.dataset.state='saved';syncPreview();});
    const roster=$('createForm');roster.querySelector('h2').before(node('p','panel-eyebrow','SQUAD / 隊伍編成'));
    roster.querySelector('p:not(.panel-eyebrow)').textContent='真人優先擔任前鋒，其他機體負責防守。每隊合計 1–3 台。';
    const invitation=$('invite');invitation.querySelector('.dialog-head').prepend(node('p','panel-eyebrow','INVITE / 邀請玩家'));invitation.querySelector('.invite-note').classList.add('settings-callout');
    $('interfaceInfo').before(node('p','invite-steps','01 掃描 QR　　02 選擇席位　　03 準備上場'));
    const direct=node('a','invite-direct-link','此裝置已完成信任？直接開啟 HTTPS →');direct.id='directJoinURL';direct.target='_blank';direct.rel='noopener';direct.hidden=true;invitation.querySelector('.invite-info').append(direct);
  }
  function player() {
    const root=document.querySelector('.advanced-content');if(!root)return;
    const preset=field('preset'),axes=$('axisMap'),deadzone=field('deadzone'),gear=field('gearButton'),actions=$('calSweep').closest('.actions'),calFinish=$('calFinish'),raw=$('padTest'),note=root.querySelector('small');
    const nav=node('nav','controller-settings-nav');nav.setAttribute('aria-label','進階設定分類');nav.setAttribute('role','tablist');
    const bodies=node('div','controller-settings-pages');
    const sections=[group('方向與按鈕','選擇配置，再確認四個方向的對應。',[preset,axes,deadzone,gear],'controller-config-page'),group('全行程校正','緩慢走完每個方向，回到中間後再儲存。',[actions],'controller-calibration-page'),group('原始訊號','需要診斷時，查看瀏覽器實際收到的數值。',[raw,note],'controller-raw-page')];
    const step=node('ol','calibration-steps');for(const text of ['搖桿依序推至最左、右、上、下','放開全部按鈕，雙桿回到中間','按「完成校正」，保存有效行程'])step.append(node('li','',text));sections[1].insertBefore(step,actions);
    const cancel=node('button','','取消本次校正');cancel.id='calCancel';cancel.type='button';cancel.disabled=true;actions.append(cancel);
    const status=node('p','calibration-state','目前保留已儲存的校正。');status.id='calibrationState';sections[1].append(status);
    const sync=()=>{const active=!calFinish.disabled;cancel.disabled=!active;status.textContent=active?'正在記錄行程；回中後按完成，或取消保留原設定。':'目前保留已儲存的校正。';status.dataset.state=active?'active':'idle';};new MutationObserver(sync).observe(calFinish,{attributes:true,attributeFilter:['disabled']});sync();
    sections.forEach((section,i)=>{const id='controllerSettingsPage'+i,button=node('button','',['方向配置','範圍校正','原始訊號'][i]);section.id=id;section.setAttribute('role','tabpanel');button.type='button';button.id='controllerSettingsTab'+i;button.setAttribute('role','tab');button.setAttribute('aria-controls',id);section.setAttribute('aria-labelledby',button.id);button.onclick=()=>{sections.forEach((s,j)=>{s.hidden=i!==j;const b=nav.children[j];b.setAttribute('aria-selected',String(i===j));b.tabIndex=i===j?0:-1;});};button.onkeydown=event=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;event.preventDefault();const j=event.key==='Home'?0:event.key==='End'?2:(i+(event.key==='ArrowRight'?1:2))%3;nav.children[j].click();nav.children[j].focus();};nav.append(button);bodies.append(section);});
    root.replaceChildren(nav,bodies);nav.firstElementChild.click();
    const heading=$('setupPanel').querySelector('.dialog-head');heading.querySelector('h2').textContent='確認操控';heading.querySelector('.eyebrow').textContent='CONTROLLER CHECK';
    const noteSave=node('p','controller-save-note','方向與按鈕變更即時儲存在此裝置；全行程校正需按完成才會保存。');bodies.append(noteSave);
  }
  if(document.body.dataset.serviceRole==='admin')admin();
  if(document.body.dataset.role==='host')host();
  if(document.body.dataset.role==='player')player();
})();
