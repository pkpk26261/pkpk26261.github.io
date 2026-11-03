/* Mirror engine-owned DOM state. Never infer whether a match may start. */
(() => {
  'use strict';
  if(document.body.dataset.role!=='host')return;
  const $=id=>document.getElementById(id);
  const assign=(id,value)=>{if($(id).textContent!==value)$(id).textContent=value;};
  function sync(){
    const active=!$('roomControls').hidden;
    document.body.dataset.roomActive=String(active);
    for(const id of ['matchOverview','hostCodeLabel','inviteOpen'])$(id).hidden=!active;
    const state=$('roomStatus').dataset.state||'';
    document.body.dataset.matchState=state;
    if(!active)return;
    assign('matchReadiness',['red','blue'].map(team=>(team==='red'?'紅隊':'藍隊')+' '+$(team+'TeamCount').textContent).join('  ·  '));
    const waiting=state==='lobby'?$('start').disabled:state==='paused'?$('resume').disabled:false;
    assign('readinessHint',state==='running'?'需要調整視角或處理席位時，請先暫停。':state==='ended'?'結果已保存，可返回管理中心安排下一場。':waiting?'等待比賽引擎與所有真人完成準備。':state==='paused'?'全員就緒，可以恢復比賽。':'全員就緒，可以開始比賽。');
    $('target').hidden=$('observer').value!=='follow';
  }
  let queued=false;
  const schedule=()=>{if(queued)return;queued=true;queueMicrotask(()=>{queued=false;sync();});};
  const observer=new MutationObserver(schedule);
  for(const id of ['roomControls','roomStatus','redTeamCount','blueTeamCount','start','resume'])observer.observe($(id),{attributes:true,attributeFilter:['hidden','disabled','data-state'],childList:true,characterData:true,subtree:true});
  $('observer').addEventListener('change',sync);
  new MutationObserver(schedule).observe($('target'),{childList:true});
  // Seat rows refresh live. Preserve a keyboard user's action and dialog return path.
  let focusedSeat=null;
  document.addEventListener('focusin',event=>{focusedSeat=event.target.closest?.('#seats button')||null;});
  const prepareSeats=()=>{
    for(const row of $('seats').children)for(const button of row.querySelectorAll('button'))button.dataset.liveKey='host-seat:'+row.dataset.team+':'+row.style.gridRow+':'+button.textContent.trim();
    if(focusedSeat&&!focusedSeat.isConnected&&document.activeElement===document.body&&!document.querySelector('dialog[open]')){
      const key=focusedSeat.dataset.liveKey;
      const target=[...$('seats').querySelectorAll('button')].find(button=>button.dataset.liveKey===key);
      (target||$('tabRoom')).focus({preventScroll:true});
    }
  };
  new MutationObserver(prepareSeats).observe($('seats'),{childList:true});
  prepareSeats();sync();
})();
