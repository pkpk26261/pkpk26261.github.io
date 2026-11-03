/* Per-aircraft motor interlock. Contact samples come from resolved geometry,
   not collision sounds/impulses; all durations use simulation time. */
(() => {
  'use strict';
  const states = new WeakMap();
  const tuning = Object.freeze({ qualify: 0.5, hold: 2, strongDown: -0.7,
    keepDown: -0.5, relativeSpeed: 0.06, neutral: 0.12 });
  const state = actor => {
    if (!states.has(actor)) states.set(actor, { locked: false, reason: '', contactSeconds: 0,
      holdSeconds: 0, down: 0, strong: false, contacts: [], neutral: false, pad: null });
    return states.get(actor);
  };
  const pilots = () => window.DroneControlSettings?.pilots || [];
  const reasons = { trapped: '受困停機', landed: '降落停機', failure: '故障停機' };
  let adapter, panel, revision = 0;
  function clearHold(s) { s.contactSeconds = s.holdSeconds = 0; s.strong = false; }
  function clearOutput(actor) {
    actor.input ||= {};
    for (const k of ['throttle', 'pitch', 'roll', 'yaw']) actor.input[k] = 0;
    actor.autoFlightState = 'none';
  }
  function stop(actor, reason = 'failure') {
    if (!actor) return;
    const s = state(actor);
    // Landing/result callbacks cannot downgrade a fault to deliberate landing.
    if (!s.locked || s.reason === 'landed') s.reason = reason;
    s.locked = true; s.neutral = false; clearHold(s);
    window.DroneFlightInput?.reset(actor);
    clearOutput(actor);
    actor.targetAltitude = actor.position.y;
    revision++;
    updateUI();
  }
  function isStopped(actor) { return states.get(actor)?.locked === true; }
  function fault(actor) { return isStopped(actor) && state(actor).reason !== 'landed'; }
  function reset(actor) { states.delete(actor); revision++; }
  function cancel(actor) { const s = states.get(actor); if (s) clearHold(s); }
  function contact(actor, normal, velocity = { x: 0, y: 0, z: 0 }, kind = 'obstacle') {
    if (!actor?.position || !normal) return;
    const s = state(actor);
    // Iterative depenetration can report the same supporting face hundreds of
    // times in one step. Keep a contact manifold, not an iteration history;
    // duplicate reports must not dominate moving-surface averages or the
    // opposing-normal search used by the trapped-motor interlock.
    if(s.contacts.some(c=>c.kind===kind
      &&(c.normal.x-normal.x)**2+(c.normal.y-normal.y)**2+(c.normal.z-normal.z)**2<1e-6
      &&(c.velocity.x-(velocity.x||0))**2+(c.velocity.y-(velocity.y||0))**2+(c.velocity.z-(velocity.z||0))**2<1e-6))return;
    // Opposite normals or support can prevent descent. A roof by itself cannot.
    s.contacts.push({ normal: { x: normal.x, y: normal.y, z: normal.z },
      velocity: { x: velocity.x || 0, y: velocity.y || 0, z: velocity.z || 0 }, kind });
  }
  function beginContacts(actors) {
    actors.forEach(actor => {
      const s = state(actor); s.contacts = [];
      s.start = { x: actor.position.x, y: actor.position.y, z: actor.position.z };
    });
  }
  function endContacts(actors, delta) {
    const dt = Math.max(0, Math.min(0.1, Number.isFinite(delta) ? delta : 0));
    for (const actor of actors) {
      const s = state(actor);
      if (s.locked || !s.start || !dt || !s.strong || document.hidden || window.DroneFlightInput?.paused
        || window.DroneMatchActions?.paused) { clearHold(s); continue; }
      const contacts = s.contacts.filter(c => c.kind !== 'floor');
      const support = contacts.some(c => c.normal.y > 0.25);
      const pinch = contacts.some((a, i) => contacts.slice(i + 1).some(b =>
        a.normal.x * b.normal.x + a.normal.y * b.normal.y + a.normal.z * b.normal.z < -0.3));
      // A side net can hold a tilted cage by friction, but a roof-only touch,
      // low hover, or a cage travelling along the wall is never enough.
      const side = contacts.some(c => Math.abs(c.normal.y) < 0.5);
      const velocity = contacts.length ? contacts.reduce((v, c) => ({x:v.x+c.velocity.x/contacts.length,
        y:v.y+c.velocity.y/contacts.length,z:v.z+c.velocity.z/contacts.length}), {x:0,y:0,z:0}) : {x:0,y:0,z:0};
      const movement = Math.hypot((actor.position.x-s.start.x)/dt-velocity.x,
        (actor.position.y-s.start.y)/dt-velocity.y,(actor.position.z-s.start.z)/dt-velocity.z);
      const measured = Math.hypot(actor.velocity.x-velocity.x,actor.velocity.y-velocity.y,actor.velocity.z-velocity.z);
      const blocked = (support || pinch || side) && Math.max(movement, measured) <= tuning.relativeSpeed;
      if (!blocked || !s.strong) { clearHold(s); continue; }
      s.contactSeconds += dt;
      if (s.contactSeconds + 1e-9 >= tuning.qualify) s.holdSeconds += dt;
      if (s.holdSeconds + 1e-9 >= tuning.hold) window.DroneFlightDynamics.stopMotors(actor, 'trapped');
    }
  }
  function neutral(actor, up, down) {
    const p = pilots().find(p => p.actor === actor), gp = state(actor).pad;
    return !up && !down && !actor.touchActive && !actor.touchActiveLeft
      && !Object.values(p?.keys || {}).some(Boolean)
      && (!state(actor).physicalPad || state(actor).physicalNeutral)
      && (!gp || !gp.connected || Math.max(Math.abs(gp.ly || 0), Math.abs(gp.lx || 0),
        Math.abs(gp.rx || 0), Math.abs(gp.ry || 0)) <= tuning.neutral);
  }
  function prepare(actor, up, down, intent) {
    const s = state(actor);
    s.down = Number.isFinite(intent) ? intent : Number(up) - Number(down);
    if (up || !down || window.DroneFlightInput?.paused) { clearHold(s); }
    else s.strong = s.down <= (s.strong ? tuning.keepDown : tuning.strongDown);
    if (!s.strong) clearHold(s);
    if (s.locked) { if (neutral(actor, up, down)) s.neutral = true; clearOutput(actor); }
  }
  function restartReason(actor, normalUp = false) {
    const s = state(actor);
    if (!s.locked) return '';
    if (document.hidden || window.DroneFlightInput?.paused || adapter?.setupOpen?.()) return '請先關閉設定並回到遊戲，再將操作回中';
    if (!s.neutral || !normalUp && !neutral(actor, false, false)) return '請先放開按鍵，將左右搖桿回中';
    const up = new THREE.Vector3(0,1,0).applyQuaternion(actor.mesh.quaternion);
    const body=window.DroneFlightDynamics.groundBody(actor)||window.DroneFlightDynamics.airBody(actor);
    if (actor.position.y > window.DroneFlightDynamics.groundHeight(actor) + 0.015
      || up.y < 0.95 || actor.velocity.length() > 0.12
      || body && Math.hypot(...body.angularVelocity) > 0.25) return '請先讓機體直立停穩在地面；受困時請使用重設';
    if (s.contacts.some(c => c.kind !== 'floor')) return '仍接觸障礙物，請先脫困或重設';
    return '';
  }
  function restart(actor, normalUp = false) {
    const s = state(actor);
    if (!s.locked) return true;
    const error = restartReason(actor, normalUp);
    if (error) { s.hint = error; updateUI(); return false; }
    s.locked = false; s.reason = ''; s.hint = ''; clearHold(s);
    window.DroneFlightInput?.reset(actor); clearOutput(actor);
    // The stopped body retains physical angular motion; stale controller yaw
    // must not inject a fresh turn when an upright resting aircraft is armed.
    actor.currentYawRate = 0;
    actor.gpAltArmed = false; actor.gpPrevLy = undefined;
    actor.targetAltitude = actor.position.y; revision++; updateUI(); return true;
  }
  function snapshot(actor) {
    const s = state(actor);
    return { locked:s.locked, reason:s.reason, qualified:s.contactSeconds >= tuning.qualify,
      holdSeconds:s.holdSeconds, contactSeconds:s.contactSeconds, neutral:s.neutral, hint:s.hint || '' };
  }
  function updateUI() {
    if(!panel)return;
    const controls=Array.from(document.querySelectorAll('.fc-flight-toolbar,.fc-camera-trigger'),el=>el.getBoundingClientRect()).filter(r=>r.width>0&&r.height>0);
    pilots().forEach((p,i)=>{
      const card=panel.children[i],s=state(p.actor);if(!card)return;
      const busy=s.locked || s.contactSeconds>0;
      card.hidden=!busy;
      if(!busy)return;
      if(pilots().length){
        // Keep active motor status in its owner's viewport and leave room
        // for the existing camera selector at the right edge.
        const rect=adapter.view?.(p.actor)?.element?.getBoundingClientRect();
        if(rect?.width>0 && rect.height>0){
          const width=Math.min(200,Math.max(80,rect.width-118));
          card.style.width=`${width}px`;card.style.maxWidth=`${width}px`;
          card.style.maxHeight=`${Math.max(36,rect.height-12)}px`;card.style.overflowY='auto';
          const height=card.getBoundingClientRect().height;
          const multi=pilots().length>1;
          const headers=multi?[]:Array.from(document.querySelectorAll('.fc-hud > header'),el=>el.getBoundingClientRect()).filter(r=>r.width>0&&r.bottom>rect.top&&r.top<rect.bottom);
          let left=multi?Math.max(rect.left+4,rect.right-112-width):rect.left+8;
          let top=multi?rect.top+6:Math.max(rect.top+80,...headers.map(r=>r.bottom+6));
          for(const control of controls){
            if(left+width<=control.left || left>=control.right || top+height<=control.top || top>=control.bottom)continue;
            if(control.left-width-6>=rect.left+4)left=control.left-width-6;
            else top=control.bottom+6;
          }
          card.style.left=`${left}px`;
          card.style.top=`${Math.max(rect.top+4,Math.min(top,rect.bottom-height-4))}px`;
        }
      }
      const label=s.locked?`${reasons[s.reason]||'馬達已停機'} · 馬達 0%`:s.contactSeconds>=tuning.qualify?`受困停機倒數 ${Math.max(0,tuning.hold-s.holdSeconds).toFixed(1)} 秒`:s.contactSeconds>0?'檢查下降是否受阻…':'馬達控制就緒';
      card.querySelector('[data-safety-status]').textContent=`P${i+1} · ${label}`;
      card.querySelector('[data-safety-hint]').textContent=s.locked?(s.hint||'先回中，再重新啟槳'):s.contactSeconds>0?'持續強下拉可停機；放開取消':'';
      card.querySelector('[data-restart-motors]').hidden=!s.locked;
      card.dataset.locked=String(s.locked);
    });
  }
  function mount(config) {
    adapter=config;
    if(panel)return;
    panel=document.createElement('aside');panel.className='fc-flight-safety fc-ui-region';panel.setAttribute('aria-label','馬達安全控制');panel.dataset.multi=String(pilots().length>1);
    pilots().forEach((p,i)=>{
      const card=document.createElement('div');card.className='fc-safety-card';
      const status=document.createElement('span');status.dataset.safetyStatus='';status.setAttribute('role','status');
      const restartButton=document.createElement('button');restartButton.dataset.restartMotors=String(i);restartButton.type='button';restartButton.textContent=`P${i+1} 重新啟槳`;restartButton.onclick=()=>{window.DroneFlightDynamics.beginTakeoff(p.actor);document.activeElement?.blur();};
      const hint=document.createElement('small');hint.dataset.safetyHint='';card.append(status,restartButton,hint);panel.append(card);
    });
    document.body.append(panel);
    const loop=()=>{updateUI();requestAnimationFrame(loop);};loop();
  }
  window.DroneFlightSafety=Object.freeze({tuning,stop,isStopped,fault,reset,cancel,contact,beginContacts,endContacts,
    prepare,restart,restartReason,snapshot,mount,
    setPad:(actor,input)=>{state(actor).pad=input;},
    observePad:(actor,pad,sticks)=>{const s=state(actor);s.physicalPad=!!pad;s.physicalNeutral=!pad || sticks.every(v=>Math.max(Math.abs(v.x),Math.abs(v.y))<=tuning.neutral);},
    get revision(){return revision;}});
})();
