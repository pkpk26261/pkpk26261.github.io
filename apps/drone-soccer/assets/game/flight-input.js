/* Shared macOS/Windows flight input. Loaded before each game's own handlers. */
(function () {
  'use strict';
  const states = new Set();
  const touchReleases = new Set();
  const padInputs = new Map();
  let unfocused = false;
  let releaseRevision = 0;
  function paused() {
    return unfocused || document.hidden || !!document.querySelector?.('dialog[open], :popover-open');
  }
  function releasePad(rig, input) {
    window.DroneFlightDynamics?.cancelBackflip?.(rig);
    input.connected = false;
    input.lx = input.ly = input.rx = input.ry = 0;
    rig.gpAltArmed = false;
    rig.gpPrevLy = undefined;
    reset(rig);
    if (rig.autoFlightState === 'none') rig.targetAltitude = Math.max(0.10, rig.position.y);
  }
  function release() {
    releaseRevision++;
    states.forEach(keys => Object.keys(keys).forEach(key => { keys[key] = false; }));
    touchReleases.forEach(reset => reset());
    padInputs.forEach((input, rig) => releasePad(rig, input));
  }
  function key(event) {
    // Retained as an inert adapter for historical engine handlers.
    return '';
  }
  // Stop old per-page shortcuts (including camera brackets and any-key start)
  // while leaving native field editing and accessible UI activation available.
  for (const type of ['keydown','keyup']) window.addEventListener(type,event=>{
    const field=event.target?.closest?.('input,textarea,select,[contenteditable="true"]');
    if (field || event.metaKey || event.ctrlKey || event.altKey || event.key==='Tab') return;
    event.stopImmediatePropagation();
    if (!event.target?.closest?.('button,a,summary,[role="button"]')) event.preventDefault();
  },true);
  function releaseOnBlur(keyStates) { keyStates.forEach(keys => states.add(keys)); }
  function releaseTouch(zone, reset) {
    // A cancelled gesture has the same motor/attitude effect as lifting the
    // final finger. Losing the page must also release controls and the knob.
    const end = event => { if (!event.targetTouches?.length) reset(); };
    zone.addEventListener('touchend', end);
    zone.addEventListener('touchcancel', end);
    for (const type of ['touchstart', 'touchmove']) zone.addEventListener(type, event => {
      if (!paused()) return;
      reset();
      event.preventDefault();
      event.stopImmediatePropagation();
    }, { capture: true, passive: false });
    touchReleases.add(reset);
  }
  window.addEventListener('blur', () => { unfocused = true; release(); });
  window.addEventListener('focus', () => { unfocused = false; });
  document.addEventListener('visibilitychange', () => { if (document.hidden) release(); });
  document.addEventListener('focusin', event => {
    if (event.target.closest?.('input,textarea,select,button,[contenteditable="true"],[role="tab"]')) release();
  });
  const padIdentities = new WeakMap();
  const liftStates = new WeakMap();
  const manualStates = new WeakMap();
  const brakeHeights = new WeakMap();
  const altitudeRates = new WeakMap();
  const liftCommands = new WeakMap();
  function keyboardMode(rig) {
    return window.DroneControlSettings?.radio === false && !rig.ai && !rig.aiAutoGear && !rig.touchActive && !rig.touchActiveLeft;
  }
  function reset(rig) {
    window.DroneFlightSafety?.cancel(rig);
    window.DroneFlightDynamics?.cancelLandingHold?.(rig);
    liftStates.delete(rig);
    manualStates.delete(rig);
    brakeHeights.delete(rig);
    altitudeRates.delete(rig);
    liftCommands.delete(rig);
  }
  function syncPad(rig, input, present, setupOpen, pad) {
    window.DroneFlightSafety?.setPad(rig, input);
    padInputs.set(rig, input);
    const controllerReady = !window.DroneControllerActions || window.DroneControllerActions.sync(rig, pad, setupOpen);
    const active = present && !setupOpen && !paused() && controllerReady;
    const identity = pad ? `${pad.bridged?'bridge':'browser'}:${pad.index}:${pad.id}:${pad.axes.length}` : null;
    if (input.connected !== active || padIdentities.get(rig) !== identity) {
      releasePad(rig, input);
    }
    padIdentities.set(rig, identity);
    return active;
  }
  // USB calibration owns its dead zone. Touch has a 3% radial/vertical dead
  // zone here. Apply the continuous response once, after removing that zone.
  const climbRate = 1;
  function axisResponse(value, deadzone = 0) {
    if (!Number.isFinite(value)) return 0;
    const dz = Math.max(0, Math.min(0.2, Number.isFinite(deadzone) ? deadzone : 0));
    const magnitude = Math.min(1, Math.abs(value));
    if (magnitude <= dz) return 0;
    const v = Math.sign(value) * (magnitude - dz) / (1 - dz);
    return 0.7 * v + 0.3 * v * v * v;
  }
  function stickCommand(rig) {
    if (rig.controllerBrakeHeld) return {pitch:0,roll:0,amount:0};
    const finite = v => Number.isFinite(v) ? Math.max(-1, Math.min(1, v)) : 0;
    // DOM touch coordinates grow to screen-right. The engine's local +X is
    // pilot-left; USB is converted at its boundary and touch is converted here.
    let pitch = finite(rig.input?.pitch), roll = finite(rig.input?.roll) * (rig.touchActive ? -1 : 1);
    ({pitch,roll}=window.DroneControllerActions?.transformCommand?.(rig,{pitch,roll}) || {pitch,roll});
    const length = Math.hypot(pitch, roll);
    const amount = axisResponse(Math.min(1, length), rig.touchActive ? 0.03 : 0);
    return { pitch: length ? pitch / length * amount : 0,
      roll: length ? roll / length * amount : 0, amount };
  }
  function verticalProfile(rig) {
    const gear = window.DroneControlSettings?.gearFor(rig);
    if (!keyboardMode(rig)) return { climb: gear?.climb ?? 1.05, descent: gear?.descent ?? 1, ramp: gear?.liftRamp ?? 0.4 };
    const airframe = window.DroneFlightDynamics?.airframe;
    const limit = gear?.powerLimit ?? 1;
    return { climb: (airframe?.maxClimbSpeed ?? 3.5) * limit, descent: (airframe?.maxLevelSpeed ?? 3) * limit, ramp: gear?.liftRamp ?? 0.3 };
  }
  function liftResponse(value, deadzone = 0.03) { return axisResponse(value, deadzone); }
  const timestep = delta => Math.max(0, Math.min(0.1, Number.isFinite(delta) ? delta : 1 / 60));
  function ramp(previous, target, delta, response) {
    // Match the horizontal acceleration of each gear; centre brakes immediately.
    if (!target) return 0;
    if (previous * target < 0) previous = 0;
    return previous + (target - previous) * (1 - Math.exp(-timestep(delta) / response));
  }
  function holdLift(rig, value, delta, source = 'pad') {
    if (rig.controllerBrakeHeld) value=0;
    if (window.DroneFlightSafety?.isStopped(rig)) return 0;
    if (value <= 0 && window.DroneFlightDynamics?.landingConfirmed?.(rig)) return 0;
    let states = liftStates.get(rig);
    if (!states) { states = {}; liftStates.set(rig, states); }
    const profile = verticalProfile(rig);
    const previous = states[source] || 0;
    const response = source === 'keyboard' && keyboardMode(rig)
      ? (Number.isFinite(value) ? Math.max(-1, Math.min(1, value)) : 0)
      : liftResponse(value, source === 'touch' ? 0.03 : 0);
    const target = response * (response > 0 ? profile.climb : profile.descent);
    if (previous && (!target || previous * target < 0) && rig.autoFlightState === 'none') rig.targetAltitude = Math.max(0.1, rig.position.y);
    states[source] = ramp(previous, target, delta, profile.ramp);
    if(target) {
      let command=states[source]/(target>0?profile.climb:profile.descent);
      if(Math.abs(command)>0.9999)command=Math.sign(command);
      const current=liftCommands.get(rig)||0;
      if(Math.abs(command)>Math.abs(current))liftCommands.set(rig,command);
    }
    if (!target) return 0;
    // Integrate the average command over this frame instead of treating the
    // endpoint as a whole-frame speed. This keeps height consistent at 30–120 Hz.
    const dt = timestep(delta), start = previous * target < 0 ? 0 : previous;
    return dt ? target + (start - target) * (-Math.expm1(-dt / profile.ramp)) * profile.ramp / dt : start;
  }
  // Arena geometry for AI/recovery and diagnostics only, never a pilot ceiling.
  function altitudeLimit() { return window.DroneArenaCollision?.bounds?.y[1] ?? 3; }
  function beginFrame(rig) { altitudeRates.set(rig, 0); liftCommands.delete(rig); }
  function throttleDemand(rig) {
    if (keyboardMode(rig)) return null;
    const command = liftCommands.get(rig);
    if (!command) return null;
    const c = rig.constants, limit = window.DroneFlightDynamics?.powerLimit(rig) ?? 1;
    const vertical = Math.max(0.08, Math.cos(rig.currentPitch || 0) * Math.cos(rig.currentRoll || 0));
    const hover = Math.min(limit, c.mass * c.gravity / (c.maxThrottleThrust * vertical));
    return Math.max(0, Math.min(limit, hover + command * (command > 0 ? limit - hover : hover)));
  }
  function motionCommand(rig) {
    const pitch = Math.max(-1, Math.min(1, rig.input?.pitch || 0));
    const roll = Math.max(-1, Math.min(1, rig.input?.roll || 0));
    const lift = liftCommands.get(rig) || 0;
    const scale = Math.max(1, Math.hypot(pitch, roll, lift));
    return { pitch: pitch / scale, roll: roll / scale, lift: lift / scale };
  }
  function flightOutput(rig) {
    if (window.DroneFlightSafety?.isStopped(rig)) return 0;
    if (!keyboardMode(rig)) return rig.input?.throttle || 0;
    if (rig.position.y <= 0.11 && rig.targetAltitude <= 0.15) return 0;
    const command = motionCommand(rig);
    const limit = window.DroneControlSettings?.gearFor(rig)?.powerLimit ?? 1;
    return Math.hypot(command.pitch, command.roll, command.lift) * limit;
  }
  function changeAltitude(rig, lift, delta) {
    // Bound accumulated demand while a net/obstacle blocks motion. Releasing
    // or reversing the stick then takes effect at the aircraft's actual height.
    const dt = timestep(delta), before = rig.targetAltitude;
    rig.targetAltitude = Math.max(0.1, Math.min(rig.position.y + 0.3,
      before + lift * climbRate * dt));
    // Velocity feed-forward must survive the nearby altitude setpoint's windup
    // bound. Net contact is resolved by physics; it must not demand raw max thrust.
    const scale = Math.max(1, Math.hypot(rig.input?.pitch || 0, rig.input?.roll || 0, liftCommands.get(rig) || 0));
    const requested = lift * climbRate / scale;
    const actual = dt ? (rig.targetAltitude - before) / dt : 0;
    altitudeRates.set(rig, keyboardMode(rig) ? requested : lift > 0 ? Math.max(0, Math.min(lift * climbRate, actual)) : Math.min(0, Math.max(lift * climbRate, actual)));
  }
  function altitudeCorrection(rig, error, kp, kd) {
    // Feed the commanded vertical speed into the existing altitude controller.
    // This reaches the gear's climb speed without accumulating a distant target
    // above a blocking net; releasing still holds the aircraft's current height.
    const rate = altitudeRates.get(rig) || 0;
    if (!keyboardMode(rig)) return error * kp - rig.velocity.y * kd + rate * (kd + rig.constants.linearDrag);
    const drag = window.DroneFlightDynamics?.verticalDrag(rig) ?? rig.constants.linearDrag;
    // Moving setpoints track velocity. A released stick restores position hold.
    return rate ? (rate - rig.velocity.y) * kd + rate * drag : error * kp - rig.velocity.y * kd;
  }
  function keyboardLift(rig, keys, delta, backup = true) {
    const up = !!(keys.w || backup && keys[' ']);
    const down = !!(keys.s || backup && keys.control);
    const lift = holdLift(rig, Number(up) - Number(down), delta, 'keyboard');
    if (lift) {
      changeAltitude(rig, lift, delta);
      rig.autoFlightState = 'none';
    }
  }
  function keyboardAttitude(rig, keys) {
    // Opposing keys cancel. A keyboard press is the same endpoint as a full
    // stick deflection; the shared attitude controller provides the ramp.
    // The nose is local +Z, so pilot-right is local -X. Positive roll input
    // drives local +X; convert the keyboard's lateral actions at this boundary.
    for (const [axis, positive, negative, touch] of [
      ['pitch', 'arrowup', 'arrowdown', rig.touchActive],
      ['roll', 'arrowleft', 'arrowright', rig.touchActive],
      ['yaw', 'd', 'a', rig.touchActiveLeft]
    ]) {
      if (keys[positive] || keys[negative] || !touch) rig.input[axis] = Number(!!keys[positive]) - Number(!!keys[negative]);
    }
  }
  function manualThrottle(rig, ly, delta) {
    if (rig.controllerBrakeHeld) {
      if(rig.position.y<=.11)return 0;
      if(!brakeHeights.has(rig))brakeHeights.set(rig,rig.targetAltitude);
      const c=rig.constants,vertical=Math.max(.08,Math.cos(rig.currentPitch||0)*Math.cos(rig.currentRoll||0));
      const force=c.mass*c.gravity+altitudeCorrection(rig,brakeHeights.get(rig)-rig.position.y,11.25,6);
      return Math.max(0,Math.min(window.DroneFlightDynamics?.powerLimit(rig)??1,force/(c.maxThrottleThrust*vertical)));
    }
    brakeHeights.delete(rig);
    if(rig.controllerBackflipActive)return (window.DroneFlightDynamics?.powerLimit(rig)??1)*Math.max(0,Math.cos(rig.currentPitch||0));
    const target = liftResponse(ly, 0);
    if (rig.position.y <= 0.11 && ly <= 0.05) { manualStates.delete(rig); return 0; }
    const c = rig.constants;
    const hoverForce = c.gravity * c.mass;
    const profile = verticalProfile(rig);
    const lift = ramp(manualStates.get(rig) || 0, target, delta, profile.ramp);
    manualStates.set(rig, lift);
    // Manual oil/power control spans the actual motor range: centre hovers,
    // fully up requests this gear's maximum, fully down requests zero.
    const vertical = Math.max(0.25, Math.cos(rig.currentPitch || 0) * Math.cos(rig.currentRoll || 0));
    const limit = window.DroneFlightDynamics?.powerLimit(rig) ?? 1;
    const hover = Math.min(limit, hoverForce / (vertical * c.maxThrottleThrust));
    return Math.max(0, Math.min(limit, hover + lift * (lift >= 0 ? limit - hover : hover)));
  }
  window.DroneFlightInput = Object.freeze({ key, release, releaseOnBlur, releaseTouch, syncPad, manualThrottle, keyboardMode, throttleDemand, motionCommand, flightOutput, climbRate, axisResponse, stickCommand, liftResponse, holdLift, altitudeLimit, beginFrame, changeAltitude, altitudeCorrection, keyboardLift, keyboardAttitude, reset, get paused() { return paused(); }, get releaseRevision() { return releaseRevision; } });
})();
