/* Stabilized flight: momentum remains in world space; recentering a stick is
   a gradual attitude/braking command, never a velocity reset. SI units. */
(() => {
  'use strict';
  const clamp = (n, lo, hi) => Math.max(lo, Math.min(hi, n));
  // Game calibration: 3 m/s horizontal/descent, slightly faster 3.5 m/s climb.
  const airframe = Object.freeze({ diameter: 0.20, mass: 0.30, speedScale: 0.25, maxLevelSpeed: 3, maxClimbSpeed: 3.5, maxBank: 0.4,
    // X-layout rotor arm projected onto pitch/roll (0.28 * 0.23 / sqrt(2)).
    motorLever: 0.0455, recoveryFrequency: 10 });
  // Effective contact pairs for the hard-plastic protective cage. These are
  // tunable game values, not measured constants for every rubber/polymer.
  // Rubber compression/hysteresis is represented by a dissipative impulse;
  // rope compliance is integrated separately by DroneRopeNet.
  const contactMaterials = Object.freeze({
    rubber: Object.freeze({ restitution: 0.05, friction: 0.5 }),
    rope: Object.freeze({ restitution: 0, friction: 0.28 }),
    plastic: Object.freeze({ restitution: 0.22, friction: 0.24 })
  });
  const horizontalLinearDrag = 0.12;
  const horizontalQuadraticDrag = (airframe.mass * 9.8 * Math.tan(0.4)
    - horizontalLinearDrag * airframe.maxLevelSpeed) / airframe.maxLevelSpeed ** 2;
  function powerLimit(actor) {
    return clamp(window.DroneControlSettings?.gearFor(actor)?.powerLimit ?? 1, 0, 1);
  }
  function throttleForVerticalForce(actor, force) {
    if (actor.controllerBackflipActive) {
      // Cut collective through the inverted half, then recover with upright
      // thrust. The maneuver still integrates forces and arena collisions.
      return powerLimit(actor)*Math.max(0,Math.cos(actor.currentPitch || 0));
    }
    // Stick input requests velocity through the altitude controller. Raw motor
    // saturation here used to bypass that controller and make every gear climb
    // near the global speed cap, regardless of the requested climb speed.
    const demand = window.DroneFlightInput?.throttleDemand(actor);
    // An attitude-priority mixer can lower collective thrust to leave room
    // for differential rotor torque after impact. Merely clipping every rotor
    // at the collective ceiling gives zero recovery authority when saturated,
    // so a tilted cage can remain pressed against the net indefinitely.
    const limit = Math.max(0, powerLimit(actor) - (airBodies.get(actor)?.thrustReserve || 0));
    if (demand !== null && demand !== undefined) return clamp(demand, 0, limit);
    const vertical = Math.cos(actor.currentPitch || 0) * Math.cos(actor.currentRoll || 0);
    return clamp(force / (actor.constants.maxThrottleThrust * Math.max(0.08, vertical)), 0, limit);
  }
  function driveTilt(actor, command = 1) {
    const c = actor.constants || {}, weight = (c.mass ?? airframe.mass) * (c.gravity ?? 9.8);
    const hover = weight / (c.maxThrottleThrust ?? 13.5);
    const reserve = clamp((powerLimit(actor) - hover) / Math.max(0.01, 1 - hover), 0, 1);
    // Normal flight needs a modest bank at this game's 3 m/s top speed.
    // Forward stick requests part of the available maneuvering power; it
    // does not mean using 100% motor thrust while maintaining level flight.
    const fullTilt = Math.min(0.4, c.maxTilt ?? 0.4) * reserve;
    // At a fixed height, horizontal force is weight * tan(bank). Interpolate
    // force, then convert back to bank so a small stick adds a small force.
    return Math.atan(Math.tan(fullTilt) * clamp(command, 0, 1));
  }
  function horizontalDrag(actor) {
    return horizontalLinearDrag + horizontalQuadraticDrag * Math.hypot(actor.velocity.x, actor.velocity.z);
  }
  function levelSpeed(actor) {
    if (window.DroneFlightInput.keyboardMode(actor)) return airframe.maxLevelSpeed * powerLimit(actor);
    const c = actor.constants || {};
    const weight = (c.mass ?? airframe.mass) * (c.gravity ?? 9.8);
    const horizontal = weight * Math.tan(driveTilt(actor));
    return (Math.sqrt(horizontalLinearDrag ** 2 + 4 * horizontalQuadraticDrag * horizontal) - horizontalLinearDrag) / (2 * horizontalQuadraticDrag);
  }
  function verticalDrag(actor) {
    const coefficient=actor.constants?.linearDrag??1.35;
    return window.DroneFlightInput.keyboardMode(actor)?Math.min(coefficient,0.8):coefficient;
  }
  function limitSpeed(actor) {
    if (!window.DroneFlightInput.keyboardMode(actor)) { actor.velocity.clampLength(0, airframe.maxLevelSpeed); return; }
    const v = actor.velocity, vertical = v.y > 0 ? airframe.maxClimbSpeed : airframe.maxLevelSpeed;
    const scale = Math.hypot(v.x / airframe.maxLevelSpeed, v.y / vertical, v.z / airframe.maxLevelSpeed);
    if (scale > 1) v.multiplyScalar(1 / scale);
  }
  function accelerate(actor, acceleration, delta) {
    actor.velocity.addScaledVector(acceleration, delta);
    // A direction-aware vector limit bounds keyboard climb at 3.5 m/s and
    // horizontal/descent at 3 m/s. USB keeps its original 3 m/s vector limit;
    // the contact solver applies its physical impulse separately afterwards.
    limitSpeed(actor);
  }
  let contactFrame = null;
  function beginContacts(actors, delta) {
    window.DroneFlightSafety?.beginContacts(actors);
    const starts = new WeakMap();
    actors.forEach(actor => { if (actor?.position) starts.set(actor, actor.position.clone()); });
    contactFrame = { starts, delta: clamp(Number.isFinite(delta) ? delta : 0, 0, 0.1), pairs: new WeakMap() };
  }
  const takeoffAltitude = 0.5;
  const groundBodies = new WeakMap();
  const airBodies = new WeakMap();
  const collisionRevisions = new WeakMap();
  const pairRevisions = new WeakMap();
  const backflips = new WeakMap();
  function cancelBackflip(actor) {
    if (!backflips.has(actor)) return;
    backflips.delete(actor);actor.controllerBackflipActive=false;
    actor.currentPitch=Math.atan2(Math.sin(actor.currentPitch||0),Math.cos(actor.currentPitch||0));
  }
  function beginBackflip(actor) {
    if (!actor || actor.controllerBackflipActive || actor.controllerBrakeHeld ||
        window.DroneFlightInput?.paused || window.DroneFlightSafety?.isStopped(actor) ||
        actor.position.y<.7 || groundBodies.has(actor) || airBodies.has(actor) ||
        Math.abs(actor.currentPitch||0)>.45 || Math.abs(actor.currentRoll||0)>.45) return false;
    backflips.set(actor,{elapsed:0,pitch:actor.currentPitch||0,revision:window.DroneFlightInput?.releaseRevision});
    actor.controllerBackflipActive=true;actor.autoFlightState='none';
    return true;
  }
  function advanceTumble(actor, body, delta, desiredUp = new THREE.Vector3(0, 1, 0), desiredYawRate = 0) {
    const dt = clamp(delta, 0, 0.05);
    advanceImpact(actor, dt);
    if ((stateOf(actor)?.age || 0) > 2.5) impacts.delete(actor);
    const steps = Math.max(1, Math.ceil(dt * 240)), h = dt / steps;
    for (let i = 0; i < steps; i++) {
      body.age += h;
      body.contactAge = (body.contactAge || 0) + h;
      const up = new THREE.Vector3(0, 1, 0).applyQuaternion(body.q);
      // The gyro is already running when cages touch. Motor response ramps
      // continuously; a bump must not disable steering for a fixed interval.
      const authority = 0.45 + 0.55 * -Math.expm1(-body.age / 0.06);
      const correction = new THREE.Vector3().crossVectors(up, desiredUp);
      if (up.dot(desiredUp) < -0.98 && correction.length() < 0.05) correction.set(1, 0, 0);
      const c = actor.constants || {}, mass = c.mass || airframe.mass;
      const maximum = (c.maxThrottleThrust ?? 13.5) * powerLimit(actor);
      const thrust = window.DroneFlightSafety?.isStopped(actor) ? 0 : aiActors.has(actor)
        ? body.collectiveThrust ?? Math.min(maximum, mass * (c.gravity ?? 9.8) / Math.max(0.35, up.y))
        : clamp(actor.input?.throttle ?? mass * (c.gravity ?? 9.8) / (c.maxThrottleThrust ?? 13.5), 0, powerLimit(actor)) * (c.maxThrottleThrust ?? 13.5);
      // Opposing rotor pairs redistribute the current collective thrust.
      // Both a stopped motor set and a fully saturated set lack headroom;
      // recovery cannot supply unlimited torque or silently level the cage.
      const headroom = Math.max(0, Math.min(thrust, maximum - thrust));
      const maxAcceleration = headroom * airframe.motorLever / (mass * 0.0065);
      const yawRate = body.omega.dot(up);
      const tiltRate = body.omega.clone().addScaledVector(up, -yawRate);
      const frequency = airframe.recoveryFrequency;
      const acceleration = correction.multiplyScalar(frequency ** 2)
        .addScaledVector(tiltRate, -1.8 * frequency).addScaledVector(up, 4 * (desiredYawRate - yawRate));
      // Next collective solve reserves at most one quarter of this gear's
      // output for the requested torque. Actual torque below still depends
      // on the delivered thrust/headroom; stopped motors get no authority.
      body.thrustReserve = Math.min(powerLimit(actor) * 0.25,
        authority * acceleration.length() * mass * 0.0065 / (airframe.motorLever * (c.maxThrottleThrust ?? 13.5)));
      if (acceleration.length() > maxAcceleration) acceleration.setLength(maxAcceleration);
      body.omega.addScaledVector(acceleration, authority * h);
      body.omega.multiplyScalar(Math.exp(-1.6 * h));
      const angle = body.omega.length() * h;
      if (angle) body.q.premultiply(new THREE.Quaternion().setFromAxisAngle(body.omega.clone().normalize(), angle)).normalize();
    }
    const up = new THREE.Vector3(0, 1, 0).applyQuaternion(body.q);
    groundPose(actor, body);
    actor.currentYawRate = -body.omega.dot(up);
    // Keep one rigid body throughout a resting/sliding contact. Deleting it
    // between tiny impulses restarts the motor delay when the next rib touches.
    if (body.age > 0.25 && body.contactAge > 0.12 && up.dot(desiredUp) > 0.9995 && body.omega.length() < 0.12) {
      airBodies.delete(actor);
      if (stateOf(actor)?.physical) impacts.delete(actor);
    }
  }
  let cageShape;
  function support(quaternion, normal = new THREE.Vector3(0, 1, 0)) {
    cageShape ||= window.DroneAppearance.groundShape(0.1);
    // Transform the normal once instead of allocating/rotating every rib at
    // every solver iteration. Dot products and patch weights are unchanged.
    const localNormal=normal.clone().applyQuaternion(quaternion.clone().conjugate());
    let lowest=Infinity;
    for(const p of cageShape.points)lowest=Math.min(lowest,p.dot(localNormal));
    // Average a resting contact patch to avoid switching equal-height ribs.
    const point=new THREE.Vector3();let count=0;
    for(const p of cageShape.points)if(p.dot(localNormal)<lowest+.0006){point.add(p);count++;}
    point.multiplyScalar(1/count).applyQuaternion(quaternion);
    point.addScaledVector(normal, -cageShape.tubeRadius);
    return { point, height: -lowest + cageShape.tubeRadius };
  }
  function groundPose(actor, body) {
    const euler = new THREE.Euler().setFromQuaternion(body.q, 'YXZ');
    actor.currentPitch = euler.x; actor.currentRoll = euler.z;
    if (actor.rotation) actor.rotation.y = euler.y;
    actor.mesh.quaternion.copy(body.q);
    actor.mesh.position.copy(actor.position);
    clearHover(actor);
  }
  function groundHeight(actor) {
    const body = groundBodies.get(actor);
    return body ? support(body.q).height : 0.1;
  }
  function groundContact(actor, delta) {
    if (!actor?.mesh) return false;
    const dt = clamp(Number.isFinite(delta) ? delta : 0, 0, 0.1);
    let body = groundBodies.get(actor);
    if (!body && actor.position.y > 0.115) return false;
    window.DroneFlightSafety?.contact(actor, {x:0,y:1,z:0}, undefined, 'floor');
    const q = body?.q || airBodies.get(actor)?.q || (aiActors.has(actor) || actor.ai ? actor.mesh.quaternion : new THREE.Quaternion().setFromEuler(new THREE.Euler(
      actor.currentPitch ?? actor.mesh.rotation.x, actor.rotation?.y ?? actor.mesh.rotation.y,
      actor.currentRoll ?? actor.mesh.rotation.z, 'YXZ')));
    const up = new THREE.Vector3(0, 1, 0).applyQuaternion(q);
    const speed = Math.hypot(actor.velocity.x, actor.velocity.z);
    if (!body && !airBodies.has(actor) && speed <= 0.12 && actor.velocity.y >= -0.4 && up.y >= 0.96) {
      if (actor.position.y < 0.1) { actor.position.y = 0.1; actor.velocity.y = 0; }
      const decay = Math.exp(-1.8 * dt);
      actor.velocity.x *= decay; actor.velocity.z *= decay;
      return false;
    }
    let contact = support(q);
    if (actor.position.y > Math.max(0.105, contact.height + 0.012)) {
      if (body) {
        groundPose(actor, body);
        const up = new THREE.Vector3(0, 1, 0).applyQuaternion(body.q);
        if (up.y < 0.95 || body.omega.length() > 0.5) airBodies.set(actor, { q: body.q.clone(), omega: body.omega.clone(), age: 0 });
        groundBodies.delete(actor);
      }
      return false;
    }
    if (!body && (speed > 0.12 || actor.velocity.y < -0.4 || up.y < 0.96)) {
      body = { q: q.clone(), omega: airBodies.get(actor)?.omega.clone() || new THREE.Vector3() };
      groundBodies.set(actor, body);
      airBodies.delete(actor);
      impacts.delete(actor);
    }
    if (!body) {
      if (actor.position.y < 0.1) { actor.position.y = 0.1; actor.velocity.y = 0; }
      const decay = Math.exp(-1.8 * dt);
      actor.velocity.x *= decay; actor.velocity.z *= decay;
      return false;
    }
    const mass = actor.constants?.mass || 0.3, gravity = actor.constants?.gravity || 9.8;
    const inertia = mass * 0.1 ** 2 * 0.65;
    const steps = Math.max(1, Math.ceil(dt / (1 / 240))), h = dt / steps;
    for (let i = 0; i < steps; i++) {
      contact = support(body.q);
      const normalImpulse = mass * (gravity * h + (i === 0 ? Math.max(0, -actor.velocity.y) : 0));
      const r = contact.point.clone();
      // A slight battery offset; cage facets still permit sideways/inverted rest.
      r.sub(new THREE.Vector3(0, -0.004, 0).applyQuaternion(body.q));
      body.omega.add(new THREE.Vector3(-r.z, 0, r.x).multiplyScalar(normalImpulse / inertia));
      const slip = new THREE.Vector3(actor.velocity.x, 0, actor.velocity.z)
        .add(new THREE.Vector3().crossVectors(body.omega, r));
      slip.y = 0;
      const impulse = slip.multiplyScalar(-1 / (1 / mass + contact.height ** 2 / inertia));
      const maxFriction = 0.45 * normalImpulse;
      if (impulse.length() > maxFriction) impulse.setLength(maxFriction);
      actor.velocity.addScaledVector(impulse, 1 / mass);
      body.omega.add(new THREE.Vector3().crossVectors(r, impulse).multiplyScalar(1 / inertia));
      body.omega.multiplyScalar(Math.exp(-1.8 * h));
      if (body.omega.length()) body.omega.setLength(Math.max(0, body.omega.length() - 0.6 * h));
      actor.velocity.x *= Math.exp(-0.35 * h); actor.velocity.z *= Math.exp(-0.35 * h);
      const horizontal = Math.hypot(actor.velocity.x, actor.velocity.z);
      const resistance = horizontal ? Math.max(0, 1 - 0.04 * h / horizontal) : 0;
      actor.velocity.x *= resistance; actor.velocity.z *= resistance;
      const angle = body.omega.length() * h;
      if (angle > 0) body.q.premultiply(new THREE.Quaternion().setFromAxisAngle(body.omega.clone().normalize(), angle)).normalize();
    }
    actor.position.y = Math.max(actor.position.y, support(body.q).height);
    if (actor.velocity.y < 0) actor.velocity.y = actor.velocity.y < -0.8 ? -actor.velocity.y * 0.08 : 0;
    groundPose(actor, body);
    return true;
  }
  function advanceGroundAI(actor, delta) {
    const body = groundBodies.get(actor);
    if (!body) return false;
    const dt = clamp(delta, 0, 0.05);
    const up = new THREE.Vector3(0, 1, 0).applyQuaternion(body.q);
    const lifting = actor.altitudeState === 'takingoff' || actor.targetAltitude > 0.2;
    const gravity = actor.constants?.gravity || 9.8;
    const maxAcceleration = powerLimit(actor) * (actor.constants?.maxThrottleThrust ?? 13.5) / (actor.constants?.mass ?? 0.3);
    const acceleration = lifting ? Math.min(gravity * 1.12, maxAcceleration) : 0;
    actor.motorThrottle = acceleration * (actor.constants?.mass ?? .3) / (actor.constants?.maxThrottleThrust ?? 13.5);
    if (actor.input) actor.input.throttle = actor.motorThrottle;
    actor.velocity.addScaledVector(up, acceleration * dt);
    actor.velocity.y -= gravity * dt;
    actor.position.addScaledVector(actor.velocity, dt);
    groundContact(actor, dt);
    return true;
  }
  function advanceAirAI(actor, delta) {
    const body = airBodies.get(actor);
    if (!body) return false;
    const dt = clamp(delta, 0, 0.05), gravity = actor.constants?.gravity || 9.8;
    // Recover against the actual drift/vertical speed, within motor authority.
    // Both attacker and guard wrappers use this path while the body tumbles.
    const station = window.DroneAITactics?.guardRecoveryVelocity?.(actor, dt)
      || window.DroneAITactics?.attackRecoveryVelocity?.(actor, dt);
    const targetAltitude = clamp(Math.max(0.5, station?.altitude ?? (actor.targetAltitude || actor.position.y)), 0.5, window.DroneFlightInput.altitudeLimit() - 0.1);
    const vertical = clamp(gravity + (targetAltitude - actor.position.y) * 3.5 - actor.velocity.y * 3, 0, gravity * 1.6);
    const steeringGain = station ? Math.min(6, actor.constants?.accel ?? 6) : 1.8;
    const desiredUp = new THREE.Vector3(((station?.x || 0) - actor.velocity.x) * steeringGain,
      Math.max(gravity * 0.5, vertical), ((station?.z || 0) - actor.velocity.z) * steeringGain);
    if (station) {
      // Keep piloting the role's target during a physical tumble.
      // The normal gyro delay, torque headroom, tilt and motor limits apply.
      const horizontal = Math.hypot(desiredUp.x, desiredUp.z);
      const limit = desiredUp.y * Math.tan(Math.min(.4, actor.constants?.maxTilt ?? .4));
      if (horizontal > limit) { desiredUp.x *= limit / horizontal; desiredUp.z *= limit / horizontal; }
    }
    desiredUp.normalize();
    const beforeUp = new THREE.Vector3(0, 1, 0).applyQuaternion(body.q);
    const mass = actor.constants?.mass ?? 0.3;
    const maxAcceleration = Math.max(0, powerLimit(actor) - (body.thrustReserve || 0))
      * (actor.constants?.maxThrottleThrust ?? 13.5) / mass;
    const thrust = Math.min(maxAcceleration, beforeUp.y > 0.15
      ? clamp(vertical / Math.max(0.35, beforeUp.y), 0, gravity * 1.6) : gravity * 0.15);
    // Hold one delivered collective for this substep. The torque solver and
    // translation must use the same motor allocation, including when inverted.
    body.collectiveThrust = thrust * mass;
    actor.motorThrottle = body.collectiveThrust / (actor.constants?.maxThrottleThrust ?? 13.5);
    if (actor.input) actor.input.throttle = actor.motorThrottle;
    advanceTumble(actor, body, dt, desiredUp);
    const up = new THREE.Vector3(0, 1, 0).applyQuaternion(body.q);
    // The same tilted thrust and weight that act on human-controlled aircraft.
    const resistance=drag(actor).multiplyScalar(1/mass);
    actor.velocity.addScaledVector(up, thrust * dt);
    actor.velocity.y -= gravity * dt;
    actor.velocity.addScaledVector(resistance,dt);
    actor.position.addScaledVector(actor.velocity, dt);
    groundContact(actor, dt);
    actor.mesh.position.copy(actor.position);
    return true;
  }
  // A rolling cage is grounded even when it still has horizontal velocity.
  function createGroundFailure({ actors, active = () => true, onFailure }) {
    let states = new WeakMap(), finished = false;
    const list = () => typeof actors === 'function' ? actors() : actors;
    function resetFailure() { states = new WeakMap(); finished = false; }
    function beginFrame() {
      if (finished || !active()) return;
      for (const actor of list()) {
        if (!actor?.mesh) continue;
        let state = states.get(actor);
        if (!state) { state = { flown: false, landing: false, elapsed: 0 }; states.set(actor, state); }
        // Latch deliberate landing before legacy AI handlers clear their state.
        if (window.DroneFlightSafety?.fault(actor)) state.landing = false;
        else if (actor.autoFlightState === 'landing' || actor.altitudeState === 'landing'
          || actor.position.y > 0.2 && actor.targetAltitude <= 0.15) state.landing = true;
      }
    }
    function update(delta) {
      if (finished) return true;
      if (!active()) { states = new WeakMap(); return false; }
      if (document.hidden || window.DroneMatchActions?.paused) return false;
      const dt = clamp(Number.isFinite(delta) ? delta : 0, 0, 0.1);
      const failed = [];
      for (const actor of list()) {
        const state = states.get(actor);
        if (!state || !actor.mesh) continue;
        if (actor.position.y > groundHeight(actor) + 0.025) {
          if (actor.position.y > 0.2) state.flown = true;
          state.elapsed = 0;
          if (actor.autoFlightState !== 'landing' && actor.altitudeState !== 'landing'
            && !(actor.targetAltitude <= 0.15)) state.landing = false;
          continue;
        }
        if (!state.flown) continue;
        const tipped = Math.cos(actor.mesh.rotation.x || 0) * Math.cos(actor.mesh.rotation.z || 0) < 0.5;
        const lifting = actor.autoFlightState === 'takingoff' || actor.altitudeState === 'takingoff'
          || actor.targetAltitude > 0.2;
        if (window.DroneFlightSafety?.fault(actor)) state.landing = false;
        if (state.landing && !tipped && !lifting) { state.elapsed = 0; continue; }
        if (lifting || tipped) state.landing = false;
        state.elapsed += dt;
        if (state.elapsed + 1e-8 >= 5) failed.push(actor);
      }
      if (failed.length) {
        finished = true;
        onFailure(failed);
      }
      return finished;
    }
    return Object.freeze({ beginFrame, update, reset: resetFailure, get finished() { return finished; } });
  }
  const groundStates = new WeakMap();
  const minimumFlightAltitude = 0.2;
  const landingHoldSeconds = 2;
  function groundState(actor) {
    if (!groundStates.has(actor)) groundStates.set(actor, {
      armed: false, upHeld: false, canLift: false, airborne: false,
      down: false, delta: 0, downSeconds: 0, floorHold: false, holdLanding: false,
      releaseRevision: window.DroneFlightInput?.releaseRevision
    });
    return groundStates.get(actor);
  }
  function landingConfirmed(actor) { return groundStates.get(actor)?.holdLanding === true; }
  function cancelLandingHold(actor) {
    const state = groundStates.get(actor);
    if (state) state.downSeconds = 0;
  }
  function stopMotors(actor, reason = 'landed') {
    const state = groundState(actor);
    state.armed = state.canLift = false;
    state.airborne = state.floorHold = state.holdLanding = false;
    state.downSeconds = 0;
    actor.targetAltitude = 0.1;
    actor.autoFlightState = 'none';
    if (actor.input) actor.input.throttle = 0;
    window.DroneFlightSafety?.stop(actor, reason);
  }
  function prepareGround(actor, up, down, delta = 0, intent = actor.touchActiveLeft ? actor.touchAltitudeInput : Number(!!up) - Number(!!down)) {
    if(actor.controllerBrakeHeld){up=false;down=false;intent=0;}
    window.DroneFlightInput?.beginFrame(actor);
    const state = groundState(actor);
    if(actor.controllerBrakeHeld){state.floorHold=state.holdLanding=false;state.downSeconds=0;}
    const safety = window.DroneFlightSafety;
    safety?.prepare(actor, up, down, intent);
    if (safety?.isStopped(actor)) {
      // Fault stops require explicit arming. Deliberate landing retains the
      // existing first upward press = idle, release, second press = lift.
      if (safety.snapshot(actor).reason === 'landed' && up && !state.upHeld
        && safety.restart(actor, true)) { state.armed = true; state.canLift = false; }
      state.upHeld = !!up;
      return;
    }
    state.delta = clamp(Number.isFinite(delta) ? delta : 0, 0, 0.1);
    const revision = window.DroneFlightInput?.releaseRevision;
    if (state.releaseRevision !== revision) state.downSeconds = 0;
    state.releaseRevision = revision;
    state.down = !!down && !up;
    if (!state.down) state.downSeconds = 0;
    // A fresh upward command can abort the automatic landing. Releasing down
    // after confirmation leaves the automatic descent running.
    if (up) {
      if (state.holdLanding) {
        actor.autoFlightState = 'none';
        actor.targetAltitude = Math.max(minimumFlightAltitude, actor.position.y);
      }
      state.floorHold = state.holdLanding = false;
    }
    if (actor.position.y > 0.11) {
      state.armed = state.canLift = true;
      state.airborne = true;
    } else if (state.holdLanding || actor.autoFlightState === 'landing' || down && actor.targetAltitude <= 0.15) {
      stopMotors(actor);
    } else if (up && !state.upHeld) {
      // The first press arms; only a fresh press after release enables lift.
      if (state.armed) state.canLift = true;
      else state.armed = true;
    }
    state.upHeld = !!up;
    if (actor.position.y <= 0.11 && !state.canLift) actor.targetAltitude = 0.1;
  }
  function heightThrottle(actor, height) {
    const c = actor.constants;
    const force = c.mass * c.gravity + 11.25 * (height - actor.position.y) - 6 * actor.velocity.y;
    const vertical = Math.max(0.08, Math.cos(actor.currentPitch || 0) * Math.cos(actor.currentRoll || 0));
    // Bypass raw stick thrust here: a fully lowered stick must still leave the
    // height sensor enough authority to brake and hover above the floor.
    return clamp(force / (c.maxThrottleThrust * vertical), 0, powerLimit(actor));
  }
  function groundThrottle(actor, throttle, manual = false) {
    if (window.DroneFlightSafety?.isStopped(actor)) return 0;
    throttle = clamp(Number.isFinite(throttle) ? throttle : 0, 0, powerLimit(actor));
    const state = groundState(actor);
    if (state.airborne) {
      if (!state.holdLanding && actor.autoFlightState !== 'landing') {
        actor.targetAltitude = Math.max(minimumFlightAltitude, actor.targetAltitude);
        // Start braking before reaching 20 cm, including direct manual power
        // control. Never teleport/clamp the position or erase collision motion.
        const approachingFloor = actor.position.y + Math.min(0, actor.velocity.y) * 0.65 <= minimumFlightAltitude + 0.02;
        if (state.down && approachingFloor) state.floorHold = true;
        if (state.floorHold && !state.upHeld) throttle = Math.max(throttle, heightThrottle(actor, minimumFlightAltitude));
        const atMinimum = Math.abs(actor.position.y - minimumFlightAltitude) <= 0.015
          && Math.abs(actor.velocity.y) <= 0.08;
        state.downSeconds = state.down && atMinimum ? state.downSeconds + state.delta : 0;
        if (state.downSeconds + 1e-9 >= landingHoldSeconds) state.holdLanding = true;
      }
      if (state.holdLanding) {
        // This runs after each page's legacy manual-takeover handlers, so the
        // held down key/touch cannot cancel the landing it just confirmed.
        actor.autoFlightState = 'landing';
        actor.targetAltitude = 0.1;
        throttle = heightThrottle(actor, 0.1);
      }
    }
    if (actor.position.y > 0.11) return throttle;
    if (state.canLift && (manual || actor.targetAltitude > 0.15)) return throttle;
    if (state.canLift && !state.upHeld) state.canLift = false;
    actor.targetAltitude = state.canLift ? actor.targetAltitude : 0.1;
    // Idle thrust stays well below weight, while powering rotor animation/audio.
    return state.armed ? 0.06 : 0;
  }
  function beginTakeoff(actor) {
    if (window.DroneFlightSafety?.isStopped(actor) && !window.DroneFlightSafety.restart(actor)) return false;
    if (!actor.ai) {
      const state = groundState(actor);
      state.armed = true;
      actor.targetAltitude = actor.position.y;
      actor.autoFlightState = 'none';
      return true;
    }
    // Start from the current height instead of asking the PD loop to jump
    // straight to goal height. Manual takeover keeps this nearby setpoint.
    actor.targetAltitude = actor.position.y;
    actor.autoFlightState = 'takingoff';
  }
  function advanceTakeoff(actor, delta, altitude = takeoffAltitude, climbRate = 0.3) {
    actor.targetAltitude = Math.min(altitude,
      actor.targetAltitude + climbRate * clamp(delta, 0, 0.1));
    return actor.targetAltitude >= altitude
      && Math.abs(actor.position.y - altitude) < 0.02
      && Math.abs(actor.velocity.y) < 0.05;
  }
  const aiTakeoffs = new WeakSet();
  function advanceAITakeoff(actor, delta, altitude = takeoffAltitude) {
    // AI commands a prompt 0.9 m/s climb through the same thrust, drag and
    // PD controller. Player takeoff retains its gradual 0.3 m/s ramp.
    if (!aiTakeoffs.has(actor)) {
      actor.targetAltitude = actor.position.y;
      aiTakeoffs.add(actor);
    }
    const dt = clamp(delta, 0, 0.1);
    const settled = advanceTakeoff(actor, dt, altitude, 0.9);
    const c = actor.constants || {};
    const mass = c.mass ?? 0.3, gravity = c.gravity ?? 9.8;
    const maxThrust = c.maxThrottleThrust ?? 13.5, linearDrag = c.linearDrag ?? 1.35;
    const hoverThrust = gravity * mass;
    let force = hoverThrust + controlGain(actor)
      * ((actor.targetAltitude - actor.position.y) * 11.25 - actor.velocity.y * 6);
    if (force < hoverThrust * 0.9) force = hoverThrust * 1.15;
    let throttle = clamp(force / maxThrust, 0, powerLimit(actor));
    if (actor.targetAltitude <= 0.15 && actor.position.y <= 0.11) throttle = 0;
    if (actor.input) actor.input.throttle = throttle;
    actor.velocity.y += (throttle * maxThrust - hoverThrust - actor.velocity.y * linearDrag) / mass * dt;
    actor.position.y += actor.velocity.y * dt;
    if (actor.position.y < 0.1) { actor.position.y = 0.1; actor.velocity.y = 0; }
    actor.mesh?.position.copy(actor.position);
    if (settled) aiTakeoffs.delete(actor);
    return settled;
  }
  const impacts = new WeakMap();
  const aiActors = new WeakSet();
  const hoverStates = new WeakMap();
  let hoverSequence = 0;
  function clearHover(actor) {
    const state = hoverStates.get(actor);
    if (!state) return;
    state.layer.position.set(0, 0, 0);
    state.layer.rotation.set(0, 0, 0);
    state.gain = 0;
  }
  function hover(actor, delta) {
    if (!actor.mesh || !actor.position) return;
    const dt = clamp(Number.isFinite(delta) ? delta : 0, 0, 0.1);
    let state = hoverStates.get(actor);
    const height = clamp((actor.position.y - 0.11) / 0.24, 0, 1);
    if (((aiActors.has(actor) || !actor.input) && actor.altitudeState === 'grounded') || height === 0) {
      clearHover(actor);
      return;
    }
    if (!state || state.root !== actor.mesh) {
      // Move only visual children. Physics, collision bounds, camera tracking
      // and the downward ground light retain their original root transform.
      const layer = new THREE.Group();
      layer.name = 'drone-hover-motion';
      actor.mesh.children.slice().forEach(child => { if (!child.isLight) layer.add(child); });
      actor.mesh.add(layer);
      state = { root: actor.mesh, layer, time: 0, gain: 0, phase: ++hoverSequence * 2.399963229728653 };
      hoverStates.set(actor, state);
    }
    state.time += dt;
    const speed = Math.hypot(actor.velocity?.x || 0, actor.velocity?.z || 0);
    const command = aiActors.has(actor) ? 0 : Math.max(Math.abs(actor.input?.pitch || 0), Math.abs(actor.input?.roll || 0));
    const landing = actor.autoFlightState === 'landing' || actor.altitudeState === 'landing';
    const target = height * height * (3 - 2 * height)
      * (0.2 + 0.8 / (1 + (speed / 0.45) ** 2)) * (1 - 0.75 * Math.min(1, command)) * (landing ? 0.25 : 1);
    state.gain += (target - state.gain) * (1 - Math.exp(-3 * dt));
    const t = state.time, p = state.phase, g = state.gain;
    // Several slow, nonmatching frequencies avoid a synchronized looping bob.
    // Small corrective tilts lag the drift, as a flight controller would.
    state.layer.position.set(
      g * (0.009 * Math.sin(1.17 * t + p) + 0.004 * Math.sin(2.03 * t + 1.7 * p)),
      g * (0.006 * Math.sin(1.61 * t + 0.7 * p) + 0.003 * Math.sin(2.71 * t + p)),
      g * (0.009 * Math.sin(0.93 * t + 1.3 * p) + 0.004 * Math.sin(1.89 * t + p)));
    state.layer.rotation.set(
      g * (0.018 * Math.sin(0.93 * t + 1.3 * p + 0.7) + 0.004 * Math.sin(3.1 * t + p)),
      g * 0.006 * Math.sin(0.71 * t + p),
      -g * (0.018 * Math.sin(1.17 * t + p + 0.7) + 0.004 * Math.sin(3.7 * t + 1.7 * p)), 'YXZ');
  }
  const stateOf = actor => impacts.get(actor);
  const yawOf = actor => aiActors.has(actor) ? actor.mesh?.rotation.y || 0 : actor.rotation?.y ?? actor.mesh?.rotation.y ?? 0;
  function localTilt(actor, state) {
    const yaw = yawOf(actor), cos = Math.cos(yaw), sin = Math.sin(yaw);
    return { pitch: sin * state.x + cos * state.z, roll: -(cos * state.x - sin * state.z) };
  }
  function controlGain(actor) {
    const state = stateOf(actor);
    if (!state) return 1;
    const severity = clamp((state.strength ?? 1.5) / 1.5, 0, 1);
    // Preserve gyro/pilot authority while momentum and angular impulse do
    // their work. Continued cage contact cannot repeatedly restart a stun.
    const age=Math.max(state.age,airBodies.get(actor)?.age||0);
    const t=clamp(age/(0.12+0.22*severity),0,1);
    return 1-(0.08+0.47*severity)*(1-t*t*(3-2*t));
  }
  function advanceImpact(actor, delta) {
    const state = stateOf(actor);
    if (!state) return;
    const dt = clamp(delta, 0, 0.1), omega = 6.5;
    const decay = Math.exp(-omega * dt);
    // Exact critically damped spring: the same recovery at 30/60/120 FPS.
    for (const axis of ['x', 'z']) {
      const rate = `${axis}Rate`, step = (state[rate] + omega * state[axis]) * dt;
      state[axis] = (state[axis] + step) * decay;
      state[rate] = (state[rate] - omega * step) * decay;
    }
    const length = Math.hypot(state.x, state.z);
    if (length > 0.34) { state.x *= 0.34 / length; state.z *= 0.34 / length; }
    state.age += dt;
  }
  function pose(actor) {
    const state = stateOf(actor);
    if (!state || !actor.mesh) return;
    const tilt = localTilt(actor, state);
    if (aiActors.has(actor) || !actor.input) {
      actor.mesh.rotation.x = (state.basePitch || 0) + tilt.pitch;
      actor.mesh.rotation.z = (state.baseRoll || 0) + tilt.roll;
      actor.mesh.rotation.order = 'YXZ';
    } else {
      actor.currentPitch = (actor.currentPitch || 0) - state.appliedPitch + tilt.pitch;
      actor.currentRoll = (actor.currentRoll || 0) - state.appliedRoll + tilt.roll;
      state.appliedPitch = tilt.pitch; state.appliedRoll = tilt.roll;
      actor.mesh.rotation.set(actor.currentPitch, actor.rotation.y, actor.currentRoll, 'YXZ');
    }
  }
  function impact(actor, impulse, surfaceNormal, solvedAngularImpulse, source = 'surface') {
    if (!actor?.velocity || ![impulse.x, impulse.y, impulse.z].every(Number.isFinite)) return;
    if (Math.hypot(impulse.x, impulse.y, impulse.z) < 0.08 && !(solvedAngularImpulse?.length() > 1e-10)) return;
    collisionRevisions.set(actor, (collisionRevisions.get(actor) || 0) + 1);
    const body = groundBodies.get(actor);
    // Δω = r × J / I. A near-centred bump has little torque; an oblique contact
    // on a rib can rotate the cage. No probability or "collision = crash" rule.
    let angularImpulse=solvedAngularImpulse;
    if(!angularImpulse){
      const q=body?.q||airBodies.get(actor)?.q||actor.mesh?.quaternion;
      const dv=new THREE.Vector3(impulse.x,impulse.y,impulse.z);
      const normal=surfaceNormal?new THREE.Vector3(surfaceNormal.x,surfaceNormal.y,surfaceNormal.z).normalize():dv.clone().normalize();
      const lever=q?support(q,normal).point.sub(new THREE.Vector3(0,-.004,0).applyQuaternion(q)):new THREE.Vector3();
      angularImpulse=new THREE.Vector3().crossVectors(lever,dv).multiplyScalar(1/(.1**2*.65));
    }
    if (body) { body.omega.add(angularImpulse); return; }
    if (angularImpulse.length() > (solvedAngularImpulse ? 1e-10 : 2.2) || airBodies.has(actor)) {
      let tumble = airBodies.get(actor);
      if (!tumble) {
        tumble = { q: actor.mesh.quaternion.clone(), omega: new THREE.Vector3(), age: 0 };
        airBodies.set(actor, tumble);
      }
      // The gyro is already running after the first impact. Continuous rope
      // contact must not restart its response delay on every tiny impulse;
      // doing so leaves a released aircraft leaning/thrusting into the net.
      // Add the new physical angular impulse while recovery keeps advancing.
      tumble.omega.add(angularImpulse);
      tumble.contactAge = 0;
      if (tumble.omega.length() > 16) tumble.omega.setLength(16);
    }
    let state = stateOf(actor);
    if (!state) {
      state = { x: 0, z: 0, xRate: 0, zRate: 0, age: 2, revision: 0, appliedPitch: 0, appliedRoll: 0 };
      impacts.set(actor, state);
    }
    state.age = 0; state.revision++;
    state.source = source;
    state.physical=!!solvedAngularImpulse;
    state.strength = Math.max(Math.hypot(impulse.x, impulse.y, impulse.z), (state.strength || 0) * 0.5);
    const normal = surfaceNormal || impulse;
    const length = Math.hypot(normal.x, normal.y, normal.z);
    if (length > 1e-8) state.normal = { x: normal.x / length, y: normal.y / length, z: normal.z / length };
    state.x = state.physical?0:clamp(state.x + impulse.x * 0.025, -0.34, 0.34);
    state.z = state.physical?0:clamp(state.z + impulse.z * 0.025, -0.34, 0.34);
    const tiltLength = Math.hypot(state.x, state.z);
    if (tiltLength > 0.34) { state.x *= 0.34 / tiltLength; state.z *= 0.34 / tiltLength; }
    state.xRate = state.physical?0:clamp(state.xRate + impulse.x * 2.7, -4, 4);
    state.zRate = state.physical?0:clamp(state.zRate + impulse.z * 2.7, -4, 4);
    if (!airBodies.has(actor)) pose(actor);
  }
  // One impulse per approaching contact. A light touch settles without a
  // bounce; friction is limited by the normal impulse, never a per-frame drag.
  function surfaceContact(actor, normal, restitution = 0.4, friction = 0.08) {
    return netContact(actor, normal, restitution, friction);
  }
  function materialContact(actor, normal, material, surface = null) {
    const pair = contactMaterials[material];
    if (!pair) throw new RangeError(`Unknown cage contact material: ${material}`);
    window.DroneFlightSafety?.contact(actor, normal, surface?.velocity);
    return netContact(actor, normal, pair.restitution, pair.friction, surface);
  }
  function contactState(actor, normal, point = null) {
    const body = groundBodies.get(actor) || airBodies.get(actor);
    const q = body?.q || actor.mesh.quaternion;
    const lever = (point ? point.clone().sub(actor.position) : support(q, normal).point)
      .sub(new THREE.Vector3(0, -0.004, 0).applyQuaternion(q));
    const mass = actor.constants?.mass || airframe.mass, inertia = mass * 0.1 ** 2 * 0.65;
    const velocity = actor.velocity.clone().add(new THREE.Vector3().crossVectors(body?.omega || new THREE.Vector3(), lever));
    const response = impulse => impulse.clone().multiplyScalar(1 / mass)
      .add(new THREE.Vector3().crossVectors(new THREE.Vector3().crossVectors(lever, impulse).multiplyScalar(1 / inertia), lever));
    return { lever, mass, inertia, velocity, response };
  }
  // Dissipative roof-net contact: solve at the actual upper cage support point.
  // Include angular contact velocity and rotational effective mass so sliding
  // produces torque without manufacturing kinetic energy. No random crash,
  // motor cut, forced downward velocity or pre-contact altitude intervention.
  function netContact(actor, normal, restitution = 0.25, friction = 0.22, surface = null) {
    if (!actor?.mesh || !actor.velocity || ![normal.x, normal.y, normal.z].every(Number.isFinite)) return null;
    const n = new THREE.Vector3(normal.x, normal.y, normal.z);
    if (n.lengthSq() < 1e-16) return null;
    n.normalize();
    const state = contactState(actor, n, surface?.actorPoint), { lever, mass, inertia } = state;
    const contactVelocity = state.velocity.clone().sub(surface?.velocity || new THREE.Vector3());
    const inward = contactVelocity.dot(n);
    if (inward >= -1e-10) return null;
    const surfaceAxis=surface?.axis || 'y',surfaceInverseMass=surface?.inverseMass || 0;
    // The relative contact response includes both bodies' rotation, or the
    // rope's moving mass. Normal/friction impulses use the same effective mass.
    const response = impulse => {
      const result = state.response(impulse);
      if (surface?.response) result.add(surface.response(impulse));
      else result[surfaceAxis] += impulse[surfaceAxis] * surfaceInverseMass;
      return result;
    };
    const inverseMass = direction => direction.dot(response(direction));
    const speed = -inward, bounce = clamp(restitution, 0, 0.7) * clamp((speed - 0.2) / 0.8, 0, 1);
    const normalImpulse = (1 + bounce) * speed / inverseMass(n);
    const impulse = n.clone().multiplyScalar(normalImpulse);
    const afterNormal = contactVelocity.clone().add(response(impulse));
    const tangent = afterNormal.clone().addScaledVector(n, -afterNormal.dot(n));
    const tangentialSpeed = tangent.length();
    if (tangentialSpeed > 1e-10) {
      tangent.multiplyScalar(1 / tangentialSpeed);
      const frictionImpulse = Math.min(Math.max(0, friction) * normalImpulse, tangentialSpeed / inverseMass(tangent));
      impulse.addScaledVector(tangent, -frictionImpulse);
    }
    // An offset rib couples friction back into the normal direction. A final
    // inelastic projection removes any remaining point approach; each solve
    // dissipates energy, and the extra normal load still bounds dry friction.
    const residual = contactVelocity.clone().add(response(impulse)).dot(n);
    if (residual < 0) impulse.addScaledVector(n, -residual / inverseMass(n));
    const dv = impulse.clone().multiplyScalar(1 / mass);
    const angular = new THREE.Vector3().crossVectors(lever, impulse).multiplyScalar(1 / inertia);
    actor.velocity.add(dv);
    surface?.applyImpulse?.(impulse);
    impact(actor, dv, n, angular, surface?.actorContact ? 'cage' : 'surface');
    const point = new THREE.Vector3().copy(surface?.point || actor.position).add(lever);
    window.DroneCollisionVisuals?.hit(actor, speed, point, n);
    return { speed, normal: n, impulse: dv, angularImpulse: angular, restitution: bounce, contact: point };
  }
  function reset(actor) {
    cancelBackflip(actor);
    actor.controllerHeadless=false;actor.controllerHeadlessYaw=undefined;
    actor.controllerBrakeHeld=false;actor.controllerViewShift=0;
    window.DroneFlightSafety?.reset(actor);
    window.DroneFlightInput?.reset(actor);
    groundStates.delete(actor);
    collisionRevisions.delete(actor);
    pairRevisions.delete(actor);
    groundBodies.delete(actor);
    airBodies.delete(actor);
    impacts.delete(actor);
    aiTakeoffs.delete(actor);
    window.DroneAITactics?.resetGear(actor);
    clearHover(actor);
    actor.currentPitch = actor.currentRoll = 0;
    if (actor.mesh) { actor.mesh.rotation.x = 0; actor.mesh.rotation.z = 0; }
  }
  function stepStopped(actor, delta) {
    if (!window.DroneFlightSafety?.isStopped(actor)) return false;
    const dt = clamp(Number.isFinite(delta) ? delta : 0, 0, 0.05);
    actor.input ||= {};
    actor.input.throttle = actor.input.pitch = actor.input.roll = actor.input.yaw = 0;
    actor.motorThrottle = 0;
    actor.autoFlightState = 'none'; actor.targetAltitude = actor.position.y;
    if (!groundBodies.has(actor)) {
      let body = airBodies.get(actor);
      if (!body) {
        // The flight controller subtracts currentYawRate from world heading.
        body = {q:actor.mesh.quaternion.clone(), omega:new THREE.Vector3(0, -(actor.currentYawRate || 0), 0), age:0};
        airBodies.set(actor,body);
      }
      advanceTumble(actor,body,dt);
    }
    actor.velocity.y -= (actor.constants?.gravity ?? 9.8) * dt;
    actor.velocity.multiplyScalar(Math.exp(-0.2 * dt));
    actor.position.addScaledVector(actor.velocity,dt);
    groundContact(actor,dt); actor.mesh.position.copy(actor.position);
    return true;
  }
  function collide(a, b, radius = 0.1, restitution = contactMaterials.plastic.restitution, continuous = true) {
    if (!a?.mesh || !b?.mesh) return null;
    const resolved = contactFrame?.pairs.get(a)?.has(b);
    let normal = b.position.clone().sub(a.position), distance = normal.length();
    const diameter = 2 * radius;
    let time = 1;
    const startA = contactFrame?.starts.get(a), startB = contactFrame?.starts.get(b);
    if (continuous && !resolved && startA && startB) {
      const offset = startB.clone().sub(startA);
      const move = normal.clone().sub(offset);
      const aa = move.lengthSq(), bb = 2 * offset.dot(move), cc = offset.lengthSq() - diameter ** 2;
      const disc = bb * bb - 4 * aa * cc;
      if (cc >= -1e-10 && aa > 1e-16 && bb < 0 && disc >= 0) {
        const t = (-bb - Math.sqrt(disc)) / (2 * aa);
        if (t >= 0 && t <= 1) {
          time = t;
          a.position.lerpVectors(startA, a.position.clone(), time);
          b.position.lerpVectors(startB, b.position.clone(), time);
          normal = b.position.clone().sub(a.position); distance = normal.length();
        }
      }
    }
    if (time === 1 && distance >= diameter) return null;
    if (contactFrame) {
      if (!contactFrame.pairs.has(a)) contactFrame.pairs.set(a, new WeakSet());
      if (!contactFrame.pairs.has(b)) contactFrame.pairs.set(b, new WeakSet());
      contactFrame.pairs.get(a).add(b); contactFrame.pairs.get(b).add(a);
    }
    if (distance > 1e-8) normal.multiplyScalar(1 / distance);
    else {
      normal.copy(a.velocity).sub(b.velocity);
      if (normal.lengthSq() > 1e-8) normal.normalize(); else normal.set(1, 0, 0);
    }
    const wa = 1 / (a.constants?.mass || 0.3), wb = 1 / (b.constants?.mass || 0.3);
    const overlap = Math.max(0, diameter - distance);
    a.position.addScaledVector(normal, -(overlap + 1e-6) * wa / (wa + wb));
    b.position.addScaledVector(normal, (overlap + 1e-6) * wb / (wa + wb));
    const closing = b.velocity.clone().sub(a.velocity).dot(normal);
    const point = a.position.clone().add(b.position).multiplyScalar(0.5);
    const peer = contactState(b, normal, point);
    const hit = netContact(a, normal.clone().negate(), clamp(restitution, 0, 0.5), contactMaterials.plastic.friction, {
      velocity: peer.velocity, response: peer.response, actorPoint: point, actorContact: true,
      applyImpulse: impulse => {
        const dv = impulse.clone().multiplyScalar(-wb);
        b.velocity.add(dv);
        impact(b, dv, normal, new THREE.Vector3().crossVectors(peer.lever, impulse).multiplyScalar(-1 / peer.inertia), 'cage');
      }
    });
    if(hit?.impulse?.lengthSq()>1e-20)for(const [actor,peer] of [[a,b],[b,a]]){
      if(!pairRevisions.has(actor))pairRevisions.set(actor,new WeakMap());
      const revisions=pairRevisions.get(actor);revisions.set(peer,(revisions.get(peer)||0)+1);
    }
    if (time < 1 && contactFrame.delta) {
      a.position.addScaledVector(a.velocity, contactFrame.delta * (1 - time));
      b.position.addScaledVector(b.velocity, contactFrame.delta * (1 - time));
    }
    a.mesh.position.copy(a.position); b.mesh.position.copy(b.position);
    return { overlap, closingSpeed: Math.max(0, -closing), normal, time, impulse: hit?.impulse || null };
  }
  function wrapAI(update, getActor = actor => actor) {
    return function (...args) {
      const actor = getActor(...args), dt = clamp(args.at(-1), 0, 0.05);
      aiActors.add(actor);
      if (stepStopped(actor, dt)) return;
      if (advanceGroundAI(actor, dt) || advanceAirAI(actor, dt)) {
        window.DroneAITactics?.syncFlightPose(actor);
        return;
      }
      if (actor.altitudeState !== 'takingoff') aiTakeoffs.delete(actor);
      const state = stateOf(actor);
      if (!state) {
        const result = update.apply(this, args);
        window.DroneAITactics?.flightPose(actor, dt);
        groundContact(actor, dt);
        hover(actor, dt);
        return result;
      }
      advanceImpact(actor, dt);
      const position = actor.position.clone(), velocity = actor.velocity.clone();
      if (actor.mesh) {
        actor.mesh.rotation.x = state.basePitch || 0;
        actor.mesh.rotation.z = state.baseRoll || 0;
      }
      const result = update.apply(this, args);
      // A reset/landing can clear the disturbance inside the original update.
      if (stateOf(actor) !== state) { hover(actor, dt); return result; }
      window.DroneAITactics?.flightPose(actor, dt);
      state.basePitch = actor.mesh?.rotation.x || 0;
      state.baseRoll = actor.mesh?.rotation.z || 0;
      if (state.age <= 2.5 && dt > 0 && actor.altitudeState !== 'grounded') {
        const desired = actor.position.clone().sub(position).multiplyScalar(1 / dt);
        const change = desired.sub(velocity).multiplyScalar(controlGain(actor));
        const limit = 3.5 * controlGain(actor) * dt;
        if (change.length() > limit) change.setLength(limit);
        actor.velocity.copy(velocity).add(change);
        actor.velocity.x += Math.sin(state.x) * 6.5 * dt;
        actor.velocity.z += Math.sin(state.z) * 6.5 * dt;
        actor.velocity.multiplyScalar(Math.exp(-0.35 * dt));
        actor.position.copy(position).addScaledVector(actor.velocity, dt);
        actor.mesh?.position.copy(actor.position);
      }
      pose(actor);
      if (state.age > 2.5) { impacts.delete(actor); }
      hover(actor, dt);
      groundContact(actor, dt);
      return result;
    };
  }
  function attitude(actor, delta) {
    const flip=backflips.get(actor);
    if (flip) {
      if (flip.revision!==window.DroneFlightInput?.releaseRevision || actor.controllerBrakeHeld ||
          window.DroneFlightSafety?.isStopped(actor) || groundBodies.has(actor) || airBodies.has(actor)) cancelBackflip(actor);
      else {
        flip.elapsed+=Math.max(0,Math.min(.1,delta));
        const t=Math.min(1,flip.elapsed/.55), smooth=t*t*(3-2*t);
        // Local +Z is the nose, so negative pitch lifts the nose backwards.
        actor.currentPitch=flip.pitch-Math.PI*2*smooth;
        actor.currentRoll=0;actor.currentYawRate=0;
        if (t<1) return;
        cancelBackflip(actor);actor.targetAltitude=Math.max(.1,actor.position.y);
      }
    }
    if (window.DroneFlightSafety?.isStopped(actor)) return;
    // A roster can change a previously automated rig back to a human pilot.
    aiActors.delete(actor);
    const body = groundBodies.get(actor);
    if (body) { groundPose(actor, body); return; }
    const tumble = airBodies.get(actor);
    if (actor.aiAutoGear) window.DroneAITactics?.resetGear(actor);
    if (actor.position.y <= 0.11 && !groundState(actor).canLift) {
      actor.currentPitch = actor.currentRoll = actor.currentYawRate = 0;
      clearHover(actor);
      return;
    }
    const dt = Math.max(0, delta);
    const { input, constants: c, velocity: v } = actor;
    const yaw = actor.rotation.y;
    const cos = Math.cos(yaw), sin = Math.sin(yaw);
    const localX = cos * v.x - sin * v.z;
    const localZ = sin * v.x + cos * v.z;
    // A small opposing tilt slows a released axis. Cross-axis stick control
    // remains available; yaw alone does not erase translational momentum.
    const gear = window.DroneControlSettings?.gearFor(actor) || { drive: 1, brake: 1 };
    const rawPitch = actor.controllerBrakeHeld?0:clamp(Number.isFinite(input.pitch) ? input.pitch : 0, -1, 1);
    const rawRoll = actor.controllerBrakeHeld?0:clamp(Number.isFinite(input.roll) ? input.roll : 0, -1, 1);
    let targetPitch, targetRoll;
    if (window.DroneFlightInput.keyboardMode(actor)) {
      const command = window.DroneFlightInput.motionCommand(actor);
      const speed = levelSpeed(actor), mass = c.mass, weight = mass * c.gravity;
      // Tilt distributes the currently requested vertical force, including climb
      // compensation. Using weight alone over-accelerates diagonally while rising.
      const verticalForce = Math.max(weight * 0.25, weight + controlGain(actor)
        * window.DroneFlightInput.altitudeCorrection(actor, actor.targetAltitude - actor.position.y, 11.25, 6));
      // Closed-loop velocity control makes half input request half speed. Motor
      // thrust still acts along the smoothed body tilt; velocity is never assigned.
      const dragForce = horizontalDrag(actor);
      const pitchForce = mass * 2.5 * (command.pitch * speed - localZ) + localZ * dragForce;
      const rollForce = mass * 2.5 * (command.roll * speed - localX) + localX * dragForce;
      targetPitch = rawPitch ? Math.atan2(pitchForce, verticalForce) : -clamp(localZ * 0.30, -0.6, 0.6);
      targetRoll = rawRoll ? -Math.atan2(rollForce, verticalForce) : clamp(localX * 0.30, -0.6, 0.6);
    } else {
      const command = window.DroneFlightInput.stickCommand(actor);
      const drive = driveTilt(actor, command.amount) / Math.max(command.amount, 1e-8);
      targetPitch = command.pitch * drive - clamp(localZ * 0.30, -0.6, 0.6) * (1 - Math.abs(command.pitch));
      targetRoll = -command.roll * drive + clamp(localX * 0.30, -0.6, 0.6) * (1 - Math.abs(command.roll));
    }
    const tilt = Math.hypot(targetPitch, targetRoll);
    // High gear needs a larger allowed tilt as well as a larger command;
    // otherwise the old tilt ceiling silently collapses it to the same speed.
    const tiltLimit = Math.min(0.4, c.maxTilt ?? 0.4);
    if (tilt > tiltLimit) {
      targetPitch *= tiltLimit / tilt;
      targetRoll *= tiltLimit / tilt;
    }
    // Convert one tilt vector to YXZ angles. Simply adding pitch and roll
    // would make a diagonal command use less power than a straight command.
    const angle = Math.hypot(targetPitch, targetRoll);
    if (angle > 1e-8) {
      const sine = Math.sin(angle);
      const forward = targetPitch / angle, lateral = targetRoll / angle;
      targetPitch = Math.asin(sine * forward);
      targetRoll = Math.atan2(sine * lateral, Math.cos(angle));
    }
    const yawInput = actor.controllerBrakeHeld?0:window.DroneFlightInput.keyboardMode(actor)
      ? clamp(Number.isFinite(input.yaw) ? input.yaw : 0, -1, 1)
      : window.DroneFlightInput.axisResponse(input.yaw, actor.touchActiveLeft ? 0.03 : 0);
    if (tumble) {
      // Keep the pilot's tilt/braking target during rigid-body recovery.
      // Levelling to world-up alone discards every direction command while
      // net contact keeps the cage rotating, leaving it unable to slide away.
      // The same delayed, thrust-limited motor torque follows this target;
      // no velocity, angular momentum or attitude is reset at contact.
      const desiredUp = new THREE.Vector3(0, 1, 0).applyEuler(new THREE.Euler(
        targetPitch, yaw, targetRoll, 'YXZ'));
      // First arrest the impact, then blend the pilot back into the recovery
      // target. Sustained contact never restarts this handover clock.
      const handover = controlGain(actor);
      desiredUp.lerp(new THREE.Vector3(0, 1, 0), 1 - handover).normalize();
      advanceTumble(actor, tumble, delta, desiredUp, -yawInput * (c.yawSpeed ?? 1.6) * (gear.yaw ?? 1) * handover);
      return;
    }
    const state = stateOf(actor);
    if (state) {
      actor.currentPitch -= state.appliedPitch; actor.currentRoll -= state.appliedRoll;
      state.appliedPitch = state.appliedRoll = 0;
      advanceImpact(actor, dt);
    }
    const alpha = 1 - Math.exp(-6 * controlGain(actor) * dt);
    actor.currentPitch += (targetPitch - actor.currentPitch) * alpha;
    actor.currentRoll += (targetRoll - actor.currentRoll) * alpha;
    if (state) {
      const tilt = localTilt(actor, state);
      actor.currentPitch += tilt.pitch; actor.currentRoll += tilt.roll;
      state.appliedPitch = tilt.pitch; state.appliedRoll = tilt.roll;
      if (state.age > 2.5) {
        actor.currentPitch -= tilt.pitch; actor.currentRoll -= tilt.roll;
        impacts.delete(actor);
      }
    }
    const yawResponse = Math.abs(yawInput) > 0.01 ? c.yawRampSpeed : 12 * gear.brake;
    const yawTarget = yawInput * c.yawSpeed * (gear.yaw ?? 1);
    const yawStart = actor.currentYawRate;
    const yawAlpha = -Math.expm1(-yawResponse * dt);
    actor.currentYawRate += (yawTarget - yawStart) * yawAlpha;
    // Integrate the whole response curve. Using the end-of-frame rate for
    // the entire frame made an identical stick gesture turn further at 30 Hz.
    actor.rotation.y -= yawResponse > 0
      ? yawTarget * dt + (yawStart - yawTarget) * yawAlpha / yawResponse
      : yawStart * dt;
    hover(actor, dt);
  }
  function drag(actor) {
    const v = actor.velocity;
    // Lower low-speed drag permits coasting. Quadratic drag limits sustained
    // speed in the small arena. Keep the existing vertical PD tuning intact.
    const dragForce = horizontalDrag(actor)+(actor.controllerBrakeHeld?actor.constants.mass*6:0);
    // Brake both local axes equally, including strafing at any heading.
    // High gear coasts further; a downshift never clamps or erases velocity.
    const gear = window.DroneControlSettings?.gearFor(actor) || { brake: 1 };
    const yaw = yawOf(actor), sin = Math.sin(yaw), cos = Math.cos(yaw);
    const forwardSpeed = sin * v.x + cos * v.z;
    const lateralSpeed = cos * v.x - sin * v.z;
    const extra = forwardSpeed * dragForce * (gear.brake - 1) * (1 - Math.min(1, Math.abs(actor.input?.pitch || 0)));
    const sideExtra = lateralSpeed * dragForce * (gear.brake - 1) * (1 - Math.min(1, Math.abs(actor.input?.roll || 0)));
    return v.clone().set(-v.x * dragForce - sin * extra - cos * sideExtra, -v.y * verticalDrag(actor), -v.z * dragForce - cos * extra + sin * sideExtra);
  }
  window.DroneFlightDynamics = Object.freeze({ airframe, contactMaterials, materialContact, accelerate, limitSpeed, powerLimit, throttleForVerticalForce, driveTilt, horizontalDrag, verticalDrag, levelSpeed, beginContacts, endContacts: () => { contactFrame = null; }, createGroundFailure, groundContact, groundHeight, stepStopped, beginTakeoff, prepareGround, groundThrottle, stopMotors, landingConfirmed, cancelLandingHold, minimumFlightAltitude, landingHoldSeconds, advanceTakeoff, advanceAITakeoff, attitude, drag, impact, surfaceContact, netContact, collide, reset, wrapAI, controlGain, hover, beginBackflip, cancelBackflip,
    contactMotion: (actor, normal) => { const state = contactState(actor, new THREE.Vector3(normal.x, normal.y, normal.z).normalize()); return { velocity: state.velocity, lever: state.lever }; },
    airBody: actor => { const body = airBodies.get(actor); return body ? { quaternion: body.q.toArray(), angularVelocity: body.omega.toArray() } : null; },
    groundBody: actor => { const body = groundBodies.get(actor); return body ? { quaternion: body.q.toArray(), angularVelocity: body.omega.toArray() } : null; },
    collisionRevision: (actor,peer) => (peer?pairRevisions.get(actor)?.get(peer):collisionRevisions.get(actor)) || 0,
    collisionState: actor => stateOf(actor) ? { ...stateOf(actor), normal: stateOf(actor).normal ? { ...stateOf(actor).normal } : null } : null });
})();
