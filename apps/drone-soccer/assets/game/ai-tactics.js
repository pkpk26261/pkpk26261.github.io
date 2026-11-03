/* Tactical calculations and automatic flight gears shared by AI games and practice. */
(() => {
  'use strict';
  const clamp = (n, lo, hi) => Math.max(lo, Math.min(hi, n));
  const guardFormations = new WeakMap();
  const guardHeights = new WeakMap();
  const guardDecisions = new WeakMap();
  const guardAssignments = new WeakMap();
  const guardChallenges = new WeakMap();
  const challengeCounts = new WeakMap();
  const challenge = Object.freeze({ range:.8, speed:2.2, follow:.28, duration:.85, followSeconds:.14, recovery:.55, cooldown:.5 });
  function goalFrame(ring) {
    const body = window.DroneGoalPhysics?.state(ring);
    const q = body?.q || ring.mesh?.quaternion;
    const rotate = (v, inverse = false) => {
      if (!q) return { x: v.x, y: v.y, z: v.z };
      const sign = inverse ? -1 : 1, x = q.x * sign, y = q.y * sign, z = q.z * sign;
      const tx = 2 * (y * v.z - z * v.y), ty = 2 * (z * v.x - x * v.z), tz = 2 * (x * v.y - y * v.x);
      return { x: v.x + q.w * tx + y * tz - z * ty,
        y: v.y + q.w * ty + z * tx - x * tz, z: v.z + q.w * tz + x * ty - y * tx };
    };
    return { body, local: v => rotate(v, true), world: v => rotate(v) };
  }
  function guardHome(guard, ring, partners = [], standoff = 0.2) {
    // Two cages cover the opening side by side at the same height and depth.
    // Leave only 1 cm for the contact solver; the cages can still press against
    // one another after an impact, without permanent penetration commands.
    // Actual cage separation remains the collision solver's responsibility.
    const offset = guard.guardHomeOffsetX || 0;
    const teammates = partners.filter(p => p && p !== guard
      && (p.altitudeState === 'flying' || p.altitudeState === 'takingoff'
        || p.altitudeState === undefined && p.position.y > 0.25)
      && Math.abs(p.position.z - ring.pos.z) < 1.2
      && Math.hypot(p.position.x - ring.pos.x, p.position.y - ring.pos.y) < 0.5);
    guardFormations.set(guard, { ring, partners, teammates, standoff,
      constants: guardFormations.get(guard)?.constants });
    const frame = goalFrame(ring);
    const lateral = p => frame.local({ x: p.position.x - ring.pos.x,
      y: p.position.y - ring.pos.y, z: p.position.z - ring.pos.z }).x;
    // After a knockback, keep the current left/right order. Crossing through
    // an occupied station to reclaim a pilot's original side blocks recovery.
    const left = !teammates.some(p => Math.abs(lateral(p) - lateral(guard)) > 0.02
      ? lateral(p) < lateral(guard) : (p.guardHomeOffsetX || 0) < offset);
    const station = frame.world({ x: teammates.length ? (left ? -0.105 : 0.105) : 0,
      y: 0, z: -ring.forwardSign * Math.max(0.26, standoff) });
    return { x: ring.pos.x + station.x, y: ring.pos.y + station.y, z: ring.pos.z + station.z };
  }
  function guardTarget(guard, enemy, ring, c, partners = [], standoff = 0.2, random = Math.random) {
    const home = guardHome(guard, ring, partners, standoff);
    guardFormations.get(guard).constants = c;
    if (c.tactics?.activeIntercept) return activeGuardTarget(guard, enemy, ring, c, partners, home, random);
    // Share one observed height while holding the same lateral/depth stations.
    // Clamp to the opening, so a high/low opponent cannot lure the pair away.
    if (partners.some(p => p && p !== guard)) {
      const frame = goalFrame(ring);
      const enemyLocal = frame.local({ x: enemy.position.x - ring.pos.x,
        y: enemy.position.y - ring.pos.y, z: enemy.position.z - ring.pos.z });
      const active = -enemyLocal.z * ring.forwardSign > 0 && enemy.readyToScore !== false
        && enemy.altitudeState !== 'grounded' && Math.abs(enemyLocal.z) <= (c.standbyRange ?? 4);
      const velocity = frame.local(enemy.velocity || { x: 0, y: 0, z: 0 });
      const closing = velocity.z * ring.forwardSign;
      const arrival = closing > 0.05 ? Math.max(0, (-enemyLocal.z * ring.forwardSign - 0.26) / closing) : 0;
      const horizon = c.tactics?.predictiveIntercept ? Math.min(arrival, c.predictionHorizon ?? 0.3) : 0;
      // Keep the whole legal opening covered during an incoming run. Following
      // a steep climb/feint by 15 cm leaves the opposite edge unprotected.
      const closeThreat = -enemyLocal.z * ring.forwardSign < 1.2;
      const heightLimit = c.tactics?.angleCoverage && closeThreat ? 0
        : closing > 0.05 && c.tactics?.predictiveIntercept
          ? (c.predictionHorizon > 0.5 ? 0.035 : 0.06) : 0.15;
      const error = (random() - 0.5) * 2 * (c.aimError ?? 0);
      const height = active ? clamp(enemyLocal.y + velocity.y * horizon + error, -heightLimit, heightLimit) : 0;
      guardHeights.set(ring, { height, enemy, range: c.standbyRange ?? 4 });
      const rise = frame.world({ x: 0, y: height, z: 0 });
      guard.targetAltitude = home.y + rise.y;
      for (const partner of guardFormations.get(guard).teammates) {
        if (guardFormations.get(partner)?.ring === ring) partner.targetAltitude = home.y + rise.y;
      }
      return { x: home.x + rise.x, y: home.y + rise.y, z: home.z + rise.z };
    }
    const homeX = ring.pos.x + (guard.guardHomeOffsetX || 0);
    const distZ = ring.pos.z - enemy.position.z;
    const inFront = distZ * ring.forwardSign > 0;
    if (!inFront || enemy.readyToScore === false || enemy.altitudeState === 'grounded'
      || Math.abs(distZ) > (c.standbyRange ?? 4)) return home;
    const timeToGoal = Math.abs(enemy.velocity.z) > 0.05 ? distZ / enemy.velocity.z : -1;
    const approaching = timeToGoal > 0;
    // Never extrapolate an opponent who is retreating away from the goal.
    const horizon = c.tactics?.predictiveIntercept && approaching
      ? clamp(timeToGoal - standoff / Math.max(0.05, Math.abs(enemy.velocity.z)), 0, c.predictionHorizon ?? 1.2) : 0;
    const px = enemy.position.x + enemy.velocity.x * horizon;
    const py = enemy.position.y + enemy.velocity.y * horizon;
    const engage = approaching && Math.abs(distZ) < (c.engageDistance || 2.4) && random() < (c.engageChance ?? 0.78);
    const homeDepth = (ring.pos.z - home.z) * ring.forwardSign;
    const depth = Math.max(homeDepth, engage ? Math.min(c.guardDepth, Math.abs(distZ) * (c.engageDepthRatio ?? 0.35)) : 0);
    let x = px;
    let y = py;
    if (c.tactics?.angleCoverage) {
      // At this depth cover the line FROM the opening TO the attacker.
      // Chasing the attacker's full lateral offset leaves the middle exposed.
      const fraction = clamp(depth / Math.max(Math.abs(distZ), 0.25), 0, 0.85);
      x = ring.pos.x + (x - ring.pos.x) * fraction;
      y = ring.pos.y + (y - ring.pos.y) * fraction;
    }
    x += (random() - 0.5) * 2 * c.aimError;
    y += (random() - 0.5) * 2 * c.aimError;
    return {
      x: clamp(x, homeX - c.guardRadiusXY, homeX + c.guardRadiusXY),
      y: clamp(y, ring.pos.y - 0.35, ring.pos.y + 0.35),
      z: ring.pos.z - ring.forwardSign * depth
    };
  }

  function guardThreat(enemy, ring, c) {
    if (!enemy?.position || enemy.readyToScore === false || enemy.altitudeState === 'grounded') return null;
    const frame = goalFrame(ring);
    const p = frame.local({ x: enemy.position.x - ring.pos.x, y: enemy.position.y - ring.pos.y, z: enemy.position.z - ring.pos.z });
    const v = frame.local(enemy.velocity || { x: 0, y: 0, z: 0 });
    const depth = -p.z * ring.forwardSign;
    if (depth <= 0.05 || depth > (c.standbyRange ?? 4) || v.z * ring.forwardSign < -0.2) return null;
    return { frame, p, v, depth };
  }
  function guardAvailable(actor, ring) {
    if (!actor?.position || actor.altitudeState !== 'flying' && !(actor.altitudeState === undefined && actor.position.y > 0.25)) return false;
    const p = goalFrame(ring).local({ x: actor.position.x - ring.pos.x, y: actor.position.y - ring.pos.y, z: actor.position.z - ring.pos.z });
    return -p.z * ring.forwardSign > 0.08 && -p.z * ring.forwardSign < 1.8 && Math.hypot(p.x, p.y) < 1.1;
  }
  function activeGuardTarget(guard, enemy, ring, c, partners, home, random) {
    const threat = guardThreat(enemy, ring, c);
    const save = (role, target) => {
      guardDecisions.set(guard, { ring, enemy, constants: c, role, target: { ...target },
        remaining: ((c.reactionMin ?? .2) + (c.reactionMax ?? .2)) / 2 });
      guard.targetAltitude = target.y;
      return target;
    };
    if (!threat || threat.depth > (c.engageDistance ?? c.standbyRange ?? 2.4)) {
      if (guardAssignments.get(ring)?.enemy === enemy) guardAssignments.delete(ring);
      return save('home', home);
    }
    const { frame, p, v, depth } = threat;
    // Only automated partners may be assigned an interception. A human still
    // owns their controls, even when their current position provides cover.
    const available = [guard, ...partners].filter((a, i, all) => a && all.indexOf(a) === i
      && (a === guard || guardFormations.has(a)) && guardAvailable(a, ring));
    let assignment = guardAssignments.get(ring);
    if (assignment?.enemy !== enemy || !available.includes(assignment.primary)) {
      const cost = a => Math.hypot(a.position.x - enemy.position.x, a.position.y - enemy.position.y, a.position.z - enemy.position.z);
      const primary = available.reduce((best, a) => !best || cost(a) < cost(best) ? a : best, null);
      assignment = { enemy, primary };
      guardAssignments.set(ring, assignment);
    }
    if (c.tactics?.coverRotation && assignment.primary && available.length > 1 && depth < .9) {
      const current = assignment.primary;
      const local = frame.local({ x:current.position.x-ring.pos.x,y:current.position.y-ring.pos.y,z:current.position.z-ring.pos.z });
      const distance = a => Math.hypot(a.position.x-enemy.position.x,a.position.y-enemy.position.y,a.position.z-enemy.position.z);
      const replacement = available.find(a => a !== current && distance(a) + .25 < distance(current));
      // The attacker is goal-side of the first keeper and has escaped its
      // lateral reach. The closer keeper takes over; the first returns home.
      if (replacement && -local.z*ring.forwardSign > depth + .2
        && Math.hypot(local.x-p.x,local.y-p.y) > .35) {
        assignment = { enemy,primary:replacement,handoff:true,handoffs:(assignment.handoffs||0)+1 };
        guardAssignments.set(ring,assignment);
      }
    }
    const primary = assignment.primary === guard;
    const hasCover = available.some(a => {
      if(a===assignment.primary)return false;
      const q=frame.local({x:a.position.x-ring.pos.x,y:a.position.y-ring.pos.y,z:a.position.z-ring.pos.z});
      const d=-q.z*ring.forwardSign;
      return d>.15&&d<.55&&Math.hypot(q.x,q.y)<.3;
    });
    const maxDepth = assignment.handoff && !hasCover ? .35 : c.guardDepth ?? .95;
    const closing = v.z * ring.forwardSign;
    // Only abandon the packed opening to meet a striker that is genuinely
    // committing: moving mostly straight at the goal rather than weaving.
    // Judge intent by direction (how lateral the motion is), not raw speed, so
    // a slow straight approach is still met while a sideways feint is held
    // against, keeping the pair from being drawn apart and beaten to the flank.
    // interceptDiscipline scales this restraint by tier (0 = always lunge like
    // a beginner; 1 = a composed keeper that never bites on a dangled feint).
    const lateral = Math.hypot(v.x, v.y);
    const direction = closing > 0.05 ? clamp(closing / (closing + 1.5 * lateral), 0, 1) : 0;
    const commitment = 1 - (c.interceptDiscipline ?? 0) * (1 - direction);
    const interceptDepth = available.length > 1
      ? Math.min(maxDepth, 0.26 + (Math.max(0.32, depth - 0.12) - 0.26) * commitment)
      : Math.min(0.45, Math.max(0.3, depth - 0.12));
    const horizon = c.tactics?.predictiveIntercept && closing > 0.05
      ? clamp((depth - interceptDepth) / closing, 0, c.predictionHorizon ?? 0.3) : 0;
    let x = p.x + v.x * horizon, y = p.y + v.y * horizon;
    let local;
    if (primary) {
      const fraction = available.length > 1 ? 1 : clamp(interceptDepth / depth, 0, 1);
      x = x * fraction + (random() - 0.5) * 2 * (c.aimError ?? 0);
      y = y * fraction + (random() - 0.5) * 2 * (c.aimError ?? 0);
      local = { x: clamp(x, -(c.guardRadiusXY ?? .62), c.guardRadiusXY ?? .62),
        y: clamp(y, -.45, .45), z: -ring.forwardSign * interceptDepth };
    } else {
      // Shift along the projected scoring line while keeping the complete
      // legal opening covered. Never send both pilots out after one feint.
      x *= .26 / Math.max(.26, depth); y *= .26 / Math.max(.26, depth);
      const scale = Math.min(1, .06 / Math.max(.0001, Math.hypot(x,y)));
      local = { x: x * scale, y: y * scale, z: -ring.forwardSign * .26 };
    }
    const offset = frame.world(local);
    return save(primary ? 'intercept' : 'cover', { x: ring.pos.x + offset.x, y: ring.pos.y + offset.y, z: ring.pos.z + offset.z });
  }

  const gearStates = new WeakMap();
  const flightStates = new WeakMap();
  const safetyStates = new WeakMap();
  const attackStates = new WeakMap();
  const attackFlights = new WeakMap();
  const attackDepartures = new WeakMap();
  const attackLaunches = new WeakMap();
  const attackMemories = new WeakMap();
  const guardReturns = new WeakMap();
  function resetGear(actor) {
    const formation = guardFormations.get(actor);
    if (formation && guardAssignments.get(formation.ring)?.primary === actor) guardAssignments.delete(formation.ring);
    guardDecisions.delete(actor);
    gearStates.delete(actor);
    flightStates.delete(actor);
    safetyStates.delete(actor);
    attackStates.delete(actor);
    attackFlights.delete(actor); attackDepartures.delete(actor); attackLaunches.delete(actor); attackMemories.delete(actor);
    guardReturns.delete(actor);
    guardChallenges.delete(actor);challengeCounts.delete(actor);
    guardFormations.delete(actor);
    if (actor.aiAutoGear) actor.controlGear = 1;
    delete actor.aiAutoGear;
  }
  function selectGear(actor, target, c, delta, goal = actor.targetRing) {
    const dx = target.x - actor.position.x, dy = target.y - actor.position.y, dz = target.z - actor.position.z;
    const distance = Math.hypot(dx, dy, dz);
    const v = actor.velocity, speed = Math.hypot(v.x, v.y, v.z);
    const alignment = speed > 0.08 && distance > 0.005 ? (v.x * dx + v.y * dy + v.z * dz) / (speed * distance) : 1;
    // Allow time to shed momentum before reaching the target; a shift never
    // clamps velocity. Tight turns get low gear's stronger braking as well.
    const stopping = speed * 0.24 + speed * speed / (2 * Math.max(0.8, c.accel * 1.65));
    const ring = goal;
    const nearRing = ring && Math.abs(actor.position.z - ring.pos.z) < 0.75;
    const offCenter = nearRing && Math.hypot(actor.position.x - target.x, actor.position.y - target.y) > 0.07;
    let state = gearStates.get(actor);
    const previous = state?.gear ?? 0;
    let next = 1;
    if (distance < (previous === 0 ? 0.26 : 0.18) || distance < stopping + 0.1 || alignment < 0.55 || offCenter || actor.jukePhase === 'decel') next = 0;
    else if (!nearRing && alignment > 0.9 && distance > (previous === 2 ? 0.8 : 1.1) + stopping * 0.6) next = 2;
    if (!state) { state = { gear: next, age: 0 }; gearStates.set(actor, state); }
    else {
      state.age += clamp(delta, 0, 0.05);
      // Safety downshifts are immediate; upshifts wait to avoid hunting.
      if (next < state.gear || next > state.gear && state.age >= 0.35) { state.gear = next; state.age = 0; }
    }
    actor.aiAutoGear = true;
    actor.controlGear = state.gear;
    return window.DroneControlSettings?.gears[state.gear] || { drive: 1, brake: 1 };
  }
  function stepFlight(guard, target, c, delta, options = {}) {
    // Small integration steps keep braking/gear boundaries consistent when
    // the browser renders at 30, 60 or 120 FPS.
    const dt = clamp(delta, 0, 0.05);
    if (dt <= 0) return;
    const steps = Math.ceil(dt * 120);
    for (let i = 0; i < steps; i++) integrateFlight(guard, target, c, dt / steps,
      { ...options, trafficTime: i * dt / steps });
  }
  function integrateFlight(guard, target, c, dt, options) {
    const plan = safeFlight(guard, target, c, options);
    target = plan.target;
    const dx = target.x - guard.position.x, dy = target.y - guard.position.y, dz = target.z - guard.position.z;
    const distance = Math.hypot(dx, dy, dz);
    let gear = selectGear(guard, target, c, dt, options.ring);
    if(options.ramming&&!plan.cautious){
      gearStates.get(guard).gear=2;guard.controlGear=2;
      gear=window.DroneControlSettings?.gears[2]||gear;
    }
    if (plan.cautious) {
      gearStates.get(guard).gear = plan.bracing ? 1 : 0;
      guard.controlGear = plan.bracing ? 1 : 0;
    }
    const activeGear = plan.cautious ? window.DroneControlSettings?.gears[plan.bracing ? 1 : 0] || { drive: 0.55, brake: 1.6 } : gear;
    const cruise = Math.min((options.maxSpeed ?? c.maxSpeed) * window.DroneFlightDynamics.airframe.speedScale,
      window.DroneFlightDynamics.levelSpeed(guard));
    const speed = Math.min(cruise * plan.speedScale, distance * (options.trackingGain ?? c.trackingGain ?? 3));
    let state = flightStates.get(guard);
    if (!state) {
      state = { x: 0, y: 0, z: 0, yaw: guard.mesh?.rotation.y ?? guard.rotation?.y ?? 0,
        yawRate: 0, pitch: 0, roll: 0 };
      flightStates.set(guard, state);
    }
    const currentSpeed = Math.hypot(guard.velocity.x, guard.velocity.y, guard.velocity.z);
    const braking = speed < currentSpeed || guard.controlGear === 0;
    const response = (options.accel ?? c.accel) * (braking ? activeGear.brake : 0.7 + 0.3 * activeGear.drive);
    // A pilot cannot instantly redirect thrust. Bound the acceleration vector
    // and its rate of change, including after a shift or a new tactical target.
    const limit = Math.min(6, response);
    const desired = {};
    for (const [axis, offset] of [['x', dx], ['y', dy], ['z', dz]]) {
      desired[axis] = distance > 0.005 ? offset / distance * speed : 0;
    }
    plan.steer(desired);
    const acceleration = {};
    for (const axis of ['x', 'y', 'z']) acceleration[axis] = (desired[axis] - guard.velocity[axis]) * response;
    const magnitude = Math.hypot(acceleration.x, acceleration.y, acceleration.z);
    const scale = magnitude > limit ? limit / magnitude : 1;
    const smoothing = 1 - Math.exp(-dt / 0.12);
    for (const axis of ['x', 'y', 'z']) state[axis] += (acceleration[axis] * scale - state[axis]) * smoothing;
    const body = guard.constants || {}, mass = body.mass ?? 0.3, gravity = body.gravity ?? 9.8;
    const drag = window.DroneFlightDynamics.horizontalDrag(guard), verticalDrag = body.linearDrag ?? 1.35;
    const force = { x: state.x * mass + guard.velocity.x * drag,
      y: Math.max(0, state.y * mass + mass * gravity + guard.velocity.y * verticalDrag),
      z: state.z * mass + guard.velocity.z * drag };
    const horizontal=Math.hypot(force.x,force.z),bankLimit=Math.min(0.4,body.maxTilt??0.4);
    const horizontalLimit=force.y*Math.tan(bankLimit);
    if(horizontal>horizontalLimit&&horizontal>0){force.x*=horizontalLimit/horizontal;force.z*=horizontalLimit/horizontal;}
    const thrust = Math.hypot(force.x, force.y, force.z), maxThrust = body.maxThrottleThrust ?? 13.5;
    const power = window.DroneFlightDynamics.powerLimit(guard);
    const thrustScale = thrust ? Math.min(1, power * maxThrust / thrust) : 0;
    guard.motorThrottle = thrust * thrustScale / maxThrust;
    if (guard.input) guard.input.throttle = guard.motorThrottle;
    for (const axis of ['x', 'y', 'z']) {
      const resistance = guard.velocity[axis] * (axis === 'y' ? verticalDrag : drag);
      guard.velocity[axis] += ((force[axis] * thrustScale - resistance) / mass - (axis === 'y' ? gravity : 0)) * dt;
      guard.position[axis] += guard.velocity[axis] * dt;
    }
  }
  function guardFlightTarget(actor, target) {
    const formation = guardFormations.get(actor);
    if (formation?.constants?.tactics?.activeIntercept && actor.altitudeState === 'flying') {
      const decision = guardDecisions.get(actor), c = formation.constants;
      if (decision?.ring === formation.ring && decision.constants === c && guardThreat(decision.enemy, formation.ring, c)) {
        const assignment = guardAssignments.get(formation.ring);
        if (decision.role === 'intercept' && assignment?.primary === actor
          || decision.role === 'cover' && assignment?.primary !== actor) {
          return { ...decision.target };
        }
      }
      if (decision) decision.role = 'home';
      return guardHome(actor, formation.ring, formation.partners, formation.standoff);
    }
    if (formation && actor.altitudeState === 'flying' && formation.partners.some(p => p && p !== actor)) {
      const home = guardHome(actor, formation.ring, formation.partners, formation.standoff);
      const observation = guardHeights.get(formation.ring);
      const enemy = observation?.enemy;
      const depth = enemy && (formation.ring.pos.z - enemy.position.z) * formation.ring.forwardSign;
      // A cached observation must not leave both keepers high/low after the
      // striker retreats, lands or loses its scoring eligibility.
      const height = enemy && enemy.readyToScore !== false && enemy.altitudeState !== 'grounded'
        && depth > 0 && depth <= observation.range ? observation.height : 0;
      const rise = goalFrame(formation.ring).world({ x: 0, y: height, z: 0 });
      target = { x: home.x + rise.x, y: home.y + rise.y, z: home.z + rise.z };
    }
    return target;
  }
  function guardChallenge(actor, home, c, delta) {
    const formation=guardFormations.get(actor);
    if(!formation||actor.altitudeState!=='flying')return null;
    const ring=formation.ring,frame=goalFrame(ring),dt=clamp(delta,0,.05);
    const local=p=>frame.local({x:p.x-ring.pos.x,y:p.y-ring.pos.y,z:p.z-ring.pos.z});
    const bounded=point=>{
      const p=local(point);p.x=clamp(p.x,-1,1);p.y=clamp(p.y,-.6,.6);
      p.z=-ring.forwardSign*clamp(-p.z*ring.forwardSign,.3,1.55);
      const q=frame.world(p);return {x:ring.pos.x+q.x,y:ring.pos.y+q.y,z:ring.pos.z+q.z};
    };
    let state=guardChallenges.get(actor);
    if(state&&state.ring!==ring){guardChallenges.delete(actor);state=null;}
    if(state){
      state.age+=dt;
      if(state.phase==='drive'){
        if(window.DroneFlightDynamics?.collisionRevision(actor,state.enemy)!==state.revision&&state.hitAt===null)state.hitAt=state.age;
        const distance=Math.hypot(actor.position.x-state.enemy.position.x,actor.position.y-state.enemy.position.y,actor.position.z-state.enemy.position.z);
        if(state.age>=challenge.duration||state.hitAt!==null&&state.age-state.hitAt>=challenge.followSeconds||distance>1.3||state.enemy.altitudeState==='grounded'){
          state.phase='recover';state.age=0;
          state.target=bounded({x:actor.position.x-state.direction.x*.35,y:home.y,z:actor.position.z-state.direction.z*.35});
        }
      }else if(state.phase==='recover'&&state.age>=challenge.recovery){state.phase='cooldown';state.age=0;}
      else if(state.phase==='cooldown'&&state.age>=challenge.cooldown){guardChallenges.delete(actor);state=null;}
      if(state)return {...state,target:state.phase==='cooldown'?home:state.target};
    }
    const position=local(actor.position),depth=-position.z*ring.forwardSign;
    if(depth<.12||depth>1.65||Math.hypot(position.x,position.y)>1.15)return null;
    const observed=guardDecisions.get(actor)?.enemy;
    const enemies=[...new Set([observed,...(window.DroneArenaCollision?.traffic||[]).map(p=>p.actor)])].filter(enemy=>{
      if(!enemy?.position||enemy===actor||formation.partners.includes(enemy)||enemy.altitudeState==='grounded'||enemy.position.y<.25)return false;
      if(actor.team&&enemy.team&&actor.team===enemy.team)return false;
      if(enemy!==observed&&!(actor.team&&enemy.team&&actor.team!==enemy.team))return false;
      const p=local(enemy.position),d=-p.z*ring.forwardSign;
      return d>.15&&d<1.7&&Math.hypot(p.x,p.y)<1.2;
    });
    const distance=other=>Math.hypot(other.position.x-actor.position.x,other.position.y-actor.position.y,other.position.z-actor.position.z);
    const enemy=enemies.sort((a,b)=>distance(a)-distance(b))[0];
    if(!enemy||distance(enemy)>challenge.range)return null;
    const length=Math.max(.001,distance(enemy));
    const direction={x:(enemy.position.x-actor.position.x)/length,y:(enemy.position.y-actor.position.y)/length,z:(enemy.position.z-actor.position.z)/length};
    const closing=['x','y','z'].reduce((sum,k)=>sum+((actor.velocity?.[k]||0)-(enemy.velocity?.[k]||0))*direction[k],0);
    if(enemy.readyToScore===false&&length>.45&&closing<.1)return null;
    // One nearby opponent gets one charger; a second keeper still covers.
    // Different close opponents can each be met by their nearest defender.
    const available=[actor,...formation.partners].filter(p=>p&&guardFormations.has(p)&&p.altitudeState==='flying');
    const committed=available.find(p=>guardChallenges.get(p)?.enemy===enemy);
    if(committed&&committed!==actor)return null;
    const cost=p=>Math.hypot(p.position.x-enemy.position.x,p.position.y-enemy.position.y,p.position.z-enemy.position.z);
    const assigned=guardAssignments.get(ring);
    let owner=available.reduce((best,p)=>!best||cost(p)<cost(best)-1e-6||Math.abs(cost(p)-cost(best))<1e-6&&(p.guardHomeOffsetX||0)<(best.guardHomeOffsetX||0)?p:best,null);
    if(assigned?.enemy===enemy&&available.includes(assigned.primary)&&cost(assigned.primary)<length+.12)owner=assigned.primary;
    if(owner!==actor)return null;
    const prediction=Math.min(.08,c.predictionHorizon||0);
    const target=bounded({x:enemy.position.x+(enemy.velocity?.x||0)*prediction+direction.x*challenge.follow,
      y:enemy.position.y+(enemy.velocity?.y||0)*prediction+direction.y*challenge.follow,
      z:enemy.position.z+(enemy.velocity?.z||0)*prediction+direction.z*challenge.follow});
    state={ring,enemy,target,direction,phase:'drive',age:0,hitAt:null,revision:window.DroneFlightDynamics?.collisionRevision(actor,enemy)||0};
    guardChallenges.set(actor,state);challengeCounts.set(actor,(challengeCounts.get(actor)||0)+1);
    return state;
  }
  function stepGuard(actor, target, c, delta, options = {}) {
    if (guardFormations.has(actor)) guardFormations.get(actor).constants = c;
    target = guardFlightTarget(actor, target);
    const contact=guardChallenge(actor,target,c,delta),ramming=contact?.phase==='drive';
    if(contact)target=contact.target;
    stepFlight(actor, target, c, delta, { ...options, defending: true,
      ...(ramming?{maxSpeed:challenge.speed/window.DroneFlightDynamics.airframe.speedScale,ramming:true,rammingTarget:contact.enemy}:{}),
      intercepting: ramming||guardDecisions.get(actor)?.role === 'intercept',
      guardRing: options.guardRing || actor.guardRing });
  }
  function guardRecoveryVelocity(actor, delta) {
    const formation = guardFormations.get(actor);
    if (!formation || actor.altitudeState !== 'flying') return null;
    const c = actor.constants || formation.constants;
    if (!c) return null;
    formation.constants = c;
    const observation = guardDecisions.get(actor);
    if (c.tactics?.activeIntercept && observation) {
      observation.remaining -= Math.max(0, Math.min(delta, .05));
      if (observation.constants !== c || observation.remaining <= 0) {
        // The rigid-body wrapper owns movement during a tumble and skips the
        // page controller. Keep observing at the same tier's decision cadence
        // instead of freezing the pre-contact home/attack target indefinitely.
        guardTarget(actor, observation.enemy, formation.ring, c, formation.partners, formation.standoff);
      }
    }
    let target = guardFlightTarget(actor, guardHome(actor, formation.ring, formation.partners, formation.standoff));
    const contact=guardChallenge(actor,target,c,delta),ramming=contact?.phase==='drive';if(contact)target=contact.target;
    const plan = safeFlight(actor, target, c, { defending: true, guardRing: formation.ring,
      ramming,rammingTarget:contact?.enemy,
      intercepting: guardDecisions.get(actor)?.role === 'intercept' });
    selectGear(actor, plan.target, c, delta, formation.ring);
    if(ramming&&!plan.cautious){gearStates.get(actor).gear=2;actor.controlGear=2;}
    const dx = plan.target.x - actor.position.x, dy = plan.target.y - actor.position.y, dz = plan.target.z - actor.position.z;
    const distance = Math.hypot(dx, dy, dz);
    const speed = Math.min(ramming?challenge.speed:c.maxSpeed * window.DroneFlightDynamics.airframe.speedScale,
      window.DroneFlightDynamics.levelSpeed(actor), distance * (c.trackingGain ?? 4)) * plan.speedScale;
    const desired = { x: 0, y: 0, z: 0 };
    if (distance > 0.005) for (const [axis, offset] of [['x',dx],['y',dy],['z',dz]]) desired[axis] = offset / distance * speed;
    plan.steer(desired);
    return { ...desired, altitude: plan.target.y };
  }
  function launchGuard(actor, ring, c, partners, delta, standoff = 0.2) {
    // Deploy straight to a useful blocking station while climbing. This uses
    // ordinary thrust, gears, banking and obstacle avoidance at every tier.
    const target = guardHome(actor, ring, partners, standoff);
    actor.targetAltitude = target.y;
    // Precision gearing at the hoop must not limit the climb two metres below
    // it. Surface avoidance still reads the complete arena goal registry.
    stepGuard(actor, target, c, delta, { ring: null, guardRing: ring, maxSpeed: 2.6 / window.DroneFlightDynamics.airframe.speedScale,
      trackingGain: 4 });
    actor.mesh?.position.copy(actor.position);
    const depth = (ring.pos.z - actor.position.z) * ring.forwardSign;
    return depth >= 0.18 && Math.hypot(actor.position.x - target.x, actor.position.y - target.y,
      actor.position.z - target.z) < 0.14;
  }

  function launchAttack(actor, ring, c, delta) {
    let launch = attackLaunches.get(actor);
    if (!launch || launch.ring !== ring) {
      const direction = ring.forwardSign || Math.sign(ring.pos.z - actor.position.z) || 1;
      const own = actor.guardRing && actor.guardRing !== ring ? actor.guardRing : null;
      // Cross our own hoop's plane low down while climbing. Reaching goal
      // height behind that hoop first leaves the striker trapped by keepers.
      launch = { ring, constants: c, direction, startZ: actor.position.z,
        target: { x: actor.position.x, y: ring.pos.y,
          z: own ? own.pos.z + direction * .9 : actor.position.z + direction * 1.25 } };
      attackLaunches.set(actor, launch);
    }
    launch.constants = c;
    actor.targetAltitude = launch.target.y;
    stepFlight(actor, launch.target, c, delta, {
      ring, maxSpeed: 2.2 / window.DroneFlightDynamics.airframe.speedScale, trackingGain: 4
    });
    actor.mesh?.position.copy(actor.position);
    const distance = Math.hypot(actor.position.x - launch.target.x,
      actor.position.y - launch.target.y, actor.position.z - launch.target.z);
    const complete = distance < .18 && Math.abs(actor.position.y - launch.target.y) < .12
      && (actor.position.z - launch.startZ) * launch.direction > .7;
    if (complete) attackLaunches.delete(actor);
    return complete;
  }

  // A pilot changes thrust, never position or collision rules. Use braking
  // distance (including motor lag), not a last-frame push away from a wall.
  function safeFlight(actor, destination, c, options = {}) {
    const settings = window.DroneAISettings;
    const pilot = settings?.safetyProfiles?.[settings.level] || { horizon: 0.9, reaction: 0.14, margin: 0.09 };
    const arena = options.environment || window.DroneArenaCollision;
    const radius = options.radius ?? 0.1, p = actor.position, v = actor.velocity;
    const target = { x: destination.x, y: destination.y, z: destination.z };
    const bankBraking=(actor.constants?.gravity??9.8)*Math.tan(Math.min(0.4,actor.constants?.maxTilt??0.4));
    const braking = Math.max(0.8, Math.min(bankBraking, c.accel));
    const delay = pilot.reaction + 0.3;
    const constraints = [];
    let cautious = false, mode = 'fly', speedScale = 1;
    const dot = (a, b) => a.x * b.x + a.y * b.y + a.z * b.z;
    function surface(normal, clearance, margin = pilot.margin, velocity = { x: 0, y: 0, z: 0 }) {
      const closing = Math.max(0, dot(velocity, normal) - dot(v, normal));
      const stopping = closing * delay + closing * closing / (2 * braking);
      const room = Math.max(0, clearance - margin);
      const allowed = Math.sqrt((braking * delay) ** 2 + 2 * braking * room) - braking * delay;
      // Correct a poor approach before contact, without giving AI an impulse.
      const escape = Math.min(0.7, Math.max(0, margin - clearance) * 3 + Math.max(0, stopping - room) * 1.5);
      if (closing > 0.03 && room < stopping + 0.03) { cautious = true; if (mode === 'fly') mode = 'brake'; }
      constraints.push({ normal, minimum: dot(velocity, normal) - allowed + escape });
    }
    const guardRing = options.defending && options.guardRing;
    if (guardRing) {
      // A front-side destination alone cannot stop traffic avoidance or motor
      // lag from carrying a keeper through the opening. Brake against the
      // live goal plane with normal thrust, including when the hoop swings.
      const frame = goalFrame(guardRing);
      const normal = frame.world({ x: 0, y: 0, z: -guardRing.forwardSign });
      const depth = dot({ x: p.x - guardRing.pos.x, y: p.y - guardRing.pos.y,
        z: p.z - guardRing.pos.z }, normal);
      let returning = guardReturns.get(actor);
      if (returning?.ring !== guardRing || depth >= 0.26) {
        guardReturns.delete(actor); returning = null;
      }
      if (!returning && depth < 0 && actor.altitudeState === 'flying') {
        const local = frame.local({ x: p.x - guardRing.pos.x, y: p.y - guardRing.pos.y, z: p.z - guardRing.pos.z });
        returning = { ring: guardRing, side: Math.sign(local.x) || Math.sign(actor.guardHomeOffsetX) || 1, stage: 'clear' };
        guardReturns.set(actor, returning);
      }
      if (returning) {
        // A teammate may already occupy the narrow opening. Re-enter around
        // the outside of the rim, rather than queue forever behind that keeper.
        const goal = arena?.goals?.get(guardRing);
        const lane = (goal?.radius ?? 0.275) + (goal?.tubeRadius ?? 0.075) + radius + pilot.margin + 0.08;
        const local = frame.local({ x: p.x - guardRing.pos.x, y: p.y - guardRing.pos.y, z: p.z - guardRing.pos.z });
        if (returning.side * local.x >= lane - 0.04) returning.stage = 'cross';
        const offset = frame.world({ x: returning.side * lane, y: 0,
          z: guardRing.forwardSign * (returning.stage === 'clear' ? Math.max(0.32, -depth) : -0.4) });
        target.x = guardRing.pos.x + offset.x;
        target.y = guardRing.pos.y + offset.y;
        target.z = guardRing.pos.z + offset.z;
      } else if (depth < 0.18) {
        // Reacquire the entrance immediately after a knockback, independently
        // of the tactical reaction timer. The hoop alignment below still owns
        // the return path; do not push through its rim from the rear side.
        target.x = guardRing.pos.x + normal.x * 0.4;
        target.y = guardRing.pos.y + normal.y * 0.4;
        target.z = guardRing.pos.z + normal.z * 0.4;
      } else {
        const targetDepth = dot({ x: target.x - guardRing.pos.x, y: target.y - guardRing.pos.y,
          z: target.z - guardRing.pos.z }, normal);
        const correction = Math.max(0, 0.26 - targetDepth);
        target.x += normal.x * correction;
        target.y += normal.y * correction;
        target.z += normal.z * correction;
      }
      // Behind the plane, first align with the opening using the ordinary
      // torus avoidance. Never replace position, velocity or contact physics.
      if (depth >= 0) surface(normal, depth, 0.18, frame.body?.velocity);
    }
    const bounds = arena?.bounds;
    if (bounds) for (const axis of ['x', 'y', 'z']) {
      const lo = bounds[axis][0] + radius, hi = bounds[axis][1] - radius;
      const margin = axis === 'y' ? Math.max(0.17, pilot.margin) : pilot.margin;
      target[axis] = clamp(target[axis], lo + margin, hi - margin);
      const n = { x: 0, y: 0, z: 0 }; n[axis] = 1;
      surface(n, p[axis] - lo, margin);
      surface({ x: -n.x, y: -n.y, z: -n.z }, hi - p[axis], margin);
    }
    // Keep an airborne pilot clear of the floor even without a cage registry.
    if (!bounds) {
      target.y = Math.max(radius + 0.17, target.y);
      surface({ x: 0, y: 1, z: 0 }, p.y - radius, 0.17);
    }
    for (const goal of arena?.goals?.values() || []) {
      const { ring, radius: major, tubeRadius } = goal;
      const frame = goalFrame(ring);
      const local = frame.local({ x: p.x - ring.pos.x, y: p.y - ring.pos.y, z: p.z - ring.pos.z });
      const aim = frame.local({ x: target.x - ring.pos.x, y: target.y - ring.pos.y, z: target.z - ring.pos.z });
      const localVelocity = frame.local(v);
      const { x: dx, y: dy, z: dz } = local;
      const radial = Math.hypot(dx, dy), inner = major - tubeRadius - radius - 0.02;
      const outer = major + tubeRadius + radius + 0.04;
      const side = Math.sign(dz) || -Math.sign(localVelocity.z) || 1;
      const crossing = aim.z * side < -0.01;
      const near = Math.abs(dz) < Math.max(1.0, Math.abs(localVelocity.z) * delay + localVelocity.z ** 2 / (2 * braking) + 0.45);
      if (crossing && near && inner > 0) {
        const fraction = clamp(-dz / (aim.z - dz), 0, 1);
        const pathX = dx + (aim.x - dx) * fraction, pathY = dy + (aim.y - dy) * fraction;
        const pathRadius = Math.hypot(pathX, pathY);
        const time = Math.abs(localVelocity.z) > 0.05 && localVelocity.z * side < 0 ? Math.min(1.2, Math.abs(dz / localVelocity.z)) : 0;
        const projected = Math.hypot(dx + localVelocity.x * time, dy + localVelocity.y * time);
        // Return flights outside the entire hoop remain free to go around it.
        if (pathRadius < outer && (pathRadius > inner || projected > inner || radial > inner)) {
          // Align to the selected legal lane, rather than erasing every
          // flank attack by forcing it back onto the same central blocker.
          const aimRadius = Math.hypot(aim.x, aim.y);
          const keepLane = options.breaching && aimRadius <= inner;
          const approach = frame.world({ x: keepLane ? aim.x : 0, y: keepLane ? aim.y : 0,
            z: side * Math.max(0.32, Math.abs(localVelocity.z) * delay + localVelocity.z ** 2 / (2 * braking) + 0.2) });
          target.x = ring.pos.x + approach.x; target.y = ring.pos.y + approach.y; target.z = ring.pos.z + approach.z;
          cautious = true; mode = 'align';
        }
      }
      // Exact distance to the torus centerline leaves the opening flyable.
      const ux = radial > 1e-8 ? dx / radial : 0, uy = radial > 1e-8 ? dy / radial : 1;
      const offset = { x: dx - major * ux, y: dy - major * uy, z: dz };
      const length = Math.hypot(offset.x, offset.y, offset.z);
      if (length > 1e-8 && length < outer + Math.hypot(v.x, v.y, v.z) * pilot.horizon) {
        const normal = frame.world({ x: offset.x / length, y: offset.y / length, z: offset.z / length });
        const lever = frame.world({ x: major * ux, y: major * uy, z: 0 });
        const velocity = frame.body?.velocity || { x: 0, y: 0, z: 0 }, spin = frame.body?.omega || { x: 0, y: 0, z: 0 };
        surface(normal, length - tubeRadius - radius, 0.012,
          { x: velocity.x + spin.y * lever.z - spin.z * lever.y,
            y: velocity.y + spin.z * lever.x - spin.x * lever.z, z: velocity.z + spin.x * lever.y - spin.y * lever.x });
      }
      // The two suspension cables are solids in the game as well.
      for (const cable of ring.cables || []) {
        const center = cable.position, height = cable.geometry.parameters.height;
        const qy = clamp(p.y, center.y - height / 2, center.y + height / 2);
        capsule({ x: center.x, y: qy, z: center.z }, 0.006, 0.025);
      }
    }
    function capsule(q, thickness, margin = pilot.margin) {
      const d = { x: p.x - q.x, y: p.y - q.y, z: p.z - q.z };
      const length = Math.hypot(d.x, d.y, d.z);
      if (length > 1e-8 && length < radius + thickness + pilot.margin + Math.hypot(v.x, v.y, v.z) * pilot.horizon + 0.1) {
        surface({ x: d.x / length, y: d.y / length, z: d.z / length }, length - radius - thickness, margin);
      }
    }
    for (const tube of arena?.tubes || []) {
      const axis = { x: tube.b.x - tube.a.x, y: tube.b.y - tube.a.y, z: tube.b.z - tube.a.z };
      const squared = dot(axis, axis);
      const t = squared > 0 ? clamp(((p.x - tube.a.x) * axis.x + (p.y - tube.a.y) * axis.y + (p.z - tube.a.z) * axis.z) / squared, 0, 1) : 0;
      capsule({ x: tube.a.x + axis.x * t, y: tube.a.y + axis.y * t, z: tube.a.z + axis.z * t }, tube.radius);
    }
    const impact = window.DroneFlightDynamics?.collisionState(actor);
    let recovery = null;
    if (impact) {
      const severity = clamp((impact.strength ?? 1.5) / 1.5, 0, 1);
      const duration = 0.25 + 0.85 * severity;
      const tilt = Math.hypot(impact.x || 0, impact.z || 0);
      if (impact.age < duration || impact.age < 1.6 && tilt > 0.12) {
        // Choose a short escape from the struck surface before reacquiring the
        // role's live target. Preserve rebound momentum; never reset velocity.
        const normal = impact.normal || { x: 0, y: 0, z: 0 };
        const blend = clamp((impact.age - 0.12) / Math.max(0.1, duration - 0.12), 0, 1);
        const retreat = (0.12 + 0.18 * severity) * (1 - blend);
        const altitude = Math.max(0.38, p.y + Math.max(0, -v.y) * delay + v.y * v.y / (2 * braking));
        const brace = !!(guardRing && c.tactics?.holdLine && impact.source === 'cage');
        const press = !!(options.breaching && impact.source === 'cage');
        const disengage = !!(options.disengaging && impact.source === 'cage');
        if (!brace && !press && !disengage) {
          target.x = p.x + (target.x - p.x) * blend + normal.x * retreat;
          target.z = p.z + (target.z - p.z) * blend + normal.z * retreat;
          target.y = Math.max(p.y + (target.y - p.y) * blend + normal.y * retreat,
            v.y < -0.05 || p.y < 0.38 ? altitude : p.y);
        } else if (p.y < 0.38) target.y = Math.max(target.y, altitude);
        if (bounds) for (const axis of ['x', 'y', 'z']) {
          const margin = axis === 'y' ? 0.17 : pilot.margin;
          target[axis] = clamp(target[axis], bounds[axis][0] + radius + margin, bounds[axis][1] - radius - margin);
        }
        speedScale = brace || press || disengage ? 1 : 0.3 + 0.7 * blend;
        cautious = true; mode = brace ? 'brace' : press ? 'press' : disengage ? 'disengage' : 'recover';
        recovery = { normal, blend, severity, altitude, brace, press, disengage };
      }
    }
    const traffic = arena?.traffic || [];
    const actorIndex = traffic.findIndex(other => other.actor === actor);
    function steer(desired) {
      if (recovery) {
        const { normal, blend, severity } = recovery;
        const outward = dot(v, normal);
        const escape = Math.min(0.65, Math.max(0.12 * severity, outward * (1 - blend)));
        const correction = recovery.brace || recovery.press || recovery.disengage ? 0 : Math.max(0, escape * (1 - blend) - dot(desired, normal));
        desired.x += normal.x * correction; desired.y += normal.y * correction; desired.z += normal.z * correction;
        // Arrest a descending rebound promptly while letting surface and
        // traffic constraints choose the available direction near the ceiling.
        if (v.y < -0.05) desired.y = Math.max(desired.y, Math.min(0.6, -v.y * 0.65 + 0.1));
      }
      for (let index = 0; index < traffic.length; index++) {
        const other = traffic[index];
        if (other.actor === actor) continue;
        // Traffic is sampled once per rendered frame. Advance that sample to
        // this physics substep so a 30 FPS pilot does not avoid a stale body.
        const time = options.trafficTime || 0;
        const r = { x: p.x - other.position.x - other.velocity.x * time,
          y: p.y - other.position.y - other.velocity.y * time,
          z: p.z - other.position.z - other.velocity.z * time };
        const relative = { x: desired.x - other.velocity.x, y: desired.y - other.velocity.y, z: desired.z - other.velocity.z };
        // Momentum still matters after a pilot selects a slower gear.
        const actual = { x: v.x - other.velocity.x, y: v.y - other.velocity.y, z: v.z - other.velocity.z };
        const missAt = (velocity, horizon = pilot.horizon) => {
          const squared = dot(velocity, velocity);
          const time = squared > 1e-8 ? clamp(-dot(r, velocity) / squared, 0, horizon) : 0;
          return Math.hypot(r.x + velocity.x * time, r.y + velocity.y * time, r.z + velocity.z * time);
        };
        // A keeper brakes at its station. Extrapolating the desired velocity
        // past that destination falsely predicts crossing the adjacent keeper
        // and makes the pair yield forever outside the opening. Actual inertia
        // retains the full safety horizon, independently of this planned stop.
        const arrival = guardRing ? Math.hypot(target.x - p.x, target.y - p.y, target.z - p.z)
          / Math.max(0.001, Math.hypot(desired.x, desired.y, desired.z)) : pilot.horizon;
        const miss = Math.min(missAt(relative, Math.min(pilot.horizon, arrival)), missAt(actual)), distance = Math.hypot(r.x, r.y, r.z);
        const separation = 2 * radius + (actor.team && actor.team === other.team ? pilot.margin : 0.035);
        if (miss >= separation || distance > 2 * radius + (c.maxSpeed + Math.hypot(other.velocity.x, other.velocity.y, other.velocity.z)) * pilot.horizon) continue;
        const enemy = actor.team && other.team && actor.team !== other.team;
        if (guardRing && !enemy && actor.team && actor.team === other.team
          && guardFormations.get(actor)?.teammates.includes(other.actor)) {
          // These two keepers intentionally press their cages together. Keep
          // normal motor control and real cage collisions; do not sidestep or
          // add a safety gap between the designated defensive partners.
          continue;
        }
        if (options.breaching && enemy && (!recovery || recovery.press) && options.blockers?.includes(other.actor)) {
          // The committed scoring lane may require contact with an opponent.
          // Keep thrust toward the goal; the ordinary collision solver alone
          // moves both cages. Teammates and solid surfaces still constrain us.
          if (!recovery?.press) mode = 'breach';
          continue;
        }
        if(options.ramming&&enemy&&other.actor===options.rammingTarget){
          // A committed bump uses ordinary full-power flight and the shared
          // contact solver. No extra impulse, mass or invulnerability is added.
          if(mode!=='recover'&&mode!=='brace')mode='challenge';continue;
        }
        if (options.defending && c.tactics?.holdLine && enemy && (!recovery || recovery.brace) && other.actor?.readyToScore !== false) {
          // A goalkeeper's job is to occupy the scoring lane. The general
          // traffic reflex used to sidestep the very attacker being blocked.
          // Brake our own charge, hold the line, and let normal collision
          // physics handle contact and recovery without changing either body.
          if (distance > 1e-8) {
            const toward = -(desired.x * r.x + desired.y * r.y + desired.z * r.z) / distance;
            const brake = Math.max(0, toward - (options.intercepting ? Math.min(c.maxSpeed, 1.4) : 0.65));
            desired.x += r.x / distance * brake;
            desired.y += r.y / distance * brake;
            desired.z += r.z / distance * brake;
          }
          if (mode !== 'recover' && mode !== 'brace') mode = 'intercept';
          continue;
        }
        // A stable passing side prevents two head-on pilots mirroring back
        // and forth. Sidestep horizontally so avoidance never dives at floor.
        const sign = actorIndex < index ? 1 : -1;
        // Keepers yield away from the teammate's actual side, rather than
        // steering both cages across each other's assigned blocking station.
        const frame = guardRing && goalFrame(guardRing);
        const lateral = frame && frame.local(r).x;
        const n = frame ? frame.world({ x: Math.abs(lateral) > 0.01 ? Math.sign(lateral)
          : Math.sign((actor.guardHomeOffsetX || 0) - (other.actor?.guardHomeOffsetX || 0)) || sign, y: 0, z: 0 })
          : Math.abs(r.z) >= Math.abs(r.x) ? { x: sign, y: 0, z: 0 } : { x: 0, y: 0, z: sign };
        const urgency = clamp((separation - miss) / separation, 0.15, 1);
        desired.x = desired.x * (1 - 0.65 * urgency) + n.x * Math.min(c.maxSpeed, 0.65) * urgency;
        desired.z = desired.z * (1 - 0.65 * urgency) + n.z * Math.min(c.maxSpeed, 0.65) * urgency;
        if (distance > 1e-8) surface({ x: r.x / distance, y: r.y / distance, z: r.z / distance }, distance - 2 * radius, separation - 2 * radius, other.velocity);
        cautious = true; if (mode !== 'recover') mode = 'yield';
      }
      // Recheck intersecting constraints (e.g. another pilot near a corner).
      for (let pass = 0; pass < 2; pass++) for (const { normal, minimum } of constraints) {
        const correction = Math.max(0, minimum - dot(desired, normal));
        desired.x += normal.x * correction; desired.y += normal.y * correction; desired.z += normal.z * correction;
      }
      safetyStates.set(actor, { mode, cautious, recovery: recovery ? { blend: recovery.blend, severity: recovery.severity } : null });
      if (cautious) {
        // Precision gear has barely enough collective/torque headroom after
        // a cage strike. Select the existing middle gear while bracing; the
        // physical gyro and motors still deliver every correction normally.
        gearStates.get(actor).gear = recovery?.brace || recovery?.press || recovery?.disengage ? 1 : 0;
        actor.controlGear = recovery?.brace || recovery?.press || recovery?.disengage ? 1 : 0;
      }
    }
    return { target, cautious, speedScale, steer, bracing: !!(recovery?.brace || recovery?.press || recovery?.disengage) };
  }

  function flightPose(actor, delta) {
    const state = flightStates.get(actor);
    if (!state || !actor.mesh || !['flying', 'takingoff'].includes(actor.altitudeState)) return;
    const dt = clamp(delta, 0, 0.05), c = actor.constants || {};
    // The role handler chooses where to look (opponent for guards, travel for
    // attackers). Follow that heading along the shortest arc, at pilot rates.
    const difference = Math.atan2(Math.sin(actor.mesh.rotation.y - state.yaw), Math.cos(actor.mesh.rotation.y - state.yaw));
    const maxYaw = c.yawSpeed ?? 1.6;
    const desiredRate = clamp(difference * 4, -maxYaw, maxYaw);
    state.yawRate += (desiredRate - state.yawRate) * (1 - Math.exp(-6 * dt));
    state.yaw += state.yawRate * dt;
    const sin = Math.sin(state.yaw), cos = Math.cos(state.yaw);
    // Bank into acceleration and counter-bank while braking, in local axes.
    // Compensate for the same horizontal drag used by the player dynamics.
    const mass = c.mass ?? 0.3, gravity = c.gravity ?? 9.8;
    const drag = window.DroneFlightDynamics.horizontalDrag(actor) / mass;
    const ax = state.x + actor.velocity.x * drag, az = state.z + actor.velocity.z * drag;
    const tiltLimit = Math.min(0.4, c.maxTilt ?? 0.4);
    let pitch = Math.atan2(sin * ax + cos * az, gravity);
    let roll = -Math.atan2(cos * ax - sin * az, gravity);
    const tilt = Math.hypot(pitch, roll);
    if (tilt > tiltLimit) { pitch *= tiltLimit / tilt; roll *= tiltLimit / tilt; }
    const alpha = 1 - Math.exp(-6 * dt);
    state.pitch += (pitch - state.pitch) * alpha;
    state.roll += (roll - state.roll) * alpha;
    actor.mesh.rotation.set(state.pitch, state.yaw, state.roll, 'YXZ');
    if (actor.rotation) actor.rotation.y = state.yaw;
  }

  function syncFlightPose(actor) {
    const state = flightStates.get(actor);
    if (!state || !actor.mesh) return;
    // During a physical tumble the body owns orientation. Resume piloting
    // from that real attitude, rather than snapping to the pre-impact yaw.
    state.yaw = actor.mesh.rotation.y;
    state.pitch = actor.mesh.rotation.x;
    state.roll = actor.mesh.rotation.z;
    state.yawRate = state.x = state.y = state.z = 0;
  }

  // The launch zone is behind our own hoop. Depart around its outer edge,
  // instead of repeatedly aligning with an opening occupied by our keepers.
  function attackDeparture(actor, targetRing) {
    const own = actor.guardRing;
    if (!own || own === targetRing || actor.readyToScore === false || actor.phase !== 'advance') return null;
    const frame = goalFrame(own);
    const p = frame.local({ x: actor.position.x - own.pos.x, y: actor.position.y - own.pos.y, z: actor.position.z - own.pos.z });
    const depth = -p.z * own.forwardSign;
    let departure = attackDepartures.get(actor);
    if (!departure && depth >= .18) return null;
    if (!departure || departure.ring !== own) {
      const launchSide = frame.local({ x: actor.team === 'blue' ? -1 : 1, y: 0, z: 0 });
      departure = { ring: own, stage: 'clear', side: Math.abs(p.x) > .1 ? Math.sign(p.x) : Math.sign(launchSide.x) || 1 };
      attackDepartures.set(actor, departure);
    }
    if (depth >= .6) { attackDepartures.delete(actor); return null; }
    const goal = window.DroneArenaCollision?.goals?.get(own);
    const clearance = Math.max(.8, (goal?.radius ?? .275) + (goal?.tubeRadius ?? .075) + .1 + .2);
    if (departure.stage === 'clear' && departure.side * p.x >= clearance - .08) departure.stage = 'forward';
    const offset = frame.world({ x: departure.side * clearance, y: 0,
      z: departure.stage === 'clear' ? own.forwardSign * Math.max(.35, -depth) : -own.forwardSign * .75 });
    return { x: own.pos.x + offset.x, y: own.pos.y + offset.y, z: own.pos.z + offset.z };
  }

  function attackMemory(actor) {
    let memory = attackMemories.get(actor);
    if (!memory) {
      memory = { attempts: 0, successes: 0, failures: 0, lanes: { '-1': 0, '1': 0 },
        active: null, ready: actor.readyToScore, phase: actor.phase, replans:0, navigation:null };
      attackMemories.set(actor, memory);
    }
    return memory;
  }
  function finishAttack(actor, success) {
    const memory = attackMemory(actor);
    if (!memory.active) return;
    const side = memory.active.side;
    // Bounded, match-local experience: a stopped route loses priority; a
    // successful route retains credit without becoming a permanent script.
    memory.lanes[side] = clamp(memory.lanes[side] * .7 + (success ? -.45 : 1), -1, 3);
    memory[success ? 'successes' : 'failures']++;
    memory.active = null;
  }
  function observeAttackOutcome(actor) {
    const memory = attackMemory(actor);
    if (memory.ready !== false && actor.readyToScore === false) finishAttack(actor, true);
    else if (memory.phase === 'advance' && (actor.phase === 'regroup' || actor.phase === 'loopback')) finishAttack(actor, false);
    memory.ready = actor.readyToScore; memory.phase = actor.phase;
  }
  function chooseAttackFlank(actor, ring, c, position, seen, preferred) {
    const memory = attackMemory(actor);
    const clearance = side => {
      const end = { x: side * .97, y: -.15 * (Math.sign(position.x) || 1), z: -ring.forwardSign * .2 };
      const d = { x: end.x-position.x, y: end.y-position.y, z: end.z-position.z };
      const length = d.x*d.x+d.y*d.y+d.z*d.z;
      return Math.min(.6, ...seen.map(b => {
        const t = clamp(((b.p.x-position.x)*d.x+(b.p.y-position.y)*d.y+(b.p.z-position.z)*d.z)/Math.max(.0001,length),0,1);
        return Math.hypot(position.x+d.x*t-b.p.x,position.y+d.y*t-b.p.y,position.z+d.z*t-b.p.z);
      }));
    };
    const score = side => clearance(side) + (side === preferred ? .25 : 0) - memory.lanes[side] * (c.adaptiveRetry ? .45 : 0);
    const side = score(-preferred) > score(preferred) + .04 ? -preferred : preferred;
    memory.attempts++; memory.active = { side };
    return side;
  }

  function regroupNavigation(actor, ring, requested, delta) {
    const memory=attackMemory(actor);
    if(actor.altitudeState!=='flying'||actor.readyToScore===false){memory.navigation=null;memory.escape=null;return requested;}
    if(memory.escape){
      const escape=memory.escape;
      if(Math.hypot(actor.position.x-escape.x,actor.position.y-escape.y,actor.position.z-escape.z)<.22){
        memory.escape=null;memory.navigation=null;actor.phase='advance';actor.reactionTimer=0;
        return requested;
      }
      actor.phase='regroup';
      if(actor.regroupTarget?.set)actor.regroupTarget.set(escape.x,escape.y,escape.z);else actor.regroupTarget={...escape};
      return{...escape};
    }
    if(actor.phase!=='regroup'){
      const n=memory.navigation;
      if(n&&Math.hypot(actor.position.x-n.origin.x,actor.position.y-n.origin.y,actor.position.z-n.origin.z)>.45)memory.navigation=null;
      return requested;
    }
    const target=actor.regroupTarget||requested;
    if(!target)return requested;
    const distance=Math.hypot(actor.position.x-target.x,actor.position.y-target.y,actor.position.z-target.z);
    let n=memory.navigation;
    if(!n){n={age:0,stalled:0,best:distance,origin:{...actor.position}};memory.navigation=n;}
    const dt=clamp(delta,0,.05);n.age+=dt;
    if(distance<n.best-.04){n.best=distance;n.stalled=0;}else n.stalled+=dt;
    const displaced=Math.hypot(actor.position.x-n.origin.x,actor.position.y-n.origin.y,actor.position.z-n.origin.z);
    if(n.age>3&&displaced<.35 || distance>.25&&(n.age>3&&n.stalled>2 || n.age>8)) {
      const frame=goalFrame(ring),p=frame.local({x:actor.position.x-ring.pos.x,y:actor.position.y-ring.pos.y,z:actor.position.z-ring.pos.z});
      const depth=-p.z*ring.forwardSign,side=Math.sign(p.x)||1;
      // A random regroup point can sit inside/behind the torus or beyond a
      // safe surface boundary. Replan away from its current side first; actual
      // arrival is still required before the page resumes attacking.
      const local=depth>.05?{x:side*.6,y:-.2,z:-ring.forwardSign*1.25}
        :{x:side*.8,y:-.35,z:ring.forwardSign*.55};
      const offset=frame.world(local),next={x:ring.pos.x+offset.x,y:ring.pos.y+offset.y,z:ring.pos.z+offset.z};
      const bounds=window.DroneArenaCollision?.bounds;
      if(bounds)for(const axis of ['x','y','z'])next[axis]=clamp(next[axis],bounds[axis][0]+.22,bounds[axis][1]-.22);
      if(actor.regroupTarget?.set)actor.regroupTarget.set(next.x,next.y,next.z);else actor.regroupTarget=next;
      actor.reactionTimer=0;memory.replans++;memory.navigation=null;memory.escape={...next};
      return next;
    }
    return{x:target.x,y:target.y,z:target.z};
  }

  function attackCruise(actor, ring, c, requestedSpeed = c.maxSpeed, departure = false) {
    const scale = window.DroneFlightDynamics.airframe.speedScale;
    const precision = requestedSpeed * scale;
    const depth = (ring.pos.z - actor.position.z) * ring.forwardSign;
    // Legacy profile values were quarter-scaled, limiting a 2.4 m/s request
    // to 0.6 m/s across the entire arena. Use normal cruise in clear space;
    // retain the existing precision speed for the defended final approach.
    let speed = precision;
    if (departure) speed = Math.max(precision, 1.4);
    else if (actor.phase === 'retreat') speed = Math.max(precision, 1.8);
    else if (actor.phase === 'advance' && (!actor.jukePhase || actor.jukePhase === 'none')) {
      const t = clamp((depth - 2) / 1, 0, 1);
      speed += (Math.min(c.maxSpeed, 2.4) - precision) * t;
    }
    return speed / scale;
  }

  // Keep a sampled lane through the final approach instead of blending a new
  // weave/juke into the opening every frame. Decisions retain each tier's delay.
  function stepAttack(actor, requested, c, delta, options = {}) {
    const ring = options.ring || actor.targetRing;
    observeAttackOutcome(actor);
    if(ring)requested=regroupNavigation(actor,ring,requested,delta);
    const advancing = options.advancing ?? actor.phase === 'advance';
    attackFlights.set(actor, { ring, constants: c, blockers: options.blockers || [], requested: { ...requested }, options, advancing, phase:actor.phase });
    let maxSpeed = options.maxSpeed ?? c.maxSpeed;
    const departure = ring && advancing && attackDeparture(actor, ring);
    if (departure) { requested = departure; attackStates.delete(actor); }
    else if (ring && advancing) {
      const depth = (ring.pos.z - actor.position.z) * ring.forwardSign;
      // Finish a feint before the braking/alignment zone, never in the hoop.
      if (actor.jukePhase && actor.jukePhase !== 'none' && depth < (c.approachDistance ?? 1.4)) {
        actor.jukePhase = 'none';
        requested = { x: ring.pos.x, y: ring.pos.y, z: ring.pos.z + ring.forwardSign * 0.45 };
        maxSpeed = c.maxSpeed;
      }
      if (!actor.jukePhase || actor.jukePhase === 'none') requested = attackTarget(actor, ring, c, options.blockers || [], requested, delta, options);
    } else attackStates.delete(actor);
    maxSpeed = ring ? attackCruise(actor, ring, c, maxSpeed, !!departure) : maxSpeed;
    const state = attackStates.get(actor);
    if (state?.maneuver?.stage === 'cut') maxSpeed = Math.min(maxSpeed, 1.1);
    const depth = ring ? (ring.pos.z - actor.position.z) * ring.forwardSign : Infinity;
    const breaching = !!(ring && advancing && actor.readyToScore !== false
      && depth > -0.1 && depth < (c.approachDistance ?? 1.4) && state?.mode !== 'reset');
    stepFlight(actor, requested, c, delta, { ...options, ring, breaching,
      disengaging: actor.phase && actor.phase!=='advance' || state?.mode==='reset', maxSpeed, trackingGain: c.trackingGain ?? 2 });
  }

  function attackTransitTarget(actor, ring) {
    const p = actor.position;
    const goal = window.DroneArenaCollision?.goals?.get(ring);
    const radius = (goal?.radius ?? .275) + (goal?.tubeRadius ?? .075);
    const ceiling = window.DroneArenaCollision?.bounds?.y?.[1] ?? 3;
    const over = ring.pos.y + radius + .4;
    const clearY = over <= Math.min(2.6,ceiling-.4) ? over : Math.max(.45,ring.pos.y-radius-.4);
    const resume = () => { actor.phase='advance';actor.reactionTimer=0;actor.jukePhase='none';attackStates.delete(actor);return null; };
    if(actor.phase==='regroup') {
      const target=actor.regroupTarget;
      if(!target)return resume();
      return Math.hypot(p.x-target.x,p.y-target.y,p.z-target.z)<.25?resume():{x:target.x,y:target.y,z:target.z};
    }
    if(actor.phase==='descend')return Math.abs(p.y-ring.pos.y)<.15?resume():{x:p.x,y:ring.pos.y,z:p.z};
    if(actor.phase==='loopback') {
      if(Math.abs(p.y-clearY)>.1)return{x:p.x,y:clearY,z:p.z};
      const depth=(ring.pos.z-p.z)*ring.forwardSign;
      return depth>=.6?resume():{x:Math.sign(p.x)*.4,y:clearY,z:ring.pos.z-ring.forwardSign*.9};
    }
    if(actor.phase==='retreat')return Math.abs(p.y-clearY)>.1?{x:p.x,y:clearY,z:p.z}
      :{x:0,y:clearY,z:-ring.forwardSign*.35};
    return null;
  }
  function attackRecoveryVelocity(actor, delta) {
    const launch = attackLaunches.get(actor);
    if (launch && actor.altitudeState === 'takingoff') {
      const c = actor.constants || launch.constants;
      const plan = safeFlight(actor, launch.target, c, { ring: launch.ring });
      selectGear(actor, plan.target, c, delta, launch.ring);
      const d = { x: plan.target.x - actor.position.x, y: plan.target.y - actor.position.y, z: plan.target.z - actor.position.z };
      const distance = Math.hypot(d.x, d.y, d.z);
      const speed = Math.min(2.2, window.DroneFlightDynamics.levelSpeed(actor), distance * 4) * plan.speedScale;
      const desired = { x: 0, y: 0, z: 0 };
      if (distance > .005) for (const axis of ['x', 'y', 'z']) desired[axis] = d[axis] / distance * speed;
      plan.steer(desired);
      return { ...desired, altitude: plan.target.y };
    }
    const flight = attackFlights.get(actor);
    observeAttackOutcome(actor);
    if (!flight?.ring || actor.altitudeState !== 'flying') return null;
    const c = actor.constants || flight.constants, ring = flight.ring;
    regroupNavigation(actor,ring,actor.regroupTarget||flight.requested,delta);
    let transit=actor.phase&&actor.phase!=='advance'?attackTransitTarget(actor,ring):null;
    const advancing=actor.phase?actor.phase==='advance':flight.advancing;
    if(!transit&&!advancing)return null;
    const depth = (ring.pos.z - actor.position.z) * ring.forwardSign;
    if (advancing && depth < -0.18 && actor.phase==='advance') {
      actor.phase='loopback';transit=attackTransitTarget(actor,ring);
    }
    if(!transit&&actor.readyToScore===false)return null;
    const departure = !transit && attackDeparture(actor, ring);
    const requested=advancing&&!flight.advancing?{x:ring.pos.x,y:ring.pos.y,z:ring.pos.z+ring.forwardSign*.45}:flight.requested;
    const target = transit || departure || attackTarget(actor, ring, c, flight.blockers, requested, delta, flight.options);
    const breaching = !transit && depth > -0.1 && depth < (c.approachDistance ?? 1.4) && attackStates.get(actor)?.mode !== 'reset';
    const plan = safeFlight(actor, target, c, { ...flight.options, ring, breaching,
      disengaging: !!transit || attackStates.get(actor)?.mode==='reset', blockers: flight.blockers });
    selectGear(actor, plan.target, c, delta, ring);
    const d = { x: plan.target.x - actor.position.x, y: plan.target.y - actor.position.y, z: plan.target.z - actor.position.z };
    const distance = Math.hypot(d.x,d.y,d.z);
    const cruise = !transit && attackStates.get(actor)?.maneuver?.stage === 'cut' ? Math.min(c.maxSpeed,1.1)
      : attackCruise(actor, ring, c, flight.options.maxSpeed ?? c.maxSpeed, !!departure);
    const speed = Math.min(cruise * window.DroneFlightDynamics.airframe.speedScale,
      window.DroneFlightDynamics.levelSpeed(actor), distance * (c.trackingGain ?? 4)) * plan.speedScale;
    const desired = { x: 0, y: 0, z: 0 };
    if (distance > .005) for (const axis of ['x','y','z']) desired[axis] = d[axis] / distance * speed;
    plan.steer(desired);
    return { ...desired, altitude: plan.target.y };
  }

  function attackTarget(actor, ring, c, blockers, requested, delta, options = {}) {
    const depth = (ring.pos.z - actor.position.z) * ring.forwardSign;
    if (actor.readyToScore === false || depth < -0.25) {
      attackStates.delete(actor);
      return requested;
    }
    let state = attackStates.get(actor);
    if (!state || state.constants !== c || state.ring !== ring) {
      state = { constants: c, ring, remaining: 0, locked: false, scans: 0,
        mode: 'attack', bestDepth: depth, stalled: 0, retries: 0 };
      attackStates.set(actor, state);
    }
    const dt = clamp(delta, 0, 0.05);
    if (state.mode === 'reset') {
      state.resetTime += dt;
      const retreatDepth = (ring.pos.z - state.resetTarget.z) * ring.forwardSign;
      const aligned = Math.hypot(actor.position.x - state.resetTarget.x, actor.position.y - state.resetTarget.y) < 0.12;
      if ((depth < retreatDepth - 0.08 || !aligned) && state.resetTime < 2.8) return { ...state.resetTarget };
      state.mode = 'attack'; state.locked = false; state.remaining = 0;
      state.bestDepth = depth; state.stalled = 0;
      state.maneuver = null;
    }
    // Even a beginner abandons repeated head-on failures. Advanced pilots
    // read the defense immediately; easy tries its direct run three times.
    if (c.tactics?.flankDefense || state.retries >= 3) {
      const waypoint = breakthroughTarget(actor, ring, c, blockers, state, dt);
      if (waypoint) return waypoint;
    }
    if (depth < state.bestDepth - 0.04 || depth > (c.approachDistance ?? 1.4)) {
      state.bestDepth = depth; state.stalled = 0;
    } else if (depth > 0.05 && blockers.some(b => {
      const p = b.position || b;
      return (p.z - actor.position.z) * ring.forwardSign >= -0.12
        && (ring.pos.z - p.z) * ring.forwardSign >= -0.12
        && Math.hypot(p.x - actor.position.x, p.y - actor.position.y, p.z - actor.position.z) < 0.65;
    })) state.stalled += dt;
    else state.stalled = 0;
    if (state.stalled >= (c.retryAfter ?? 1.7)) {
      finishAttack(actor, false);
      state.mode = 'reset'; state.resetTime = 0; state.retries++;
      state.locked = false;
      // Back off under normal thrust to gain room for a fresh approach.
      // Smarter pilots change the entry side instead of repeating the jam.
      const side = state.route?.x >= ring.pos.x ? -1 : 1;
      state.resetTarget = { x: ring.pos.x + (c.adaptiveRetry ? side * 0.28 : 0),
        y: ring.pos.y + (c.adaptiveRetry ? (state.route?.y >= ring.pos.y ? -0.18 : 0.18) : 0),
        z: ring.pos.z - ring.forwardSign * Math.max(0.9, depth + 0.55) };
      return { ...state.resetTarget };
    }
    if (depth > (c.laneCommitDistance ?? 0.65) + 0.25) state.locked = false;
    state.remaining -= dt;
    if (!state.route || !state.locked && state.remaining <= 0) {
      const random = options.random || Math.random;
      const min = c.gapReactionMin ?? c.reactionMin;
      const max = c.gapReactionMax ?? c.reactionMax;
      state.remaining = min + random() * (max - min);
      const cage = options.cageRadius ?? 0.1, clear = options.clearRadius ?? 0.2;
      const gap = findGap(ring, actor.position, blockers, clear, cage, {
        speed: Math.max(0.35, Math.hypot(actor.velocity.x, actor.velocity.y, actor.velocity.z), c.maxSpeed * 0.7),
        horizon: c.tactics?.predictiveGap ? c.predictionHorizon ?? 0.4 : 0
      });
      const commitment = c.commitToGapMax ?? 0.65;
      const radius = (gap?.radius || 0) * commitment;
      let x = Math.cos(gap?.angle || 0) * radius;
      let y = Math.sin(gap?.angle || 0) * radius;
      // Errors are decisions too, not fresh random thrust at rendering speed.
      if (gap) {
        x += (random() - 0.5) * c.aimError;
        y += (random() - 0.5) * c.aimError;
      }
      const limit = Math.max(0, clear - cage - 0.025), length = Math.hypot(x, y);
      if (length > limit) { x *= limit / length; y *= limit / length; }
      state.route = { x: ring.pos.x + x, y: ring.pos.y + y };
      state.blocked = !!gap && !gap.reliable;
      state.hasBlockers = !!gap;
      state.scans++;
    }
    if (depth < (c.laneCommitDistance ?? 0.65)) state.locked = true;
    const approach = c.approachDistance ?? 1.4;
    const blend = state.hasBlockers ? clamp((approach - depth) / Math.max(0.2, approach - 0.8), 0, 1) : 1;
    return {
      x: requested.x + (state.route.x - requested.x) * blend,
      y: requested.y + (state.route.y - requested.y) * blend,
      z: requested.z
    };
  }

  function breakthroughTarget(actor, ring, c, blockers, state, dt) {
    const frame = goalFrame(ring);
    const p = frame.local({ x: actor.position.x-ring.pos.x, y: actor.position.y-ring.pos.y, z: actor.position.z-ring.pos.z });
    const depth = -p.z * ring.forwardSign;
    const defense = blockers.map(b=>({ actor:b, p:frame.local({ x:(b.position||b).x-ring.pos.x,
      y:(b.position||b).y-ring.pos.y,z:(b.position||b).z-ring.pos.z }) }))
      .filter(b=>b.actor.altitudeState!=='grounded'&&-b.p.z*ring.forwardSign>-.1&&-b.p.z*ring.forwardSign<1.5&&Math.hypot(b.p.x,b.p.y)<1.0);
    let m = state.maneuver;
    if (!m && defense.length>=2 && depth>.85) {
      m=state.maneuver={stage:'draw',age:0,side:Math.sign(p.x)||1,flank:0};
    }
    if (!m || m.stage==='finish') return null;
    if (!defense.length) {m.stage='finish';state.remaining=0;state.locked=false;return null;}
    m.age+=dt;
    if (m.stage==='draw') {
      m.observeRemaining=(m.observeRemaining||0)-dt;
      if(!m.seen||m.observeRemaining<=0){
        m.seen=defense.map(b=>({p:{...b.p}}));m.observeRemaining=((c.reactionMin??.2)+(c.reactionMax??.2))/2;
      }
      const forward=m.seen.reduce((best,b)=>!best||-b.p.z*ring.forwardSign>-best.p.z*ring.forwardSign?b:best,null);
      if (depth<1.85 && (forward&&-forward.p.z*ring.forwardSign>.6&&m.age>.35 || m.age>2.0) || m.age>6.0) {
        m.flank=chooseAttackFlank(actor,ring,c,p,m.seen,-(Math.sign(forward?.p.x)||m.side));m.stage='slip';m.age=0;
      }
    } else if (m.stage==='slip') {
      if (depth<.3&&m.flank*p.x>.82) {m.stage='cut';m.age=0;}
      else if(m.age>5.0){m.stage='finish';state.remaining=0;state.locked=false;}
    } else if (m.stage==='cut') {
      if (Math.hypot(p.x,p.y)<.075&&depth<.4 || m.age>5.0) {
        m.stage='finish';state.remaining=0;state.locked=false;state.bestDepth=depth;state.stalled=0;
      }
    }
    if(m.stage==='finish')return null;
    const local=m.stage==='draw'?{x:m.side*.5,y:m.side*.12,z:-ring.forwardSign*1.4}
      :m.stage==='slip'?{x:m.flank*.97,y:-m.side*.15,z:-ring.forwardSign*.2}
        :{x:0,y:0,z:-ring.forwardSign*.2};
    const offset=frame.world(local),target={x:ring.pos.x+offset.x,y:ring.pos.y+offset.y,z:ring.pos.z+offset.z};
    const bounds=window.DroneArenaCollision?.bounds;
    if(bounds)for(const axis of ['x','y','z'])target[axis]=clamp(target[axis],bounds[axis][0]+.22,bounds[axis][1]-.22);
    return target;
  }

  const gapSamples = Array.from({length:36}, (_, i) => {
    const angle=i/36*Math.PI*2;
    return {x:Math.cos(angle),y:Math.sin(angle)};
  });
  function closestSeparation(x, y, z, vx, vy, vz, duration) {
    const squared=vx*vx+vy*vy+vz*vz;
    const t=squared>1e-9?clamp(-(x*vx+y*vy+z*vz)/squared,0,duration):0;
    return Math.hypot(x+vx*t,y+vy*t,z+vz*t);
  }
  function findGap(ring, attackerPos, blockers, clearRadius, cageRadius, prediction = {}) {
    // Score actual 3D flight segments. Defenders behind the attacker/goal cannot block this run.
    const lo = Math.min(attackerPos.z, ring.pos.z) - cageRadius;
    const hi = Math.max(attackerPos.z, ring.pos.z) + cageRadius;
    const horizon = prediction.horizon ?? 0;
    const active = blockers.map(b => ({ position: b.position || b, velocity: b.velocity || { x: 0, y: 0, z: 0 } }))
      .filter(b => Math.max(b.position.z, b.position.z + b.velocity.z * horizon) >= lo
        && Math.min(b.position.z, b.position.z + b.velocity.z * horizon) <= hi);
    if (!active.length) return null;
    const radius = Math.max(0, clearRadius - cageRadius - 0.015);
    const clearanceAt = (x, y) => {
      const vx = x - attackerPos.x, vy = y - attackerPos.y, vz = ring.pos.z - attackerPos.z;
      const length2 = vx * vx + vy * vy + vz * vz;
      const travel = Math.sqrt(length2) / (prediction.speed || 1);
      const fx=vx/travel,fy=vy/travel,fz=vz/travel;
      const movingTime=Math.min(horizon,travel),stationaryTime=travel-movingTime;
      let clearance=Infinity;
      for(const {position:b,velocity} of active){
        const x0=attackerPos.x-b.x,y0=attackerPos.y-b.y,z0=attackerPos.z-b.z;
        if(travel<1e-8){clearance=Math.min(clearance,Math.hypot(x0,y0,z0)-2*cageRadius);continue;}
        // Compare the moving and stopped portions of the same swept path.
        // Avoid allocating vectors, closures and a temporary array per blocker
        // at each of the 73 candidate lanes.
        const moving=closestSeparation(x0,y0,z0,fx-velocity.x,fy-velocity.y,fz-velocity.z,movingTime);
        const stopped=closestSeparation(x0+(fx-velocity.x)*movingTime,
          y0+(fy-velocity.y)*movingTime,z0+(fz-velocity.z)*movingTime,fx,fy,fz,stationaryTime);
        clearance=Math.min(clearance,moving-2*cageRadius,stopped-2*cageRadius);
      }
      return clearance;
    };
    const centerClearance = clearanceAt(ring.pos.x, ring.pos.y);
    let best = { x: 0, y: 0, score: Math.min(centerClearance, 0.12), clearance: centerClearance };
    for (const fraction of [0.5, 1]) {
      for (const sample of gapSamples) {
        const x = sample.x * radius * fraction, y = sample.y * radius * fraction;
        const clearance = clearanceAt(ring.pos.x + x, ring.pos.y + y);
        const score = Math.min(clearance, 0.12) - Math.hypot(x, y) * 0.06;
        if (score > best.score + 0.002) best = { x, y, score, clearance };
      }
    }
    // reliable means a collision-free sampled route exists; a fully blocked opening stays uncertain.
    return { angle: Math.atan2(best.y, best.x), clearance: best.clearance, reliable: best.clearance > 0.015, radius: Math.hypot(best.x, best.y) };
  }
  window.DroneAITactics = Object.freeze({ guardHome, guardTarget, launchGuard, launchAttack, guardRecoveryVelocity, attackRecoveryVelocity, attackTarget, stepAttack, stepGuard, stepFlight, flightPose, syncFlightPose, selectGear, resetGear, findGap,
    guardState: actor => {
      const s=guardDecisions.get(actor),contact=guardChallenges.get(actor);
      return s||contact?{role:contact?.phase==='drive'?'intercept':s?.role||'home',
        target:{...(contact&&contact.phase!=='cooldown'?contact.target:s?.target)},
        action:contact?.phase||'guard',challenges:challengeCounts.get(actor)||0,
        handoffs:guardAssignments.get(s?.ring||contact?.ring)?.handoffs||0}:null;
    },
    brainState: actor => { const m = attackMemories.get(actor); return m ? { attempts:m.attempts,successes:m.successes,failures:m.failures,replans:m.replans,navigation:m.navigation?{age:m.navigation.age,stalled:m.navigation.stalled}:null,escape:m.escape?{...m.escape}:null,lanes:{...m.lanes},active:m.active?{...m.active}:null } : null; },
    attackState: actor => { const s = attackStates.get(actor); return s ? { locked: s.locked, scans: s.scans, blocked: s.blocked,
      mode: s.mode, retries: s.retries, stalled: s.stalled, route: { ...s.route }, maneuver: s.maneuver
        ? {stage:s.maneuver.stage,age:s.maneuver.age,side:s.maneuver.side,flank:s.maneuver.flank} : null } : null; },
    safetyState: actor => safetyStates.get(actor) ? { ...safetyStates.get(actor) } : null });
})();
