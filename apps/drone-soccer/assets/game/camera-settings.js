/* Camera preferences, mode controls and adaptive center framing for all games. */
(() => {
  'use strict';
  const storageKey = 'droneSoccer.camera.followPitch';
  let followPitch = false;
  try { followPitch = localStorage.getItem(storageKey) === 'true'; } catch (_) { /* Session-only when storage is unavailable. */ }
  const requestedPitch = new URLSearchParams(location.search).get('followpitch');
  if (requestedPitch === '1' || requestedPitch === '0') followPitch = requestedPitch === '1';

  // Keep the observer upright and centered. Only the vertical aim follows the
  // flight envelope; include both complete goal rings so looking down at a
  // grounded drone cannot crop the goal above it. Widen only when necessary,
  // retreating on very narrow panels instead of using an extreme wide angle.
  function frameCenter(camera, options) {
    const { endSign, netZ, back, eyeHeight, goals, actor, actorRadius,
      goalTubeRadius, preferredFov, launchWidth, peers } = options;
    const points = [];
    const box = (x, y, z, rx, ry, rz) => {
      for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1])
        points.push([x + sx * rx, y + sy * ry, z + sz * rz]);
    };
    for (const goal of goals) {
      if (goal.halfSize) {
        const h = goal.halfSize;
        box(goal.pos.x, goal.pos.y, goal.pos.z, h.x, h.y, h.z);
      } else {
        const radius = goal.size / 2 + goalTubeRadius;
        box(goal.pos.x, goal.pos.y, goal.pos.z, radius, radius, goalTubeRadius);
      }
    }
    // All starting slots share the same framing. During flight, follow height
    // without yawing toward an individual slot or depending on the pilot role.
    const p = actor.position;
    const halfWidth = Math.max(launchWidth / 2, Math.abs(p.x) + actorRadius);
    box(0, p.y, p.z, halfWidth, actorRadius, actorRadius);
    const aspect = Math.max(0.05, Math.min(camera.aspect, ...(peers || [])
      .filter(r => r.team === actor.team && r.camera).map(r => r.camera.aspect)));
    // Derive vertical aim at the chosen standing position. Any automatic
    // framing retreat must not reverse that aim while the drone is ascending.
    const standingDistance = Math.max(back, 0.06);
    let low = Infinity, high = -Infinity;
    for (const [, y, z] of points) {
      const angle = Math.atan2(y - eyeHeight, netZ + standingDistance - endSign * z);
      low = Math.min(low, angle); high = Math.max(high, angle);
    }
    const pitchLimit = Math.PI / 9;
    const pitch = Math.max(-pitchLimit, Math.min(pitchLimit, (low + high) / 2));
    const sin = Math.sin(pitch), cos = Math.cos(pitch);
    const evaluate = distance => {
      let tangent = 0;
      for (const [x, y, z] of points) {
        const depth = netZ + distance - endSign * z, dy = y - eyeHeight;
        const forward = depth * cos + dy * sin;
        if (forward <= camera.near) return { pitch, fov: 180 };
        tangent = Math.max(tangent, Math.abs(dy * cos - depth * sin) / (forward * 0.82),
          Math.abs(x) / (forward * aspect * 0.86));
      }
      return { pitch, fov: Math.atan(tangent) * 360 / Math.PI };
    };
    // Use a normal 60-degree lens in flight. A small ground-only allowance
    // keeps the launch pad visible above the foreground arena bumper without
    // stretching airborne or far-goal views into a 100+ degree wide angle.
    const baseFov = Math.min(60, Math.max(30, preferredFov));
    let distance = standingDistance, frame = evaluate(distance);
    const groundAllowance = 14 * Math.max(0, Math.min(1, (1.2 - p.y) / 1.1));
    const maxFov = 60 + groundAllowance;
    if (frame.fov > maxFov) {
      let low = distance, high = Math.max(1, distance * 2);
      while (evaluate(high).fov > maxFov && high < 64) high *= 2;
      for (let i = 0; i < 18; i++) {
        const middle = (low + high) / 2;
        if (evaluate(middle).fov > maxFov) low = middle; else high = middle;
      }
      distance = high; frame = evaluate(distance);
    }
    const fov = Math.max(baseFov, frame.fov);
    if (Math.abs(camera.fov - fov) > 1e-7) { camera.fov = fov; camera.updateProjectionMatrix(); }
    // LT/RT slide the standing observer along their own left/right axis.
    const standingX = endSign * (actor.controllerViewShift || 0);
    camera.position.set(standingX, eyeHeight, endSign * (netZ + distance));
    camera.lookAt(standingX, eyeHeight + Math.sin(frame.pitch),
      camera.position.z - endSign * Math.cos(frame.pitch));
  }

  window.DroneCameraSettings = Object.freeze({
    get followPitch() { return followPitch; },
    pitchFor(pitch) { return followPitch ? pitch : 0; },
    frameCenter,
    setFollowPitch(enabled) {
      followPitch = Boolean(enabled);
      try { localStorage.setItem(storageKey, String(followPitch)); } catch (_) { /* Keep the live preference working. */ }
      window.dispatchEvent(new Event('drone-camera-settings-change'));
      if (window.parent !== window) window.parent.postMessage({ type:'flight-setting-change', key:'followpitch', value:followPitch }, location.protocol === 'file:' ? '*' : location.origin);
    }
  });
  window.addEventListener('storage', event => {
    if (event.key !== storageKey && event.key !== null) return;
    followPitch = event.key === storageKey && event.newValue === 'true';
    window.dispatchEvent(new Event('drone-camera-settings-change'));
  });

  // The engine owns camera motion; the shared UI selects modes through this API.
  const modes = Object.freeze([
    { id: 'cockpit', label: '第一人稱', hint: '從機身前方觀看', icon: 'cockpit' },
    { id: 'chase', label: '第三人稱', hint: '在機身後方跟隨', icon: 'chase' },
    { id: 'free', label: '自由視角', hint: '拖曳旋轉・滾輪縮放', icon: 'free' },
    { id: 'center', label: '固定中間', hint: '依飛行高度調整上下視線', icon: 'center' },
    { id: 'left', label: '固定左邊', hint: '從己方左邊觀察', icon: 'left' },
    { id: 'right', label: '固定右邊', hint: '從己方右邊觀察', icon: 'right' }
  ].map(Object.freeze));
  const controllers = new Map();
  const current = new Map();
  const rigs = new WeakSet();
  const cycle = ['chase', 'center', 'left', 'right', 'cockpit', 'free'];
  // Every new official match starts centered; pre-flight choices are match-local.
  let preflightMode = /\/drone_soccer_competition_(?:add_ai|addadd)_realgame\.html$/.test(location.pathname) ? 'center' : null;
  window.DroneCameraControls = Object.freeze({
    modes,
    get preflightMode() { return preflightMode; },
    setPreflightMode(mode) {
      if (!modes.some(option => option.id === mode)) return;
      preflightMode = mode;
      controllers.forEach(controller => controller.setMode(mode));
    },
    get: id => controllers.get(id),
    shiftStanding(direction,delta) { controllers.forEach(controller=>controller.shiftView?.(direction,delta)); },
    register(id, controller) {
      let spectatorShift=0;
      const actor=()=>controller.actor || window.DroneControlSettings?.pilots.find(p=>{
        const a=p.actor;return (a.id?`${a.id}-btnCam`:a.prefix?`${a.prefix}btnCam`:'btnCam')===id;
      })?.actor;
      controller.getShift=()=>actor()?.controllerViewShift || spectatorShift;
      controller.shiftView=(direction,delta=1/60)=>{
        controller.setMode('center');
        const shift=Math.max(-1.25,Math.min(1.25,controller.getShift()+direction*Math.max(0,Math.min(.1,delta))*.9));
        const a=actor();if(a)a.controllerViewShift=shift;else spectatorShift=shift;
      };
      controllers.set(id, controller);
      window.dispatchEvent(new CustomEvent('drone-camera-ready', { detail: { id } }));
      this.changed(id);
    },
    changed(id) {
      const mode = controllers.get(id)?.getMode();
      if (current.get(id) === mode) return;
      current.set(id, mode);
      window.dispatchEvent(new CustomEvent('drone-camera-change', { detail: { id, mode } }));
    },
    bindRig(rig, id, defaultViews, onChange) {
      if (rigs.has(rig)) return;
      rigs.add(rig);
      const previous = preflightMode || (rig.camViews || defaultViews)[rig.camIdx || 0];
      rig.camViews = cycle.slice();
      rig.camIdx = Math.max(0, rig.camViews.indexOf(previous));
      rig.orbitControls.enabled = previous === 'free';
      if (previous === 'free') rig.orbitControls.target.copy(rig.position);
      this.register(id, {
        actor:rig,
        getMode: () => rig.camViews[rig.camIdx],
        setMode: mode => {
          const index = rig.camViews.indexOf(mode);
          if (index < 0 || index === rig.camIdx) return;
          rig.camIdx = index;
          rig.orbitControls.enabled = mode === 'free';
          if (mode === 'free') rig.orbitControls.target.copy(rig.position);
          onChange();
          this.changed(id);
        }
      });
    }
  });
})();
