/* Display-paced rendering and time-based camera smoothing. No frame limiter. */
(() => {
  'use strict';
  const storageKey = 'droneSoccer.render.targetFPS';
  const valid = value => value === 'auto' || /^(?:[3-9]\d|[1-9]\d{2}|1000)$/.test(String(value));
  // The current display is configured at 120 Hz. This is a quality target,
  // never a cap: requestAnimationFrame still renders every display callback.
  let preference = '120';
  try {
    const saved = localStorage.getItem(storageKey);
    if (valid(saved)) preference = saved;
  } catch (_) { /* Preferences also work without storage. */ }
  const query = new URLSearchParams(location.search).get('targetfps');
  if (valid(query)) preference = query;

  function setTarget(value) {
    value = String(value);
    if (!valid(value)) return false;
    preference = value;
    try { localStorage.setItem(storageKey, value); } catch (_) { /* Session only. */ }
    window.dispatchEvent(new CustomEvent('drone-frame-settings-change'));
    if (window.parent !== window) window.parent.postMessage({ type:'flight-setting-change', key:'targetfps', value }, location.protocol === 'file:' ? '*' : location.origin);
    return true;
  }

  window.addEventListener('storage', event => {
    if (event.key !== storageKey && event.key !== null) return;
    const next = valid(event.newValue) ? String(event.newValue) : '120';
    if (next === preference) return;
    preference = next;
    window.dispatchEvent(new CustomEvent('drone-frame-settings-change'));
  });

  function create() {
    let time = 0, last = null, estimated = 60, estimateAt = 0;
    const intervals = [];
    function reset() { last = null; intervals.length = 0; estimated = 60; estimateAt = 0; }
    document.addEventListener('visibilitychange', reset);
    // Re-estimate after moving the window to another display.
    window.addEventListener('resize', reset);
    return {
      frame(timestamp, hidden = document.hidden) {
        time = Number.isFinite(timestamp) ? timestamp : performance.now();
        const dt = last === null ? 0 : time - last;
        last = time;
        if (hidden || dt > 250 || dt < 0) { reset(); return; }
        if (dt <= 0) return;
        intervals.push(dt);
        if (intervals.length > 120) intervals.shift();
        if (intervals.length < 12 || time - estimateAt < 500) return;
        estimateAt = time;
        const sorted = intervals.slice().sort((a, b) => a - b);
        // Reject occasional very short callbacks. This estimates observed
        // cadence, not a hardware refresh-rate query; explicit targets are
        // useful when rendering is too busy to reach the display's rate.
        const observed = Math.round(1000 / sorted[Math.floor(sorted.length * 0.2)]);
        if (observed > estimated * 1.05) estimated = observed;
      },
      get time() { return time; },
      get targetFPS() { return preference === 'auto' ? estimated : Number(preference); }
    };
  }

  // Match the old 60 FPS camera feel at every refresh rate, including 120 Hz.
  function alpha(valueAt60, delta) {
    const dt = Number.isFinite(delta) ? Math.max(0, Math.min(delta, 0.1)) : 1 / 60;
    return 1 - Math.pow(1 - valueAt60, dt * 60);
  }
  window.DroneFrameTiming = Object.freeze({
    create, alpha, setTarget,
    get preference() { return preference; }
  });
})();
