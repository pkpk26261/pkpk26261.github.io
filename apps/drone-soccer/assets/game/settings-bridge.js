/* Narrow bridge for the parent game's settings. Standalone games remain standalone. */
(() => {
  'use strict';
  if (window.parent === window) return;
  const origin = location.protocol === 'file:' ? '*' : location.origin;
  const validParent = event => event.source === window.parent && (location.protocol === 'file:' ? event.origin === 'null' : event.origin === location.origin);
  function send(type, extra = {}) { window.parent.postMessage({type,...extra}, origin); }
  window.addEventListener('message', event => {
    if (!validParent(event) || !event.data || typeof event.data !== 'object') return;
    const data = event.data;
    if (data.type === 'flight-settings-apply') {
      if (typeof data.targetfps === 'string' && data.targetfps !== window.DroneFrameTiming?.preference) window.DroneFrameTiming?.setTarget(data.targetfps);
      if (typeof data.followpitch === 'boolean' && data.followpitch !== window.DroneCameraSettings?.followPitch) window.DroneCameraSettings?.setFollowPitch(data.followpitch);
    } else if (data.type === 'flight-open-settings') {
      if (data.kind === 'controls' && window.DroneControlSettings?.pilots.length) {
        window.DroneControlSettings.show(); send('flight-settings-opened',{kind:'controls'});
      } else if (data.kind === 'pad') {
        const panel = document.querySelector('.fc-pad-overlay');
        const button = document.querySelector('[id$="btnPad"],.js-btn-pad');
        if (panel?.getClientRects().length) send('flight-settings-opened',{kind:'pad'});
        else if (button) { button.click(); send('flight-settings-opened',{kind:'pad'}); }
      }
    }
  });
  const ready = () => {
    send('flight-settings-ready');
    send('flight-setting-change',{key:'targetfps',value:window.DroneFrameTiming?.preference});
    send('flight-setting-change',{key:'followpitch',value:window.DroneCameraSettings?.followPitch});
  };
  if (document.readyState === 'loading') window.addEventListener('DOMContentLoaded',ready,{once:true}); else ready();
})();
