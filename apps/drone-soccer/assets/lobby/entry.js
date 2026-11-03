(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const defaults = { sound:true, motion:!reduced.matches };
  let preferences = { ...defaults };
  try { preferences = { ...defaults, ...JSON.parse(localStorage.getItem('flight-club-menu-v2') || '{}') }; } catch (_) {}
  let audio;
  let lastSound = 0;
  function sound(confirm = false) {
    if (!preferences.sound || performance.now() - lastSound < 65) return;
    lastSound = performance.now();
    try {
      audio ||= new (window.AudioContext || window.webkitAudioContext)();
      if (audio.state === 'suspended') void audio.resume();
      const oscillator = audio.createOscillator(), gain = audio.createGain();
      const now = audio.currentTime;
      oscillator.type = 'sine';
      oscillator.frequency.setValueAtTime(confirm ? 640 : 440, now);
      oscillator.frequency.exponentialRampToValueAtTime(confirm ? 960 : 350, now + .055);
      gain.gain.setValueAtTime(0, now);
      gain.gain.linearRampToValueAtTime(confirm ? .035 : .018, now + .004);
      gain.gain.exponentialRampToValueAtTime(.0001, now + .075);
      oscillator.connect(gain); gain.connect(audio.destination);
      oscillator.start(now); oscillator.stop(now + .08);
    } catch (_) { /* Optional menu feedback never gates play. */ }
  }
  function applyPreferences() {
    $('menu-sound-setting').checked = !!preferences.sound;
    $('menu-motion-setting').checked = !!preferences.motion;
    $('home-sound').setAttribute('aria-pressed', String(!!preferences.sound));
    $('home-sound').querySelector('span').textContent = preferences.sound ? '選單音效' : '選單已靜音';
    document.documentElement.dataset.menuMotion = preferences.motion ? 'on' : 'off';
    let saved = true;
    try { localStorage.setItem('flight-club-menu-v2', JSON.stringify(preferences)); } catch (_) { saved = false; }
    window.dispatchEvent(new CustomEvent('flight-menu-preferences', {detail:{...preferences,saved}}));
    return saved;
  }
  window.FlightMenuPreferences = Object.freeze({
    get current(){ return {...preferences}; },
    set(key,value){ if(!['sound','motion'].includes(key))return false; preferences[key]=!!value; return applyPreferences(); },
    reset(){ preferences={...defaults}; return applyPreferences(); }
  });
  const copy = {
    battle:['自由對戰','選好編隊，讓對決開始。'],
    practice:['飛行訓練','先找到手感，再創造你的精彩。'],
    racing:['競速挑戰','選擇航線，挑戰下一個最佳成績。'],
    official:['正式賽','三局兩勝，為每一場勝利而飛。']
  };
  function refreshSelection(detail = window.FlightLobby.snapshot()) {
    const [title,description] = copy[detail.category] || copy.battle;
    $('setup-screen-title').textContent = title;
    $('setup-screen-description').textContent = description;
    document.querySelectorAll('[data-home-category]').forEach(button => button.dataset.current = String(button.dataset.homeCategory === detail.category));
    window.flightShowroom?.setTheme(detail.category);
    updateDevice();
  }
  function updateDevice() {
    let pads = [];
    try { pads = window.DroneUSBInput.controllerPads(window.DroneUSBInput.browserPads()); } catch (_) {}
    const connected = pads.length > 0;
    $('home-input-icon').setAttribute('href', '#i-controller');
    $('home-input-label').textContent = connected ? '搖桿已連線' : '請接上搖桿，按任一按鈕啟用';
  }
  function configure(category) {
    sound(true);
    window.FlightLobby.configure(category);
    $('setup-screen-title').focus({ preventScroll:true });
    window.scrollTo({ top:0, behavior:'instant' });
  }
  $('home-play').addEventListener('click', () => configure('battle'));
  $('home-practice').addEventListener('click', () => configure('practice'));
  document.querySelectorAll('[data-home-category]').forEach(button => {
    button.addEventListener('click', () => configure(button.dataset.homeCategory));
    button.addEventListener('pointerenter', event => {
      if (event.pointerType !== 'touch') window.flightShowroom?.setTheme(button.dataset.homeCategory);
    });
  });
  $('home-return').addEventListener('click', () => {
    sound(); window.FlightLobby.home(); window.scrollTo({top:0,behavior:'instant'}); $('home-play').focus({preventScroll:true});
  });
  $('home-settings').addEventListener('click', () => { sound(); window.FlightSettings?.open(); });
  $('home-help').addEventListener('click', () => { sound(); $('help-dialog').showModal(); });
  $('home-sound').addEventListener('click', () => { preferences.sound = !preferences.sound; applyPreferences(); if (preferences.sound) sound(); });
  $('menu-sound-setting').addEventListener('change', event => { preferences.sound = event.target.checked; applyPreferences(); sound(); });
  $('menu-motion-setting').addEventListener('change', event => { preferences.motion = event.target.checked; applyPreferences(); });
  window.addEventListener('flight-lobby-change', event => refreshSelection(event.detail));
  window.addEventListener('flight-lobby-view', event => {
    const screen = $(event.detail.screen === 'home' ? 'home-screen' : 'lobby');
    if (event.detail.screen !== 'game') { screen.classList.remove('screen-arrival'); requestAnimationFrame(() => screen.classList.add('screen-arrival')); }
  });
  window.addEventListener('gamepadconnected', () => updateDevice());
  window.addEventListener('gamepaddisconnected', () => updateDevice());
  window.addEventListener('pagehide', () => { window.flightShowroom?.setActive(false); if (audio?.state === 'running') void audio.suspend(); });
  window.addEventListener('pageshow', () => window.flightShowroom?.setActive(document.documentElement.dataset.screen === 'home'));
  applyPreferences(); refreshSelection();
})();
