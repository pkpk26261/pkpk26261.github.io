'use strict';
(() => {
  const dock = document.querySelector('.music-dock');
  if (!dock) return;
  const toggle = dock.querySelector('.music-toggle');
  const panel = dock.querySelector('.music-panel');
  const play = dock.querySelector('.music-play');
  const status = dock.querySelector('.music-status');
  const audio = dock.querySelector('audio');
  const volume = dock.querySelector('input[type="range"]');
  const updateToggle = () => {
    const action = panel.hidden ? '開啟配樂控制' : '關閉配樂控制';
    const state = {playing:'配樂播放中',paused:'配樂已暫停',blocked:'點擊開啟配樂',loading:'配樂準備中',error:'配樂載入失敗',unavailable:'尚未設定配樂'}[dock.dataset.state];
    toggle.setAttribute('aria-label', action);
    toggle.title = state ? `${state} · ${action}` : action;
  };
  const setState = state => { dock.dataset.state = state; updateToggle(); };
  const setStatus = text => { status.textContent = text; };
  const expand = () => { panel.hidden = false; toggle.setAttribute('aria-expanded', 'true'); updateToggle(); };
  const collapse = () => { panel.hidden = true; toggle.setAttribute('aria-expanded', 'false'); updateToggle(); };
  const mobileHeader = matchMedia('(max-width: 760px)');
  let headerSlot = document.querySelector('.header-music-slot');
  let menuToggle = document.querySelector('.menu-toggle');
  const dockHome = document.createComment('Desktop music controls');
  dock.before(dockHome);
  // Keep audio in one document location so resizing never interrupts playback.
  if (audio) dock.after(audio);
  function placeControls() {
    if (mobileHeader.matches && headerSlot) headerSlot.append(dock);
    else { const main = document.querySelector('main'); main ? main.after(dock) : dockHome.after(dock); }
    collapse();
  }
  mobileHeader.addEventListener('change', placeControls);
  menuToggle?.addEventListener('click', collapse);
  placeControls();
  document.addEventListener('reader:navigate', () => {
    headerSlot = document.querySelector('.header-music-slot');
    menuToggle = document.querySelector('.menu-toggle');
    menuToggle?.addEventListener('click', collapse);
    placeControls();
  });
  toggle.addEventListener('click', () => {
    if (panel.hidden) {
      if (menuToggle?.getAttribute('aria-expanded') === 'true') menuToggle.click();
      expand();
    } else collapse();
    if (audio && dock.dataset.state === 'blocked') { stoppedByUser = false; attemptPlay(true); }
  });
  document.addEventListener('pointerdown', event => { if (!dock.contains(event.target)) collapse(); });
  dock.addEventListener('keydown', event => { if (event.key === 'Escape' && !panel.hidden) { collapse(); toggle.focus(); } });
  updateToggle();
  if (!audio) return;
  const tracks = JSON.parse(dock.dataset.tracks || '[]');
  const trackButtons = [...dock.querySelectorAll('[data-track]')];
  const randomTrackIndex = (exclude = -1) => {
    const choices = tracks.map((track, index) => index).filter(index => index !== exclude);
    return choices.length ? choices[Math.floor(Math.random() * choices.length)] : Math.max(0, exclude);
  };
  let activeIndex = 0, stoppedByUser = false;
  let previous = {};
  try { previous = JSON.parse(sessionStorage.getItem('yc-audio') || '{}'); } catch { /* Storage is optional. */ }
  stoppedByUser = previous.stopped === true;
  const savedIndex = tracks.findIndex(t => t.audio_src === previous.source);
  if (tracks.length) {
    activeIndex = savedIndex >= 0 ? savedIndex : randomTrackIndex();
    audio.src = tracks[activeIndex].audio_src;
  }
  let requestedVolume = Math.max(0, Math.min(1, Number.isFinite(previous.volume) ? previous.volume : Number(dock.dataset.volume) || 0));
  let audioContext = null, gain = null;
  const volumeOutput = dock.querySelector('.music-volume output');
  // A gain node controls actual output on iOS, even when .volume appears writable.
  function prepareAudio() {
    if (gain || audioContext) return;
    const Context = window.AudioContext || window.webkitAudioContext;
    if (!Context) return;
    try {
      audioContext = new Context();
      const source = audioContext.createMediaElementSource(audio);
      gain = audioContext.createGain();
      source.connect(gain); gain.connect(audioContext.destination);
      audio.volume = 1;
      audioContext.onstatechange = () => {
        if (audioContext.state === 'running' && !audio.paused) showPlaying();
        else if (!stoppedByUser && !audio.paused) {
          setState('blocked'); setStatus('輕觸頁面，繼續閱讀配樂。');
        }
      };
    } catch { audioContext = null; gain = null; }
  }
  function applyVolume() {
    if (gain) gain.gain.setValueAtTime(requestedVolume, audioContext.currentTime);
    else audio.volume = requestedVolume;
    volume.value = String(Math.round(requestedVolume * 100));
    volume.setAttribute('aria-valuetext', volume.value + '%');
    if (volumeOutput) volumeOutput.textContent = volume.value + '%';
  }
  applyVolume();
  const trackTitle = () => tracks[activeIndex]?.title || '閱讀配樂';
  const updateTrack = () => trackButtons.forEach((button, index) => button.setAttribute('aria-pressed', String(index === activeIndex)));
  updateTrack();
  const save = () => {
    try { sessionStorage.setItem('yc-audio', JSON.stringify({ source: audio.getAttribute('src'), stopped: stoppedByUser, time: audio.currentTime, volume: requestedVolume })); } catch { /* Storage is optional. */ }
  };
  let playSequence = 0, pendingPlay = false;
  const wantsPlayback = () => dock.dataset.autoplay === 'true' && !stoppedByUser;
  const needsPlayback = () => audio.paused || (audioContext && audioContext.state !== 'running');
  const attemptPlay = async (gesture = false) => {
    const sequence = ++playSequence;
    pendingPlay = true;
    setState('loading');
    setStatus('正在準備播放…');
    try {
      // Do not route permitted native autoplay through a still-locked AudioContext.
      // Attach the volume graph during a real interaction, when it can be unlocked.
      if (gesture) prepareAudio();
      applyVolume();
      const resumed = audioContext?.resume();
      const playing = audio.play();
      // resume() can stay pending while policy blocks Web Audio. Media playback
      // and graph state are checked separately, so the UI cannot hang forever.
      resumed?.catch(() => {
        if (sequence === playSequence && !stoppedByUser) {
          setState('blocked'); setStatus('輕觸頁面，繼續閱讀配樂。');
        }
      });
      await playing;
      if (sequence !== playSequence) return;
      if (stoppedByUser) { audio.pause(); return; }
      if (!audioContext || audioContext.state === 'running') showPlaying();
      if (audioContext && audioContext.state !== 'running') {
        setState('blocked'); setStatus('輕觸頁面，開啟閱讀配樂。');
      }
    }
    catch (error) {
      if (sequence !== playSequence) return;
      if (error.name === 'AbortError') return;
      if (error.name === 'NotAllowedError') {
        setState('blocked');
        setStatus('輕觸頁面，開啟閱讀配樂。');
      } else {
        setState('error');
        setStatus('音訊目前無法播放，請稍後重試。');
        expand();
      }
    } finally {
      if (sequence === playSequence) pendingPlay = false;
    }
  };
  function selectTrack(index, gesture = true) {
    stoppedByUser = false;
    if (index !== activeIndex) {
      activeIndex = index;
      audio.src = tracks[index].audio_src;
      previous = {};
      updateTrack();
    }
    attemptPlay(gesture);
    save();
  }
  trackButtons.forEach(button => button.addEventListener('click', () => selectTrack(Number(button.dataset.track))));
  play.addEventListener('click', () => {
    if (!audio.paused && dock.dataset.state === 'playing') { stoppedByUser = true; ++playSequence; pendingPlay = false; audio.pause(); }
    else { stoppedByUser = false; attemptPlay(true); }
    save();
  });
  volume.addEventListener('input', () => {
    requestedVolume = Number(volume.value) / 100; prepareAudio(); applyVolume();
    if (audioContext?.state === 'suspended') audioContext.resume().catch(() => {});
    save();
  });
  // Touch release, pointer release and keyboard input all retain their actual
  // browser gesture. This also repairs incidental pause/suspend states.
  const resumeOnInteraction = event => {
    if (!wantsPlayback() || event.target.closest('.music-dock')) return;
    if (needsPlayback()) attemptPlay(true);
  };
  document.addEventListener('pointerup', resumeOnInteraction);
  document.addEventListener('touchend', resumeOnInteraction, {passive:true});
  document.addEventListener('click', resumeOnInteraction);
  document.addEventListener('keydown', event => {
    if (event.key === 'Enter' || event.key === ' ') resumeOnInteraction(event);
  });
  const resumeAutomatically = () => {
    if (wantsPlayback() && needsPlayback() && !pendingPlay && !document.hidden && dock.dataset.state !== 'error') attemptPlay();
  };
  audio.addEventListener('canplay', resumeAutomatically);
  document.addEventListener('visibilitychange', resumeAutomatically);
  document.addEventListener('reader:navigate', resumeAutomatically);
  function showPlaying() {
    setState('playing'); play.setAttribute('aria-label', '暫停背景音樂');
    setStatus('正在播放 · ' + trackTitle()); save();
  }
  audio.addEventListener('playing', () => {
    if (audioContext && audioContext.state !== 'running') {
      setState('blocked'); setStatus('輕觸頁面，開啟閱讀配樂。'); return;
    }
    setState('playing');
    play.setAttribute('aria-label', '暫停背景音樂');
    setStatus('正在播放 · ' + trackTitle());
    save();
  });
  audio.addEventListener('pause', () => {
    setState(stoppedByUser ? 'paused' : 'blocked');
    play.setAttribute('aria-label', '播放背景音樂');
    setStatus(stoppedByUser ? '已暫停 · ' + trackTitle() : '輕觸頁面，繼續閱讀配樂。');
    save();
  });
  audio.addEventListener('ended', () => { if (!stoppedByUser && tracks.length > 1) selectTrack(randomTrackIndex(activeIndex), false); });
  audio.addEventListener('error', () => {
    setState('error');
    setStatus('音訊載入失敗，請確認檔案或網址。');
    expand();
  });
  const restorePosition = () => {
    if (previous.source === audio.getAttribute('src') && Number.isFinite(previous.time) && previous.time > 0 && Number.isFinite(audio.duration) && audio.duration > 0) {
      audio.currentTime = previous.time % audio.duration;
    }
  };
  if (audio.readyState >= 1) restorePosition();
  else audio.addEventListener('loadedmetadata', restorePosition, { once: true });
  audio.addEventListener('timeupdate', save);
  window.addEventListener('pagehide', save);
  window.addEventListener('pageshow', resumeAutomatically);
  if (dock.dataset.autoplay === 'true' && !stoppedByUser) attemptPlay();
  else { setState('paused'); setStatus('點擊播放，開啟閱讀配樂。'); }
})();
