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
  const headerSlot = document.querySelector('.header-music-slot');
  const menuToggle = document.querySelector('.menu-toggle');
  const dockHome = document.createComment('Desktop music controls');
  dock.before(dockHome);
  // Keep audio in one document location so resizing never interrupts playback.
  if (audio) dock.after(audio);
  function placeControls() {
    if (mobileHeader.matches && headerSlot) headerSlot.append(dock);
    else dockHome.after(dock);
    collapse();
  }
  mobileHeader.addEventListener('change', placeControls);
  menuToggle?.addEventListener('click', collapse);
  placeControls();
  toggle.addEventListener('click', () => {
    if (panel.hidden) {
      if (menuToggle?.getAttribute('aria-expanded') === 'true') menuToggle.click();
      expand();
    } else collapse();
    if (audio && dock.dataset.state === 'blocked') { stoppedByUser = false; attemptPlay(); }
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
  audio.volume = Math.max(0, Math.min(1, Number.isFinite(previous.volume) ? previous.volume : Number(dock.dataset.volume)));
  volume.value = String(Math.round(audio.volume * 100));
  volume.setAttribute('aria-valuetext', volume.value + '%');
  const trackTitle = () => tracks[activeIndex]?.title || '閱讀配樂';
  const updateTrack = () => trackButtons.forEach((button, index) => button.setAttribute('aria-pressed', String(index === activeIndex)));
  updateTrack();
  const save = () => {
    try { sessionStorage.setItem('yc-audio', JSON.stringify({ source: audio.getAttribute('src'), stopped: stoppedByUser, time: audio.currentTime, volume: audio.volume })); } catch { /* Storage is optional. */ }
  };
  const attemptPlay = async () => {
    setState('loading');
    setStatus('正在準備播放…');
    try { await audio.play(); }
    catch (error) {
      if (error.name === 'AbortError') return;
      if (error.name === 'NotAllowedError') {
        setState('blocked');
        setStatus('點擊播放，開啟閱讀配樂。');
      } else {
        setState('error');
        setStatus('音訊目前無法播放，請稍後重試。');
        expand();
      }
    }
  };
  function selectTrack(index) {
    stoppedByUser = false;
    if (index !== activeIndex) {
      activeIndex = index;
      audio.src = tracks[index].audio_src;
      previous = {};
      updateTrack();
    }
    attemptPlay();
    save();
  }
  trackButtons.forEach(button => button.addEventListener('click', () => selectTrack(Number(button.dataset.track))));
  play.addEventListener('click', () => {
    if (!audio.paused) { stoppedByUser = true; audio.pause(); }
    else { stoppedByUser = false; attemptPlay(); }
    save();
  });
  volume.addEventListener('input', () => { audio.volume = Number(volume.value) / 100; volume.setAttribute('aria-valuetext', volume.value + '%'); save(); });
  audio.addEventListener('playing', () => {
    setState('playing');
    play.setAttribute('aria-label', '暫停背景音樂');
    setStatus('正在播放 · ' + trackTitle());
    save();
  });
  audio.addEventListener('pause', () => {
    setState('paused');
    play.setAttribute('aria-label', '播放背景音樂');
    setStatus('已暫停 · ' + trackTitle());
    save();
  });
  audio.addEventListener('ended', () => { if (!stoppedByUser && tracks.length > 1) selectTrack(randomTrackIndex(activeIndex)); });
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
  window.addEventListener('pagehide', save);
  if (dock.dataset.autoplay === 'true' && !stoppedByUser) attemptPlay();
  else { setState('paused'); setStatus('點擊播放，開啟閱讀配樂。'); }
})();
