'use strict';
(() => {
  const loader = document.querySelector('.site-loader');
  if (!loader) return;
  try { if (sessionStorage.getItem('yc-entry-seen') === 'true') { loader.remove(); return; } sessionStorage.setItem('yc-entry-seen', 'true'); } catch { /* Each new document can still open without storage. */ }
  loader.hidden = false;
  const meter = loader.querySelector('[role="progressbar"]');
  const fill = loader.querySelector('.loader-fill');
  const percent = loader.querySelector('.loader-percent');
  const stage = loader.querySelector('.loader-stage');
  const started = Date.now();
  let finished = false;
  const progress = (value, text) => {
    if (finished) return;
    meter.setAttribute('aria-valuenow', String(value));
    fill.style.width = value + '%'; percent.textContent = value + '%'; stage.textContent = text;
  };
  const close = () => {
    if (finished) return;
    finished = true; clearTimeout(timeout); clearTimeout(skipTimer);
    loader.classList.add('is-leaving');
    setTimeout(() => loader.remove(), 340);
  };
  loader.querySelector('.loader-skip').addEventListener('click', close);
  const skipTimer = setTimeout(() => { loader.querySelector('.loader-skip').hidden = false; }, 2500);
  // Slow images/fonts cannot keep readers trapped behind the intro.
  const timeout = setTimeout(close, 6000);
  progress(12, '準備學習手札');
  const ready = async () => {
    progress(40, '整理文字與插圖');
    const images = [...document.querySelectorAll('.loader-usagi,.identity-avatar,.hero .usagi-art')];
    const tasks = images.map(img => img.complete ? Promise.resolve() : new Promise(resolve => {
      img.addEventListener('load', resolve, {once:true}); img.addEventListener('error', resolve, {once:true});
    }));
    if (document.fonts?.ready) tasks.push(document.fonts.ready);
    let completed = 0;
    await Promise.all(tasks.map(task => Promise.resolve(task).catch(() => {}).then(() => {
      completed++; progress(40 + Math.round(completed / tasks.length * 50), '準備好，一起探索');
    })));
    if (finished) return;
    progress(100, '呀哈！歡迎來到學以成之');
    setTimeout(close, Math.max(180, 800 - (Date.now() - started)));
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', ready, {once:true});
  else ready();
})();
