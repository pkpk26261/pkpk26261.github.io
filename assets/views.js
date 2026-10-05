'use strict';
window.ReaderViews = (() => {
  let request, currentDisplay, currentURL;
  let state = {text:'統計中…', ready:false};
  const endpoint = 'https://cdn.busuanzi.cc/api.php';
  const publicOrigin = 'https://pkpk26261.github.io';
  function render() {
    if (!currentDisplay?.isConnected) return;
    currentDisplay.querySelector('[data-views-value]').textContent = state.text;
    currentDisplay.querySelector('[data-views-suffix]').hidden = !state.ready;
  }
  function init() {
    currentDisplay = document.querySelector('[data-article-views]');
    const canonical = document.querySelector('link[rel="canonical"]')?.href;
    if (!currentDisplay || !canonical || location.origin !== publicOrigin) {
      request?.abort(); request = null; currentURL = null;
      state = {text:'發布後開始統計', ready:false};
      render();
      return;
    }
    const url = new URL(canonical);
    if (url.origin !== publicOrigin) return;
    url.hash = ''; url.search = '';
    // Hash history may replace the page DOM; reuse this article's request/result.
    if (currentURL === url.href) { render(); return; }
    request?.abort();
    currentURL = url.href;
    const controller = new AbortController();
    request = controller;
    state = {text:'統計中…', ready:false};
    render();
    const timer = setTimeout(() => controller.abort(), 8000);
    // Same page-PV request as the provider's official script, using the canonical
    // URL so chapter hashes and search parameters share one article counter.
    fetch(endpoint, {
      method:'POST', credentials:'omit', referrerPolicy:'no-referrer',
      body:JSON.stringify({url:url.href, referrer:''}), signal:controller.signal
    }).then(async response => {
      if (!response.ok) throw new Error('Count service unavailable');
      const data = await response.json();
      const count = data.busuanzi_page_pv;
      if (!['number','string'].includes(typeof count) || !/^\d+$/.test(String(count)) || !Number.isSafeInteger(Number(count))) throw new Error('Invalid page count');
      if (request !== controller) return;
      state = {text:Number(count).toLocaleString('zh-TW'), ready:true};
      render();
    }).catch(() => {
      if (request !== controller) return;
      state = {text:'暫時無法取得', ready:false};
      render();
    }).finally(() => clearTimeout(timer));
  }
  document.addEventListener('reader:navigate', init);
  window.addEventListener('pageshow', event => { if (event.persisted) { currentURL = null; init(); } });
  init();
  return {init};
})();
