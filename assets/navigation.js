'use strict';
(() => {
  // Keep the audio element and AudioContext alive while reader pages change.
  let request = null;
  let displayedURL = location.href;
  const positions = new Map();
  const route = url => url.origin === location.origin && !/^\/(editor|api|content|design|rating)(\/|$)/.test(url.pathname) && !/\.[a-z0-9]+$/i.test(url.pathname);
  const remember = () => positions.set(displayedURL, [scrollX, scrollY]);
  const move = (url, back) => {
    if (url.hash) {
      let id; try { id = decodeURIComponent(url.hash.slice(1)); } catch { id = url.hash.slice(1); }
      document.getElementById(id)?.scrollIntoView();
    } else {
      const point = back ? positions.get(url.href) : null;
      scrollTo({left:point?.[0] || 0, top:point?.[1] || 0, behavior:'instant'});
    }
  };
  async function navigate(url, back = false) {
    request?.abort();
    const current = new AbortController(); request = current;
    document.documentElement.setAttribute('aria-busy', 'true');
    const announcement = document.getElementById('site-status');
    if (announcement) announcement.textContent = '正在開啟頁面…';
    try {
      const response = await fetch(url.href, {signal:current.signal});
      if (!response.ok || !response.headers.get('content-type')?.includes('text/html')) throw new Error('Not a reader page');
      const next = new DOMParser().parseFromString(await response.text(), 'text/html');
      if (!next.querySelector('main') || !next.querySelector('.music-dock') || !next.querySelector('.site-header')) throw new Error('Not a reader shell');
      if (current.signal.aborted) return;
      if (!back) { remember(); history.pushState(null, '', url.href); }
      window.ReaderPage?.dispose(); window.ReaderMotion?.dispose();
      const dock = document.querySelector('.music-dock');
      const audio = document.querySelector('audio');
      // Audio remains attached at its original body location; only controls move.
      if (dock) document.body.append(dock);
      [...document.body.children].forEach(node => {
        if (node !== dock && node !== audio && node.tagName !== 'SCRIPT') node.remove();
      });
      for (const node of [...next.body.children]) {
        if (node.matches('.music-dock, .site-loader, script')) continue;
        document.body.append(node);
      }
      document.body.className = next.body.className;
      for (const key of Object.keys(document.body.dataset)) delete document.body.dataset[key];
      Object.assign(document.body.dataset, next.body.dataset);
      displayedURL = url.href;
      document.title = next.title;
      document.head.querySelectorAll('meta[property],meta[name="description"],link[rel="canonical"],script[type="application/ld+json"]').forEach(node => node.remove());
      next.head.querySelectorAll('meta[property],meta[name="description"],link[rel="canonical"],script[type="application/ld+json"]').forEach(node => document.head.append(node));
      document.documentElement.style.removeProperty('--reading-progress');
      window.StudioCode?.render(document);
      window.ReaderPage?.init(); window.ReaderMotion?.init();
      document.dispatchEvent(new Event('reader:navigate'));
      document.getElementById('main')?.focus({preventScroll:true});
      move(url, back);
      document.getElementById('site-status').textContent = '已開啟：' + document.title;
    } catch (error) {
      if (error.name !== 'AbortError') location.assign(url.href);
    } finally {
      if (request === current) document.documentElement.removeAttribute('aria-busy');
    }
  }
  document.addEventListener('click', event => {
    const link = event.target.closest('a[href]');
    if (!link || event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || link.hasAttribute('download') || (link.target && link.target !== '_self')) return;
    const url = new URL(link.href);
    if (!route(url) || (url.pathname === location.pathname && url.search === location.search)) return;
    event.preventDefault();
    navigate(url);
  });
  if ('scrollRestoration' in history) history.scrollRestoration = 'manual';
  window.addEventListener('popstate', () => {
    remember();
    navigate(new URL(location.href), true);
  });
})();
