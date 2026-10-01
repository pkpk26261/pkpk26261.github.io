'use strict';
window.ReaderPage = (() => {
  let dispose;
  function init() {
  dispose?.();
  const scope = new AbortController();
  const observers = [];
  const listen = (target, type, callback, options = {}) => target.addEventListener(type, callback, { ...options, signal: scope.signal });
  const intersection = (...args) => { const item = new IntersectionObserver(...args); observers.push(item); return item; };
  const resizeObserver = (...args) => { const item = new ResizeObserver(...args); observers.push(item); return item; };
  dispose = () => { scope.abort(); observers.forEach(item => item.disconnect()); };
  const menu = document.querySelector('.menu-toggle');
  const mobileNav = document.getElementById('mobile-nav');
  function setMenu(open) {
    if (!menu || !mobileNav) return;
    menu.setAttribute('aria-expanded', String(open));
    menu.setAttribute('aria-label', open ? '關閉導覽選單' : '開啟導覽選單');
    mobileNav.hidden = !open;
    document.body.classList.toggle('nav-open', open);
  }
  function closeMenu() { setMenu(false); }
  menu?.addEventListener('click', () => setMenu(menu.getAttribute('aria-expanded') !== 'true'));
  listen(document, 'pointerdown', e => { if (!e.target.closest('.site-header')) closeMenu(); });
  listen(document, 'focusin', e => { if (!e.target.closest('.site-header')) closeMenu(); });
  listen(document, 'keydown', e => { if (e.key === 'Escape' && menu?.getAttribute('aria-expanded') === 'true') { closeMenu(); menu.focus(); } });
  mobileNav?.addEventListener('click', e => { if (e.target.closest('a')) closeMenu(); });
  listen(window.matchMedia('(min-width: 761px)'), 'change', e => { if (e.matches) closeMenu(); });

  const articleToc = document.querySelector('.article-toc');
  if (articleToc) {
    const inlineToc = matchMedia('(max-width: 850px)');
    const placeToc = () => { articleToc.open = !inlineToc.matches; };
    placeToc();
    listen(inlineToc, 'change', placeToc);
    articleToc.querySelectorAll('a').forEach(link => link.addEventListener('click', () => {
      if (inlineToc.matches) articleToc.open = false;
    }));
  }

  const normalized = value => value.normalize('NFKC').toLocaleLowerCase();
  document.querySelectorAll('[data-list]').forEach(list => {
    const cards = [...list.querySelectorAll('.article-card')];
    const input = list.querySelector('input[type="search"]');
    const sort = list.querySelector('[data-sort]');
    const grid = list.querySelector('.article-grid');
    const status = list.querySelector('.filter-status');
    const limit = Number(list.dataset.limit) || Infinity;
    let topic = 'all';
    const params = new URLSearchParams(location.search);
    if (sort && params.get('sort') === 'updated') sort.value = 'updated';
    if (input && params.get('q')) input.value = params.get('q');
    if ([...list.querySelectorAll('[data-filter]')].some(b => b.dataset.filter === params.get('topic'))) topic = params.get('topic');
    function filter() {
      const query = normalized(input?.value.trim() || '');
      let matches = 0, shown = 0;
      if (sort) {
        cards.sort((a,b) => ReaderSearch.compareDates(a.dataset, b.dataset, sort.value));
        cards.forEach(card => grid.append(card));
      }
      cards.forEach(card => {
        const found = (topic === 'all' || card.dataset.topics.split(' ').includes(topic)) && (!query || query.split(/\s+/).every(word => normalized(card.dataset.searchText).includes(word)));
        if (found) matches++;
        card.hidden = !(found && shown < limit);
        if (!card.hidden) shown++;
      });
      list.querySelectorAll('[data-filter]').forEach(button => { const active = button.dataset.filter === topic; button.classList.toggle('active', active); button.setAttribute('aria-pressed', String(active)); });
      if (status) status.textContent = matches === 0 ? '沒有符合的文章' : limit === Infinity ? `顯示 ${matches} 篇文章` : `此主題共 ${matches} 篇，顯示 ${shown} 篇文章`;
      list.querySelector('.empty-state').hidden = matches > 0;
      if (input || sort) {
        const url = new URL(location.href);
        topic === 'all' ? url.searchParams.delete('topic') : url.searchParams.set('topic', topic);
        if (input) input.value.trim() ? url.searchParams.set('q', input.value.trim()) : url.searchParams.delete('q');
        if (sort) sort.value === 'updated' ? url.searchParams.set('sort', 'updated') : url.searchParams.delete('sort');
        history.replaceState(null, '', url);
      }
    }
    list.querySelectorAll('[data-filter]').forEach(button => button.addEventListener('click', () => { topic = button.dataset.filter; filter(); }));
    input?.addEventListener('input', filter);
    sort?.addEventListener('change', filter);
    filter();
  });

  const totalArticles = document.body.dataset.articleCount;
  const search = document.querySelector('.search-dialog');
  const searchInput = search?.querySelector('input');
  const searchStatus = document.getElementById('search-status');
  const searchResults = search?.querySelector('.search-results');
  // The keyboard shrinks iOS's visual viewport without shrinking the layout.
  // Keep the mobile dialog inside the part of the page the reader can see.
  function placeSearch() {
    if (!search?.open) return;
    const viewport = window.visualViewport;
    const values = {width:viewport?.width || window.innerWidth, height:viewport?.height || window.innerHeight, left:viewport?.offsetLeft || 0, top:viewport?.offsetTop || 0};
    for (const [key,value] of Object.entries(values)) search.style.setProperty(`--search-viewport-${key}`, `${value}px`);
  }
  if (window.visualViewport) {
    listen(window.visualViewport, 'resize', placeSearch);
    listen(window.visualViewport, 'scroll', placeSearch);
  }
  listen(window, 'resize', placeSearch);
  let searchData = null, loading = null, searchTrigger = null;
  async function loadSearch() {
    if (searchData) return searchData;
    if (!loading) loading = fetch('/assets/search-index.json').then(response => { if (!response.ok) throw new Error('Search index unavailable'); return response.json(); }).then(data => searchData = data).catch(error => { loading = null; throw error; });
    return loading;
  }
  function resultNode(post, terms) {
    const link = document.createElement('a'); link.className = 'search-result'; link.href = post.url;
    const meta = document.createElement('small'); meta.textContent = `${post.label} · ${post.date}`;
    const title = document.createElement('h3'); ReaderSearch.appendHighlighted(title, post.title, terms);
    const excerpt = document.createElement('p');
    ReaderSearch.appendHighlighted(excerpt, ReaderSearch.snippet(post.text, terms, post.summary), terms);
    link.append(meta, title, excerpt); return link;
  }
  async function runSearch() {
    const query = searchInput.value.trim();
    searchResults.replaceChildren();
    if (!query) { searchStatus.textContent = `輸入關鍵字，搜尋所有 ${totalArticles} 篇文章。`; return; }
    searchStatus.textContent = '正在搜尋…';
    try {
      const data = await loadSearch();
      if (query !== searchInput.value.trim()) return;
      const terms = normalized(query).split(/\s+/);
      const results = data.map(post => {
        const title = normalized(post.title), body = normalized(`${post.text} ${post.label}`);
        return { post, found: terms.every(term => title.includes(term) || body.includes(term)), score: terms.reduce((s,term) => s + (title.includes(term) ? 2 : 0), 0) };
      }).filter(item => item.found).sort((a,b) => b.score - a.score);
      searchStatus.textContent = results.length ? `找到 ${results.length} 篇與「${query}」相關的文章` : `找不到與「${query}」相關的文章，試試 Python、YOLO 或 Ubuntu。`;
      searchResults.replaceChildren(...results.map(item => resultNode(item.post, terms)));
    } catch { searchStatus.textContent = '目前無法載入搜尋資料。請稍後再試，或前往文章頁依主題瀏覽。'; const link = document.createElement('a'); link.href = '/articles/'; link.className = 'search-result'; link.textContent = '前往全部文章 →'; searchResults.replaceChildren(link); }
  }
  document.querySelectorAll('[data-search]').forEach(button => button.addEventListener('click', () => { closeMenu(); searchTrigger = button; search.showModal(); document.body.classList.add('modal-open'); placeSearch(); searchInput.focus({preventScroll:true}); runSearch(); }));
  searchInput?.addEventListener('input', runSearch);
  search?.querySelector('form').addEventListener('submit', e => { e.preventDefault(); runSearch(); });
  search?.querySelector('.close-search').addEventListener('click', () => search.close());
  search?.addEventListener('keydown', e => { if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); search.close(); } });
  search?.addEventListener('close', () => { document.body.classList.remove('modal-open'); searchTrigger?.focus({preventScroll:true}); });
  search?.addEventListener('click', e => { if (e.target === search) { const r = search.getBoundingClientRect(); if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) search.close(); } });

  const imageDialog = document.querySelector('.image-dialog');
  let imageTrigger = null;
  document.querySelectorAll('.prose img').forEach(img => {
    if (!img.alt) img.alt = img.closest('p')?.textContent.trim().slice(0, 100) || '原文章的插圖';
    const anchor = img.closest('a');
    // Linked images keep their original navigation destination.
    if (anchor) return;
    img.tabIndex = 0; img.setAttribute('role', 'button'); img.setAttribute('aria-label', `放大圖片：${img.alt}`);
    function enlarge() { imageTrigger = img; const full = imageDialog.querySelector('img'); full.src = img.currentSrc || img.src; full.alt = img.alt; imageDialog.querySelector('p').textContent = img.alt; imageDialog.showModal(); document.body.classList.add('modal-open'); }
    img.addEventListener('click', enlarge);
    img.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); enlarge(); } });
  });
  imageDialog?.querySelector('.close-image').addEventListener('click', () => imageDialog.close());
  imageDialog?.addEventListener('close', () => { document.body.classList.remove('modal-open'); imageTrigger?.focus(); });
  imageDialog?.addEventListener('click', e => { if (e.target === imageDialog) imageDialog.close(); });

  document.querySelectorAll('.highlight').forEach(figure => {
    const code = figure.querySelector('.code pre') || figure.querySelector('pre'); if (!code) return;
    const button = document.createElement('button'); button.type = 'button'; button.className = 'copy-code'; button.textContent = '複製'; button.setAttribute('aria-label', '複製這段程式碼');
    button.addEventListener('click', async () => {
      try { await navigator.clipboard.writeText(window.StudioCode?StudioCode.textOf(code):code.innerText); button.textContent = '已複製'; document.getElementById('site-status').textContent = '程式碼已複製'; }
      catch { button.textContent = '請手動選取複製'; document.getElementById('site-status').textContent = '瀏覽器無法複製，請手動選取程式碼'; }
      setTimeout(() => button.textContent = '複製', 2000);
    });
    const scroller = document.createElement('div'); scroller.className = 'code-scroll'; scroller.tabIndex = 0; scroller.setAttribute('role', 'region'); scroller.setAttribute('aria-label', '程式碼，可左右捲動');
    [...figure.childNodes].forEach(node => scroller.append(node));
    const toolbar = document.createElement('div'); toolbar.className = 'code-toolbar';
    const dots = document.createElement('span'); dots.className = 'code-window-dots'; dots.setAttribute('aria-hidden', 'true');
    for (let i = 0; i < 3; i++) dots.append(document.createElement('i'));
    const caption = document.createElement('span'); caption.className = 'code-caption'; caption.textContent = figure.dataset.codeCaption || '程式碼';
    toolbar.append(dots, caption, button); figure.append(toolbar, scroller);
  });
  document.querySelectorAll('.pdfobject-container').forEach(container => { if (!container.dataset.target) return; const frame = document.createElement('iframe'); frame.title = '文章內的 PDF 文件'; frame.loading = 'lazy'; frame.src = `/lib/pdf/web/viewer.html?file=${encodeURIComponent(container.dataset.target)}`; container.append(frame); });
  document.querySelectorAll('.prose a[target="_blank"]').forEach(link => { link.rel = [...new Set((link.rel + ' noopener noreferrer').split(/\s+/))].join(' '); });
  if ('IntersectionObserver' in window && !document.getElementById('article-body')) {
    const headings = [...document.querySelectorAll('.prose h2[id], .prose h3[id]')];
    const tocLinks = [...document.querySelectorAll('.article-toc a')];
    const observer = intersection(entries => { const visible = entries.find(entry => entry.isIntersecting); if (!visible) return; tocLinks.forEach(link => { decodeURIComponent(link.hash.slice(1)) === visible.target.id ? link.setAttribute('aria-current', 'location') : link.removeAttribute('aria-current'); }); }, { rootMargin: '-110px 0px -65% 0px' });
    headings.forEach(heading => observer.observe(heading));
  }
  const topButton = document.querySelector('.back-top');
  let scheduled = false;
  let returnIdleTimer;
  let returnControlActive = false;
  function hideIdleReturn() {
    clearTimeout(returnIdleTimer);
    if (topButton && !topButton.hidden && !returnControlActive) returnIdleTimer = setTimeout(() => { topButton.hidden = true; }, 3500);
  }
  function updateReturnControls() {
    const viewportHeight = window.innerHeight;
    const longPage = document.documentElement.scrollHeight > viewportHeight * 2;
    if (topButton) topButton.hidden = !longPage || window.scrollY < Math.max(650, viewportHeight);
    hideIdleReturn();
  }
  function scheduleReturnControls() {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(() => { updateReturnControls(); scheduled = false; });
  }
  listen(window, 'scroll', scheduleReturnControls, { passive: true });
  listen(window, 'resize', scheduleReturnControls);
  listen(window, 'load', scheduleReturnControls);
  if ('ResizeObserver' in window) resizeObserver(scheduleReturnControls).observe(document.querySelector('main'));
  topButton?.addEventListener('focus', () => { returnControlActive = true; clearTimeout(returnIdleTimer); });
  topButton?.addEventListener('blur', () => { returnControlActive = false; hideIdleReturn(); });
  document.querySelectorAll('[data-return-top]').forEach(button => button.addEventListener('click', () => {
    window.scrollTo({ top: 0, behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth' });
    document.querySelector('.skip-link')?.focus({ preventScroll: true });
  }));
  updateReturnControls();
  const baseDispose = dispose;
  dispose = () => { baseDispose(); clearTimeout(returnIdleTimer); };
  }
  init();
  return { init, dispose: () => dispose?.() };
})();
