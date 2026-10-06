'use strict';
window.ReaderMotion = (() => {
  let dispose;
  let lastHomeQuote = -1;
  const homeQuotes = [
    ['今天也有', '新的發現！'],
    ['帶著好奇', '出發吧！'],
    ['靈感來了', '快記下來！'],
    ['慢慢探索', '也很棒呀！'],
    ['小小實作', '大大進步！'],
    ['翻開手札', '一起學習！'],
    ['試試看嘛', '會有驚喜！'],
    ['呀哈！今天', '也要開心！']
  ];
  function init() {
  dispose?.();
  const scope = new AbortController();
  const observers = [];
  const listen = (target, type, callback, options = {}) => target.addEventListener(type, callback, { ...options, signal: scope.signal });
  const intersection = (...args) => { const item = new IntersectionObserver(...args); observers.push(item); return item; };
  const resizeObserver = (...args) => { const item = new ResizeObserver(...args); observers.push(item); return item; };
  dispose = () => { scope.abort(); observers.forEach(item => item.disconnect()); };
  const root = document.documentElement;
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const finePointer = matchMedia('(hover: hover) and (pointer: fine)');
  const scene = document.querySelector('.usagi-scene');
  const homeQuote = scene?.querySelector('.book-note > span');
  if (homeQuote) {
    try {
      const saved = sessionStorage.getItem('usagi-home-quote');
      if (saved !== null && /^\d+$/.test(saved)) lastHomeQuote = Number(saved);
    } catch { /* Keep visits varied in memory when storage is unavailable. */ }
    const choices = homeQuotes.map((_, index) => index).filter(index => index !== lastHomeQuote);
    lastHomeQuote = choices[Math.floor(Math.random() * choices.length)];
    const [first, second] = homeQuotes[lastHomeQuote];
    homeQuote.replaceChildren(document.createTextNode(first), document.createElement('br'), document.createTextNode(second));
    try { sessionStorage.setItem('usagi-home-quote', String(lastHomeQuote)); } catch { /* Optional session persistence. */ }
  }
  const mascot = scene?.querySelector('.usagi-mascot');
  const heroArt = mascot?.querySelector('.hero-usagi[data-animated-src]');
  const heroStill = heroArt?.getAttribute('src');
  const animations = new Set();
  const canMove = () => !reduced.matches && !document.hidden;
  function syncHeroArt() {
    if (!heroArt) return;
    // GIFs cannot use CSS animation-play-state: swap to the original still.
    const source = canMove() && !scene.classList.contains('scene-asleep')
      ? heroArt.dataset.animatedSrc : heroStill;
    if (heroArt.getAttribute('src') !== source) heroArt.setAttribute('src', source);
  }
  function syncMotion() {
    const paused = reduced.matches;
    root.classList.toggle('motion-paused', paused);
    root.classList.toggle('motion-hidden', document.hidden);
    syncHeroArt();
    if (!canMove()) animations.forEach(animation => animation.cancel());
    if (paused) {
      scene?.style.setProperty('--pointer-x', '0');
      scene?.style.setProperty('--pointer-y', '0');
    }
  }
  listen(reduced, 'change', syncMotion);
  listen(document, 'visibilitychange', syncMotion);
  syncMotion();
  function animate(node, frames, options) {
    if (!canMove() || !node.animate) return;
    const animation = node.animate(frames, options);
    animations.add(animation);
    animation.finished.catch(() => {}).finally(() => animations.delete(animation));
    return animation;
  }
  // Keep the original portrait intact; animate its camera framing and frame separately.
  const portrait = document.querySelector('.about-portrait');
  if (portrait) {
    const resetTilt = () => {
      portrait.style.setProperty('--portrait-rx', '0deg');
      portrait.style.setProperty('--portrait-ry', '0deg');
    };
    const syncPortrait = () => {
      if (reduced.matches || document.hidden) resetTilt();
    };
    portrait.classList.add('portrait-live');
    listen(reduced, 'change', syncPortrait);
    listen(document, 'visibilitychange', syncPortrait);
    listen(portrait, 'pointermove', event => {
      if (!canMove() || !finePointer.matches || event.pointerType === 'touch') return;
      const box = portrait.getBoundingClientRect();
      const x = Math.max(-1, Math.min(1, (event.clientX - box.left) / box.width * 2 - 1));
      const y = Math.max(-1, Math.min(1, (event.clientY - box.top) / box.height * 2 - 1));
      portrait.style.setProperty('--portrait-rx', `${(-y * 2).toFixed(2)}deg`);
      portrait.style.setProperty('--portrait-ry', `${(x * 3).toFixed(2)}deg`);
    });
    listen(portrait, 'pointerleave', resetTilt);
    if ('IntersectionObserver' in window) intersection(entries => {
      portrait.classList.toggle('portrait-offscreen', !entries[0].isIntersecting);
    }, { threshold: 0 }).observe(portrait);
    syncPortrait();
  }
  // A bounded CSS animation layer: no canvas, scroll loop, or touch listeners.
  const dessertLayer = document.createElement('div');
  dessertLayer.className = 'dessert-atmosphere';
  dessertLayer.setAttribute('aria-hidden', 'true');
  const smallDessertScreen = matchMedia('(max-width: 760px)');
  const dessertKinds = ['cake', 'donut', 'pudding', 'dango', 'cake-chocolate', 'macaron', 'cupcake', 'ice-cream'];
  const desktopDesserts = [...dessertKinds, 'donut', 'pudding', 'dango', 'macaron', 'donut', 'pudding', 'dango', 'cupcake'];
  function arrangeDesserts() {
    const compact = smallDessertScreen.matches;
    const positions = compact ? [2, 86, 9, 80, 4, 89, 7, 84] : [2, 11, 23, 36, 48, 59, 71, 83, 93, 6, 44, 88, 18, 65, 32, 95];
    const fragments = document.createDocumentFragment();
    positions.forEach((left, index) => {
      const particle = document.createElement('span');
      particle.className = 'dessert-particle';
      particle.dataset.dessert = (compact ? dessertKinds : desktopDesserts)[index];
      particle.style.setProperty('--dessert-left', `${left}%`);
      particle.style.setProperty('--dessert-size', `${(compact ? 27 : 37) + index % 3 * 5}px`);
      particle.style.setProperty('--dessert-duration', `${30 + index % 5 * 5}s`);
      particle.style.setProperty('--dessert-delay', `${-(index + .6) / positions.length * (30 + index % 5 * 5)}s`);
      particle.style.setProperty('--dessert-drift', `${(index % 2 ? -1 : 1) * (compact ? 10 : 24)}px`);
      particle.style.setProperty('--dessert-angle', `${index % 2 ? 14 : -14}deg`);
      if (!compact) {
        // Each dessert follows its own winding route, bounded by the screen edges.
        const min = Math.max(-7, 1 - left);
        const max = Math.min(7, 96 - left);
        for (let step = 1; step <= 6; step++) {
          const band = (step + index) % 2 ? .65 : .05;
          const offset = min + (max - min) * (band + Math.random() * .3);
          particle.style.setProperty(`--dessert-route-${step}`, `${offset.toFixed(2)}vw`);
          particle.style.setProperty(`--dessert-turn-${step}`, `${(-19 + Math.random() * 38).toFixed(1)}deg`);
        }
      }
      const illustration = document.createElement('img');
      illustration.src = `/assets/illustrations/desserts/${particle.dataset.dessert}.svg`;
      illustration.alt = '';
      illustration.width = 96; illustration.height = 96;
      illustration.decoding = 'async'; illustration.draggable = false;
      particle.append(illustration); fragments.append(particle);
    });
    dessertLayer.replaceChildren(fragments);
  }
  document.querySelector('.dessert-atmosphere')?.remove();
  document.body.prepend(dessertLayer);
  arrangeDesserts();
  listen(smallDessertScreen, 'change', arrangeDesserts);
  // A staged entrance gives the title, illustration, and controls their own rhythm.
  document.querySelectorAll('.hero-copy > *, .hero-line').forEach((node, index) => {
    animate(node, [{ opacity:0, transform:'translateY(22px)' }, { opacity:1, transform:'translateY(0)' }], {
      duration:900, delay:Math.min(index * 85, 420), easing:'cubic-bezier(.22,1,.36,1)', fill:'backwards'
    });
  });
  if (scene) animate(scene, [{ opacity:0, transform:'scale(.92) translateY(24px)' }, { opacity:1, transform:'scale(1) translateY(0)' }], { duration:1100, delay:150, easing:'cubic-bezier(.22,1,.36,1)', fill:'backwards' });
  if ('IntersectionObserver' in window) {
    const reveal = intersection(entries => {
      entries.forEach(entry => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add('is-visible');
        reveal.unobserve(entry.target);
      });
    }, { threshold:.08, rootMargin:'0px 0px 30px 0px' });
    document.querySelectorAll('.article-card, .topic-card, .collection-card, .section-heading, .about-strip, .project-card, .history-list article').forEach((node, index) => {
      node.dataset.reveal = '';
      node.style.setProperty('--reveal-delay', `${(index % 3) * 65}ms`);
      reveal.observe(node);
    });
  }
  // Pointer movement is coalesced into one frame; touch screens never run it.
  let pointerFrame = 0;
  let pointerX = 0, pointerY = 0;
  scene?.addEventListener('pointermove', event => {
    if (!canMove() || !finePointer.matches || event.pointerType === 'touch') return;
    const rect = scene.getBoundingClientRect();
    pointerX = Math.max(-1, Math.min(1, (event.clientX - rect.left) / rect.width * 2 - 1));
    pointerY = Math.max(-1, Math.min(1, (event.clientY - rect.top) / rect.height * 2 - 1));
    if (pointerFrame) return;
    pointerFrame = requestAnimationFrame(() => {
      scene.style.setProperty('--pointer-x', pointerX.toFixed(3));
      scene.style.setProperty('--pointer-y', pointerY.toFixed(3));
      pointerFrame = 0;
    });
  });
  scene?.addEventListener('pointerleave', () => {
    cancelAnimationFrame(pointerFrame); pointerFrame = 0;
    scene.style.setProperty('--pointer-x', '0');
    scene.style.setProperty('--pointer-y', '0');
  });
  let greetingTimer = 0;
  const greetings = ['呀哈！一起探索吧。', '烏拉！又多一點靈感。', '今天也要保持好奇！'];
  let greetingIndex = 0;
  mascot?.addEventListener('click', () => {
    clearTimeout(greetingTimer);
    scene.querySelector('.usagi-speech').textContent = greetings[greetingIndex++ % greetings.length];
    scene.classList.add('is-greeting');
    greetingTimer = setTimeout(() => scene.classList.remove('is-greeting'), 2600);
    // Clear a previous burst before creating a new one; rapid input stays bounded.
    scene.querySelector('.scene-confetti').replaceChildren();
    animations.forEach(animation => { if (animation.effect?.target === mascot || animation.effect?.target?.classList?.contains('confetti-piece')) animation.cancel(); });
    animate(mascot, [
      { transform:'translateY(0) rotate(0) scale(1)', offset:0 },
      { transform:'translateY(8px) rotate(-5deg) scale(1.08,.9)', offset:.18 },
      { transform:'translateY(-36px) rotate(12deg) scale(.96,1.06)', offset:.43 },
      { transform:'translateY(0) rotate(-7deg) scale(1.05,.96)', offset:.7 },
      { transform:'translateY(-9px) rotate(3deg) scale(1)', offset:.86 },
      { transform:'translateY(0) rotate(0) scale(1)', offset:1 }
    ], { duration:950, easing:'cubic-bezier(.2,.7,.3,1)' });
    if (!canMove()) return;
    const colors = ['#bd8b28', '#95a578', '#ce977d', '#d4b66a'];
    for (let i=0; i<16; i++) {
      const particle = document.createElement('span');
      particle.className = 'confetti-piece'; particle.textContent = i % 3 ? '✦' : '✿';
      particle.style.setProperty('--particle-color', colors[i % colors.length]);
      scene.querySelector('.scene-confetti').append(particle);
      const angle = i / 16 * Math.PI * 2;
      const distance = Math.min(scene.clientWidth * .42, 190);
      const x = Math.cos(angle) * distance, y = Math.sin(angle) * distance;
      const burst = animate(particle, [
        { opacity:0, transform:'translate(0,0) scale(.3)' },
        { opacity:1, offset:.2 },
        { opacity:0, transform:`translate(${x}px,${y}px) rotate(${i * 30}deg) scale(.5)` }
      ], { duration:1100 + (i % 4) * 100, easing:'cubic-bezier(.12,.7,.3,1)' });
      if (burst) burst.finished.catch(() => {}).finally(() => particle.remove()); else particle.remove();
    }
  });
  // Keep decorative loops asleep once the hero leaves the viewport.
  if (scene && 'IntersectionObserver' in window) {
    intersection(entries => {
      scene.classList.toggle('scene-asleep', !entries[0].isIntersecting);
      syncHeroArt();
    }).observe(scene);
  }

  const aboutScene = document.querySelector('.about-usagi-stage');
  const aboutCharacter = aboutScene?.querySelector('.about-usagi-character');
  aboutCharacter?.addEventListener('click', () => {
    aboutScene.querySelector('.about-usagi-message').textContent = '烏拉！把好奇，寫成下一步。';
    animations.forEach(animation => { if (animation.effect?.target === aboutCharacter) animation.cancel(); });
    animate(aboutCharacter, [
      { transform:'translateY(0) rotate(0)' },
      { transform:'translateY(6px) scale(1.04,.94)', offset:.18 },
      { transform:'translateY(-28px) rotate(9deg)', offset:.45 },
      { transform:'translateY(0) rotate(-4deg)', offset:.75 },
      { transform:'translateY(0) rotate(0)' }
    ], { duration:850, easing:'cubic-bezier(.2,.7,.3,1)' });
  });
  if (aboutScene && 'IntersectionObserver' in window) {
    intersection(entries => aboutScene.classList.toggle('scene-asleep', !entries[0].isIntersecting)).observe(aboutScene);
  }

  document.querySelectorAll('.index-usagi-scene').forEach(indexScene => {
    const character = indexScene.querySelector('.index-usagi-character');
    const topics = indexScene.dataset.scene === 'topics';
    character.addEventListener('click', () => {
      indexScene.querySelector('.index-usagi-message').textContent = topics ? '呀哈！一起找到想探索的主題。' : '烏拉！又寫下了一點新發現。';
      animations.forEach(animation => { if (animation.effect?.target === character) animation.cancel(); });
      animate(character, topics ? [
        { transform:'rotate(0) translateY(0)' },
        { transform:'rotate(-7deg) translateY(3px)', offset:.2 },
        { transform:'rotate(7deg) translateY(-17px)', offset:.5 },
        { transform:'rotate(0) translateY(0)' }
      ] : [
        { transform:'translateY(0) scale(1)' },
        { transform:'translateY(4px) scale(1.04,.96)', offset:.25 },
        { transform:'translateY(-13px) scale(.98,1.02)', offset:.55 },
        { transform:'translateY(0) scale(1)' }
      ], { duration:800, easing:'cubic-bezier(.2,.7,.3,1)' });
    });
    if ('IntersectionObserver' in window) {
      intersection(entries => indexScene.classList.toggle('scene-asleep', !entries[0].isIntersecting)).observe(indexScene);
    }
  });

  const article = document.getElementById('article-body');
  const progress = document.querySelector('.reading-progress');
  const percent = document.querySelector('.reader-percent');
  const label = document.querySelector('.reader-progress-label');
  const remaining = document.querySelector('.reader-remaining');
  const minutes = Number(remaining?.textContent.match(/\d+/)?.[0]) || 1;
  const readingFinish = document.querySelector('.reading-finish');
  let readingCompleted = false;
  let frame = 0, articleTop = 0, articleBottom = 0, offset = 0, viewport = innerHeight;
  const articleToc = document.querySelector('.article-toc');
  const tocNav = articleToc?.querySelector('nav');
  let followedChapter = null, followToc = true;
  const tocLinks = [...document.querySelectorAll('.article-toc a')];
  function revealCurrentChapter(link) {
    if (!link || !articleToc.open || !tocNav || tocNav.clientHeight === 0 || tocNav.scrollHeight <= tocNav.clientHeight) return;
    // Scroll only the TOC viewport; scrolling ancestors would move the article.
    const viewport = tocNav.getBoundingClientRect();
    const item = link.getBoundingClientRect();
    const top = viewport.top + tocNav.clientTop + 8;
    const bottom = viewport.top + tocNav.clientTop + tocNav.clientHeight - 8;
    if (item.top < top) tocNav.scrollTop += item.top - top;
    else if (item.bottom > bottom) tocNav.scrollTop += Math.min(item.bottom - bottom, item.top - top);
  }
  const chapterTitles = new Map(tocLinks.map(link => [decodeURIComponent(link.hash.slice(1)), link.querySelector('.toc-title')?.textContent.trim() || link.textContent.trim()]));
  let sections = [];
  function measure() {
    if (scope.signal.aborted) return;
    viewport = innerHeight;
    if (!article) return;
    const rect = article.getBoundingClientRect();
    articleTop = rect.top + scrollY;
    articleBottom = rect.bottom + scrollY;
    offset = (document.querySelector('.site-header')?.getBoundingClientRect().height || 76) + (document.querySelector('.reading-companion')?.getBoundingClientRect().height || 44);
    sections = [...article.querySelectorAll('h1[id],h2[id],h3[id]')].filter(node => chapterTitles.has(node.id)).map(node => ({ id:node.id, title:chapterTitles.get(node.id), top:node.getBoundingClientRect().top + scrollY }));
  }
  function update() {
    frame = 0;
    if (article && progress) {
      // 0% when the first body content reaches the reading edge; 100% when its end is visible.
      const start = articleTop - offset;
      const end = Math.max(start + 1, articleBottom - viewport);
      const ratio = Math.max(0, Math.min(1, (scrollY - start) / (end - start)));
      const rounded = Math.floor(ratio * 100);
      const complete = ratio >= 1;
      if (readingFinish && complete !== readingCompleted) {
        readingFinish.hidden = !complete;
        if (complete) animate(readingFinish, [{opacity:0,transform:'translateY(16px)'},{opacity:1,transform:'translateY(0)'}], {duration:600,easing:'cubic-bezier(.22,1,.36,1)'});
        readingCompleted = complete;
      }
      root.style.setProperty('--reading-progress', ratio.toFixed(5));
      progress.setAttribute('aria-valuenow', String(rounded));
      progress.setAttribute('aria-valuetext', `已閱讀 ${rounded}%`);
      percent.textContent = `${rounded}%`;
      label.textContent = ratio >= 1 ? '讀完了，烏拉！' : '一起慢慢讀';
      remaining.textContent = ratio >= 1 ? '本篇已讀完' : `約剩 ${Math.max(1, Math.ceil(minutes * (1 - ratio)))} 分鐘`;
      const active = sections.filter(section => section.top <= scrollY + offset + 45).at(-1);
      const currentChapter = document.querySelector('.toc-current');
      if (currentChapter) currentChapter.textContent = active?.title || '從第一章開始';
      let currentLink;
      tocLinks.forEach(link => {
        if (decodeURIComponent(link.hash.slice(1)) === active?.id) {
          link.setAttribute('aria-current', 'location');
          currentLink = link;
        } else link.removeAttribute('aria-current');
      });
      if (followToc || followedChapter !== active?.id) revealCurrentChapter(currentLink);
      followedChapter = active?.id;
      followToc = false;
    }
    if (scene && canMove() && finePointer.matches && !scene.classList.contains('scene-asleep')) {
      scene.style.setProperty('--scroll-shift', `${Math.min(scrollY * .055, 24).toFixed(1)}px`);
    }
  }
  function schedule() { if (!scope.signal.aborted && !frame) frame = requestAnimationFrame(update); }
  listen(window, 'scroll', schedule, { passive:true });
  listen(window, 'resize', () => { followToc = true; measure(); schedule(); }, { passive:true });
  listen(window, 'pageshow', () => { measure(); schedule(); });
  listen(articleToc || document, 'toggle', () => { followToc = true; measure(); schedule(); });
  if (article && 'ResizeObserver' in window) resizeObserver(() => { measure(); schedule(); }).observe(article);
  article?.querySelectorAll('img,iframe').forEach(node => node.addEventListener('load', () => { measure(); schedule(); }));
  document.fonts?.ready.then(() => { measure(); schedule(); });
  measure(); schedule();
  const baseDispose = dispose;
  dispose = () => { baseDispose(); animations.forEach(item => item.cancel()); cancelAnimationFrame(frame); cancelAnimationFrame(pointerFrame); clearTimeout(greetingTimer); };
  }
  init();
  return { init, dispose: () => dispose?.() };
})();
