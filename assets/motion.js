'use strict';
(() => {
  const root = document.documentElement;
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const finePointer = matchMedia('(hover: hover) and (pointer: fine)');
  const scene = document.querySelector('.usagi-scene');
  const mascot = scene?.querySelector('.usagi-mascot');
  const animations = new Set();
  const canMove = () => !reduced.matches && !document.hidden;
  function syncMotion() {
    const paused = reduced.matches;
    root.classList.toggle('motion-paused', paused);
    root.classList.toggle('motion-hidden', document.hidden);
    if (!canMove()) animations.forEach(animation => animation.cancel());
    if (paused) {
      scene?.style.setProperty('--pointer-x', '0');
      scene?.style.setProperty('--pointer-y', '0');
    }
  }
  reduced.addEventListener('change', syncMotion);
  document.addEventListener('visibilitychange', syncMotion);
  syncMotion();
  function animate(node, frames, options) {
    if (!canMove() || !node.animate) return;
    const animation = node.animate(frames, options);
    animations.add(animation);
    animation.finished.catch(() => {}).finally(() => animations.delete(animation));
    return animation;
  }
  // A bounded CSS animation layer: no canvas, scroll loop, or touch listeners.
  const dessertLayer = document.createElement('div');
  dessertLayer.className = 'dessert-atmosphere';
  dessertLayer.setAttribute('aria-hidden', 'true');
  const smallDessertScreen = matchMedia('(max-width: 760px)');
  const dessertKinds = ['cake', 'donut', 'pudding', 'dango'];
  function arrangeDesserts() {
    const compact = smallDessertScreen.matches;
    const positions = compact ? [2, 86, 9, 80, 4, 89, 7, 84] : [2, 11, 23, 36, 48, 59, 71, 83, 93, 6, 44, 88, 18, 65, 32, 95];
    const fragments = document.createDocumentFragment();
    positions.forEach((left, index) => {
      const particle = document.createElement('span');
      particle.className = 'dessert-particle';
      particle.dataset.dessert = dessertKinds[index % dessertKinds.length];
      particle.style.setProperty('--dessert-left', `${left}%`);
      particle.style.setProperty('--dessert-size', `${(compact ? 27 : 37) + index % 3 * 5}px`);
      particle.style.setProperty('--dessert-duration', `${30 + index % 5 * 5}s`);
      particle.style.setProperty('--dessert-delay', `${-(index + .6) / positions.length * (30 + index % 5 * 5)}s`);
      particle.style.setProperty('--dessert-drift', `${(index % 2 ? -1 : 1) * (compact ? 10 : 24)}px`);
      particle.style.setProperty('--dessert-angle', `${index % 2 ? 14 : -14}deg`);
      const illustration = document.createElement('img');
      illustration.src = `/assets/illustrations/desserts/${particle.dataset.dessert}.svg`;
      illustration.alt = '';
      illustration.width = 96; illustration.height = 96;
      illustration.decoding = 'async'; illustration.draggable = false;
      particle.append(illustration); fragments.append(particle);
    });
    dessertLayer.replaceChildren(fragments);
  }
  document.body.prepend(dessertLayer);
  arrangeDesserts();
  smallDessertScreen.addEventListener('change', arrangeDesserts);
  // A staged entrance gives the title, illustration, and controls their own rhythm.
  document.querySelectorAll('.hero-copy > *, .hero-line').forEach((node, index) => {
    animate(node, [{ opacity:0, transform:'translateY(22px)' }, { opacity:1, transform:'translateY(0)' }], {
      duration:900, delay:Math.min(index * 85, 420), easing:'cubic-bezier(.22,1,.36,1)', fill:'backwards'
    });
  });
  if (scene) animate(scene, [{ opacity:0, transform:'scale(.92) translateY(24px)' }, { opacity:1, transform:'scale(1) translateY(0)' }], { duration:1100, delay:150, easing:'cubic-bezier(.22,1,.36,1)', fill:'backwards' });
  if ('IntersectionObserver' in window) {
    const reveal = new IntersectionObserver(entries => {
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
    new IntersectionObserver(entries => scene.classList.toggle('scene-asleep', !entries[0].isIntersecting)).observe(scene);
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
    new IntersectionObserver(entries => aboutScene.classList.toggle('scene-asleep', !entries[0].isIntersecting)).observe(aboutScene);
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
      new IntersectionObserver(entries => indexScene.classList.toggle('scene-asleep', !entries[0].isIntersecting)).observe(indexScene);
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
  const tocLinks = [...document.querySelectorAll('.article-toc a')];
  let sections = [];
  function measure() {
    viewport = innerHeight;
    if (!article) return;
    const rect = article.getBoundingClientRect();
    articleTop = rect.top + scrollY;
    articleBottom = rect.bottom + scrollY;
    offset = (document.querySelector('.site-header')?.getBoundingClientRect().height || 76) + (document.querySelector('.reading-companion')?.getBoundingClientRect().height || 44);
    sections = [...article.querySelectorAll('h2[id],h3[id]')].map(node => ({ id:node.id, top:node.getBoundingClientRect().top + scrollY }));
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
      tocLinks.forEach(link => {
        if (decodeURIComponent(link.hash.slice(1)) === active?.id) link.setAttribute('aria-current', 'location');
        else link.removeAttribute('aria-current');
      });
    }
    if (scene && canMove() && finePointer.matches && !scene.classList.contains('scene-asleep')) {
      scene.style.setProperty('--scroll-shift', `${Math.min(scrollY * .055, 24).toFixed(1)}px`);
    }
  }
  function schedule() { if (!frame) frame = requestAnimationFrame(update); }
  addEventListener('scroll', schedule, { passive:true });
  addEventListener('resize', () => { measure(); schedule(); }, { passive:true });
  addEventListener('pageshow', () => { measure(); schedule(); });
  if (article && 'ResizeObserver' in window) new ResizeObserver(() => { measure(); schedule(); }).observe(article);
  article?.querySelectorAll('img,iframe').forEach(node => node.addEventListener('load', () => { measure(); schedule(); }));
  document.fonts?.ready.then(() => { measure(); schedule(); });
  measure(); schedule();
})();
