/* One preference for the lobby, keyboard games and USB games. Works on file:// too. */
(() => {
  'use strict';
  const key = 'droneSoccer.ai.difficulty';
  const levels = Object.freeze({
    easy: { label: '簡單', description: '同等機體能力：一台防守直接迎擊、一台補門，判斷較慢；進攻先嘗試直接突破，受阻約 3 秒重試；連續三次失敗後側繞。' },
    medium: { label: '中等', description: '同等機體能力：防守預判攔截、隊友補門，失位時交接；進攻引開迎擊者、側切找空檔，記住受阻路線並換線再攻。' },
    hard: { label: '困難', description: '同等機體能力：防守提前迎擊、快速交接，隊友持續補門；進攻預判空檔、引守側切，依本場突破成敗調整路線，受阻約 1.6 秒後重攻。' }
  });
  // Survival reflexes remain enabled at every level; difficulty controls how
  // early a pilot reads danger, rather than deliberately crashing beginners.
  const safetyProfiles = Object.freeze({
    easy: Object.freeze({ horizon: 0.65, reaction: 0.24, margin: 0.07 }),
    medium: Object.freeze({ horizon: 0.9, reaction: 0.14, margin: 0.09 }),
    hard: Object.freeze({ horizon: 1.15, reaction: 0.08, margin: 0.11 })
  });
  // Difficulty changes decisions, never the airframe or flight controller.
  // Keep speed, acceleration, tracking and burst limits identical for all tiers.
  const flightProfile = Object.freeze({ maxSpeed: 2.4, accel: 7, trackingGain: 4 });
  const profiles = {
    guard: {
      easy: { ...flightProfile, label: '簡單',
        guardRadiusXY: 0.62, guardDepth: 0.95, reactionMin: 0.85, reactionMax: 1.25,
        aimError: 0.24, engageDistance: 1.5, engageChance: 0.05, standbyRange: 1.5,
        tactics: { angleCoverage: false, predictiveIntercept: false, coverRotation: false, holdLine: true, activeIntercept: true } },
      medium: { ...flightProfile, label: '中等',
        guardRadiusXY: 0.62, guardDepth: 0.95, reactionMin: 0.28, reactionMax: 0.44,
        aimError: 0.10, engageDistance: 2.2, engageChance: 0.50, standbyRange: 2.7,
        predictionHorizon: 0.3, coverSpread: 0.18, engageDepthRatio: 0.45, interceptDiscipline: 0.68,
        tactics: { angleCoverage: true, predictiveIntercept: true, coverRotation: true, holdLine: true, activeIntercept: true } },
      hard: { ...flightProfile, label: '困難',
        guardRadiusXY: 0.62, guardDepth: 0.95, reactionMin: 0.06, reactionMax: 0.12,
        aimError: 0.015, engageDistance: 3.4, engageChance: 0.97, standbyRange: 4,
        predictionHorizon: 0.85, coverSpread: 0.12, engageDepthRatio: 0.7, interceptDiscipline: 1.0,
        tactics: { angleCoverage: true, predictiveIntercept: true, coverRotation: true, holdLine: true, activeIntercept: true } }
    },
    attack: {
      easy: { ...flightProfile, label: '簡單', weaveAmplitude: 1.0, weaveAmplitudeY: 0.45, diagonalBias: 0.25,
        approachDistance: 1.05, laneCommitDistance: 0.5, predictionHorizon: 0, retryAfter: 3.0, adaptiveRetry: false,
        reactionMin: 0.75, reactionMax: 1.10, aimError: 0.10, gapReactionMin: 1.4, gapReactionMax: 2.0,
        commitToGapMin: 0.05, commitToGapMax: 0.20, evadeDistance: 0.25,
        patterns: ['diagonal_weave', 'low_skim'], jukeChance: 0, jukeBurstMultiplier: 1.35,
        tactics: { smartFeint: false, predictiveGap: false, smartEvade: false, smartJukeDirection: false } },
      medium: { ...flightProfile, label: '中等', weaveAmplitude: 1.6, weaveAmplitudeY: 0.9, diagonalBias: 0.4,
        approachDistance: 1.4, laneCommitDistance: 0.5, predictionHorizon: 0.25, retryAfter: 2.2, adaptiveRetry: true,
        reactionMin: 0.30, reactionMax: 0.48, aimError: 0.045, gapReactionMin: 0.60, gapReactionMax: 0.90,
        commitToGapMin: 0.30, commitToGapMax: 0.65, evadeDistance: 0.50,
        patterns: ['diagonal_weave', 's_curve', 'high_dive', 'low_skim'], jukeChance: 0.22, jukeBurstMultiplier: 1.35,
        tactics: { smartFeint: false, predictiveGap: true, smartEvade: true, smartJukeDirection: false, flankDefense: true } },
      hard: { ...flightProfile, label: '困難', weaveAmplitude: 1.9, weaveAmplitudeY: 1.15, diagonalBias: 0.5,
        approachDistance: 1.8, laneCommitDistance: 0.35, predictionHorizon: 0.65, retryAfter: 1.6, adaptiveRetry: true,
        reactionMin: 0.06, reactionMax: 0.12, aimError: 0.01, gapReactionMin: 0.12, gapReactionMax: 0.22,
        commitToGapMin: 0.70, commitToGapMax: 0.98, evadeDistance: 0.85,
        // Active 1v2 trials favoured stable lateral curves over repeated
        // vertical jukes, which shed alignment and momentum before contact.
        patterns: ['diagonal_weave', 's_curve'], jukeChance: 0, jukeBurstMultiplier: 1.35,
        tactics: { smartFeint: true, predictiveGap: true, smartEvade: true, smartJukeDirection: true, flankDefense: true } }
    },
    practice: {
      easy: { ...flightProfile, reactionMin: 0.85, reactionMax: 1.25,
        aimError: 0.07, guardRadiusXY: 0.62, guardDepth: 0.95, engageChance: 0, pause: 3.5, standbyRange: 1.5, engageDistance: 1.5,
        approachDistance: 1.05, laneCommitDistance: 0.5, predictionHorizon: 0, retryAfter: 3.0, adaptiveRetry: false, commitToGapMax: 0.2,
        tactics: { predictiveIntercept: false, angleCoverage: false, coverRotation: false, predictiveGap: false, holdLine: true, activeIntercept: true } },
      medium: { ...flightProfile, reactionMin: 0.28, reactionMax: 0.44,
        aimError: 0.035, guardRadiusXY: 0.62, guardDepth: 0.95, engageChance: 0.45, pause: 1.7, standbyRange: 2.7, engageDistance: 2.2,
        approachDistance: 1.4, laneCommitDistance: 0.5, predictionHorizon: 0.25, retryAfter: 2.2, adaptiveRetry: true, commitToGapMax: 0.65, coverSpread: 0.18, engageDepthRatio: 0.45, interceptDiscipline: 0.68,
        tactics: { predictiveIntercept: true, angleCoverage: true, coverRotation: true, predictiveGap: true, flankDefense: true, holdLine: true, activeIntercept: true } },
      hard: { ...flightProfile, reactionMin: 0.06, reactionMax: 0.12,
        aimError: 0.01, guardRadiusXY: 0.62, guardDepth: 0.95, engageChance: 0.97, pause: 0.6, standbyRange: 4, engageDistance: 3.4,
        approachDistance: 1.8, laneCommitDistance: 0.35, predictionHorizon: 0.65, retryAfter: 1.6, adaptiveRetry: true, commitToGapMax: 0.98, coverSpread: 0.12, engageDepthRatio: 0.7, interceptDiscipline: 1.0,
        tactics: { predictiveIntercept: true, angleCoverage: true, coverRotation: true, predictiveGap: true, smartFeint: true, flankDefense: true, holdLine: true, activeIntercept: true } }
    }
  };
  const valid = value => Object.hasOwn(levels, value);
  let level = 'medium';
  let adapter = null;
  try {
    const saved = localStorage.getItem(key);
    if (valid(saved)) level = saved;
  } catch (_) { /* Keep session controls usable when storage is disabled. */ }
  const requested = new URLSearchParams(location.search).get('difficulty');
  if (valid(requested)) level = requested;
  window.DroneAISettings = Object.freeze({
    levels, profiles, safetyProfiles, valid,
    get level() { return level; },
    register(game) { adapter = game; },
    applyLevel(value) {
      if (!valid(value)) return false;
      if (adapter) adapter.apply(value);
      else window.DroneAISettings.setLevel(value);
      return true;
    },
    setLevel(value) {
      if (!valid(value)) return false;
      level = value;
      try { localStorage.setItem(key, level); } catch (_) { /* Session-only fallback. */ }
      if (/\/games\//.test(location.pathname)) {
        try {
          const url = new URL(location.href);
          url.searchParams.set('difficulty', level);
          history.replaceState(history.state, '', url);
        } catch (_) { /* History may be restricted on file://. */ }
      }
      document.querySelectorAll('[data-ai-level]').forEach(select => { select.value = level; });
      document.querySelectorAll('[class*="difficulty-option"][data-level]').forEach(option => {
        option.setAttribute('aria-pressed', String(option.dataset.level === level));
      });
      window.dispatchEvent(new CustomEvent('drone-ai-difficulty-change', { detail: { level } }));
      return true;
    }
  });
  window.addEventListener('storage', event => {
    if (event.key !== key || !valid(event.newValue)) return;
    if (adapter) { adapter.apply(event.newValue); return; }
    level = event.newValue;
    window.dispatchEvent(new CustomEvent('drone-ai-difficulty-change', { detail: { level } }));
  });
  document.addEventListener('DOMContentLoaded', () => {
    // Constants have already been built using this preference. Synchronize the menus and HUD.
    adapter?.refresh?.();
    document.querySelectorAll('[class*="difficulty-option"][data-level]').forEach(option => {
      option.setAttribute('role', 'button');
      option.tabIndex = 0;
      option.setAttribute('aria-pressed', String(option.dataset.level === level));
      if(levels[option.dataset.level])option.title=levels[option.dataset.level].description;
      const detail = option.querySelector('p:nth-of-type(2)');
      if (detail) detail.textContent = { easy:'直接迎擊・隊友補門', medium:'預判攔截・分工補門', hard:'提前迎擊・失位交接' }[option.dataset.level] || detail.textContent;
    });
    const groups=new Set(Array.from(document.querySelectorAll('.difficulty-option[data-level]'),option=>option.parentElement));
    groups.forEach(group=>{
      const hint=document.createElement('p');hint.className='fc-ai-description';hint.setAttribute('role','status');
      const refresh=()=>{hint.textContent=levels[level].description;};
      group.after(hint);refresh();window.addEventListener('drone-ai-difficulty-change',refresh);
    });
    // The human attacker + human defender mode previously had no AI teammate controls.
    if (adapter?.kind === 'ally') {
      const row = document.createElement('label');
      row.className = 'fc-camera-setting fc-ui-region fc-ai-setting';
      row.textContent = 'AI 隊友智能難度 ';
      const select = document.createElement('select');
      select.dataset.aiLevel = '';
      select.setAttribute('aria-label', 'AI 隊友智能難度');
      for (const [value, info] of Object.entries(levels)) select.add(new Option(info.label, value));
      select.value = level;
      select.addEventListener('change', () => window.DroneAISettings.applyLevel(select.value));
      row.append(select);
      document.querySelector('[id$="startScreen"] .fc-page')?.prepend(row);
      const live = row.cloneNode(true);
      live.querySelector('select').value = level;
      live.querySelector('select').addEventListener('change', event => window.DroneAISettings.applyLevel(event.target.value));
      const actions = document.querySelector('.fc-more-actions') || document.querySelector('.fc-system-actions');
      if (actions) actions.insertBefore(live, actions.querySelector('.fc-action-reset'));
    }
  });
})();
