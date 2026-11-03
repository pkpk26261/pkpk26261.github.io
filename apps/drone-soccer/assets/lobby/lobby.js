(() => {
  'use strict';

  try { localStorage.removeItem('flight-club-color-theme-v1'); } catch (_) { /* Storage is optional. */ }

  // All playable modes use the shared USB/gamepad controls.
  const modes = [
    { id: 'attack', category: 'battle', title: '單人進攻', description: '突破 AI 防線，挑戰你的進球能力。', detail: '操控攻擊機突破 AI 防線，穿越對方球門得分。', players: 1, lineup: '玩家攻擊機 vs 2 台 AI 防守機', tag: '進攻挑戰', badge: '挑戰 AI', file: 'drone_soccer_competition.html' },
    { id: 'defense', category: 'battle', title: '單人防守', description: '守住球門，攔下 AI 的每次進攻。', detail: '駕駛防守機攔截 AI 攻擊機，練習站位與防守反應。', players: 1, lineup: '玩家防守機 vs AI 攻擊機', tag: '防守挑戰', badge: '挑戰 AI', file: 'drone_soccer_defense.html' },
    { id: 'attack-defense', category: 'battle', title: '攻守對決', description: '一攻一守，再由 AI 助陣防線。', detail: '兩位玩家分別操控攻擊機與防守機，另有一台 AI 防守機協助防守。', players: 2, lineup: '1 攻擊機 vs 1 防守機＋1 AI', tag: '雙人對抗', badge: 'AI 助陣', file: 'drone_soccer_competition_ad.html' },
    { id: 'double-defense', category: 'battle', title: '雙人聯防', description: '兩位玩家合作，阻擋 AI 攻擊機。', detail: '兩位玩家各駕駛一台防守機，一起建立防線，阻擋 AI 攻擊機。', players: 2, lineup: 'AI 攻擊機 vs 2 玩家防守機', tag: '合作防守', badge: '雙人合作', file: 'drone_soccer_competition_dd.html' },
    { id: 'one-vs-two', category: 'battle', title: '一攻二守', description: '一位前鋒、兩位守門，正面交鋒。', detail: '三位玩家各自駕駛一台無人機，由一位攻擊手挑戰兩位防守球員。', players: 3, lineup: '1 玩家攻擊機 vs 2 玩家防守機', tag: '三人對抗', badge: '玩家對戰', file: 'drone_soccer_competition_add.html' },
    { id: 'team-ai', category: 'battle', title: '組隊挑戰 AI', description: '藍隊三人集結，迎戰紅隊 AI。', detail: '三位玩家組成藍隊，分工進攻與防守，迎戰紅隊 AI。', players: 3, lineup: '紅隊 AI vs 藍隊 3 位玩家', tag: '團隊合作', badge: '3 vs AI', file: 'drone_soccer_competition_add_ai.html' },
    { id: 'six-player', category: 'battle', title: '紅藍全員對戰', description: '六人分成兩隊，來場完整對決。', detail: '六位玩家分為紅、藍兩隊，每隊一位攻擊手與兩位防守球員。', players: 6, lineup: '紅隊 3 位玩家 vs 藍隊 3 位玩家', tag: '六人對抗', badge: '3 vs 3', file: 'drone_soccer_competition_addadd.html' },
    { id: 'practice', category: 'practice', title: '自由飛行練習', description: '進攻挑戰兩台 AI，防守有 AI 隊友協助。', detail: '前鋒機突破兩台 AI 防守；防守機與一台 AI 隊友共同攔截 AI 對手；不限時，可切換角色與 AI 強度，依自己的步調練習。', players: 1, lineup: '進攻 1 對 2・防守 2 對 1', tag: '基礎飛行', badge: '新手推薦', file: 'drone_soccer_practise.html' },
    { id: 'rings', category: 'racing', title: '高低環', description: '高低交錯穿環，掌握升降節奏。', detail: '挑戰不同高度的環門，在升降與前進之間找到穿越的節奏。', players: 1, lineup: '高低環門賽道', tag: '高度控制', badge: '單人競速', file: 'drone_soccer_racing_1.html' },
    { id: 'maze', category: 'racing', title: '立體迷宮', description: '穿梭立體路徑，挑戰空間判斷。', detail: '在立體迷宮中規劃路線，練習空間判斷與精準轉向。', players: 1, lineup: '立體迷宮賽道', tag: '空間挑戰', badge: '單人競速', file: 'drone_soccer_racing_2.html' },
    { id: 'grid', category: 'racing', title: '九宮格', description: '瞄準格位，練習精準穿越。', detail: '控制高度與方向，穿越九宮格，提升飛行的準確度。', players: 1, lineup: '九宮格穿越賽道', tag: '精準穿越', badge: '單人競速', file: 'drone_soccer_racing_3.html' },
    { id: 'tunnel', category: 'racing', title: '單軌隧道', description: '沿著隧道前進，保持穩定航線。', detail: '沿單軌隧道持續前進，挑戰穩定操控與路徑追蹤能力。', players: 1, lineup: '單軌隧道賽道', tag: '穩定操控', badge: '單人競速', file: 'drone_soccer_racing_4.html' },
    { id: 'slalom', category: 'racing', title: '蛇形繞標', description: '左右交錯穿環，練習連續側移。', detail: '三個難度，依序穿越左右交錯的環門，最後一環通過即過關並顯示完成秒數。', players: 1, lineup: '蛇形環門賽道・3 個難度', tag: '連續轉向', badge: '單人競速', file: 'drone_soccer_racing_5.html' },
    { id: 'spiral', category: 'racing', title: '螺旋爬升', description: '沿螺旋環門盤旋，穩定爬升。', detail: '三個難度，沿螺旋航線依序穿環爬升，最後一環通過即過關並顯示完成秒數。', players: 1, lineup: '螺旋爬升賽道・3 個難度', tag: '盤旋爬升', badge: '單人競速', file: 'drone_soccer_racing_6.html' },
    { id: 'official-ai', category: 'official', title: '正式賽・挑戰 AI', description: '藍隊玩家迎戰紅隊 AI，三局兩勝。', detail: '使用原始遊戲的 FAI 正式賽版，三位玩家組成藍隊迎戰紅隊 AI，採三局兩勝。', players: 3, lineup: '紅隊 AI vs 藍隊 3 位玩家', tag: 'FAI 賽制版', badge: '三局兩勝', file: 'drone_soccer_competition_add_ai_realgame.html' },
    { id: 'official-six', category: 'official', title: '正式賽・紅藍對抗', description: '六人分隊，三局兩勝決定勝負。', detail: '使用原始遊戲的 FAI 正式賽版，紅、藍兩隊各三位玩家，採三局兩勝。', players: 6, lineup: '紅隊 3 位玩家 vs 藍隊 3 位玩家', tag: 'FAI 賽制版', badge: '三局兩勝', file: 'drone_soccer_competition_addadd_realgame.html' }
  ];
  const categories = {
    battle: { title: '自由對戰', description: '自由搭配真人與 AI，每隊合計 1–3 台。', icon: 'battle' },
    practice: { title: '飛行練習', description: '從第一次起飛開始，慢慢找到自己的飛行節奏。', icon: 'target' },
    racing: { title: '競速挑戰', description: '六種賽道，練習速度、路線與精準操控。', icon: 'racing' },
    official: { title: '正式賽', description: '共用隊伍編成，採三局兩勝；可使用不足三台的自訂陣容。', icon: 'trophy' }
  };
  const $ = id => document.getElementById(id);
  const icon = name => `<svg class="icon" aria-hidden="true"><use href="#i-${name}"/></svg>`;
  const storageKey = 'flight-club-preferences-v1';
  const state = { category: 'battle', controller: 'radio', selected: 'team-ai', roster: window.DroneMatchRoster.defaults() };
  const beginnerButton = $('beginner-button');
  let activeFrame = null;
  let launchTrigger = null;

  function isTeamMatch() { return state.category === 'battle' || state.category === 'official'; }
  function selectedMode() {
    const mode = modes.find(mode => mode.id === state.selected);
    if (!isTeamMatch()) return mode;
    const humans = state.roster.red.humans + state.roster.blue.humans;
    return { ...mode, title: state.category === 'official' ? '正式賽' : '自由編成對戰', players: humans,
      detail: state.category === 'official' ? '三局兩勝。得分後全隊退回己方半場，再次進攻。' : '每隊一位前鋒，其餘防守。真人優先擔任前鋒，穿越對方球門得分。',
      lineup: ['red','blue'].map(team => `${team === 'red' ? '紅隊' : '藍隊'} ${state.roster[team].humans} 真人＋${state.roster[team].ai} AI`).join(' ／ ') };
  }
  function controllerLabel() { return '搖桿控制'; }
  function visibleModes() { return modes.filter(mode => mode.category === state.category && (!isTeamMatch() || mode.id === (state.category === 'official' ? 'official-ai' : 'team-ai'))); }
  function hasAI(mode) { return isTeamMatch() ? state.roster.red.ai + state.roster.blue.ai > 0 : formations[mode?.id]?.some(member => member[1]) || false; }
  function gamePath(mode) {
    const params = new URLSearchParams();
    if (window.DroneFrameTiming) params.set('targetfps', window.DroneFrameTiming.preference);
    if (window.DroneCameraSettings) params.set('followpitch', window.DroneCameraSettings.followPitch ? '1' : '0');
    if (hasAI(mode)) params.set('difficulty', window.DroneAISettings.level);
    if (isTeamMatch()) window.DroneMatchRoster.write(params, state.roster);
    return `games/radio/${mode.file}${params.size ? '?' + params : ''}`;
  }

  const modeIcons = { attack: 'goal', defense: 'shield', 'attack-defense': 'battle', 'double-defense': 'shield', 'one-vs-two': 'battle', 'team-ai': 'chip', 'six-player': 'users', practice: 'route', rings: 'goal', maze: 'maze', grid: 'grid', tunnel: 'tunnel', slalom: 'route', spiral: 'racing', 'official-ai': 'trophy', 'official-six': 'trophy' };
  const formations = {
    attack: [['red', false, 'drone'], ['blue', true, 'shield'], ['blue', true, 'shield']],
    defense: [['red', true, 'drone'], ['blue', false, 'shield'], ['blue', true, 'shield']],
    'attack-defense': [['red', false, 'drone'], ['blue', false, 'shield'], ['blue', true, 'shield']],
    'double-defense': [['red', true, 'drone'], ['blue', false, 'shield'], ['blue', false, 'shield']],
    'one-vs-two': [['red', false, 'drone'], ['blue', false, 'shield'], ['blue', false, 'shield']],
    practice: [['red', false, 'drone'], ['blue', true, 'shield'], ['blue', true, 'shield']],
    'team-ai': [['red', true, 'drone'], ['red', true, 'shield'], ['red', true, 'shield'], ['blue', false, 'drone'], ['blue', false, 'shield'], ['blue', false, 'shield']],
    'six-player': [['red', false, 'drone'], ['red', false, 'shield'], ['red', false, 'shield'], ['blue', false, 'drone'], ['blue', false, 'shield'], ['blue', false, 'shield']]
  };
  formations['official-ai'] = formations['team-ai'];
  formations['official-six'] = formations['six-player'];

  const presets = {
    solo: { red: { humans: 0, ai: 1 }, blue: { humans: 1, ai: 0 } },
    coop: { red: { humans: 0, ai: 3 }, blue: { humans: 2, ai: 1 } },
    friends: { red: { humans: 3, ai: 0 }, blue: { humans: 3, ai: 0 } }
  };
  function renderTeams() {
    const definitions = window.DroneMatchRoster.definitions(state.roster);
    formations[state.selected] = definitions.map(def => [def.team, def.ai, def.role === 'attacker' ? 'drone' : 'shield']);
    $('mode-list').innerHTML = `<div class="team-builder">
      <div class="builder-intro"><span class="eyebrow">BUILD YOUR MATCH</span><h3>這場，誰來上場？</h3><p>兩隊分別調整，空位不會自動補 AI。</p></div>
      <div class="roster-presets" role="group" aria-label="快速套用隊伍"><span>快速開始</span>${[['solo','單人挑戰'],['coop','好友合作'],['friends','六人對戰']].map(([id,label]) => `<button type="button" data-preset="${id}" aria-pressed="${JSON.stringify(state.roster) === JSON.stringify(presets[id])}">${label}</button>`).join('')}</div>
      <div class="team-editors">${['red','blue'].map(team => {
        const row = state.roster[team], total = row.humans + row.ai, label = team === 'red' ? '紅隊' : '藍隊';
        return `<section class="team-editor ${team}" aria-labelledby="${team}-heading"><div class="team-editor-heading"><h3 id="${team}-heading"><i></i>${label}</h3><span>${total}<small> / 3 台</small></span></div>
          <div class="team-slots">${Array.from({length:3},(_,i) => `<div class="team-slot${i >= total ? ' is-empty' : ''}">${icon(i < row.humans ? 'users' : i < total ? 'chip' : 'drone')}<strong>${i < row.humans ? '真人' : i < total ? 'AI' : '空位'}</strong><small>${i < total ? (i === 0 ? '前鋒' : '防守') : '—'}</small></div>`).join('')}</div>
          ${[['humans','真人','users'],['ai','AI','chip']].map(([kind,name,symbol]) => `<div class="roster-counter"><span>${icon(symbol)}${name}</span><div class="roster-stepper" role="group" aria-label="${label}${name}數量"><button type="button" data-team="${team}" data-kind="${kind}" data-step="-1" aria-label="減少${label}${name}" ${row[kind] === 0 || total === 1 ? 'disabled' : ''}>−</button><output id="${team}-${kind}" aria-label="${label}${name}數量">${row[kind]}</output><button type="button" data-team="${team}" data-kind="${kind}" data-step="1" aria-label="增加${label}${name}" ${total === 3 ? 'disabled' : ''}>＋</button></div></div>`).join('')}
        </section>`;
      }).join('')}</div>
      <p class="roster-note">${icon('help')}每隊第 1 台為前鋒，其餘防守；真人優先擔任前鋒。真人皆為本機玩家。</p>
      <div class="roster-total" role="status">${icon('users')}<strong>${state.roster.red.humans + state.roster.blue.humans} 位真人</strong><span>＋ ${state.roster.red.ai + state.roster.blue.ai} 台 AI</span><span class="roster-match-size">${state.roster.red.humans + state.roster.red.ai} vs ${state.roster.blue.humans + state.roster.blue.ai}</span></div>
    </div>`;
  }
  function teamPreview(mode, roster = state.roster) {
    const definitions = window.DroneMatchRoster.definitions(roster);
    const project = (x, z, height = 0) => [600 + z * 132 + x * 78, 285 + x * 38 - z * 30 - height];
    const polygon = points => points.map(point => project(...point).join(',')).join(' ');
    const corners = [[-1.5,-3],[-1.5,3],[1.5,3],[1.5,-3]];
    const outline = polygon(corners);
    const stripes = Array.from({length: 8}, (_, i) => {
      const z = -3 + i * .75;
      return `<polygon points="${polygon([[-1.5,z],[-1.5,z+.75],[1.5,z+.75],[1.5,z]])}" fill="${i % 2 ? '#213c31' : '#1d352d'}"/>`;
    }).join('');
    const center = Array.from({length: 49}, (_, i) => {
      const a = i / 48 * Math.PI * 2;
      return project(Math.cos(a) * .64, Math.sin(a) * .64).join(',');
    }).join(' ');
    const zones = ['red','blue'].map(team => {
      const z = team === 'red' ? -3 : 2.2, color = team === 'red' ? '#ff8793' : '#87baff';
      return `<g data-launch-zone="${team}"><polygon points="${polygon([[-.5,z],[-.5,z+.8],[.5,z+.8],[.5,z]])}" fill="${color}" fill-opacity=".15" stroke="${color}" stroke-opacity=".6" stroke-width="2"/></g>`;
    }).join('');
    const posts = corners.map(point => {
      const p = project(...point), top = project(...point, 95);
      return `<path d="M${p}L${top}" stroke="#53636b" stroke-width="3"/>`;
    }).join('');
    const drones = definitions.map(def => {
      const start = window.DroneMatchRoster.launchPosition(definitions, def);
      const [x,y] = project(start.x, start.z, 16);
      const color = def.team === 'red' ? '#ff8793' : '#87baff';
      return `<g data-launch-drone="${def.id}" data-team="${def.team}" transform="translate(${x} ${y})"><ellipse cy="18" rx="18" ry="5" fill="#030809" opacity=".6"/><circle r="14" fill="#152229" stroke="${color}" stroke-width="2.5"/><ellipse rx="5" ry="14" fill="none" stroke="${color}" opacity=".65"/><ellipse rx="14" ry="5" fill="none" stroke="${color}" opacity=".65"/><path d="m-6-5 12 10m-12 0 12-10" stroke="${color}" stroke-width="2.5"/><text y="-25" text-anchor="middle" fill="${color}" font-size="18" font-weight="600">${def.ai ? 'AI' : 'P'}</text></g>`;
    }).join('');
    const goals = [-2.25,2.25].map((z,i) => {
      const [x,y] = project(0,z,48), color = i ? '#87baff' : '#ff8793';
      return `<path d="M${x},${y-47}V${y-91}" stroke="#53636b" stroke-width="2"/><ellipse cx="${x}" cy="${y}" rx="18" ry="44" transform="rotate(-12 ${x} ${y})" fill="none" stroke="#071111" stroke-width="11"/><ellipse cx="${x}" cy="${y}" rx="18" ry="44" transform="rotate(-12 ${x} ${y})" fill="none" stroke="${color}" stroke-width="6"/>`;
    }).join('');
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 500"><title>紅藍兩隊起飛區配置</title><rect width="1200" height="500" fill="#0f171c"/><polygon points="${outline}" transform="translate(0 12)" fill="#050b0e" stroke="#1f3038" stroke-width="14"/>${stripes}<polygon points="${outline}" fill="none" stroke="#83958a" stroke-width="2"/><path d="M${project(-1.5,0)}L${project(1.5,0)}" stroke="#83958a" stroke-width="2"/><polyline points="${center}" fill="none" stroke="#83958a" stroke-width="2"/>${zones}<polygon points="${corners.map(p=>project(...p,95).join(',')).join(' ')}" fill="none" stroke="#53636b" stroke-width="2" stroke-dasharray="5 6" opacity=".5"/>${posts}<g font-family="Segoe UI,sans-serif">${goals}${drones}<text x="190" y="467" fill="#ff8793" font-size="22" font-weight="600">RED TEAM</text><text x="1010" y="467" text-anchor="end" fill="#87baff" font-size="22" font-weight="600">BLUE TEAM</text><text x="600" y="467" text-anchor="middle" fill="#819795" font-size="18" letter-spacing="2">STARTING FORMATION</text></g></svg>`;
    $('arena-preview').src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg);
    $('arena-preview').alt = mode.lineup + '。全員停在己方起飛區，每隊第一台為前鋒，其餘防守。';
    $('arena-preview').dataset.previewMode = mode.id;
    $('preview-visual').dataset.category = mode.category;
    $('preview-category').textContent = '即時隊伍預覽';
    $('preview-note').hidden = false;
    $('preview-icon-use').setAttribute('href', '#i-battle');
    ['red','blue'].forEach((team,i) => { $(`preview-legend-${i ? 'right' : 'left'}`).textContent = `${team === 'red' ? '紅隊' : '藍隊'} · ${roster[team].humans} 真人＋${roster[team].ai} AI`; });
    $('preview-legend-center').textContent = 'VS';
  }

  function formationHTML(mode) {
    if (!mode) return '';
    const members = formations[mode.id];
    const member = ([team, ai, role]) => `<span class="formation-member ${team}${ai ? ' is-ai' : ''}" title="${ai ? 'AI' : '玩家'}${role === 'shield' ? '防守機' : '攻擊機'}">${icon(ai ? 'chip' : role)}</span>`;
    if (!members) return `<span class="formation-team">${member(['blue', false, 'drone'])}</span><span class="formation-vs">→</span><span class="formation-member">${icon(mode.category === 'practice' ? 'route' : modeIcons[mode.id])}</span>`;
    return `<span class="formation-team">${members.filter(item => item[0] === 'red').map(member).join('')}</span><span class="formation-vs">VS</span><span class="formation-team">${members.filter(item => item[0] === 'blue').map(member).join('')}</span>`;
  }

  function modeArt(mode) {
    if (formations[mode.id]) return matchupArt(mode);
    return flightArt(mode);
  }

  function flightArt(mode) {
    const id = `flight-${mode.id}`;
    const accents = { practice: '#329887', rings: '#cc8b36', maze: '#8771bc', grid: '#37967d', tunnel: '#448caf', slalom: '#38bdf8', spiral: '#a78bfa' };
    const accent = accents[mode.id];
    const ring = (x, y, rx, ry, color) => `<ellipse cx="${x + 3}" cy="${y + 2}" rx="${rx}" ry="${ry}" stroke="${color}" stroke-width="4" opacity=".2"/><ellipse cx="${x}" cy="${y}" rx="${rx}" ry="${ry}" stroke="${color}" stroke-width="3"/><path d="M${x - rx * .7} ${y - ry * .7}Q${x} ${y - ry * 1.3} ${x + rx * .7} ${y - ry * .7}" stroke="#fff" stroke-width="1.5" opacity=".9"/>`;
    const route = path => `<path d="${path}" stroke="${accent}" stroke-width="2.3" stroke-dasharray="4 5" stroke-linecap="round" marker-end="url(#${id}-arrow)"/>`;
    const scenes = {
      practice: () => `<g stroke="#a7c4d5" stroke-width="1.4"><ellipse cx="71" cy="118" rx="34" ry="10"/><ellipse cx="71" cy="118" rx="23" ry="6"/><path d="M65 118h12m-6-4v8M24 88V48m-4 5 4-5 4 5"/></g>${route('M88 80Q123 13 180 59T229 81')}${ring(182, 60, 23, 37, accent)}${flightDrone(id, 71, 71, 1.02)}`,
      rings: () => racingArt(mode, id),
      maze: () => mazeArt(id),
      grid: () => racingArt(mode, id),
      slalom: () => racingArt(mode, id),
      spiral: () => racingArt(mode, id),
      tunnel: () => racingArt(mode, id)
    };
    return `<svg class="mode-art flight-art flight-art-${mode.category}" viewBox="0 0 260 150" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" focusable="false">
      <defs><radialGradient id="${id}-sphere" cx=".3" cy=".25" r=".8"><stop stop-color="#fff"/><stop offset="1" stop-color="#c7dff5"/></radialGradient><marker id="${id}-arrow" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="m1 1 5 3-5 3" fill="none" stroke="${accent}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>
      <g stroke-linecap="round" stroke-linejoin="round">${scenes[mode.id]()}</g>
    </svg>`;
  }

  // Racing cards share the maze palette, route weight and drone scale.
  function racingArt(mode, id) {
    const colors = { frame: '#cbbced', rail: '#a995d5', gold: '#ffd176', mint: '#8be4ce', blue: '#9bc9ea' };
    const marker = `${id}-route-arrow`;
    const route = path => `<path d="${path}" stroke="#263342" stroke-width="6" opacity=".85"/><path d="${path}" stroke="${colors.gold}" stroke-width="2.8" marker-end="url(#${marker})"/>`;
    const start = (x, y) => `<circle cx="${x}" cy="${y}" r="3" fill="${colors.mint}" stroke="#213849" stroke-width="1.3"/>`;
    const gate = (x, y, rx, ry, color = colors.frame) => `<ellipse cx="${x + 3}" cy="${y + 2}" rx="${rx}" ry="${ry}" stroke="${colors.rail}" stroke-width="2" opacity=".35"/><ellipse cx="${x}" cy="${y}" rx="${rx}" ry="${ry}" fill="${color}" fill-opacity=".06" stroke="${color}" stroke-width="2.4"/><path d="M${x - rx * .7} ${y - ry * .7}Q${x} ${y - ry * 1.2} ${x + rx * .7} ${y - ry * .7}" stroke="#ede6ff" stroke-width="1" opacity=".75"/>`;
    const guides = points => `<g stroke="${colors.rail}" stroke-width="1" opacity=".38" stroke-dasharray="2 4">${points.map(([x, y, floor]) => `<path d="M${x} ${y}V${floor}"/>`).join('')}</g>`;
    const scenes = {
      rings: () => `${guides([[84, 50, 132], [145, 94, 132], [207, 42, 132]])}
        <path d="M62 132H225" stroke="${colors.rail}" opacity=".3"/>
        ${gate(84, 50, 17, 30)}${gate(145, 94, 17, 30, colors.blue)}${gate(207, 42, 14, 25)}
        ${route('M51 112C67 112 70 50 84 50S126 94 145 94 188 42 207 42H232')}${start(51, 112)}`,
      grid: () => `<g transform="translate(103 36) skewY(-8)">
        <path d="M0 0 7-5H119V91L112 96M112 0l7-5" stroke="${colors.rail}" stroke-width="1.3"/>
        <rect width="112" height="96" fill="${colors.rail}" fill-opacity=".08" stroke="${colors.frame}" stroke-width="2.4"/>
        <path d="M37.33 0V96M74.67 0V96M0 32H112M0 64H112" stroke="${colors.frame}" stroke-width="1.7"/>
        <rect x="37.33" y="32" width="37.34" height="32" fill="${colors.mint}" fill-opacity=".2" stroke="${colors.mint}" stroke-width="2.5"/>
        <path d="M-7 101H7M105 101H119" stroke="${colors.rail}" stroke-width="2"/>
        </g>${route('M51 112C91 116 125 96 159 76L169 70')}${start(51, 112)}`,
      slalom: () => {
        const centers = [[71, 105], [134, 89], [108, 73], [171, 57], [145, 41]];
        return `<path d="m51 129 77 12 75-87-77-12Z" fill="${colors.rail}" fill-opacity=".06" stroke="${colors.rail}" stroke-opacity=".25"/>
          <path d="M93 132 168 46" stroke="${colors.rail}" stroke-dasharray="3 5" opacity=".45"/>
          ${guides(centers.map(([x, y]) => [x, y + 18, y + 25]))}
          ${[...centers].reverse().map(([x, y]) => gate(x, y, 10, 18, colors.blue)).join('')}
          ${route('M51 120 71 105 134 89 108 73 171 57 145 41 190 27')}${start(51, 120)}`;
      },
      spiral: () => {
        const points = Array.from({ length: 91 }, (_, i) => {
          const t = i / 90, angle = Math.PI + 3 * Math.PI * t;
          return [145 + 65 * Math.cos(angle), 119 - 85 * t + 17 * Math.sin(angle)];
        });
        const path = points.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(2)} ${y.toFixed(2)}`).join('');
        return `<ellipse cx="145" cy="130" rx="66" ry="11" stroke="${colors.rail}" opacity=".25"/>
          <path d="M145 125V16m-4 5 4-5 4 5" stroke="${colors.rail}" stroke-dasharray="3 5" opacity=".6"/>
          ${[0, 18, 36, 54, 72, 90].map(i => gate(...points[i], 10, 16)).join('')}
          ${route(path)}${start(...points[0])}`;
      },
      tunnel: () => {
        // Draw only the cells along one bent passage, rather than a full maze.
        const cells = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 1, 0], [2, 1, 1], [2, 1, 2], [3, 1, 2]];
        const project = ([x, y, z]) => [65 + 27 * x + 17 * z, 110 + 7 * x - 11 * z - 25 * y];
        const point = p => project(p).join(' ');
        const edges = new Set();
        const addEdge = (a, b) => edges.add([a.join(','), b.join(',')].sort().join('|'));
        const faces = cells.map(([x, y, z]) => {
          for (let a = 0; a <= 1; a++) for (let b = 0; b <= 1; b++) {
            addEdge([x, y + a, z + b], [x + 1, y + a, z + b]);
            addEdge([x + a, y, z + b], [x + a, y + 1, z + b]);
            addEdge([x + a, y + b, z], [x + a, y + b, z + 1]);
          }
          return `<path d="M${point([x, y + 1, z])}L${point([x + 1, y + 1, z])}L${point([x + 1, y + 1, z + 1])}L${point([x, y + 1, z + 1])}Z" fill="${colors.blue}" fill-opacity=".1"/>`;
        }).join('');
        const lattice = [...edges].map(edge => { const [a, b] = edge.split('|').map(p => p.split(',').map(Number)); return `M${point(a)}L${point(b)}`; }).join('');
        const centers = [[-.35, .5, .5], ...cells.map(cell => cell.map(v => v + .5)), [4.35, 1.5, 2.5]];
        const path = centers.map((cell, i) => `${i ? 'L' : 'M'}${point(cell)}`).join('');
        return `${faces}<path d="${lattice}" stroke="${colors.frame}" stroke-width="1.4" opacity=".8"/>
          ${route(path)}${start(...project(centers[0]))}`;
      }
    };
    return `<defs><marker id="${marker}" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="3.5" markerHeight="3.5" orient="auto"><path d="m1 1 5 3-5 3" stroke="${colors.gold}" stroke-width="1.6" fill="none"/></marker></defs>
      <ellipse cx="132" cy="139" rx="86" ry="6" fill="#0b1026" opacity=".24"/>
      ${scenes[mode.id]()}${flightDrone(id, 30, 116, .59)}`;
  }

  // A three-level lattice keeps the card recognizable at small sizes.
  function mazeArt(id) {
    const project = ([x, y, z]) => [112 + 30 * x - 21 * z, 78 + 10 * x + 11 * z - 23 * y];
    const point = cell => project(cell).join(' ');
    const polygon = cells => cells.map(cell => project(cell).join(',')).join(' ');
    const line = (a, b) => `M${point(a)}L${point(b)}`;
    const panel = (cells, fill) => `<polygon points="${polygon(cells)}" fill="${fill}" fill-opacity=".8" stroke="#b5a0eb" stroke-width="1.2"/>`;
    const lattice = [];
    for (let a = 0; a <= 3; a++) for (let b = 0; b <= 3; b++) {
      lattice.push(line([0, a, b], [3, a, b]), line([a, 0, b], [a, 3, b]), line([a, b, 0], [a, b, 3]));
    }
    const corners = [[0, 0, 0], [3, 0, 0], [3, 0, 3], [0, 0, 3]];
    const frame = [...corners.map((c, i) => line(c, corners[(i + 1) % 4])),
      ...corners.map((c, i) => line([c[0], 3, c[2]], [corners[(i + 1) % 4][0], 3, corners[(i + 1) % 4][2]])),
      ...corners.map(c => line(c, [c[0], 3, c[2]]))].join('');
    // Show turns and vertical movement through open cells, away from the panels.
    const cells = [[.5, .5, 3.3], [.5, .5, 2.5], [.5, 1.5, 2.5],
      [1.5, 1.5, 2.5], [1.5, 1.5, 1.5], [1.5, 2.5, 1.5],
      [2.5, 2.5, 1.5], [2.5, 2.5, .5], [3.25, 2.5, .5]];
    const path = cells.map((cell, index) => `${index ? 'L' : 'M'}${point(cell)}`).join('');
    return `<defs><marker id="${id}-route-arrow" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="3.5" markerHeight="3.5" orient="auto"><path d="m1 1 5 3-5 3" stroke="#ffd176" stroke-width="1.6" fill="none"/></marker></defs>
      <ellipse cx="132" cy="139" rx="86" ry="6" fill="#0b1026" opacity=".24"/>
      <polygon points="${polygon(corners)}" fill="#7561a3" fill-opacity=".13"/>
      <polygon points="${polygon([[0, 0, 0], [3, 0, 0], [3, 3, 0], [0, 3, 0]])}" fill="#aa94de" fill-opacity=".12"/>
      <polygon points="${polygon([[0, 0, 0], [0, 0, 3], [0, 3, 3], [0, 3, 0]])}" fill="#c8b9ef" fill-opacity=".09"/>
      <path d="${lattice.join('')}" stroke="#b2a0d4" stroke-width=".9" opacity=".38"/>
      ${panel([[1, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0]], '#8070af')}
      ${panel([[0, 1, 1], [0, 1, 2], [0, 2, 2], [0, 2, 1]], '#baabe1')}
      ${panel([[1, 2, 0], [2, 2, 0], [2, 3, 0], [1, 3, 0]], '#a895d4')}
      ${panel([[2, 0, 2], [3, 0, 2], [3, 1, 2], [2, 1, 2]], '#8e7abb')}
      <path d="${frame}" stroke="#cbbced" stroke-width="1.6"/>
      <path d="${path}" stroke="#263342" stroke-width="6" opacity=".85"/>
      <path d="${path}" stroke="#ffd176" stroke-width="2.8" marker-end="url(#${id}-route-arrow)"/>
      <circle cx="${project(cells[0])[0]}" cy="${project(cells[0])[1]}" r="3" fill="#8be4ce" stroke="#213849" stroke-width="1.3"/>
      ${flightDrone(id, 30, 116, .59)}`;
  }

  function flightDrone(id, x, y, scale) {
    return `<g transform="translate(${x} ${y}) scale(${scale}) rotate(-15)"><circle r="25" fill="url(#${id}-sphere)" stroke="#477eae" stroke-width="1.3"/><circle r="21" fill="#f4f9ff"/><g fill="#7eadd5" stroke="#3d729f" stroke-width=".8"><ellipse cx="-9" cy="-7" rx="7" ry="3"/><ellipse cx="9" cy="-7" rx="7" ry="3"/><ellipse cx="-9" cy="7" rx="7" ry="3"/><ellipse cx="9" cy="7" rx="7" ry="3"/><path d="m-9-7 18 14m-18 0L9-7"/></g><rect x="-5" y="-4" width="10" height="9" rx="2" fill="#407eb7"/><g stroke="#4e84b4" stroke-width="1.6"><circle r="24"/><ellipse rx="11" ry="24"/><ellipse rx="24" ry="9"/><path d="M-24 0h48M-19-15q19 9 38 0M-19 15q19-9 38 0"/></g><path d="M-17-17a24 24 0 0 1 27-4" stroke="#fff" stroke-width="2.5" stroke-linecap="round"/></g>`;
  }

  // Keep each illustration tied to the same roster used by the match details.
  function matchupArt(mode) {
    const id = `match-${mode.id}`;
    const official = mode.category === 'official';
    const members = formations[mode.id];
    const label = team => {
      const roster = members.filter(member => member[0] === team);
      const ai = roster.filter(member => member[1]).length;
      const players = roster.length - ai;
      return ai && players ? `玩家 ${players}＋AI ${ai}` : ai ? `${ai} AI` : `${players} 玩家`;
    };
    const ball = (team, x, y, scale, ai, angle) => {
      const color = team === 'red' ? '#d34261' : '#286fc3';
      return `<g transform="translate(${x} ${y}) scale(${scale}) rotate(${angle})">
        <circle r="29" fill="url(#${id}-${team})" stroke="${color}" stroke-width="1.8"/>
        <circle r="25" fill="${team === 'red' ? '#fff5f6' : '#f1f7ff'}"/>
        <g fill="${color}" fill-opacity=".65" stroke="${color}" stroke-width=".7">
          <ellipse cx="-11" cy="-8" rx="9" ry="3.5"/><ellipse cx="11" cy="-8" rx="9" ry="3.5"/>
          <ellipse cx="-11" cy="9" rx="9" ry="3.5"/><ellipse cx="11" cy="9" rx="9" ry="3.5"/>
        </g>
        <path d="m-11-8 22 17m-22 0 22-17" stroke="${color}" stroke-width="2"/>
        <rect x="-6" y="-5" width="12" height="12" rx="3" fill="${color}" stroke="#fff" stroke-width="1"/>
        <g fill="none" stroke="${color}" stroke-width="1.8">
          <ellipse rx="12" ry="28"/><ellipse rx="23" ry="28"/>
          <ellipse rx="28" ry="11"/><path d="M-28 0h56M-23-16q23 10 46 0M-23 16q23-10 46 0"/>
        </g>
        <path d="M-20-20a28 28 0 0 1 32-5" fill="none" stroke="#fff" stroke-width="2.5" stroke-linecap="round" opacity=".9"/>
        <circle r="28" fill="none" stroke="${color}" stroke-width="2"/>
        ${ai ? `<rect x="-12" y="20" width="24" height="15" rx="4" fill="#fff" stroke="${color}" stroke-width="1.2"/><text y="31" fill="${color}" text-anchor="middle" font-size="10" font-weight="800" font-family="Segoe UI,sans-serif">AI</text>` : ''}
      </g>`;
    };
    const teamArt = team => {
      const roster = members.filter(member => member[0] === team);
      const left = team === 'red';
      const positions = left ? [[48, 73, 1.13], [23, 29, .54], [21, 108, .48]] : [[212, 73, 1.13], [237, 29, .54], [239, 108, .48]];
      // Draw the smaller team members behind the captain.
      return roster.map((member, index) => ({ member, index })).reverse().map(({ member, index }) => ball(team, ...positions[index], member[1], left ? -23 : 23)).join('');
    };
    return `<svg class="mode-art matchup-art${official ? ' championship-art' : ''}" viewBox="0 0 260 150" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" focusable="false">
      <defs>
        <radialGradient id="${id}-red" cx=".3" cy=".25" r=".8"><stop stop-color="#fff"/><stop offset="1" stop-color="#efd0d8"/></radialGradient>
        <radialGradient id="${id}-blue" cx=".3" cy=".25" r=".8"><stop stop-color="#fff"/><stop offset="1" stop-color="#c7dcf3"/></radialGradient>
        <linearGradient id="${id}-vs"><stop stop-color="#b52547"/><stop offset=".46" stop-color="#183d65"/><stop offset="1" stop-color="#2567b6"/></linearGradient>
      </defs>
      <g stroke="#d34261" stroke-width="2.5" stroke-linecap="round" opacity=".65"><path d="m2 58 33 8M4 80l29-3m-20 25 22-11"/></g>
      <g stroke="#286fc3" stroke-width="2.5" stroke-linecap="round" opacity=".65"><path d="m225 66 33-8m-31 19 29 3m-31 11 22 11"/></g>
      <path d="m142 29-25 30 16-2-17 37 14-4-8 29 24-36-15 3 18-35-17 4Z" fill="#efd18b" opacity=".75"/>
      <g class="matchup-red">${teamArt('red')}</g><g class="matchup-blue">${teamArt('blue')}</g>
      <text x="129" y="95" fill="url(#${id}-vs)" stroke="#fff" stroke-width="4" paint-order="stroke fill" text-anchor="middle" font-family="Arial Black,Segoe UI,Arial,sans-serif" font-style="italic" font-size="62" font-weight="900" letter-spacing="-5">VS</text>
      <path d="m91 108 29-5m15-1 29-5" stroke="${official ? '#bb8730' : '#6685a8'}" stroke-width="3" stroke-linecap="round"/>
      <g font-family="Segoe UI,Microsoft JhengHei,Arial,sans-serif" text-anchor="middle" font-size="15" font-weight="800">
        <text x="50" y="142" fill="#bd294c">${label('red')}</text><text x="210" y="142" fill="#2365b2">${label('blue')}</text>
      </g>
      ${official ? `<text x="130" y="19" fill="#96702d" font-family="Segoe UI,Arial,sans-serif" text-anchor="middle" font-size="13" font-weight="800" letter-spacing=".7">BEST OF 3</text><g transform="translate(130 132) scale(1.1)" stroke="#b78a3b" stroke-width="1.3" fill="#f5ddb0"><path d="M-6-7H6V0a6 6 0 0 1-12 0Z"/><path d="M-6-5h-4v3l5 4M6-5h4v3L5 2M0 6v5m-5 0H5" fill="none"/><path d="M-13 7q-7-8-1-15m27 15q7-8 1-15m-28 7-4-2m3 7-4-1m31-4 4-2m-3 7 4-1" fill="none"/></g>` : ''}
    </svg>`;
  }

  function renderBrief() {
    const brief = $('category-brief');
    brief.hidden = false;
    brief.dataset.category = state.category;
    const copy = {
      battle: { title: '攻守之間，找到你的節奏。', note: '看準目標，和隊友一起完成每次進攻。', steps: [['goal', '進攻', '對準環門，穩定穿越。'], ['shield', '防守', '守住位置，預判來球。'], ['users', '協作', '分配角色，支援隊友。']] },
      practice: { title: '從穩定起飛，到精準穿越。', note: '不急著搶快，先建立自己的操控節奏。', steps: [['drone', '起飛與定高', '熟悉油門，控制飛行高度。'], ['route', '轉向與移動', '練習偏航、前後與左右側移。'], ['goal', '穿越與降落', '對準目標，平穩完成航線。']] },
      racing: { title: '六種賽道，練出不同的飛行能力。', note: '先熟悉路線，再挑戰自己的完成時間。', steps: [['goal', '高度與節奏', '高低環與螺旋爬升：調整高度與節奏。'], ['maze', '空間與路線', '立體迷宮與蛇形繞標：掌握轉向路線。'], ['target', '精度與穩定', '九宮格與隧道：保持準確航線。']] },
      official: { title: '三局兩勝，每一局都算數。', note: '每局 3 分鐘；局間休息可在進場時設定。', steps: [['trophy', '第 1 局', '建立節奏，掌握對手。'], ['trophy', '第 2 局', '調整戰術，爭取勝局。'], ['trophy', '第 3 局', '需要時進行決勝局。']] }
    }[state.category];
    const route = state.category === 'official'
      ? '<path d="M58 40h304" stroke="#d1bd91" stroke-width="1.8" stroke-dasharray="4 5"/><g fill="#fff8e9" stroke="#c6a86a" stroke-width="2"><circle cx="58" cy="40" r="24"/><circle cx="210" cy="40" r="24"/><circle cx="362" cy="40" r="24"/></g><g fill="#9c7937" font-family="Segoe UI,sans-serif" font-size="23" font-weight="800" text-anchor="middle"><text x="58" y="48">1</text><text x="210" y="48">2</text><text x="362" y="48">3</text></g><path d="m91 35 5 5-5 5m152-10 5 5-5 5" stroke="#ae8c48" stroke-width="1.8"/>'
      : `<defs><radialGradient id="brief-sphere" cx=".3" cy=".25" r=".8"><stop stop-color="#fff"/><stop offset="1" stop-color="#c7dff5"/></radialGradient></defs><path d="M25 61q47-68 101-27t94 0 96-5 80-6" stroke="#629f9a" stroke-width="2" stroke-dasharray="4 5"/><g stroke-width="2.5"><ellipse cx="194" cy="39" rx="15" ry="25" stroke="#548fba"/><ellipse cx="344" cy="36" rx="14" ry="22" stroke="#c49a50"/></g>${flightDrone('brief', 48, 42, .63)}<circle cx="397" cy="23" r="4" fill="#499a86"/>`;
    const summary = { battle: '攻守協作', practice: '穩定起飛', racing: '精準穿越', official: '三局兩勝' }[state.category];
    brief.innerHTML = `<div class="brief-heading"><div class="brief-label"><span class="brief-category-icon">${icon(categories[state.category].icon)}</span><div><p class="eyebrow">玩法重點</p><h3>${summary}</h3></div></div><button type="button" class="brief-open" aria-haspopup="dialog" aria-controls="brief-dialog" aria-label="查看${categories[state.category].title}完整玩法重點">完整說明${icon('arrow')}</button></div><div class="brief-steps">${copy.steps.map(([, title, text], index) => `<div><span class="brief-step-number" aria-hidden="true">0${index + 1}</span><div><strong>${title}</strong><p>${text}</p></div></div>`).join('')}</div>`;
    $('brief-dialog-title').textContent = `${categories[state.category].title}・玩法重點`;
    $('brief-dialog-content').innerHTML = `<p class="brief-dialog-lead">${copy.title}</p><p class="brief-dialog-note">${copy.note}</p><svg class="brief-dialog-route" viewBox="0 0 420 82" fill="none" aria-hidden="true">${route}</svg><ol class="brief-dialog-steps">${copy.steps.map(([symbol, title, text]) => `<li>${icon(symbol)}<div><h3>${title}</h3><p>${text}</p></div></li>`).join('')}</ol>`;
  }

  function restoreState() {
    const params = new URLSearchParams(location.search);
    if (window.DroneAISettings.valid(params.get('difficulty'))) window.DroneAISettings.setLevel(params.get('difficulty'));
    let saved = {};
    try { saved = JSON.parse(localStorage.getItem(storageKey) || '{}') || {}; } catch (_) { /* Storage may be disabled. */ }
    // A shared URL takes precedence over this browser's remembered choices.
    const source = params.has('mode') || params.has('category') || params.has('controller') ? Object.fromEntries(params) : saved;
    // Migrate remembered preferences and old keyboard links to joystick play.
    state.controller = 'radio';
    state.category = Object.hasOwn(categories, source.category) ? source.category : 'battle';
    const mode = modes.find(item => item.id === (source.mode || source.selected));
    if (mode && (!source.category || mode.category === state.category)) {
      state.selected = mode.id;
      state.category = mode.category;
    } else {
      state.selected = isTeamMatch() ? (state.category === 'official' ? 'official-ai' : 'team-ai') : modes.find(item => item.category === state.category).id;
    }
    const fromURL = window.DroneMatchRoster.read(params);
    if (fromURL) state.roster = fromURL;
    else if (window.DroneMatchRoster.valid(source.roster)) state.roster = source.roster;
    else if (!params.has('mode') && window.DroneMatchRoster.valid(saved.roster)) state.roster = saved.roster;
    else if (params.has('mode') && formations[state.selected]) {
      state.roster = Object.fromEntries(['red','blue'].map(team => [team, {
        humans: formations[state.selected].filter(m => m[0] === team && !m[1]).length,
        ai: formations[state.selected].filter(m => m[0] === team && m[1]).length
      }]));
    }
    if (isTeamMatch()) state.selected = state.category === 'official' ? 'official-ai' : 'team-ai';
    const visible = visibleModes();
    if (!visible.some(item => item.id === state.selected)) state.selected = visible[0]?.id || null;
  }

  function stateURL(play = false) {
    const url = new URL(location.href);
    url.search = '';
    url.searchParams.set('screen', play ? 'game' : window.FlightEntry?.screen === 'home' ? 'home' : 'setup');
    url.searchParams.set('category', state.category);
    url.searchParams.set('controller', state.controller);
    if (state.selected) url.searchParams.set('mode', state.selected);
    if (hasAI(selectedMode())) url.searchParams.set('difficulty', window.DroneAISettings.level);
    if (isTeamMatch()) window.DroneMatchRoster.write(url.searchParams, state.roster);
    if (play) url.searchParams.set('play', '1');
    return url;
  }

  function remember() {
    try { localStorage.setItem(storageKey, JSON.stringify(state)); } catch (_) { /* Keep working without storage. */ }
    // file:// browser policies can reject History API updates; the lobby still works.
    try { history.replaceState(null, '', stateURL()); } catch (_) { /* No URL persistence available. */ }
  }

  function renderPreview(mode) {
    if (isTeamMatch()) { teamPreview(mode); return; }
    $('preview-note').hidden = true;
    const preview = window.FLIGHT_CLUB_PREVIEWS[mode.id];
    const artwork = $('arena-preview');
    if (artwork.getAttribute('src') !== preview.src) artwork.setAttribute('src', preview.src);
    artwork.alt = preview.alt;
    artwork.dataset.previewMode = mode.id;
    $('preview-visual').dataset.category = mode.category;
    $('preview-category').textContent = mode.title;
    $('preview-icon-use').setAttribute('href', `#i-${modeIcons[mode.id]}`);
    ['left', 'center', 'right'].forEach((position, index) => {
      $(`preview-legend-${position}`).textContent = preview.legend[index];
    });
  }

  function render({ announce = true, persist = true } = {}) {
    const category = categories[state.category];
    const visible = visibleModes();
    if (!visible.some(mode => mode.id === state.selected)) state.selected = visible[0]?.id || null;
    const mode = selectedMode();
    document.querySelector('.mode-section').classList.toggle('short-list', visible.length <= 4);
    document.querySelector('.mode-section').dataset.category = state.category;
    document.querySelectorAll('[data-category]').forEach(button => {
      if (!button.matches('button')) return;
      const active = button.dataset.category === state.category;
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-pressed', String(active));
    });
    document.querySelectorAll('[data-controller]').forEach(button => {
      const active = button.dataset.controller === state.controller;
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-pressed', String(active));
    });
    $('category-title').textContent = category.title;
    $('category-description').textContent = category.description;
    $('mode-count').textContent = isTeamMatch() ? '每隊 1–3 台' : `${visible.length} 種模式`;
    $('mode-list').classList.toggle('is-team-builder', isTeamMatch());
    $('mode-list').innerHTML = visible.map(item => `<button type="button" class="mode-card${item.id === state.selected ? ' is-selected' : ''}" data-mode="${item.id}" aria-label="${item.title}，${item.players} 人，${item.description}" aria-pressed="${item.id === state.selected}"><div class="card-copy"><span class="card-top"><span class="card-icon">${icon(modeIcons[item.id])}</span><span class="card-tag">${item.tag}</span></span><h3>${item.title.replace(/^正式賽・/, '<span class="card-title-prefix">正式賽・</span>')}</h3><p class="card-description">${item.description}</p><span class="card-bottom">${icon('users')}<span>${item.players} 人</span><span class="card-badge">${item.badge}</span></span></div>${modeArt(item)}<span class="card-check">${icon('check')}</span></button>`).join('');
    if (visible.length === 7) {
      const tip = document.createElement('div');
      tip.className = 'catalog-tip';
      tip.innerHTML = `<div class="tip-copy"><p class="eyebrow">FIRST FLIGHT</p><h3>第一次飛？從這裡開始。</h3><p>先練習起飛與轉向，再上場挑戰。</p></div>${modeArt(modes.find(item => item.id === 'practice'))}`;
      tip.querySelector('.tip-copy').append(beginnerButton);
      $('mode-list').append(tip);
    } else $('brief-dialog-actions').prepend(beginnerButton);
    if (isTeamMatch()) renderTeams();
    renderBrief();
    if (mode) renderPreview(mode);
    $('selected-title').textContent = mode?.title || '選擇適合的玩法';
    $('selected-description').textContent = mode?.detail || '請先選擇要遊玩的模式。';
    $('selected-players').textContent = mode ? (mode.players ? `${mode.players} 位真人` : 'AI 觀戰') : '—';
    $('selected-controller').textContent = controllerLabel();
    $('ai-difficulty-setting').hidden = !hasAI(mode);
    $('ai-difficulty').value = window.DroneAISettings.level;
    $('ai-difficulty-description').textContent = window.DroneAISettings.levels[window.DroneAISettings.level].description;
    $('selected-lineup').textContent = mode?.lineup || '—';
    $('formation').innerHTML = formationHTML(mode);
    document.querySelector('.formation-legend').hidden = !mode || !formations[mode.id]?.some(member => member[1]);
    document.querySelector('.ready-badge').hidden = !mode;
    $('mobile-title').textContent = mode?.title || '請選擇模式';
    $('mobile-controller').textContent = mode ? `${controllerLabel()} · ${mode.players ? mode.players + ' 位真人' : 'AI 觀戰'}` : controllerLabel();
    $('launch-button').disabled = !mode;
    $('lan-official-button').hidden = state.category !== 'official';
    $('launch-note').textContent = isTeamMatch() && mode.players === 0 ? '全 AI 對戰：進場以 X 確認開始，可自由切換視角。' : '接上搖桿，進場確認左右雙桿；X 確認、Y 取消、BACK 返回。';
    if (announce) $('selection-status').textContent = mode ? `已選擇${mode.title}，${controllerLabel()}，${mode.players} 人。` : '請選擇遊戲模式。';
    if (persist) remember();
    window.dispatchEvent(new CustomEvent('flight-lobby-change', { detail: { category: state.category, controller: state.controller, title: mode?.title, players: mode?.players } }));
  }

  function setLobbyScreen(screen, { push = false, persist = true } = {}) {
    if (window.FlightEntry) window.FlightEntry.screen = screen;
    document.documentElement.dataset.screen = screen;
    $('home-screen').hidden = screen !== 'home';
    $('lobby').hidden = screen !== 'setup';
    window.dispatchEvent(new CustomEvent('flight-home-visibility', { detail: { active: screen === 'home' } }));
    window.dispatchEvent(new CustomEvent('flight-lobby-view', { detail: { screen } }));
    if (push) { try { history.pushState({ entryScreen: screen }, '', stateURL()); } catch (_) {} }
    else if (persist) remember();
  }

  function chooseCategory(category) {
    state.category = category;
    state.selected = isTeamMatch() ? (category === 'official' ? 'official-ai' : 'team-ai') : modes.find(mode => mode.category === category).id;
    render();
  }

  let stopGamepadRelay = null;
  let pendingFrameSettings = null;
  function syncFrameSettings() {
    activeFrame?.contentWindow?.postMessage({ type:'flight-settings-apply', targetfps:window.DroneFrameTiming?.preference, followpitch:window.DroneCameraSettings?.followPitch }, location.protocol === 'file:' ? '*' : location.origin);
  }
  function requestFrameSettings() {
    syncFrameSettings();
    if (pendingFrameSettings) activeFrame?.contentWindow?.postMessage({type:'flight-open-settings',kind:pendingFrameSettings}, location.protocol === 'file:' ? '*' : location.origin);
  }
  window.addEventListener('drone-frame-settings-change',syncFrameSettings);
  window.addEventListener('drone-camera-settings-change',syncFrameSettings);
  window.addEventListener('message', event => {
    if (event.source !== activeFrame?.contentWindow || (location.protocol === 'file:' ? event.origin !== 'null' : event.origin !== location.origin)) return;
    const data = event.data;
    if (!data || typeof data !== 'object') return;
    if (data.type === 'flight-club:gamepad-menu') return menu();
    if (data.type === 'flight-club:gamepad-back') return back();
    if (data.type === 'flight-settings-ready') requestFrameSettings();
    if (data.type === 'flight-settings-opened') { pendingFrameSettings = null; $('game-settings-hint').hidden = true; }
    if (data.type === 'flight-setting-change') {
      if (data.key === 'targetfps' && typeof data.value === 'string' && data.value !== window.DroneFrameTiming?.preference) window.DroneFrameTiming?.setTarget(data.value);
      if (data.key === 'followpitch' && typeof data.value === 'boolean' && data.value !== window.DroneCameraSettings?.followPitch) window.DroneCameraSettings?.setFollowPitch(data.value);
    }
  });
  function showGame({ pushHistory = true, openSettings = null } = {}) {
    const mode = selectedMode();
    if (!mode) return;
    pendingFrameSettings = openSettings;
    setLobbyScreen('game', { persist: false });
    launchTrigger = document.activeElement;
    const path = gamePath(mode);
    $('game-title').textContent = mode.title;
    $('game-controller').textContent = controllerLabel();
    $('game-loading').hidden = false;
    const frame = document.createElement('iframe');
    frame.title = `${mode.title} — ${controllerLabel()}`;
    frame.setAttribute('allow', 'gamepad *; fullscreen');
    frame.allowFullscreen = true;
    frame.addEventListener('load', () => {
      if (activeFrame !== frame) return;
      $('game-loading').hidden = true;

      // Keep the embedded game's opening menu focused.
      frame.focus();
      requestFrameSettings();
    });
    stopGamepadRelay?.();
    stopGamepadRelay = null;
    activeFrame?.remove();
    activeFrame = frame;
    // Set the destination before insertion so about:blank cannot fire a load
    // event that dismisses the loading cover before the game UI is ready.
    frame.src = path;
    $('game-host').append(frame);
    $('game-settings-hint').hidden = !openSettings;
    $('game-settings-hint').textContent = openSettings === 'pad' ? '正在開啟遙控器設定…上方可返回遊戲設定。' : '正在開啟按鍵與檔位設定…上方可返回遊戲設定。';
    stopGamepadRelay = window.DroneUSBInput.relayTo(frame);
    $('lobby').hidden = true;
    $('game-shell').hidden = false;
    document.body.classList.add('is-playing');
    document.title = `${mode.title}｜FLIGHT CLUB`;
    if (pushHistory) {
      try { history.pushState({ lobbyGame: true }, '', stateURL(true)); } catch (_) { /* file:// fallback */ }
    }
  }

  function hideGame() {
    try {
      const level = activeFrame?.contentWindow.DroneAISettings?.level;
      if (level) window.DroneAISettings.setLevel(level);
    } catch (_) { /* file:// browsers may isolate the frame. */ }
    // Remove the iframe, stopping its render loop, audio and controller polling.
    stopGamepadRelay?.();
    stopGamepadRelay = null;
    activeFrame?.remove();
    activeFrame = null;
    pendingFrameSettings = null;
    $('game-settings-hint').hidden = true;
    $('game-shell').hidden = true;
    setLobbyScreen('setup', { persist: false });
    document.body.classList.remove('is-playing');
    document.title = 'FLIGHT CLUB｜無人機足球';
    if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
    if (launchTrigger?.isConnected) launchTrigger.focus({ preventScroll: true });
  }

  function menu() {
    window.dispatchEvent(new CustomEvent('flight-lobby-menu'));
    document.querySelectorAll('dialog[open]').forEach(dialog => dialog.close());
    if (activeFrame) hideGame();
    setLobbyScreen('home');
    $('home-play').focus({ preventScroll: true });
  }

  function back() {
    if (activeFrame) { hideGame(); remember(); }
    else if (document.documentElement.dataset.screen === 'setup') {
      setLobbyScreen('home');
      $('home-play').focus({ preventScroll: true });
    }
  }

  window.addEventListener('drone-gamepad-menu', event => { event.preventDefault(); menu(); });
  window.addEventListener('drone-gamepad-back', event => { event.preventDefault(); back(); });

  $('category-nav').addEventListener('click', event => {
    const button = event.target.closest('button[data-category]');
    if (button) { chooseCategory(button.dataset.category); setLobbyView('modes'); }
  });
  function setLobbyView(view) {
    document.querySelector('.selection-layout').dataset.view = view;
    document.querySelectorAll('[data-lobby-view]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.lobbyView === view)));
  }
  $('mobile-view-tabs').addEventListener('click', event => {
    const button = event.target.closest('[data-lobby-view]');
    if (button) setLobbyView(button.dataset.lobbyView);
  });
  $('beginner-button').addEventListener('click', () => {
    $('brief-dialog').close();
    chooseCategory('practice');
    setLobbyView('modes');
    $('launch-button').focus({ preventScroll: true });
  });
  $('controller-toggle').addEventListener('click', event => {
    const button = event.target.closest('button[data-controller]');
    if (!button) return;
    state.controller = 'radio';
    render();
  });
  $('mode-list').addEventListener('click', event => {
    const preset = event.target.closest('[data-preset]');
    const stepper = event.target.closest('[data-step]');
    if (preset || stepper) {
      let focusSelector;
      if (preset) {
        state.roster = JSON.parse(JSON.stringify(presets[preset.dataset.preset]));
        focusSelector = `[data-preset="${preset.dataset.preset}"]`;
      } else {
        const { team, kind, step } = stepper.dataset;
        const next = JSON.parse(JSON.stringify(state.roster));
        next[team][kind] += Number(step);
        if (!window.DroneMatchRoster.valid(next)) return;
        state.roster = next;
        focusSelector = `[data-team="${team}"][data-kind="${kind}"][data-step="${step}"]`;
      }
      render();
      const focus = document.querySelector(focusSelector);
      if (focus && !focus.disabled) focus.focus({ preventScroll: true });
      else if (stepper) document.getElementById(`${stepper.dataset.team}-${stepper.dataset.kind}`).closest('.roster-stepper').querySelector('button:not(:disabled)')?.focus({ preventScroll: true });
      $('selection-status').textContent = selectedMode().lineup;
      return;
    }

    const button = event.target.closest('button[data-mode]');
    if (!button) return;
    state.selected = button.dataset.mode;
    render();
    // Rendering replaces the selected card. Restore its focus for navigation.
    document.querySelector(`[data-mode="${state.selected}"]`).focus({ preventScroll: true });
  });
  $('launch-button').addEventListener('click', () => showGame());
  $('lan-official-button').addEventListener('click', async () => {
    const query = new URLSearchParams({ difficulty: window.DroneAISettings.level });
    window.DroneMatchRoster.write(query, state.roster);
    if (location.protocol !== 'file:') {
      try {
        const response = await fetch('/api/config');
        if (response.ok && response.headers.get('content-type')?.includes('application/json')) {
          const config = await response.json();
          if (config.secure && new URL(config.origin).origin === location.origin) {
            location.href = '/admin?create=1&' + query;
            return;
          }
        }
      } catch (_) { /* The ordinary static server has no LAN room service. */ }
    }
    location.href = 'lan/setup.html';
  });
  $('ai-difficulty').addEventListener('change', event => {
    window.DroneAISettings.setLevel(event.target.value);
    render();
  });
  window.addEventListener('drone-ai-difficulty-change', () => {
    $('ai-difficulty').value = window.DroneAISettings.level;
    $('ai-difficulty-description').textContent = window.DroneAISettings.levels[window.DroneAISettings.level].description;
  });
  $('back-button').addEventListener('click', () => {
    if (history.state?.lobbyGame) history.back();
    else { hideGame(); remember(); }
  });
  $('fullscreen-button').addEventListener('click', async () => {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else await $('game-shell').requestFullscreen();
    } catch (_) { $('fullscreen-button').title = '此瀏覽器不支援全螢幕'; }
  });
  document.addEventListener('fullscreenchange', () => {
    const button = $('fullscreen-button');
    const active = Boolean(document.fullscreenElement);
    button.title = active ? '離開全螢幕' : '全螢幕';
    button.querySelector('span').textContent = button.title;
    button.setAttribute('aria-pressed', String(active));
  });
  const previewDialog = $('preview-dialog');
  $('preview-expand').addEventListener('click', () => {
    const preview = $('preview-visual').cloneNode(true);
    preview.removeAttribute('id');
    preview.querySelector('.arena-caption').remove();
    preview.querySelectorAll('[id]').forEach(element => element.removeAttribute('id'));
    $('preview-dialog-title').textContent = $('preview-category').textContent;
    $('preview-dialog-content').replaceChildren(preview);
    previewDialog.showModal();
  });
  $('close-preview').addEventListener('click', () => previewDialog.close());
  previewDialog.addEventListener('click', event => {
    if (event.target !== previewDialog) return;
    const bounds = previewDialog.getBoundingClientRect();
    if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) previewDialog.close();
  });
  const helpDialog = $('help-dialog');
  const briefDialog = $('brief-dialog');
  $('category-brief').addEventListener('click', event => {
    if (event.target.closest('.brief-open')) briefDialog.showModal();
  });
  $('close-brief').addEventListener('click', () => briefDialog.close());
  $('brief-done').addEventListener('click', () => briefDialog.close());
  briefDialog.addEventListener('click', event => {
    if (event.target !== briefDialog) return;
    const bounds = briefDialog.getBoundingClientRect();
    if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) briefDialog.close();
  });
  $('help-button').addEventListener('click', () => helpDialog.showModal());
  $('close-help').addEventListener('click', () => helpDialog.close());
  $('got-it').addEventListener('click', () => helpDialog.close());
  helpDialog.addEventListener('click', event => {
    if (event.target !== helpDialog) return;
    const bounds = helpDialog.getBoundingClientRect();
    if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) helpDialog.close();
  });
  window.addEventListener('popstate', () => {
    const targetScreen = new URLSearchParams(location.search).get('screen') || 'setup';
    const wasPlaying = Boolean(activeFrame);
    hideGame();
    const playedLevel = window.DroneAISettings.level;
    restoreState();
    if (wasPlaying) window.DroneAISettings.setLevel(playedLevel);
    render({ announce: false, persist: false });
    if (targetScreen === 'home') setLobbyScreen('home', { persist: false });
    if (new URLSearchParams(location.search).get('play') === '1') showGame({ pushHistory: false });
  });

  const startInGame = new URLSearchParams(location.search).get('play') === '1';

  restoreState();
  render({ announce: false, persist: !startInGame });
  window.FlightLobby = Object.freeze({
    configure(category) { if (Object.hasOwn(categories, category)) chooseCategory(category); setLobbyScreen('setup', { push: true }); },
    home() { setLobbyScreen('home'); },
    menu,
    back,
    setController(controller) { if (activeFrame || controller !== 'radio') return false; state.controller='radio'; render(); return true; },
    openSettings(kind) { if (!['controls','pad'].includes(kind)) return false; if (activeFrame) { pendingFrameSettings=kind;requestFrameSettings();return true; } showGame({openSettings:kind});return true; },
    snapshot() { return { category: state.category, controller: state.controller, title: selectedMode()?.title, players:selectedMode()?.players, playing:!!activeFrame }; }
  });
  if (!startInGame) setLobbyScreen(window.FlightEntry?.screen === 'home' ? 'home' : 'setup', { persist: false });
  if (startInGame) showGame({ pushHistory: false });
})();
