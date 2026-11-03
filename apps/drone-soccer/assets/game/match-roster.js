/* Team sizes and input assignments shared by the lobby and match engines. */
(() => {
  'use strict';
  const teams = ['red', 'blue'];
  const defaults = { red: { humans: 0, ai: 3 }, blue: { humans: 1, ai: 2 } };
  const clone = value => JSON.parse(JSON.stringify(value));
  function valid(roster) {
    return !!roster && teams.every(team => {
      const row = roster[team];
      return row && Number.isInteger(row.humans) && Number.isInteger(row.ai) &&
        row.humans >= 0 && row.ai >= 0 && row.humans + row.ai >= 1 && row.humans + row.ai <= 3;
    });
  }
  function read(params) {
    const keys = ['redHumans', 'redAI', 'blueHumans', 'blueAI'];
    if (!keys.some(key => params.has(key))) return null;
    if (!keys.every(key => /^[0-3]$/.test(params.get(key) || ''))) return clone(defaults);
    const roster = Object.fromEntries(teams.map(team => [team, {
      humans: Number(params.get(team + 'Humans')), ai: Number(params.get(team + 'AI'))
    }]));
    return valid(roster) ? roster : clone(defaults);
  }
  function write(params, roster) {
    if (!valid(roster)) throw new Error('每隊必須配置 1–3 台無人機');
    teams.forEach(team => {
      params.set(team + 'Humans', roster[team].humans);
      params.set(team + 'AI', roster[team].ai);
    });
  }
  const controls = [
    { id:'r-atk',  team:'red',  role:'attacker', label:'紅隊．前鋒機',
      keyMap:{ w:'w', s:'s', a:'a', d:'d', arrowup:'arrowup', arrowdown:'arrowdown', arrowleft:'arrowleft', arrowright:'arrowright' },
      keyLabels:['W','S','A','D','↑','↓','←','→'] },
    { id:'r-def1', team:'red',  role:'defender', label:'紅隊．防守機1',
      keyMap:{ g:'w', v:'s', f:'a', b:'d', u:'arrowup', h:'arrowdown', y:'arrowleft', j:'arrowright' },
      keyLabels:['G','V','F','B','U','H','Y','J'] },
    { id:'r-def2', team:'red',  role:'defender', label:'紅隊．防守機2',
      keyMap:{ '1':'w', '2':'s', '3':'a', '4':'d', '5':'arrowup', '6':'arrowdown', '7':'arrowleft', '8':'arrowright' },
      keyLabels:['1','2','3','4','5','6','7','8'] },
    { id:'b-atk',  team:'blue', role:'attacker', label:'藍隊．前鋒機',
      keyMap:{ q:'w', e:'s', z:'a', x:'d', i:'arrowup', k:'arrowdown', o:'arrowleft', p:'arrowright' },
      keyLabels:['Q','E','Z','X','I','K','O','P'] },
    { id:'b-def1', team:'blue', role:'defender', label:'藍隊．防守機1',
      keyMap:{ '[':'w', ']':'s', ';':'a', "'":'d', '.':'arrowup', ',':'arrowdown', '/':'arrowleft', '\\':'arrowright' },
      keyLabels:['[',']',';',"'",'.',',','/','\\'] },
    { id:'b-def2', team:'blue', role:'defender', label:'藍隊．防守機2',
      keyMap:{ '9':'w', '0':'s', n:'a', '`':'d', '-':'arrowup', '=':'arrowdown', ' ':'arrowleft', 'enter':'arrowright' },
      keyLabels:['9','0','N','`','-','=','Space','Enter'] }
];
  function definitions(roster) {
    if (!valid(roster)) throw new Error('Invalid match roster');
    let humanIndex = 0;
    return teams.flatMap(team => {
      const row = roster[team];
      return Array.from({ length: row.humans + row.ai }, (_, index) => {
        const ai = index >= row.humans, role = index === 0 ? 'attacker' : 'defender';
        const control = ai ? { keyMap: {}, keyLabels: [] } : controls[humanIndex++];
        return { id: `${team[0]}-${index === 0 ? 'atk' : 'def' + index}`, team, role, ai,
          label: `${team === 'red' ? '紅隊' : '藍隊'}．${ai ? 'AI ' : '玩家 '}${role === 'attacker' ? '前鋒' : '防守 ' + index}`,
          guardHomeOffsetX: index === 1 ? (row.humans + row.ai === 2 ? 0 : -.28) : .28,
          keyMap: { ...control.keyMap }, keyLabels: [...control.keyLabels] };
      });
    });
  }
  function launchPosition(rigs, rig, takeoffZ = 2.6) {
    const teammates = rigs.filter(r => r.team === rig.team);
    const x = teammates.length === 2
      ? (teammates.indexOf(rig) === 0 ? -.14 : .14)
      : rig.role === 'attacker' ? 0 : rig.guardHomeOffsetX;
    return { x, y: .1, z: rig.team === 'red' ? -takeoffZ : takeoffZ };
  }
  function setup(rigs, rings, colors, applyColor, takeoffZ, pickPattern) {
    rigs.forEach(rig => {
      const own = rings.find(r => r.team === rig.team);
      applyColor(rig, colors[rig.team]);
      rig.guardRing = own;
      rig.targetRing = rings.find(r => r.team !== rig.team);
      // Keep every cage inside the team's 1 m launch zone, with no overlaps.
      // Tactical guard offsets remain separate from these ground positions.
      const start = launchPosition(rigs, rig, takeoffZ);
      rig.homePosition.set(start.x, start.y, start.z);
      rig.position.copy(rig.homePosition);
      rig.velocity.set(0, 0, 0); rig.rotation.set(0, rig.team === 'red' ? 0 : Math.PI, 0);
      rig.targetAltitude = .1; rig.autoFlightState = 'none'; rig.input.throttle = 0;
      Object.keys(rig.keys).forEach(key => { rig.keys[key] = false; });
      rig.currentPitch = 0; rig.currentRoll = 0; rig.currentYawRate = 0;
      rig.altitudeState = 'grounded'; rig.phase = 'advance'; rig.reactionTimer = 0;
      rig.jukePhase = 'none'; rig.jukeCheckTimer = Math.random() * .6;
      rig.gapTimer = 0; rig.gapHasBlockers = false;
      if (rig.ai && rig.role === 'attacker') pickPattern(rig);
      rig.aiAimTarget.copy(rig.position);
      rig.ringSide = {}; rig.ringLastPass = {}; rig.ringPending = {}; rig.centerSide = null; rig.readyToScore = true;
      window.DroneGoalPhysics?.resetPasses(rig);
      if (rig.mesh) { rig.mesh.position.copy(rig.position); rig.mesh.rotation.copy(rig.rotation); }
    });
  }
  window.DroneMatchRoster = Object.freeze({ valid, read, write, definitions, launchPosition, setup, defaults: () => clone(defaults) });
})();
