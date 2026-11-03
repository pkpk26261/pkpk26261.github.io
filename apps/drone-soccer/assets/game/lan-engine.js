/* Optional LAN adapter. Ordinary radio/offline games never enter this code. */
(() => {
  'use strict';
  const mode = new URLSearchParams(location.search).get('lan');
  if (!['authority','viewer'].includes(mode)) return;
  let bridge, control, appliedAt = 0, command = 0, lastPublish = 0, sequence = 0;
  let renderer, camera, surface, config, samples = [], observer = 'all', observerTarget = '', ready = false;
  let rendering = null;
  let arenaCorners, arenaCenter, broadcastFrameKey='',previewActive=null;const previewMaterials=[],endNets=[];
  const actionStates=new WeakMap();
  let playerViewShift=0;
  function releaseActions(actor) {
    actor.controllerBrakeHeld=false;
    window.DroneFlightDynamics?.cancelBackflip?.(actor);
    actionStates.delete(actor);
  }
  function playerActions(actor,input) {
    const key=`${control?.epoch}:${control?.command}`,previous=actionStates.get(actor);
    const buttons=input?.buttons||[],down=i=>(buttons[i]||0)>.65;
    actor.controllerBrakeHeld=false;
    if(!input?.connected){releaseActions(actor);return;}
    const state=previous?.key===key?previous:{key,ready:false,buttons:[]};
    if(!state.ready){
      state.ready=['lx','ly','rx','ry'].every(k=>Math.abs(input[k]||0)<=.12)&&buttons.every(v=>v<.35);
    }else{
      const edge=i=>down(i)&&!(state.buttons[i]>.35);
      actor.controllerBrakeHeld=down(1);
      if(edge(1)){
        window.DroneFlightInput.reset(actor);
        actor.targetAltitude=Math.max(.1,actor.position.y);actor.autoFlightState='none';
        window.DroneFlightDynamics.cancelBackflip?.(actor);
      }
      if(edge(0)){actor.controllerHeadless=!actor.controllerHeadless;if(actor.controllerHeadless)actor.controllerHeadlessYaw=actor.rotation.y;}
      if(edge(5))window.DroneFlightDynamics.beginBackflip?.(actor);
    }
    state.buttons=buttons.slice();actionStates.set(actor,state);
  }
  const copy = v => JSON.parse(JSON.stringify(v));
  const send = data => parent.postMessage({ source:'drone-lan-scene', ...data }, location.origin);
  function publish(now) {
    if (!bridge || now-lastPublish < 50 || typeof window.lanSnapshot !== 'function') return;
    lastPublish = now;
    const m = bridge.match;
    window.lanSnapshot({ seq:++sequence, epoch:control?.epoch || 0, engineRevision:Number(new URLSearchParams(location.search).get('lanRevision')),
      match:{ phase:m.phase, set:m.set, setRemaining:m.setRemaining, phaseLeft:m.phaseLeft,
        setWins:copy(m.setWins), history:copy(m.history), returning:copy(m.returning), winner:m.winner,
        winReason:m.winReason, goldenElapsed:m.goldenElapsed, shootout:!!m.shootout }, scores:copy(bridge.scores),
      failure:bridge.failed() ? '無人機失去恢復能力，比賽已停止；請由主持人重建比賽' : '',
      checkpoint:bridge.checkpoint(),
      goals:bridge.goals.map(r=>({team:r.team,...window.DroneGoalPhysics.snapshot(r)})),
      rigs:bridge.rigs.map(r => ({ id:r.id, ai:r.ai, p:r.position.toArray(), q:r.mesh.quaternion.toArray(),
        v:r.velocity.toArray(), rotation:r.rotation.toArray().slice(0,3), pitch:r.currentPitch, roll:r.currentRoll,
        throttle:r.input.throttle, gear:r.controlGear || 0, readyToScore:r.readyToScore,
        headless:!!r.controllerHeadless,braking:!!r.controllerBrakeHeld,backflip:!!r.controllerBackflipActive,
        rotors:r.rotors.map(rotor => rotor.rotation.y), safety:window.DroneFlightSafety.snapshot(r) }))
    }).catch(() => {});
  }
  function configure(data) {
    // FOV, height, distance and team end remain host-owned. LT/RT may adjust
    // only the player's lateral standing position along that same short end.
    config = data; samples = [];broadcastFrameKey='';playerViewShift=0;
    if (mode === 'viewer') send({ type:'configured', revision:data.camera.revision, engineRevision:data.engineRevision });
  }
  function setPreview(enabled){
    if(previewActive===enabled)return;previewActive=enabled;
    for(const item of previewMaterials){item.material.opacity=enabled?item.preview:item.opacity;item.material.transparent=enabled||item.transparent;item.material.depthWrite=enabled?false:item.depthWrite;item.material.needsUpdate=true;}
  }
  function frameBroadcast(aspect,preview) {
    const key=observer+'|'+aspect+'|'+preview;if(key===broadcastFrameKey)return;broadcastFrameKey=key;
    const direction=new THREE.Vector3(...({left:[-1,.4,.08],right:[1,.4,-.08],center:[0,.4,1]}[observer]||(preview?[1,1.15,.5]:[1,.62,.28]))).normalize();
    camera.position.copy(arenaCenter).add(direction);camera.lookAt(arenaCenter);camera.updateMatrixWorld();
    const inverse=camera.quaternion.clone().invert(),tangent=Math.tan(camera.fov*Math.PI/360);let distance=0;
    for(const point of arenaCorners){
      const local=point.clone().sub(arenaCenter).applyQuaternion(inverse);
      distance=Math.max(distance,local.z+Math.abs(local.x)/(tangent*aspect*.88),local.z+Math.abs(local.y)/(tangent*.88),local.z+camera.near+.1);
    }
    camera.position.copy(arenaCenter).addScaledVector(direction,distance);camera.lookAt(arenaCenter);
  }
  function render(now) {
    if (!bridge || !config || !samples.length) return;
    const target = now - 80;
    while (samples.length > 2 && samples[1].at <= target) samples.shift();
    const a = samples[0], z = samples[1] || a;
    const t = Math.max(0,Math.min(1,(target-a.at)/Math.max(1,z.at-a.at)));
    for (const state of z.value.rigs) {
      const r = bridge.rigs.find(r => r.id === state.id); if (!r) continue;
      const old = a.value.rigs.find(r => r.id === state.id) || state;
      r.position.fromArray(old.p).lerp(new THREE.Vector3().fromArray(state.p), t);
      r.mesh.position.copy(r.position);
      r.mesh.quaternion.fromArray(old.q).slerp(new THREE.Quaternion().fromArray(state.q), t);
      r.rotation.set(...state.rotation,'YXZ');
      const yawDelta=state.rotation[1]-old.rotation[1];r.rotation.y=old.rotation[1]+Math.atan2(Math.sin(yawDelta),Math.cos(yawDelta))*t;
      r.currentPitch=old.pitch+(state.pitch-old.pitch)*t;r.currentRoll=old.roll+(state.roll-old.roll)*t;
      r.readyToScore=state.readyToScore; r.input.throttle=state.throttle;
      r.rotors.forEach((rotor,i) => { rotor.rotation.y=state.rotors[i] || 0; }); r.mesh.visible=true;
    }
    for(const state of z.value.goals || []){
      const ring=bridge.goals.find(r=>r.team===state.team);if(!ring)continue;
      const old=(a.value.goals || []).find(r=>r.team===state.team)||state;
      const p=new THREE.Vector3().fromArray(old.p).lerp(new THREE.Vector3().fromArray(state.p),t).toArray();
      const q=new THREE.Quaternion().fromArray(old.q).slerp(new THREE.Quaternion().fromArray(state.q),t).toArray();
      window.DroneGoalPhysics.applySnapshot(ring,{...state,p,q});
    }
    Object.assign(bridge.scores,z.value.scores); Object.assign(bridge.match,z.value.match); bridge.updateGoals();
    const rules = config.camera;
    const r = bridge.rigs.find(r => r.id === (config.role === 'player' ? config.seat : observerTarget)) || bridge.rigs[0];
    const w=innerWidth,h=innerHeight,playerView=config.role==='player';
    const preview=!playerView&&z.value.match.phase==='idle';setPreview(preview);
    camera.fov = playerView ? rules.fov : preview?40:45;
    camera.aspect = playerView ? 16/9 : w/Math.max(1,h);camera.updateProjectionMatrix();
    if(playerView){
      const endSign=r.team==='red'?-1:1,lateral=rules.mode==='left'?-1:rules.mode==='right'?1:0;
      const bounds=window.DroneArenaCollision.bounds,netZ=endSign<0?bounds.z[0]:bounds.z[1];
      const sideX=Math.max(-.9,Math.min(.9,(lateral*.9+playerViewShift)*endSign))||0;
      camera.position.set(sideX,rules.height,netZ+endSign*rules.distance);
      camera.lookAt(0,1.35,0);
    }else if(observer==='follow'){
      const yaw = r.rotation.y, quat = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0,1,0),yaw);
        camera.position.copy(r.position).add(new THREE.Vector3(0,.55,-1.6).applyQuaternion(quat));
        camera.position.y=Math.min(2.55,Math.max(.15,camera.position.y));
        camera.lookAt(r.position.clone().add(new THREE.Vector3(0,.05,0)));
    } else {
      frameBroadcast(camera.aspect,preview);
    }
    const vw=playerView?Math.min(w,h*16/9):w,vh=playerView?vw*9/16:h;
    renderer.setSize(w,h,false); renderer.setScissorTest(false); renderer.setViewport(0,0,w,h); renderer.clear();
    renderer.setViewport((w-vw)/2,(h-vh)/2,vw,vh);
    if(playerView)for(const net of endNets)if(net.mesh.position.z*(r.team==='red'?-1:1)>0)net.mesh.visible=false;
    window.DroneArenaLighting.update(bridge.scene); renderer.render(bridge.scene,camera);
    for(const net of endNets)net.mesh.visible=net.visible;
    bridge.rigs.forEach(r => { r.mesh.visible=true; });
    if (!ready) { ready=true; send({ type:'rendered', revision:rules.revision, engineRevision:config.engineRevision }); }
  }
  window.addEventListener('message', event => {
    if (mode !== 'viewer' || event.source !== parent || event.origin !== location.origin || event.data?.source !== 'drone-lan-app') return;
    const data=event.data;
    if(data.type==='config'){ ready=false;configure(data); }
    if(data.type==='snapshot' && config && data.value.epoch >= (samples.at(-1)?.value.epoch || 0)) {
      if(samples.at(-1)?.value.epoch!==data.value.epoch)samples=[];
      samples.push({at:performance.now(),value:data.value}); if(samples.length>8)samples.shift();
    }
    if(data.type==='observer' && config?.role !== 'player' && ['all','left','right','center','follow'].includes(data.mode)) {
      observer=data.mode; observerTarget=data.target;broadcastFrameKey='';
    }
    if(data.type==='player-view-shift' && config?.role==='player' && [-1,1].includes(data.direction) && Number.isFinite(data.delta)) {
      const base=({left:-.9,right:.9})[config.camera.mode]||0;
      playerViewShift=Math.max(-.9-base,Math.min(.9-base,playerViewShift+data.direction*Math.max(0,Math.min(.1,data.delta))*.7));
    }
  });
  // Capture local controls in LAN scenes; only the calibrated tablet client
  // sends inputs. Legacy keyboard, touch, zoom and orbit listeners cannot fly.
  for(const type of ['keydown','keyup','wheel','pointerdown','touchstart','touchmove']) window.addEventListener(type,e=>{
    if(mode==='viewer'){e.preventDefault();e.stopImmediatePropagation();}
  },{capture:true,passive:false});
  window.DroneLANEngine = {
    active:true,
    get rendering() { return rendering; },
    mount(value) {
      bridge=value;
      if(mode==='authority') { document.title='LAN authority'; return; }
      bridge.rigs.forEach(r=>r.orbitControls?.dispose());
      // Fit the playing enclosure; decorative light heads do not dictate scale.
      const bounds=window.DroneArenaCollision.bounds||{x:[-1.5,1.5],y:[0,3],z:[-3,3]};
      arenaCorners=[];
      for(const x of [bounds.x[0]-.22,bounds.x[1]+.22])for(const y of [Math.min(0,bounds.y[0])-.05,bounds.y[1]+.2])for(const z of [bounds.z[0]-.22,bounds.z[1]+.22])arenaCorners.push(new THREE.Vector3(x,y,z));
      arenaCenter=new THREE.Box3().setFromPoints(arenaCorners).getCenter(new THREE.Vector3());
      const floorGrid=new THREE.GridHelper(18,36,0x284454,0x172c3b);floorGrid.position.y=-.015;floorGrid.material.transparent=true;floorGrid.material.opacity=.3;bridge.scene.add(floorGrid);
      const style=document.createElement('style');style.textContent='html body>*:not(#lanWorld){display:none!important}body *{visibility:hidden!important}#lanWorld,#lanWorld canvas{visibility:visible!important}html,body{margin:0!important;overflow:hidden!important;background:#000!important}#lanWorld{display:block!important;position:fixed!important;inset:0!important}#lanWorld canvas{display:block!important;position:absolute!important;inset:0!important;width:100%!important;height:100%!important;pointer-events:none!important}';document.head.append(style);
      surface=document.createElement('div');surface.id='lanWorld';document.body.append(surface);
      renderer=bridge.rigs.find(r=>r.renderer).renderer;surface.append(renderer.domElement);
      // The ordinary split renderer prepares lighting lazily. LAN viewers use
      // their own render loop, so they must apply the same art direction here.
      window.DroneArenaLighting.prepare(bridge.scene,renderer);
      // Software WebGL can spend longer than the input watchdog on dynamic
      // shadows. Keep the venue and contact shadows while reducing viewer GPU
      // work; the authority and its physics do not enter this mount branch.
      let software=false;
      try {
        const context=renderer.getContext(),debug=context.getExtension('WEBGL_debug_renderer_info');
        if(debug)software=/swiftshader|llvmpipe|softpipe|software rasterizer/i.test(String(context.getParameter(debug.UNMASKED_RENDERER_WEBGL)||''));
      } catch (_) { /* Restricted renderer diagnostics retain normal quality. */ }
      const scale=Math.min(devicePixelRatio||1,software ? .65 : 1.5);
      if(software)renderer.shadowMap.enabled=false;
      renderer.setPixelRatio(scale);
      rendering=Object.freeze({software,scale,dynamicShadows:!!renderer.shadowMap.enabled});
      bridge.scene.traverse(object=>{
        if(object.name.startsWith('flexible-rope-z-'))endNets.push({mesh:object,visible:object.visible});
        if(!object.isMesh||!object.material||Array.isArray(object.material))return;
        const roofRail=object.userData.perfMerge==='tube'&&object.position.y>2.6;
        const net=object.name.startsWith('flexible-rope-');
        if(!roofRail&&!net)return;
        object.material=object.material.clone();const material=object.material;
        previewMaterials.push({material,opacity:material.opacity,transparent:material.transparent,depthWrite:material.depthWrite,preview:roofRail ? .2 : object.name.includes('rope-y-') ? .035 : .09});
      });
      renderer.autoClear=true;
      camera=new THREE.PerspectiveCamera(60,16/9,.025,100);
      send({type:'loaded'});
    },
    apply(value) {
      if(mode!=='authority'||!bridge)return;
      control=value;appliedAt=performance.now();
      if(value.command!==command){command=value.command;bridge.start(value.breakSec,value.difficulty);}
      for(const s of value.seats){const r=bridge.rigs.find(r=>r.id===s.id);if(r&&!r.ai&&s.ai){r.ai=true;r.aiAutoGear=true;r.altitudeState=r.position.y>.2?'flying':'takingoff';}}
      if(value.status!=='running')for(const r of bridge.rigs)if(!r.ai){r.gp={connected:false,lx:0,ly:0,rx:0,ry:0};r.gpAltArmed=false;releaseActions(r);window.DroneFlightInput.reset(r);}
    },
    restore(value){if(mode==='authority'&&bridge){command=value.command;bridge.restore(value.checkpoint);}},
    frame(now=performance.now()) {
      if(!bridge)return true;
      if(mode==='viewer'){render(now);return true;}
      publish(now);
      if(!control || performance.now()-appliedAt>350 || control.status!=='running') {for(const r of bridge.rigs)if(!r.ai)releaseActions(r);bridge.resetClock();return true;}
      for(const r of bridge.rigs)if(!r.ai){const input=control.inputs[r.id];r.gp=input || {connected:false,lx:0,ly:0,rx:0,ry:0};if(input)r.controlGear=input.gear;
        playerActions(r,input);
        window.DroneFlightSafety.setPad(r,r.gp);window.DroneFlightSafety.observePad(r,r.gp.connected?r.gp:null,[{x:r.gp.lx,y:r.gp.ly},{x:r.gp.rx,y:r.gp.ry}]);}
      return false;
    },
    skipRender:mode==='authority'
  };
})();
