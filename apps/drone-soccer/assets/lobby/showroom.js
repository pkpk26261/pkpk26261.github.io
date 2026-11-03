/* Real Flight Club aircraft, lit for the title screen. No simulation state. */
(function (global) {
  'use strict';
  const source = document.currentScript?.src || new URL('assets/lobby/showroom.js', location.href).href;
  const base = new URL('.', source);
  let dependencyPromise;
  function load(relative, ready) {
    if (ready()) return Promise.resolve();
    return new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = new URL(relative, base).href;
      script.onload = () => ready() ? resolve() : reject(new Error('Showroom dependency unavailable'));
      script.onerror = () => reject(new Error('Showroom dependency failed to load'));
      document.head.append(script);
    });
  }
  function dependencies() {
    return dependencyPromise ||= load('../../../../vendor/three-r128/three.min.js', () => !!global.THREE)
      .then(() => load('../../../../vendor/three-r128/BufferGeometryUtils.js', () => !!global.THREE.BufferGeometryUtils))
      .then(() => load('../game/drone-appearance.js', () => !!global.DroneAppearance));
  }
  async function createScene(host, options = {}) {
    if (!host) return null;
    if (host.flightShowroom) return host.flightShowroom;
    host.dataset.showroom = 'loading';
    host.style.pointerEvents = 'none';
    const posterBackground = 'url("' + new URL('showroom-poster.png', base).href + '")';
    host.style.backgroundImage = posterBackground;
    host.style.backgroundSize = 'cover'; host.style.backgroundPosition = 'center';
    const fallback = { setActive() {}, setTheme() {}, dispose() { delete host.flightShowroom;delete host.flightShowroomMount; }, available: false };
    try { await dependencies(); } catch (_) { host.dataset.showroom = 'fallback'; return fallback; }
    const THREE = global.THREE;
    let renderer;
    try { renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'low-power' }); }
    catch (_) { host.dataset.showroom = 'fallback'; return fallback; }
    renderer.domElement.setAttribute('aria-hidden', 'true');
    renderer.domElement.style.cssText = 'display:block;width:100%;height:100%;pointer-events:none;opacity:0;transition:opacity .6s';
    renderer.outputEncoding = THREE.sRGBEncoding;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = .95;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    host.append(renderer.domElement);
    const ownedGeometries = new Set(), ownedMaterials = new Set(), ownedTextures = new Set();
    const scene = new THREE.Scene(); scene.background = new THREE.Color(0x080b12);
    scene.fog = new THREE.FogExp2(0x080b12, .045);
    const camera = new THREE.PerspectiveCamera(32, 1, .1, 70);
    camera.position.set(3.7, 2.55, 5.8);
    const target = new THREE.Vector3(0, 1.45, 0);
    const ambient = new THREE.HemisphereLight(0x8da9c6, 0x080a12, .58); scene.add(ambient);
    const key = new THREE.DirectionalLight(0xffeddd, 2.3); key.position.set(-3, 5.8, 4); scene.add(key);
    key.castShadow = true; key.shadow.mapSize.set(1024, 1024);
    Object.assign(key.shadow.camera, { left:-4, right:4, top:4, bottom:-4, near:.1, far:16 });
    key.shadow.camera.updateProjectionMatrix();
    key.shadow.bias = -.0003; key.shadow.normalBias = .015; key.shadow.radius = 4;
    const rim = new THREE.DirectionalLight(0x71beff, 2.6); rim.position.set(2.5, 2.8, -3.8); scene.add(rim);
    const fill = new THREE.DirectionalLight(0xe6edf9, .4); fill.position.set(5, 1, 3.5); scene.add(fill);
    const coral = new THREE.PointLight(0xff654d, 1.7, 8, 2); coral.position.set(-1.8, 1.8, 1.5); scene.add(coral);
    function material(color, metalness = .3, roughness = .5, extra = {}) {
      const value = new THREE.MeshStandardMaterial({ color, metalness, roughness, ...extra });
      value.color.convertSRGBToLinear(); if(extra.emissive)value.emissive.convertSRGBToLinear();
      ownedMaterials.add(value); return value;
    }
    function object(geometry, mat, position, rotation) {
      ownedGeometries.add(geometry); const mesh = new THREE.Mesh(geometry, mat);
      if (position) mesh.position.set(...position); if (rotation) mesh.rotation.set(...rotation);
      mesh.receiveShadow = true; scene.add(mesh); return mesh;
    }
    const floor = object(new THREE.PlaneGeometry(90,90), material(0x0a111b,.38,.48), [0,-.12,0], [-Math.PI/2,0,0]);
    const platform = object(new THREE.CylinderGeometry(1.72,1.79,.15,96), material(0x18212b,.64,.36), [0,-.025,0]);
    platform.castShadow = true;
    object(new THREE.CylinderGeometry(1.54,1.54,.018,96), material(0x111923,.45,.64), [0,.06,0]);
    object(new THREE.TorusGeometry(1.61,.009,6,160), material(0xff654d,.22,.5,{emissive:0xff654d,emissiveIntensity:1.4}), [0,.066,0], [Math.PI/2,0,0]);
    object(new THREE.TorusGeometry(1.84,.002,4,144), material(0x486278,.25,.6,{emissive:0x243646,emissiveIntensity:.4}), [0,-.11,0], [Math.PI/2,0,0]);
    // Inlaid runway marks and broad architectural light strips keep the scene a hangar.
    const darkMetal = material(0x111a26,.35,.65), strip = material(0x8fa4b8,.25,.45,{emissive:0x45677e,emissiveIntensity:.7});
    for (const x of [-3.5,3.5]) {
      object(new THREE.BoxGeometry(.035,.008,30), material(0x34485a,.3,.6), [x,-.113,-6]);
      object(new THREE.BoxGeometry(.09,7,.18), darkMetal, [x,3,-6.2]);
      object(new THREE.BoxGeometry(.032,4,.04), strip, [x + (x<0?.07:-.07),2.6,-6.07]);
    }
    for (const z of [-2.5,-5,-7.5]) object(new THREE.BoxGeometry(12,.008,.017), material(0x1c2a37,.3,.6), [0,-.111,z]);
    for (const x of [-1.13,1.13]) for (let i=0;i<3;i++) {
      object(new THREE.BoxGeometry(.16,.008,.025), material(0xb4b5b5,.12,.65), [x,.071,-.44+i*.14]);
    }
    // A soft contact gradient complements the real shadow while the aircraft hovers.
    const shadowCanvas = document.createElement('canvas'); shadowCanvas.width = shadowCanvas.height = 128;
    const ctx = shadowCanvas.getContext('2d'), gradient = ctx.createRadialGradient(64,64,3,64,64,62);
    gradient.addColorStop(0,'rgba(0,0,0,.75)'); gradient.addColorStop(.45,'rgba(0,0,0,.4)'); gradient.addColorStop(1,'rgba(0,0,0,0)');
    ctx.fillStyle = gradient; ctx.fillRect(0,0,128,128);
    const shadowTexture = new THREE.CanvasTexture(shadowCanvas); ownedTextures.add(shadowTexture);
    const shadowMaterial = new THREE.MeshBasicMaterial({map:shadowTexture,transparent:true,depthWrite:false,opacity:.74}); ownedMaterials.add(shadowMaterial);
    const shadow = object(new THREE.PlaneGeometry(2.85,2.85),shadowMaterial,[0,.083,0],[-Math.PI/2,0,0]);
    const aircraft = new THREE.Group(); aircraft.name = 'showroom-authentic-flight-club-aircraft';
    const body = global.DroneAppearance.createBody(0xff654d); body.group.scale.setScalar(.23); aircraft.add(body.group);
    const cage = global.DroneAppearance.createCage(.1,0xff654d); aircraft.add(cage.group);
    const beacon = global.DroneAppearance.createTailBeacon();
    beacon.group.position.set(0,global.DroneAppearance.REAR_LIGHT_Y,global.DroneAppearance.REAR_LIGHT_Z); aircraft.add(beacon.group);
    // Clone only materials: the source geometry and proportions are the actual game asset.
    const clones = new Map();
    aircraft.traverse(node => {
      if (!node.isMesh && !node.isSprite) return;
      const clone = original => {
        if (!clones.has(original)) { const value = original.clone(); clones.set(original,value); ownedMaterials.add(value); }
        return clones.get(original);
      };
      node.material = Array.isArray(node.material) ? node.material.map(clone) : clone(node.material);
      if (node.isMesh) { node.castShadow = true; node.receiveShadow = true; }
    });
    const heroCage = cage.group.children[0].material;
    heroCage.color.set(0xff654d).convertSRGBToLinear(); heroCage.emissive.set(0x5a130a).convertSRGBToLinear();
    heroCage.emissiveIntensity = .08; heroCage.toneMapped = true; heroCage.roughness = .33; heroCage.metalness = .16;
    // All team-colored cage stays use the same clone, retaining the authentic geometry.
    body.canopy.material.color.set(0xff654d).convertSRGBToLinear();
    aircraft.scale.setScalar(11.4); aircraft.position.set(0,1.65,0);
    aircraft.rotation.set(.08,-.55,-.1); scene.add(aircraft);
    let active = options.active !== false, intersecting = true, disposed = false, lost = false;
    let animation = 0, lastFrame = 0, elapsed = 0, width = 1, height = 1;
    const reduced = matchMedia('(prefers-reduced-motion: reduce)');
    const pointer = new THREE.Vector2(), smoothPointer = new THREE.Vector2();
    const originalCamera = camera.position.clone();
    let firstFrame = true;
    function draw(delta = 0) {
      if (disposed || lost) return;
      if (motionEnabled()) {
        elapsed += delta;
        smoothPointer.lerp(pointer,Math.min(1,delta*3));
        aircraft.rotation.y = -.55 + Math.sin(elapsed*.17)*.24;
        aircraft.rotation.z = -.09 + Math.sin(elapsed*.27)*.016;
        aircraft.position.y = 1.65 + Math.sin(elapsed*.75)*.027;
        for (let i=0;i<body.rotors.length;i++) body.rotors[i].rotation.y += delta*(i%2?-.6:.6);
      }
      camera.position.copy(originalCamera);
      camera.position.x += smoothPointer.x*.12; camera.position.y += smoothPointer.y*.08;
      camera.lookAt(target); renderer.render(scene,camera);
      if (firstFrame) { firstFrame=false; host.dataset.showroom='ready'; host.style.backgroundImage='none'; renderer.domElement.style.opacity='1'; }
    }
    function motionEnabled() { return !reduced.matches && document.documentElement.dataset.menuMotion!=='off'; }
    function canRender() { return active && intersecting && !document.hidden && !disposed && !lost; }
    function canAnimate() { return canRender() && motionEnabled(); }
    function frame(time) {
      animation=0;
      if (!canAnimate()) return;
      if (time-lastFrame >= 1000/30) { const delta=lastFrame?Math.min(.08,(time-lastFrame)/1000):0;lastFrame=time;draw(delta); }
      if (canAnimate()) animation=requestAnimationFrame(frame);
    }
    function wake() {
      if (animation) cancelAnimationFrame(animation); animation=0;lastFrame=0;
      if (canRender()) { draw(); if (canAnimate()) animation=requestAnimationFrame(frame); }
    }
    function resize() {
      const bounds=host.getBoundingClientRect(); width=Math.max(1,bounds.width);height=Math.max(1,bounds.height);
      const narrow=width<700;
      renderer.setPixelRatio(Math.min(global.devicePixelRatio||1,narrow?1.25:1.6));
      renderer.setSize(width,height,false); camera.aspect=width/height;camera.fov=narrow?55:32;
      camera.clearViewOffset();
      const align=options.align||'right';
      if (align==='right') camera.setViewOffset(width,height,-width*.22,height*(narrow?.13:.015),width,height);
      camera.updateProjectionMatrix();
      key.shadow.mapSize.set(narrow?512:1024,narrow?512:1024);
      wake();
    }
    function move(event) {
      if (!motionEnabled() || options.interactive===false || event.pointerType==='touch') return;
      const bounds=host.getBoundingClientRect();
      pointer.set(Math.max(-1,Math.min(1,(event.clientX-bounds.left)/width*2-1)),Math.max(-1,Math.min(1,(event.clientY-bounds.top)/height*2-1)));
    }
    function leave() { pointer.set(0,0); }
    function contextLost(event) { event.preventDefault(); lost=true;wake();host.style.backgroundImage=posterBackground;host.dataset.showroom='fallback';renderer.domElement.style.opacity='0'; }
    function contextRestored() { lost=false;wake();host.style.backgroundImage='none';renderer.domElement.style.opacity='1';host.dataset.showroom='ready'; }
    function homeVisibility(event) { active=!!event.detail?.active;wake(); }
    const observer = new ResizeObserver(resize); observer.observe(host);
    const intersection = new IntersectionObserver(entries=>{intersecting=entries[0]?.isIntersecting!==false;wake();}); intersection.observe(host);
    document.addEventListener('visibilitychange',wake);
    global.addEventListener('flight-home-visibility',homeVisibility);
    global.addEventListener('flight-menu-preferences',wake);
    global.addEventListener('pointermove',move,{passive:true}); global.addEventListener('pointerleave',leave);
    reduced.addEventListener?.('change',wake);
    renderer.domElement.addEventListener('webglcontextlost',contextLost); renderer.domElement.addEventListener('webglcontextrestored',contextRestored);
    function setTheme(category) {
      const color={battle:0xff654d,practice:0x62cabd,racing:0x77bcff,official:0xf2c87b}[category]||0xff654d;
      heroCage.color.set(color).convertSRGBToLinear(); body.canopy.material.color.set(color).convertSRGBToLinear();
      coral.color.set(color); wake();
    }
    const handle = {
      available:true, canvas:renderer.domElement,
      setActive(value) { active=!!value;wake(); }, setTheme,
      capturePoster() { draw(); return renderer.domElement.toDataURL('image/png'); },
      renderInfo() { return {calls:renderer.info.render.calls,triangles:renderer.info.render.triangles,frames:renderer.info.render.frame,active:canAnimate(),visible:canRender(),motion:motionEnabled(),reducedMotion:reduced.matches,width,height}; },
      dispose() {
        if(disposed)return;disposed=true;if(animation)cancelAnimationFrame(animation);
        observer.disconnect();intersection.disconnect();document.removeEventListener('visibilitychange',wake);
        global.removeEventListener('flight-home-visibility',homeVisibility);
        global.removeEventListener('flight-menu-preferences',wake);
        global.removeEventListener('pointermove',move);global.removeEventListener('pointerleave',leave);reduced.removeEventListener?.('change',wake);
        renderer.domElement.removeEventListener('webglcontextlost',contextLost);renderer.domElement.removeEventListener('webglcontextrestored',contextRestored);
        ownedGeometries.forEach(geometry=>geometry.dispose());ownedMaterials.forEach(mat=>mat.dispose());ownedTextures.forEach(texture=>texture.dispose());
        renderer.dispose();renderer.domElement.remove();delete host.flightShowroom;delete host.flightShowroomMount;host.style.backgroundImage=posterBackground;host.dataset.showroom='fallback';
      }
    };
    host.flightShowroom=handle;
    if(options.theme)setTheme(options.theme);resize();
    return handle;
  }
  function mount(host, options = {}) {
    if (!host) return Promise.resolve(null);
    if (host.flightShowroom) return Promise.resolve(host.flightShowroom);
    if (!host.flightShowroomMount) host.flightShowroomMount=createScene(host,options).then(handle=>{host.flightShowroom=handle;return handle;});
    return host.flightShowroomMount;
  }
  global.FlightShowroom=Object.freeze({mount});
  async function autoMount() {
    const host=document.getElementById('showroom-canvas-host');
    const screen=document.documentElement.dataset.screen;
    if(host)global.flightShowroom=await mount(host,{align:'right',active:!screen||screen==='home'});
  }
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',autoMount,{once:true});else autoMount();
})(window);
