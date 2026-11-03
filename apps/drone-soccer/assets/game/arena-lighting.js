/* Shared arena art direction for Three.js r128. No gameplay or camera changes. */
(() => {
  'use strict';
  const states = new WeakMap();
  const NET_PITCH = 0.05; // 5 cm openings, well below the 20 cm drone cage diameter.

  function tuneNet(material, geometry, maps, renderer, THREE) {
    const { width, height } = geometry.parameters;
    const key = `${width}:${height}`;
    if (!maps.has(key)) {
      const canvas = document.createElement('canvas');
      canvas.width = canvas.height = 128;
      const ctx = canvas.getContext('2d');
      ctx.strokeStyle = '#93b5c8'; ctx.lineWidth = 3;
      ctx.strokeRect(0, 0, 128, 128);
      // Material.clone() retains the same map: each plane aspect ratio needs
      // its own texture transform so the roof cannot stretch the wall cells.
      const map = new THREE.CanvasTexture(canvas);
      map.wrapS = map.wrapT = THREE.RepeatWrapping;
      map.repeat.set(width / NET_PITCH, height / NET_PITCH);
      map.encoding = THREE.sRGBEncoding;
      map.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
      maps.set(key, map);
    }
    material.map = maps.get(key);
    material.opacity = 0.34;
  }

  function radialTexture(THREE) {
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 128;
    const ctx = canvas.getContext('2d');
    const gradient = ctx.createRadialGradient(64, 64, 4, 64, 64, 64);
    gradient.addColorStop(0, 'rgba(5,12,20,0.8)');
    gradient.addColorStop(0.26, 'rgba(5,12,20,0.6)');
    gradient.addColorStop(0.6, 'rgba(5,12,20,0.22)');
    gradient.addColorStop(1, 'rgba(5,12,20,0)');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 128, 128);
    return new THREE.CanvasTexture(canvas);
  }

  function tuneField(material, THREE) {
    const source = material.map.image;
    const canvas = document.createElement('canvas');
    canvas.width = source.width; canvas.height = source.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(source, 0, 0);
    // Subtle, baked edge occlusion: the middle remains evenly lit and legible.
    const gradient = ctx.createRadialGradient(canvas.width / 2, canvas.height / 2, canvas.width * 0.3,
      canvas.width / 2, canvas.height / 2, canvas.height * 0.68);
    gradient.addColorStop(0, 'rgba(3,21,16,0)');
    gradient.addColorStop(1, 'rgba(3,21,16,0.22)');
    ctx.fillStyle = gradient; ctx.fillRect(0, 0, canvas.width, canvas.height);
    const map = new THREE.CanvasTexture(canvas);
    map.encoding = THREE.sRGBEncoding;
    map.anisotropy = material.map.anisotropy;
    material.map = map;
    material.emissiveMap = null;
    material.emissiveIntensity = 0;
    material.roughness = 0.94;
    material.metalness = 0;
    material.envMapIntensity = 0.12;
  }

  function venueFloorTexture(THREE, renderer) {
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 1024;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#252d37'; ctx.fillRect(0, 0, 1024, 1024);
    // Baked concrete grain and broad slab joints, rather than a wire grid.
    let seed = 71429;
    const random = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
    for (let i = 0; i < 12000; i++) {
      ctx.fillStyle = i % 2 ? 'rgba(255,255,255,.025)' : 'rgba(0,0,0,.04)';
      ctx.fillRect(random() * 1024, random() * 1024, 1 + random() * 3, 1 + random() * 2);
    }
    ctx.strokeStyle = '#1e2630'; ctx.lineWidth = 2;
    [256, 512, 768].forEach(p => { ctx.beginPath(); ctx.moveTo(p, 0); ctx.lineTo(p, 1024); ctx.stroke(); });
    [170, 512, 854].forEach(p => { ctx.beginPath(); ctx.moveTo(0, p); ctx.lineTo(1024, p); ctx.stroke(); });
    const map = new THREE.CanvasTexture(canvas);
    map.encoding = THREE.sRGBEncoding;
    map.anisotropy = Math.min(4, renderer.capabilities.getMaxAnisotropy());
    return map;
  }

  function venueBrandTexture(THREE) {
    const canvas = document.createElement('canvas');
    canvas.width = 1536; canvas.height = 384;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#0d1722'; ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#ff765e'; ctx.fillRect(66, 66, 72, 5);
    ctx.fillStyle = '#e9eff4'; ctx.font = 'italic 900 135px Segoe UI, sans-serif';
    ctx.textAlign = 'left'; ctx.fillText('FLIGHT CLUB', 60, 233);
    ctx.fillStyle = '#8aa5b9'; ctx.font = '600 24px Segoe UI, sans-serif';
    ctx.fillText('D R O N E   S O C C E R   A R E N A', 68, 302);
    ctx.strokeStyle = '#819ab733'; ctx.lineWidth = 2; ctx.strokeRect(20, 20, 1496, 344);
    ctx.fillStyle = '#ff765e'; ctx.fillRect(1432, 266, 39, 39);
    const map = new THREE.CanvasTexture(canvas); map.encoding = THREE.sRGBEncoding;
    return map;
  }

  function buildVenue(scene, renderer, THREE) {
    // This root is deliberately outside the cage group registered by physics.
    // Its solids all sit below the field or beyond the playable X/Z bounds.
    const venue = new THREE.Group(); venue.name = 'arena-venue-decoration';
    venue.userData.visualOnly = true; venue.userData.collision = false;
    const linear = hex => new THREE.Color(hex).convertSRGBToLinear();
    const material = (hex, roughness = .8, metalness = .1) => new THREE.MeshStandardMaterial({ color: linear(hex), roughness, metalness });
    const deck = material(0x1c2733, .7, .3), structure = material(0x243344, .72, .34);
    const wall = material(0x172330, .92, .06);
    wall.emissive.setHex(0x111c2b); wall.emissiveIntensity = .16;
    const warm = new THREE.MeshBasicMaterial({ color: linear(0xc36a51), toneMapped: false });
    const cool = new THREE.MeshBasicMaterial({ color: linear(0x77adc5), toneMapped: false });
    const white = new THREE.MeshBasicMaterial({ color: linear(0x8ca7b9), toneMapped: false });
    const boxGeometry = new THREE.BoxGeometry(1, 1, 1);
    let drawCalls = 0, instances = 0;
    function batch(name, entries, mat) {
      const mesh = new THREE.InstancedMesh(boxGeometry, mat, entries.length);
      const object = new THREE.Object3D();
      entries.forEach(([x,y,z,w,h,d,ry = 0], index) => {
        object.position.set(x,y,z); object.rotation.set(0,ry,0); object.scale.set(w,h,d); object.updateMatrix();
        mesh.setMatrixAt(index, object.matrix);
      });
      mesh.name = name; mesh.castShadow = false; mesh.receiveShadow = true; mesh.frustumCulled = false;
      mesh.userData.visualOnly = true; venue.add(mesh); drawCalls++; instances += entries.length;
      return mesh;
    }
    // A real, low plinth supports the existing grass and cage. Top remains below Y=0.
    batch('venue-stage', [[0,-.12,0,4.2,.2,7.4],[0,-.031,0,4.08,.022,7.28],
      [-2.25,-.145,-2.8,.3,.07,.6],[2.25,-.145,2.8,.3,.07,.6]], deck);
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(32, 36), new THREE.MeshStandardMaterial({
      color: 0xffffff, map: venueFloorTexture(THREE, renderer), roughness: .96, metalness: .04,
      emissive: linear(0x18212b), emissiveIntensity: .12
    }));
    floor.name = 'venue-concrete-floor'; floor.rotation.x = -Math.PI / 2; floor.position.y = -.224;
    floor.receiveShadow = true; floor.castShadow = false; floor.userData.visualOnly = true; venue.add(floor); drawCalls++;
    batch('venue-red-edge', [[-2.108,-.047,-1.83,.018,.025,3.61],[2.108,-.047,-1.83,.018,.025,3.61],[0,-.047,-3.708,4.2,.025,.018]], warm);
    batch('venue-blue-edge', [[-2.108,-.047,1.83,.018,.025,3.61],[2.108,-.047,1.83,.018,.025,3.61],[0,-.047,3.708,4.2,.025,.018]], cool);
    // The enclosed venue starts well beyond the pilot's fixed stations.
    const walls = [[0,3.9,-11.5,22,8,.22],[0,3.9,11.5,22,8,.22],[-11,3.9,0,.22,8,23],[11,3.9,0,.22,8,23]];
    [-1,1].forEach(side => {
      [-8,-4,0,4,8].forEach(z => walls.push([side*10.86,2.1,z,.1,3.8,3.75]));
      [-7,-3.5,3.5,7].forEach(x => walls.push([x,1.2,side*11.3,3.3,2.1,.12]));
    });
    batch('venue-wall-panels', walls, wall);
    const beams = [];
    [-1,1].forEach(side => {
      [-10,-5,0,5,10].forEach(z => beams.push([side*10.5,3.6,z,.18,7.3,.23]));
      [1.1,5.9].forEach(y => beams.push([side*10.5,y,0,.2,.15,20.3]));
      [-7,-3.5,0,3.5,7].forEach(x => beams.push([x,3.6,side*11.05,.18,7.3,.2]));
      beams.push([0,6.8,side*11.05,17.5,.18,.2]);
    });
    batch('venue-structural-frames', beams, structure);
    const lightStrips = [];
    [-1,1].forEach(side => {
      lightStrips.push([side*10.37,5.87,0,.025,.025,19.2]);
      lightStrips.push([0,6.67,side*10.9,15,.025,.025]);
      [-7,-3.5,3.5,7].forEach(x => lightStrips.push([x,3.35,side*10.91,.018,.7,.018]));
    });
    batch('venue-architectural-led', lightStrips, white);
    const brandMaterial = new THREE.MeshBasicMaterial({ map: venueBrandTexture(THREE), color: 0xffffff, toneMapped: false });
    const brand = new THREE.InstancedMesh(new THREE.PlaneGeometry(1,1), brandMaterial, 6);
    const object = new THREE.Object3D();
    const signs = [[0,4.4,-10.92,6.6,1.65,0],[0,4.4,10.92,6.6,1.65,Math.PI],
      [-10.35,2.5,-4.6,3.8,.95,Math.PI/2],[-10.35,2.5,4.6,3.8,.95,Math.PI/2],
      [10.35,2.5,-4.6,3.8,.95,-Math.PI/2],[10.35,2.5,4.6,3.8,.95,-Math.PI/2]];
    signs.forEach(([x,y,z,w,h,ry],i) => { object.position.set(x,y,z); object.rotation.set(0,ry,0); object.scale.set(w,h,1); object.updateMatrix(); brand.setMatrixAt(i,object.matrix); });
    brand.name = 'venue-flight-club-lightboxes'; brand.frustumCulled = false; brand.castShadow = false;
    brand.userData.visualOnly = true; venue.add(brand); drawCalls++; instances += signs.length;
    scene.add(venue);
    scene.userData.arenaVenue = { version: 1, decorativeRoot: venue.name, drawCalls, instances,
      shadowLightsAdded: 0, collisionObjectsAdded: 0, playableBoundsChanged: false };
  }

  function prepare(scene, renderer) {
    if (states.has(scene)) return;
    const THREE = window.THREE;
    const state = { contacts: [], beacons: [], position: new THREE.Vector3(), shadowTexture: radialTexture(THREE) };
    states.set(scene, state);
    renderer.outputEncoding = THREE.sRGBEncoding;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.02;
    scene.background = new THREE.Color(0x0b1421);
    scene.fog = new THREE.FogExp2(0x0b1421, 0.024);

    const materials = new Set(), netMaps = new Map(), roots = new Set();
    scene.traverse(object => {
      if (object.name === 'drone-tail-lens') {
        state.beacons.push({ material: object.material, light: object.parent.getObjectByName('drone-tail-light') });
      }
      if (object.isAmbientLight) { object.color.setHex(0xc4d3e5); object.intensity = 0.18; }
      if (object.isHemisphereLight) {
        object.color.setHex(0xd5e7ff); object.groundColor.setHex(0x223b35); object.intensity = 0.42;
      }
      if (object.isSpotLight) {
        // Only the vertical overhead key casts shadows. Colored lights are fill.
        object.castShadow = false;
        if (object.parent === scene && object.position.y > 3.2) {
          object.intensity = 1.55; object.penumbra = 1; object.decay = 1.5;
          object.angle = Math.PI / 3;
        } else if (object.parent === scene) {
          object.color.setHex(0xfff3dd); object.intensity = 0.65; object.penumbra = 1;
        } else {
          object.intensity = 0.7;
        }
      }
      if (object.userData.perfMerge === 'head') object.scale.setScalar(0.48);
      if (object.userData.perfDrone) {
        let root = object;
        while (root.parent && root.parent !== scene) root = root.parent;
        roots.add(root);
      }
      if (!object.isMesh) return;
      const list = Array.isArray(object.material) ? object.material : [object.material];
      list.forEach(material => {
        if (materials.has(material)) return;
        materials.add(material);
        if (material.isMeshStandardMaterial) {
          material.envMapIntensity = 0.32;
          if (object.userData.perfMerge === 'tube') {
            material.color.convertSRGBToLinear();
            material.emissiveIntensity = 0.025;
            material.roughness = 0.57; material.metalness = 0.06;
          }
          if (object.userData.perfMerge === 'pole') {
            material.color.setHex(0x29394d).convertSRGBToLinear();
            material.metalness = 0.35; material.roughness = 0.46;
          }
          const p = object.geometry.parameters;
          if (p?.width === 3 && p?.height === 6 && material.map) tuneField(material, THREE);
          if (p?.width === 60 && p?.height === 60) {
            material.color.setHex(0x101b2b).convertSRGBToLinear();
            material.metalness = 0.08; material.roughness = 0.92;
            object.scale.set(4, 4, 1);
            // The venue now supplies the visible floor, below the raised stage.
            object.visible = false;
          }
        }
        // Dense, evenly sized protective netting on all four walls and roof.
        if (material.isMeshBasicMaterial && material.map && material.transparent && material.opacity === 0.38) {
          tuneNet(material, object.geometry, netMaps, renderer, THREE);
        }
      });
      if (object.userData.perfMerge === 'tube') object.receiveShadow = true;
    });

    // Position and target share X/Z: every physical shadow falls straight down.
    const key = new THREE.DirectionalLight(0xfff5e7, 1.25);
    key.name = 'arena-overhead-key'; key.position.set(0, 7, 0); key.target.position.set(0, 0, 0);
    key.castShadow = true;
    const size = matchMedia('(pointer: coarse)').matches ? 1024 : 2048;
    key.shadow.mapSize.set(size, size);
    Object.assign(key.shadow.camera, { left: -2.15, right: 2.15, top: 3.7, bottom: -3.7, near: 0.5, far: 9 });
    key.shadow.bias = -0.00015; key.shadow.normalBias = 0.0025;
    key.shadow.radius = 2;
    scene.add(key, key.target);
    const fill = new THREE.DirectionalLight(0xb3d5ff, 0.34);
    fill.name = 'arena-soft-fill'; fill.position.set(-4, 3, 2); scene.add(fill);

    // Soft grounding persists at every performance tier, centered on the drone.
    roots.forEach(root => {
      const material = new THREE.MeshBasicMaterial({ map: state.shadowTexture, transparent: true,
        depthWrite: false, opacity: 0.45, toneMapped: false, polygonOffset: true, polygonOffsetFactor: -1 });
      const mesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), material);
      mesh.name = 'drone-contact-shadow'; mesh.rotation.x = -Math.PI / 2; mesh.renderOrder = 1;
      scene.add(mesh); state.contacts.push({ root, mesh });
    });
    scene.userData.arenaLighting = { version: 1, shadowDirection: 'vertical', contactShadows: roots.size,
      shadowMapSize: size, shadowLights: 1 };
    buildVenue(scene, renderer, THREE);
    update(scene);
  }

  function update(scene) {
    const state = states.get(scene);
    if (!state) return;
    const now = performance.now();
    state.beacons.forEach(({ material, light }) => window.DroneAppearance.updateTailBeacon(material, light, now));
    state.contacts.forEach(({ root, mesh }) => {
      const position = root.getWorldPosition(state.position);
      const height = Math.max(0, position.y - 0.1);
      const diameter = 0.29 + height * 0.13;
      mesh.position.set(position.x, 0.006, position.z);
      mesh.scale.set(diameter, diameter, 1);
      mesh.material.opacity = 0.48 / (1 + height * 1.8);
      // A cockpit camera hides its own drone model; its ground shadow still exists.
      mesh.visible = Math.abs(position.x) < 1.5 && Math.abs(position.z) < 3;
      mesh.updateMatrixWorld(true);
    });
  }
  window.DroneArenaLighting = Object.freeze({ prepare, update });
})();
