/* Shared visual geometry only. Flight state, rotor animation and team colors stay with each game. */
(function (global) {
  'use strict';
  const THREE = global.THREE;
  const cache = new Map();
  const tailStates = new WeakMap();
  const INNER_SCALE = 0.23, ARM_LENGTH = 0.28, ARM_Y = -0.006;
  function cached(key, build) {
    if (!cache.has(key)) cache.set(key, build());
    return cache.get(key);
  }
  function standard(color, metalness = 0.35, roughness = 0.5) {
    return new THREE.MeshStandardMaterial({ color, metalness, roughness });
  }
  function carbon() {
    return cached('carbon-material', () => {
      const canvas = document.createElement('canvas'); canvas.width = canvas.height = 64;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#202833'; ctx.fillRect(0, 0, 64, 64);
      for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) {
        ctx.fillStyle = (x + y) % 2 ? '#293340' : '#151e28';
        ctx.fillRect(x * 8, y * 8, 7, 7);
        ctx.fillStyle = '#35404c';
        ctx.fillRect(x * 8 + 1, y * 8 + 1, (x + y) % 2 ? 1 : 5, (x + y) % 2 ? 5 : 1);
      }
      const map = new THREE.CanvasTexture(canvas); map.encoding = THREE.sRGBEncoding;
      map.wrapS = map.wrapT = THREE.RepeatWrapping; map.repeat.set(3, 3);
      const material = standard(0xc6d3df, 0.2, 0.64); material.map = map;
      return material;
    });
  }
  function mesh(geometry, material, x = 0, y = 0, z = 0) {
    const result = new THREE.Mesh(geometry, material); result.position.set(x, y, z);
    result.castShadow = true; result.receiveShadow = true; return result;
  }
  function box(x, y, z) { return new THREE.BoxGeometry(x, y, z); }
  function cylinder(r, h, segments = 12) { return new THREE.CylinderGeometry(r, r, h, segments); }
  function plate(width, length, depth, bevel = 0.012) {
    const x = width / 2, z = length / 2, shape = new THREE.Shape();
    shape.moveTo(-x + bevel, -z); shape.lineTo(x - bevel, -z); shape.lineTo(x, -z + bevel);
    shape.lineTo(x, z - bevel); shape.lineTo(x - bevel, z); shape.lineTo(-x + bevel, z);
    shape.lineTo(-x, z - bevel); shape.lineTo(-x, -z + bevel); shape.closePath();
    const geometry = new THREE.ExtrudeGeometry(shape, { depth, bevelEnabled: true,
      bevelSegments: 1, steps: 1, bevelSize: 0.003, bevelThickness: 0.003, curveSegments: 6 });
    geometry.rotateX(-Math.PI / 2); geometry.translate(0, -depth / 2, 0); return geometry;
  }
  function strut(a, b, radius, segments = 6) {
    const direction = new THREE.Vector3().subVectors(b, a);
    const geometry = cylinder(radius, direction.length(), segments);
    geometry.applyMatrix4(new THREE.Matrix4().makeRotationFromQuaternion(
      new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.normalize())));
    geometry.translate((a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2); return geometry;
  }
  function combine(parts) {
    const normalized = parts.map(part => part.index ? part.toNonIndexed() : part.clone());
    const result = THREE.BufferGeometryUtils.mergeBufferGeometries(normalized, false);
    normalized.forEach(part => part.dispose());
    return result;
  }
  // Merge by material once, cache immutable geometry, and keep moving rotors separate.
  function batch(group) {
    const bins = new Map();
    group.children.slice().forEach(child => {
      if (!child.isMesh) return;
      child.updateMatrix();
      if (!bins.has(child.material)) bins.set(child.material, []);
      bins.get(child.material).push(child.geometry.clone().applyMatrix4(child.matrix)); group.remove(child);
      child.geometry.dispose();
    });
    bins.forEach((geometries, material) => {
      const merged = combine(geometries);
      geometries.forEach(geometry => geometry.dispose());
      group.add(mesh(merged, material));
    });
    return group;
  }
  function soccerCageLayout(radius) {
    return cached(`soccer-cage-layout-${radius}`, () => {
      // Truncate each icosahedron corner at one third of its edges: the classic
      // football has 20 hexagons and 12 pentagons, with no diagonals or hoops.
      const ico = new THREE.IcosahedronGeometry(1, 0);
      const source = ico.attributes.position, corners = [], triangles = [], lookup = new Map();
      for (let i = 0; i < source.count; i++) {
        const point = new THREE.Vector3().fromBufferAttribute(source, i);
        const key = point.toArray().map(value => value.toFixed(5)).join(',');
        if (!lookup.has(key)) { lookup.set(key, corners.length); corners.push(point); }
        triangles.push(lookup.get(key));
      }
      ico.dispose();
      const vertices = [], panels = [], edges = [], directed = new Map(), seenEdges = new Set();
      const neighbors = corners.map(() => new Set());
      function cut(a, b) {
        const key = `${a},${b}`;
        if (!directed.has(key)) {
          directed.set(key, vertices.length);
          vertices.push(corners[a].clone().multiplyScalar(2).add(corners[b]).normalize().multiplyScalar(radius * 0.98));
        }
        return directed.get(key);
      }
      for (let i = 0; i < triangles.length; i += 3) {
        const [a, b, c] = triangles.slice(i, i + 3);
        for (const [u, v] of [[a, b], [b, c], [c, a]]) { neighbors[u].add(v); neighbors[v].add(u); }
        panels.push([cut(a, b), cut(b, a), cut(b, c), cut(c, b), cut(c, a), cut(a, c)]);
      }
      corners.forEach((corner, a) => {
        const normal = corner.clone().normalize();
        const adjacent = [...neighbors[a]];
        const tangent = corners[adjacent[0]].clone().projectOnPlane(normal).normalize();
        const bitangent = new THREE.Vector3().crossVectors(normal, tangent);
        adjacent.sort((b, c) => Math.atan2(corners[b].dot(bitangent), corners[b].dot(tangent))
          - Math.atan2(corners[c].dot(bitangent), corners[c].dot(tangent)));
        panels.push(adjacent.map(b => cut(a, b)));
      });
      panels.forEach(panel => panel.forEach((a, i) => {
        const b = panel[(i + 1) % panel.length], key = [Math.min(a, b), Math.max(a, b)].join(',');
        if (!seenEdges.has(key)) { seenEdges.add(key); edges.push([a, b]); }
      }));
      const segments = [];
      edges.forEach(([a, b]) => {
        let previous = vertices[a];
        for (let step = 1; step <= 4; step++) {
          const next = vertices[a].clone().lerp(vertices[b], step / 4).normalize().multiplyScalar(radius * 0.98);
          segments.push([previous, next]); previous = next;
        }
      });
      return { vertices, panels, edges, segments };
    });
  }
  function frameMounts(radius = 0.1) {
    return cached(`frame-mounts-${radius}`, () => {
      const layout = soccerCageLayout(radius);
      const height = ARM_Y * INNER_SCALE;
      const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -height);
      return [[1, 1], [-1, 1], [1, -1], [-1, -1]].map(([x, z]) => {
        const target = new THREE.Vector3(x, 0, z).normalize().multiplyScalar(radius * 0.98); target.y = height;
        const candidates = layout.segments.map(([a, b]) => plane.intersectLine(new THREE.Line3(a, b), new THREE.Vector3()))
          .filter(point => point && point.x * x > 0 && point.z * z > 0);
        const cage = candidates.reduce((closest, point) => point.distanceToSquared(target) < closest.distanceToSquared(target) ? point : closest);
        // One radial axis drives the black arm, motor/rotor and colored stay.
        // The motor remains at the original distance from the center.
        const motor = new THREE.Vector3(cage.x, 0, cage.z).normalize().multiplyScalar(ARM_LENGTH); motor.y = ARM_Y;
        return { motor, cage };
      });
    });
  }
  function createCage(radius, colorHex) {
    const layout = soccerCageLayout(radius);
    const geometry = cached(`cage-${radius}`, () => {
      const parts = [];
      // Follow the sphere gently along each panel edge; all junctions still have
      // just three struts, so the six-sided openings remain clearly readable.
      layout.segments.forEach(([a, b]) => parts.push(strut(a, b, radius * 0.018, 6)));
      const result = combine(parts);
      parts.forEach(part => part.dispose());
      result.userData.soccerPanels = {
        vertices: layout.vertices.map(point => point.toArray()),
        panels: layout.panels, edges: layout.edges
      };
      return result;
    });
    const material = standard(colorHex, 0, 0.86);
    material.emissive.setHex(colorHex); material.emissiveIntensity = 0.12;
    material.toneMapped = false;
    // Games keep UI hex colors in color/emissive and update them on team changes.
    // Decode those sRGB values in the shader, avoiding washed-out pink/cyan cages
    // under the arena's sRGB output while retaining the existing material refs.
    material.onBeforeCompile = shader => {
      shader.fragmentShader = shader.fragmentShader
        .replace('#include <color_fragment>', '#include <color_fragment>\n diffuseColor.rgb = sRGBToLinear(diffuseColor).rgb;')
        .replace('#include <emissivemap_fragment>', '#include <emissivemap_fragment>\n totalEmissiveRadiance = sRGBToLinear(vec4(totalEmissiveRadiance / 0.12, 1.0)).rgb * 0.12;');
    };
    material.customProgramCacheKey = () => 'drone-cage-srgb-v1';
    const group = new THREE.Group(); group.name = 'drone-protective-cage'; group.userData.perfDrone = true;
    group.add(mesh(geometry, material));
    const staysGeometry = cached(`cage-stays-${radius}`, () => {
      const parts = [], mounts = [], stayRadius = radius * 0.028;
      // Begin at the motor seat on the black arm's exact centerline, rather
      // than crossing the arm at an independently chosen angle and height.
      frameMounts(radius).forEach(({ motor, cage: to }) => {
        const from = motor.clone().multiplyScalar(INNER_SCALE);
        parts.push(strut(from, to, stayRadius, 8));
        const collar = new THREE.SphereGeometry(radius * 0.019, 8, 5);
        collar.translate(to.x, to.y, to.z); parts.push(collar);
        mounts.push({ from: from.toArray(), to: to.toArray() });
      });
      const result = combine(parts); parts.forEach(part => part.dispose());
      result.userData.cageMounts = mounts; result.userData.stayRadius = stayRadius;
      return result;
    });
    const stays = mesh(staysGeometry, material); stays.name = 'drone-cage-stays'; group.add(stays);
    const fittings = cached(`cage-fittings-${radius}`, () => {
      const parts = [];
      layout.vertices.forEach(point => {
        const fitting = new THREE.SphereGeometry(radius * 0.019, 8, 5);
        fitting.translate(point.x, point.y, point.z); parts.push(fitting);
      });
      const result = combine(parts);
      parts.forEach(part => part.dispose()); return result;
    });
    group.add(mesh(fittings, cached('fitting-material', () => standard(0x253345, 0.65, 0.38))));
    group.userData.appearanceVersion = 5;
    return { group, material };
  }
  function core() {
    return cached('core', () => {
      const group = new THREE.Group(), composite = carbon();
      const graphite = standard(0x172332, 0.45, 0.48), metal = standard(0xa8bacb, 0.72, 0.29);
      const copper = standard(0xba7942, 0.65, 0.4), rubber = standard(0x080d15, 0.02, 0.9);
      const circuit = standard(0x154c42, 0.18, 0.6);
      group.add(mesh(plate(0.25, 0.255, 0.014), composite, 0, -0.035, 0));
      group.add(mesh(plate(0.22, 0.23, 0.012), composite, 0, 0.025, 0));
      group.add(mesh(plate(0.137, 0.17, 0.011), circuit, 0, 0.001, 0));
      group.add(mesh(plate(0.125, 0.16, 0.046), graphite, 0, -0.073, -0.008));
      for (const x of [-0.045, 0.045]) group.add(mesh(box(0.018, 0.052, 0.171), rubber, x, -0.073, -0.008));
      for (const x of [-0.09, 0.09]) for (const z of [-0.09, 0.09]) {
        group.add(mesh(cylinder(0.007, 0.052, 8), copper, x, -0.005, z));
        group.add(mesh(cylinder(0.01, 0.004, 8), metal, x, 0.035, z));
        group.add(mesh(box(0.009, 0.001, 0.002), rubber, x, 0.0375, z));
      }
      for (let i = 0; i < 5; i++) {
        group.add(mesh(box(0.013, 0.006, 0.021), graphite, -0.042 + i * 0.021, 0.012, -0.06));
        group.add(mesh(box(0.008, 0.007, 0.008), metal, -0.04 + i * 0.02, 0.012, 0.056));
      }
      // Camera barrel, dark optical face and a recessed turquoise lens (forward is +Z).
      const barrel = mesh(cylinder(0.029, 0.034, 20), graphite, 0, 0.014, 0.133);
      barrel.rotation.x = Math.PI / 2; group.add(barrel);
      const rim = mesh(new THREE.TorusGeometry(0.023, 0.004, 6, 20), metal, 0, 0.014, 0.151); group.add(rim);
      const optic = mesh(cylinder(0.019, 0.003, 20), rubber, 0, 0.014, 0.153);
      optic.rotation.x = Math.PI / 2; group.add(optic);
      const glass = standard(0x165c70, 0.65, 0.13);
      const lens = mesh(new THREE.SphereGeometry(0.014, 16, 8), glass, 0, 0.014, 0.154);
      lens.scale.z = 0.3; group.add(lens);
      const glint = mesh(new THREE.SphereGeometry(0.003, 6, 4), metal, -0.004, 0.019, 0.158); group.add(glint);
      for (const x of [-0.043, 0.043]) for (let i = 0; i < 4; i++) {
        group.add(mesh(box(0.014, 0.004, 0.006), rubber, x, 0.072, -0.045 + i * 0.017));
      }
      frameMounts().forEach(({ motor }) => {
        const px = motor.x, pz = motor.z;
        const arm = mesh(box(0.032, 0.018, ARM_LENGTH), composite, px / 2, ARM_Y, pz / 2);
        arm.rotation.y = Math.atan2(px, pz); group.add(arm);
        group.add(mesh(strut(new THREE.Vector3(px * 0.36, 0.007, pz * 0.36), new THREE.Vector3(px, 0.007, pz), 0.003, 5), copper));
        group.add(mesh(cylinder(0.035, 0.058, 16), graphite, px, 0.016, pz));
        group.add(mesh(cylinder(0.036, 0.006, 16), copper, px, 0.042, pz));
        group.add(mesh(cylinder(0.036, 0.006, 16), metal, px, -0.012, pz));
        for (let i = 0; i < 8; i++) {
          const angle = i * Math.PI / 4;
          const slot = mesh(box(0.004, 0.027, 0.006), rubber, px + Math.cos(angle) * 0.034, 0.02, pz + Math.sin(angle) * 0.034);
          slot.rotation.y = -angle; group.add(slot);
        }
        group.add(mesh(cylinder(0.014, 0.008, 10), metal, px, 0.051, pz));
      });
      // The four horizontal stays are built with the cage, using its real lattice
      // and mutable team material instead of this graphite frame.
      return batch(group);
    });
  }
  function rotorGeometry() {
    return cached('rotor-geometry', () => {
      const shape = new THREE.Shape();
      shape.moveTo(0.008, -0.004); shape.bezierCurveTo(0.035, -0.014, 0.074, -0.02, 0.104, -0.01);
      shape.quadraticCurveTo(0.115, -0.002, 0.101, 0.006);
      shape.bezierCurveTo(0.072, 0.012, 0.035, 0.004, 0.008, 0.004); shape.closePath();
      const blade = new THREE.ExtrudeGeometry(shape, { depth: 0.003, bevelEnabled: false, curveSegments: 8 });
      blade.rotateX(-Math.PI / 2);
      const parts = [0, 1, 2].map(i => blade.clone().rotateY(i * Math.PI * 2 / 3));
      parts.push(cylinder(0.015, 0.009, 12));
      const result = combine(parts);
      parts.forEach(part => part.dispose()); blade.dispose(); return result;
    });
  }
  function createBody(colorHex) {
    const group = new THREE.Group(); group.name = 'drone-flight-mechanism'; group.userData.appearanceVersion = 1;
    const fixed = core().clone(); fixed.name = 'drone-machined-frame'; group.add(fixed);
    const canopyMat = standard(colorHex, 0.23, 0.38);
    canopyMat.emissive.setHex(colorHex); canopyMat.emissiveIntensity = 0.035;
    const canopy = mesh(cached('canopy', () => plate(0.113, 0.15, 0.033, 0.026)), canopyMat, 0, 0.05, 0.007);
    canopy.name = 'drone-canopy'; group.add(canopy);
    const rotorMat = cached('rotor-material', () => standard(0xc5d1db, 0.18, 0.55));
    const green = cached('nav-green', () => new THREE.MeshBasicMaterial({ color: 0x32e1b1 }));
    const red = cached('nav-red', () => new THREE.MeshBasicMaterial({ color: 0xf45460 }));
    const rotors = [], lights = [];
    frameMounts().forEach(({ motor }, index) => {
      const px = motor.x, pz = motor.z, z = Math.sign(pz);
      const rotor = new THREE.Group(); rotor.name = `drone-rotor-${index}`;
      rotor.position.set(px, 0.059, pz); rotor.rotation.y = index * 0.7;
      rotor.add(mesh(rotorGeometry(), rotorMat)); group.add(rotor); rotors.push(rotor);
      const light = mesh(cached('nav-lens', () => box(0.022, 0.009, 0.012)), z > 0 ? green : red, px, -0.02, pz);
      light.name = 'drone-navigation-light'; group.add(light);
      lights.push({ mesh: light, originalColor: z > 0 ? 0x32e1b1 : 0xf45460 });
    });
    // Light animation changes material colors; each aircraft retains its own pair.
    const navMaterials = new Map();
    lights.forEach(light => {
      const original = light.mesh.material;
      if (!navMaterials.has(original)) navMaterials.set(original, original.clone());
      light.mesh.material = navMaterials.get(original);
    });
    return { group, canopy, rotors, lights };
  }
  function beaconHaloTexture() {
    return cached('beacon-halo-texture', () => {
      const canvas = document.createElement('canvas'); canvas.width = canvas.height = 256;
      const ctx = canvas.getContext('2d'), pixels = ctx.createImageData(256, 256);
      // Continuous falloff, with zero alpha at the edge: no nested sphere silhouettes.
      for (let y = 0; y < 256; y++) for (let x = 0; x < 256; x++) {
        const r = Math.hypot((x - 127.5) / 127.5, (y - 127.5) / 127.5);
        const alpha = r < 1 ? Math.exp(-7 * r * r) * (1 - r * r) ** 2 : 0;
        const i = (y * 256 + x) * 4;
        pixels.data[i] = pixels.data[i + 1] = pixels.data[i + 2] = 255;
        pixels.data[i + 3] = Math.round(alpha * 255);
      }
      ctx.putImageData(pixels, 0, 0);
      return new THREE.CanvasTexture(canvas);
    });
  }
  function createTailBeacon() {
    const group = new THREE.Group(); group.name = 'drone-tail-beacon';
    const graphite = cached('beacon-housing-material', () => standard(0x172332, 0.58, 0.34));
    const metal = cached('beacon-bezel-material', () => standard(0x8194a8, 0.78, 0.26));
    // A shallow rear panel sits at the PCB edge, inside the carbon frame.
    // Rounded faces replace the former stalk and projecting cylindrical lamp.
    function panel(key, width, height, depth, radius) {
      return cached(key, () => {
        const shape = new THREE.Shape(), x = width / 2, y = height / 2;
        shape.moveTo(-x + radius, -y); shape.lineTo(x - radius, -y);
        shape.quadraticCurveTo(x, -y, x, -y + radius); shape.lineTo(x, y - radius);
        shape.quadraticCurveTo(x, y, x - radius, y); shape.lineTo(-x + radius, y);
        shape.quadraticCurveTo(-x, y, -x, y - radius); shape.lineTo(-x, -y + radius);
        shape.quadraticCurveTo(-x, -y, -x + radius, -y);
        const geometry = new THREE.ExtrudeGeometry(shape, { depth, bevelEnabled: true,
          bevelSize: 0.00035, bevelThickness: 0.00025, bevelSegments: 2, steps: 1, curveSegments: 6 });
        geometry.translate(0, 0, -depth / 2); return geometry;
      });
    }
    const housing = mesh(panel('rear-board-housing', 0.021, 0.008, 0.006, 0.0018), graphite);
    housing.name = 'drone-rear-board-housing'; group.add(housing);
    const flange = mesh(cached('rear-board-flange', () => box(0.018, 0.0015, 0.008)), graphite, 0, -0.002, 0.0035);
    group.add(flange);
    group.add(mesh(panel('rear-board-bezel', 0.0175, 0.0057, 0.0008, 0.0012), metal, 0, 0, -0.0033));
    const screwGeometry = cached('rear-board-screw', () => cylinder(0.0009, 0.0007, 8));
    for (const x of [-0.0091, 0.0091]) {
      const screw = mesh(screwGeometry, metal, x, 0, -0.00335); screw.rotation.x = Math.PI / 2; group.add(screw);
    }
    const beaconMat = new THREE.MeshBasicMaterial({ color: 0xff2438, toneMapped: false });
    const lens = mesh(panel('rear-board-lens', 0.0148, 0.0037, 0.001, 0.0011), beaconMat, 0, 0, -0.0039);
    lens.name = 'drone-tail-lens';
    lens.castShadow = lens.receiveShadow = false; group.add(lens);
    const coreMat = new THREE.MeshBasicMaterial({ color: 0xffe3e6, toneMapped: false });
    const emitter = mesh(panel('rear-board-emitter', 0.0108, 0.0011, 0.0003, 0.00045), coreMat, 0, 0, -0.00465);
    emitter.name = 'drone-tail-emitter';
    emitter.castShadow = emitter.receiveShadow = false; group.add(emitter);
    const haloMat = new THREE.SpriteMaterial({ map: beaconHaloTexture(), color: 0xff2438,
      transparent: true, opacity: 0.48, blending: THREE.AdditiveBlending,
      depthWrite: false, depthTest: true, toneMapped: false });
    const halo = new THREE.Sprite(haloMat); halo.name = 'drone-tail-soft-glow';
    halo.position.set(0, 0, -0.0053); halo.scale.set(0.036, 0.022, 1); group.add(halo);
    haloMat.userData.baseOpacity = 0.48;
    beaconMat.userData.haloMats = [haloMat]; beaconMat.userData.coreMat = coreMat;
    const beaconLight = new THREE.PointLight(0xff2438, 0.65, 0.24, 2);
    beaconLight.name = 'drone-tail-light';
    beaconLight.position.copy(lens.position); group.add(beaconLight);
    tailStates.set(beaconMat, { actor: null, blinking: false, startedAt: 0 });
    setTailBeaconColor(beaconMat, beaconLight, 0xff2438);
    return { group, beaconMat, beaconLight };
  }
  function setTailBeaconColor(beaconMat, beaconLight, colorHex, boost = 1) {
    if (!beaconMat) return;
    // The arena renders to sRGB; decode the UI hex once to preserve saturated team hues.
    const color = new THREE.Color(colorHex).convertSRGBToLinear();
    beaconMat.userData.baseColor = color; beaconMat.color.copy(color);
    // Keep the bright center small so the surrounding lens still reads as team color.
    beaconMat.userData.coreColor = color.clone().lerp(new THREE.Color(0xffffff), 0.78);
    const gain = Math.min(1.35, Math.sqrt(Math.max(1, boost)));
    beaconMat.userData.haloMats.forEach(material => {
      material.color.copy(color); material.opacity = Math.min(0.95, material.userData.baseOpacity * gain);
    });
    if (beaconLight) {
      beaconLight.color.copy(color); beaconLight.userData.boost = gain;
      beaconLight.intensity = 0.65 * gain;
    }
    updateTailBeacon(beaconMat, beaconLight);
  }
  function bindTailBeacon(beaconMat, actor) {
    const state = tailStates.get(beaconMat);
    if (state) state.actor = actor;
  }
  function updateTailBeacon(beaconMat, beaconLight, now = performance.now()) {
    if (!beaconMat) return;
    const state = tailStates.get(beaconMat), actor = state?.actor;
    // Takeoff commands count before the aircraft leaves the ground. Manual USB
    // throttle and airborne motion also count, including the full landing phase.
    const active = !!actor && !window.DroneFlightSafety?.isStopped(actor) && (actor.autoFlightState === 'takingoff' || actor.altitudeState === 'takingoff'
      || actor.position.y > 0.115 || (actor.altitudeState !== 'grounded' && (actor.input?.throttle || 0) > 0.01));
    let level = 1;
    if (state) {
      if (active && !state.blinking) state.startedAt = now;
      state.blinking = active;
      if (active) {
        const phase = Math.max(0, now - state.startedAt) % 500;
        const smooth = x => x * x * (3 - 2 * x);
        level = phase < 180 ? 1 : phase < 230 ? 1 - 0.9 * smooth((phase - 180) / 50)
          : phase < 450 ? 0.1 : 0.1 + 0.9 * smooth((phase - 450) / 50);
      }
    }
    const gain = beaconLight?.userData.boost || 1;
    beaconMat.color.copy(beaconMat.userData.baseColor).multiplyScalar(level);
    beaconMat.userData.coreMat.color.copy(beaconMat.userData.coreColor).multiplyScalar(level);
    beaconMat.userData.haloMats.forEach(material => {
      material.opacity = Math.min(0.95, material.userData.baseOpacity * gain) * level;
    });
    if (beaconLight) beaconLight.intensity = 0.65 * gain * level;
    // Some modes do not keep a glowRef. Resolve the visible belly lamp from the
    // finished aircraft once, so player and AI models share this exact phase/hue.
    if (actor?.mesh && state.bellyRoot !== actor.mesh) {
      state.bellyRoot = actor.mesh;
      state.bellyMaterial = actor.mesh.getObjectByName('drone-belly-light')?.material;
      state.bellySpot = actor.mesh.children.find(child => child.isSpotLight);
      if (state.bellyMaterial && state.bellyMaterial.toneMapped) {
        state.bellyMaterial.toneMapped = false;
        state.bellyMaterial.needsUpdate = true;
      }
      if (state.bellySpot) state.bellySpot.name = 'drone-belly-spotlight';
    }
    if (state?.bellyMaterial) state.bellyMaterial.color.copy(beaconMat.userData.baseColor).multiplyScalar(level);
    if (state?.bellySpot) {
      state.bellySpot.color.copy(beaconMat.userData.baseColor);
      // Preserve the arena's existing 0.7 belly fill; only hue and pulse change.
      state.bellySpot.intensity = 0.7 * level;
    }
  }
  global.DroneAppearance = Object.freeze({ createCage, createBody, createTailBeacon,
    groundShape: (radius = 0.1) => ({ points: soccerCageLayout(radius).segments.flatMap(edge => edge.map(p => p.clone())), tubeRadius: radius * 0.018 }),
    setTailBeaconColor, bindTailBeacon, updateTailBeacon, REAR_LIGHT_Y: 0.001, REAR_LIGHT_Z: -0.026,
    BELLY_LIGHT_Y: -0.0265, BELLY_LIGHT_RADIUS: 0.0045 });
})(window);
