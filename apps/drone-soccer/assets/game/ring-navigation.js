/* Visual guidance reads the production gate order and crossing plane only. */
(function (root) {
    'use strict';
    function create({ kind = 'ring', halfSize = 0.3 } = {}) {
        const THREE = root.THREE;
        const isGrid = kind === 'grid';
        const unit = isGrid ? '格' : '環';
        const panel = document.createElement('aside');
        panel.id = 'ringNavigation'; panel.hidden = true;
        panel.dataset.kind = kind;
        panel.setAttribute('aria-label', isGrid ? '九宮格穿越導航' : '穿環導航');
        panel.innerHTML = '<div class="ring-eyebrow"><i class="ring-light"></i>下一個穿越目標</div>' +
            '<div class="ring-heading"><strong></strong><span class="ring-count"></span></div>' +
            (isGrid ? '<p class="ring-crossing"></p><p class="ring-reference">前側＝起飛側</p>' : '') +
            '<p class="ring-direction"></p><div class="ring-meta"><span class="ring-distance"></span><span class="ring-progress"></span></div>' +
            '<div class="ring-track"><span></span></div>';
        const label = document.createElement('div');
        label.id = 'ringTargetLabel'; label.hidden = true;
        label.innerHTML = '<i></i><span></span>';
        // Announce target changes without announcing every distance update.
        const heading = panel.querySelector('strong');
        heading.setAttribute('role', 'status'); heading.setAttribute('aria-live', 'polite');
        document.body.append(panel, label);
        const q = selector => panel.querySelector(selector);
        const count = q('.ring-count'), direction = q('.ring-direction');
        if (isGrid) { direction.setAttribute('role', 'status'); direction.setAttribute('aria-live', 'polite'); }
        const distance = q('.ring-distance'), progress = q('.ring-progress'), bar = q('.ring-track span');
        const group = new THREE.Group(); group.name = isGrid ? 'active-grid-guidance' : 'active-ring-guidance'; group.visible = false;
        function outlineGeometry(radius, tube) {
            if (!isGrid) return new THREE.TorusGeometry(radius, tube, 8, 64);
            // Follow the square beams, with a face visible on either side of the grid.
            const outer = radius + tube, inner = radius - tube;
            const shape = new THREE.Shape();
            shape.moveTo(-outer, -outer); shape.lineTo(outer, -outer);
            shape.lineTo(outer, outer); shape.lineTo(-outer, outer); shape.closePath();
            const hole = new THREE.Path();
            hole.moveTo(-inner, -inner); hole.lineTo(-inner, inner);
            hole.lineTo(inner, inner); hole.lineTo(inner, -inner); hole.closePath();
            shape.holes.push(hole);
            return new THREE.ExtrudeGeometry(shape, { depth: 0.07, bevelEnabled: false, steps: 1 }).translate(0, 0, -0.035);
        }
        const layers = [0.009, 0.023, 0.045].map((tube, i) => {
            const material = new THREE.MeshBasicMaterial({ color: 0xffbd45, transparent: true,
                opacity: [0.85, 0.16, 0.045][i], depthWrite: false, toneMapped: false,
                blending: i === 0 ? THREE.NormalBlending : THREE.AdditiveBlending });
            material.color.convertSRGBToLinear();
            const mesh = new THREE.Mesh(outlineGeometry(1, tube), material);
            group.add(mesh); return mesh;
        });
        // A bevelled dart and two travelling chevrons read as a flight cue.
        // Local +Z remains the exact crossing direction used by the gate.
        const arrow = new THREE.Group(); arrow.name = 'ring-direction-arrow';
        const ink = (color, opacity = 1) => {
            const material = new THREE.MeshBasicMaterial({ color, transparent: opacity < 1,
                opacity, depthWrite: opacity === 1, toneMapped: false, side: THREE.DoubleSide });
            material.color.convertSRGBToLinear();
            return material;
        };
        const face = ink(0xffd36c), edge = ink(0xb97a26), highlight = ink(0xfff2c8);
        const dartShape = new THREE.Shape();
        [[-0.027, -0.22], [0.027, -0.22], [0.038, 0.035], [0.113, -0.015],
            [0, 0.25], [-0.113, -0.015], [-0.038, 0.035]].forEach(([x, y], i) => {
            if (i === 0) dartShape.moveTo(x, y); else dartShape.lineTo(x, y);
        });
        dartShape.closePath();
        const dart = new THREE.Mesh(new THREE.ExtrudeGeometry(dartShape, {
            depth: 0.026, bevelEnabled: true, bevelSegments: 2, steps: 1,
            bevelSize: 0.006, bevelThickness: 0.006
        }), [face, edge]);
        dart.rotation.x = Math.PI / 2; dart.position.y = 0.013;
        arrow.add(dart);
        // A smaller upright fin keeps the arrow legible from a low side view.
        const finShape = new THREE.Shape();
        finShape.moveTo(-0.065, -0.015); finShape.lineTo(0, 0.25);
        finShape.lineTo(0.065, -0.015); finShape.lineTo(0, 0.035); finShape.closePath();
        const fin = new THREE.Mesh(new THREE.ExtrudeGeometry(finShape, {
            depth: 0.012, bevelEnabled: true, bevelSegments: 1, steps: 1,
            bevelSize: 0.004, bevelThickness: 0.004
        }), [face, edge]);
        fin.rotation.set(Math.PI / 2, Math.PI / 2, 0);
        fin.position.x = -0.006;
        arrow.add(fin);
        const inlay = new THREE.Mesh(new THREE.ShapeGeometry(dartShape), highlight);
        inlay.scale.set(0.46, 0.81, 1); inlay.rotation.x = Math.PI / 2;
        inlay.position.set(0, 0.021, 0.007); arrow.add(inlay);
        const chevronShape = new THREE.Shape();
        [[-0.087, -0.047], [0, 0.028], [0.087, -0.047], [0.087, -0.014],
            [0, 0.061], [-0.087, -0.014]].forEach(([x, y], i) => {
            if (i === 0) chevronShape.moveTo(x, y); else chevronShape.lineTo(x, y);
        });
        chevronShape.closePath();
        const chevronGeometry = new THREE.ExtrudeGeometry(chevronShape, {
            depth: 0.012, bevelEnabled: false, steps: 1
        });
        const chevrons = [0, 1].map(i => {
            const mesh = new THREE.Mesh(chevronGeometry, ink(0xffd36c, 0.65));
            mesh.rotation.x = Math.PI / 2; mesh.position.z = -0.34 - i * 0.16;
            arrow.add(mesh); return mesh;
        });
        group.add(arrow);
        const projected = new THREE.Vector3(), view = new THREE.Vector3(), relative = new THREE.Vector3();
        const gridForward = new THREE.Vector3(), up = new THREE.Vector3(0, 1, 0);
        let active = null, lastGates = null, lastIndex = -1, lastState = '', lastTop = 0;
        const reducedMotion = root.matchMedia('(prefers-reduced-motion: reduce)');
        const setText = (node, value) => { if (node.textContent !== value) node.textContent = value; };
        function redraw(gate, color) {
            const data = gate.numberSprite.userData, ctx = data.ctx, size = data.canvas.width;
            ctx.clearRect(0, 0, size, size);
            ctx.font = `bold ${Math.round(size * 0.78)}px sans-serif`;
            ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.lineJoin = 'round';
            ctx.lineWidth = size * 0.10; ctx.strokeStyle = '#07101de0';
            ctx.strokeText(String(data.number), size / 2, size * 0.54);
            ctx.lineWidth = size * 0.055; ctx.strokeStyle = '#ffffff';
            ctx.strokeText(String(data.number), size / 2, size * 0.54);
            ctx.fillStyle = color; ctx.fillText(String(data.number), size / 2, size * 0.54);
            data.texture.needsUpdate = true;
        }
        function update({ gates, index, state, scene, camera, drone }) {
            if (group.parent !== scene) scene.add(group);
            const target = gates[index];
            const visible = !!target && (state === 'idle' || state === 'racing');
            panel.hidden = label.hidden = !visible; group.visible = visible;
            if (gates !== lastGates || index !== lastIndex || state !== lastState) {
                gates.forEach((gate, i) => {
                    const isTarget = visible && i === index;
                    const color = gate.passed ? 0x10b981 : isTarget ? 0xffd36c : 0x64748b;
                    if (!isGrid) {
                        gate.mesh.material.color.setHex(color);
                        gate.mesh.material.emissive.setHex(color);
                        gate.mesh.material.emissiveIntensity = isTarget ? 0.5 : gate.passed ? 0.28 : 0.06;
                    }
                    redraw(gate, gate.passed ? '#10b981' : isTarget ? '#ffda7c' : isGrid ? (gate.dir === 1 ? '#ef4444' : '#f97316') : '#9aadc3');
                });
                active = target; lastGates = gates; lastIndex = index; lastState = state;
                if (visible) {
                    group.position.copy(target.posVec);
                    if (isGrid) group.quaternion.setFromAxisAngle(up, target.dir === 1 ? 0 : Math.PI);
                    else group.quaternion.copy(target.mesh.quaternion);
                    const radius = isGrid ? halfSize : target.r;
                    layers.forEach((mesh, i) => {
                        mesh.geometry.dispose();
                        mesh.geometry = outlineGeometry(radius + (isGrid ? 0 : 0.033), [0.009, 0.023, 0.045][i]);
                    });
                    // Runs beneath the opening, along the same normal used by scoring.
                    arrow.position.set(0, isGrid ? -radius * 0.65 : -radius - 0.105, 0);
                    arrow.scale.setScalar(THREE.MathUtils.clamp(radius / 0.32, 0.78, 1.12));
                    setText(heading, `第 ${index + 1} ${unit}`);
                    setText(count, `共 ${gates.length} ${unit}`);
                    if (isGrid) setText(q('.ring-crossing'), `穿越方向：${target.dir === 1 ? '前 → 後' : '後 → 前'}`);
                }
            }
            if (!visible) return;
            const now = performance.now();
            const pulse = reducedMotion.matches ? 0.5 : 0.5 + 0.5 * Math.sin(now * 0.0025);
            if (!isGrid) active.mesh.material.emissiveIntensity = 0.25 + pulse * 0.18;
            layers[1].material.opacity = 0.10 + pulse * 0.06;
            layers[2].material.opacity = 0.025 + pulse * 0.025;
            chevrons.forEach((mesh, i) => {
                const travel = reducedMotion.matches ? i * 0.5 + 0.25 : (now * 0.00065 + i * 0.5) % 1;
                mesh.position.z = -0.60 + travel * 0.32;
                mesh.material.opacity = reducedMotion.matches ? 0.65 : 0.18 + Math.sin(travel * Math.PI) * 0.62;
            });
            relative.copy(drone.position).sub(target.posVec);
            const wrongSide = relative.dot(isGrid ? gridForward.set(0, 0, target.dir) : target.frame.forward) > 0.045;
            panel.dataset.wrongSide = String(wrongSide);
            const entrance = isGrid && target.dir === 1 ? '前側' : '後側';
            setText(direction, isGrid
                ? (wrongSide ? `目前在出口側，請繞到${entrance}入口` : `從${entrance}入口，對準金色宮格穿越`)
                : (wrongSide ? '繞到光環另一側，再順箭頭穿越' : '對準金色光環，順箭頭穿越'));
            setText(distance, `距離 ${relative.length().toFixed(1)} m`);
            setText(progress, `已通過 ${gates.filter(g => g.passed).length} / ${gates.length}`);
            bar.style.width = `${gates.filter(g => g.passed).length / gates.length * 100}%`;
            // Respect the responsive toolbar, including wrapped mobile controls.
            const header = document.querySelector('.fc-hud > header');
            const top = Math.round((header ? header.getBoundingClientRect().bottom : 80) + 10);
            if (top !== lastTop) { panel.style.top = `${top}px`; lastTop = top; }
            // Determine visibility from the opening centre. Its upper label
            // anchor may leave the screen while the opening is still visible.
            view.copy(target.posVec).applyMatrix4(camera.matrixWorldInverse);
            projected.copy(target.posVec).project(camera);
            const behind = view.z >= 0;
            let x = projected.x, y = -projected.y;
            if (behind) { x = -x; y = -y; }
            if (!Number.isFinite(x)) x = Math.sign(view.x) || 0.001;
            if (!Number.isFinite(y)) y = -Math.sign(view.y) || 0.001;
            const minY = top + panel.offsetHeight + 35;
            let px = (x + 1) * innerWidth / 2, py = (y + 1) * innerHeight / 2;
            const offscreen = behind || projected.z > 1 || px < 90 || px > innerWidth - 90 || py < top + 25 || py > innerHeight - 50;
            if (offscreen) {
                const dx = x || 0.001, dy = y || 0.001;
                const scale = Math.min((innerWidth / 2 - 90) / Math.abs(dx), Math.max(10, (innerHeight - minY) / 2 - 25) / Math.abs(dy));
                px = innerWidth / 2 + dx * scale; py = (minY + innerHeight - 30) / 2 + dy * scale;
            } else {
                projected.copy(target.posVec); projected.y += (isGrid ? halfSize : target.r) + 0.14;
                projected.project(camera);
                px = (projected.x + 1) * innerWidth / 2;
                py = (1 - projected.y) * innerHeight / 2;
            }
            const bearing = Math.abs(x) > Math.abs(y) ? (x < 0 ? '←' : '→') : (y < 0 ? '↑' : '↓');
            const crossing = isGrid ? ` · ${target.dir === 1 ? '前→後' : '後→前'}` : '';
            const compact = isGrid && innerHeight <= 550 && innerWidth <= 600;
            setText(label.lastElementChild, compact ? `${offscreen ? bearing + ' ' : ''}第 ${index + 1} ${unit}${crossing}`
                : offscreen ? `${bearing} 第 ${index + 1} ${unit}・${behind ? '在視角後方' : '畫面外'}${crossing}` : `下一個 · 第 ${index + 1} ${unit}${crossing}`);
            const labelMargin = label.offsetWidth / 2 + 10;
            px = THREE.MathUtils.clamp(px, labelMargin, innerWidth - labelMargin);
            py = THREE.MathUtils.clamp(py, top + 28, innerHeight - 35);
            if (isGrid) {
                const telemetry = document.querySelector('.fc-telemetry');
                const rect = telemetry && telemetry.getBoundingClientRect();
                if (rect && px + label.offsetWidth / 2 > rect.left && px - label.offsetWidth / 2 < rect.right &&
                    py > rect.top && py - label.offsetHeight < rect.bottom) {
                    const left = rect.left - labelMargin, right = rect.right + labelMargin;
                    if (left >= labelMargin) px = left;
                    else if (right <= innerWidth - labelMargin) px = right;
                    else py = rect.top - 10;
                }
            }
            const panelRect = panel.getBoundingClientRect();
            if (px + label.offsetWidth / 2 > panelRect.left - 8 && px - label.offsetWidth / 2 < panelRect.right + 8 &&
                py > panelRect.top - 8 && py - label.offsetHeight < panelRect.bottom + 8) {
                py = Math.min(innerHeight - 10, panelRect.bottom + label.offsetHeight + 10);
            }
            label.style.left = `${Math.round(px)}px`; label.style.top = `${Math.round(py)}px`;
            label.dataset.offscreen = String(offscreen);
        }
        return { update, group, panel, label };
    }
    root.DroneRingNavigation = { create };
})(window);
