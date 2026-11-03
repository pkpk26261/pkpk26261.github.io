/* Visual guidance follows the actual tunnel path; it never changes collision or timing. */
(function (root) {
    'use strict';
    const FACES = [[-1, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
    function direction(normal, yaw) {
        if (normal[1]) return normal[1] > 0 ? '↑ 上升' : '↓ 下降';
        const forward = normal[0] * Math.sin(yaw) + normal[2] * Math.cos(yaw);
        const right = -normal[0] * Math.cos(yaw) + normal[2] * Math.sin(yaw);
        const labels = ['↑ 前進', '↗ 右前方', '→ 向右', '↘ 右後方', '↶ 回轉', '↙ 左後方', '← 向左', '↖ 左前方'];
        return labels[(Math.round(Math.atan2(right, forward) / (Math.PI / 4)) + 8) % 8];
    }
    function create(THREE, data, parent) {
        const unit = data.unit, half = data.n * unit / 2;
        const center = c => new THREE.Vector3(-half + (c.ix + .5) * unit,
            (c.iy + .5) * unit, -half + (c.iz + .5) * unit);
        const centers = data.path.map(center);
        const indexByCell = new Map(data.path.map((c, i) => [`${c.ix},${c.iy},${c.iz}`, i]));
        function indexAt(p) {
            const x = Math.floor((p.x + half) / unit), y = Math.floor(p.y / unit), z = Math.floor((p.z + half) / unit);
            return indexByCell.get(`${x},${y},${z}`) ?? -1;
        }
        const group = new THREE.Group(); group.name = 'tunnel-navigation'; parent.add(group);
        const material = (color, opacity = 1) => new THREE.MeshBasicMaterial({ color, transparent: opacity < 1,
            opacity, side: THREE.DoubleSide, depthWrite: false, toneMapped: false });
        const cyan = material(0x59e5f5), amber = material(0xffce75), muted = material(0x547585, .4);
        const wallMat = material(0x172e43, .62), crossMat = material(0xf6a6a0, .95);
        function frame(mat) {
            const node = new THREE.Group(), span = unit * .89, thick = unit * .025;
            for (const sign of [-1, 1]) {
                const h = new THREE.Mesh(new THREE.BoxGeometry(span, thick, .012), mat);
                h.position.y = sign * span / 2; node.add(h);
                const v = new THREE.Mesh(new THREE.BoxGeometry(thick, span, .012), mat);
                v.position.x = sign * span / 2; node.add(v);
            }
            return node;
        }
        function orient(node, normal) {
            node.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), normal);
        }
        const gates = [cyan, amber].map(mat => {
            const node = frame(mat); group.add(node); return node;
        });
        // Only current-cell walls are emphasized. Remote walls stay visually quiet.
        const localWalls = FACES.map(normal => {
            const node = new THREE.Group(); orient(node, new THREE.Vector3(...normal));
            const localWallMat = wallMat.clone(), localCrossMat = crossMat.clone();
            node.add(new THREE.Mesh(new THREE.PlaneGeometry(unit * .96, unit * .96), localWallMat));
            for (const sign of [-1, 1]) {
                const bar = new THREE.Mesh(new THREE.BoxGeometry(unit * .105, unit * .014, .012), localCrossMat);
                bar.rotation.z = sign * Math.PI / 4; node.add(bar);
            }
            group.add(node); return node;
        });
        const routeGroup = new THREE.Group(); group.add(routeGroup);
        // Fixed-size pool avoids allocating/discarding GPU resources while flying.
        const legs = Array.from({ length: 4 }, (_, i) => {
            const mat = i === 0 ? cyan : i === 1 ? amber : muted;
            const line = new THREE.Mesh(new THREE.CylinderGeometry(unit * .012, unit * .012, 1, 6), mat);
            const arrow = new THREE.Mesh(new THREE.ConeGeometry(unit * .055, unit * .14, 8), mat);
            routeGroup.add(line, arrow); return { line, arrow };
        });
        let panel = document.getElementById('tunnelNavigation');
        if (!panel) {
            panel = document.createElement('section'); panel.id = 'tunnelNavigation';
            panel.setAttribute('aria-label', '隧道路線導引');
            panel.innerHTML = '<div class="tunnel-guide-heading"><span>航線導引</span><span data-tunnel-step></span></div>' +
                '<strong data-tunnel-direction></strong><div data-tunnel-detail></div>' +
                '<div class="tunnel-guide-progress"><span></span></div>' +
                '<p><span class="tunnel-cyan">■ 下一道亮框</span>　<span class="tunnel-amber">■ 後續亮框</span><br>方向依機頭 · × 為牆面</p>';
            const header = document.getElementById('missionLevelVal').closest('header');
            const hud = header.parentElement; hud.appendChild(panel);
            const place = () => { panel.style.top = `${header.getBoundingClientRect().bottom - hud.getBoundingClientRect().top + 10}px`; };
            new ResizeObserver(place).observe(header); place();
        }
        const step = panel.querySelector('[data-tunnel-step]'), title = panel.querySelector('strong');
        const detail = panel.querySelector('[data-tunnel-detail]'), progress = panel.querySelector('.tunnel-guide-progress span');
        let lastIndex = -99, lastText = '';
        function targetFor(i) {
            if (i < 0) return { point: centers[0].clone().add(new THREE.Vector3(0, 0, -unit / 2)), normal: new THREE.Vector3(0, 0, 1) };
            const a = centers[i], b = centers[i + 1];
            return b ? { point: a.clone().add(b).multiplyScalar(.5), normal: b.clone().sub(a).normalize() }
                : { point: a.clone().add(new THREE.Vector3(0, 0, unit / 2)), normal: new THREE.Vector3(0, 0, 1) };
        }
        function blocked(c, normal) {
            return normal[0] ? data.xBlocked[c.ix + (normal[0] > 0 ? 1 : 0)][c.iy][c.iz]
                : normal[1] ? data.yBlocked[c.iy + (normal[1] > 0 ? 1 : 0)][c.ix][c.iz]
                : data.zBlocked[c.iz + (normal[2] > 0 ? 1 : 0)][c.ix][c.iy];
        }
        return { group, gates, localWalls, centers, indexAt, targetFor,
            update(position, yaw, mission, camera) {
                const index = indexAt(position), done = mission.completed;
                // Outside the route we point to the entry, without pretending to know progress.
                group.visible = !done;
                if (index !== lastIndex) {
                    gates.forEach((gate, offset) => {
                        const i = index + offset;
                        gate.visible = i < centers.length;
                        if (!gate.visible) return;
                        const target = targetFor(i); gate.position.copy(target.point); orient(gate, target.normal);
                    });
                    localWalls.forEach((wall, face) => {
                        wall.visible = index >= 0 && !!blocked(data.path[index], FACES[face]);
                        if (index >= 0) wall.position.copy(centers[index]).addScaledVector(new THREE.Vector3(...FACES[face]), unit / 2 - .01);
                    });
                    const points = index < 0 ? [centers[0].clone().add(new THREE.Vector3(0, 0, -unit * 1.7)), ...centers.slice(0, 4)]
                        : centers.slice(index, index + 5);
                    if (index >= 0 && points.length < 5) points.push(centers[centers.length - 1].clone().add(new THREE.Vector3(0, 0, unit * .7)));
                    legs.forEach(({ line, arrow }, i) => {
                        line.visible = arrow.visible = i + 1 < points.length;
                        if (!line.visible) return;
                        const a = points[i].clone(), b = points[i + 1].clone();
                        a.y -= unit * .24; b.y -= unit * .24;
                        const vector = b.clone().sub(a), length = vector.length();
                        const rotation = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), vector.normalize());
                        line.position.copy(a).lerp(b, .5); line.scale.y = length; line.quaternion.copy(rotation);
                        arrow.position.copy(a).lerp(b, .65); arrow.quaternion.copy(rotation);
                    });
                    lastIndex = index;
                }
                // A chase camera can sit outside the cell. Fade the intervening wall
                // so its surface and cross do not cover the drone or the route.
                if (camera && index >= 0) localWalls.forEach((wall, face) => {
                    const n = FACES[face], p = wall.position, c = camera.position;
                    const outside = (c.x-p.x)*n[0] + (c.y-p.y)*n[1] + (c.z-p.z)*n[2] > .01;
                    wall.children[0].material.opacity = outside ? .08 : .62;
                    wall.children[1].material.opacity = outside ? .18 : .95;
                });
                const target = targetFor(index), next = targetFor(Math.min(index + 1, centers.length - 1));
                const normal = [target.normal.x, target.normal.y, target.normal.z];
                const command = direction(normal, yaw);
                const text = done ? '完成穿越' : index < 0 ? '對準入口亮框' : `${command}${index === centers.length - 1 ? ' · 穿出終點' : ''}`;
                const sub = done ? '已抵達出口，可重新挑戰' : index < 0 ? `入口中心高度 ${centers[0].y.toFixed(2)} m`
                    : index === centers.length - 1 ? `出口高度 ${target.point.y.toFixed(2)} m · 穿出即完成`
                    : `目標高度 ${target.point.y.toFixed(2)} m · 接著${direction([next.normal.x, next.normal.y, next.normal.z], yaw).replace(/^\S+ /, '')}`;
                const state = `${index}|${text}|${sub}`;
                if (state !== lastText) {
                    step.textContent = done ? '完成' : index < 0 ? '入口' : `${index + 1} / ${centers.length} 格`;
                    title.textContent = text; detail.textContent = sub;
                    progress.style.width = `${done ? 100 : Math.max(0, index) / centers.length * 100}%`;
                    lastText = state;
                }
            }
        };
    }
    root.DroneTunnelNavigation = { create, direction };
})(typeof window !== 'undefined' ? window : globalThis);
