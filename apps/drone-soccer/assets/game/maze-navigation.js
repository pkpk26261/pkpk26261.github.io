/* Local passage cues use the same six wall arrays as collision detection. */
(function (root) {
    'use strict';
    const FACES = [
        { axis: 'x', sign: -1, normal: [-1, 0, 0] },
        { axis: 'x', sign: 1, normal: [1, 0, 0] },
        { axis: 'y', sign: 1, normal: [0, 1, 0] },
        { axis: 'y', sign: -1, normal: [0, -1, 0] },
        { axis: 'z', sign: 1, normal: [0, 0, 1] },
        { axis: 'z', sign: -1, normal: [0, 0, -1] }
    ];
    function cellAt(data, position) {
        const half = data.n * data.unit / 2;
        const cell = [Math.floor((position.x + half) / data.unit),
            Math.floor(position.y / data.unit), Math.floor((position.z + half) / data.unit)];
        return cell.every(v => v >= 0 && v < data.n) ? cell : null;
    }
    function passages(data, cell, yaw) {
        const [x, y, z] = cell;
        return FACES.map(face => {
            const { axis, sign, normal } = face;
            const blocked = axis === 'x' ? data.xBlocked[x + (sign > 0 ? 1 : 0)][y][z]
                : axis === 'y' ? data.yBlocked[y + (sign > 0 ? 1 : 0)][x][z]
                : data.zBlocked[z + (sign > 0 ? 1 : 0)][x][y];
            let label;
            if (axis === 'y') label = sign > 0 ? '上升' : '下降';
            else {
                // The drone flies toward local +Z; its right side is local -X.
                const forward = normal[0] * Math.sin(yaw) + normal[2] * Math.cos(yaw);
                const right = -normal[0] * Math.cos(yaw) + normal[2] * Math.sin(yaw);
                const angle = Math.atan2(right, forward);
                const labels = ['前方', '右前', '右側', '右後', '後方', '左後', '左側', '左前'];
                label = labels[(Math.round(angle / (Math.PI / 4)) + 8) % 8];
            }
            const gate = axis === 'z' && ((sign < 0 && z === 0) || (sign > 0 && z === data.n - 1));
            return { ...face, blocked: !!blocked, label, gate: gate && !blocked ? (sign > 0 ? '出口' : '入口') : '' };
        });
    }
    function isConnected(data) {
        const start = [data.entrance3D.ix, data.entrance3D.iy, data.entrance3D.iz];
        const queue = [start], seen = new Set([start.join(',')]);
        for (let i = 0; i < queue.length; i++) {
            for (const face of passages(data, queue[i], 0)) {
                if (face.blocked) continue;
                const next = queue[i].map((v, axis) => v + face.normal[axis]);
                const key = next.join(',');
                if (next.some(v => v < 0 || v >= data.n) || seen.has(key)) continue;
                seen.add(key); queue.push(next);
            }
        }
        return seen.size === data.n ** 3;
    }
    // Deterministic fallback visits every cell once. Each cell has two openings,
    // including the outer IN/OUT doors; no invalid last random candidate is used.
    function buildFallbackCore(n) {
        const cube = (a, b, c, value) => Array.from({ length: a }, () =>
            Array.from({ length: b }, () => Array(c).fill(value)));
        const xBlocked = cube(n + 1, n, n, true), yBlocked = cube(n + 1, n, n, true);
        const zBlocked = cube(n + 1, n, n, true), openCount = cube(n, n, n, 0);
        const entrance3D = { ix: n - 1, iy: 0, iz: 0 }, exit3D = { ix: 0, iy: n - 1, iz: n - 1 };
        const path = [[n - 1, 0, 0]], current = [...path[0]];
        const step = (axis, sign) => { current[axis] += sign; path.push([...current]); };
        for (let y = 0; y < n; y++) {
            // Even grids flip X on all but the second layer, which flips Z.
            const rowAxis = n % 2 === 0 && y !== 1 ? 0 : 2;
            const innerAxis = rowAxis === 0 ? 2 : 0;
            const rowSign = current[rowAxis] === 0 ? 1 : -1;
            for (let row = 0; row < n; row++) {
                const innerSign = current[innerAxis] === 0 ? 1 : -1;
                for (let i = 1; i < n; i++) step(innerAxis, innerSign);
                if (row < n - 1) step(rowAxis, rowSign);
            }
            if (y < n - 1) step(1, 1);
        }
        for (let i = 1; i < path.length; i++) {
            const a = path[i - 1], b = path[i];
            if (a[0] !== b[0]) xBlocked[Math.max(a[0], b[0])][a[1]][a[2]] = false;
            else if (a[1] !== b[1]) yBlocked[Math.max(a[1], b[1])][a[0]][a[2]] = false;
            else zBlocked[Math.max(a[2], b[2])][a[0]][a[1]] = false;
            for (const c of [a, b]) openCount[c[0]][c[1]][c[2]]++;
        }
        zBlocked[0][n - 1][0] = false; zBlocked[n][0][n - 1] = false;
        openCount[n - 1][0][0]++; openCount[0][n - 1][n - 1]++;
        return { n, xBlocked, yBlocked, zBlocked, openCount, entrance3D, exit3D,
            entranceCell: { ix: n - 1, iy: 0 }, exitCell: { ix: 0, iy: n - 1 } };
    }
    function create(THREE, data, parent, makeLabel) {
        const group = new THREE.Group();
        group.name = 'maze-passage-guidance';
        parent.add(group);
        const unit = data.unit, half = data.n * unit / 2;
        const material = color => new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide, toneMapped: false });
        const green = material(0x59f4cb), amber = material(0xffd276), blue = material(0x76c7ff);
        const wall = new THREE.MeshBasicMaterial({ color: 0x243b50, side: THREE.DoubleSide,
            transparent: true, opacity: 0.78, depthWrite: false, toneMapped: false });
        const red = material(0xff827e);
        function frame(mat) {
            const result = new THREE.Group();
            // Border sits beside the physical pipes, leaving the opening clear.
            const span = unit * 0.90, thick = unit * 0.035;
            for (const sign of [-1, 1]) {
                const h = new THREE.Mesh(new THREE.BoxGeometry(span, thick, 0.009), mat);
                h.position.y = sign * span / 2; result.add(h);
                const v = new THREE.Mesh(new THREE.BoxGeometry(thick, span, 0.009), mat);
                v.position.x = sign * span / 2; result.add(v);
                // Inset chevrons stay in view when the cockpit is too close to see the full frame.
                const arrow = new THREE.BufferGeometry();
                arrow.setAttribute('position', new THREE.Float32BufferAttribute([
                    sign * unit * 0.29, 0, 0,
                    sign * unit * 0.40, unit * 0.045, 0,
                    sign * unit * 0.40, -unit * 0.045, 0
                ], 3));
                result.add(new THREE.Mesh(arrow, mat));
            }
            return result;
        }
        const localFaces = FACES.map(face => {
            const node = new THREE.Group();
            node.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), new THREE.Vector3(...face.normal));
            const open = frame(face.axis === 'y' ? amber : green);
            const shut = new THREE.Group();
            shut.add(new THREE.Mesh(new THREE.PlaneGeometry(unit * 0.96, unit * 0.96), wall));
            for (const sign of [-1, 1]) {
                const cross = new THREE.Mesh(new THREE.BoxGeometry(unit * 0.24, unit * 0.025, 0.012), red);
                cross.rotation.z = sign * Math.PI / 4; shut.add(cross);
            }
            node.add(open, shut); group.add(node);
            return { node, open, shut };
        });
        // Permanent gate frames and overhead labels distinguish the two outer doors.
        for (const [cell, z, text, mat, color] of [
            [data.entranceCell, -half, '入口 IN', green, '#59f4cb'],
            [data.exitCell, half, '出口 OUT', blue, '#76c7ff']
        ]) {
            const gate = frame(mat);
            gate.position.set(-half + (cell.ix + 0.5) * unit, (cell.iy + 0.5) * unit, z);
            parent.add(gate);
            const label = makeLabel(text, color, Math.min(0.85, unit * 0.8));
            label.position.copy(gate.position); label.position.y += unit * 0.65;
            parent.add(label);
        }
        let panel = document.getElementById('mazeNavigation');
        if (!panel) {
            panel = document.createElement('section'); panel.id = 'mazeNavigation';
            panel.setAttribute('aria-label', '目前格位通道');
            panel.innerHTML = '<strong id="mazeCellStatus"></strong><div class="maze-passages"></div>' +
                '<p>方向依機頭 · 亮框可通行 · × 為牆</p>';
            const header = document.getElementById('missionLevelVal').closest('header');
            const hud = header.closest('.fc-hud') || header.parentElement;
            hud.appendChild(panel);
            // Keep the navigation out of header flow so small screens retain their toolbar position.
            const place = () => { panel.style.top = `${header.getBoundingClientRect().bottom - hud.getBoundingClientRect().top + 8}px`; };
            new ResizeObserver(place).observe(header);
            place();
        }
        const status = panel.querySelector('strong'), list = panel.querySelector('.maze-passages');
        let lastCell = '', lastText = '';
        return {
            group, localFaces,
            update(position, yaw, mission) {
                const cell = cellAt(data, position);
                group.visible = !!cell;
                if (!cell) {
                    const text = mission.completed ? '已通過出口，恭喜過關！' : mission.entered
                        ? '已離開迷宮範圍，請辨認藍色出口' : `前往綠色入口 · 中心高度 ${((data.entranceCell.iy + 0.5) * unit).toFixed(2)} m`;
                    if (lastText !== text) { status.textContent = text; list.replaceChildren(); lastText = text; }
                    return;
                }
                const faces = passages(data, cell, yaw), key = cell.join(',');
                if (lastCell !== key) {
                    const center = new THREE.Vector3(-half + (cell[0] + 0.5) * unit,
                        (cell[1] + 0.5) * unit, -half + (cell[2] + 0.5) * unit);
                    faces.forEach((face, i) => {
                        const view = localFaces[i];
                        view.node.position.copy(center).addScaledVector(new THREE.Vector3(...face.normal), unit / 2 - 0.008);
                        view.open.visible = !face.blocked; view.shut.visible = face.blocked;
                        view.open.children.forEach(mesh => { mesh.material = face.gate === '出口' ? blue : face.axis === 'y' ? amber : green; });
                    });
                    lastCell = key;
                }
                const text = `第 ${cell[1] + 1} / ${data.n} 層 · ${faces.filter(f => !f.blocked).length} 個通道`;
                const signature = text + faces.map(f => f.label + f.blocked + f.gate).join('|');
                if (lastText === signature) return;
                status.textContent = text;
                list.replaceChildren(...faces.map(face => {
                    const item = document.createElement('span');
                    item.dataset.open = String(!face.blocked);
                    item.dataset.vertical = String(face.axis === 'y');
                    item.textContent = `${face.blocked ? '×' : '○'} ${face.label}${face.gate ? '・' + face.gate : ''}`;
                    item.setAttribute('aria-label', `${face.label}：${face.blocked ? '牆面' : face.gate || '可通行'}`);
                    return item;
                }));
                lastText = signature;
            }
        };
    }
    root.DroneMazeNavigation = { cellAt, passages, isConnected, buildFallbackCore, create };
    if (typeof module !== 'undefined' && module.exports) module.exports = root.DroneMazeNavigation;
})(typeof window !== 'undefined' ? window : globalThis);
