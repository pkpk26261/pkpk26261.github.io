/* Score LEDs follow the existing torus surface; no housing or extra collider. */
(function (root) {
    'use strict';

    function create(ring, scene, radius, tubeRadius) {
        root.DroneArenaCollision?.registerGoal(ring, radius, tubeRadius);
        const T = root.THREE;
        const canvas = document.createElement('canvas');
        canvas.width = 1024;
        canvas.height = 256;
        const texture = new T.CanvasTexture(canvas);
        texture.anisotropy = 4;
        texture.minFilter = T.LinearFilter;
        texture.generateMipmaps = false;
        const positions = [], uvs = [], indices = [];
        // Match the 120 x 40 torus grid. Both surfaces stay on the original tube.
        // Upper 120 degrees, with a white rim beside the 108-degree tube face.
        const columns = 40, rows = 12;
        for (let y = 0; y <= rows; y++) {
            const v = y / rows;
            const phi = (v * 108 - 54) * Math.PI / 180;
            const r = radius + tubeRadius * Math.sin(phi);
            for (let x = 0; x <= columns; x++) {
                const u = x / columns;
                const theta = (u * 120 - 60) * Math.PI / 180;
                positions.push(r * Math.sin(theta), r * Math.cos(theta), tubeRadius * Math.cos(phi));
                uvs.push(u, v);
                if (x < columns && y < rows) {
                    const a = y * (columns + 1) + x, b = a + 1, c = a + columns + 1;
                    indices.push(a, b, c, b, c + 1, c);
                }
            }
        }
        const geometry = new T.BufferGeometry();
        geometry.setAttribute('position', new T.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('uv', new T.Float32BufferAttribute(uvs, 2));
        geometry.setIndex(indices);
        geometry.computeVertexNormals();
        const material = new T.MeshBasicMaterial({ map: texture, toneMapped: false,
            polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -4 });
        const group = new T.Group();
        group.name = 'goal-ring-score-band';
        const front = new T.Mesh(geometry, material);
        front.name = 'goal-score-front';
        const back = new T.Mesh(geometry, material);
        back.name = 'goal-score-back';
        back.rotation.y = Math.PI;
        group.add(front, back);
        group.position.copy(ring.pos);
        scene.add(group);
        const display = { canvas, ctx: canvas.getContext('2d'), texture, group, key: null,
            center: group.position.clone() };
        ring.display = display;
        root.DroneGoalPhysics.register(ring, radius, tubeRadius);
        return display;
    }

    function draw(display, info) {
        const g = display.ctx, W = display.canvas.width, H = display.canvas.height;
        const color = info.team === 'red' ? '#fb7185' : '#60a5fa';
        g.clearRect(0, 0, W, H);
        g.fillStyle = '#091321';
        g.fillRect(0, 0, W, H);
        g.fillStyle = info.flashing ? '#34d399' : color;
        g.fillRect(0, 0, W, 10);
        g.fillRect(0, H - 10, W, 10);
        g.textBaseline = 'middle';
        g.textAlign = 'center';
        g.fillStyle = color;
        g.font = 'bold 48px "Microsoft JhengHei",sans-serif';
        g.fillText(info.team === 'red' ? '紅方球門' : '藍方球門', 175, 103, 290);
        g.fillStyle = '#b9c9df';
        g.font = '28px "Microsoft JhengHei",sans-serif';
        g.fillText(info.tag, 175, 163, 290);
        // Always keep the number visible, including the goal celebration.
        const number = String(info.goals).padStart(2, '0');
        g.fillStyle = info.flashing ? '#6ee7b7' : '#ffffff';
        g.font = `bold ${number.length > 2 ? 142 : 178}px Consolas,monospace`;
        g.fillText(number, W / 2, 116, 330);
        g.font = 'bold 24px "Microsoft JhengHei",sans-serif';
        g.fillStyle = info.flashing ? '#6ee7b7' : '#b9c9df';
        g.fillText(info.flashing ? 'GOAL!  進球' : '進球數', W / 2, 222);
        g.fillStyle = info.flashing ? '#6ee7b7' : info.status[0];
        g.font = `bold ${info.status[1].includes('\n') ? 40 : 32}px "Microsoft JhengHei",sans-serif`;
        // Wrap long match instructions instead of shrinking them to illegible text.
        const status = info.flashing ? '進球得分' : info.status[1].replace(/^●\s*/, '');
        const chunks = status.split('\n').flatMap(line => line.match(/.{1,8}/gu) || ['']);
        chunks.slice(0, 3).forEach((line, i, lines) => {
            g.fillText(line, 852, 132 + (i - (lines.length - 1) / 2) * 42, 290);
        });
        display.texture.needsUpdate = true;
    }

    root.DroneGoalRingDisplay = { create, draw };
})(window);
