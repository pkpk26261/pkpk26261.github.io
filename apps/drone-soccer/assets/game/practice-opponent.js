/* Local, untimed team practice for keyboard and USB entrypoints. */
(() => {
    'use strict';
    const profiles = window.DroneAISettings.profiles.practice;
    function create({ player, rings, radius, paint, notify }) {
        const makeActor = () => ({
            position: new THREE.Vector3(), velocity: new THREE.Vector3(),
            rotation: new THREE.Euler(0, Math.PI, 0, 'YXZ'), rotors: [], lights: [],
            mesh: null, altitudeState: 'grounded', guardHomeOffsetX: 0
        });
        const actor = makeActor(), support = makeActor();
        const actors = [actor, support];
        actor.team = support.team = 'blue';
        let role = 'attack', started = false, phase = 'ready', elapsed = 0, wait = 0;
        let playerGoals = 0, aiGoals = 0, attempts = 0, stops = 0, active = false;
        let aim = { x: 0, y: 0 }, lastContact = -Infinity, time = 0, statusText = '';
        const config = () => profiles[window.DroneAISettings.level];
        const decisions = new Map();
        function guardDecision(guard, enemy, ring, partners, standoff, dt) {
            const c = config();
            let decision = decisions.get(guard);
            if (!decision || (decision.remaining -= dt) <= 0) {
                decision = { remaining: c.reactionMin + Math.random() * (c.reactionMax - c.reactionMin),
                    target: window.DroneAITactics.guardTarget(guard, enemy, ring, c, partners, standoff) };
                decisions.set(guard, decision);
            }
            return decision.target;
        }
        function clearPasses() {
            rings.forEach(ring => { ring.aiPassStates = new WeakMap(); });
        }
        function reset(nextRole = role) {
            decisions.clear();
            player.team = 'red';
            role = nextRole;
            phase = 'ready'; elapsed = wait = time = 0; active = false; lastContact = -Infinity;
            playerGoals = aiGoals = attempts = stops = 0;
            actor.position.set(role === 'attack' ? -0.14 : 0, 0.1, 2.6);
            actor.velocity.set(0, 0, 0);
            actor.altitudeState = 'grounded';
            actor.readyToScore = false;
            window.DroneFlightDynamics.reset(actor);
            support.team = role === 'defense' ? 'red' : 'blue';
            actor.guardHomeOffsetX = role === 'attack' ? -0.14 : 0;
            support.guardHomeOffsetX = role === 'attack' ? 0.14 : -0.16;
            parkSupport();
            clearPasses();
            if (support.mesh) paint(support, 'defense');
            if (actor.mesh) {
                paint(actor, role === 'attack' ? 'defense' : 'attack');
                actor.mesh.position.copy(actor.position);
            }
            refresh();
        }
        function refresh() {
            const objective = role === 'attack'
                ? '進攻目標：正向穿越藍隊球門（2 台 AI 防守）'
                : '防守目標：守住紅隊球門（1 台 AI 隊友協防）';
            const progress = role === 'attack'
                ? `你的進球：${playerGoals}`
                : `失球：${aiGoals}｜未失球回合：${stops}`;
            const state = !started ? '準備練習' : '練習進行中｜AI 自動起飛並持續攻防';
            const text = `${objective}\n${state}｜${progress}`;
            if (text === statusText) return;
            statusText = text;
            document.querySelectorAll('[data-practice-status]').forEach(node => {
                node.textContent = text;
            });
        }
        function beginRun() {
            const c = config(), ring = rings[0];
            const gap = c.tactics.predictiveGap ? window.DroneAITactics.findGap(ring, actor.position, [player.position, support.position], 0.2, radius) : null;
            const offset = gap?.reliable ? Math.min(gap.radius, 0.065) : 0;
            aim = { x: Math.cos(gap?.angle || 0) * offset + (Math.random() - 0.5) * c.aimError,
                y: Math.sin(gap?.angle || 0) * offset + (Math.random() - 0.5) * c.aimError };
            phase = 'run'; elapsed = 0; attempts++; actor.readyToScore = true;
        }
        function retreat(scored) {
            if (!scored) stops++;
            phase = 'clear'; elapsed = 0; actor.readyToScore = false;
        }
        function goal(who, ring) {
            if (!started || !active) return false;
            if (who === player && role === 'attack' && ring === rings[1]) {
                playerGoals++;
                notify(`進球成功！你已進 ${playerGoals} 球`);
            } else if (who === actor && role === 'defense' && ring === rings[0] && phase === 'run') {
                aiGoals++; retreat(true);
                notify(`本回合失球，累計 ${aiGoals} 球；AI 將繞回重新進攻`);
            } else return false;
            ring.passes = who === player ? playerGoals : aiGoals;
            refresh();
            return true;
        }
        function update(delta) {
            const dt = Math.max(0, Math.min(delta, 0.05));
            // AI 由練習開始控制；真人起降不重設 AI 位置、回合或計分。
            const canPlay = started;
            if (!canPlay) {
                if (active) { phase = 'ready'; elapsed = wait = 0; clearPasses(); decisions.clear(); }
                active = false;
                actor.velocity.set(0, 0, 0);
                actor.altitudeState = 'grounded';
                window.DroneFlightDynamics.reset(actor);
                actor.position.set(role === 'attack' ? -0.14 : 0, 0.1, 2.6);
            } else {
                active = true;
                const c = config();
                let target;
                if (role === 'attack') {
                    target = guardDecision(actor, player, rings[1], [support], 0.25, dt);
                } else {
                    const ring = rings[0];
                    if (phase === 'ready') {
                        target = { x: 0, y: ring.pos.y, z: 1.25 };
                        wait += dt;
                        if (wait >= c.pause && actor.position.distanceTo(new THREE.Vector3(target.x, target.y, target.z)) < 0.1) beginRun();
                    } else if (phase === 'run') {
                        // Fly beyond the plane so the entire cage clears the opening.
                        target = { x: ring.pos.x + aim.x, y: ring.pos.y + aim.y, z: ring.pos.z - 0.3 };
                    } else if (phase === 'clear') {
                        // Move away from the ring before taking the outside return lane.
                        target = { x: 0.85, y: 1.55, z: actor.position.z };
                        if (actor.position.x > 0.7) phase = 'return';
                    } else {
                        target = { x: 0.85, y: 1.55, z: 1.25 };
                        if (actor.position.z > 1.1) { phase = 'ready'; wait = elapsed = 0; }
                    }
                }
                if (actor.altitudeState === 'grounded') actor.altitudeState = 'takingoff';
                if (actor.altitudeState === 'takingoff') {
                    if (role === 'attack'
                        ? window.DroneAITactics.launchGuard(actor, rings[1], c, [support], dt, 0.25)
                        : window.DroneAITactics.launchAttack(actor, rings[0], c, dt)) {
                        actor.altitudeState = 'flying';
                    }
                } else if (role === 'defense') {
                    window.DroneAITactics.stepAttack(actor, target, c, dt, {
                        ring: rings[0], advancing: phase === 'run', blockers: [player, support], cageRadius: radius
                    });
                } else window.DroneAITactics.stepGuard(actor, target, c, dt, { guardRing: rings[1] });
            }
            if (actor.mesh) {
                actor.mesh.position.copy(actor.position);
                if (actor.velocity.lengthSq() > 0.01) actor.rotation.y = Math.atan2(actor.velocity.x, actor.velocity.z);
                actor.mesh.rotation.set(-actor.velocity.z * 0.08, actor.rotation.y, -actor.velocity.x * 0.08, 'YXZ');
                actor.rotors.forEach((rotor, i) => { rotor.rotation.y += (i % 2 ? -1 : 1) * (active ? 70 : 12) * dt; });
            }
            refresh();
        }
        function parkSupport() {
            support.position.set(0.14, 0.1, role === 'attack' ? 2.6 : -2.6);
            support.velocity.set(0, 0, 0);
            support.altitudeState = 'grounded';
            window.DroneFlightDynamics.reset(support);
            if (support.mesh) support.mesh.position.copy(support.position);
        }
        const updateSupport = window.DroneFlightDynamics.wrapAI(function (dt) {
            if (!active) parkSupport();
            else {
                const defending = role === 'defense';
                const target = guardDecision(support, defending ? actor : player,
                    rings[defending ? 0 : 1], [defending ? player : actor], defending ? 0.4 : 0.25, dt);
                if (support.altitudeState === 'grounded') support.altitudeState = 'takingoff';
                if (support.altitudeState === 'takingoff') {
                    if (window.DroneAITactics.launchGuard(support,
                        rings[defending ? 0 : 1], config(), [defending ? player : actor], dt, defending ? 0.4 : 0.25)) {
                        support.altitudeState = 'flying';
                    }
                } else window.DroneAITactics.stepGuard(support, target, config(), dt,
                    { guardRing: rings[role === 'defense' ? 0 : 1] });
            }
            if (support.mesh) {
                support.mesh.position.copy(support.position);
                if (support.velocity.lengthSq() > 0.01) support.rotation.y = Math.atan2(support.velocity.x, support.velocity.z);
                support.mesh.rotation.set(-support.velocity.z * 0.08, support.rotation.y, -support.velocity.x * 0.08, 'YXZ');
                support.rotors.forEach((rotor, i) => { rotor.rotation.y += (i % 2 ? -1 : 1) * (active ? 70 : 12) * dt; });
            }
        }, () => support);
        function collide() {
            if (!active) return;
            for (const [a, b] of [[player, actor], [player, support], [actor, support]]) {
                const hit = window.DroneFlightDynamics.collide(a, b, radius);
                if (!hit) continue;
                if (a === player && hit.closingSpeed > 0.12) {
                    window.DroneCollisionVisuals.hit(player, hit.closingSpeed,
                        player.position.clone().addScaledVector(hit.normal, radius), hit.normal.clone().negate());
                }
                if (a === player && b.team !== 'red' && time - lastContact > 0.75 && hit.closingSpeed > 0.15) {
                    lastContact = time;
                    notify(role === 'defense' ? '與 AI 前鋒碰撞，請穩住機身繼續防守' : '與 AI 防守機碰撞，請穩住機身再調整進攻角度');
                }
            }
        }
        function mount() {
            const makeSelect = () => {
                const label = document.createElement('label');
                label.className = 'fc-camera-setting fc-ui-region';
                label.textContent = 'AI 陪練智能難度 ';
                const select = document.createElement('select');
                select.dataset.aiLevel = '';
                select.setAttribute('aria-label', 'AI 陪練智能難度');
                Object.entries(window.DroneAISettings.levels).forEach(([key, info]) => select.add(new Option(info.label, key)));
                select.value = window.DroneAISettings.level;
                select.addEventListener('change', () => window.DroneAISettings.applyLevel(select.value));
                label.addEventListener('click', event => event.stopPropagation());
                label.append(select); return label;
            };
            document.getElementById('roleDesc').after(makeSelect());
            const controls = document.querySelector('.fc-more-actions') || document.querySelector('.fc-system-actions');
            controls?.append(makeSelect());
            const status = document.createElement('p');
            status.dataset.practiceStatus = '';
            status.setAttribute('role', 'status');
            status.setAttribute('aria-live', 'polite');
            status.style.cssText = 'position:fixed;top:76px;left:50%;transform:translateX(-50%);max-width:94vw;width:max-content;padding:6px 10px;border-radius:9px;background:#0f172ae8;color:#e2e8f0;font:12px/1.6 sans-serif;text-align:center;white-space:pre-line;z-index:15;pointer-events:none';
            document.body.append(status);
            // Narrow HUDs wrap the toolbar below the score card. Keep the new
            // practice summary below both rows instead of covering the score.
            const anchors = [...document.querySelectorAll('#gameUI > header, .fc-flight-toolbar, .fc-flight-status')];
            const positionStatus = () => {
                const center = document.querySelector('.fc-flight-status');
                const top = Math.max(76, (center?.getBoundingClientRect().bottom || 0) + 10,
                    ...(innerWidth < 900 ? anchors.map(node => node.getBoundingClientRect().bottom + 8) : []));
                status.style.top = `${top}px`;
                document.documentElement.style.setProperty('--fc-practice-status-bottom', `${status.getBoundingClientRect().bottom + 12}px`);
            };
            const observer = new ResizeObserver(positionStatus);
            anchors.forEach(node => observer.observe(node));
            observer.observe(status);
            window.addEventListener('resize', positionStatus);
            positionStatus();
            statusText = '';
            refresh();
        }
        window.DroneAISettings.register({ apply(level) { window.DroneAISettings.setLevel(level); decisions.clear(); refresh(); } });
        document.addEventListener('DOMContentLoaded', mount);
        const updateOpponent = window.DroneFlightDynamics.wrapAI(update, () => actor);
        return { actor, support, actors, reset, update(delta) {
            const dt = Math.max(0, Math.min(delta, 0.05)); time += dt;
            // A real collision can keep the rigid-body wrapper in charge of
            // movement. Round time and blocked-run completion must keep going
            // even while that wrapper skips the original role update.
            if (started && role === 'defense') {
                elapsed += dt;
                if (phase === 'run' && (elapsed > Math.max(9, 5 / config().maxSpeed + 4)
                    || actor.position.z < rings[0].pos.z - 0.24)) retreat(false);
            }
            updateOpponent(dt); updateSupport(dt);
        }, collide, goal, start() { started = true; },
            get active() { return active; },
            get state() { return { role, started, active, phase, playerGoals, aiGoals, attempts, stops, opponents: role === 'attack' ? 2 : 1, teammates: role === 'defense' ? 1 : 0 }; } };
    }
    window.DronePracticeOpponent = Object.freeze({ create });
})();
