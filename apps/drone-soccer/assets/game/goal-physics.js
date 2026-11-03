/* Suspended rubber goals: dissipative contact, hoop motion and tension-only ropes. SI units. */
(function (root) {
    'use strict';
    const bodies = new Map();
    const GRAVITY = 9.81, MASS = 0.9, SKIN = 1e-5;
    let starts = new WeakMap(), contacts = new WeakMap(), frame = 0;
    const T = () => root.THREE;
    const vector = p => new (T().Vector3)(p?.x || 0, p?.y || 0, p?.z || 0);

    function inverseInertia(body, torque) {
        const local = torque.clone().applyQuaternion(body.q.clone().invert());
        local.x /= body.inertia; local.y /= body.inertia; local.z /= body.inertia * 2;
        return local.applyQuaternion(body.q);
    }

    function rotate(body, rotation) {
        const angle = rotation.length();
        if (angle > 1e-12) body.q.premultiply(new (T().Quaternion)()
            .setFromAxisAngle(rotation.multiplyScalar(1 / angle), angle)).normalize();
    }

    function attachment(body, index) {
        return body.attachments[index].clone().applyQuaternion(body.q).add(body.ring.pos);
    }

    function sync(body) {
        const { ring } = body;
        ring.mesh.position.copy(ring.pos);
        ring.mesh.quaternion.copy(body.q);
        if (ring.display) {
            ring.display.group.position.copy(ring.pos);
            ring.display.group.quaternion.copy(body.q);
            ring.display.center.copy(ring.pos);
        }
        (ring.cables || []).forEach((cable, index) => {
            const end = attachment(body, index), anchor = body.anchors[index];
            const direction = anchor.clone().sub(end);
            cable.position.copy(end).add(anchor).multiplyScalar(0.5);
            cable.quaternion.setFromUnitVectors(new (T().Vector3)(0, 1, 0), direction.clone().normalize());
            cable.scale.y = direction.length() / cable.geometry.parameters.height;
        });
    }

    function register(ring, radius, tubeRadius) {
        if (bodies.has(ring)) return bodies.get(ring);
        // Attach above the centre of mass so the hoop also rights itself after a pitch impact.
        const attachments = [-1, 1].map(sign => new (T().Vector3)(sign * radius * 0.8, radius * 0.6, 0));
        const anchors = attachments.map(p => new (T().Vector3)(ring.pos.x + p.x, 3, ring.pos.z));
        const body = { ring, radius, tubeRadius, mass: MASS,
            inertia: MASS * (0.5 * radius ** 2 + 0.625 * tubeRadius ** 2),
            home: ring.pos.clone(), q: ring.mesh.quaternion.clone(), homeQ: ring.mesh.quaternion.clone(),
            velocity: new (T().Vector3)(), omega: new (T().Vector3)(), attachments, anchors,
            lengths: anchors.map((a, i) => a.distanceTo(attachments[i].clone().add(ring.pos))),
            previous: ring.pos.clone(), previousQ: ring.mesh.quaternion.clone(), samples: new WeakMap(), passes: new WeakMap() };
        bodies.set(ring, body);
        sync(body);
        return body;
    }

    function update(delta) {
        const dt = Math.min(Math.max(Number.isFinite(delta) ? delta : 0, 0), 0.1);
        for (const body of bodies.values()) {
            body.previous.copy(body.ring.pos); body.previousQ.copy(body.q);
            const steps = Math.max(1, Math.ceil(dt * 120)), h = dt / steps;
            if (!h) continue;
            for (let step = 0; step < steps; step++) {
                const p0 = body.ring.pos.clone(), q0 = body.q.clone();
                body.velocity.y -= GRAVITY * h;
                body.velocity.multiplyScalar(Math.exp(-0.48 * h));
                body.omega.multiplyScalar(Math.exp(-0.75 * h));
                body.ring.pos.addScaledVector(body.velocity, h);
                rotate(body, body.omega.clone().multiplyScalar(h));
                // Position constraints also transfer load into the ceiling supports.
                // Only tension: a rope cannot push a goal down when it becomes slack.
                for (let pass = 0; pass < 10; pass++) for (let i = 0; i < 2; i++) {
                    const lever = body.attachments[i].clone().applyQuaternion(body.q);
                    const n = body.ring.pos.clone().add(lever).sub(body.anchors[i]);
                    const length = n.length(), error = length - body.lengths[i];
                    if (error <= 1e-9) continue;
                    n.multiplyScalar(1 / length);
                    const cross = lever.clone().cross(n), angular = inverseInertia(body, cross);
                    const impulse = -error / (1 / body.mass + cross.dot(angular));
                    body.ring.pos.addScaledVector(n, impulse / body.mass);
                    rotate(body, angular.multiplyScalar(impulse));
                }
                // The enclosing arena also stops an exceptionally hard swing.
                const bounds = root.DroneArenaCollision?.bounds;
                if (bounds) for (const axis of ['x', 'y', 'z']) {
                    const margin = body.radius + body.tubeRadius;
                    body.ring.pos[axis] = Math.max(bounds[axis][0] + margin,
                        Math.min(bounds[axis][1] - margin, body.ring.pos[axis]));
                }
                body.velocity.copy(body.ring.pos).sub(p0).multiplyScalar(1 / h);
                const dq = body.q.clone().multiply(q0.invert());
                if (dq.w < 0) { dq.x *= -1; dq.y *= -1; dq.z *= -1; dq.w *= -1; }
                const sine = Math.hypot(dq.x, dq.y, dq.z);
                body.omega.set(dq.x, dq.y, dq.z).multiplyScalar(sine > 1e-10
                    ? 2 * Math.atan2(sine, dq.w) / (sine * h) : 2 / h);
            }
            sync(body);
        }
    }

    function beginFrame(rigs, delta) {
        frame++;
        starts = new WeakMap(); contacts = new WeakMap();
        for (const rig of rigs) if (rig?.position) starts.set(rig, vector(rig.position));
        update(delta);
    }

    function localPosition(ring, position) {
        const body = bodies.get(ring);
        return vector(position).sub(ring.pos).applyQuaternion(body ? body.q.clone().invert() : new (T().Quaternion)());
    }

    function distance(body, point, center = body.ring.pos, q = body.q) {
        const p = point.clone().sub(center).applyQuaternion(q.clone().invert());
        return Math.hypot(Math.hypot(p.x, p.y) - body.radius, p.z);
    }

    function collide(rig, ring, bodyRadius = 0.1) {
        const body = bodies.get(ring);
        if (!body || !rig?.velocity || !rig.position) return false;
        if (distance(body, vector(rig.position)) <= body.tubeRadius + bodyRadius + 0.0005) {
            const p=localPosition(ring,rig.position), radial=Math.hypot(p.x,p.y);
            const lever=new (T().Vector3)(radial?body.radius*p.x/radial:body.radius,radial?body.radius*p.y/radial:0,0).applyQuaternion(body.q);
            const normal=vector(rig.position).sub(ring.pos).sub(lever).normalize();
            const velocity=body.omega.clone().cross(lever).add(body.velocity);
            root.DroneFlightSafety?.contact(rig,normal,velocity);
        }
        let records = contacts.get(rig);
        if (!records) contacts.set(rig, records = new Map());
        const saved = records.get(ring);
        if (saved && saved.position.distanceToSquared(vector(rig.position)) < 1e-18) return saved.hit;
        const end = vector(rig.position), from = starts.get(rig);
        const threshold = body.tubeRadius + bodyRadius;
        let point = end.clone(), hit = distance(body, point) < threshold;
        // Conservative advancement of the sphere against the moving torus. The
        // distance function is Lipschitz; this also catches a fast, complete transit.
        if (from && distance(body, from, body.previous, body.previousQ) >= threshold) {
            const travel = end.distanceTo(from) + body.ring.pos.distanceTo(body.previous)
                + body.q.angleTo(body.previousQ) * (body.radius + body.tubeRadius);
            if (travel > 1e-9) for (let t = 0, iteration = 0; t <= 1 && iteration < 160; iteration++) {
                const center = body.previous.clone().lerp(body.ring.pos, t);
                const q = body.previousQ.clone().slerp(body.q, t);
                const sample = from.clone().lerp(end, t);
                const gap = distance(body, sample, center, q) - threshold;
                if (gap <= 1e-7) { point.copy(sample); hit = true; break; }
                t += Math.max(gap / travel * 0.9, 1e-7);
            }
        }
        if (hit) {
            const p = localPosition(ring, point), radial = Math.hypot(p.x, p.y);
            const onAxis = radial < 1e-10;
            const localContact = new (T().Vector3)(onAxis ? body.radius : body.radius * p.x / radial,
                onAxis ? 0 : body.radius * p.y / radial, 0);
            const lever = localContact.applyQuaternion(body.q);
            const centerline = lever.clone().add(ring.pos);
            let normal = point.clone().sub(centerline);
            if (normal.lengthSq() < 1e-14) {
                normal = vector(rig.velocity).sub(body.velocity).negate();
                if (normal.lengthSq() < 1e-14) normal.set(0, 0, -(ring.forwardSign || 1)).applyQuaternion(body.q);
            }
            normal.normalize();
            const contactLever = lever.clone().addScaledVector(normal, body.tubeRadius);
            const surfaceVelocity = body.omega.clone().cross(contactLever).add(body.velocity);
            const relative = vector(rig.velocity).sub(surfaceVelocity), inward = relative.dot(normal);
            // Rewind swept impacts and separate exact overlaps before scoring.
            point.copy(centerline).addScaledVector(normal, threshold + SKIN);
            Object.assign(rig.position, { x: point.x, y: point.y, z: point.z });
            if (root.DroneFlightDynamics?.materialContact) {
                // Include the cage's angular point velocity and inertia too;
                // hoop and aircraft receive equal/opposite contact impulses.
                root.DroneFlightDynamics.materialContact(rig, normal, 'rubber', {
                    velocity: surfaceVelocity,
                    actorPoint: centerline.clone().addScaledVector(normal, body.tubeRadius),
                    response: impulse => impulse.clone().multiplyScalar(1 / body.mass)
                        .add(inverseInertia(body, contactLever.clone().cross(impulse)).cross(contactLever)),
                    applyImpulse: impulse => {
                        body.velocity.addScaledVector(impulse, -1 / body.mass);
                        body.omega.sub(inverseInertia(body, contactLever.clone().cross(impulse)));
                    }
                });
            } else if (inward < 0) {
                const mass = rig.constants?.mass || 0.3;
                const torque = contactLever.clone().cross(normal);
                const effective = 1 / mass + 1 / body.mass + torque.dot(inverseInertia(body, torque));
                const bounce = 0.05 * Math.min(1, Math.max(0, (-inward - 0.2) / 0.8));
                const magnitude = -(1 + bounce) * inward / effective;
                const impulse = normal.clone().multiplyScalar(magnitude);
                const dv = impulse.clone().multiplyScalar(1 / mass);
                rig.velocity.x += dv.x; rig.velocity.y += dv.y; rig.velocity.z += dv.z;
                body.velocity.addScaledVector(impulse, -1 / body.mass);
                body.omega.sub(inverseInertia(body, contactLever.clone().cross(impulse)));
                root.DroneFlightDynamics?.impact(rig, dv, normal);
                root.DroneCollisionVisuals?.hit(rig, -inward, centerline.clone().addScaledVector(normal, body.tubeRadius), normal);
            }
            if (rig.mesh) rig.mesh.position.copy(rig.position);
            // Scoring samples the resolved position below. Touching the inner
            // tube does not discard a valid passage through the opening.
        }
        records.set(ring, { position: vector(rig.position), hit });
        return hit;
    }

    function resolve(rigs, radius) {
        for (const rig of rigs) for (const ring of bodies.keys()) collide(rig, ring, radius);
    }

    // Both sides of the crossing are measured using today's pose. A swinging
    // goal passing a stationary drone therefore cannot manufacture a goal.
    function crossedForward(ring, rig, forwardSign, openingRadius) {
        const body = bodies.get(ring);
        if (!body) return false;
        const now = localPosition(ring, rig.position), previous = starts.get(rig) || body.samples.get(rig);
        body.samples.set(rig, vector(rig.position));
        if (!previous) return false;
        const before = localPosition(ring, previous);
        if (!(before.z * forwardSign < 0 && now.z * forwardSign >= 0)) return false;
        const t = before.z / (before.z - now.z);
        return Math.hypot(before.x + (now.x - before.x) * t, before.y + (now.y - before.y) * t) < openingRadius;
    }

    // One event per complete forward transit. Test the opening at the plane
    // intersection, not at the end of the frame: a diagonal exit can already
    // be outside its projection after the cage has safely passed the hoop.
    function completedForward(ring, rig, forwardSign, openingRadius, bodyRadius = 0.1) {
        const body = bodies.get(ring);
        if (!body || !rig?.position) return false;
        const saved = body.passes.get(rig);
        const start = saved?.frame === frame ? null : starts.get(rig);
        const previous = start || saved?.position;
        const before = start ? vector(start).sub(body.previous).applyQuaternion(body.previousQ.clone().invert()) : saved?.local;
        // Resolve the frame first, including modes with a separate collision
        // callback. Evaluate the actual, separated position even on contact.
        collide(rig, ring, bodyRadius);
        const now = localPosition(ring, rig.position);
        const state = { position: vector(rig.position), local: now, pending: saved?.pending || false, frame };
        body.passes.set(rig, state);
        if (!previous || !before) return false;
        const a = before.z * forwardSign, b = now.z * forwardSign;
        const normal = new (T().Vector3)(0, 0, forwardSign).applyQuaternion(body.q);
        const advances = vector(rig.position).sub(previous).dot(normal) > 1e-9;
        if (a < 0 && b >= 0 && advances) {
            const t = a / (a - b);
            state.pending = Math.hypot(before.x + (now.x - before.x) * t,
                before.y + (now.y - before.y) * t) < openingRadius;
        }
        // Leaving around the outside of the tube is not a passage. Inner-rim
        // contact may keep the entry armed until the entire cage clears.
        const radial = Math.hypot(now.x, now.y);
        if (b < 0 || (Math.abs(b) < bodyRadius + body.tubeRadius && radial >= body.radius)) state.pending = false;
        // Hoop motion alone must not create an entry or finish a passage.
        if (state.pending && b >= bodyRadius && advances) {
            state.pending = false;
            return true;
        }
        return false;
    }

    function resetPasses(rig) {
        starts.delete(rig); contacts.delete(rig);
        for (const body of bodies.values()) {
            body.passes.delete(rig); body.samples.delete(rig);
        }
    }

    function reset() {
        starts = new WeakMap(); contacts = new WeakMap();
        for (const body of bodies.values()) {
            body.ring.pos.copy(body.home); body.q.copy(body.homeQ);
            body.velocity.set(0, 0, 0); body.omega.set(0, 0, 0);
            body.previous.copy(body.home); body.previousQ.copy(body.homeQ);
            body.samples = new WeakMap();
            body.passes = new WeakMap();
            sync(body);
        }
    }

    // A follower applies the host's pose without advancing a second simulation.
    // The same record restores a suspended goal after authority recovery.
    function snapshot(ring) {
        const body = bodies.get(ring);
        return body ? { p:ring.pos.toArray(), q:body.q.toArray(), v:body.velocity.toArray(), w:body.omega.toArray() } : null;
    }
    function applySnapshot(ring, value) {
        const body = bodies.get(ring);
        if (!body || !value || ![['p',3],['q',4],['v',3],['w',3]].every(([key,size]) =>
            Array.isArray(value[key]) && value[key].length === size && value[key].every(Number.isFinite)) ||
            Math.hypot(...value.q) < 1e-8) return false;
        ring.pos.fromArray(value.p); body.q.fromArray(value.q).normalize();
        body.velocity.fromArray(value.v); body.omega.fromArray(value.w);
        body.previous.copy(ring.pos); body.previousQ.copy(body.q);
        body.samples = new WeakMap(); body.passes = new WeakMap();
        sync(body);
        return true;
    }
    root.DroneGoalPhysics = { register, beginFrame, endFrame: () => { starts = new WeakMap(); }, update, collide, resolve, localPosition, crossedForward, completedForward, resetPasses, reset, snapshot, applySnapshot,
        state: ring => bodies.get(ring) };
})(typeof window !== 'undefined' ? window : globalThis);
