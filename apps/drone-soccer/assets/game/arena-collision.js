/* Rubber arena tubes. Geometry is captured before PERF merges meshes. */
(function (root) {
    'use strict';
    const SKIN = 1e-6;
    const sub = (a, b) => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });
    const add = (a, b, t = 1) => ({ x: a.x + b.x * t, y: a.y + b.y * t, z: a.z + b.z * t });
    const dot = (a, b) => a.x * b.x + a.y * b.y + a.z * b.z;
    const copy = p => ({ x: p.x, y: p.y, z: p.z });

    function closest(p, tube) {
        const axis = sub(tube.b, tube.a);
        const t = Math.max(0, Math.min(1, dot(sub(p, tube.a), axis) / dot(axis, axis)));
        return add(tube.a, axis, t);
    }

    function normalAt(p, tube) {
        const offset = sub(p, closest(p, tube));
        const length = Math.hypot(offset.x, offset.y, offset.z);
        if (length > 1e-10) return add({ x: 0, y: 0, z: 0 }, offset, 1 / length);
        // An exact center-line overlap (e.g. an old saved position): push into the arena.
        const axis = sub(tube.b, tube.a);
        const inward = sub({ x: 0, y: 1.5, z: 0 }, p);
        const perpendicular = add(inward, axis, -dot(inward, axis) / dot(axis, axis));
        const size = Math.hypot(perpendicular.x, perpendicular.y, perpendicular.z);
        return size > 1e-10 ? add({ x: 0, y: 0, z: 0 }, perpendicular, 1 / size) : { x: 1, y: 0, z: 0 };
    }

    // Earliest segment hit on a capsule: the cylinder and its two end spheres.
    // Sweeping the full frame displacement prevents fast flight from tunnelling.
    function sweep(start, move, tube, radius) {
        const axis = sub(tube.b, tube.a), offset = sub(start, tube.a);
        const axisSq = dot(axis, axis), along = dot(offset, axis) / axisSq;
        const speedAlong = dot(move, axis) / axisSq;
        const radial = add(offset, axis, -along);
        const radialMove = add(move, axis, -speedAlong);
        let best = null;
        function quadratic(a, b, c, accept) {
            if (a < 1e-20) return;
            const disc = b * b - 4 * a * c;
            if (disc < 0) return;
            const t = (-b - Math.sqrt(disc)) / (2 * a);
            if (t < -1e-9 || t > 1 || !accept(t)) return;
            const time = Math.max(0, t);
            const normal = normalAt(add(start, move, time), tube);
            if (dot(move, normal) >= -1e-12) return;
            if (!best || time < best.time) best = { time, normal };
        }
        quadratic(dot(radialMove, radialMove), 2 * dot(radial, radialMove), dot(radial, radial) - radius * radius,
            t => along + speedAlong * t >= 0 && along + speedAlong * t <= 1);
        for (const end of [tube.a, tube.b]) {
            const d = sub(start, end);
            quadratic(dot(move, move), 2 * dot(d, move), dot(d, d) - radius * radius,
                t => end === tube.a ? along + speedAlong * t <= 0 : along + speedAlong * t >= 1);
        }
        return best;
    }

    class Arena {
        constructor() {
            this.tubes = [];
            this.starts = new WeakMap();
            this.bounds = null;
            this.goals = new Map();
            this.traffic = [];
            this.roof = null;
            this.nets = [];
            this.delta = 0;
        }

        register(group) {
            group.updateMatrixWorld(true);
            this.tubes = [];
            group.traverse(mesh => {
                const p = mesh.geometry && mesh.geometry.parameters;
                if (!p || !p.height || !p.radiusTop || !p.radiusBottom) return;
                const a = mesh.localToWorld(new root.THREE.Vector3(0, -p.height / 2, 0));
                const b = mesh.localToWorld(new root.THREE.Vector3(0, p.height / 2, 0));
                const scale = mesh.getWorldScale(new root.THREE.Vector3());
                this.tubes.push({ a: copy(a), b: copy(b), radius: Math.max(p.radiusTop, p.radiusBottom) * Math.max(scale.x, scale.z) });
            });
            this.bounds = {};
            for (const axis of ['x', 'y', 'z']) {
                const values = this.tubes.flatMap(tube => [tube.a[axis], tube.b[axis]]);
                this.bounds[axis] = [Math.min(...values), Math.max(...values)];
            }
            this.nets = root.DroneRopeNet ? [['x',1],['x',-1],['y',-1],['z',1],['z',-1]]
                .map(([axis,sign])=>root.DroneRopeNet.create(this.bounds,axis,sign)) : [];
            this.nets.forEach(net=>net.attach(group));
            this.roof = this.nets.find(net=>net.axis==='y') || null;
        }

        beginFrame(rigs, delta) {
            this.delta = Math.max(0, Math.min(Number.isFinite(delta) ? delta : 0, 0.1));
            this.nets.forEach(net=>net.advance(delta));
            root.DroneFlightDynamics?.beginContacts(rigs, delta);
            root.DroneGoalPhysics?.beginFrame(rigs, delta);
            // All pilots see the same traffic sample, regardless of update order.
            this.traffic = rigs.filter(rig => rig?.position && rig.velocity).map(rig => ({
                actor: rig, position: copy(rig.position), velocity: copy(rig.velocity), team: rig.team
            }));
            for (const rig of rigs) if (rig && rig.position) this.starts.set(rig, copy(rig.position));
        }

        registerGoal(ring, radius, tubeRadius) {
            this.goals.set(ring, { ring, radius, tubeRadius });
        }

        resolve(rigs, bodyRadius) {
            const stabilizers = [];
            // Include every human/AI and same-team pair, even in legacy modes
            // whose feedback handler only enumerates opponents.
            for (let i = 0; i < rigs.length; i++) for (let j = i + 1; j < rigs.length; j++)
                root.DroneFlightDynamics?.collide(rigs[i], rigs[j], bodyRadius);
            root.DroneGoalPhysics?.resolve(rigs, bodyRadius);
            for (const rig of rigs) {
                if (!rig || !rig.position) continue;
                let current = copy(this.starts.get(rig) || rig.position);
                let remaining = sub(rig.position, current);
                let remainingTime = this.delta;
                const contactedNets = new Set();
                const planes = [];
                if (this.bounds) for (const axis of ['x', 'y', 'z']) {
                    const [min, max] = this.bounds[axis];
                    if (!this.nets.length || axis==='y') planes.push({ axis, limit: min + (axis === 'y' ? root.DroneFlightDynamics?.groundHeight(rig) ?? bodyRadius : bodyRadius), sign: 1 });
                    if (!this.nets.length) planes.push({ axis, limit: max - bodyRadius, sign: -1 });
                }
                const reflect = (normal, net = null) => {
                    if (!rig.velocity) return;
                    if (net) return net.contact(rig, current);
                    const inward = dot(rig.velocity, normal);
                    if (root.DroneFlightDynamics?.materialContact) {
                        // A rotating rib may still approach while its centre
                        // separates. The rigid contact solver checks point speed.
                        return root.DroneFlightDynamics.materialContact(rig, normal, 'rubber', {
                            // The sweep has already reached contact; the actor's
                            // original end-of-step position is not the contact.
                            point: current
                        });
                    }
                    if (inward < 0) {
                        root.DroneFlightDynamics?.impact(rig, add({ x: 0, y: 0, z: 0 }, normal, -1.05 * inward));
                        if (root.DroneCollisionVisuals) root.DroneCollisionVisuals.hit(rig, -inward, add(current, normal, -bodyRadius), normal);
                        Object.assign(rig.velocity, add(rig.velocity, normal, -1.05 * inward));
                    }
                };
                // Recover pre-existing overlaps before sweeping (including exact-axis cases).
                const depenetrate = () => {
                    for (let pass = 0; pass < 48; pass++) {
                        let moved = false;
                        for (const tube of this.tubes) {
                            const d = sub(current, closest(current, tube));
                            const distance = Math.hypot(d.x, d.y, d.z);
                            const radius = tube.radius + bodyRadius + SKIN;
                            if (distance >= radius - 1e-10) continue;
                            const normal = normalAt(current, tube);
                            current = add(current, normal, radius - distance);
                            reflect(normal);
                            moved = true;
                        }
                        for (const plane of planes) {
                            if ((current[plane.axis] - plane.limit) * plane.sign >= 0) continue;
                            current[plane.axis] = plane.limit;
                            const normal = { x: 0, y: 0, z: 0 };
                            normal[plane.axis] = plane.sign;
                            reflect(normal);
                            moved = true;
                        }
                        for (const net of this.nets) {
                            const limit = net.limit(current, bodyRadius);
                            if ((current[net.axis]-limit)*net.sign < -1e-10) {
                                current[net.axis] = limit;reflect(net.normal(current),net);moved = true;
                            }
                        }
                        if (!moved) break;
                    }
                };
                depenetrate();
                for (let pass = 0; pass < 12; pass++) {
                    // A compliant net can share the cage's outward velocity.
                    // Its geometry is frozen during this substep, so sweeping
                    // it again at t=0 repeatedly consumes the entire iteration
                    // budget and discards sliding travel. Retain that contact
                    // as a positional constraint while sweeping other objects.
                    // Only clip motion through the net; retreat stays free.
                    for (const net of contactedNets) {
                        const target = add(current, remaining);
                        const limit = net.limit(target, bodyRadius);
                        if ((target[net.axis] - limit) * net.sign < 0)
                            remaining[net.axis] = limit - current[net.axis];
                    }
                    let hit = null;
                    for (const tube of this.tubes) {
                        const candidate = sweep(current, remaining, tube, tube.radius + bodyRadius + SKIN);
                        if (candidate && (!hit || candidate.time < hit.time)) hit = candidate;
                    }
                    for (const plane of planes) {
                        if (remaining[plane.axis] * plane.sign >= 0) continue;
                        const rawTime = (plane.limit - current[plane.axis]) / remaining[plane.axis];
                        if (rawTime < -1e-9 || rawTime > 1) continue;
                        const time = Math.max(0, rawTime);
                        if (hit && time >= hit.time) continue;
                        const normal = { x: 0, y: 0, z: 0 };
                        normal[plane.axis] = plane.sign;
                        hit = { time, normal };
                    }
                    for (const net of this.nets) {
                        if (contactedNets.has(net)) continue;
                        const netHit = net.sweep(current, remaining, bodyRadius);
                        if (netHit && (!hit || netHit.time < hit.time)) hit = netHit;
                    }
                    if (!hit) { current = add(current, remaining); break; }
                    current = add(current, remaining, hit.time);
                    remainingTime *= 1 - hit.time;
                    remaining = add({ x: 0, y: 0, z: 0 }, remaining, 1 - hit.time);
                    const inward = dot(remaining, hit.normal);
                    if (inward < 0) remaining = add(remaining, hit.normal, -inward);
                    reflect(hit.normal, hit.net);
                    if (hit.net) contactedNets.add(hit.net);
                    // Continue the part of this step after the impact using
                    // resolved velocity, including rebound and contact friction.
                    // Geometry-only callers without a time step retain sliding.
                    if (remainingTime > 0 && rig.velocity) remaining = add({ x: 0, y: 0, z: 0 }, rig.velocity, remainingTime);
                }
                // Reconcile simultaneous rail/floor contacts after numerical rounding.
                depenetrate();
                Object.assign(rig.position, current);
                root.DroneFlightDynamics?.limitSpeed(rig);
                if (rig.mesh) rig.mesh.position.copy(rig.position);
                stabilizers.push(() => {
                    current = copy(rig.position);
                    depenetrate();
                    Object.assign(rig.position, current);
                    if (rig.mesh) rig.mesh.position.copy(rig.position);
                });
                this.starts.delete(rig);
            }
            // A third cage (or wall) can close a pair just separated above.
            // Revisit only actual overlaps, without rewinding the sweep or
            // replaying restitution. Usually there is no second pass at all;
            // the finite cap also handles a six-cage chain pressed into a wall.
            if (root.DroneFlightDynamics?.collide && rigs.length > 1) for (let pass = 0; pass < 192; pass++) {
                let moved = false;
                for (let i = 0; i < rigs.length; i++) for (let j = i + 1; j < rigs.length; j++) {
                    if (!rigs[i]?.position || !rigs[j]?.position) continue;
                    const gap = sub(rigs[i].position, rigs[j].position);
                    if (dot(gap, gap) >= (2 * bodyRadius - 1e-7) ** 2) continue;
                    root.DroneFlightDynamics.collide(rigs[i], rigs[j], bodyRadius, 0, false);
                    moved = true;
                }
                if (!moved) break;
                root.DroneGoalPhysics?.resolve(rigs, bodyRadius);
                stabilizers.forEach(stabilize => stabilize());
            }
            for (const rig of rigs) if (rig?.velocity) root.DroneFlightDynamics?.limitSpeed(rig);
            // Sample resting contacts too: an impact callback alone disappears
            // once the cage settles on a rail or against a net.
            for (const rig of rigs) if (rig?.position) {
                for (const tube of this.tubes) {
                    const offset = sub(rig.position, closest(rig.position, tube));
                    if (Math.hypot(offset.x,offset.y,offset.z) <= tube.radius + bodyRadius + 0.0005)
                        root.DroneFlightSafety?.contact(rig,normalAt(rig.position,tube));
                }
                for (const net of this.nets) if ((rig.position[net.axis]-net.limit(rig.position,bodyRadius))*net.sign <= 0.0005) {
                    const velocity={x:0,y:0,z:0};velocity[net.axis]=net.outward*net.sampleAt(rig.position).velocity;
                    root.DroneFlightSafety?.contact(rig,net.normal(rig.position),velocity);
                }
            }
            root.DroneFlightSafety?.endContacts(rigs,this.delta);
            root.DroneGoalPhysics?.endFrame();
            root.DroneFlightDynamics?.endContacts();
        }
    }

    root.DroneArenaCollision = new Arena();
    if (typeof module !== 'undefined' && module.exports) module.exports = { Arena, closest, sweep };
})(typeof window !== 'undefined' ? window : globalThis);
