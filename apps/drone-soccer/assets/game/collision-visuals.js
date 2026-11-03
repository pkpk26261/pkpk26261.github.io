/* Hard-plastic cages, rubber and rope do not emit collision sparks.
   Keep the legacy solver adapter: physical recovery must work independently
   of optional visual feedback, audio settings and rendering availability. */
(function (global) {
  'use strict';
  const owners = new WeakMap();
  function create(options) {
    const actors = options.actors.filter(Boolean);
    const feedback = {
      hit() { return false; },
      wrap(callback) {
        return function (...args) {
          const before = actors.map(actor => ({ actor,
            revision: global.DroneFlightDynamics?.collisionRevision(actor) || 0,
            vx: actor.velocity.x, vy: actor.velocity.y, vz: actor.velocity.z }));
          const result = callback.apply(this, args);
          for (const state of before) {
            const actor = state.actor;
            const impulse = { x: actor.velocity.x - state.vx,
              y: actor.velocity.y - state.vy, z: actor.velocity.z - state.vz };
            if (Math.hypot(impulse.x, impulse.y, impulse.z) > 0.08 &&
                (global.DroneFlightDynamics?.collisionRevision(actor) || 0) === state.revision) {
              global.DroneFlightDynamics?.impact(actor, impulse);
            }
          }
          return result;
        };
      },
      wall(actor, axis) {
        const impulse = { x: 0, y: 0, z: 0 };
        impulse[axis] = -(1 + global.DroneFlightDynamics.contactMaterials.rubber.restitution) * actor.velocity[axis];
        global.DroneFlightDynamics?.impact(actor, impulse);
        return false;
      },
      update() {}
    };
    actors.forEach(actor => owners.set(actor, feedback));
    return feedback;
  }
  function wall(actor, axis) {
    return owners.get(actor)?.wall(actor, axis) || false;
  }
  global.DroneCollisionVisuals = Object.freeze({ create, hit: () => false, wall });
})(window);
