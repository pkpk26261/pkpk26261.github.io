/* Local procedural quadrotor audio. Power comes from the flight controller's
   final throttle, so altitude hold and manual flight use the same response. */
(function (global) {
  'use strict';
  const buses = new WeakMap();
  const clamp = (value, lo, hi) => Math.min(hi, Math.max(lo, Number.isFinite(value) ? value : 0));
  const glide = (param, value, time, seconds) => param.setTargetAtTime(value, time, seconds);

  function getBus(context) {
    if (buses.has(context)) return buses.get(context);
    const mix = context.createGain();
    const limiter = context.createDynamicsCompressor();
    limiter.threshold.value = -10;
    limiter.knee.value = 6;
    limiter.ratio.value = 8;
    limiter.attack.value = 0.004;
    limiter.release.value = 0.15;
    const output = context.createGain();
    output.gain.value = typeof document !== 'undefined' && document.hidden ? 0 : 1;
    mix.connect(limiter).connect(output).connect(context.destination);
    const bus = { mix, output, voices: new Set(), serial: 0 };
    if (typeof document !== 'undefined') {
      document.addEventListener('visibilitychange', () => {
        glide(output.gain, document.hidden ? 0 : 1, context.currentTime, 0.025);
      });
      global.addEventListener('pagehide', () => {
        // The animation loop may stop before it can send another throttle update.
        output.gain.cancelScheduledValues(context.currentTime);
        output.gain.setValueAtTime(0, context.currentTime);
      });
      global.addEventListener('pageshow', () => {
        glide(output.gain, document.hidden ? 0 : 1, context.currentTime, 0.025);
      });
    }
    buses.set(context, bus);
    return bus;
  }

  function create(context) {
    const bus = getBus(context);
    const serial = bus.serial++;
    const sources = [];
    const nodes = [];
    const master = context.createGain();
    master.gain.value = 0;
    master.connect(bus.mix);
    nodes.push(master);
    // A rounded blade-passage waveform supplies the body without a raw saw buzz.
    const harmonics = new Float32Array([0, 0.8, 0.38, 0.2, 0.11, 0.065, 0.035, 0.018]);
    const wave = context.createPeriodicWave(new Float32Array(harmonics.length), harmonics);
    const motors = Array.from({ length: 4 }, (_, index) => {
      const blade = context.createOscillator();
      blade.setPeriodicWave(wave);
      blade.frequency.value = 55;
      const body = context.createGain();
      body.gain.value = 0.19;
      const whine = context.createOscillator();
      whine.type = 'sine';
      whine.frequency.value = 385;
      const electrical = context.createGain();
      electrical.gain.value = 0.018;
      const pan = context.createStereoPanner();
      pan.pan.value = index % 2 ? 0.18 : -0.18;
      blade.connect(body).connect(pan);
      whine.connect(electrical).connect(pan);
      pan.connect(master);
      // Independent, subtle RPM drift avoids four perfectly locked electronic tones.
      const drift = context.createOscillator();
      drift.frequency.value = 8.7 + index * 1.13 + serial * 0.17;
      const depth = context.createGain();
      depth.gain.value = 1.5;
      drift.connect(depth).connect(blade.frequency);
      const at = context.currentTime + index * 0.003;
      blade.start(at); whine.start(at); drift.start(at);
      sources.push(blade, whine, drift);
      nodes.push(blade, body, whine, electrical, pan, drift, depth);
      return { blade, whine, electrical, depth, offset: (index - 1.5) * 0.007 + serial * 0.0013 };
    });

    // A continuous, deterministic noise bed models prop wash; no sample download.
    const buffer = context.createBuffer(1, context.sampleRate * 2, context.sampleRate);
    const data = buffer.getChannelData(0);
    let seed = 9173 + serial * 131;
    for (let i = 0; i < data.length; i++) {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      data[i] = seed / 2147483648 - 1;
    }
    const wash = context.createBufferSource();
    wash.buffer = buffer;
    wash.loop = true;
    const highpass = context.createBiquadFilter();
    highpass.type = 'highpass'; highpass.frequency.value = 170; highpass.Q.value = 0.5;
    const lowpass = context.createBiquadFilter();
    lowpass.type = 'lowpass'; lowpass.frequency.value = 750; lowpass.Q.value = 0.5;
    const air = context.createGain();
    air.gain.value = 0.1;
    wash.connect(highpass).connect(lowpass).connect(air).connect(master);
    wash.start();
    sources.push(wash);
    nodes.push(wash, highpass, lowpass, air);
    let lastPower = 0;
    let disposed = false;
    const voice = {
      update(input, enabled) {
        if (disposed) return;
        const power = clamp(input && input.throttle, 0, 1);
        const spinning = power > 0.02;
        const now = context.currentTime;
        const response = power > lastPower ? 0.12 : 0.22;
        // Nonlinear loudness makes increased thrust clearly audible.
        glide(master.gain, enabled && spinning ? 0.018 + 0.24 * Math.pow(power, 1.45) : 0,
          now, !enabled ? 0.018 : spinning ? 0.07 : 0.12);
        const bladeHz = spinning ? 75 + 440 * Math.sqrt(power) : 45;
        const pitch = clamp(input && input.pitch, -1, 1);
        const roll = clamp(input && input.roll, -1, 1);
        const yaw = clamp(input && input.yaw, -1, 1);
        motors.forEach((motor, index) => {
          const load = (index < 2 ? pitch : -pitch) + (index % 2 ? roll : -roll)
            + (index === 0 || index === 3 ? yaw : -yaw) * 0.5;
          const hz = bladeHz * (1 + motor.offset + load * 0.018);
          glide(motor.blade.frequency, hz, now, response);
          glide(motor.whine.frequency, hz * 7.03, now, response);
          glide(motor.electrical.gain, 0.008 + power * 0.018, now, 0.1);
          glide(motor.depth.gain, 0.6 + power * 2.4, now, 0.1);
        });
        glide(lowpass.frequency, 650 + power * 4400, now, response);
        glide(air.gain, 0.09 + power * 0.48, now, 0.1);
        lastPower = power;
      },
      dispose() {
        if (disposed) return;
        disposed = true;
        sources.forEach(source => source.stop());
        nodes.forEach(node => node.disconnect());
        bus.voices.delete(voice);
        glide(bus.mix.gain, 0.9 / Math.sqrt(Math.max(1, bus.voices.size)), context.currentTime, 0.05);
      }
    };
    bus.voices.add(voice);
    glide(bus.mix.gain, 0.9 / Math.sqrt(bus.voices.size), context.currentTime, 0.05);
    return voice;
  }
  global.DroneBrushlessAudio = Object.freeze({ create });
})(window);
