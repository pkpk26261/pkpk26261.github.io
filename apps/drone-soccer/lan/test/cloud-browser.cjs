'use strict';
// A real HTTPS reverse proxy and separate host/player browser contexts.
// Uses isolated test certificates and synthetic controllers; no cloud deployment.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const https = require('node:https');
const net = require('node:net');
const { randomBytes } = require('node:crypto');
const { chromium } = require('playwright');
const { start } = require('../server.cjs');

const output = path.resolve(__dirname, '../../../../artifacts/cloud-deploy-20261103/browser');
const report = { checks: [], errors: [], externalRequests: [], clientProfile: { viewport: { width: 800, height: 600 }, hasTouch: true, renderFPS: 4, graphics: 'SwANGLE' }, limitations: ['HTTPS proxy test, not a public cloud deployment.', 'Synthetic controllers, not physical hardware.', 'Simulated clients use 800x600 touch views with the documented SwANGLE software driver at 4 FPS; the authoritative engine is unmodified.'] };
// Chromium's GPU-less bot configuration:
// https://chromium.googlesource.com/chromium/src/+/main/docs/gpu/swiftshader.md
const launchClientBrowser = () => chromium.launch({ executablePath: chromium.executablePath(), headless: true, args: ['--use-gl=angle', '--use-angle=swiftshader'] });
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(fn, label, timeout = 40000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) { if (await fn()) return; await delay(75); }
  throw Error('Timeout: ' + label);
}
async function freePort() {
  const listener = net.createServer();
  await new Promise((resolve, reject) => { listener.once('error', reject); listener.listen(0, '127.0.0.1', resolve); });
  const port = listener.address().port;
  await new Promise(resolve => listener.close(resolve));
  return port;
}
async function clickEnabled(button) {
  await button.waitFor({ state: 'visible' });
  await until(() => button.isEnabled(), 'enabled user action');
  await button.evaluate(element => element.scrollIntoView({ block: 'center', inline: 'nearest' }));
  await until(() => button.evaluate(element => {
    const bounds = element.getBoundingClientRect();
    return !window.FlightPanels?.busy && element.contains(document.elementFromPoint(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2));
  }), 'uncovered user action');
  // Physical clicks do not wait for two renderer frames. Semantic clicks keep
  // independent simulated devices from sharing the driver's mouse focus.
  await button.evaluate(element => element.click());
}
async function preparePlayers(players, room) {
  for (let attempt = 0; attempt < 4; attempt++) {
    for (const [i, player] of players.entries()) {
      const seat = i ? 'b-atk' : 'r-atk';
      if ([...room.players.values()].some(value => value.seat === seat && value.ready)) continue;
      if (!await player.locator('#ready').isVisible()) await clickEnabled(player.locator('#reviewController'));
      await clickEnabled(player.locator('#ready'));
      await until(() => [...room.players.values()].some(value => value.seat === seat && value.ready), 'server confirms player preparation');
    }
    if (room.conditions()) return;
    report.preparationRetries = (report.preparationRetries || 0) + 1;
    await delay(600);
  }
  throw Error('Players cannot retain synchronized, fresh preparation.');
}
function syntheticPad() {
  window.__pads = [{ id: 'Synthetic cloud-test controller', index: 0, connected: true, mapping: 'standard', axes: [0, 0, 0, 0], buttons: Array.from({ length: 16 }, () => ({ value: 0, pressed: false, touched: false })), timestamp: 1 }];
  Object.defineProperty(navigator, 'getGamepads', { value: () => window.__pads });
}
function clientRendering() {
  // These clients share a GPU-less CI machine, unlike real players' devices.
  // Keep rendering separate from the real 32 ms controller/WSS timers.
  window.__cloudTestRenderFrames = 0;
  window.requestAnimationFrame = callback => setTimeout(() => { window.__cloudTestRenderFrames++; callback(performance.now()); }, 250);
  window.cancelAnimationFrame = id => clearTimeout(id);
}

(async () => {
  let service, proxy, browser, inputProbe;
  const playerBrowsers = [];
  const tunnels = new Set();
  fs.mkdirSync(output, { recursive: true });
  try {
    assert(process.env.LAN_TEST_CERT_DIR, 'LAN_TEST_CERT_DIR must name isolated test certificates');
    const certDir = path.resolve(process.env.LAN_TEST_CERT_DIR);
    const internalPort = await freePort(), externalPort = await freePort();
    const origin = 'https://127.0.0.1:' + externalPort;
    const adminToken = randomBytes(32).toString('base64url');
    service = await start({ cloud: true, port: internalPort, persist: false, 'public-origin': origin, 'admin-token': adminToken });
    proxy = https.createServer({ cert: fs.readFileSync(path.join(certDir, 'server.pem')), key: fs.readFileSync(path.join(certDir, 'server-key.pem')) }, (req, res) => {
      const upstream = http.request({ hostname: '127.0.0.1', port: internalPort, path: req.url, method: req.method, headers: req.headers }, reply => {
        res.writeHead(reply.statusCode, reply.headers); reply.pipe(res);
      });
      upstream.on('error', error => { if (!res.headersSent) res.writeHead(502); res.end(error.message); });
      req.pipe(upstream);
    });
    proxy.on('upgrade', (req, socket, head) => {
      const upstream = net.connect(internalPort, '127.0.0.1', () => {
        upstream.write(req.method + ' ' + req.url + ' HTTP/1.1\r\n' + req.rawHeaders.reduce((lines, value, i, rows) => i % 2 ? lines : lines + value + ': ' + rows[i + 1] + '\r\n', '') + '\r\n');
        if (head.length) upstream.write(head);
        socket.pipe(upstream); upstream.pipe(socket);
      });
      tunnels.add(socket); tunnels.add(upstream);
      for (const connection of [socket, upstream]) {
        connection.on('error', () => { socket.destroy(); upstream.destroy(); });
        connection.on('close', () => { tunnels.delete(connection); socket.destroy(); upstream.destroy(); });
      }
    });
    await new Promise((resolve, reject) => { proxy.once('error', reject); proxy.listen(externalPort, '127.0.0.1', resolve); });
    browser = await launchClientBrowser();
    const newContext = async (deviceBrowser = browser) => {
      const context = await deviceBrowser.newContext({ ignoreHTTPSErrors: true, viewport: { width: 800, height: 600 }, hasTouch: true, deviceScaleFactor: 1 });
      await context.addInitScript(clientRendering);
      context.on('page', page => {
        page.on('pageerror', error => report.errors.push(error.message));
        page.on('request', request => { if (!request.url().startsWith(origin + '/') && !/^(data|blob):/.test(request.url())) report.externalRequests.push(request.url()); });
      });
      return context;
    };
    const hostContext = await newContext(), host = await hostContext.newPage();
    const configResponse = await hostContext.request.get(origin + '/api/config');
    const config = await configResponse.json();
    assert.equal(config.joinURL, origin + '/play');
    assert.equal(config.certificateTrustRequired, false);
    assert.equal(config.publicReady, true);
    assert.equal(service.internalOrigin, 'http://127.0.0.1:' + internalPort);
    report.checks.push('HTTPS public configuration and private HTTP authority work through an actual TLS reverse proxy without server certificates.');
    await host.goto(origin + '/admin');
    await host.locator('#adminKey').fill(adminToken);
    await host.locator('#loginForm button').click();
    await host.locator('#adminDashboard').waitFor({ state: 'visible' });
    const cookies = await hostContext.cookies();
    const session = cookies.find(cookie => cookie.name === 'lan_session');
    assert(session?.secure && session.httpOnly && session.sameSite === 'Strict');
    // Exercise the existing standalone host screen using its authenticated cookie.
    await host.goto(origin + '/host');
    await host.locator('#redHumans').fill('1');
    await host.locator('#redAI').fill('0');
    await host.locator('#blueHumans').fill('1');
    await host.locator('#blueAI').fill('0');
    await host.locator('#create').click();
    await until(() => service.roomMap.size === 1, 'host creates room');
    const room = [...service.roomMap.values()][0];
    await until(() => room.engineReady && room.snapshot, 'authoritative game snapshots');
    await host.locator('#inviteClose').click();
    report.checks.push('A browser-only host logs in using a Secure/HttpOnly cookie, creates a room and starts the original authoritative game engine.');
    const players = [];
    for (const [i, seat] of ['r-atk', 'b-atk'].entries()) {
      const deviceBrowser = await launchClientBrowser();
      playerBrowsers.push(deviceBrowser);
      const context = await newContext(deviceBrowser);
      await context.addInitScript(syntheticPad);
      const player = await context.newPage(); players.push(player);
      await player.goto(origin + '/play?code=' + room.code);
      await player.locator('#nickname').fill('雲端玩家 ' + (i + 1));
      await player.locator('#seat').selectOption(seat);
      await player.locator('#joinSubmit').click();
      await until(() => [...room.players.values()].some(p => p.seat === seat && p.pad && p.cameraAck === room.camera.revision && p.viewAck === room.engineRevision), 'player ' + seat + ' scene and controller ready');
      const scene = player.frames().find(frame => frame.url().includes('lan=viewer'));
      assert(scene, 'Player has a live game view.');
      const rendering = await scene.evaluate(() => window.DroneLANEngine.rendering);
      assert.equal(rendering.software, true, 'The forced software driver selects the real lightweight viewer.');
      assert.equal(rendering.dynamicShadows, false);
      assert(rendering.scale <= 0.65);
      (report.renderProfiles ||= []).push(rendering);
    }
    const watchContext = await newContext(), watch = await watchContext.newPage();
    await watch.goto(origin + '/watch?code=' + room.code);
    await watch.locator('#scene:not([hidden])').waitFor();
    await preparePlayers(players, room);
    await clickEnabled(host.locator('#tabRoom'));
    report.inputFreshness = { samples: 0, maxAgeMilliseconds: 0 };
    inputProbe = setInterval(() => {
      if (room.status !== 'running') return;
      for (const player of room.players.values()) {
        report.inputFreshness.samples++;
        report.inputFreshness.maxAgeMilliseconds = Math.max(report.inputFreshness.maxAgeMilliseconds, Date.now() - player.lastInput);
      }
    }, 25);
    await clickEnabled(host.locator('#start'));
    for (let recovery = 0; ; recovery++) {
      await until(() => room.status === 'paused' || room.status === 'running' && room.snapshot?.match.phase === 'play', 'countdown or safety pause');
      if (room.status === 'running') break;
      assert(recovery < 2, 'The match must become stable after bounded, explicit user recovery.');
      assert(/輸入逾時|權威運算中斷/.test(room.reason), 'Only observed startup watchdog pauses may be recovered.');
      (report.startupPauses ||= []).push({ reason: room.reason, inputFreshness: { ...report.inputFreshness } });
      await until(() => room.engineReady, 'authority available for explicit resume');
      await preparePlayers(players, room);
      report.inputFreshness = { samples: 0, maxAgeMilliseconds: 0 };
      await clickEnabled(host.locator('#resume'));
    }
    await delay(1000);
    clearInterval(inputProbe); inputProbe = null;
    assert.equal(room.status, 'running');
    assert(report.inputFreshness.samples > 0 && report.inputFreshness.maxAgeMilliseconds < 600, 'Player inputs stay fresh through countdown and match play.');
    assert([...room.players.values()].every(player => player.pad && Date.now() - player.lastInput < 600));
    report.checks.push('Two independent player browsers join different seats over WSS, synchronize their 3D views, prepare and enter the official match; an independent observer watches the same room.');
    await host.locator('#pause').click();
    await until(() => room.status === 'paused', 'host pause');
    await host.screenshot({ path: path.join(output, 'host.png') });
    await players[0].screenshot({ path: path.join(output, 'player.png') });
    await watch.screenshot({ path: path.join(output, 'watch.png') });
    await host.reload();
    await until(() => host.locator('#roomCode').textContent().then(value => value === room.code), 'host reload retains room');
    assert.equal(service.roomMap.size, 1);
    report.checks.push('Host pause and browser refresh retain the same server-owned room and engine.');
    assert.deepEqual(report.errors, []);
    assert.deepEqual(report.externalRequests, []);
    report.result = 'PASS';
    console.log('CLOUD BROWSER PASS', report.checks.length);
  } catch (error) {
    report.result = 'FAIL'; report.failure = error.stack;
    report.rooms = [...(service?.roomMap.values() || [])].map(room => ({
      code: room.code, status: room.status, reason: room.reason, engineReady: room.engineReady,
      engineRevision: room.engineRevision, cameraRevision: room.camera.revision,
      players: [...room.players.values()].map(player => ({ seat: player.seat, connected: player.connected, pad: player.pad, ready: player.ready, cameraAck: player.cameraAck, viewAck: player.viewAck, lastInput: player.lastInput }))
    }));
    report.pages = [];
    const contexts = [browser, ...playerBrowsers].filter(Boolean).flatMap(deviceBrowser => deviceBrowser.contexts());
    for (const [i, context] of contexts.entries()) for (const [j, page] of context.pages().entries()) {
      const record = { url: page.url(), frames: [] };
      try {
        record.body = (await page.locator('body').innerText({ timeout: 3000 })).slice(0, 5000);
        await page.screenshot({ path: path.join(output, 'failure-' + i + '-' + j + '.png'), timeout: 10000 });
        for (const frame of page.frames()) record.frames.push(await frame.evaluate(() => ({ url: location.href, ready: document.readyState, secureContext: isSecureContext, hidden: document.hidden, focused: document.hasFocus(), canvases: document.querySelectorAll('canvas').length, gamepads: navigator.getGamepads ? Array.from(navigator.getGamepads()).filter(Boolean).map(p => ({ id: p.id, connected: p.connected, axes: p.axes.length, buttons: p.buttons.length })) : null, three: typeof THREE, lanEngine: window.DroneLANEngine ? { active: window.DroneLANEngine.active } : null })));
      } catch (diagnosticError) { record.diagnosticError = diagnosticError.message; }
      report.pages.push(record);
    }
    console.error(error.stack); process.exitCode = 1;
  } finally {
    clearInterval(inputProbe);
    await Promise.allSettled([browser, ...playerBrowsers].filter(Boolean).map(deviceBrowser => deviceBrowser.close()));
    for (const socket of tunnels) socket.destroy();
    if (proxy) await new Promise(resolve => proxy.close(resolve));
    await service?.close();
    fs.writeFileSync(path.join(output, 'report.json'), JSON.stringify(report, null, 2));
  }
})();
