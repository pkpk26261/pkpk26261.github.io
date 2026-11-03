#!/usr/bin/env node
// Build a credential-free cloud source tree; never copy the desktop installation.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const outputRoot = path.join(root, 'output');
const destination = path.join(outputRoot, 'flight-club-cloud');
const manifestName = 'cloud-manifest.json';
const format = 'flight-club-cloud-source-v1';
const args = process.argv.slice(2);
if (args.length > 1 || args.some(arg => arg !== '--verify')) {
  throw Error('Usage: node scripts/package-cloud.mjs [--verify]');
}

const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const portable = value => value.split(path.sep).join('/');
const exists = value => fs.existsSync(value);
const skip = name => name.startsWith('.') || name === 'Thumbs.db' || name === 'node_modules' || name === '__pycache__';

function assertUnder(parent, target) {
  const relative = path.relative(parent, target);
  if (!relative || relative.startsWith('..') || path.isAbsolute(relative)) {
    throw Error('Refusing a path outside the expected directory: ' + target);
  }
}

function assertNoSymlinks(target) {
  assertUnder(root, target);
  let current = root;
  for (const segment of path.relative(root, target).split(path.sep)) {
    current = path.join(current, segment);
    if (exists(current) && fs.lstatSync(current).isSymbolicLink()) {
      throw Error('Symlinks are not allowed in the cloud package: ' + current);
    }
  }
}

function filesBelow(directory) {
  assertNoSymlinks(directory);
  return fs.readdirSync(directory, { withFileTypes: true })
    .sort((a, b) => a.name.localeCompare(b.name, 'en'))
    .filter(entry => !skip(entry.name))
    .flatMap(entry => {
      const full = path.join(directory, entry.name);
      if (entry.isSymbolicLink()) throw Error('Symlinks are not allowed: ' + full);
      return entry.isDirectory() ? filesBelow(full) : entry.isFile() ? [full] : [];
    });
}

function manifestAt(directory) {
  const manifestPath = path.join(directory, manifestName);
  assertNoSymlinks(manifestPath);
  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
  if (manifest.format !== format || !Array.isArray(manifest.files)) {
    throw Error('The existing output is not a generated Flight Club cloud package.');
  }
  return manifest;
}

function verify(directory, manifest = manifestAt(directory)) {
  const actual = [];
  function visit(folder) {
    for (const entry of fs.readdirSync(folder, { withFileTypes: true })) {
      const full = path.join(folder, entry.name);
      if (entry.isSymbolicLink()) throw Error('Unexpected symlink in cloud output: ' + full);
      if (entry.isDirectory()) visit(full);
      else if (entry.isFile() && full !== path.join(directory, manifestName)) actual.push(portable(path.relative(directory, full)));
      else if (!entry.isFile()) throw Error('Unexpected file type in cloud output: ' + full);
    }
  }
  visit(directory);
  actual.sort();
  const expected = manifest.files.map(entry => entry.path).sort();
  if (JSON.stringify(actual) !== JSON.stringify(expected)) throw Error('Cloud output contains missing or unexpected files.');
  for (const entry of manifest.files) {
    const full = path.resolve(directory, entry.path);
    assertUnder(directory, full);
    const bytes = fs.readFileSync(full);
    if (bytes.length !== entry.bytes || digest(bytes) !== entry.sha256) throw Error('Cloud package checksum mismatch: ' + entry.path);
  }
  return manifest;
}

assertNoSymlinks(destination);
if (args.includes('--verify')) {
  const manifest = verify(destination);
  console.log(JSON.stringify({ directory: destination, verified: true, files: manifest.files.length, bytes: manifest.bytes }, null, 2));
  process.exit(0);
}

const exact = [
  'index.html', 'Dockerfile', '.dockerignore', 'render.yaml',
  '.github/workflows/cloud-check.yml', 'scripts/package-cloud.mjs',
  'apps/drone-soccer/index.html', 'apps/drone-soccer/README.md',
  'apps/drone-soccer/lan/package.json', 'apps/drone-soccer/lan/pnpm-lock.yaml',
  'apps/drone-soccer/lan/README.md', 'apps/drone-soccer/lan/setup.html',
  'apps/drone-soccer/lan/test/cloud.cjs', 'apps/drone-soccer/lan/test/cloud-browser.cjs',
  'apps/drone-soccer/cloud/README.md', 'apps/drone-soccer/cloud/.env.example', 'apps/drone-soccer/cloud/help.html',
  'vendor/drone-soccer-dependencies.md'
];
const directories = [
  'apps/drone-soccer/assets', 'apps/drone-soccer/games', 'apps/drone-soccer/lan/public',
  'vendor/three-r128', 'vendor/drone-soccer-fonts', 'vendor/fontawesome-6.4.0', 'vendor/tailwindcss-3.4.17'
];
for (const relative of [...exact, ...directories]) {
  const full = path.join(root, relative);
  assertNoSymlinks(full);
  if (!exists(full)) throw Error('Missing cloud source input: ' + relative);
}

const inputs = new Set(exact.map(relative => path.join(root, relative)));
for (const directory of directories) for (const full of filesBelow(path.join(root, directory))) inputs.add(full);
for (const entry of fs.readdirSync(path.join(root, 'apps/drone-soccer/lan'), { withFileTypes: true })) {
  if (entry.isFile() && !skip(entry.name) && entry.name.endsWith('.cjs')) inputs.add(path.join(root, 'apps/drone-soccer/lan', entry.name));
}
// Preserve any repository-level attribution files when present.
for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
  if (entry.isFile() && /^(?:licen[sc]e|copying|notice)(?:[.-].*)?$/i.test(entry.name)) inputs.add(path.join(root, entry.name));
}

const planned = [...inputs].map(full => ({ full, relative: portable(path.relative(root, full)) })).sort((a, b) => a.relative.localeCompare(b.relative, 'en'));
const forbidden = /(?:^|\/)(?:runtime|desktop|node_modules|test|tests|tools|artifacts|\.codex-work|\.git|__pycache__)(?:\/|$)|\.(?:pem|key|pfx|p12|exe|pyc)$/i;
// CI runs these two focused checks against the image via bind mounts. Docker's
// allowlist deliberately keeps them out of the production image itself.
const ciTests = new Set(['apps/drone-soccer/lan/test/cloud.cjs', 'apps/drone-soccer/lan/test/cloud-browser.cjs']);
for (const entry of planned) {
  assertNoSymlinks(entry.full);
  if ((forbidden.test(entry.relative) && !ciTests.has(entry.relative)) || /(?:^|\/)state\.json(?:\.bak)?$/i.test(entry.relative)) throw Error('Forbidden deployment input: ' + entry.relative);
}

fs.mkdirSync(outputRoot, { recursive: true });
assertNoSymlinks(outputRoot);
// Only the fixed generated directory may be replaced, and only after its prior
// manifest proves that it has no unrelated files to accidentally delete.
if (exists(destination)) verify(destination);
const staging = fs.mkdtempSync(path.join(outputRoot, '.flight-club-cloud-build-'));
function removeStaging() {
  assertUnder(outputRoot, staging);
  if (!path.basename(staging).startsWith('.flight-club-cloud-build-')) throw Error('Unexpected staging path.');
  assertNoSymlinks(staging);
  fs.rmSync(staging, { recursive: true, force: true });
}
try {
  const entries = [];
  for (const input of planned) {
    const bytes = fs.readFileSync(input.full);
    const target = path.join(staging, input.relative);
    assertUnder(staging, target);
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.writeFileSync(target, bytes);
    entries.push({ path: input.relative, bytes: bytes.length, sha256: digest(bytes) });
  }
  const manifest = { format, bytes: entries.reduce((sum, entry) => sum + entry.bytes, 0), files: entries };
  fs.writeFileSync(path.join(staging, manifestName), JSON.stringify(manifest, null, 2) + '\n');
  verify(staging, manifest);
  for (const input of planned) {
    const source = fs.readFileSync(input.full);
    const entry = entries.find(entry => entry.path === input.relative);
    if (digest(source) !== entry.sha256) throw Error('Source changed during packaging; rerun after edits finish: ' + input.relative);
  }
  if (exists(destination)) {
    assertUnder(outputRoot, destination);
    if (path.basename(destination) !== 'flight-club-cloud') throw Error('Unexpected output path.');
    assertNoSymlinks(destination);
    verify(destination);
    fs.rmSync(destination, { recursive: true });
  }
  fs.renameSync(staging, destination);
  console.log(JSON.stringify({ directory: destination, files: entries.length, bytes: manifest.bytes, megabytes: Number((manifest.bytes / 1024 / 1024).toFixed(2)), manifest: path.join(destination, manifestName) }, null, 2));
} finally {
  if (exists(staging)) removeStaging();
}
