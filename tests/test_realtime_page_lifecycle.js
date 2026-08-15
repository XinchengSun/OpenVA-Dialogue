const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const html = fs.readFileSync(
  path.join(__dirname, '..', 'static', 'index.html'),
  'utf8'
);
const scriptMatch = html.match(/<script>([\s\S]*)<\/script>/);

function section(start, end) {
  const startIndex = html.indexOf(start);
  const endIndex = html.indexOf(end, startIndex + start.length);
  assert.notEqual(startIndex, -1, `missing section start: ${start}`);
  assert.notEqual(endIndex, -1, `missing section end: ${end}`);
  return html.slice(startIndex, endIndex);
}

function numericConstant(name) {
  const match = html.match(new RegExp(`const ${name} = ([0-9.]+);`));
  assert.ok(match, `missing numeric constant: ${name}`);
  return Number(match[1]);
}

test('realtime page script parses', () => {
  assert.ok(scriptMatch, 'inline script must exist');
  new vm.Script(scriptMatch[1]);
});

test('leaving for customization releases all realtime resources', () => {
  assert.match(html, /id="customizeLink"/);
  const release = section('function releaseRealtimePage()', 'startBtn.onclick');
  assert.match(release, /stopMic\(\);/);
  assert.match(release, /mediaWs = null;/);
  assert.match(release, /mediaWsGeneration \+= 1;/);
  assert.match(release, /closePageSocket\(oldMediaWs\);/);
  assert.match(release, /logWs = null;/);
  assert.match(release, /logWsGeneration \+= 1;/);
  assert.match(release, /closePageSocket\(oldLogWs\);/);
  assert.match(release, /teardownMSE\(\);/);
  assert.match(release, /customizeLink\.addEventListener\('click', releaseRealtimePage\);/);
  assert.match(release, /window\.addEventListener\('pagehide', releaseRealtimePage\);/);
});

test('MSE keeps enough live-edge reserve for a short public-network gap', () => {
  const targetLatencySec = numericConstant('TARGET_LATENCY_SEC');
  const lowBufferEnterSec = numericConstant('LOW_BUFFER_ENTER_SEC');
  const lowBufferExitSec = numericConstant('LOW_BUFFER_EXIT_SEC');
  const lowBufferPlaybackRate = numericConstant('LOW_BUFFER_PLAYBACK_RATE');

  // A 60-second public-media probe observed gaps up to 0.476 s. Keep useful
  // scheduling headroom beyond that gap without returning to a deep buffer.
  assert.ok(targetLatencySec >= 0.60);
  assert.ok(lowBufferEnterSec >= 0.48);
  assert.equal(lowBufferExitSec, targetLatencySec);
  assert.ok(lowBufferPlaybackRate >= 0.90 && lowBufferPlaybackRate <= 0.95);

  const recovery = section('function liveTail(slot)', 'function appendNextSegment(slot)');
  assert.match(recovery, /latency < LOW_BUFFER_ENTER_SEC/);
  assert.match(recovery, /latency >= LOW_BUFFER_EXIT_SEC/);
  assert.match(recovery, /desiredRate = LOW_BUFFER_PLAYBACK_RATE/);
  assert.doesNotMatch(recovery, /video\.pause\(\)/);
});
