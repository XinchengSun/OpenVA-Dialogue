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

test('one client id gates status, microphone, and media', () => {
  assert.match(html, /fetch\(withClientId\('\/api\/realtime\/status'\)/);
  assert.match(html, /wsUrl\(withClientId\('\/ws\/mic'\)\)/);
  assert.match(html, /wsUrl\(withClientId\('\/ws\/media'\)\)/);
  assert.match(html, /encodeURIComponent\(realtimeClientId\)/);
});

test('microphone lease is granted before media or device capture starts', () => {
  const startMic = section('async function startMic(', 'function closeRealtimeAuxiliarySockets()');
  const admission = startMic.indexOf('await waitForMicAdmission');
  const media = startMic.indexOf('connectMedia();');
  const logs = startMic.indexOf('connectLogs();');
  const capture = startMic.indexOf('getUserMedia');
  assert.ok(admission >= 0);
  assert.ok(media > admission);
  assert.ok(logs > admission);
  assert.ok(capture > media);

  const clickHandler = section('startBtn.onclick = async () =>', 'stopBtn.onclick');
  assert.doesNotMatch(clickHandler, /connectMedia\(\)/);
  assert.doesNotMatch(clickHandler, /connectLogs\(\)/);
  assert.match(clickHandler, /await startMic\(generation\)/);
});

test('customization and unavailable states cannot enable Start', () => {
  const functions = section(
    'function syncStartButtonWithCapacity()',
    'function stopRealtimeStatusPolling()'
  );
  const startBtn = {};
  const capacityStatusEl = { dataset: {} };
  const api = vm.runInNewContext(`
    let latestRealtimeStatus = null;
    let startInProgress = false;
    let appStarted = false;
    let micLeaseOwned = false;
    ${functions}
    ({ renderRealtimeStatus });
  `, { startBtn, capacityStatusEl, Number, Math });

  api.renderRealtimeStatus({
    service_ready: true,
    phase: 'customizing',
    conversation: { state: 'unavailable' },
    media_clients: 0,
    speech: {},
  });
  assert.equal(startBtn.disabled, true);
  assert.equal(startBtn.textContent, '数字人定制中');
  assert.match(capacityStatusEl.textContent, /正在更换数字人，服务暂不可用/);

  api.renderRealtimeStatus({
    service_ready: true,
    phase: 'ready',
    conversation: { state: 'unavailable' },
    media_clients: 0,
    speech: {},
  });
  assert.equal(startBtn.disabled, true);
  assert.equal(startBtn.textContent, '服务暂不可用');
});

test('service busy is mapped to unavailable before any downstream startup', () => {
  const connectMic = section('function connectMicWS(', 'function floatToPcm16LE');
  assert.match(connectMic, /obj\.type === 'service_busy'/);
  assert.match(connectMic, /error\.code = serviceBusy \? 'SERVICE_BUSY' : 'LEASE_BUSY'/);
  assert.doesNotMatch(connectMic, /connectMedia\(\)|connectLogs\(\)|getUserMedia/);
});

test('status polling starts on load and stops on page release', () => {
  assert.match(html, /ensureIdlePlaying\(\);\s*startRealtimeStatusPolling\(\);\s*<\/script>/);
  const release = section('function releaseRealtimePage()', 'startBtn.onclick');
  assert.match(release, /stopRealtimeStatusPolling\(\);/);
  const visibility = section('function handleMediaVisibilityChange()', 'function connectMedia()');
  assert.match(visibility, /scheduleRealtimeStatusPoll\(realtimeStatusGeneration, 0\);/);
});
