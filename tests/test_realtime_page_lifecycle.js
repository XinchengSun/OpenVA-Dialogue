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
