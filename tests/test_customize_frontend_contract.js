const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const html = fs.readFileSync(
  path.join(__dirname, '..', 'static', 'customize.html'),
  'utf8'
);
const scriptMatch = html.match(/<script>([\s\S]*)<\/script>/);
const loginHtml = fs.readFileSync(
  path.join(__dirname, '..', 'static', 'customize_login.html'),
  'utf8'
);
const loginScriptMatch = loginHtml.match(/<script>([\s\S]*)<\/script>/);

function section(start, end) {
  const startIndex = html.indexOf(start);
  const endIndex = html.indexOf(end, startIndex + start.length);
  assert.notEqual(startIndex, -1, `missing section start: ${start}`);
  assert.notEqual(endIndex, -1, `missing section end: ${end}`);
  return html.slice(startIndex, endIndex);
}

test('customization page script parses', () => {
  assert.ok(scriptMatch, 'inline script must exist');
  new vm.Script(scriptMatch[1]);
});

test('one-link login strips the fragment before submitting the credential', () => {
  assert.ok(loginScriptMatch, 'login inline script must exist');
  new vm.Script(loginScriptMatch[1]);
  const stripIndex = loginHtml.indexOf('history.replaceState');
  const submitIndex = loginHtml.indexOf('void submitToken(sharedToken)');
  assert.notEqual(stripIndex, -1);
  assert.notEqual(submitIndex, -1);
  assert.ok(stripIndex < submitIndex, 'fragment must be removed before login starts');
  assert.match(loginHtml, /const rawFragment = window\.location\.hash;/);
  assert.match(loginHtml, /new URLSearchParams\(rawFragment\.slice\(1\)\)/);
  assert.match(loginHtml, /fragmentParams\.get\('token'\)/);
  assert.match(loginHtml, /fetch\('\/api\/customization\/session'/);
  assert.doesNotMatch(loginHtml, /localStorage|sessionStorage/);
});

test('one-link login exchanges the fragment through POST after removing it', async () => {
  const token = 'a'.repeat(64);
  const events = [];
  const elements = {
    loginForm: {addEventListener() {}},
    adminToken: {value: ''},
    submitBtn: {disabled: false},
    status: {textContent: ''}
  };
  const context = {
    document: {getElementById: id => elements[id]},
    URLSearchParams,
    history: {
      replaceState(_state, _title, url) {
        events.push(['strip', url]);
      }
    },
    window: {
      location: {
        hash: `#token=${token}`,
        pathname: '/customize/login',
        search: '',
        replace(url) {
          events.push(['redirect', url]);
        }
      }
    },
    async fetch(url, options) {
      events.push(['fetch', url, options]);
      return {ok: true, status: 200};
    }
  };

  vm.runInNewContext(loginScriptMatch[1], context);
  await new Promise(resolve => setImmediate(resolve));

  assert.equal(events[0][0], 'strip');
  assert.equal(events[0][1], '/customize/login');
  assert.equal(events[1][0], 'fetch');
  assert.equal(events[1][1], '/api/customization/session');
  assert.equal(events[1][2].method, 'POST');
  assert.equal(JSON.parse(events[1][2].body).token, token);
  assert.equal(events[2][0], 'redirect');
  assert.equal(events[2][1], '/customize');
  assert.equal(elements.adminToken.value, '');
});

test('failed one-link login stays stripped and a reload does not resubmit', async () => {
  const token = 'b'.repeat(64);
  let fetchCount = 0;
  let strippedUrl = null;
  const makeContext = hash => {
    const elements = {
      loginForm: {addEventListener() {}},
      adminToken: {value: ''},
      submitBtn: {disabled: false},
      status: {textContent: ''}
    };
    const location = {
      hash,
      pathname: '/customize/login',
      search: '',
      replace() {
        throw new Error('failed login must not redirect');
      }
    };
    return {
      elements,
      context: {
        document: {getElementById: id => elements[id]},
        URLSearchParams,
        history: {
          replaceState(_state, _title, url) {
            strippedUrl = url;
            location.hash = '';
          }
        },
        window: {location},
        async fetch() {
          fetchCount += 1;
          return {ok: false, status: 401};
        }
      }
    };
  };

  const first = makeContext(`#token=${token}`);
  vm.runInNewContext(loginScriptMatch[1], first.context);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(strippedUrl, '/customize/login');
  assert.equal(fetchCount, 1);
  assert.equal(first.elements.adminToken.value, '');

  const reload = makeContext('');
  vm.runInNewContext(loginScriptMatch[1], reload.context);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(fetchCount, 1, 'a stripped reload must not submit again');
});

test('customization page exposes provider and language selectors', () => {
  for (const id of ['ttsBackend', 'referenceLanguage', 'targetLanguage']) {
    assert.match(html, new RegExp(`id="${id}"`));
  }
  assert.match(html, /data\.tts_options/);
  assert.match(html, /data\.tts_backends/);
  assert.match(html, /data\.language_options/);
  assert.match(html, /option\.disabled = !selectable/);
  assert.match(html, /backend\.disabled_reason/);
});

test('prepare and activation preserve all selected voice settings', () => {
  for (const field of ['tts_backend', 'reference_language', 'target_language']) {
    assert.match(html, new RegExp(`form\\.append\\('${field}'`));
    assert.match(html, new RegExp(`${field}:`));
  }
  assert.match(html, /transcript: transcript\.value\.trim\(\)/);
});

test('prepare waits for explicit confirmation without regressing status checks', () => {
  assert.match(html, /restorePrepared\(result\);/);
  assert.doesNotMatch(
    html,
    /restorePrepared\(result\);\s*await submitActivation\(currentJob\)/
  );
  assert.match(html, /\['auto', 'sensevoice_auto'\]\.includes\(prepared\.transcript_source\)/);
  assert.match(html, /确认文本并启用/);
  assert.match(html, /checking: '状态待确认'/);
  assert.match(html, /“立即检查状态”只查询状态，不会重复启用/);
});

test('lost connectivity and the five-minute limit converge on low-frequency checking', () => {
  const polling = section('async function pollJob', 'async function submitActivation');
  assert.match(html, /const STATUS_CHECK_POLL_MS = 10 \* 1000;/);
  assert.match(polling, /phase = 'checking';\s*continueCheckingAtActivationLimit\(jobId\);/);
  assert.match(polling, /phase === 'checking' \? STATUS_CHECK_POLL_MS : 2200/);
  assert.match(
    polling,
    /catch \(_\) \{[\s\S]*?phase = 'checking';[\s\S]*?showStatusChecking\([\s\S]*?'服务正在切换，正在等待恢复连接…'/
  );
  assert.doesNotMatch(html, /timed_out|自动检查已停止|stopAtActivationLimit/);
});

test('successful job GET restores availability and initialization does not invent activation', () => {
  const fetchJob = section('async function fetchJob', 'async function fetchActiveSnapshot');
  const initialize = section('async function initialize', "window.addEventListener('beforeunload'");
  assert.equal((fetchJob.match(/markServiceConnected\(\);/g) || []).length, 2);
  assert.match(html, /if \(phase === 'activating'\) ensureActivationStartedAt\(jobId\);/);
  assert.doesNotMatch(html, /phase === 'checking'\) ensureActivationStartedAt\(jobId\)/);
  assert.match(
    initialize,
    /not infer an activation state until persisted status is available\.[\s\S]*?pollJob\(currentJob, 'checking'\);/
  );
});

test('automatic and manual GET checks converge on authoritative server states', () => {
  const polling = section('async function pollJob', 'async function submitActivation');
  const manualCheck = section("checkStatusBtn.addEventListener('click'", 'async function loadActive');
  for (const state of ['ready', 'failed', 'prepared']) {
    assert.match(polling, new RegExp(`status\\.state === '${state}'`));
    assert.match(manualCheck, new RegExp(`status\\.state === '${state}'`));
  }
  assert.match(manualCheck, /const status = await fetchJob\(jobId\);/);
  assert.doesNotMatch(manualCheck, /method:\s*'POST'|submitActivation\(/);
});

test('rollback copy describes an attempted recovery', () => {
  assert.match(html, /rolling_back: '尝试恢复上一版本'/);
  assert.match(html, /启用失败时，系统会尝试恢复当前数字人/);
});
