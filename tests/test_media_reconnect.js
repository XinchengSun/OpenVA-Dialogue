const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const html = fs.readFileSync(
  path.join(__dirname, '..', 'static', 'index.html'),
  'utf8'
);

function section(start, end) {
  const startIndex = html.indexOf(start);
  const endIndex = html.indexOf(end, startIndex + start.length);
  assert.notEqual(startIndex, -1, `missing section start: ${start}`);
  assert.notEqual(endIndex, -1, `missing section end: ${end}`);
  return html.slice(startIndex, endIndex);
}

function makeHarness() {
  const sockets = [];
  const timers = new Map();
  const intervals = new Map();
  let timerId = 0;
  let intervalId = 0;
  let teardownCount = 0;
  let now = 0;
  const document = { visibilityState: 'visible' };

  class FakeWebSocket {
    static CONNECTING = 0;
    static OPEN = 1;
    static CLOSING = 2;
    static CLOSED = 3;

    constructor(url) {
      this.url = url;
      this.readyState = FakeWebSocket.CONNECTING;
      this.closeCalls = [];
      sockets.push(this);
    }

    close(code, reason) {
      this.closeCalls.push({ code, reason });
      this.readyState = FakeWebSocket.CLOSING;
    }
  }

  const mediaFunctions = section(
    'function cancelMediaReconnect()',
    'function connectMicWS('
  );
  const context = {
    WebSocket: FakeWebSocket,
    wsUrl: value => value,
    withClientId: value => `${value}?client_id=test-client-123456`,
    realtimeStatusGeneration: 0,
    scheduleRealtimeStatusPoll() {},
    setTimeout(callback, delay) {
      const id = ++timerId;
      timers.set(id, { callback, delay });
      return id;
    },
    clearTimeout(id) {
      timers.delete(id);
    },
    setInterval(callback, delay) {
      const id = ++intervalId;
      intervals.set(id, { callback, delay });
      return id;
    },
    clearInterval(id) {
      intervals.delete(id);
    },
    performance: { now: () => now },
    document,
    log() {},
    setStatus() {},
    teardownMSE() { teardownCount += 1; },
    ensureIdlePlaying() {},
    resetAssistantAudioGate() {},
  };
  const api = vm.runInNewContext(`
    let mediaWs = null;
    let mediaWsGeneration = 0;
    let mediaReconnectTimer = null;
    let mediaConnectTimer = null;
    let mediaReconnectAttempt = 0;
    let mediaWatchdogTimer = null;
    let lastMediaBytesAt = 0;
    let appStarted = true;
    let startInProgress = false;
    let currentStreamEpoch = 0;
    let expectedStreamGeneration = -1;
    let highestAssistantTurnId = -1;
    let lastBoundaryEventSeq = -1;
    let ingestSlot = null;
    let activeSlot = null;
    let logWs = null;
    const MEDIA_RECONNECT_BASE_DELAY_MS = 250;
    const MEDIA_RECONNECT_MAX_DELAY_MS = 2000;
    const MEDIA_CONNECT_TIMEOUT_MS = 5000;
    const MEDIA_STALL_TIMEOUT_MS = 2000;
    const MEDIA_WATCHDOG_INTERVAL_MS = 500;
    function maybePlay() {}
    ${mediaFunctions}
    ({
      connectMedia,
      stopAndInvalidate() {
        appStarted = false;
        mediaWsGeneration += 1;
        cancelMediaReconnect();
        cancelMediaConnectTimeout();
        cancelMediaWatchdog();
        mediaReconnectAttempt = 0;
      },
      pauseConversationKeepMedia() {
        appStarted = false;
        cancelMediaReconnect();
        cancelMediaConnectTimeout();
        cancelMediaWatchdog();
      },
      resumeConversation() {
        appStarted = true;
        return connectMedia();
      },
      checkCurrentWatchdog() {
        if (mediaWs) checkMediaWatchdog(mediaWs, mediaWsGeneration);
      },
      handleMediaVisibilityChange,
      generation: () => mediaWsGeneration,
      reconnectAttempt: () => mediaReconnectAttempt,
    });
  `, context);

  return {
    api,
    sockets,
    timers,
    intervals,
    document,
    teardownCount: () => teardownCount,
    advanceClock(milliseconds) {
      now += milliseconds;
    },
    runNextTimer() {
      const next = timers.entries().next().value;
      assert.ok(next, 'expected a pending reconnect timer');
      const [id, timer] = next;
      timers.delete(id);
      timer.callback();
      return timer.delay;
    },
    runWatchdog() {
      const next = intervals.values().next().value;
      assert.ok(next, 'expected an active media watchdog');
      next.callback();
      return next.delay;
    },
  };
}

test('closed media socket reconnects and stale generation cannot reconnect', () => {
  const harness = makeHarness();
  const first = harness.api.connectMedia();
  assert.match(first.url, /\/ws\/media\?client_id=test-client-123456$/);
  first.readyState = harness.sockets[0].constructor.CLOSED;
  first.onclose();

  assert.equal(harness.teardownCount(), 1);
  assert.equal(harness.timers.size, 1);
  assert.equal(harness.runNextTimer(), 250);
  assert.equal(harness.sockets.length, 2);
  assert.equal(harness.api.generation(), 2);

  first.onclose();
  // The replacement socket owns only its 5 s connect timeout. The stale
  // socket must not schedule another reconnect.
  assert.equal(harness.timers.size, 1);
  assert.equal(harness.sockets.length, 2);
});

test('explicit lifecycle invalidation cancels a queued media reconnect', () => {
  const harness = makeHarness();
  const first = harness.api.connectMedia();
  first.readyState = harness.sockets[0].constructor.CLOSED;
  first.onclose();
  assert.equal(harness.timers.size, 1);

  harness.api.stopAndInvalidate();

  assert.equal(harness.timers.size, 0);
  assert.equal(harness.api.generation(), 2);
  assert.equal(harness.sockets.length, 1);
});

test('visible media stall rebuilds websocket and MSE within watchdog bound', () => {
  const harness = makeHarness();
  const first = harness.api.connectMedia();
  first.readyState = first.constructor.OPEN;
  first.onopen();
  assert.equal(harness.intervals.size, 1);

  harness.advanceClock(2100);
  assert.equal(harness.runWatchdog(), 500);

  assert.equal(first.closeCalls.length, 1);
  assert.equal(first.closeCalls[0].code, 4000);
  assert.equal(harness.teardownCount(), 1);
  assert.equal(harness.intervals.size, 0);
  assert.equal(harness.timers.size, 1);
  assert.equal(harness.api.generation(), 2);
  assert.equal(harness.runNextTimer(), 250);
  assert.equal(harness.sockets.length, 2);

  first.onclose();
  assert.equal(harness.timers.size, 1);
  assert.equal(harness.sockets.length, 2);
});

test('hidden tab defers stall recovery and visibility check runs immediately', () => {
  const harness = makeHarness();
  const first = harness.api.connectMedia();
  first.readyState = first.constructor.OPEN;
  first.onopen();
  harness.advanceClock(3000);
  harness.document.visibilityState = 'hidden';

  harness.runWatchdog();
  assert.equal(first.closeCalls.length, 0);

  harness.document.visibilityState = 'visible';
  harness.api.checkCurrentWatchdog();
  assert.equal(first.closeCalls.length, 1);
  assert.equal(harness.timers.size, 1);
});

test('stop cancels both reconnect and watchdog work', () => {
  const harness = makeHarness();
  const first = harness.api.connectMedia();
  first.readyState = first.constructor.OPEN;
  first.onopen();
  assert.equal(harness.intervals.size, 1);

  harness.api.stopAndInvalidate();

  assert.equal(harness.intervals.size, 0);
  assert.equal(harness.timers.size, 0);
  assert.equal(harness.api.reconnectAttempt(), 0);
});

test('failed reconnects back off until media bytes prove recovery', () => {
  const harness = makeHarness();
  const first = harness.api.connectMedia();
  first.readyState = first.constructor.CLOSED;
  first.onclose();
  assert.equal(harness.runNextTimer(), 250);

  const second = harness.sockets[1];
  second.readyState = second.constructor.CLOSED;
  second.onclose();
  assert.equal(harness.runNextTimer(), 500);
  assert.equal(harness.sockets.length, 3);
});

test('media websocket cannot remain connecting forever', () => {
  const harness = makeHarness();
  const first = harness.api.connectMedia();

  assert.equal(harness.timers.size, 1);
  assert.equal(harness.runNextTimer(), 5000);
  assert.equal(first.closeCalls.length, 1);
  assert.equal(first.closeCalls[0].code, 4001);
  assert.equal(harness.api.generation(), 2);
  assert.equal(harness.timers.size, 1);
  assert.equal(harness.runNextTimer(), 250);
  assert.equal(harness.sockets.length, 2);
});

test('resuming conversation rearms watchdog on an existing open socket', () => {
  const harness = makeHarness();
  const first = harness.api.connectMedia();
  first.readyState = first.constructor.OPEN;
  first.onopen();
  assert.equal(harness.intervals.size, 1);

  harness.api.pauseConversationKeepMedia();
  assert.equal(harness.intervals.size, 0);
  const resumed = harness.api.resumeConversation();

  assert.equal(resumed, first);
  assert.equal(harness.sockets.length, 1);
  assert.equal(harness.intervals.size, 1);
});

test('resuming conversation rearms timeout on an existing connecting socket', () => {
  const harness = makeHarness();
  const first = harness.api.connectMedia();
  assert.equal(harness.timers.size, 1);

  harness.api.pauseConversationKeepMedia();
  assert.equal(harness.timers.size, 0);
  const resumed = harness.api.resumeConversation();

  assert.equal(resumed, first);
  assert.equal(harness.sockets.length, 1);
  assert.equal(harness.timers.size, 1);
  assert.equal(harness.runNextTimer(), 5000);
});

test('page release cancels reconnect before invalidating media generation', () => {
  const release = section('function releaseRealtimePage()', 'startBtn.onclick');
  assert.match(release, /cancelMediaReconnect\(\);/);
  assert.match(release, /cancelMediaConnectTimeout\(\);/);
  assert.match(release, /cancelMediaWatchdog\(\);/);
  assert.match(release, /mediaWsGeneration \+= 1;/);
  assert.ok(
    release.indexOf('cancelMediaReconnect();')
      < release.indexOf('mediaWsGeneration += 1;')
  );
});

test('returning to a visible page checks media freshness immediately', () => {
  const visibility = section(
    'function handleMediaVisibilityChange()',
    'startBtn.onclick'
  );
  assert.match(visibility, /document\.visibilityState !== 'visible'/);
  assert.match(
    visibility,
    /checkMediaWatchdog\(mediaWs, mediaWsGeneration\);/
  );
  assert.match(visibility, /cancelMediaReconnect\(\);/);
  assert.match(visibility, /connectMedia\(\);/);
});

test('visible page bypasses a background-throttled reconnect timer', () => {
  const harness = makeHarness();
  const first = harness.api.connectMedia();
  first.readyState = first.constructor.CLOSED;
  first.onclose();
  assert.equal(harness.timers.size, 1);

  harness.document.visibilityState = 'visible';
  harness.api.handleMediaVisibilityChange();

  // The throttled reconnect timer was replaced by the new socket's connect
  // timeout rather than left pending in the background.
  assert.equal(harness.timers.size, 1);
  assert.equal(harness.sockets.length, 2);
});
