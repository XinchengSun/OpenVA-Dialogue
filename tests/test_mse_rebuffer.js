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

function numericConstant(name) {
  const match = html.match(new RegExp(`const ${name} = ([0-9.]+);`));
  assert.ok(match, `missing numeric constant: ${name}`);
  return Number(match[1]);
}

function makeHarness() {
  let end = 10;
  let now = 1000;
  let currentGeneration = 7;
  const logs = [];
  const video = {
    currentTime: 10,
    playbackRate: 0.96,
    readyState: 3,
    pauseCalls: 0,
    playCalls: 0,
    pause() {
      this.pauseCalls += 1;
    },
    play() {
      this.playCalls += 1;
      return Promise.resolve();
    },
  };
  const sourceBuffer = {
    updating: false,
    buffered: {
      length: 1,
      start() { return 0; },
      end() { return end; },
    },
    remove() {},
  };
  const slot = {
    video,
    sourceBuffer,
    playbackStarted: true,
    hasPlayed: true,
    lowBufferRecovery: true,
    rebuffering: false,
    rebufferCatchup: false,
    rebufferStartedAt: 0,
    rebufferCount: 0,
    replacing: false,
    lastPrebufferLogAt: 0,
    lastTailSeekAt: 0,
    lastBufferTrimAt: now,
    generation: 7,
    disposed: false,
  };
  const recovery = section(
    'function enterPlaybackRebuffer(slot, reason)',
    'function maybePlay(slot)'
  );
  const maybePlay = section('function maybePlay(slot)', 'function liveTail(slot)');
  const liveTail = section('function liveTail(slot)', 'function teardownMSE()');
  const api = vm.runInNewContext(`
    const START_BUFFER_SEC = 0.60;
    const HANDOFF_BUFFER_SEC = 0.16;
    const TARGET_LATENCY_SEC = 0.48;
    const HANDOFF_TARGET_LATENCY_SEC = 0.08;
    const LOW_BUFFER_ENTER_SEC = 0.32;
    const LOW_BUFFER_EXIT_SEC = 0.48;
    const LOW_BUFFER_PLAYBACK_RATE = 0.96;
    const REBUFFER_BUFFER_SEC = START_BUFFER_SEC;
    const REBUFFER_CATCHUP_RATE = 1.03;
    const SOFT_CATCHUP_SEC = 0.80;
    const FAST_CATCHUP_SEC = 1.30;
    const HARD_CATCHUP_SEC = 2.50;
    const LIVE_TAIL_SEEK_INTERVAL_MS = 5000;
    const BUFFER_RETENTION_SEC = 12;
    const BUFFER_TRIM_INTERVAL_MS = 5000;
    const Date = { now: () => nowValue() };
    ${recovery}
    ${maybePlay}
    ${liveTail}
    ({ enterPlaybackRebuffer, maybePlay, liveTail });
  `, {
    HTMLMediaElement: { HAVE_FUTURE_DATA: 3 },
    isCurrentSlot: candidate => (
      candidate === slot
      && !candidate.disposed
      && candidate.generation === currentGeneration
    ),
    bufferedAhead: candidate => Math.max(0, end - candidate.video.currentTime),
    bufferedEnd: () => end,
    armSlotActivation() {},
    log: message => logs.push(message),
    nowValue: () => now,
  });

  return {
    api,
    logs,
    slot,
    video,
    setAhead(seconds) {
      end = video.currentTime + seconds;
    },
    advance(milliseconds) {
      now += milliseconds;
      slot.lastBufferTrimAt = now;
    },
    replaceGeneration() {
      currentGeneration += 1;
    },
  };
}

test('normal live-tail latency stays low outside actual stalls', () => {
  assert.equal(numericConstant('START_BUFFER_SEC'), 0.60);
  assert.equal(numericConstant('TARGET_LATENCY_SEC'), 0.48);
  assert.equal(numericConstant('LOW_BUFFER_ENTER_SEC'), 0.32);
  assert.equal(numericConstant('LOW_BUFFER_EXIT_SEC'), 0.48);
  assert.equal(numericConstant('LOW_BUFFER_PLAYBACK_RATE'), 0.96);
  assert.equal(numericConstant('REBUFFER_CATCHUP_RATE'), 1.03);
  assert.match(html, /const REBUFFER_BUFFER_SEC = START_BUFFER_SEC;/);
});

test('prebuffer and false waiting signals do not add latency', () => {
  const harness = makeHarness();
  harness.slot.hasPlayed = false;
  harness.api.enterPlaybackRebuffer(harness.slot, 'waiting');
  assert.equal(harness.video.pauseCalls, 0);

  harness.slot.hasPlayed = true;
  harness.setAhead(0.32);
  harness.video.readyState = 3;
  harness.api.enterPlaybackRebuffer(harness.slot, 'stalled');
  assert.equal(harness.video.pauseCalls, 0);
  assert.equal(harness.slot.rebuffering, false);
});

test('real waiting pauses once and resumes once without skipping media', async () => {
  const harness = makeHarness();
  harness.setAhead(0);
  harness.video.readyState = 2;

  harness.api.enterPlaybackRebuffer(harness.slot, 'waiting');
  harness.api.enterPlaybackRebuffer(harness.slot, 'waiting');

  assert.equal(harness.slot.rebuffering, true);
  assert.equal(harness.slot.rebufferCount, 1);
  assert.equal(harness.slot.lowBufferRecovery, false);
  assert.equal(harness.slot.rebufferCatchup, false);
  assert.equal(harness.video.playbackRate, 1.0);
  assert.equal(harness.video.pauseCalls, 1);
  const originalTime = harness.video.currentTime;

  harness.setAhead(0.24);
  harness.api.maybePlay(harness.slot);
  harness.setAhead(0.47);
  harness.api.maybePlay(harness.slot);
  assert.equal(harness.video.playCalls, 0);
  assert.equal(harness.slot.rebuffering, true);

  harness.advance(720);
  harness.setAhead(0.72);
  harness.api.maybePlay(harness.slot);
  assert.equal(harness.video.playCalls, 1);
  assert.equal(harness.slot.rebufferResumePending, true);
  await Promise.resolve();
  assert.equal(harness.slot.rebuffering, false);
  assert.equal(harness.slot.rebufferResumePending, false);
  assert.equal(harness.slot.rebufferCatchup, true);
  assert.equal(harness.video.playbackRate, 1.03);
  assert.equal(harness.video.currentTime, originalTime, 'recovery must not seek over TTS');
  assert.equal(harness.logs.filter(line => line.includes('video rebuffer start')).length, 1);
  assert.equal(harness.logs.filter(line => line.includes('video rebuffer end')).length, 1);
});

test('one pending play is retried after rejection and stale completion is fenced', async () => {
  const harness = makeHarness();
  harness.setAhead(0);
  harness.video.readyState = 2;
  harness.api.enterPlaybackRebuffer(harness.slot, 'waiting');
  harness.setAhead(0.72);

  let resolveFirst;
  harness.video.play = function playPending() {
    this.playCalls += 1;
    return new Promise(resolve => { resolveFirst = resolve; });
  };
  harness.api.maybePlay(harness.slot);
  harness.api.maybePlay(harness.slot);
  assert.equal(harness.video.playCalls, 1);
  assert.equal(harness.slot.rebufferResumePending, true);

  harness.replaceGeneration();
  resolveFirst();
  await Promise.resolve();
  assert.equal(harness.slot.rebuffering, true);
  assert.equal(harness.slot.rebufferResumePending, true);

  const retry = makeHarness();
  retry.setAhead(0);
  retry.video.readyState = 2;
  retry.api.enterPlaybackRebuffer(retry.slot, 'waiting');
  retry.setAhead(0.72);
  retry.video.play = function rejectOnce() {
    this.playCalls += 1;
    return Promise.reject(new Error('autoplay blocked'));
  };
  retry.api.maybePlay(retry.slot);
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(retry.slot.rebuffering, true);
  assert.equal(retry.slot.rebufferResumePending, false);
  assert.equal(retry.video.playbackRate, 1.0);

  retry.video.play = function acceptRetry() {
    this.playCalls += 1;
    return Promise.resolve();
  };
  retry.api.maybePlay(retry.slot);
  await Promise.resolve();
  assert.equal(retry.video.playCalls, 2);
  assert.equal(retry.slot.rebuffering, false);
});

test('temporary catch-up survives fast refill and ends at the 0.48 second target', () => {
  const harness = makeHarness();
  harness.slot.lowBufferRecovery = false;
  harness.slot.rebufferCatchup = true;
  harness.video.playbackRate = 1.03;
  harness.setAhead(0.96);

  harness.api.liveTail(harness.slot);

  assert.equal(harness.slot.rebufferCatchup, true);
  assert.equal(harness.video.playbackRate, 1.06);

  harness.setAhead(0.48);

  harness.api.liveTail(harness.slot);

  assert.equal(harness.slot.rebufferCatchup, false);
  assert.equal(harness.video.playbackRate, 1.0);
});

test('rebuffering blocks hard seeks and stale slots are inert', () => {
  const harness = makeHarness();
  harness.slot.rebuffering = true;
  harness.video.playbackRate = 1.0;
  harness.setAhead(3.0);
  const originalTime = harness.video.currentTime;

  harness.api.liveTail(harness.slot);
  assert.equal(harness.video.currentTime, originalTime);
  assert.equal(harness.video.playbackRate, 1.0);

  harness.slot.rebuffering = false;
  harness.replaceGeneration();
  harness.api.enterPlaybackRebuffer(harness.slot, 'waiting');
  assert.equal(harness.video.pauseCalls, 0);
});

test('page wiring handles waiting and stalled without touching audio gates', () => {
  const build = section('function buildMSE(mime, replacing)', 'function setupMSE(mime)');
  const recovery = section(
    'function enterPlaybackRebuffer(slot, reason)',
    'function maybePlay(slot)'
  );

  assert.match(build, /targetVideo\.onplaying[\s\S]*slot\.hasPlayed = true;/);
  assert.match(build, /targetVideo\.onwaiting[\s\S]*enterPlaybackRebuffer\(slot, 'waiting'\);/);
  assert.match(build, /targetVideo\.onstalled[\s\S]*enterPlaybackRebuffer\(slot, 'stalled'\);/);
  assert.match(build, /if \(slot\.rebuffering && !slot\.rebufferResumePending\)[\s\S]*targetVideo\.pause\(\);/);
  assert.doesNotMatch(recovery, /teardownMSE|holdLiveAudio|setLiveVisible/);
});
