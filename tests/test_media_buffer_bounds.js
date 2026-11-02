#!/usr/bin/env node
"use strict";

// Execute the entire production script with browser API fakes. In particular,
// failMediaStream, stopMic, the Start handler and audio gates are never mocked.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const html = fs.readFileSync(path.resolve(__dirname, "../static/index.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function harness() {
  const elements = new Map();
  const sockets = [];
  const tracks = [];
  const element = id => {
    if (!elements.has(id)) elements.set(id, {
      disabled: false, textContent: "", dataset: {}, currentTime: 30,
      muted: true, playbackRate: 1, classList: { add() {}, remove() {}, toggle() {} },
      append(text) { this.textContent += text; }, pause() {}, load() {},
      removeAttribute() {}, play() { return Promise.resolve(); }, addEventListener() {},
    });
    return elements.get(id);
  };
  class Socket {
    static OPEN = 1; static CONNECTING = 0; static CLOSING = 2; static CLOSED = 3;
    constructor(url) { this.url = url; this.readyState = 1; sockets.push(this); }
    send() {}
    close() { this.readyState = 3; this.onclose?.(); }
  }
  class AudioContext {
    constructor() { this.state = "running"; this.sampleRate = 48000; this.audioWorklet = {addModule: async () => {}}; }
    resume() { return Promise.resolve(); }
    createMediaStreamSource() { return {connect() {}}; }
    createScriptProcessor() { return {connect() {}, disconnect() {}}; }
    createGain() { return {gain: {value: 1}, connect() {}, disconnect() {}}; }
    close() { this.state = "closed"; return Promise.resolve(); }
  }
  const context = vm.createContext({
    document: {getElementById: element, createTextNode: value => value,
      baseURI: 'http://localhost/', querySelector: () => ({open: false})},
    WebSocket: Socket, window: {AudioContext, MediaSource: class {}},
    AudioWorkletNode: class {
      constructor() { this.port = {onmessage: null, close() {}}; }
      connect() {} disconnect() {}
    },
    navigator: {mediaDevices: {getUserMedia: async () => {
      const track = {stopped: false, stop() { this.stopped = true; }};
      tracks.push(track);
      return {getTracks: () => [track], getAudioTracks: () => [track]};
    }}},
    location: {protocol: "http:", host: "localhost", href: "http://localhost/"},
    URL: class extends URL { static revokeObjectURL() {} }, setInterval() {}, setTimeout, clearTimeout,
    performance, console,
  });
  vm.runInContext(script, context);
  const run = source => vm.runInContext(source, context);
  function setSlot(sourceBuffer) {
    context.fixtureBuffer = sourceBuffer;
    return run(`globalThis.fixtureSlot = {
      video: videoA, generation: mseGeneration, pendingSegments: [], pendingBytes: 0,
      quotaRetryAttempted: false, sourceBuffer: fixtureBuffer,
      mediaSource: {readyState: 'open'}, disposed: false
    };
    slotByVideo.set(videoA, fixtureSlot); ingestSlot = fixtureSlot; activeSlot = fixtureSlot;
    fixtureSlot;`);
  }
  return {context, run, setSlot, sockets, tracks, elements};
}

const ranges = {length: 1, start: () => 0, end: () => 35};
const quota = () => Object.assign(new Error("full"), {name: "QuotaExceededError"});

async function main() {
  // Do not enforce the previous unmeasured 8 MiB / 256-chunk cutoff by default.
  {
    const h = harness();
    const slot = h.setSlot(null);
    h.context.chunk = new Uint8Array(32769);
    h.run("for (let i=0; i<257; i++) queueMediaSegment(fixtureSlot, chunk)");
    assert.equal(slot.pendingSegments.length, 257);
    assert.ok(slot.pendingBytes > 8 * 1024 * 1024);
    assert.equal(slot.disposed, false);
  }
  // Preserve each byte chunk and its order while SourceBuffer is busy.
  {
    const h = harness(), accepted = [];
    const sb = {updating: true, buffered: ranges, appendBuffer: data => accepted.push(data[0])};
    const slot = h.setSlot(sb);
    h.run("for (const value of [1,2,3]) queueMediaSegment(fixtureSlot, new Uint8Array([value])); appendNextSegment(fixtureSlot)");
    assert.equal(slot.pendingBytes, 3);
    sb.updating = false;
    h.run("for (let i=0; i<3; i++) appendNextSegment(fixtureSlot)");
    assert.deepEqual(accepted, [1,2,3]);
    assert.equal(slot.pendingBytes, 0);
  }
  // Quota errors retain the exact head chunk; safe pruning permits one retry.
  {
    const h = harness(), removed = [], accepted = [];
    await h.run("startBtn.onclick()");
    let first = true;
    const sb = {updating: false, buffered: ranges,
      appendBuffer(data) { if (first) { first = false; throw quota(); } accepted.push(data); },
      remove(start, end) { removed.push([start,end]); this.updating = true; }, abort() {},
    };
    const slot = h.setSlot(sb);
    h.run("queueMediaSegment(fixtureSlot, new Uint8Array([4,5])); appendNextSegment(fixtureSlot)");
    const pending = slot.pendingSegments[0];
    assert.deepEqual(removed, [[0,18]]); // Existing 12-second retention boundary.
    assert.equal(h.run("appStarted"), true);
    sb.updating = false; // Simulate completion of asynchronous range removal.
    h.run("appendNextSegment(fixtureSlot)");
    assert.equal(accepted[0], pending);
    assert.equal(slot.pendingBytes, 0);
    assert.equal(slot.quotaRetryAttempted, false);
  }
  // If pruning cannot recover, stop microphone and timeline, then Start works.
  {
    const h = harness();
    await h.run("startBtn.onclick()");
    const sb = {updating: false, buffered: ranges, appendBuffer() { throw quota(); },
      remove() { this.updating = true; }, abort() {},
    };
    const slot = h.setSlot(sb);
    h.run("queueMediaSegment(fixtureSlot, new Uint8Array([6])); appendNextSegment(fixtureSlot)");
    sb.updating = false;
    h.run("appendNextSegment(fixtureSlot)");
    assert.equal(slot.disposed, true);
    assert.equal(h.run("appStarted"), false);
    assert.equal(h.run("micWs"), null);
    assert.equal(h.run("mediaWs"), null);
    assert.equal(h.run("activeSlot"), null);
    assert.equal(h.run("liveAudioHeld"), true);
    assert.equal(h.elements.get("startBtn").disabled, false);
    assert.equal(h.tracks[0].stopped, true);
    await h.run("startBtn.onclick()");
    assert.equal(h.run("appStarted"), true);
    assert.equal(h.sockets.filter(socket => socket.url.endsWith("/ws/media")).length, 2);
  }
  // No safe old range exists: fail cleanly without discarding and continuing.
  {
    const h = harness();
    await h.run("startBtn.onclick()");
    h.elements.get("videoA").currentTime = 1;
    const sb = {updating: false, buffered: ranges, appendBuffer() { throw quota(); }, abort() {}};
    h.setSlot(sb);
    h.run("queueMediaSegment(fixtureSlot, new Uint8Array([7])); appendNextSegment(fixtureSlot)");
    assert.equal(h.run("appStarted"), false);
    assert.equal(h.tracks[0].stopped, true);
    assert.equal(h.elements.get("startBtn").disabled, false);
  }
  // Server-initiated closure must not leave an invisible live microphone.
  {
    const h = harness();
    await h.run("startBtn.onclick()");
    h.run("mediaWs.close()");
    assert.equal(h.run("appStarted"), false);
    assert.equal(h.run("micWs"), null);
    assert.equal(h.run("liveAudioHeld"), true);
    assert.equal(h.tracks[0].stopped, true);
    assert.equal(h.elements.get("startBtn").disabled, false);
    await h.run("startBtn.onclick()");
    assert.equal(h.run("appStarted"), true);
  }
  console.log("media stream state: 6 full-script scenarios passed");
}
main().catch(error => { console.error(error); process.exitCode = 1; });
