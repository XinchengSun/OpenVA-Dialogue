#!/usr/bin/env node
"use strict";

// Run the unchanged browser script with media API fakes. This verifies gate
// behavior, not decoding, network timing, audible output, or real browser delay.
const assert = require("node:assert/strict");
const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const sourcePath = path.resolve(__dirname, "../static/index.html");
const source = fs.readFileSync(sourcePath, "utf8");
const script = source.match(/<script>([\s\S]*?)<\/script>/)[1];

function replay(ahead, replacing, playbackStarted = false) {
  const elements = new Map();
  let playCalls = 0;
  const element = id => {
    if (!elements.has(id)) elements.set(id, {
      disabled: false, textContent: "", dataset: {}, currentTime: 0,
      muted: true, playbackRate: 1, classList: {add() {}, remove() {}, toggle() {}},
      append(text) { this.textContent += text; }, pause() {}, load() {},
      removeAttribute() {}, play() { playCalls++; return Promise.resolve(); },
      addEventListener() {},
    });
    return elements.get(id);
  };
  const context = vm.createContext({
    document: {getElementById: element, createTextNode: value => value},
    WebSocket: {OPEN: 1}, window: {}, navigator: {},
    location: {protocol: "http:", host: "localhost", href: "http://localhost/"},
    URL: class extends URL { static revokeObjectURL() {} }, setInterval() {}, setTimeout() { return 1; },
    clearTimeout() {}, performance, console,
  });
  vm.runInContext(script, context);
  Object.assign(context, {ahead, replacing, wasStarted: playbackStarted});
  vm.runInContext(`
    const slot = {
      video: videoA, generation: mseGeneration, disposed: false,
      replacing, playbackStarted: wasStarted, lastPrebufferLogAt: 0,
      sourceBuffer: {buffered: {length: 1, start: () => 0, end: () => ahead}},
    };
    slotByVideo.set(videoA, slot);
    maybePlay(slot);
    globalThis.observed = {
      ahead_seconds: ahead, replacing, already_started: wasStarted,
      playback_started: slot.playbackStarted,
      retained_media_seconds: ahead - slot.video.currentTime,
    };
  `, context);
  return {...context.observed, play_calls: playCalls};
}

const rows = [replay(0.59, false), replay(0.60, false),
              replay(0.15, true), replay(0.16, true), replay(0.10, false, true)];
assert.equal(rows[0].play_calls, 0);
assert.equal(rows[1].play_calls, 1);
assert.equal(rows[2].play_calls, 0);
assert.equal(rows[3].play_calls, 1);
assert.equal(rows[4].play_calls, 1);
console.log(JSON.stringify({
  scope: "Production JS startup gate replay with fake media APIs; not measured playback latency",
  production_behavior_changed: false,
  source: sourcePath,
  source_sha256: crypto.createHash("sha256").update(source).digest("hex"),
  cases_passed: rows.length,
  cases: rows,
}, null, 2));
