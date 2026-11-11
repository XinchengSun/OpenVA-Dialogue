"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const html = fs.readFileSync(path.join(__dirname, "../static/index.html"), "utf8");
const source = html.slice(html.indexOf("let lastVoicedPosition ="),
  html.indexOf("function resetMicPending16()"));
const sent = [];
let now = 1000;
const slot = {
  timelineOrigin: 1,
  streamEpoch: 4,
  video: {
    currentTime: 5, paused: false, seeking: false,
    buffered: {length: 1, start: () => 1, end: () => 6},
  },
};
const context = vm.createContext({
  activeSlot: slot,
  performance: {now: () => now},
  micWs: {send: value => sent.push(JSON.parse(value))},
});
vm.runInContext(source, context);
const report = samples => context.reportListenerPosition(samples);
const voiced = new Float32Array(1600);
voiced[800] = 0.1;
report(voiced);
assert.equal(sent[0].type, "playout_clock");
assert.equal(sent[0].pts, 4);
assert.equal(sent[0].buffered, 1);
assert.equal(sent[0].advancing, false);
report(new Float32Array(1600));
assert.equal(sent[1].type, "listener_position");
assert.equal(sent[1].epoch, 4);
assert.equal(sent[1].pts, 4 - 799 / 16000);
report(new Float32Array(1600));
assert.equal(sent.length, 2, "a silence packet reports a voice boundary once");
now += 160;
slot.video.currentTime += 0.16;
report(new Float32Array(1600));
assert.equal(sent.at(-1).advancing, true);
report(voiced);
slot.streamEpoch += 1;
report(new Float32Array(1600));
assert.equal(sent.length, 3, "a reset discards the old voice boundary");
report(voiced);
slot.video.paused = true;
report(new Float32Array(1600));
slot.video.paused = false;
report(new Float32Array(1600));
assert.equal(sent.length, 3, "a pause discards the old voice boundary");
console.log("listener position, playout feedback, epoch and pause checks passed");
