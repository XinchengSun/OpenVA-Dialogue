"use strict";
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../static/mic-capture.js'), 'utf8');
for (const sampleRate of [44100, 48000]) {
  let Capture;
  const queued = [];
  class AudioWorkletProcessor {
    constructor() { this.port = {postMessage: bytes => queued.push(new Float32Array(bytes))}; }
  }
  vm.runInNewContext(source, {AudioWorkletProcessor, sampleRate, Float32Array,
    registerProcessor: (_, implementation) => { Capture = implementation; }});
  const capture = new Capture();
  const samples = Float32Array.from({length: sampleRate * 5}, (_, i) => Math.sin(i * 0.031));
  // The UI does not consume messages during capture, as with a busy main thread.
  for (let i = 0; i < samples.length; i += 128) capture.process([[samples.subarray(i, i + 128)]]);
  assert.equal(queued.length, 125);
  const result = Float32Array.from(queued.flatMap(chunk => Array.from(chunk)));
  assert.deepEqual(result, samples, `${sampleRate}: capture lost or altered samples`);
  assert.equal(capture.process([[]]), true);
}
console.log('AudioWorklet capture: both sample rates retain all samples while delivery is delayed');
