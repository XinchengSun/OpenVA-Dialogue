#!/usr/bin/env node
"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");


function extractFunction(source, name) {
  const start = source.indexOf(`function ${name}(`);
  assert.notEqual(start, -1, `${name} is missing from index.html`);
  const bodyStart = source.indexOf("{", start);
  let depth = 0;
  for (let index = bodyStart; index < source.length; index++) {
    if (source[index] === "{") depth += 1;
    if (source[index] === "}") depth -= 1;
    if (depth === 0) return source.slice(start, index + 1);
  }
  throw new Error(`${name} has no closing brace`);
}


const htmlPath = path.resolve(__dirname, "../static/index.html");
const html = fs.readFileSync(htmlPath, "utf8");
const downsampleSource = extractFunction(html, "downsampleTo16k");


function makeResampler() {
  const context = { Float32Array, Number, Error, Math };
  vm.runInNewContext(
    `let micResampleSourceRate = 0;
     let micResampleWeightedSum = 0;
     let micResampleWeight = 0;
     ${downsampleSource}
     this.run = downsampleTo16k;`,
    context,
  );
  return context.run;
}


function concatenate(parts) {
  const length = parts.reduce((total, part) => total + part.length, 0);
  const output = new Float32Array(length);
  let offset = 0;
  for (const part of parts) {
    output.set(part, offset);
    offset += part.length;
  }
  return output;
}


function resampleInChunks(input, rate, chunkSizes) {
  const run = makeResampler();
  const output = [];
  let offset = 0;
  let chunkIndex = 0;
  while (offset < input.length) {
    const size = chunkSizes[chunkIndex % chunkSizes.length];
    output.push(run(input.subarray(offset, offset + size), rate));
    offset += size;
    chunkIndex += 1;
  }
  return concatenate(output);
}


function sine(rate, seconds, frequency) {
  const output = new Float32Array(Math.round(rate * seconds));
  for (let index = 0; index < output.length; index++) {
    output[index] = Math.sin(2 * Math.PI * frequency * index / rate);
  }
  return output;
}


function rms(input) {
  let energy = 0;
  for (const value of input) energy += value * value;
  return Math.sqrt(energy / input.length);
}


for (const rate of [44_100, 48_000]) {
  const input = sine(rate, 5, 1_000);
  const whole = resampleInChunks(input, rate, [input.length]);
  const chunked = resampleInChunks(input, rate, [127, 2048, 997, 4096, 333]);
  assert.equal(chunked.length, 80_000, `${rate} Hz output length drifted`);
  assert.equal(whole.length, chunked.length, `${rate} Hz chunking changed length`);
  let maxDifference = 0;
  for (let index = 0; index < whole.length; index++) {
    maxDifference = Math.max(maxDifference, Math.abs(whole[index] - chunked[index]));
  }
  assert.ok(maxDifference < 1e-7, `${rate} Hz callback boundaries changed samples`);
  assert.ok(rms(chunked) / rms(input) > 0.98, `${rate} Hz damaged 1 kHz speech band`);
}

const aliasedInput = sine(48_000, 2, 12_000);
const filtered = resampleInChunks(aliasedInput, 48_000, [2048]);
const nearest = new Float32Array(Math.floor(aliasedInput.length / 3));
for (let index = 0; index < nearest.length; index++) nearest[index] = aliasedInput[index * 3];
assert.ok(
  rms(filtered) < rms(nearest) * 0.4,
  "area averaging did not materially attenuate above-Nyquist alias energy",
);

console.log("MIC_RESAMPLER_OK rates=44100,48000 seconds=5 output_samples=80000");
