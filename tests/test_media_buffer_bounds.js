#!/usr/bin/env node
"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const html = fs.readFileSync(path.resolve(__dirname, "../static/index.html"), "utf8");

function extractFunction(name) {
  const start = html.indexOf(`function ${name}(`);
  assert.notEqual(start, -1);
  const end = html.indexOf("\nfunction ", start + 1);
  return html.slice(start, end);
}

function harness() {
  const stopped = [];
  const context = {
    MEDIA_PENDING_MAX_BYTES: 8,
    MEDIA_PENDING_MAX_SEGMENTS: 3,
    isCurrentSlot: slot => !slot.disposed,
    failMediaStream: (slot, reason) => { slot.disposed = true; stopped.push(reason); },
    log: () => {},
    describeBuffered: () => "",
  };
  vm.runInNewContext(
    `${extractFunction("queueMediaSegment")}\n${extractFunction("appendNextSegment")}`,
    context,
  );
  return { ...context, stopped };
}

// Bound bytes while MSE is waiting for sourceopen or updateend.
{
  const h = harness();
  const slot = { pendingSegments: [], pendingBytes: 0, sourceBuffer: null };
  assert.equal(h.queueMediaSegment(slot, new Uint8Array(5)), true);
  assert.equal(h.queueMediaSegment(slot, new Uint8Array(3)), true);
  assert.equal(h.queueMediaSegment(slot, new Uint8Array(1)), false);
  assert.equal(slot.pendingBytes, 8);
  assert.equal(h.stopped.length, 1);
  assert.equal(h.queueMediaSegment(slot, new Uint8Array(1)), false);
}

// Tiny chunks cannot bypass the item bound, and accepted bytes retain FIFO order.
{
  const h = harness();
  const appended = [];
  const slot = { pendingSegments: [], pendingBytes: 0,
    sourceBuffer: { updating: true, appendBuffer: data => appended.push(data[0]) } };
  for (const value of [1, 2, 3]) h.queueMediaSegment(slot, new Uint8Array([value]));
  h.appendNextSegment(slot);
  assert.equal(slot.pendingBytes, 3);
  slot.sourceBuffer.updating = false;
  for (let i = 0; i < 3; i++) h.appendNextSegment(slot);
  assert.deepEqual(appended, [1, 2, 3]);
  assert.equal(slot.pendingBytes, 0);
  assert.equal(slot.pendingSegments.length, 0);
  for (const value of [4, 5, 6]) h.queueMediaSegment(slot, new Uint8Array([value]));
  assert.equal(h.queueMediaSegment(slot, new Uint8Array([7])), false);
  assert.equal(h.stopped.length, 1);
}

// A failed append must stop the timeline, never discard bytes and continue.
{
  const h = harness();
  const chunk = new Uint8Array([4, 5]);
  const slot = { pendingSegments: [chunk], pendingBytes: 2,
    sourceBuffer: { updating: false, appendBuffer: () => { throw new Error("quota"); } } };
  h.appendNextSegment(slot);
  assert.equal(h.stopped.length, 1);
  assert.equal(slot.pendingSegments[0], chunk);
  assert.equal(slot.pendingBytes, 2);
}

console.log("media buffer bounds: 3 scenarios passed");
