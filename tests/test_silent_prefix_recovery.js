"use strict";
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '../static/index.html'), 'utf8');
const source = html.slice(html.indexOf('function recoverSilentPrefix('), html.indexOf('function maybeReturnToListenerAtBoundary('));
function run(overrides = {}) {
  const slot = {video:{currentTime:10,muted:true,seeking:false},sourceBuffer:{buffered:{length:1,start:()=>0,end:()=>25}}};
  Object.assign(slot.video, overrides.video);
  if (overrides.ranges) slot.sourceBuffer.buffered = overrides.ranges;
  const context = {slot,liveAudioHeld:true,visibleAssistantTurn:null,START_BUFFER_SEC:2,
    assistantTurnKey:(...xs)=>xs.join(':'),log:()=>{},...overrides.globals};
  vm.runInNewContext(source + ';recoverSilentPrefix(slot,{streamEpoch:0,generation:0,turnId:2},20);',context);
  return {slot,context};
}
const recovered=run();
assert.equal(recovered.slot.video.currentTime,19.84);
assert.ok(recovered.slot.video.currentTime<20,'must preserve the first reply audio');
for(const options of [{video:{muted:false}},{video:{seeking:true}},
  {globals:{liveAudioHeld:false}},{globals:{visibleAssistantTurn:{turnId:1}}},
  {video:{currentTime:19}},{video:{currentTime:21}},
  {ranges:{length:1,start:()=>20,end:()=>25}},
  {ranges:{length:1,start:()=>0,end:()=>20.2}},
  {ranges:{length:1,start:()=>0,end:()=>21.9}}]) {
 const {slot}=run(options);assert.equal(slot.video.currentTime,options.video?.currentTime??10);
}
// A duplicate boundary cannot initiate a second seek.
recovered.slot.video.currentTime=10;
vm.runInNewContext(source+';recoverSilentPrefix(slot,{streamEpoch:0,generation:0,turnId:2},20);',recovered.context);
assert.equal(recovered.slot.video.currentTime,10);
console.log('Silent prefix recovery: first audio, visible speech, buffering and duplicate boundaries protected');
