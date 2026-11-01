"""Execute the real browser tail policy: spoken audio must never be skipped."""
from pathlib import Path
import re
import shutil
import subprocess
import unittest


@unittest.skipUnless(shutil.which('node'), 'Node.js required for browser policy check')
class SpeechPlaybackTests(unittest.TestCase):
    def test_speech_keeps_cadence_and_idle_still_catches_up(self):
        html=(Path(__file__).resolve().parents[1]/'static/index.html').read_text(encoding='utf-8')
        constants='\n'.join(re.findall(r'^const (?:START_BUFFER_SEC|TARGET_LATENCY_SEC|LOW_BUFFER_[A-Z_]+|SOFT_CATCHUP_SEC|FAST_CATCHUP_SEC|HARD_CATCHUP_SEC|LIVE_TAIL_SEEK_INTERVAL_MS|BUFFER_[A-Z_]+) = .*;',html,re.M))
        function=html[html.index('function liveTail(slot)'):html.index('function teardownMSE()')]
        checks=r'''
const assert = require('node:assert/strict');
let assistantResponseArmed=false, visibleAssistantTurn=null;
const isCurrentSlot=()=>true, log=()=>{};
function slot(ahead) {
  return {sourceBuffer:{updating:true,buffered:{length:1,end:()=>10+ahead}},
    video:{currentTime:10,playbackRate:1},playbackStarted:true,
    lowBufferRecovery:false,lastTailSeekAt:0,lastBufferTrimAt:Date.now()};
}
visibleAssistantTurn={turnId:1};
for(const ahead of [0.05,1,HARD_CATCHUP_SEC+1]) {
  const s=slot(ahead);s.video.playbackRate=1.1;liveTail(s);
  assert.equal(s.video.currentTime,10,'never seek across spoken words');
  assert.equal(s.video.playbackRate,1,'preserve natural speech cadence');
}
visibleAssistantTurn=null;assistantResponseArmed=true;
const armed=slot(HARD_CATCHUP_SEC+1);liveTail(armed);assert.equal(armed.video.currentTime,10);
assistantResponseArmed=false;
const idle=slot(HARD_CATCHUP_SEC+1);liveTail(idle);
assert.equal(idle.video.currentTime,11+HARD_CATCHUP_SEC-TARGET_LATENCY_SEC);
assert.ok(START_BUFFER_SEC>=TARGET_LATENCY_SEC);
'''
        result=subprocess.run(['node','-e',constants+'\n'+function+'\n'+checks],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_preroll_keeps_first_speech_frame_and_waits_for_jitter_buffer(self):
        html=(Path(__file__).resolve().parents[1]/'static/index.html').read_text(encoding='utf-8')
        constants='\n'.join(re.findall(r'^const (?:START_BUFFER_SEC|HANDOFF_BUFFER_SEC|TARGET_LATENCY_SEC|HANDOFF_TARGET_LATENCY_SEC|ASSISTANT_PREROLL_SEC) = .*;',html,re.M))
        prepare=html[html.index('function prepareAssistantPreroll('):html.index('function maybeReleaseLiveAudioAtBoundary()')]
        play=html[html.index('function maybePlay(slot)'):html.index('function liveTail(slot)')]
        gate=html[html.index('function maybeReleaseLiveAudioAtBoundary()'):html.index('function maybeReturnToListenerAtBoundary()')]
        checks=r'''
const assert=require('node:assert/strict');
const isCurrentSlot=()=>true, log=()=>{}, armSlotActivation=()=>{};
const bufferedEnd=s=>s.sourceBuffer.buffered.end(0);
const bufferedAhead=s=>bufferedEnd(s)-s.video.currentTime;
let end=4.5, pauses=0, plays=0;
const s={video:{muted:true,currentTime:1,pause:()=>{pauses++;},play:()=>{plays++;return Promise.resolve();}},
  sourceBuffer:{buffered:{length:1,start:()=>0,end:()=>end}},playbackStarted:false,replacing:false};
prepareAssistantPreroll(s,4,'turn1');
assert.equal(pauses,1);assert.equal(plays,0);
assert.equal(s.video.currentTime,4-ASSISTANT_PREROLL_SEC);
assert.ok(s.video.currentTime<4,'retain the first speech frame');
assert.ok(s.assistantPreroll,'keep playback paused with insufficient headroom');
end=8;maybePlay(s);
assert.equal(plays,1);assert.equal(s.video.currentTime,4-ASSISTANT_PREROLL_SEC,
  'startup must not seek past speech after preroll');
assert.equal(s.assistantPreroll,null);
prepareAssistantPreroll(s,4,'turn1');assert.equal(pauses,1,'never rearm the same boundary');
s.video.muted=false;prepareAssistantPreroll(s,7,'turn2');assert.equal(pauses,1,'never seek audible speech');
s.video.muted=true;s.video.currentTime=3.9;prepareAssistantPreroll(s,4,'turn3');
assert.equal(pauses,1,'do not seek backwards once already in the preroll');
let pendingAssistantBoundary={streamEpoch:1,generation:4,turnId:7,safeMediaTimeSec:4};
let assistantResponseArmed=true,activeAssistantTurn={streamEpoch:1,generation:4,turnId:8};
const activeSlot=s,ingestSlot=s;let expectedStreamGeneration=4,releasedAssistantTurnKey=null,releases=0;
const assistantTurnKey=(...args)=>args.join(':'),releaseLiveAudio=()=>{releases++;};
s.streamEpoch=1;s.timelineOrigin=0;
maybeReleaseLiveAudioAtBoundary();assert.equal(pauses,1,'ignore old turn boundaries');
activeAssistantTurn.turnId=7;expectedStreamGeneration=5;
maybeReleaseLiveAudioAtBoundary();assert.equal(pauses,1,'ignore old generation boundaries');
expectedStreamGeneration=4;maybeReleaseLiveAudioAtBoundary();assert.equal(releases,0);
s.video.currentTime=4;maybeReleaseLiveAudioAtBoundary();assert.equal(releases,1);
'''
        result=subprocess.run(['node','-e',constants+'\n'+prepare+'\n'+play+'\n'+gate+'\n'+checks],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
