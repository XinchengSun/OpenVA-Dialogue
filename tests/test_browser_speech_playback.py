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
for(const ahead of [0.05,1,4]) {
  const s=slot(ahead);s.video.playbackRate=1.1;liveTail(s);
  assert.equal(s.video.currentTime,10,'never seek across spoken words');
  assert.equal(s.video.playbackRate,1,'preserve natural speech cadence');
}
visibleAssistantTurn=null;assistantResponseArmed=true;
const armed=slot(4);liveTail(armed);assert.equal(armed.video.currentTime,10);
assistantResponseArmed=false;
const idle=slot(4);liveTail(idle);assert.equal(idle.video.currentTime,14-TARGET_LATENCY_SEC);
assert.ok(START_BUFFER_SEC>=TARGET_LATENCY_SEC);
'''
        result=subprocess.run(['node','-e',constants+'\n'+function+'\n'+checks],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
