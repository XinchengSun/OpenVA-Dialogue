"""Listener, thinking and speech must keep the native media clock."""
from pathlib import Path
import re
import shutil
import subprocess
import unittest


@unittest.skipUnless(shutil.which('node'), 'Node.js required')
class SpeechPlaybackTests(unittest.TestCase):
    def test_no_role_can_seek_pause_or_speed_up_the_live_tail(self):
        html=(Path(__file__).resolve().parents[1]/'static/index.html').read_text(encoding='utf-8')
        constants='\n'.join(re.findall(r'^const BUFFER_[A-Z_]+ = .*;',html,re.M))
        function=html[html.index('function liveTail(slot)'):html.index('function teardownMSE()')]
        checks=r'''
const assert=require('node:assert/strict');
const isCurrentSlot=()=>true,log=()=>{};
for(const ahead of [0.05,1,4,10]) {
 let pauses=0;
 const s={sourceBuffer:{updating:true,buffered:{length:1,end:()=>10+ahead}},
  video:{currentTime:10,playbackRate:1.1,pause:()=>{pauses++;}},playbackStarted:true};
 liveTail(s);assert.equal(s.video.currentTime,10);assert.equal(s.video.playbackRate,1);assert.equal(pauses,0);
}
'''
        result=subprocess.run(['node','-e',constants+'\n'+function+'\n'+checks],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_late_start_keeps_first_frame_and_audio_instead_of_seeking(self):
        html=(Path(__file__).resolve().parents[1]/'static/index.html').read_text(encoding='utf-8')
        constants='\n'.join(re.findall(r'^const (?:START_BUFFER_SEC|HANDOFF_BUFFER_SEC) = .*;',html,re.M))
        play=html[html.index('function maybePlay(slot)'):html.index('function liveTail(slot)')]
        checks=r'''
const assert=require('node:assert/strict');
const isCurrentSlot=()=>true,log=()=>{},armSlotActivation=()=>{},bufferedAhead=s=>s.end-s.video.currentTime;
let plays=0;const s={end:8,sourceBuffer:{buffered:{length:1}},video:{currentTime:0,play:()=>{plays++;return Promise.resolve();}},playbackStarted:false};
maybePlay(s);assert.equal(plays,1);assert.equal(s.video.currentTime,0,'keep the first word after slow connection startup');
assert.equal(s.playbackStarted,true);
'''
        result=subprocess.run(['node','-e',constants+'\n'+play+'\n'+checks],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
