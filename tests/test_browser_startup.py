"""Exercise startup errors and gesture-time audio initialization in real JS."""
from pathlib import Path
import shutil
import subprocess
import unittest


@unittest.skipUnless(shutil.which('node'), 'Node.js required')
class BrowserStartupTests(unittest.TestCase):
    def test_permission_errors_are_visible_and_start_can_be_retried(self):
        html=(Path(__file__).resolve().parents[1]/'static/index.html').read_text(encoding='utf-8')
        error=html[html.index('function setStatus('):html.index('function setLiveVisible(')]
        guard=html[html.index('function requireServicePage('):html.index('function wsUrl(')]
        handler=html[html.index('startBtn.onclick ='):html.index('stopBtn.onclick =')]
        script=r'''
const assert=require('node:assert/strict');
const panel={open:false},startupErrorEl={hidden:true},document={querySelector:()=>panel};
const log=()=>{},ensureIdlePlaying=()=>{},setLiveVisible=()=>{},statusEl={dataset:{}};
let startInProgress=false,appStarted=false,micLifecycleGeneration=0,activeSlot=null;
const startBtn={},navigator={mediaDevices:{getUserMedia:()=>{}}},window={MediaSource:{}};
const location={protocol:'https:'};
let connections=0;
const connectLogs=()=>{connections++;},connectMedia=()=>{connections++;};
const startMic=async()=>{throw Object.assign(new Error('permission denied'),{name:'NotAllowedError'});};
'''+guard+'\n'+error+'\n'+handler+r'''
(async()=>{
 await startBtn.onclick();
 assert.match(startupErrorEl.textContent,/麦克风权限/);
 assert.equal(startupErrorEl.hidden,false);assert.equal(panel.open,true);
 setStatus('listening');assert.equal(statusEl.textContent,'启动失败');
 assert.equal(startBtn.disabled,false);assert.equal(startInProgress,false);
 assert.equal(appStarted,false);assert.equal(connections,2);
 connections=0;window.MediaSource=null;await startBtn.onclick();
 assert.match(startupErrorEl.textContent,/浏览器不支持视频流/);
 assert.equal(connections,0,'fail before opening sockets in unsupported browsers');
 window.MediaSource={};location.protocol='file:';await startBtn.onclick();
 assert.match(startupErrorEl.textContent,/本地页面文件/);
 assert.equal(connections,0,'file pages must fail before sockets or microphone acquisition');
})().catch(e=>{console.error(e);process.exitCode=1;});
'''
        result=subprocess.run(['node','-e',script],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_audio_resume_is_invoked_before_waiting_for_websocket(self):
        html=(Path(__file__).resolve().parents[1]/'static/index.html').read_text(encoding='utf-8')
        function=html[html.index('async function startMic('):html.index('function stopMic()')]
        script=r'''
const assert=require('node:assert/strict'),order=[];
let processor=null,micStream=null,audioCtx=null,zeroGain=null,micLifecycleGeneration=1;
const ws={readyState:1,close(){}},micWs=ws,WebSocket={OPEN:1,CLOSING:2};
const micMode={value:'speaker'};
const resetMicPending16=()=>{},connectMicWS=()=>ws,setStatus=()=>{},ensureIdlePlaying=()=>{},log=()=>{};
const downsampleTo16k=()=>{},sendMic16k=()=>{};
const waitForMicOpen=()=>{order.push('wait');return new Promise(r=>setTimeout(r,10));};
const node=()=>({connect(){},disconnect(){}});
class Audio {
 constructor(){order.push('context');this.state='running';this.sampleRate=48000;this.audioWorklet={addModule:async()=>{}};}
 resume(){order.push('resume');return Promise.resolve();}
 createMediaStreamSource(){return node();}createScriptProcessor(){return node();}
 createGain(){return {...node(),gain:{value:1}};}
 close(){return Promise.resolve();}
}
class AudioWorkletNode {constructor(){this.port={};}connect(){}disconnect(){}}
const document={baseURI:'https://example.test/proxy/18786/',getElementById:()=>({})};
const window={AudioContext:Audio},navigator={mediaDevices:{getUserMedia:async()=>({getTracks:()=>[],getAudioTracks:()=>[]})}};
'''+function+r'''
(async()=>{
 await startMic(1);assert.deepEqual(order.slice(0,3),['context','resume','wait']);
 assert.ok(processor);assert.ok(micStream);assert.equal(audioCtx.state,'running');
})().catch(e=>{console.error(e);process.exitCode=1;});
'''
        result=subprocess.run(['node','-e',script],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
