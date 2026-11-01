"""Real WebSocket regression for stale reply events in the dialogue probe."""
import asyncio
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
import wave

from scripts.probe_custom_e2e import _run
from websockets.asyncio.server import serve


class CustomProbeTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, valid_start):
        peers={}
        async def handler(ws):
            role=ws.request.path.rsplit('/',1)[-1]
            peers[role]=ws
            if role!='mic':
                await ws.wait_closed();return
            await ws.recv()
            await ws.send(json.dumps({'type':'ok','message':'session running'}))
            async def event(kind,turn):
                await peers['media'].send(json.dumps({'type':'assistant_'+kind,'turn_id':turn,'generation':2}))
            # A prior reply is replayed on media connect, before current speech.
            for kind in ['turn_started','media_boundary','media_ended']:await event(kind,9)
            for _ in range(4):await ws.recv()  # 3 voiced blocks and one silence.
            await asyncio.sleep(.02)
            await peers['logs'].send(json.dumps({'message':'[PIPECAT PROVIDER] first audio engine_turn=77 sample_rate=48000'}))
            if valid_start:await event('turn_started',77)
            await event('media_boundary',77)
            await event('media_ended',77)
            await ws.wait_closed()
        with tempfile.TemporaryDirectory() as directory:
            wav=Path(directory)/'query.wav'
            with wave.open(str(wav),'wb') as output:
                output.setparams((1,2,16000,0,'NONE',''))
                output.writeframes(struct.pack('<h',1000)*640*3)
            async with serve(handler,'127.0.0.1',0) as server:
                port=server.sockets[0].getsockname()[1]
                args=SimpleNamespace(base=f'ws://127.0.0.1:{port}',input_wav=wav,turns=1,timeout=.25,
                    instruction='test',voice_threshold=500,silence_chunks=1,between_turns=0)
                return await _run(args)

    async def test_stale_complete_reply_cannot_supply_current_measurement(self):
        result=(await self.exercise(True))[0]
        self.assertEqual(result['engine_turn_id'],77)
        self.assertTrue(result['lifecycle_complete'])
        for name in ['voice_stop_to_provider_first_audio_ms','voice_stop_to_media_boundary_ms','voice_stop_to_media_ended_ms']:
            self.assertGreaterEqual(result[name],0)

    async def test_boundary_and_end_without_current_start_fail(self):
        with self.assertRaises(TimeoutError):
            await self.exercise(False)


if __name__=='__main__':unittest.main()
