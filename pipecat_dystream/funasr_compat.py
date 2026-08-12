"""Compatibility fixes scoped to the Pipecat services used by this demo."""

from pipecat.services.funasr.stt import FunASRSTTService


class RawPCMFunASRSTTService(FunASRSTTService):
    """Keep Pipecat 1.6 FunASR segments as raw PCM, as ``run_stt`` expects."""

    @property
    def wants_wav_segments(self) -> bool:
        return False
