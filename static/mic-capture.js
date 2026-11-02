// Capture on the audio thread so page rendering cannot discard input blocks.
class OpenVAMicCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.chunkSize = Math.round(sampleRate * 0.04);
    this.chunk = new Float32Array(this.chunkSize);
    this.offset = 0;
  }

  process(inputs) {
    const input = inputs[0]?.[0];
    if (!input) return true;
    let start = 0;
    while (start < input.length) {
      const count = Math.min(this.chunkSize - this.offset, input.length - start);
      this.chunk.set(input.subarray(start, start + count), this.offset);
      start += count;
      this.offset += count;
      if (this.offset === this.chunkSize) {
        this.port.postMessage(this.chunk.buffer, [this.chunk.buffer]);
        this.chunk = new Float32Array(this.chunkSize);
        this.offset = 0;
      }
    }
    // Outputs remain zero: never feed the microphone back to the speakers.
    return true;
  }
}

registerProcessor('openva-mic-capture', OpenVAMicCapture);
