// AudioWorkletProcessor for high-performance, low-latency mic capture.
// Resamples any input sample rate (44.1kHz, 48kHz) to 16kHz mono Int16 PCM chunks.

class MicDownsamplerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetSampleRate = 16000;
    this.buffer = [];
    this.chunkSize = 512; // 32ms @ 16kHz (exact frame size for Silero VAD)
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;

    const channelData = input[0];
    if (!channelData || channelData.length === 0) return true;

    // Resample from current AudioContext sampleRate to 16000Hz via linear interpolation
    const ratio = sampleRate / this.targetSampleRate;
    const newLen = Math.floor(channelData.length / ratio);

    for (let i = 0; i < newLen; i++) {
      const srcIdx = i * ratio;
      const i0 = Math.floor(srcIdx);
      const i1 = Math.min(i0 + 1, channelData.length - 1);
      const frac = srcIdx - i0;
      const sample = channelData[i0] * (1 - frac) + channelData[i1] * frac;

      // Convert float [-1.0, 1.0] to 16-bit signed integer
      const clamped = Math.max(-1.0, Math.min(1.0, sample));
      const s16 = clamped < 0 ? clamped * 32768 : clamped * 32767;
      this.buffer.push(Math.round(s16));

      if (this.buffer.length >= this.chunkSize) {
        const int16Array = new Int16Array(this.buffer.slice(0, this.chunkSize));
        this.port.postMessage(int16Array.buffer, [int16Array.buffer]);
        this.buffer = this.buffer.slice(this.chunkSize);
      }
    }

    return true;
  }
}

registerProcessor('mic-downsampler', MicDownsamplerProcessor);
