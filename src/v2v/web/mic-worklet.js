// AudioWorkletProcessor for high-performance, glitch-free 16kHz mono mic capture.
// Maintains a continuous fractional sample offset across 128-sample WebAudio blocks
// so zero samples are ever dropped at block boundaries.

class MicDownsamplerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetSampleRate = 16000;
    this.buffer = new Int16Array(512);
    this.bufIdx = 0;
    this.readOffset = 0.0;
    this.prevSample = 0.0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;

    const channelData = input[0];
    if (!channelData || channelData.length === 0) return true;

    const inLen = channelData.length;
    const ratio = sampleRate / this.targetSampleRate;

    // Fast path when AudioContext is already running natively at 16000 Hz
    if (Math.abs(ratio - 1.0) < 0.001) {
      for (let i = 0; i < inLen; i++) {
        const clamped = Math.max(-1.0, Math.min(1.0, channelData[i]));
        this.buffer[this.bufIdx++] = clamped < 0 ? Math.round(clamped * 32768) : Math.round(clamped * 32767);
        if (this.bufIdx >= 512) {
          const out = this.buffer.slice(0);
          this.port.postMessage(out.buffer, [out.buffer]);
          this.bufIdx = 0;
        }
      }
      return true;
    }

    // Continuous phase-preserving resampler with anti-aliasing averaging
    while (this.readOffset < inLen) {
      const i0 = Math.floor(this.readOffset);
      const frac = this.readOffset - i0;
      const s0 = i0 >= 0 ? channelData[i0] : this.prevSample;
      const s1 = i0 + 1 < inLen ? channelData[i0 + 1] : s0;
      let sample = s0 * (1.0 - frac) + s1 * frac;

      // Simple 2-point anti-aliasing smoothing when downsampling from 44.1/48kHz
      if (i0 + 2 < inLen && ratio >= 2.0) {
        sample = 0.7 * sample + 0.3 * channelData[i0 + 2];
      }

      const clamped = Math.max(-1.0, Math.min(1.0, sample));
      this.buffer[this.bufIdx++] = clamped < 0 ? Math.round(clamped * 32768) : Math.round(clamped * 32767);

      if (this.bufIdx >= 512) {
        const out = this.buffer.slice(0);
        this.port.postMessage(out.buffer, [out.buffer]);
        this.bufIdx = 0;
      }

      this.readOffset += ratio;
    }

    this.readOffset -= inLen;
    this.prevSample = channelData[inLen - 1];
    return true;
  }
}

registerProcessor('mic-downsampler', MicDownsamplerProcessor);
