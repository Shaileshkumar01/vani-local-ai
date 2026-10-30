// Vāṇī Web Client: High-performance WebAudio + WebSocket streaming

class VoiceClient {
  constructor() {
    this.ws = null;
    this.audioCtx = null;
    this.micStream = null;
    this.workletNode = null;
    this.micAnalyser = null;
    this.speakerAnalyser = null;
    this.bargeInVoiceFrames = 0;

    this.isPlaying = false;
    this.currentTurnId = 0;
    this.nextPlayTime = 0;
    this.activeSourceNodes = [];
    this.bargeInEnabled = true;


    this.state = 'disconnected'; // disconnected, ready, listening, thinking, speaking
    this.currentAgentBubble = null;

    this.initDOMElements();
    this.initCanvasVisualizer();
    this.initEventListeners();
    this.pollSystemStatus();
  }

  initDOMElements() {
    this.connectBtn = document.getElementById('connect-btn');
    this.connectBtnText = document.getElementById('connect-btn-text');
    this.statusBadge = document.getElementById('status-badge');
    this.statusHint = document.getElementById('status-hint');
    this.voiceSelect = document.getElementById('voice-select');
    this.clearBtn = document.getElementById('clear-btn');
    this.gpuNameEl = document.getElementById('gpu-name');
    this.vramStatEl = document.getElementById('vram-stat');
    this.messagesContainer = document.getElementById('messages-container');
    this.orbContainer = document.querySelector('.orb-container');
    this.bargeInToggle = document.getElementById('barge-in-toggle');
    this.bargeInText = document.getElementById('barge-in-text');
  }

  initEventListeners() {
    this.connectBtn.addEventListener('click', () => this.toggleConnection());

    this.voiceSelect.addEventListener('change', (e) => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: 'set_voice', voice: e.target.value }));
      }
    });

    this.clearBtn.addEventListener('click', () => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: 'clear_history' }));
      }
      this.messagesContainer.innerHTML = `
        <div class="system-welcome">
          <p>Conversation memory cleared.</p>
        </div>`;
    });

    if (this.bargeInToggle) {
      this.bargeInToggle.addEventListener('click', () => {
        this.bargeInEnabled = !this.bargeInEnabled;
        this.updateBargeInUI();
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
          this.ws.send(JSON.stringify({ type: 'set_barge_in', enabled: this.bargeInEnabled }));
        }
      });
    }
  }

  async pollSystemStatus() {
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const data = await res.json();
        if (data.gpu) this.gpuNameEl.textContent = data.gpu;
        if (data.vram) this.vramStatEl.textContent = data.vram;
        if (data.tts_voice && this.voiceSelect) {
          this.voiceSelect.value = data.tts_voice;
        }
        if (typeof data.barge_in === 'boolean') {
          this.bargeInEnabled = data.barge_in;
          this.updateBargeInUI();
        }
      }
    } catch (e) {
      // Server warming up or offline
    }
    setTimeout(() => this.pollSystemStatus(), 5000);
  }

  updateBargeInUI() {
    if (!this.bargeInToggle || !this.bargeInText) return;
    if (this.bargeInEnabled) {
      this.bargeInToggle.classList.remove('safe-mode');
      this.bargeInText.textContent = 'Barge-in Active';
      this.bargeInToggle.title = 'Barge-in Active: You can speak over the agent. Click to enable Safe Mode.';
    } else {
      this.bargeInToggle.classList.add('safe-mode');
      this.bargeInText.textContent = 'Safe Mode (No Barge-in)';
      this.bargeInToggle.title = 'Safe Mode: Mic muted while agent speaks (prevents speaker echo). Click to enable Barge-in.';
    }
  }

  // ------------------------------------------------------------- Connection
  async toggleConnection() {
    if (this.state === 'disconnected') {
      await this.start();
    } else {
      this.stop();
    }
  }

  async start() {
    this.updateStatus('connecting', 'Connecting to local edge AI engine...');
    try {
      // 1. Initialize AudioContext (must happen on user gesture)
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      this.audioCtx = new AudioContextClass();
      if (this.audioCtx.state === 'suspended') {
        await this.audioCtx.resume();
      }

      // 2. Setup Analysers for Mic (barge-in detection) and Speaker (audio playback)
      this.micAnalyser = this.audioCtx.createAnalyser();
      this.micAnalyser.fftSize = 128;
      this.micAnalyser.smoothingTimeConstant = 0.5;

      this.speakerAnalyser = this.audioCtx.createAnalyser();
      this.speakerAnalyser.fftSize = 128;
      this.speakerAnalyser.smoothingTimeConstant = 0.8;

      // 3. Connect WebSocket
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}/ws`;
      this.ws = new WebSocket(wsUrl);
      this.ws.binaryType = 'arraybuffer';

      this.ws.onopen = async () => {
        await this.startMicrophone();
        this.updateStatus('ready', 'Connected! Speak into your microphone...');
        this.connectBtn.classList.add('btn-active');
        this.connectBtnText.textContent = 'Stop Session';
        this.ws.send(JSON.stringify({ type: 'set_barge_in', enabled: this.bargeInEnabled }));
      };

      this.ws.onmessage = (event) => this.handleMessage(event);

      this.ws.onclose = () => {
        this.stop();
        this.updateStatus('disconnected', 'Session closed. Click start to reconnect.');
      };

      this.ws.onerror = (err) => {
        console.error('WebSocket error:', err);
        this.updateStatus('disconnected', 'Connection error.');
      };
    } catch (err) {
      console.error('Failed to start voice session:', err);
      alert(`Could not access microphone or connect: ${err.message}`);
      this.stop();
    }
  }

  stop() {
    if (this.workletNode) {
      this.workletNode.disconnect();
      this.workletNode = null;
    }
    if (this.micStream) {
      this.micStream.getTracks().forEach((t) => t.stop());
      this.micStream = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.stopAllAudio();
    this.state = 'disconnected';
    this.connectBtn.classList.remove('btn-active');
    this.connectBtnText.textContent = 'Start Voice Agent';
    this.updateStatus('disconnected', 'Click Start Voice Agent to begin');
  }

  // ------------------------------------------------------------- Microphone
  async startMicrophone() {
    this.micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
      },
    });

    const micSource = this.audioCtx.createMediaStreamSource(this.micStream);
    micSource.connect(this.micAnalyser);


    // Try AudioWorklet first, with ScriptProcessor fallback
    try {
      await this.audioCtx.audioWorklet.addModule('/static/mic-worklet.js');
      this.workletNode = new AudioWorkletNode(this.audioCtx, 'mic-downsampler');
      this.workletNode.port.onmessage = (event) => {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
          this.ws.send(event.data);
        }
      };
      micSource.connect(this.workletNode);
    } catch (e) {
      console.warn('AudioWorklet failed, using ScriptProcessor fallback:', e);
      this.startScriptProcessorFallback(micSource);
    }
  }

  startScriptProcessorFallback(micSource) {
    const bufferSize = 2048;
    const spNode = this.audioCtx.createScriptProcessor(bufferSize, 1, 1);
    const targetSampleRate = 16000;
    let downsampleBuffer = [];

    spNode.onaudioprocess = (e) => {
      if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
      const input = e.inputBuffer.getChannelData(0);
      const ratio = this.audioCtx.sampleRate / targetSampleRate;
      const newLen = Math.floor(input.length / ratio);

      for (let i = 0; i < newLen; i++) {
        const srcIdx = i * ratio;
        const i0 = Math.floor(srcIdx);
        const i1 = Math.min(i0 + 1, input.length - 1);
        const frac = srcIdx - i0;
        const sample = input[i0] * (1 - frac) + input[i1] * frac;
        const clamped = Math.max(-1.0, Math.min(1.0, sample));
        const s16 = clamped < 0 ? clamped * 32768 : clamped * 32767;
        downsampleBuffer.push(Math.round(s16));

        if (downsampleBuffer.length >= 512) {
          const chunk = new Int16Array(downsampleBuffer.slice(0, 512));
          this.ws.send(chunk.buffer);
          downsampleBuffer = downsampleBuffer.slice(512);
        }
      }
    };
    micSource.connect(spNode);
    spNode.connect(this.audioCtx.destination);
    this.workletNode = spNode;
  }

  // ------------------------------------------------------------- Message Handling
  handleMessage(event) {
    if (typeof event.data === 'string') {
      const data = JSON.parse(event.data);
      this.handleJsonEvent(data);
    } else {
      this.handleAudioChunk(event.data);
    }
  }

  handleJsonEvent(data) {
    switch (data.type) {
      case 'status':
        if (data.status === 'listening') {
          this.updateStatus('listening', 'Listening to you...');
        } else if (data.status === 'processing') {
          this.updateStatus('thinking', 'Processing utterance...');
        } else if (data.status === 'ready') {
          // Only switch to ready if audio playback has actually finished
          if (this.activeSourceNodes.length === 0) {
            this.updateStatus('ready', 'Ready for next turn');
          }
        }
        break;

      case 'user_transcript':
        this.appendUserMessage(data.text);
        this.currentTurnId = data.turn_id;
        this.currentAgentBubble = this.createAgentMessage();
        break;

      case 'agent_token':
        if (this.currentAgentBubble) {
          this.appendAgentToken(data.token);
        }
        break;

      case 'interrupt':
        this.stopAllAudio();
        if (data.new_turn_id) {
          this.currentTurnId = data.new_turn_id;
        }
        if (this.currentAgentBubble) {
          this.markInterrupted(this.currentAgentBubble);
        }
        this.updateStatus('listening', 'Barge-in: listening to you...');
        break;

      case 'metrics':
        this.updateLatencyMetrics(data.data);
        break;

      case 'turn_complete':
        if (this.currentAgentBubble) {
          const cursor = this.currentAgentBubble.querySelector('.typing-cursor');
          if (cursor) cursor.remove();
        }
        break;

      case 'barge_in_updated':
        this.bargeInEnabled = Boolean(data.enabled);
        this.updateBargeInUI();
        break;
    }
  }

  // ------------------------------------------------------------- Audio Playback
  handleAudioChunk(arrayBuffer) {
    if (arrayBuffer.byteLength < 4) return;
    const view = new DataView(arrayBuffer);
    const turnId = view.getUint32(0, true); // Little-endian turn ID

    // Drop stale audio chunks from canceled/older turns
    if (turnId < this.currentTurnId) {
      return;
    }
    if (turnId > this.currentTurnId) {
      this.currentTurnId = turnId;
    }

    const wasIdle = this.activeSourceNodes.length === 0;
    this.updateStatus('speaking', 'Speaking...');

    if (wasIdle && this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'playback_start' }));
    }

    // Extract Int16 PCM (24kHz mono)
    const pcmData = new Int16Array(arrayBuffer, 4);
    const floatData = new Float32Array(pcmData.length);
    for (let i = 0; i < pcmData.length; i++) {
      floatData[i] = pcmData[i] / 32768.0;
    }

    const sampleRate = 24000;
    const audioBuffer = this.audioCtx.createBuffer(1, floatData.length, sampleRate);
    audioBuffer.getChannelData(0).set(floatData);

    const source = this.audioCtx.createBufferSource();
    source.buffer = audioBuffer;

    // Connect to speaker visualizer analyser and destination
    if (this.speakerAnalyser) {
      source.connect(this.speakerAnalyser);
    }
    source.connect(this.audioCtx.destination);

    // Schedule back-to-back without gap
    const now = this.audioCtx.currentTime;
    if (this.nextPlayTime < now) {
      this.nextPlayTime = now + 0.015; // 15ms lead buffer
    }

    source.start(this.nextPlayTime);
    this.nextPlayTime += audioBuffer.duration;
    this.activeSourceNodes.push(source);

    source.onended = () => {
      const idx = this.activeSourceNodes.indexOf(source);
      if (idx !== -1) this.activeSourceNodes.splice(idx, 1);
      if (this.activeSourceNodes.length === 0) {
        this.updateStatus('ready', 'Ready for next turn');
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
          this.ws.send(JSON.stringify({ type: 'playback_end' }));
        }
      }
    };
  }

  stopAllAudio() {
    const wasPlaying = this.activeSourceNodes.length > 0;
    for (const src of this.activeSourceNodes) {
      try {
        src.stop();
        src.disconnect();
      } catch (e) {}
    }
    this.activeSourceNodes = [];
    if (this.audioCtx) {
      this.nextPlayTime = this.audioCtx.currentTime;
    }
    if (wasPlaying && this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'playback_end' }));
    }
    this.updateStatus('ready', 'Ready for next turn');
  }

  // ------------------------------------------------------------- UI Updates

  updateStatus(state, hintText) {
    this.state = state;
    this.statusHint.textContent = hintText;

    this.orbContainer.className = 'orb-container';
    if (state === 'listening') {
      this.orbContainer.classList.add('state-listening');
      this.statusBadge.textContent = '● Listening';
      this.statusBadge.style.color = 'var(--accent-emerald)';
    } else if (state === 'thinking') {
      this.orbContainer.classList.add('state-thinking');
      this.statusBadge.textContent = '✦ Thinking';
      this.statusBadge.style.color = 'var(--accent-amber)';
    } else if (state === 'speaking') {
      this.orbContainer.classList.add('state-speaking');
      this.statusBadge.textContent = '🔊 Speaking';
      this.statusBadge.style.color = 'var(--accent-cyan)';
    } else if (state === 'ready') {
      this.statusBadge.textContent = '✓ Ready';
      this.statusBadge.style.color = 'var(--text-muted)';
    } else {
      this.statusBadge.textContent = 'Disconnected';
      this.statusBadge.style.color = 'var(--text-dim)';
    }
  }

  appendUserMessage(text) {
    const welcome = this.messagesContainer.querySelector('.system-welcome');
    if (welcome) welcome.remove();

    const bubble = document.createElement('div');
    bubble.className = 'chat-bubble user';
    bubble.innerHTML = `
      <div class="bubble-sender">You</div>
      <div class="bubble-body">${this.escapeHtml(text)}</div>
    `;
    this.messagesContainer.appendChild(bubble);
    this.scrollDialogue();
  }

  createAgentMessage() {
    const bubble = document.createElement('div');
    bubble.className = 'chat-bubble agent';
    bubble.innerHTML = `
      <div class="bubble-sender">Vāṇī Agent</div>
      <div class="bubble-body"><span class="token-content"></span><span class="typing-cursor"></span></div>
    `;
    this.messagesContainer.appendChild(bubble);
    this.scrollDialogue();
    return bubble;
  }

  appendAgentToken(tok) {
    if (!this.currentAgentBubble) return;
    const contentSpan = this.currentAgentBubble.querySelector('.token-content');
    if (contentSpan) {
      contentSpan.textContent += tok;
      this.scrollDialogue();
    }
  }

  markInterrupted(bubble) {
    const body = bubble.querySelector('.bubble-body');
    if (body && !body.querySelector('.interrupted-tag')) {
      const cursor = body.querySelector('.typing-cursor');
      if (cursor) cursor.remove();
      const tag = document.createElement('span');
      tag.className = 'interrupted-tag';
      tag.textContent = 'Interrupted';
      body.appendChild(tag);
    }
  }

  scrollDialogue() {
    this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
  }

  updateLatencyMetrics(d) {
    const fmt = (v) => (v !== null && v !== undefined ? `${Math.round(v)} ms` : '-- ms');

    document.getElementById('val-endpoint').textContent = fmt(d.endpoint_ms);
    document.getElementById('val-stt').textContent = fmt(d.stt_done_ms);
    document.getElementById('val-llm').textContent = fmt(d.llm_first_token_ms);
    document.getElementById('val-phrase').textContent = fmt(d.first_phrase_ms);
    document.getElementById('val-tts').textContent = fmt(d.tts_first_audio_ms);
    document.getElementById('val-total').textContent = fmt(d.processing_ms);
    document.getElementById('val-full-sub').textContent = `${fmt(d.playback_start_ms)} from speech end`;

    // Highlight total turnaround card green if sub-300ms
    const totalCard = document.getElementById('card-total');
    if (d.processing_ms !== null && d.processing_ms <= 300) {
      totalCard.style.borderColor = 'var(--accent-emerald)';
    } else {
      totalCard.style.borderColor = 'var(--border-glow)';
    }

    // Update waterfall progress bars
    const total = (d.playback_start_ms || 500) - (d.endpoint_ms || 0);
    if (total > 0) {
      const sttPct = Math.max(5, (((d.stt_done_ms || 0) - (d.endpoint_ms || 0)) / total) * 100);
      const llmPct = Math.max(5, (((d.llm_first_token_ms || 0) - (d.stt_done_ms || 0)) / total) * 100);
      const phrasePct = Math.max(5, (((d.first_phrase_ms || 0) - (d.llm_first_token_ms || 0)) / total) * 100);
      const ttsPct = Math.max(5, (((d.tts_first_audio_ms || 0) - (d.first_phrase_ms || 0)) / total) * 100);

      document.getElementById('bar-stt').style.width = `${sttPct}%`;
      document.getElementById('bar-llm').style.width = `${llmPct}%`;
      document.getElementById('bar-phrase').style.width = `${phrasePct}%`;
      document.getElementById('bar-tts').style.width = `${ttsPct}%`;
    }
  }

  escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  // ------------------------------------------------------------- Visualizer Canvas
  initCanvasVisualizer() {
    const canvas = document.getElementById('orb-canvas');
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    canvas.width = 460 * dpr;
    canvas.height = 460 * dpr;
    ctx.scale(dpr, dpr);

    const freqData = new Uint8Array(64);
    let angle = 0;

    const render = () => {
      requestAnimationFrame(render);
      ctx.clearRect(0, 0, 460, 460);

      let audioEnergy = 0;
      if (this.audioCtx && this.state !== 'disconnected') {
        let micSum = 0;
        let spkSum = 0;

        if (this.micAnalyser) {
          this.micAnalyser.getByteFrequencyData(freqData);
          for (let i = 0; i < 32; i++) micSum += freqData[i];
        }
        const micEnergy = micSum / 32 / 255;

        if (this.speakerAnalyser && this.state === 'speaking') {
          this.speakerAnalyser.getByteFrequencyData(freqData);
          for (let i = 0; i < 32; i++) spkSum += freqData[i];
        }
        const spkEnergy = spkSum / 32 / 255;

        // Visualizer reflects either speaker or mic
        audioEnergy = this.state === 'speaking' ? Math.max(spkEnergy, micEnergy) : micEnergy;
      }



      const cx = 230;
      const cy = 230;
      const baseRadius = 100 + audioEnergy * 45;
      angle += 0.02 + audioEnergy * 0.05;

      // Color scheme based on state
      let c1 = 'rgba(0, 242, 254, 0.7)';
      let c2 = 'rgba(79, 172, 254, 0.4)';
      let c3 = 'rgba(142, 45, 226, 0.6)';

      if (this.state === 'listening') {
        c1 = 'rgba(16, 185, 129, 0.8)';
        c2 = 'rgba(0, 242, 254, 0.5)';
        c3 = 'rgba(16, 185, 129, 0.3)';
      } else if (this.state === 'thinking') {
        c1 = 'rgba(245, 158, 11, 0.8)';
        c2 = 'rgba(236, 72, 153, 0.6)';
        c3 = 'rgba(142, 45, 226, 0.4)';
      } else if (this.state === 'speaking') {
        c1 = 'rgba(0, 242, 254, 0.9)';
        c2 = 'rgba(79, 172, 254, 0.7)';
        c3 = 'rgba(0, 242, 254, 0.4)';
      }

      // Outer undulating waves
      for (let ring = 0; ring < 3; ring++) {
        ctx.beginPath();
        const pts = 64;
        const rOffset = ring * 12;
        for (let i = 0; i <= pts; i++) {
          const theta = (i / pts) * Math.PI * 2;
          const wave =
            Math.sin(theta * 5 + angle + ring) * (8 + audioEnergy * 24) +
            Math.cos(theta * 3 - angle * 0.8) * (6 + audioEnergy * 16);
          const r = baseRadius + wave - rOffset;
          const x = cx + Math.cos(theta) * r;
          const y = cy + Math.sin(theta) * r;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.strokeStyle = ring === 0 ? c1 : ring === 1 ? c2 : c3;
        ctx.lineWidth = 2 + (2 - ring) * 0.5;
        ctx.stroke();
      }

      // Inner glowing core
      const grad = ctx.createRadialGradient(cx, cy, 20, cx, cy, baseRadius);
      grad.addColorStop(0, c1);
      grad.addColorStop(0.6, c2);
      grad.addColorStop(1, 'transparent');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(cx, cy, baseRadius * 0.95, 0, Math.PI * 2);
      ctx.fill();
    };

    render();
  }
}

// Instantiate client when DOM is loaded
window.addEventListener('DOMContentLoaded', () => {
  window.v2vClient = new VoiceClient();
});
