// ---------- Synthesized retro sound effects (WebAudio) ----------
'use strict';

class Sfx {
  constructor() {
    this.ctx = null; this.muted = false; this.vol = 0.5;
    this.last = {};
    this.alarmT = 0;
  }
  ensure() {
    if (this.ctx) return true;
    try {
      this.ctx = new (window.AudioContext || window.webkitAudioContext)();
      this.master = this.ctx.createGain(); this.master.gain.value = this.vol; this.master.connect(this.ctx.destination);
      const len = this.ctx.sampleRate;
      this.noise = this.ctx.createBuffer(1, len, this.ctx.sampleRate);
      const d = this.noise.getChannelData(0);
      for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
    } catch (e) { this.ctx = null; }
    return !!this.ctx;
  }
  tone(type, f0, f1, dur, vol = 0.3, delay = 0) {
    const c = this.ctx, t = c.currentTime + delay;
    const o = c.createOscillator(), g = c.createGain();
    o.type = type; o.frequency.setValueAtTime(f0, t);
    if (f1 !== f0) o.frequency.exponentialRampToValueAtTime(Math.max(1, f1), t + dur);
    g.gain.setValueAtTime(vol, t); g.gain.exponentialRampToValueAtTime(0.001, t + dur);
    o.connect(g); g.connect(this.master); o.start(t); o.stop(t + dur + 0.02);
  }
  noiseBurst(dur, vol = 0.3, filt = 1000, q = 1, delay = 0, type = 'lowpass') {
    const c = this.ctx, t = c.currentTime + delay;
    const s = c.createBufferSource(); s.buffer = this.noise;
    const f = c.createBiquadFilter(); f.type = type; f.frequency.value = filt; f.Q.value = q;
    const g = c.createGain(); g.gain.setValueAtTime(vol, t); g.gain.exponentialRampToValueAtTime(0.001, t + dur);
    s.connect(f); f.connect(g); g.connect(this.master); s.start(t); s.stop(t + dur + 0.02);
  }
  // ---- original chiptune loop: scheduled a bar ahead ----
  startMusic() {
    if (!this.ensure() || this.musicOn) return;
    this.musicOn = true;
    this.musicGain = this.ctx.createGain(); this.musicGain.gain.value = 0.22; this.musicGain.connect(this.master);
    this.bar = 0; this.nextBarT = this.ctx.currentTime + 0.1;
    this.musicTimer = setInterval(() => this.scheduleMusic(), 200);
  }
  stopMusic() {
    this.musicOn = false;
    clearInterval(this.musicTimer);
    if (this.musicGain) { this.musicGain.gain.setTargetAtTime(0, this.ctx.currentTime, 0.2); this.musicGain = null; }
  }
  note(type, freq, t, dur, vol) {
    const c = this.ctx, o = c.createOscillator(), g = c.createGain();
    o.type = type; o.frequency.value = freq;
    g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(vol, t + 0.02); g.gain.exponentialRampToValueAtTime(0.001, t + dur);
    o.connect(g); g.connect(this.musicGain); o.start(t); o.stop(t + dur + 0.05);
  }
  scheduleMusic() {
    if (!this.musicOn || !this.musicGain) return;
    const c = this.ctx;
    const beat = 60 / 84;
    const hz = (m) => 440 * Math.pow(2, (m - 69) / 12);
    // D major: I  V  vi  IV / I  IV  V  V  (MIDI roots)
    const prog = [[50, 54, 57], [57, 61, 64], [59, 62, 66], [55, 59, 62], [50, 54, 57], [55, 59, 62], [57, 61, 64], [57, 61, 64]];
    // melody per bar: [beat offset, midi, length in beats]
    const mel = [
      [[0, 74, 1.5], [1.5, 76, 0.5], [2, 78, 2]],
      [[0, 76, 1], [1, 73, 1], [2, 69, 2]],
      [[0, 71, 1.5], [1.5, 73, 0.5], [2, 74, 1], [3, 76, 1]],
      [[0, 74, 3], [3, 71, 1]],
      [[0, 74, 1.5], [1.5, 76, 0.5], [2, 78, 1], [3, 81, 1]],
      [[0, 79, 2], [2, 78, 1], [3, 76, 1]],
      [[0, 76, 1], [1, 78, 1], [2, 76, 1], [3, 73, 1]],
      [[0, 74, 4]],
    ];
    while (this.nextBarT < c.currentTime + 1.2) {
      const t0 = this.nextBarT, i = this.bar % 8, chord = prog[i];
      const quiet = Math.floor(this.bar / 8) % 2 === 1; // every other phrase: no lead, just the bed
      this.note('triangle', hz(chord[0] - 12), t0, beat * 3.8, 0.22);
      for (let k = 0; k < 8; k++) this.note('triangle', hz(chord[k % 3] + (k >= 4 ? 12 : 0)), t0 + k * beat / 2, beat * 0.45, 0.07);
      if (!quiet) for (const [o, m, l] of mel[i]) this.note('square', hz(m), t0 + o * beat, l * beat * 0.95, 0.045);
      else this.note('sine', hz(chord[1] + 12), t0, beat * 4, 0.05);
      this.nextBarT += beat * 4; this.bar++;
    }
  }

  play(name, vol = 1) {
    if (this.muted || !this.ensure()) return;
    const now = performance.now();
    const minGap = { screech: 700, honk: 600, bellow: 1500, roar: 900, zap: 120, thud: 100, scream: 400, dart: 80, chomp: 150, alarm: 2500, thunder: 300, crash: 200 }[name] || 60;
    if (this.last[name] && now - this.last[name] < minGap) return;
    this.last[name] = now;
    const v = clamp(vol, 0, 1.2);
    switch (name) {
      case 'click': this.tone('square', 880, 660, 0.05, 0.08 * v); break;
      case 'build': this.tone('square', 220, 440, 0.08, 0.12 * v); this.noiseBurst(0.08, 0.15 * v, 1500); break;
      case 'demolish': this.noiseBurst(0.25, 0.3 * v, 600); this.tone('sawtooth', 120, 40, 0.25, 0.15 * v); break;
      case 'error': this.tone('square', 200, 150, 0.15, 0.12 * v); break;
      case 'cash': this.tone('square', 1320, 1320, 0.05, 0.08 * v); this.tone('square', 1760, 1760, 0.08, 0.08 * v, 0.05); break;
      case 'roar':
        this.noiseBurst(1.2, 0.5 * v, 400, 4); this.tone('sawtooth', 90, 45, 1.2, 0.25 * v); this.tone('sawtooth', 140, 60, 1.0, 0.15 * v, 0.05); break;
      case 'screech': this.tone('sawtooth', 1400, 2400, 0.12, 0.08 * v); this.tone('sawtooth', 2200, 900, 0.35, 0.08 * v, 0.1); this.noiseBurst(0.3, 0.1 * v, 3000, 3, 0.05, 'bandpass'); break;
      case 'honk': this.tone('triangle', 220, 260, 0.35, 0.12 * v); this.tone('triangle', 330, 300, 0.4, 0.06 * v); break;
      case 'bellow': this.tone('sine', 70, 55, 1.4, 0.3 * v); this.tone('triangle', 140, 110, 1.2, 0.08 * v); break;
      case 'zap': this.noiseBurst(0.15, 0.25 * v, 4000, 2, 0, 'highpass'); this.tone('square', 1800, 300, 0.12, 0.1 * v); break;
      case 'thud': this.tone('sine', 120, 40, 0.18, 0.4 * v); this.noiseBurst(0.1, 0.2 * v, 300); break;
      case 'crash': this.noiseBurst(0.6, 0.5 * v, 900); this.tone('sawtooth', 80, 30, 0.5, 0.25 * v); break;
      case 'chomp': this.noiseBurst(0.08, 0.35 * v, 800); this.noiseBurst(0.08, 0.3 * v, 600, 1, 0.12); break;
      case 'scream': this.tone('sawtooth', 900, 1400, 0.3, 0.08 * v); this.tone('sawtooth', 1400, 600, 0.4, 0.08 * v, 0.3); break;
      case 'dart': this.noiseBurst(0.06, 0.2 * v, 3000, 1, 0, 'highpass'); this.tone('square', 600, 1200, 0.05, 0.05 * v); break;
      case 'alarm': for (let i = 0; i < 4; i++) { this.tone('square', 660, 880, 0.28, 0.1 * v, i * 0.6); this.tone('square', 880, 660, 0.28, 0.1 * v, i * 0.6 + 0.3); } break;
      case 'thunder': this.noiseBurst(1.8, 0.6 * v, 200, 1); this.noiseBurst(0.2, 0.4 * v, 3000, 1); break;
      case 'boom': this.noiseBurst(2.5, 0.7 * v, 120, 1); this.tone('sine', 60, 25, 2.0, 0.4 * v); break;
      case 'powerdown': this.tone('sawtooth', 440, 40, 1.5, 0.2 * v); break;
      case 'powerup': this.tone('sawtooth', 60, 440, 1.0, 0.15 * v); break;
      case 'hatch': this.tone('triangle', 660, 990, 0.1, 0.15 * v); this.tone('triangle', 990, 1320, 0.15, 0.15 * v, 0.1); break;
      case 'fanfare': [523, 659, 784, 1046].forEach((f, i) => this.tone('square', f, f, 0.18, 0.09 * v, i * 0.12)); break;
      case 'chirp': { const f = 2000 + Math.random() * 1800; this.tone('sine', f, f * 1.3, 0.07, 0.025 * v); this.tone('sine', f * 1.1, f * 0.9, 0.08, 0.025 * v, 0.1); break; }
      case 'cricket': for (let i = 0; i < 3; i++) this.tone('square', 4200, 4300, 0.03, 0.008 * v, i * 0.06); break;
      case 'caw': this.tone('sawtooth', 700, 420, 0.25, 0.03 * v); break;
      case 'heli': this.noiseBurst(0.8, 0.15 * v, 200, 6); break;
    }
  }
}
