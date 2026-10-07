/**
 * The learned listener's head motion (rmr/listen_model.py LearnedHead), streaming in the browser.
 * Speaker features (speakerFeatures, causal, one frame at a time) -> GRU -> sampled change of pitch, yaw, roll.
 * The weights are CC-BY-NC (trained on Meta's Seamless Interaction dataset) and are not bundled: the server serves
 * them at api/listener when started with LISTENER_MODEL; without them the viewer keeps the rule-based head.
 */
const FPS = 25, BINS = 31, DMAX = 3.0, H = 96;
const HP_ALPHA = 1 - Math.exp(-2 * Math.PI * 0.3 / FPS);   // one-pole 0.3 Hz low-pass (rmr.listen_model)

/** Causal per-frame version of rmr.listen_model.speaker_features. */
export class SpeakerFeatures {
    constructor() { this.ref = null; this.a = 1 - Math.exp(-1 / (FPS * 20)); this.prev = 0; this.since = 0; this.talk = 0; }
    step(db, vad) {
        const v = vad ? 1 : 0;
        if (vad) this.ref = this.ref === null ? db : this.ref + this.a * (db - this.ref);
        const level = this.ref === null ? -60 : this.ref;
        const rel = Math.max(-40, Math.min(15, db - level)) / 20;
        const onset = Math.max(0, v - this.prev), offset = Math.max(0, this.prev - v);
        this.prev = v;
        this.since = (onset || offset) ? 0 : this.since + 1 / FPS;
        this.talk = vad ? this.talk + 1 / FPS : 0;
        return [rel, v, onset, offset, Math.tanh(this.since / 2), Math.tanh(this.talk / 5)];
    }
}

const sig = (x) => 1 / (1 + Math.exp(-x));
const matvec = (W, x) => W.map((row) => row.reduce((s, w, j) => s + w * x[j], 0));

export class LearnedHead {
    /** weights: the JSON from scripts/train_listener.py; random: () => [0, 1) */
    constructor(weights, temperature = 0.8, random = Math.random) {
        this.Wi = weights['gru.weight_ih_l0']; this.Wh = weights['gru.weight_hh_l0'];
        this.bi = weights['gru.bias_ih_l0']; this.bh = weights['gru.bias_hh_l0'];
        this.Wo = weights['head.weight']; this.bo = weights['head.bias'];
        this.h = new Array(H).fill(0); this.y = [0, 0, 0]; this.d = [0, 0, 0]; this.slow = [0, 0, 0];
        this.temperature = temperature; this.random = random;
    }

    logits(f) {
        const x = [...f, ...this.y.map((v) => v / 10), ...this.d.map((v) => v / DMAX)];
        const gi = matvec(this.Wi, x).map((v, i) => v + this.bi[i]), gh = matvec(this.Wh, this.h).map((v, i) => v + this.bh[i]);
        const h = this.h.map((hp, k) => {
            const r = sig(gi[k] + gh[k]), z = sig(gi[H + k] + gh[H + k]), n = Math.tanh(gi[2 * H + k] + r * gh[2 * H + k]);
            return (1 - z) * n + z * hp;
        });
        this.h = h;
        const o = matvec(this.Wo, h).map((v, i) => v + this.bo[i]);
        return [0, 1, 2].map((a) => o.slice(a * BINS, (a + 1) * BINS));
    }

    /** One frame -> [pitch, yaw, roll] of the fast head motion, degrees (pitch + = down). u: optional uniforms. */
    step(f, u = null) {
        const lg = this.logits(f), t = Math.max(this.temperature, 1e-3);
        this.d = lg.map((row, a) => {
            const m = Math.max(...row), p = row.map((v) => Math.exp((v - m) / t)), s = p.reduce((x, y) => x + y, 0);
            const r = u ? u[a] : this.random();
            let c = 0, k = BINS - 1;
            for (let i = 0; i < BINS; i++) { c += p[i] / s; if (r <= c) { k = i; break; } }
            return k / (BINS - 1) * 2 * DMAX - DMAX;
        });
        this.y = this.y.map((v, a) => v + this.d[a]);
        // the same causal 0.3 Hz high-pass as the training targets: no slow drift
        this.slow = this.slow.map((v, a) => v + HP_ALPHA * (this.y[a] - v));
        return this.y.map((v, a) => v - this.slow[a]);
    }
}
