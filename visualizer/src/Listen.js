import { headJoints } from './StewartIK.js';
import { SpeakerFeatures } from './ListenModel.js';

/**
 * Listening behaviour, live from the microphone: lean in, nod at pauses, perk the antennas when the voice lifts.
 * A line-for-line port of rmr/listen.py (keep the constants and the order of the update steps identical).
 *
 *   const lis = new Listener();  lis.step(dbfs) -> {head_pose, head_joints, antennas_position} once per 1/25 s
 */
export const FPS = 25;
const FLOOR_RISE = 0.004, VOICE_DB = 9.0, MIN_DB = -55.0, HANGOVER = 0.2, PAUSE = 0.3, MIN_TALK = 1.0;
const END_OF_TURN = 0.9, MIN_TURN = 0.6;
const DOUBLE_TALK = 3.0, REFRACTORY = 1.2, PERK_GAP = 3.0, ENGAGED = 4.0, FAST = 0.5, SLOW = 0.05, RISE_DB = 8.0;
const NOD_DEG = 7.0, NOD_S = 0.55, NOD_Z = -2.0;
const PERK_EARS = 25.0, PERK_ATTACK = 0.12, PERK_DECAY = 0.8, PERK_PITCH = -3.0, PERK_Z = 3.0;
const ATTENTIVE = { ears: 6.0, pitch: -2.0, roll: 6.0, z: 5.0 };
const IDLE = { ears: 15.0, pitch: 0.0, roll: 0.0, z: 3.0 };
const POSE_TAU = 0.6, BREATH_MM = 1.2, BREATH_S = 4.0;

/** The listening style (rmr.listen.STYLE): what a person tunes by comparing listeners side by side. */
export const STYLE = { nod_deg: NOD_DEG, pause: PAUSE, min_talk: MIN_TALK, double_talk: DOUBLE_TALK, lean: 1.0, perk: 1.0,
    sway_deg: 0.0, glances: 0.0 };
const GLANCE_DEG = 8.0, GLANCE_S = 1.2;
const SWAY_HZ = [[0.11, 0.23, 0.41], [0.07, 0.17, 0.31], [0.13, 0.19, 0.37]];   // pitch, yaw, roll

/** mulberry32, the same 32-bit arithmetic as rmr.listen._rng. */
function rng(seed) {
    let a = seed >>> 0;
    return () => {
        a = (a + 0x6D2B79F5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}
const ease = (u) => u <= 0 ? 0 : u >= 1 ? 1 : u * u * (3 - 2 * u);

const bump = (u) => (u <= 0 || u >= 1) ? 0 : Math.sin(Math.PI * u ** 0.7) ** 2;
const rad = (d) => d * Math.PI / 180;

/** Head pose (4x4, nested) from z (m) and roll, pitch, yaw (rad): R = Rz(yaw) Ry(pitch) Rx(roll), as rmr.motion.rpy_to_mat. */
function pose(z, roll, pitch, yaw = 0) {
    const cr = Math.cos(roll), sr = Math.sin(roll), cp = Math.cos(pitch), sp = Math.sin(pitch);
    const cy = Math.cos(yaw), sy = Math.sin(yaw);
    return [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, 0],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, 0],
        [-sp, cp * sr, cp * cr, z], [0, 0, 0, 1]];
}

export class Listener {
    /** head: optional LearnedHead (ListenModel.js); it then drives pitch, yaw and roll in place of the rule nods.
     *  style: overrides of STYLE; seed: for the sway phases and the glances. */
    constructor(side = 1.0, head = null, style = null, seed = 1) {
        this.head = head; this.feats = head ? new SpeakerFeatures() : null;
        this.style = { ...STYLE, ...(style || {}) }; this.rng = rng(seed);
        this.phase = [0, 1, 2].map(() => [0, 1, 2].map(() => 2 * Math.PI * this.rng()));
        this.glance = null;
        this.dt = 1 / FPS; this.t = 0; this.floor = null; this.fast = this.slow = null;
        this.sinceVoice = 1e9; this.talk = 0; this.lastNod = this.lastPerk = -1e9;
        this.nods = []; this.perkT = -1e9; this.side = side; this.engaged = 0; this.events = [];
        this.turnTalk = 0; this.turnStart = null; this.turnEnd = null;   // turnEnd: [start, end] on the ending frame
    }

    /** One frame: the speaker's loudness in dBFS -> viewer joint state. */
    step(db) {
        const t = this.t, dt = this.dt;
        if (this.floor === null || db < this.floor) this.floor = db;
        else this.floor += FLOOR_RISE * (db - this.floor);
        const voiced = db > this.floor + VOICE_DB && db > MIN_DB;
        if (voiced) {
            if (this.sinceVoice > ENGAGED) this.side = -this.side;
            this.sinceVoice = 0;
            this.fast = this.fast === null ? db : this.fast + FAST * (db - this.fast);
            this.slow = this.slow === null ? db : this.slow + SLOW * (db - this.slow);
        } else this.sinceVoice += dt;
        const speaking = this.sinceVoice < HANGOVER;
        if (speaking) {
            this.talk += dt; this.turnTalk += dt;
            if (this.turnStart === null) this.turnStart = t;
        }
        this.turnEnd = null;
        if (this.turnStart !== null && this.sinceVoice >= END_OF_TURN) {
            if (this.turnTalk >= MIN_TURN) { this.turnEnd = [this.turnStart, t]; this.events.push([t, 'turn']); }
            this.turnStart = null; this.turnTalk = 0;
        }

        const st = this.style;
        if (st.nod_deg > 0 && !speaking && this.sinceVoice >= st.pause && this.talk >= st.min_talk && t - this.lastNod >= REFRACTORY) {
            const double = this.talk >= st.double_talk;
            this.nods.push([t, 1.0, NOD_S]);
            if (double) this.nods.push([t + NOD_S * 0.9, 0.6, NOD_S * 0.8]);
            if (!this.head) this.events.push([t, double ? 'nod2' : 'nod']);
            this.lastNod = t; this.talk = 0;
        }
        if (st.perk > 0 && voiced && this.fast !== null && this.fast - this.slow > RISE_DB && t - this.lastPerk >= PERK_GAP) {
            this.perkT = this.lastPerk = t;
            this.events.push([t, 'perk']);
        }

        const target = this.sinceVoice < ENGAGED ? 1 : 0;
        this.engaged += (target - this.engaged) * (1 - Math.exp(-dt / POSE_TAU));
        const p = {};
        for (const k in IDLE) p[k] = IDLE[k] + st.lean * this.engaged * (ATTENTIVE[k] - IDLE[k]);
        p.roll *= this.side;

        const nod = this.nods.reduce((s, [s0, a, d]) => s + a * bump((t - s0) / d), 0);
        this.nods = this.nods.filter((n) => t - n[0] < n[2]);
        const u = t - this.perkT;
        const perk = st.perk * (u < 0 ? 0 : (u < PERK_ATTACK ? u / PERK_ATTACK : Math.exp(-(u - PERK_ATTACK) / PERK_DECAY)));

        const ears = p.ears - PERK_EARS * perk;
        const sway = [0, 1, 2].map((a) => st.sway_deg * this.engaged *
            SWAY_HZ[a].reduce((s, f, i) => s + Math.sin(2 * Math.PI * f * t + this.phase[a][i]), 0) / 3);
        let g = 0;
        if (this.glance === null && st.glances > 0 && this.engaged > 0.5) {
            if (this.rng() < st.glances / 60 / FPS) {
                const dir = this.rng() < 0.5 ? 1 : -1;
                this.glance = [t, dir * GLANCE_DEG * (0.6 + 0.4 * this.rng())];
            }
        }
        if (this.glance !== null) {
            const v = t - this.glance[0];
            g = this.glance[1] * (ease(v / 0.25) - ease((v - GLANCE_S + 0.35) / 0.35));
            if (v >= GLANCE_S) this.glance = null;
        }
        p.pitch += sway[0]; p.roll += sway[2];
        let yaw = sway[1] + g, n = nod;
        if (this.head) {
            const [hp, hy, hr] = this.head.step(this.feats.step(db, speaking));
            n = 0; yaw += hy; p.pitch += hp; p.roll += hr;
        }
        const pitch = p.pitch + st.nod_deg * n + PERK_PITCH * perk;
        const z = p.z + NOD_Z * n + PERK_Z * perk + BREATH_MM * Math.sin(2 * Math.PI * t / BREATH_S);
        this.t += dt;
        const head = pose(z / 1000, rad(p.roll), rad(pitch), rad(yaw));
        return { head_pose: head.flat(), head_joints: headJoints(head, 0), antennas_position: [-rad(ears), rad(ears)] };
    }
}

/** Mean-square of a block of samples -> dBFS (rmr.listen.loudness). */
export const dbfs = (ms) => 10 * Math.log10(ms + 1e-10);
