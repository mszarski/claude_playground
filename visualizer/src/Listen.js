import { headJoints } from './StewartIK.js';

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

const bump = (u) => (u <= 0 || u >= 1) ? 0 : Math.sin(Math.PI * u ** 0.7) ** 2;
const rad = (d) => d * Math.PI / 180;

/** Head pose (4x4, nested) from z (m), roll and pitch (rad): R = Ry(pitch) @ Rx(roll), as rmr.motion.rpy_to_mat. */
function pose(z, roll, pitch) {
    const cr = Math.cos(roll), sr = Math.sin(roll), cp = Math.cos(pitch), sp = Math.sin(pitch);
    return [[cp, sp * sr, sp * cr, 0], [0, cr, -sr, 0], [-sp, cp * sr, cp * cr, z], [0, 0, 0, 1]];
}

export class Listener {
    constructor(side = 1.0) {
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

        if (!speaking && this.sinceVoice >= PAUSE && this.talk >= MIN_TALK && t - this.lastNod >= REFRACTORY) {
            const double = this.talk >= DOUBLE_TALK;
            this.nods.push([t, 1.0, NOD_S]);
            if (double) this.nods.push([t + NOD_S * 0.9, 0.6, NOD_S * 0.8]);
            this.events.push([t, double ? 'nod2' : 'nod']);
            this.lastNod = t; this.talk = 0;
        }
        if (voiced && this.fast !== null && this.fast - this.slow > RISE_DB && t - this.lastPerk >= PERK_GAP) {
            this.perkT = this.lastPerk = t;
            this.events.push([t, 'perk']);
        }

        const target = this.sinceVoice < ENGAGED ? 1 : 0;
        this.engaged += (target - this.engaged) * (1 - Math.exp(-dt / POSE_TAU));
        const p = {};
        for (const k in IDLE) p[k] = IDLE[k] + this.engaged * (ATTENTIVE[k] - IDLE[k]);
        p.roll *= this.side;

        const nod = this.nods.reduce((s, [s0, a, d]) => s + a * bump((t - s0) / d), 0);
        this.nods = this.nods.filter((n) => t - n[0] < n[2]);
        const u = t - this.perkT;
        const perk = u < 0 ? 0 : (u < PERK_ATTACK ? u / PERK_ATTACK : Math.exp(-(u - PERK_ATTACK) / PERK_DECAY));

        const ears = p.ears - PERK_EARS * perk;
        const pitch = p.pitch + NOD_DEG * nod + PERK_PITCH * perk;
        const z = p.z + NOD_Z * nod + PERK_Z * perk + BREATH_MM * Math.sin(2 * Math.PI * t / BREATH_S);
        this.t += dt;
        const head = pose(z / 1000, rad(p.roll), rad(pitch));
        return { head_pose: head.flat(), head_joints: headJoints(head, 0), antennas_position: [-rad(ears), rad(ears)] };
    }
}

/** Mean-square of a block of samples -> dBFS (rmr.listen.loudness). */
export const dbfs = (ms) => 10 * Math.log10(ms + 1e-10);
