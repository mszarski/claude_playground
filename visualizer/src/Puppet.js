import { feasible, headJoints } from './StewartIK.js';
import { pose } from './Listen.js';

/**
 * Puppeteering: a face tracked by the phone's camera (MediaPipe Face Landmarker) drives Reachy Mini.
 *
 * Head: the face's rotation and height relative to a neutral pose captured at the start (`calibrate`) become head
 * yaw, pitch, roll and height. In mirror mode (the default) Reachy moves like your reflection: turn to your left and
 * it turns to the same side of the screen. Antennas: eyebrows. Raised brows perk an antenna up, a frown droops it
 * (and a frowning mouth droops both a little). Everything is smoothed and kept inside the recipe limits
 * (rmr.recipe.LIMITS), and a pose the head cannot reach holds the last one that it can.
 *
 * Units out of `frame`: degrees (pitch + = head down, roll, yaw + = to Reachy's left), mm (z), antenna droop in
 * degrees (0 up, 15 relaxed, 150 drooped), as in rmr.plan.
 */
const LIMIT = { pitch: 30, roll: 25, yaw: 50, z: [-20, 25], ear: [-25, 150] };
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const deg = (r) => r * 180 / Math.PI, rad = (d) => d * Math.PI / 180;

/** MediaPipe facial transformation matrix (column-major Float32Array(16), cm) -> {R: 3x3 rows, t: [x, y, z]}. */
export function fromMediaPipe(data) {
    const m = (r, c) => data[c * 4 + r];
    return { R: [0, 1, 2].map((r) => [0, 1, 2].map((c) => m(r, c))), t: [m(0, 3), m(1, 3), m(2, 3)] };
}

const mul = (A, B) => A.map((r) => [0, 1, 2].map((j) => r[0] * B[0][j] + r[1] * B[1][j] + r[2] * B[2][j]));
const T = (A) => [0, 1, 2].map((i) => [0, 1, 2].map((j) => A[j][i]));

export class Retarget {
    constructor({ mirror = true, smooth = 0.45 } = {}) {
        this.mirror = mirror; this.smooth = smooth; this.ref = null; this.out = null;
    }

    calibrate(m) { this.ref = m; this.out = null; }

    /** m: fromMediaPipe(...); shapes: {categoryName: score}. -> {pitch, roll, yaw, z, earR, earL}. */
    frame(m, shapes = {}) {
        if (!this.ref) this.calibrate(m);
        const R = mul(m.R, T(this.ref.R));                        // rotation since the neutral pose, camera axes
        const f = [R[0][2], R[1][2], R[2][2]], u = [R[0][1], R[1][1], R[2][1]];
        const yawC = Math.atan2(f[0], f[2]), up = Math.atan2(f[1], Math.hypot(f[0], f[2])), tilt = Math.atan2(u[0], u[1]);
        const s = this.mirror ? 1 : -1;
        const b = (k) => shapes[k] || 0;
        const ear = (side) => 15 - 45 * (b(`browOuterUp${side}`) + 0.6 * b('browInnerUp')) + 110 * b(`browDown${side}`)
            + 30 * (b('mouthFrownLeft') + b('mouthFrownRight')) / 2;
        const [rightFrom, leftFrom] = this.mirror ? ['Left', 'Right'] : ['Right', 'Left'];
        const raw = {
            pitch: clamp(-deg(up), -LIMIT.pitch, LIMIT.pitch),
            roll: clamp(s * deg(tilt), -LIMIT.roll, LIMIT.roll),
            yaw: clamp(-s * deg(yawC), -LIMIT.yaw, LIMIT.yaw),
            z: clamp(3 + 8 * (m.t[1] - this.ref.t[1]), ...LIMIT.z),
            earR: clamp(ear(rightFrom), ...LIMIT.ear),
            earL: clamp(ear(leftFrom), ...LIMIT.ear),
        };
        if (!this.out) this.out = raw;
        else for (const k in raw) this.out[k] += (1 - this.smooth) * (raw[k] - this.out[k]);
        return { ...this.out };
    }
}

/** Plan-unit pose -> a move frame and the viewer state; `last` (a previous result) is held if this one is out of reach. */
export function toFrame(p, last = null) {
    const head = pose(p.z / 1000, rad(p.roll), rad(p.pitch), rad(p.yaw));
    const joints = headJoints(head, 0);
    if (!feasible(joints) && last) return last;
    const antennas = [-rad(p.earR), rad(p.earL)];
    return { move: { head, antennas, body_yaw: 0, check_collision: false },
             view: { head_pose: head.flat(), head_joints: joints, antennas_position: antennas } };
}
