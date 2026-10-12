import { feasible, headJoints } from './StewartIK.js';

/**
 * Plays Reachy Mini moves (SDK recorded-move format) on the 3D robot.
 * Each frame: head_pose (row-major 4x4) + antennas + body_yaw -> motor angles via the browser IK -> RobotManager.
 */
export class Player {
    constructor(robotManager, onTick) {
        this.robot = robotManager; this.onTick = onTick;
        this.frames = null; this.times = null; this.t0 = 0; this.pausedAt = null;
        this.speed = 1; this.loop = true;
        this.live = null;   // a joint state set from outside (the listening behaviour) overrides the move
        requestAnimationFrame(this.tick.bind(this));
    }

    /** Precompute the viewer state for every frame of a move (IK runs once, not per render). With `from` (a viewer
     * state, e.g. `this.shown`), the first `blendS` seconds ease from that pose into the move, so the robot glides
     * from listening into its answer instead of jumping. */
    load(move, from = null, blendS = 0.4) {
        const fr = move.set_target_data, t = move.time;
        const t0 = t[0];
        this.times = t.map((x) => x - t0);
        this.frames = fr.map((f) => ({
            head_pose: f.head.flat(),
            head_joints: headJoints(f.head, f.body_yaw ?? 0),
            antennas_position: f.antennas,
        }));
        if (from?.head_pose && from.head_joints) {
            for (let i = 0; i < this.frames.length && this.times[i] < blendS; i++) {
                const w = ease(this.times[i] / blendS), f = fr[i];
                const H = blendPose(from.head_pose, f.head.flat(), w);
                const yaw = (1 - w) * from.head_joints[0] + w * (f.body_yaw ?? 0);   // head_joints[0] is the body yaw
                const ant = f.antennas.map((x, k) => (1 - w) * (from.antennas_position?.[k] ?? x) + w * x);
                const joints = headJoints(H, yaw);
                if (feasible(joints)) this.frames[i] = { head_pose: H.flat(), head_joints: joints, antennas_position: ant };
            }
        }
        // an unreachable frame (NaN motor angle, or a motor past its range) holds the previous valid one
        for (let i = 1; i < this.frames.length; i++) {
            if (!feasible(this.frames[i].head_joints)) this.frames[i] = this.frames[i - 1];
        }
        this.t0 = performance.now(); this.pausedAt = null;
        return this.duration();
    }

    duration() { return this.times ? this.times[this.times.length - 1] : 0; }
    playing() { return this.frames && this.pausedAt === null; }
    toggle() {
        if (!this.frames) return;
        if (this.pausedAt === null) this.pausedAt = this.elapsed();
        else { this.t0 = performance.now() - this.pausedAt * 1000 / this.speed; this.pausedAt = null; }
    }
    restart() { this.t0 = performance.now(); this.pausedAt = null; }
    setSpeed(s) { const e = this.elapsed(); this.speed = s; this.t0 = performance.now() - e * 1000 / s; }
    seek(frac) {
        const e = frac * this.duration();
        if (this.pausedAt !== null) this.pausedAt = e; else this.t0 = performance.now() - e * 1000 / this.speed;
    }
    elapsed() {
        if (!this.frames) return 0;
        if (this.pausedAt !== null) return this.pausedAt;
        let e = (performance.now() - this.t0) / 1000 * this.speed;
        const d = this.duration();
        if (e > d) { if (this.loop) { e = e % (d + 0.6); if (e > d) e = d; } else e = d; }   // brief rest between loops
        return e;
    }

    tick() {
        if (this.live) { this.robot.updateJoints(this.live); this.shown = this.live; }
        else if (this.frames) {
            const e = this.elapsed();
            let i = this.times.findIndex((x) => x > e); if (i < 0) i = this.times.length; i = Math.max(0, i - 1);
            this.robot.updateJoints(this.frames[i]); this.shown = this.frames[i];
            this.onTick?.(e, this.duration());
        }
        requestAnimationFrame(this.tick.bind(this));
    }
}

const ease = (x) => (x <= 0 ? 0 : x >= 1 ? 1 : 0.5 - 0.5 * Math.cos(Math.PI * x));

/** Row-major 4x4 (16 numbers) -> quaternion [w, x, y, z] of its rotation. */
function quat(m) {
    const [a, b, c, , d, e, f, , g, h, k] = m, tr = a + e + k;
    let q;
    if (tr > 0) { const s = 2 * Math.sqrt(tr + 1); q = [s / 4, (h - f) / s, (c - g) / s, (d - b) / s]; }
    else if (a > e && a > k) { const s = 2 * Math.sqrt(1 + a - e - k); q = [(h - f) / s, s / 4, (b + d) / s, (c + g) / s]; }
    else if (e > k) { const s = 2 * Math.sqrt(1 + e - a - k); q = [(c - g) / s, (b + d) / s, s / 4, (f + h) / s]; }
    else { const s = 2 * Math.sqrt(1 + k - a - e); q = [(d - b) / s, (c + g) / s, (f + h) / s, s / 4]; }
    return q;
}

/** Blend two row-major head poses: slerp the rotation, lerp the position. Returns a 4x4 (array of rows). */
function blendPose(A, B, w) {
    const qa = quat(A); let qb = quat(B);
    let dot = qa.reduce((s, x, i) => s + x * qb[i], 0);
    if (dot < 0) { qb = qb.map((x) => -x); dot = -dot; }
    let q;
    if (dot > 0.9995) q = qa.map((x, i) => x + w * (qb[i] - x));
    else {
        const th = Math.acos(dot), s0 = Math.sin((1 - w) * th) / Math.sin(th), s1 = Math.sin(w * th) / Math.sin(th);
        q = qa.map((x, i) => s0 * x + s1 * qb[i]);
    }
    const n = Math.hypot(...q), [qw, qx, qy, qz] = q.map((x) => x / n);
    const p = [3, 7, 11].map((j) => (1 - w) * A[j] + w * B[j]);
    return [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw), p[0]],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw), p[1]],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy), p[2]],
        [0, 0, 0, 1],
    ];
}
