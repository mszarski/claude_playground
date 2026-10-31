// A tiny software synthesizer: sound effects plus two sequenced music loops
// (a Hijaz-flavoured lyre tune for the steppe and a war drum loop for Humbaba).
#include "game.h"

#define RATE 22050
enum { W_TRI, W_SQR, W_SAW, W_SINE, W_NOISE, W_HPNOISE };
typedef struct { bool on; int w; float ph, f, f1, t, dur, vol, lp, cut, cut1, music; } Voice;
#define NV 18
static Voice V[NV];
static float s_master = 0.55f, s_noise = 1;
static uint32_t s_rng = 12345;
static bool s_muted;

static float noise1(void) { s_rng = s_rng * 1664525u + 1013904223u; return ((s_rng >> 9) / 4194304.0f) - 1.0f; }

static void voice(int w, float f, float dur, float vol, float slideTo, float cutoff, float cut1, float delay, bool music) {
    int slot = -1;
    for (int i = 0; i < NV; i++) if (!V[i].on) { slot = i; break; }
    if (slot < 0) { float best = 0; for (int i = 0; i < NV; i++) if (V[i].t / V[i].dur > best) { best = V[i].t / V[i].dur; slot = i; } }
    if (slot < 0) slot = 0;
    V[slot] = (Voice){true, w, 0, f, slideTo > 0 ? slideTo : f, -delay, dur, vol, 0, cutoff, cut1 > 0 ? cut1 : cutoff, music ? 1 : 0};
}
static inline float st(float base, float semis) { return base * powf(2.0f, semis / 12.0f); }

// ------------------------------------------------------------------ music
static const int SC[7] = {0, 1, 4, 5, 7, 8, 10};
#define R (-99)
static float deg_freq(int root, int d) { int o = (int)floorf(d / 7.0f), i = ((d % 7) + 7) % 7; int m = root + 12 * o + SC[i]; return 440.0f * powf(2.0f, (m - 69) / 12.0f); }
static const int MEL_FIELD[64] = {4, R, 5, 4, 3, R, 2, R, 1, R, 2, 3, 4, R, R, R, 7, R, 6, 5, 6, R, 5, 4, 3, 4, 3, 2, 2, R, R, R,
                                  0, R, 1, R, 2, R, 1, 0, 1, R, R, 2, 1, 0, R, R, 4, R, 4, 5, 4, R, 3, 2, 1, R, 2, 1, 0, R, R, R};
static const int BASS_FIELD[8] = {0, 0, 3, 4, 0, 0, 1, 0};
static const int MEL_BOSS[32] = {0, R, 0, 1, 0, R, 4, 3, 1, R, 0, R, -1, R, 0, R, 7, R, 7, 8, 7, R, 4, R, 5, 4, 3, 1, 0, R, R, R};
static const int BASS_BOSS[4] = {0, 0, 1, 0};
static int s_track, s_step;
static float s_stepT, s_droneT;

static void play_step(void) {
    int i = s_step;
    if (s_track == 1) {
        int n = MEL_FIELD[i % 64];
        if (n != R) { voice(W_TRI, deg_freq(62, n), 0.7f, 0.22f, 0, 0, 0, 0, true); voice(W_SINE, deg_freq(62, n) * 2.003f, 0.5f, 0.07f, 0, 0, 0, 0, true); }
        int b8 = i % 8;
        if (b8 == 0 || b8 == 4) voice(W_SINE, 120, 0.3f, 0.5f, 45, 0, 0, 0, true);
        if (b8 == 1 || b8 == 3 || b8 == 6) voice(W_HPNOISE, 0, 0.06f, 0.12f, 0, 2500, 0, 0, true);
        if (b8 == 0) voice(W_TRI, deg_freq(38, BASS_FIELD[(i / 8) % 8]), 1.4f, 0.25f, 0, 0, 0, 0, true);
    } else if (s_track == 2) {
        int n = MEL_BOSS[i % 32];
        if (n != R) voice(W_SAW, deg_freq(50, n), 0.16f, 0.1f, 0, 2400, 500, 0, true);
        int b8 = i % 8;
        if (b8 == 0 || b8 == 3 || b8 == 6) voice(W_SINE, 120, 0.3f, 0.6f, 45, 0, 0, 0, true);
        if (i & 1) voice(W_HPNOISE, 0, 0.06f, 0.1f, 0, 2500, 0, 0, true);
        else voice(W_SAW, deg_freq(38, BASS_BOSS[(i / 2) % 4]), 0.14f, 0.16f, 0, 1200, 300, 0, true);
    }
    s_step++;
}
void synth_set_track(int t) {
    if (t == s_track) return;
    s_track = t; s_step = 0; s_stepT = 0;
    for (int i = 0; i < NV; i++) if (V[i].music) V[i].on = false;
}
int synth_track(void) { return s_track; }
void synth_toggle_mute(void) { s_muted = !s_muted; }

// ------------------------------------------------------------------ sfx
void sfx(int id) {
    switch (id) {
    case SFX_SWING:  voice(W_NOISE, 0, 0.16f, 0.25f, 0, 2600, 500, 0, false); break;
    case SFX_SWING3: voice(W_NOISE, 0, 0.18f, 0.28f, 0, 1800, 400, 0, false); break;
    case SFX_HIT:    voice(W_SQR, 180, 0.09f, 0.16f, 0, 0, 0, 0, false); voice(W_NOISE, 0, 0.08f, 0.25f, 0, 1200, 0, 0, false); break;
    case SFX_KILL:   voice(W_SQR, 520, 0.25f, 0.12f, 90, 0, 0, 0, false); voice(W_NOISE, 0, 0.3f, 0.2f, 0, 600, 100, 0, false); break;
    case SFX_HURT:   voice(W_SAW, 320, 0.22f, 0.18f, 120, 3000, 0, 0, false); break;
    case SFX_BLOCK:  voice(W_SQR, 1400, 0.06f, 0.1f, 0, 0, 0, 0, false); voice(W_TRI, 2100, 0.08f, 0.1f, 0, 0, 0, 0.02f, false); break;
    case SFX_ROLL:   voice(W_NOISE, 0, 0.22f, 0.12f, 0, 500, 200, 0, false); break;
    case SFX_HEART:  for (int i = 0; i < 3; i++) voice(W_TRI, st(880, (int[]){0, 4, 7}[i]), 0.12f, 0.15f, 0, 0, 0, i * 0.05f, false); break;
    case SFX_LAPIS:  voice(W_SQR, 1568, 0.08f, 0.07f, 0, 0, 0, 0, false); voice(W_SQR, 2093, 0.12f, 0.07f, 0, 0, 0, 0.06f, false); break;
    case SFX_SEAL:
        for (int i = 0; i < 6; i++) voice(W_TRI, st(523, (int[]){0, 4, 7, 12, 7, 12}[i]), i == 5 ? 0.7f : 0.14f, 0.2f, 0, 0, 0, i * 0.12f, false);
        for (int i = 0; i < 3; i++) voice(W_SQR, st(262, (int[]){0, 4, 7}[i]), 0.7f, 0.04f, 0, 0, 0, 0.6f, false);
        break;
    case SFX_SECRET: for (int i = 0; i < 8; i++) voice(W_TRI, st(392, (int[]){0, 1, 4, 7, 8, 11, 12, 16}[i]), 0.18f, 0.13f, 0, 0, 0, i * 0.08f, false); break;
    case SFX_DOOR:   voice(W_NOISE, 0, 2.6f, 0.35f, 0, 180, 60, 0, false); voice(W_SAW, 55, 2.4f, 0.08f, 40, 600, 0, 0, false); break;
    case SFX_ROAR:   voice(W_SAW, 140, 1.3f, 0.22f, 45, 2000, 400, 0, false); voice(W_SQR, 150, 1.3f, 0.08f, 50, 0, 0, 0, false); voice(W_NOISE, 0, 1.3f, 0.3f, 0, 700, 150, 0, false); break;
    case SFX_SLAM:   voice(W_SINE, 80, 0.5f, 0.45f, 30, 0, 0, 0, false); voice(W_NOISE, 0, 0.5f, 0.4f, 0, 300, 60, 0, false); break;
    case SFX_FIRE:   voice(W_NOISE, 0, 0.35f, 0.22f, 0, 900, 300, 0, false); break;
    case SFX_REFLECT: voice(W_SQR, 900, 0.15f, 0.12f, 1800, 0, 0, 0, false); break;
    case SFX_TEXT:   voice(W_SQR, 640 + (s_rng & 63), 0.025f, 0.025f, 0, 0, 0, 0, false); break;
    case SFX_SELECT: voice(W_SQR, 880, 0.06f, 0.07f, 0, 0, 0, 0, false); break;
    case SFX_DIE:    for (int i = 0; i < 4; i++) voice(W_TRI, st(392, (int[]){12, 8, 5, 0}[i]), 0.3f, 0.15f, 0, 0, 0, i * 0.22f, false); break;
    case SFX_BOSSHIT: voice(W_SQR, 120, 0.12f, 0.2f, 0, 0, 0, 0, false); voice(W_NOISE, 0, 0.12f, 0.3f, 0, 900, 0, 0, false); break;
    case SFX_WIN:    for (int i = 0; i < 7; i++) voice(W_TRI, st(392, (int[]){0, 4, 7, 12, 16, 19, 24}[i]), i == 6 ? 1.2f : 0.16f, 0.18f, 0, 0, 0, i * 0.13f, false); break;
    case SFX_THUD:   voice(W_SINE, 60, 0.3f, 0.4f, 35, 0, 0, 0, false); voice(W_NOISE, 0, 0.25f, 0.3f, 0, 200, 0, 0, false); break;
    }
}

// ------------------------------------------------------------------ mixing
static float s_droneLp[2], s_dronePh[2];
static inline float render_voice(Voice *v, float dt) {
    if (v->t < 0) { v->t += dt; return 0; }
    float k = v->t / v->dur;
    if (k >= 1) { v->on = false; return 0; }
    float env = v->t < 0.006f ? v->t / 0.006f : expf(-5.0f * k);
    float f = v->f1 != v->f ? v->f * powf(v->f1 / v->f, k) : v->f;
    float s;
    switch (v->w) {
    case W_TRI: v->ph += f * dt; v->ph -= floorf(v->ph); s = 4 * fabsf(v->ph - 0.5f) - 1; break;
    case W_SQR: v->ph += f * dt; v->ph -= floorf(v->ph); s = v->ph < 0.5f ? 0.6f : -0.6f; break;
    case W_SINE: v->ph += f * dt; v->ph -= floorf(v->ph); s = sinf(v->ph * TAU_F); break;
    case W_SAW: {
        v->ph += f * dt; v->ph -= floorf(v->ph); float raw = 2 * v->ph - 1;
        float cut = v->cut > 0 ? v->cut * powf(v->cut1 / v->cut, k) : 4000;
        v->lp += (raw - v->lp) * minf(1, cut * TAU_F * dt); s = v->lp; break;
    }
    case W_NOISE: {
        float cut = v->cut * powf(v->cut1 / v->cut, k);
        v->lp += (noise1() - v->lp) * minf(1, cut * TAU_F * dt); s = v->lp * 2.5f; break;
    }
    default: { float n = noise1(); v->lp += (n - v->lp) * minf(1, v->cut * TAU_F * dt); s = (n - v->lp) * 0.8f; break; }
    }
    v->t += dt;
    return s * env * v->vol;
}
void synth_init(void) {
#ifndef DEBUG_NOAUDIO
    audio_init(RATE, 4);
#endif
}
void synth_update(void) {
#ifdef DEBUG_NOAUDIO
    return;
#endif
    const float dt = 1.0f / RATE;
    while (audio_can_write()) {
        short *buf = audio_write_begin();
        int n = audio_get_buffer_length();
        float stepLen = s_track == 2 ? 60.0f / 150 / 2 : 60.0f / 92 / 2;
        for (int i = 0; i < n; i++) {
            if (s_track) { s_stepT += dt; if (s_stepT >= stepLen) { s_stepT -= stepLen; play_step(); } }
            float mix = 0;
            for (int k = 0; k < NV; k++) if (V[k].on) mix += render_voice(&V[k], dt);
            if (s_track) {   // low droning strings under the music
                static const float droneF[2][2] = {{73.4f, 110.0f}, {73.4f, 73.4f}};
                for (int d = 0; d < 2; d++) {
                    s_dronePh[d] += droneF[s_track - 1][d] * dt; s_dronePh[d] -= floorf(s_dronePh[d]);
                    s_droneLp[d] += ((2 * s_dronePh[d] - 1) - s_droneLp[d]) * 0.09f;
                    mix += s_droneLp[d] * 0.05f;
                }
            }
            mix *= s_muted ? 0 : s_master;
            int v = (int)(clampf(mix, -1, 1) * 32000);
            buf[i * 2] = buf[i * 2 + 1] = (short)v;
        }
        audio_write_end();
    }
    (void)s_noise; (void)s_droneT;
}
