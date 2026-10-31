// Gilgamesh 64 - shared declarations
#pragma once

#include <libdragon.h>
#include <t3d/t3d.h>
#include <t3d/t3dmath.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>

// Game logic runs in meters; the renderer uses decimeters (RS units per meter)
// because tiny3d's fixed-point pipeline prefers larger coordinates.
#define RS        10.0f
#define FB_COUNT  3
#define PI_F      3.14159265f
#define TAU_F     6.28318531f

#define FONT_TEXT  1
#define FONT_TITLE 2

static inline float clampf(float v, float a, float b) { return v < a ? a : (v > b ? b : v); }
static inline float lerpf(float a, float b, float t) { return a + (b - a) * t; }
static inline float dampf(float a, float b, float l, float dt) { return lerpf(a, b, 1.0f - expf(-l * dt)); }
static inline float smoothf(float t) { return t * t * (3.0f - 2.0f * t); }
static inline float minf(float a, float b) { return a < b ? a : b; }
static inline float maxf(float a, float b) { return a > b ? a : b; }
float ang_diff(float a, float b);
float turn_toward(float a, float b, float step);
float frand(void);            // gameplay randomness, 0..1
float srand01(void);          // deterministic level-generation randomness, 0..1

typedef struct { float x, y, z; } V3;
static inline V3 v3(float x, float y, float z) { return (V3){x, y, z}; }
static inline float dist2d(V3 a, V3 b) { float dx = a.x - b.x, dz = a.z - b.z; return sqrtf(dx * dx + dz * dz); }

// ------------------------------------------------------------------ colors
#define RGBA(r, g, b, a) ((uint32_t)(((r) << 24) | ((g) << 16) | ((b) << 8) | (a)))
#define RGB(hex)         ((uint32_t)(((hex) << 8) | 0xFF))
uint32_t col_lerp(uint32_t a, uint32_t b, float t);
uint32_t col_shade(uint32_t c, float k);

// ------------------------------------------------------------------ mesh.c
// A procedural mesh builder: primitives are emitted with a current transform,
// then baked into tiny3d vertex batches recorded inside an rspq block.
typedef struct MeshB MeshB;
typedef struct {
    rspq_block_t *block;
    T3DVertPacked *verts;
    int tris;
} Mesh;

MeshB *mb_new(float unit);
void mb_free(MeshB *mb);
int  mb_tris(MeshB *mb);
void mb_identity(MeshB *mb);
void mb_xform(MeshB *mb, float px, float py, float pz, float rx, float ry, float rz, float sx, float sy, float sz);
int  mb_vert(MeshB *mb, float x, float y, float z, float nx, float ny, float nz, uint32_t rgba);
void mb_tri(MeshB *mb, int a, int b, int c);
void mb_quad_raw(MeshB *mb, const float p[4][3], uint32_t col);
void mb_box(MeshB *mb, float cx, float cy, float cz, float w, float h, float d, uint32_t col);
void mb_cyl(MeshB *mb, float cx, float cy, float cz, float rTop, float rBot, float h, int seg, uint32_t col, bool caps);
void mb_disc(MeshB *mb, float cx, float cy, float cz, float r, int seg, uint32_t col);
Mesh mb_bake(MeshB *mb);
void mesh_draw(const Mesh *m);

// ------------------------------------------------------------------ math helpers for draw matrices
typedef T3DMat4 M4;
void m4_trs(M4 *m, float px, float py, float pz, float rx, float ry, float rz, float s);
void m4_mul(M4 *out, const M4 *a, const M4 *b);
void m4_scale_cols(M4 *m, float s);
T3DMat4FP *mat_alloc(void);          // per-frame uncached matrix pool
void mat_frame_begin(int frameIdx);
void draw_mesh_at(const Mesh *mesh, const M4 *m);

// ------------------------------------------------------------------ world.c
#define BX0 (-30.0f)
#define BX1 (30.0f)
#define BZ0 (-76.0f)
#define BZ1 (44.0f)
#define GATE_Z (-22.0f)
#define ARENA_X 0.0f
#define ARENA_Z (-52.0f)
#define ARENA_R 17.0f

float height_at(float x, float z);
void  world_build(void);
void  world_draw(V3 cam, float camYaw, float fogFar);
void  world_draw_sky(V3 cam);
bool  resolve(V3 *pos, float r);
void  camera_push(V3 *c);           // keep the camera out of trunks and canopies   // push out of colliders, returns true if moved
void  world_open_doors(float k);   // 0 closed .. 1 open
void  world_set_door_collider(bool on);
void  world_update(float dt);

typedef struct { V3 pos; const char *who; const char *lines[3]; int nlines; } Stele;
extern Stele g_steles[2];

// ------------------------------------------------------------------ synth.c
void synth_init(void);
void synth_update(void);
void synth_set_track(int track);   // 0 none, 1 field, 2 boss
int  synth_track(void);
void synth_toggle_mute(void);
enum {
    SFX_SWING, SFX_SWING3, SFX_HIT, SFX_KILL, SFX_HURT, SFX_BLOCK, SFX_ROLL, SFX_HEART, SFX_LAPIS, SFX_SEAL,
    SFX_SECRET, SFX_DOOR, SFX_ROAR, SFX_SLAM, SFX_FIRE, SFX_REFLECT, SFX_TEXT, SFX_SELECT, SFX_DIE, SFX_BOSSHIT,
    SFX_WIN, SFX_THUD,
};
void sfx(int id);

// ------------------------------------------------------------------ ui.c
typedef struct { const char *who; const char *text; } Line;
void ui_init(void);
void ui_say(const Line *lines, int n, void (*onDone)(void));
bool ui_dialog_active(void);
void ui_dialog_update(float dt, bool advance);
void ui_area_card(const char *top, const char *name);
void ui_draw_hud(void);
void ui_draw_dialog(void);
void ui_draw_overlays(void);
void ui_update(float dt);

// ------------------------------------------------------------------ game state (main.c / actors.c)
enum { ST_TITLE, ST_PLAY, ST_PAUSE, ST_GAMEOVER, ST_ENDING };
typedef struct {
    int state;
    float time, playTime;
    int hp, maxHp, lapis;
    bool seals[3];          // dawn, noon, dusk
    bool gateOpen, bossStarted, bossDead, duskSpawned;
    V3 checkpoint;
    float shake, hitstop;
    int region;             // 0 steppe, 1 forest
    const char *aLabel;
    float bossHpFrac;
    bool bossBar;
} Game;
extern Game G;

typedef struct {
    float mx, my;           // move stick, -1..1
    float cx;               // camera turn
    bool a, b, z, r, start; // pressed this frame
    bool zHeld, rHeld, cl, cr;
} Input;
extern Input IN;

// actors.c
void actors_init(void);
void actors_update(float dt);
void actors_draw(void);
void actors_draw_shadows(void);
void actors_title_anim(float dt);
void actors_continue(void);
V3   player_pos(void);
float player_facing(void);
bool player_dead(void);
void player_set(V3 p, float facing);
void *lock_target(void);
V3   lock_target_pos(float *camH, bool *isBoss);
void boss_begin_fight(void);
bool boss_active(void);
V3   boss_pos(void);
void boss_set_state_roar(void);
void particles_burst(V3 p, int n, int colorIdx, float speed, float up, float y);

// story hooks called by actors (main.c)
extern float g_camYaw;      // direction the camera looks, forward = (sin, cos)
void on_seal_collected(int id);
void on_ring_cleared(void);
void on_boss_defeated(void);
const char *context_label(V3 p);
bool context_run(V3 p);
void enkidu_face_player(void);

// camera & cutscenes (main.c)
void cutscene_gate(void);
void cutscene_boss(void);
void game_over(void);
void game_ending(void);
void hud_dirty(void);
