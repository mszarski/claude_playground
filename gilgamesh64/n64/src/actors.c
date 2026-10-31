// Characters, enemies, Humbaba, pickups, pots, projectiles and particles.
#include "game.h"

// ===================================================================== models
// A model is a small hierarchy of parts. Each part owns a mesh built in
// centimeters, a rest offset (meters) from its parent and per-instance pose.
#define MAX_PARTS 18
typedef struct { int parent; float px, py, pz; Mesh mesh; } PartDef;
typedef struct { PartDef p[MAX_PARTS]; int n; float scale; } ModelDef;
typedef struct { float rx[MAX_PARTS], ry[MAX_PARTS], rz[MAX_PARTS], ox[MAX_PARTS], oy[MAX_PARTS], oz[MAX_PARTS], sy[MAX_PARTS]; } Pose;

static MeshB *s_pb[MAX_PARTS];
static int part_new(ModelDef *md, int parent, float x, float y, float z) {
    int i = md->n++;
    md->p[i] = (PartDef){parent, x, y, z, {0}};
    s_pb[i] = mb_new(100.0f);
    return i;
}
static MeshB *pb(int i) { return s_pb[i]; }
static void model_bake(ModelDef *md) {
    for (int i = 0; i < md->n; i++) { md->p[i].mesh = mb_bake(s_pb[i]); mb_free(s_pb[i]); s_pb[i] = NULL; }
    if (md->scale == 0) md->scale = 1;
}
static void pose_reset(Pose *p) {
    for (int i = 0; i < MAX_PARTS; i++) { p->rx[i] = p->ry[i] = p->rz[i] = p->ox[i] = p->oy[i] = p->oz[i] = 0; p->sy[i] = 1; }
}
// optional per-part mesh override (e.g. Humbaba's eye colours)
static const Mesh *s_override[MAX_PARTS];

static void model_draw(const ModelDef *md, const Pose *ps, V3 pos, float facing, float rootRx, float rootRz, float flash) {
    M4 world[MAX_PARTS], root, tilt, L, D;
    m4_trs(&root, pos.x * RS, pos.y * RS, pos.z * RS, 0, facing, 0, RS * md->scale);
    if (rootRx != 0 || rootRz != 0) { m4_trs(&tilt, 0, 0, 0, rootRx, 0, rootRz, 1); m4_mul(&root, &root, &tilt); }
    if (flash > 0) rdpq_set_prim_color(RGBA32(255, 60, 30, (uint8_t)(flash * 200)));
    for (int i = 0; i < md->n; i++) {
        const PartDef *p = &md->p[i];
        m4_trs(&L, p->px + ps->ox[i], p->py + ps->oy[i], p->pz + ps->oz[i], ps->rx[i], ps->ry[i], ps->rz[i], 1);
        if (ps->sy[i] != 1) for (int r = 0; r < 3; r++) L.m[1][r] *= ps->sy[i];
        m4_mul(&world[i], p->parent >= 0 ? &world[p->parent] : &root, &L);
        const Mesh *mesh = s_override[i] ? s_override[i] : &p->mesh;
        if (!mesh->block) continue;
        D = world[i]; m4_scale_cols(&D, 0.01f);
        draw_mesh_at(mesh, &D);
    }
    if (flash > 0) rdpq_set_prim_color(RGBA32(0, 0, 0, 0));
}

// ------------------------------------------------------------------ humanoids
enum { H_HIPS, H_LEGL, H_LEGR, H_TORSO, H_HEAD, H_ARML, H_ARMR, H_HANDR, H_HANDL, H_COUNT };
typedef struct { uint32_t skin, hair, kilt, belt, boot, sash; } HumanCols;
static void build_humanoid(ModelDef *md, HumanCols c) {
    md->n = 0;
    part_new(md, -1, 0, 0.95f, 0);                    // hips
    part_new(md, H_HIPS, 0.16f, 0, 0);                // legL (character's left is +x)
    part_new(md, H_HIPS, -0.16f, 0, 0);               // legR
    part_new(md, H_HIPS, 0, 0.12f, 0);                // torso
    part_new(md, H_TORSO, 0, 0.85f, 0);               // head
    part_new(md, H_TORSO, 0.39f, 0.56f, 0);           // armL
    part_new(md, H_TORSO, -0.39f, 0.56f, 0);          // armR
    part_new(md, H_ARMR, 0, -0.62f, 0);               // handR
    part_new(md, H_ARML, 0, -0.62f, 0);               // handL
    for (int l = H_LEGL; l <= H_LEGR; l++) { mb_box(pb(l), 0, -41, 0, 20, 82, 22, c.skin); mb_box(pb(l), 0, -86, 5, 24, 14, 34, c.boot); }
    mb_box(pb(H_HIPS), 0, -12, 0, 64, 50, 44, c.kilt);
    mb_box(pb(H_HIPS), 0, 12, 0, 66, 10, 46, c.belt);
    mb_box(pb(H_TORSO), 0, 32, 0, 60, 62, 36, c.skin);
    if (c.sash) { mb_xform(pb(H_TORSO), 0, 32, 0, 0, 0, 0.7f, 1, 1, 1); mb_box(pb(H_TORSO), 0, 0, 0, 13, 82, 38, c.sash); mb_identity(pb(H_TORSO)); }
    MeshB *h = pb(H_HEAD);
    mb_box(h, 0, 0, 0, 38, 40, 38, c.skin);
    mb_box(h, 0, 17, -2, 42, 18, 42, c.hair);
    mb_box(h, 0, -5, -17, 42, 46, 14, c.hair);
    mb_box(h, 0, -29, 15, 36, 36, 14, c.hair);
    mb_box(h, 9, 4, 19.5f, 6, 6, 2, RGB(0x161010));
    mb_box(h, -9, 4, 19.5f, 6, 6, 2, RGB(0x161010));
    for (int a = H_ARML; a <= H_ARMR; a++) { mb_box(pb(a), 0, -30, 0, 17, 60, 18, c.skin); mb_box(pb(a), 0, -45, 0, 20, 10, 20, c.belt); }
}

static ModelDef M_GIL, M_ENK, M_SCORP, M_LION, M_HUMB;
static Mesh MESH_ARC, MESH_SHADOW, MESH_RETICLE, MESH_HEART, MESH_LAPIS, MESH_SEAL, MESH_BEAM, MESH_POT, MESH_FIRE, MESH_FIRE_R, MESH_SHOCK;
static Mesh MESH_PART[8];
static Mesh MESH_EYES[3];
static const uint32_t PART_COLORS[8] = {RGB(0xFFF6C0), RGB(0x7A321A), RGB(0xC89A52), RGB(0x8A5AFF), RGB(0xB5562B), RGB(0xFF7A20), RGB(0x2F5A2A), RGB(0xFF5040)};
enum { PC_SPARK, PC_SHELL, PC_FUR, PC_SPIRIT, PC_CLAY, PC_FIRE, PC_LEAF, PC_BLOOD };

static void build_gilgamesh(void) {
    build_humanoid(&M_GIL, (HumanCols){RGB(0xC68A5A), RGB(0x15100C), RGB(0xEEE0C0), RGB(0xE2B54A), RGB(0x5A3A1A), RGB(0x2A4DBF)});
    MeshB *h = pb(H_HEAD);
    mb_cyl(h, 0, 30, 0, 25, 23, 14, 8, RGB(0xF2C14E), false);   // crown
    for (int i = 0; i < 8; i++) { float a = i / 8.0f * TAU_F; mb_box(h, cosf(a) * 24, 42, sinf(a) * 24, 6, 12, 6, RGB(0xF2C14E)); }
    for (int i = 0; i < 3; i++) mb_box(h, 0, -20 - i * 8, 22.5f, 38, 3, 2, RGB(0x2A2018));   // beard ringlets
    MeshB *s = pb(H_HANDR);   // bronze sword, blade along +z
    mb_box(s, 0, 0, 2, 7, 7, 24, RGB(0x5A3A1A));
    mb_box(s, 0, 0, 16, 30, 6, 7, RGB(0xE2B54A));
    mb_box(s, 0, 0, 66, 5, 14, 95, RGB(0xD9A14A));
    mb_box(s, 0, 0, 118, 4, 8, 15, RGB(0xFFD98A));
    MeshB *sh = pb(H_HANDL);  // round lapis shield
    mb_xform(sh, 10, 5, 5, 0, 0, PI_F / 2, 1, 1, 1);
    mb_cyl(sh, 0, 0, 0, 42, 42, 8, 10, RGB(0x2A4DBF), true);
    mb_cyl(sh, 0, 5, 0, 10, 12, 4, 6, RGB(0xE2B54A), true);
    mb_identity(sh);
    model_bake(&M_GIL);
}
static void build_enkidu(void) {
    build_humanoid(&M_ENK, (HumanCols){RGB(0x9A6440), RGB(0x3B2614), RGB(0xB8925A), RGB(0x5A3A1A), RGB(0x6B4A2A), 0});
    mb_box(pb(H_HEAD), 0, 5, -8, 62, 55, 50, RGB(0x3B2614));
    mb_box(pb(H_TORSO), 0, 50, -2, 66, 30, 40, RGB(0x4A2E18));
    for (int i = 0; i < 5; i++) mb_box(pb(H_HIPS), -26 + i * 13, -32, 0, 10, 12, 46, RGB(0x7A5A30));
    mb_xform(pb(H_TORSO), 0, 40, -25, 0, 0, 0.6f, 1, 1, 1);
    mb_box(pb(H_TORSO), 0, 0, 0, 6, 110, 6, RGB(0x6A4A2A));
    mb_box(pb(H_TORSO), 14, 45, 0, 32, 22, 4, RGB(0xC98D3A));
    mb_identity(pb(H_TORSO));
    M_ENK.scale = 1.06f;
    model_bake(&M_ENK);
}

// scorpion-man
enum { S_BODY, S_LEG0, S_TAIL0 = 7, S_TAIL3 = 10, S_TORSO, S_ARMR, S_COUNT };
static void build_scorpion(void) {
    ModelDef *md = &M_SCORP; md->n = 0;
    uint32_t shell = RGB(0x7A321A), shellD = RGB(0x42190C), skin = RGB(0x9A5A34), bronze = RGB(0xD09A3A);
    part_new(md, -1, 0, 0.55f, 0);
    mb_box(pb(S_BODY), 0, 0, -10, 90, 38, 110, shell);
    mb_box(pb(S_BODY), 0, 5, -75, 70, 30, 40, shellD);
    for (int s = 0; s < 2; s++) for (int i = 0; i < 3; i++) {
        float side = s ? 1 : -1;
        int l = part_new(md, S_BODY, side * 0.45f, -0.05f, 0.25f - i * 0.35f);
        mb_xform(pb(l), side * 30, -14, 0, 0, 0, side * -0.55f, 1, 1, 1);
        mb_box(pb(l), 0, 0, 0, 62, 10, 10, shellD);
    }
    int parent = S_BODY;
    for (int i = 0; i < 4; i++) {
        int t = part_new(md, parent, 0, i == 0 ? 0.1f : 0, i == 0 ? -0.95f : -0.42f);
        mb_box(pb(t), 0, 0, -21, 26 - i * 3, 24 - i * 3, 44, shell);
        if (i == 3) { mb_xform(pb(t), 0, 0, -55, -PI_F / 2, 0, 0, 1, 1, 1); mb_cyl(pb(t), 0, 0, 0, 0, 10, 38, 5, RGB(0xE8C860), false); mb_identity(pb(t)); }
        parent = t;
    }
    part_new(md, S_BODY, 0, 0.15f, 0.42f);    // torso
    MeshB *t = pb(S_TORSO);
    mb_box(t, 0, 30, 0, 50, 60, 32, skin);
    mb_box(t, 0, 78, 0, 32, 34, 32, skin);
    mb_box(t, 0, 95, 0, 38, 16, 38, bronze);
    mb_box(t, 0, 60, 16, 30, 20, 10, RGB(0x1A1008));
    mb_box(t, 8, 80, 17, 6, 5, 2, RGB(0xFFD040));
    mb_box(t, -8, 80, 17, 6, 5, 2, RGB(0xFFD040));
    mb_xform(t, 32, 50, 0, -0.4f, 0, 0, 1, 1, 1); mb_box(t, 0, -25, 0, 14, 50, 14, skin); mb_identity(t);
    part_new(md, S_TORSO, -0.32f, 0.5f, 0);   // spear arm
    MeshB *a = pb(S_ARMR);
    mb_box(a, 0, -25, 0, 14, 50, 14, skin);
    mb_box(a, 0, -50, 30, 5, 5, 190, RGB(0x6A4A2A));
    mb_xform(a, 0, -50, 135, PI_F / 2, 0, 0, 1, 1, 1); mb_cyl(a, 0, 0, 0, 0, 7, 30, 4, bronze, false); mb_identity(a);
    model_bake(md);
}
enum { L_BODY, L_HEAD, L_LEG0, L_TAIL = 6, L_COUNT };
static void build_lion(void) {
    ModelDef *md = &M_LION; md->n = 0;
    uint32_t fur = RGB(0xC89A52), mane = RGB(0x8A4F1E);
    part_new(md, -1, 0, 0.75f, 0);
    mb_box(pb(L_BODY), 0, 0, 0, 70, 62, 145, fur);
    part_new(md, L_BODY, 0, 0.28f, 0.8f);
    MeshB *h = pb(L_HEAD);
    mb_box(h, 0, 0, -8, 88, 88, 50, mane);
    mb_box(h, 0, -2, 20, 50, 50, 45, RGB(0xD4A860));
    mb_box(h, 0, -13, 46, 30, 20, 22, RGB(0xE2C08A));
    mb_box(h, 0, -4, 58, 10, 7, 4, RGB(0x1A0A05));
    mb_box(h, 12, 8, 43, 7, 6, 2, RGB(0x201000)); mb_box(h, -12, 8, 43, 7, 6, 2, RGB(0x201000));
    mb_box(h, 30, 42, -5, 14, 14, 10, mane); mb_box(h, -30, 42, -5, 14, 14, 10, mane);
    static const float lp[4][2] = {{1, 1}, {-1, 1}, {1, -1}, {-1, -1}};
    for (int i = 0; i < 4; i++) { int l = part_new(md, L_BODY, lp[i][0] * 0.24f, -0.25f, lp[i][1] * 0.52f); mb_box(pb(l), 0, -30, 0, 20, 65, 22, fur); mb_box(pb(l), 0, -62, 4, 24, 10, 28, RGB(0xA87A3A)); }
    part_new(md, L_BODY, 0, 0.2f, -0.72f);
    mb_box(pb(L_TAIL), 0, 0, -40, 7, 7, 80, fur); mb_box(pb(L_TAIL), 0, 0, -85, 16, 16, 20, mane);
    model_bake(md);
}
enum { B_HIPS, B_LEGL, B_LEGR, B_TORSO, B_HEAD, B_MOUTH, B_EYES, B_ARMP, B_ARMN, B_COUNT };
static void build_humbaba(void) {
    ModelDef *md = &M_HUMB; md->n = 0;
    uint32_t hide = RGB(0x5B4A2A), bark = RGB(0x6B5532), face = RGB(0xA0784A), coil = RGB(0x7A3424), leaf = RGB(0x2F5A2A), bone = RGB(0xE8DCB0);
    part_new(md, -1, 0, 2.2f, 0);
    part_new(md, B_HIPS, 0.8f, 0, 0); part_new(md, B_HIPS, -0.8f, 0, 0);
    for (int l = B_LEGL; l <= B_LEGR; l++) { mb_box(pb(l), 0, -110, 0, 75, 220, 85, hide); mb_box(pb(l), 0, -205, 15, 95, 35, 125, RGB(0x2E2414)); }
    part_new(md, B_HIPS, 0, 0.2f, 0);
    MeshB *t = pb(B_TORSO);
    mb_box(t, 0, 100, 0, 260, 210, 160, bark);
    mb_box(t, 0, 80, 82, 180, 120, 20, RGB(0x8A7048));
    mb_box(t, 0, 5, 0, 290, 40, 190, RGB(0x3A2A14));
    for (int i = 0; i < 12; i++) { float a = i / 12.0f * TAU_F; mb_xform(t, cosf(a) * 125, 205, sinf(a) * 90, 0, -a, 0.3f, 1, 1, 1); mb_box(t, 0, 0, 0, 70, 28, 100, leaf); }
    mb_identity(t);
    part_new(md, B_TORSO, 0, 2.5f, 0.25f);
    MeshB *h = pb(B_HEAD);
    mb_box(h, 0, 0, 0, 190, 170, 150, face);
    static const float coils[8][3] = {{-55, 55, 20}, {55, 55, 20}, {0, 15, 16}, {-70, -10, 17}, {70, -10, 17}, {-35, -70, 13}, {35, -70, 13}, {0, 60, 12}};
    for (int i = 0; i < 8; i++) { mb_xform(h, coils[i][0], coils[i][1], 77, PI_F / 2, srand01() * TAU_F, 0, 1, 1, 1); mb_cyl(h, 0, 0, 0, coils[i][2], coils[i][2], 8, 6, coil, true); mb_cyl(h, 0, 2, 0, coils[i][2] * 0.5f, coils[i][2] * 0.5f, 9, 6, face, true); }
    mb_identity(h);
    mb_box(h, 0, 35, 72, 175, 25, 30, RGB(0x4A2A1A));
    for (int s = -1; s <= 1; s += 2) { mb_xform(h, s * 85, 105, 0, 0, 0, -s * 0.5f, 1, 1, 1); mb_cyl(h, 0, 0, 0, 0, 20, 110, 5, bone, false); }
    mb_identity(h);
    part_new(md, B_HEAD, 0, -0.38f, 0.72f);   // mouth
    mb_box(pb(B_MOUTH), 0, 0, 0, 115, 35, 20, RGB(0x220A05));
    for (int i = 0; i < 6; i++) mb_box(pb(B_MOUTH), -45 + i * 18, 14, 10, 10, 12, 8, RGB(0xF2EAD0));
    part_new(md, B_HEAD, 0, 0.14f, 0.78f);    // eyes (mesh swapped per state)
    part_new(md, B_TORSO, 1.65f, 1.75f, 0); part_new(md, B_TORSO, -1.65f, 1.75f, 0);
    for (int a = B_ARMP; a <= B_ARMN; a++) {
        mb_box(pb(a), 0, -105, 0, 65, 210, 75, hide);
        mb_box(pb(a), 0, -5, 0, 80, 40, 90, leaf);
        for (int k = -1; k <= 1; k++) { mb_xform(pb(a), k * 20, -235, 20, PI_F, 0, 0, 1, 1, 1); mb_cyl(pb(a), 0, 0, 0, 0, 10, 45, 4, bone, false); }
        mb_identity(pb(a));
    }
    model_bake(md);
    static const uint32_t eyeCols[3] = {RGB(0xFFD040), RGB(0xFF3020), RGB(0x604818)};
    for (int e = 0; e < 3; e++) { MeshB *m = mb_new(100); mb_box(m, -45, 0, 0, 32, 18, 10, eyeCols[e]); mb_box(m, 45, 0, 0, 32, 18, 10, eyeCols[e]); MESH_EYES[e] = mb_bake(m); mb_free(m); }
    M_HUMB.p[B_EYES].mesh = MESH_EYES[0];
}

static void build_misc(void) {
    MeshB *m;
    // sword trail: a flat arc in front of the hero (drawn additively)
    m = mb_new(100);
    for (int i = 0; i < 12; i++) {
        float a0 = -1.6f + 3.2f * i / 12, a1 = -1.6f + 3.2f * (i + 1) / 12;
        float p[4][3] = {{sinf(a0) * 70, 0, cosf(a0) * 70}, {sinf(a0) * 205, 0, cosf(a0) * 205}, {sinf(a1) * 205, 0, cosf(a1) * 205}, {sinf(a1) * 70, 0, cosf(a1) * 70}};
        mb_quad_raw(m, p, RGBA(255, 240, 190, 150));
        float q[4][3] = {{p[3][0], 0, p[3][2]}, {p[2][0], 0, p[2][2]}, {p[1][0], 0, p[1][2]}, {p[0][0], 0, p[0][2]}};
        mb_quad_raw(m, q, RGBA(255, 240, 190, 150));
    }
    MESH_ARC = mb_bake(m); mb_free(m);
    m = mb_new(100); mb_disc(m, 0, 0, 0, 100, 10, RGBA(0, 0, 0, 90)); MESH_SHADOW = mb_bake(m); mb_free(m);
    m = mb_new(100); mb_xform(m, 0, 0, 0, PI_F, 0, 0, 1, 1, 1); mb_cyl(m, 0, 0, 0, 0, 28, 50, 4, RGB(0xFF6A2A), true); MESH_RETICLE = mb_bake(m); mb_free(m);
    m = mb_new(100);   // heart: two lobes and a point
    mb_box(m, -11, 6, 0, 22, 20, 12, RGB(0xFF3A4A)); mb_box(m, 11, 6, 0, 22, 20, 12, RGB(0xFF3A4A));
    mb_xform(m, 0, -8, 0, PI_F, PI_F / 4, 0, 1, 1, 1); mb_cyl(m, 0, 0, 0, 0, 22, 22, 4, RGB(0xE8303E), false);
    MESH_HEART = mb_bake(m); mb_free(m);
    m = mb_new(100); mb_cyl(m, 0, 11, 0, 0, 22, 22, 4, RGB(0x3A64FF), false); mb_xform(m, 0, -11, 0, PI_F, 0, 0, 1, 1, 1); mb_cyl(m, 0, 0, 0, 0, 22, 22, 4, RGB(0x2342C0), false); MESH_LAPIS = mb_bake(m); mb_free(m);
    m = mb_new(100); mb_cyl(m, 0, 0, 0, 22, 22, 70, 8, RGB(0xF6D272), false); mb_cyl(m, 0, 37, 0, 24, 24, 6, 8, RGB(0x2A4DBF), true); mb_cyl(m, 0, -37, 0, 24, 24, 6, 8, RGB(0x2A4DBF), true);
    for (int k = 0; k < 3; k++) mb_cyl(m, 0, -20 + k * 20, 0, 23, 23, 4, 8, RGB(0xB9852A), false);
    MESH_SEAL = mb_bake(m); mb_free(m);
    m = mb_new(100); mb_cyl(m, 0, 450, 0, 50, 70, 900, 6, RGBA(255, 224, 128, 70), false); MESH_BEAM = mb_bake(m); mb_free(m);
    m = mb_new(100);   // clay jar
    static const float prof[][2] = {{0, 26}, {20, 38}, {42, 40}, {64, 30}, {74, 15}, {88, 20}};
    for (int i = 0; i < 5; i++) { float y0 = prof[i][0], y1 = prof[i + 1][0]; mb_cyl(m, 0, (y0 + y1) / 2, 0, prof[i + 1][1], prof[i][1], y1 - y0, 8, i == 2 ? RGB(0x2A1A10) : RGB(0xB5562B), false); }
    mb_disc(m, 0, 88, 0, 18, 8, RGB(0x3A1A08));
    MESH_POT = mb_bake(m); mb_free(m);
    m = mb_new(100); mb_cyl(m, 0, 20, 0, 0, 40, 40, 5, RGB(0xFF7A20), false); mb_xform(m, 0, -20, 0, PI_F, 0, 0, 1, 1, 1); mb_cyl(m, 0, 0, 0, 0, 40, 40, 5, RGB(0xFFD040), false); MESH_FIRE = mb_bake(m); mb_free(m);
    m = mb_new(100); mb_cyl(m, 0, 20, 0, 0, 40, 40, 5, RGB(0x80C0FF), false); mb_xform(m, 0, -20, 0, PI_F, 0, 0, 1, 1, 1); mb_cyl(m, 0, 0, 0, 0, 40, 40, 5, RGB(0xE0F0FF), false); MESH_FIRE_R = mb_bake(m); mb_free(m);
    m = mb_new(100);   // unit shock ring, radius 100cm
    for (int i = 0; i < 20; i++) {
        float a0 = i / 20.0f * TAU_F, a1 = (i + 1) / 20.0f * TAU_F;
        float p[4][3] = {{cosf(a0) * 85, 10, sinf(a0) * 85}, {cosf(a1) * 85, 10, sinf(a1) * 85}, {cosf(a1) * 115, 10, sinf(a1) * 115}, {cosf(a0) * 115, 10, sinf(a0) * 115}};
        mb_quad_raw(m, p, RGBA(255, 176, 80, 200));
    }
    MESH_SHOCK = mb_bake(m); mb_free(m);
    for (int c = 0; c < 8; c++) { m = mb_new(100); mb_box(m, 0, 0, 0, 100, 100, 100, PART_COLORS[c]); MESH_PART[c] = mb_bake(m); mb_free(m); }
}

static void draw_simple(const Mesh *mesh, V3 p, float ry, float rx, float s) {
    M4 m; m4_trs(&m, p.x * RS, p.y * RS, p.z * RS, rx, ry, 0, s * RS * 0.01f);
    draw_mesh_at(mesh, &m);
}

// ===================================================================== particles
#define PMAX 56
typedef struct { V3 p, v; float life, max, s; int c; } Particle;
static Particle s_parts[PMAX];
static int s_pIdx;
void particles_burst(V3 pos, int n, int c, float speed, float up, float y) {
    for (int k = 0; k < n; k++) {
        Particle *P = &s_parts[s_pIdx]; s_pIdx = (s_pIdx + 1) % PMAX;
        float a = frand() * TAU_F, sp = speed * (0.4f + frand() * 0.6f);
        P->p = v3(pos.x, pos.y + y, pos.z);
        P->v = v3(cosf(a) * sp, up * (0.5f + frand()), sinf(a) * sp);
        P->life = P->max = 0.5f + frand() * 0.4f; P->s = 0.1f + frand() * 0.1f; P->c = c;
    }
}
static void particles_update(float dt) {
    for (int i = 0; i < PMAX; i++) {
        Particle *P = &s_parts[i]; if (P->life <= 0) continue;
        P->life -= dt; P->v.y -= 12 * dt;
        P->p.x += P->v.x * dt; P->p.y += P->v.y * dt; P->p.z += P->v.z * dt;
        float gy = height_at(P->p.x, P->p.z);
        if (P->p.y < gy + 0.05f) { P->p.y = gy + 0.05f; P->v.y *= -0.3f; P->v.x *= 0.6f; P->v.z *= 0.6f; }
    }
}
static void particles_draw(void) {
    for (int i = 0; i < PMAX; i++) {
        Particle *P = &s_parts[i]; if (P->life <= 0) continue;
        float k = P->life / P->max;
        draw_simple(&MESH_PART[P->c], P->p, P->life * 7, P->life * 5, P->s * (0.3f + 0.7f * k));
    }
}

// ===================================================================== entities
enum { P_NORMAL, P_ATTACK, P_ROLL, P_HURT, P_DEAD };
typedef struct { float dur, from, to, h0, h1, range, arc; int dmg; float lunge; } AttackDef;
static const AttackDef ATTACKS[3] = {
    {0.32f, -1.6f, 1.4f, 0.08f, 0.24f, 2.3f, 1.35f, 1, 3},
    {0.32f, 1.4f, -1.6f, 0.08f, 0.24f, 2.3f, 1.35f, 1, 3},
    {0.44f, 0, 0, 0.10f, 0.26f, 2.8f, 0.6f, 2, 8},
};
#define ROLL_T 0.48f

typedef struct {
    V3 pos, vel, rollDir, knock;
    float facing, t, inv, walk;
    int state, combo;
    bool queued, shield;
    uint32_t hitMask;   // targets already hit by this swing (bit per target slot)
    Pose pose;
} Player;
static Player P;

typedef struct { V3 pos, vel; float facing, walk; Pose pose; } Companion;
static Companion E;

enum { E_IDLE, E_CHASE, E_RETURN, E_WINDUP, E_LUNGE, E_RECOVER, E_HURT, E_DEAD, E_GONE };
typedef struct { int hp; float speed, radius; int dmg; float aggro, range, windup, lungeSpeed, lungeTime, camH; int col0, col1; } EnemyDef;
static const EnemyDef EDEF[2] = {
    {3, 3.0f, 0.6f, 1, 9, 2.3f, 0.55f, 9.0f, 0.30f, 1.6f, PC_SHELL, PC_SPARK},
    {4, 4.6f, 0.6f, 1, 11, 3.4f, 0.45f, 12.5f, 0.30f, 1.3f, PC_FUR, PC_SPARK},
};
#define MAX_ENEMIES 12
typedef struct { int kind, state, hp; V3 pos, home, vel, wander; float facing, t, flash, walk, seed; bool hitDone, ring; Pose pose; } Enemy;
static Enemy s_en[MAX_ENEMIES];
static int s_nen;

enum { BS_DORMANT, BS_ROAR, BS_IDLE, BS_WALK, BS_SLAMW, BS_CHARGEW, BS_CHARGE, BS_FIREW, BS_RECOVER, BS_STUNNED, BS_DYING, BS_DEAD };
typedef struct { V3 pos, home, vel; float facing, t, flash, walk, recoverT; int state, hp, maxHp, last; bool active, phase2; Pose pose; } Boss;
static Boss B;
#define BOSS_RADIUS 1.7f
#define BOSS_HP 24

#define MAX_POTS 16
typedef struct { V3 pos; bool alive; float rot; } Pot;
static Pot s_pots[MAX_POTS];
static int s_npots;

enum { PK_HEART, PK_LAPIS, PK_SEAL };
#define MAX_PICKUPS 24
typedef struct { bool used; int kind, id; V3 pos; float h, vy, vx, vz, baseH, life, reach, t, rot; } Pickup;
static Pickup s_pk[MAX_PICKUPS];

#define MAX_FB 8
typedef struct { bool used, reflected; V3 pos, vel; float life, rot; } Fireball;
static Fireball s_fb[MAX_FB];
typedef struct { bool used, hit; float x, z, r; } Shock;
static Shock s_sw[3];

// lock-on target slots: 0..MAX_ENEMIES-1 enemies, 100 boss
static int s_lock = -1;
static float s_camRecenter;

// ------------------------------------------------------------------ public queries
V3 player_pos(void) { return P.pos; }
V3 enkidu_pos(void) { return E.pos; }
float player_facing(void) { return P.facing; }
bool player_dead(void) { return P.state == P_DEAD; }
void player_set(V3 p, float facing) { P.pos = p; P.pos.y = height_at(p.x, p.z); P.facing = facing; P.vel = v3(0, 0, 0); }
void *lock_target(void) { return s_lock >= 0 ? (void *)&s_lock : NULL; }
bool boss_active(void) { return B.active; }
V3 boss_pos(void) { return B.pos; }
void boss_set_state_roar(void) { B.state = BS_ROAR; }
void boss_begin_fight(void) { B.active = true; B.state = BS_IDLE; B.t = 0; G.bossBar = true; G.bossHpFrac = 1; }
void enkidu_face_player(void) { E.facing = atan2f(P.pos.x - E.pos.x, P.pos.z - E.pos.z); }
V3 lock_target_pos(float *camH, bool *isBoss) {
    if (s_lock == 100) { *camH = 5.2f; *isBoss = true; return B.pos; }
    *isBoss = false;
    if (s_lock >= 0) { *camH = EDEF[s_en[s_lock].kind].camH; return s_en[s_lock].pos; }
    *camH = 0; return P.pos;
}

// ------------------------------------------------------------------ pickups
static Pickup *spawn_pickup(int kind, float x, float z, bool pop, float baseH, float life, float reach, int id) {
    for (int i = 0; i < MAX_PICKUPS; i++) if (!s_pk[i].used) {
        Pickup *u = &s_pk[i];
        *u = (Pickup){true, kind, id, v3(x, height_at(x, z), z), pop ? 0.4f : baseH, pop ? 6 : 0, pop ? (frand() - 0.5f) * 3 : 0, pop ? (frand() - 0.5f) * 3 : 0, baseH, life, reach, 0, 0};
        return u;
    }
    return NULL;
}
static void drop_loot(V3 p, float chance) {
    if (frand() > chance) return;
    if (G.hp < G.maxHp && frand() < 0.55f) spawn_pickup(PK_HEART, p.x, p.z, true, 0.5f, 12, 1.0f, 0);
    else { int n = frand() < 0.3f ? 3 : 1; for (int i = 0; i < n; i++) spawn_pickup(PK_LAPIS, p.x, p.z, true, 0.5f, 12, 1.0f, 0); }
}
void spawn_seal(int id, float x, float z) { spawn_pickup(PK_SEAL, x, z, false, id == 2 ? 1.4f : 1.5f, 1e9f, 1.7f, id); }
static void pickups_update(float dt) {
    for (int i = 0; i < MAX_PICKUPS; i++) {
        Pickup *u = &s_pk[i]; if (!u->used) continue;
        u->t += dt; u->life -= dt; u->rot += dt * (u->kind == PK_SEAL ? 1.5f : 2.5f);
        if (u->vy != 0 || u->h > u->baseH + 0.01f) {
            u->vy -= 18 * dt; u->h += u->vy * dt; u->pos.x += u->vx * dt; u->pos.z += u->vz * dt;
            resolve(&u->pos, 0.2f); u->pos.y = height_at(u->pos.x, u->pos.z);
            if (u->h < u->baseH) { u->h = u->baseH; u->vy = u->vy < -2.5f ? -u->vy * 0.4f : 0; u->vx *= 0.5f; u->vz *= 0.5f; }
        }
        if (u->life <= 0) { u->used = false; continue; }
        if (P.state != P_DEAD && dist2d(u->pos, P.pos) < u->reach && u->t > 0.25f) {
            u->used = false;
            if (u->kind == PK_HEART) { G.hp = G.hp + 2 > G.maxHp ? G.maxHp : G.hp + 2; sfx(SFX_HEART); }
            else if (u->kind == PK_LAPIS) { G.lapis++; sfx(SFX_LAPIS); }
            else { G.seals[u->id] = true; sfx(SFX_SEAL); particles_burst(P.pos, 20, PC_SPARK, 5, 6, 1); on_seal_collected(u->id); }
            hud_dirty();
        }
    }
}
static void pickups_draw(void) {
    for (int i = 0; i < MAX_PICKUPS; i++) {
        Pickup *u = &s_pk[i]; if (!u->used) continue;
        if (u->life < 3 && ((int)(u->life * 8) & 1)) continue;
        float bob = u->vy == 0 ? sinf(u->t * 3) * 0.1f : 0;
        V3 p = v3(u->pos.x, u->pos.y + u->h + bob, u->pos.z);
        draw_simple(u->kind == PK_HEART ? &MESH_HEART : u->kind == PK_LAPIS ? &MESH_LAPIS : &MESH_SEAL, p, u->rot, 0, 1);
    }
}
static void pickups_draw_additive(void) {
    for (int i = 0; i < MAX_PICKUPS; i++) {
        Pickup *u = &s_pk[i]; if (!u->used || u->kind != PK_SEAL) continue;
        draw_simple(&MESH_BEAM, v3(u->pos.x, u->pos.y + u->h, u->pos.z), u->rot * 0.3f, 0, 1);
    }
}

// ------------------------------------------------------------------ pots
static void add_pot(float x, float z) { if (s_npots < MAX_POTS) s_pots[s_npots++] = (Pot){v3(x, height_at(x, z), z), true, srand01() * TAU_F}; }
void pots_resolve(V3 *pos, float r, bool *moved) {
    for (int i = 0; i < s_npots; i++) {
        Pot *p = &s_pots[i]; if (!p->alive) continue;
        float dx = pos->x - p->pos.x, dz = pos->z - p->pos.z, R = 0.42f + r, q = dx * dx + dz * dz;
        if (q < R * R) { float d = sqrtf(q); if (d < 1e-4f) { d = 1e-4f; dx = 1e-4f; } pos->x = p->pos.x + dx / d * R; pos->z = p->pos.z + dz / d * R; *moved = true; }
    }
}
static void pot_break(int i) {
    Pot *p = &s_pots[i]; if (!p->alive) return;
    p->alive = false; sfx(SFX_HIT);
    particles_burst(p->pos, 10, PC_CLAY, 4, 4, 0.5f);
    drop_loot(p->pos, 0.85f);
}

// ------------------------------------------------------------------ combat
static int hurt_player(int dmg, V3 src, bool unblockable) {
    if (G.state != ST_PLAY || P.state == P_DEAD || P.inv > 0) return 0;
    if (P.state == P_ROLL && P.t < ROLL_T * 0.8f) return 0;
    float dx = src.x - P.pos.x, dz = src.z - P.pos.z, d = sqrtf(dx * dx + dz * dz); if (d < 1e-3f) d = 1e-3f;
    if (P.shield && !unblockable) {
        float dot = (dx * sinf(P.facing) + dz * cosf(P.facing)) / d;
        if (dot > 0.35f) {
            sfx(SFX_BLOCK);
            particles_burst(v3(P.pos.x + dx / d * 0.6f, P.pos.y, P.pos.z + dz / d * 0.6f), 6, PC_SPARK, 3, 2, 1.1f);
            P.vel = v3(-dx / d * 5, 0, -dz / d * 5);
            return 2;   // blocked
        }
    }
    G.hp -= dmg; if (G.hp < 0) G.hp = 0;
    P.inv = 1.1f; P.state = P_HURT; P.t = 0; P.knock = v3(-dx / d * 9, 0, -dz / d * 9);
    sfx(SFX_HURT); G.shake = 0.25f; hud_dirty();
    particles_burst(P.pos, 6, PC_BLOOD, 3, 3, 1);
    if (G.hp <= 0) { P.state = P_DEAD; P.t = 0; s_lock = -1; sfx(SFX_DIE); synth_set_track(0); }
    return 1;       // hit
}

static void enemy_hit(Enemy *e, int dmg, float nx, float nz) {
    if (e->state == E_DEAD || e->state == E_GONE) return;
    e->hp -= dmg; e->flash = 0.18f; sfx(SFX_HIT); G.hitstop = 0.05f;
    e->vel = v3(nx * 9, 0, nz * 9);
    particles_burst(e->pos, 5, EDEF[e->kind].col0, 3, 3, 0.8f);
    if (e->hp <= 0) { e->state = E_DEAD; e->t = 0; sfx(SFX_KILL); }
    else { e->state = E_HURT; e->t = 0; }
}
static void boss_hit(int dmg) {
    if (B.state == BS_DYING || B.state == BS_DEAD || !B.active) return;
    int mult = B.state == BS_STUNNED ? 2 : 1;
    B.hp -= dmg * mult; B.flash = 0.18f; sfx(SFX_BOSSHIT); G.hitstop = 0.06f;
    particles_burst(B.pos, 8, PC_LEAF, 4, 3, 2.5f);
    G.bossHpFrac = B.hp > 0 ? (float)B.hp / B.maxHp : 0;
    if (!B.phase2 && B.hp <= B.maxHp / 2) { B.phase2 = true; sfx(SFX_ROAR); G.shake = 0.6f; }
    if (B.hp <= 0) { B.state = BS_DYING; B.t = 0; s_lock = -1; synth_set_track(0); sfx(SFX_ROAR); G.shake = 1; }
}

static int find_target(void) {
    int best = -1; float bs = 1e9f, fx = sinf(P.facing), fz = cosf(P.facing);
    for (int i = 0; i < s_nen; i++) {
        Enemy *e = &s_en[i]; if (e->state == E_DEAD || e->state == E_GONE) continue;
        float dx = e->pos.x - P.pos.x, dz = e->pos.z - P.pos.z, d = sqrtf(dx * dx + dz * dz) + 1e-3f;
        if (d > 15) continue;
        float dot = (dx * fx + dz * fz) / d; if (dot < -0.2f && d > 5) continue;
        float sc = d * (1.6f - dot); if (sc < bs) { bs = sc; best = i; }
    }
    if (B.active && B.state != BS_DYING && B.state != BS_DEAD) {
        float dx = B.pos.x - P.pos.x, dz = B.pos.z - P.pos.z, d = sqrtf(dx * dx + dz * dz) + 1e-3f;
        float dot = (dx * fx + dz * fz) / d;
        if (d < 26 && !(dot < -0.2f && d > 5)) { float sc = d * (1.6f - dot); if (sc < bs) best = 100; }
    }
    return best;
}

// ===================================================================== player
static void player_update(float dt) {
    P.inv = maxf(0, P.inv - dt);
    float fx = sinf(g_camYaw), fz = cosf(g_camYaw);
    float rx = -fz, rz = fx;   // camera right
    float wx = fx * IN.my + rx * IN.mx, wz = fz * IN.my + rz * IN.mx;
    float mag = minf(1, sqrtf(wx * wx + wz * wz));
    float dx = 0, dz = 0; if (mag > 0.01f) { float l = sqrtf(wx * wx + wz * wz); dx = wx / l; dz = wz / l; }

    if (IN.z) { s_lock = find_target(); if (s_lock < 0) s_camRecenter = 0.35f; }
    if (!IN.zHeld) s_lock = -1;
    if (s_lock >= 0 && s_lock < 100) { Enemy *e = &s_en[s_lock]; if (e->state == E_DEAD || e->state == E_GONE || dist2d(e->pos, P.pos) > 28) s_lock = -1; }
    if (s_lock == 100 && (!B.active || B.state == BS_DYING || B.state == BS_DEAD)) s_lock = -1;

    const char *ctx = P.state == P_NORMAL ? context_label(P.pos) : NULL;
    G.aLabel = ctx ? ctx : (P.shield ? "Guard" : "Roll");
    P.shield = IN.rHeld && P.state == P_NORMAL;
    float tx = 0, tz = 0; bool locked = false;
    if (s_lock >= 0) { float ch; bool ib; V3 tp = lock_target_pos(&ch, &ib); tx = tp.x; tz = tp.z; locked = true; }

    switch (P.state) {
    case P_NORMAL: {
        float speed = P.shield ? 0 : (locked ? 4.6f : 6.6f) * mag;
        P.vel.x = dampf(P.vel.x, dx * speed, 14, dt); P.vel.z = dampf(P.vel.z, dz * speed, 14, dt);
        if (locked) P.facing = turn_toward(P.facing, atan2f(tx - P.pos.x, tz - P.pos.z), 14 * dt);
        else if (mag > 0.15f && !P.shield) P.facing = turn_toward(P.facing, atan2f(dx, dz), 12 * dt);
        if (IN.b) { P.state = P_ATTACK; P.combo = 0; P.t = 0; P.queued = false; P.hitMask = 0; sfx(SFX_SWING); if (locked) P.facing = atan2f(tx - P.pos.x, tz - P.pos.z); }
        else if (IN.a) {
            if (ctx && context_run(P.pos)) { P.vel = v3(0, 0, 0); }
            else {
                P.state = P_ROLL; P.t = 0; sfx(SFX_ROLL);
                if (mag > 0.2f) P.rollDir = v3(dx, 0, dz); else P.rollDir = v3(sinf(P.facing), 0, cosf(P.facing));
                if (!locked) P.facing = atan2f(P.rollDir.x, P.rollDir.z);
            }
        }
        break;
    }
    case P_ATTACK: {
        const AttackDef *a = &ATTACKS[P.combo]; P.t += dt;
        float lk = P.t < a->dur * 0.4f ? a->lunge : 0;
        P.vel.x = dampf(P.vel.x, sinf(P.facing) * lk, 20, dt); P.vel.z = dampf(P.vel.z, cosf(P.facing) * lk, 20, dt);
        if (IN.b && P.t > a->dur * 0.25f) P.queued = true;
        if (P.t >= a->h0 && P.t <= a->h1) {
            float ffx = sinf(P.facing), ffz = cosf(P.facing);
            // enemies
            for (int i = 0; i < s_nen; i++) {
                Enemy *e = &s_en[i]; if (e->state == E_DEAD || e->state == E_GONE || (P.hitMask & (1u << i))) continue;
                float ex = e->pos.x - P.pos.x, ez = e->pos.z - P.pos.z, d = sqrtf(ex * ex + ez * ez) + 1e-3f;
                if (d > a->range + EDEF[e->kind].radius) continue;
                if (d > EDEF[e->kind].radius + 0.4f && (ex * ffx + ez * ffz) / d < cosf(a->arc)) continue;
                P.hitMask |= 1u << i; enemy_hit(e, a->dmg, ex / d, ez / d);
                particles_burst(v3(P.pos.x + ex / d * 1.2f, e->pos.y, P.pos.z + ez / d * 1.2f), 4, PC_SPARK, 3, 2, 1.1f);
            }
            // pots
            for (int i = 0; i < s_npots; i++) {
                Pot *p = &s_pots[i]; if (!p->alive || (P.hitMask & (1u << (16 + i % 15)))) continue;
                float ex = p->pos.x - P.pos.x, ez = p->pos.z - P.pos.z, d = sqrtf(ex * ex + ez * ez) + 1e-3f;
                if (d > a->range + 0.42f) continue;
                if (d > 0.8f && (ex * ffx + ez * ffz) / d < cosf(a->arc)) continue;
                pot_break(i);
            }
            // boss
            if (B.active && !(P.hitMask & (1u << 31))) {
                float ex = B.pos.x - P.pos.x, ez = B.pos.z - P.pos.z, d = sqrtf(ex * ex + ez * ez) + 1e-3f;
                if (d <= a->range + BOSS_RADIUS && (d <= BOSS_RADIUS + 0.4f || (ex * ffx + ez * ffz) / d >= cosf(a->arc))) {
                    P.hitMask |= 1u << 31; boss_hit(a->dmg);
                    particles_burst(v3(P.pos.x + ex / d * 1.4f, B.pos.y, P.pos.z + ez / d * 1.4f), 4, PC_SPARK, 3, 2, 1.3f);
                }
            }
        }
        if (P.t >= a->dur) {
            if (P.queued && P.combo < 2) { P.combo++; P.t = 0; P.queued = false; P.hitMask = 0; sfx(P.combo == 2 ? SFX_SWING3 : SFX_SWING); if (locked) P.facing = atan2f(tx - P.pos.x, tz - P.pos.z); }
            else { P.state = P_NORMAL; P.t = 0; }
        }
        break;
    }
    case P_ROLL:
        P.t += dt; { float sp = P.t < ROLL_T * 0.8f ? 11 : 4; P.vel = v3(P.rollDir.x * sp, 0, P.rollDir.z * sp); }
        if (P.t >= ROLL_T) { P.state = P_NORMAL; P.t = 0; }
        break;
    case P_HURT:
        P.t += dt; P.vel = P.knock; { float k = expf(-8 * dt); P.knock.x *= k; P.knock.z *= k; }
        if (P.t >= 0.35f) { P.state = P_NORMAL; P.t = 0; }
        break;
    case P_DEAD:
        P.t += dt; { float k = expf(-6 * dt); P.vel.x *= k; P.vel.z *= k; }
        if (P.t > 1.8f && G.state == ST_PLAY) game_over();
        break;
    }
    P.pos.x += P.vel.x * dt; P.pos.z += P.vel.z * dt;
    resolve(&P.pos, 0.45f);
    P.pos.y = height_at(P.pos.x, P.pos.z);
}

static float animate_humanoid(Pose *ps, float *walk, V3 vel, float dt) {
    float sp = sqrtf(vel.x * vel.x + vel.z * vel.z); *walk += sp * dt * 2.1f;
    float ph = *walk, sw = sinf(ph), k = minf(1, sp / 5);
    pose_reset(ps);
    ps->oy[H_HIPS] = fabsf(cosf(ph)) * 0.07f * k;
    if (k < 0.1f) ps->oy[H_TORSO] = sinf(G.time * 2.2f) * 0.012f;
    ps->rx[H_LEGL] = sw * 0.85f * k; ps->rx[H_LEGR] = -sw * 0.85f * k;
    ps->rx[H_ARML] = -sw * 0.6f * k; ps->rz[H_ARML] = 0.05f;
    ps->rx[H_ARMR] = sw * 0.6f * k; ps->rz[H_ARMR] = -0.05f;
    return k;
}
static float s_arcAlpha, s_arcSign;
static void animate_player(float dt) {
    Pose *ps = &P.pose;
    animate_humanoid(ps, &P.walk, P.vel, dt);
    s_arcAlpha = 0;
    if (P.shield) { ps->rx[H_ARML] = -1.35f; ps->ry[H_ARML] = 0; ps->rz[H_ARML] = -0.55f; ps->ry[H_TORSO] = 0.2f; }
    switch (P.state) {
    case P_ATTACK: {
        const AttackDef *a = &ATTACKS[P.combo]; float k = clampf(P.t / a->dur, 0, 1);
        if (P.combo < 2) {
            float s = smoothf(clampf((k - 0.05f) / 0.6f, 0, 1));
            ps->ry[H_TORSO] = a->from + (a->to - a->from) * s;
            ps->rx[H_ARMR] = -1.5f; ps->rz[H_ARMR] = 0; ps->rx[H_HANDR] = PI_F / 2;
            ps->rx[H_LEGL] = 0.45f; ps->rx[H_LEGR] = -0.35f;
            float ak = clampf((k - 0.05f) / 0.65f, 0, 1); s_arcAlpha = sinf(ak * PI_F); s_arcSign = P.combo == 0 ? 1 : -1;
        } else {
            float e = clampf(k / 0.3f, 0, 1);
            ps->rx[H_ARMR] = -0.3f - 1.25f * e; ps->rx[H_HANDR] = (PI_F / 2) * e;
            ps->oy[H_HIPS] = -0.1f; ps->rx[H_LEGL] = 0.8f; ps->rx[H_LEGR] = -0.6f; ps->rx[H_TORSO] = 0.15f;
        }
        break;
    }
    case P_ROLL: { float k = P.t / ROLL_T; ps->oy[H_HIPS] = -0.4f; ps->rx[H_HIPS] = smoothf(clampf(k / 0.8f, 0, 1)) * TAU_F; ps->rx[H_LEGL] = ps->rx[H_LEGR] = -1.4f; ps->rx[H_ARML] = ps->rx[H_ARMR] = -1.2f; break; }
    case P_HURT: ps->rx[H_TORSO] = -0.4f; ps->rz[H_ARML] = 0.6f; ps->rz[H_ARMR] = -0.6f; break;
    case P_DEAD: { float k = minf(1, P.t / 0.8f); ps->rx[H_HIPS] = -k * 1.5f; ps->oy[H_HIPS] = -0.6f * k; break; }
    }
}

// ===================================================================== Enkidu
static void enkidu_update(float dt) {
    float tx, tz;
    if (G.bossStarted && !G.bossDead && B.active) { tx = 2.5f; tz = ARENA_Z + ARENA_R + 1.5f; }
    else { float fx = sinf(P.facing), fz = cosf(P.facing); tx = P.pos.x - fx * 2.6f + fz * 1.3f; tz = P.pos.z - fz * 2.6f - fx * 1.3f; }
    float dx = tx - E.pos.x, dz = tz - E.pos.z, d = sqrtf(dx * dx + dz * dz) + 1e-3f;
    if (d > 24) { E.pos = v3(tx, 0, tz); resolve(&E.pos, 0.45f); }
    float want = d > 1.3f ? minf(7.6f, d * 2.4f) : 0;
    E.vel.x = dampf(E.vel.x, dx / d * want, 8, dt); E.vel.z = dampf(E.vel.z, dz / d * want, 8, dt);
    E.pos.x += E.vel.x * dt; E.pos.z += E.vel.z * dt;
    float sx = E.pos.x - P.pos.x, sz = E.pos.z - P.pos.z, sd = sqrtf(sx * sx + sz * sz);
    if (sd < 0.9f && sd > 0) { E.pos.x = P.pos.x + sx / sd * 0.9f; E.pos.z = P.pos.z + sz / sd * 0.9f; }
    resolve(&E.pos, 0.45f); E.pos.y = height_at(E.pos.x, E.pos.z);
    float sp = sqrtf(E.vel.x * E.vel.x + E.vel.z * E.vel.z);
    if (sp > 0.6f) E.facing = turn_toward(E.facing, atan2f(E.vel.x, E.vel.z), 10 * dt);
    else { V3 look = B.active ? B.pos : P.pos; E.facing = turn_toward(E.facing, atan2f(look.x - E.pos.x, look.z - E.pos.z), 4 * dt); }
}
static void animate_enkidu(float dt) {
    float k = animate_humanoid(&E.pose, &E.walk, E.vel, dt);
    if (B.active && k < 0.1f) { float c = sinf(G.time * 6); E.pose.rx[H_ARMR] = -2.6f + c * 0.3f; E.pose.rz[H_ARMR] = -0.2f; E.pose.rx[H_ARML] = -2.6f - c * 0.3f; E.pose.rz[H_ARML] = 0.2f; }
}

// ===================================================================== enemies
static void spawn_enemy(int kind, float x, float z, bool ring) {
    if (s_nen >= MAX_ENEMIES) return;
    Enemy *e = &s_en[s_nen++];
    memset(e, 0, sizeof(*e));
    e->kind = kind; e->hp = EDEF[kind].hp; e->pos = v3(x, height_at(x, z), z); e->home = e->pos;
    e->facing = srand01() * TAU_F; e->state = E_IDLE; e->t = srand01() * 2; e->seed = srand01() * 10; e->ring = ring;
    pose_reset(&e->pose);
}
static void enemy_update(Enemy *e, int idx, float dt) {
    const EnemyDef *d = &EDEF[e->kind];
    float dx = P.pos.x - e->pos.x, dz = P.pos.z - e->pos.z, dist = sqrtf(dx * dx + dz * dz) + 1e-3f, toP = atan2f(dx, dz);
    float homeD = dist2d(e->pos, e->home); bool pAlive = P.state != P_DEAD;
    e->t += dt; e->flash = maxf(0, e->flash - dt);
    float want = 0;
    switch (e->state) {
    case E_IDLE: {
        if (e->wander.x == 0 && e->wander.z == 0) e->wander = e->home;
        if (e->t > 3) { e->t = 0; float a = frand() * TAU_F, r = frand() * 3; e->wander = v3(e->home.x + cosf(a) * r, 0, e->home.z + sinf(a) * r); }
        float wx = e->wander.x - e->pos.x, wz = e->wander.z - e->pos.z;
        if (sqrtf(wx * wx + wz * wz) > 0.5f) { e->facing = turn_toward(e->facing, atan2f(wx, wz), 3 * dt); want = d->speed * 0.35f; }
        if (pAlive && dist < d->aggro) { e->state = E_CHASE; e->t = 0; }
        break;
    }
    case E_CHASE:
        e->facing = turn_toward(e->facing, toP, 6 * dt); want = d->speed;
        if (dist < d->range && fabsf(ang_diff(e->facing, toP)) < 0.4f) { e->state = E_WINDUP; e->t = 0; }
        if (!pAlive || dist > d->aggro * 1.8f || homeD > 16) { e->state = E_RETURN; e->t = 0; }
        break;
    case E_RETURN: {
        float hx = e->home.x - e->pos.x, hz = e->home.z - e->pos.z;
        e->facing = turn_toward(e->facing, atan2f(hx, hz), 5 * dt); want = d->speed * 0.8f;
        if (homeD < 1) { e->state = E_IDLE; e->t = 0; }
        else if (pAlive && dist < d->aggro * 0.7f && homeD < 12) { e->state = E_CHASE; e->t = 0; }
        break;
    }
    case E_WINDUP:
        e->facing = turn_toward(e->facing, toP, 3 * dt);
        if (e->t >= d->windup) { e->state = E_LUNGE; e->t = 0; e->hitDone = false; e->vel = v3(sinf(e->facing) * d->lungeSpeed, 0, cosf(e->facing) * d->lungeSpeed); }
        break;
    case E_LUNGE:
        if (!e->hitDone && dist < d->radius + 0.45f + 0.5f) {
            e->hitDone = true;
            if (hurt_player(d->dmg, e->pos, false) == 2) { e->state = E_RECOVER; e->t = -0.3f; e->vel = v3(-dx / dist * 7, 0, -dz / dist * 7); e->flash = 0.1f; }
        }
        if (e->state == E_LUNGE && e->t >= d->lungeTime) { e->state = E_RECOVER; e->t = 0; }
        break;
    case E_RECOVER: if (e->t >= 0.65f) { e->state = E_CHASE; e->t = 0; } break;
    case E_HURT: if (e->t >= 0.35f) { e->state = E_CHASE; e->t = 0; } break;
    case E_DEAD:
        if (e->t >= 0.5f) {
            e->state = E_GONE;
            particles_burst(e->pos, 14, PC_SPIRIT, 4, 5, 0.8f);
            drop_loot(e->pos, 0.7f);
            if (s_lock == idx) s_lock = -1;
            if (e->ring && !G.duskSpawned) {
                bool all = true;
                for (int i = 0; i < s_nen; i++) if (s_en[i].ring && s_en[i].state != E_GONE) all = false;
                if (all) { G.duskSpawned = true; on_ring_cleared(); }
            }
            return;
        }
        break;
    }
    bool free = e->state != E_LUNGE && e->state != E_HURT && e->state != E_DEAD && !(e->state == E_RECOVER && e->t < 0);
    if (free) { e->vel.x = dampf(e->vel.x, sinf(e->facing) * want, 10, dt); e->vel.z = dampf(e->vel.z, cosf(e->facing) * want, 10, dt); }
    else { float k = expf(-(e->state == E_LUNGE ? 1 : 6) * dt); e->vel.x *= k; e->vel.z *= k; }
    e->pos.x += e->vel.x * dt; e->pos.z += e->vel.z * dt;
    if (pAlive) { float sx = e->pos.x - P.pos.x, sz = e->pos.z - P.pos.z, sd = sqrtf(sx * sx + sz * sz), R = d->radius + 0.45f; if (sd < R && sd > 0) { e->pos.x = P.pos.x + sx / sd * R; e->pos.z = P.pos.z + sz / sd * R; } }
    for (int i = 0; i < s_nen; i++) {
        Enemy *o = &s_en[i]; if (o == e || o->state == E_GONE) continue;
        float sx = e->pos.x - o->pos.x, sz = e->pos.z - o->pos.z, sd = sqrtf(sx * sx + sz * sz), R = d->radius + EDEF[o->kind].radius;
        if (sd < R && sd > 0) { e->pos.x += sx / sd * (R - sd) * 0.5f; e->pos.z += sz / sd * (R - sd) * 0.5f; }
    }
    resolve(&e->pos, d->radius);
    e->pos.y = height_at(e->pos.x, e->pos.z);
}
static void animate_enemy(Enemy *e, float dt) {
    Pose *ps = &e->pose; pose_reset(ps);
    float sp = sqrtf(e->vel.x * e->vel.x + e->vel.z * e->vel.z); e->walk += sp * dt * 3;
    float k = minf(1, sp / 2);
    if (e->kind == 0) {
        for (int i = 0; i < 6; i++) { int side = i < 3 ? 0 : 1; ps->ry[S_LEG0 + i] = sinf(e->walk * 2 + (i % 3) * 2 + (side ? PI_F : 0)) * 0.45f * k; }
        float curl = 0.75f + sinf(G.time * 3 + e->seed) * 0.06f, t0 = 0.9f;
        if (e->state == E_WINDUP) { float w = minf(1, e->t / EDEF[0].windup); curl += 0.25f * w; ps->oy[S_BODY] = -0.12f * w; }
        else if (e->state == E_LUNGE) { curl = 1.25f; t0 = 1.5f; }
        for (int i = 0; i < 4; i++) ps->rx[S_TAIL0 + i] = i == 0 ? t0 : curl;
        ps->rx[S_ARMR] = e->state == E_LUNGE ? -1.5f : e->state == E_WINDUP ? 0.7f : -0.3f;
        ps->rx[S_TORSO] = e->state == E_WINDUP ? -0.25f : e->state == E_LUNGE ? 0.3f : 0;
    } else {
        float gal = e->walk * 1.6f;
        for (int i = 0; i < 4; i++) { bool front = i < 2; ps->rx[L_LEG0 + i] = sinf(gal + (front ? 0 : PI_F) + ((i & 1) ? 0 : 0.4f)) * 0.7f * k; }
        ps->oy[L_BODY] = fabsf(sinf(gal)) * 0.08f * k;
        if (e->state == E_WINDUP) { float w = minf(1, e->t / EDEF[1].windup); ps->oy[L_BODY] = -0.2f * w; ps->rx[L_BODY] = 0.12f * w; ps->rx[L_HEAD] = 0.2f * w; }
        else if (e->state == E_LUNGE) { ps->rx[L_BODY] = -0.3f; for (int i = 0; i < 4; i++) ps->rx[L_LEG0 + i] = i < 2 ? -1.1f : 0.9f; }
        ps->ry[L_TAIL] = sinf(G.time * 4 + e->seed) * 0.5f; ps->rx[L_TAIL] = 0.5f;
    }
}

// ===================================================================== Humbaba
static void spawn_shock(float x, float z) { for (int i = 0; i < 3; i++) if (!s_sw[i].used) { s_sw[i] = (Shock){true, false, x, z, 0.5f}; return; } }
static void spawn_fireball(float x, float z, float dir) {
    for (int i = 0; i < MAX_FB; i++) if (!s_fb[i].used) { s_fb[i] = (Fireball){true, false, v3(x, 0, z), v3(sinf(dir) * 9, 0, cosf(dir) * 9), 3.2f, 0}; return; }
}
static void boss_update(float dt) {
    if (!B.active) return;
    float dx = P.pos.x - B.pos.x, dz = P.pos.z - B.pos.z, dist = sqrtf(dx * dx + dz * dz) + 1e-3f, toP = atan2f(dx, dz);
    float ax = P.pos.x - ARENA_X, az = P.pos.z - ARENA_Z;
    bool inArena = sqrtf(ax * ax + az * az) < ARENA_R + 1 && P.state != P_DEAD;
    bool ph2 = B.phase2;
    B.t += dt; B.flash = maxf(0, B.flash - dt);
    float want = 0; bool charging = false;
    switch (B.state) {
    case BS_IDLE:
        if (!inArena) { float hx = B.home.x - B.pos.x, hz = B.home.z - B.pos.z; if (sqrtf(hx * hx + hz * hz) > 1) { B.facing = turn_toward(B.facing, atan2f(hx, hz), 2 * dt); want = 2.5f; } break; }
        B.facing = turn_toward(B.facing, toP, 2.5f * dt);
        if (B.t > (ph2 ? 0.55f : 1.0f)) {
            float r = frand(); B.t = 0;
            if (dist < 6.5f) B.state = r < 0.6f ? BS_SLAMW : r < 0.8f ? BS_WALK : BS_FIREW;
            else B.state = r < 0.4f ? BS_CHARGEW : r < 0.75f ? BS_FIREW : BS_WALK;
            if (B.state == BS_FIREW) sfx(SFX_FIRE);
        }
        break;
    case BS_WALK:
        B.facing = turn_toward(B.facing, toP, 3 * dt); want = ph2 ? 3.8f : 2.7f;
        if (B.t > 1.8f || dist < 4.5f) { B.state = dist < 5.5f ? BS_SLAMW : BS_IDLE; B.t = 0; }
        break;
    case BS_SLAMW:
        if (B.t < 0.4f) B.facing = turn_toward(B.facing, toP, 3 * dt);
        if (B.t >= (ph2 ? 0.65f : 0.85f)) {
            float fx = B.pos.x + sinf(B.facing) * 2.4f, fz = B.pos.z + cosf(B.facing) * 2.4f;
            spawn_shock(fx, fz); sfx(SFX_SLAM); G.shake = 0.6f;
            particles_burst(v3(fx, B.pos.y, fz), 12, PC_CLAY, 7, 5, 0.2f);
            B.state = BS_RECOVER; B.t = 0; B.recoverT = 0.9f; B.last = 1;
        }
        break;
    case BS_CHARGEW:
        B.facing = turn_toward(B.facing, toP, 4 * dt);
        if (B.t >= 0.75f) { B.state = BS_CHARGE; B.t = 0; sfx(SFX_ROAR); }
        break;
    case BS_CHARGE:
        charging = true; want = ph2 ? 17 : 15;
        if (B.t >= 1.2f) { B.state = BS_RECOVER; B.t = 0; B.recoverT = 0.6f; B.last = 2; }
        break;
    case BS_FIREW:
        B.facing = turn_toward(B.facing, toP, 3 * dt);
        if (B.t >= 0.6f) {
            int n = ph2 ? 5 : 3; float mx = B.pos.x + sinf(B.facing) * 1.8f, mz = B.pos.z + cosf(B.facing) * 1.8f;
            for (int i = 0; i < n; i++) spawn_fireball(mx, mz, toP + (i - (n - 1) / 2.0f) * 0.28f);
            sfx(SFX_FIRE); B.state = BS_RECOVER; B.t = 0; B.recoverT = 1.0f; B.last = 3;
        }
        break;
    case BS_RECOVER: if (B.t >= B.recoverT) { B.state = BS_IDLE; B.t = 0; } break;
    case BS_STUNNED:
        if (frand() < dt * 6) particles_burst(v3(B.pos.x, B.pos.y + 5.6f, B.pos.z), 1, PC_SPARK, 1.5f, 1, 0);
        if (B.t >= 2.0f) { B.state = BS_IDLE; B.t = 0; }
        break;
    case BS_DYING:
        if (frand() < dt * 10) particles_burst(B.pos, 3, frand() < 0.5f ? PC_LEAF : PC_SPARK, 5, 6, 2 + frand() * 3);
        if (B.t >= 3.2f) { B.active = false; B.state = BS_DEAD; G.bossBar = false; on_boss_defeated(); }
        break;
    default: break;
    }
    float fx = sinf(B.facing), fz = cosf(B.facing);
    if (charging) B.vel = v3(fx * want, 0, fz * want);
    else { B.vel.x = dampf(B.vel.x, fx * want, 6, dt); B.vel.z = dampf(B.vel.z, fz * want, 6, dt); }
    if (B.state == BS_DYING || B.state == BS_STUNNED) { float k = expf(-6 * dt); B.vel.x *= k; B.vel.z *= k; }
    B.pos.x += B.vel.x * dt; B.pos.z += B.vel.z * dt;
    float bx = B.pos.x - ARENA_X, bz = B.pos.z - ARENA_Z, bd = sqrtf(bx * bx + bz * bz), lim = ARENA_R - BOSS_RADIUS;
    if (bd > lim) {
        B.pos.x = ARENA_X + bx / bd * lim; B.pos.z = ARENA_Z + bz / bd * lim;
        if (charging) { B.state = BS_STUNNED; B.t = 0; B.vel = v3(0, 0, 0); sfx(SFX_THUD); G.shake = 0.8f; particles_burst(B.pos, 14, PC_LEAF, 6, 6, 4); }
    }
    B.pos.y = height_at(B.pos.x, B.pos.z);
    if (P.state != P_DEAD && B.state != BS_DYING) {
        float R = BOSS_RADIUS + 0.45f;
        if (dist < R + 0.2f && (B.state == BS_WALK || charging)) hurt_player(charging ? 2 : 1, B.pos, charging);
        float sx = P.pos.x - B.pos.x, sz = P.pos.z - B.pos.z, sd = sqrtf(sx * sx + sz * sz);
        if (sd < R && sd > 0) { P.pos.x = B.pos.x + sx / sd * R; P.pos.z = B.pos.z + sz / sd * R; }
    }
}
static int s_bossEye;
static float s_bossRootRx;
static void animate_boss(float dt) {
    Pose *ps = &B.pose; pose_reset(ps);
    float sp = sqrtf(B.vel.x * B.vel.x + B.vel.z * B.vel.z); B.walk += sp * dt;
    float sw = sinf(B.walk), k = minf(1, sp / 3);
    ps->rx[B_LEGL] = sw * 0.5f * k; ps->rx[B_LEGR] = -sw * 0.5f * k;
    ps->oy[B_HIPS] = fabsf(cosf(B.walk)) * 0.15f * k;
    ps->oy[B_TORSO] = sinf(G.time * 2) * 0.05f;
    ps->rx[B_ARMP] = -sw * 0.3f * k; ps->rz[B_ARMP] = 0.12f;
    ps->rx[B_ARMN] = sw * 0.3f * k; ps->rz[B_ARMN] = -0.12f;
    s_bossEye = 0; s_bossRootRx = 0;
    float t = B.t;
    switch (B.state) {
    case BS_DORMANT: ps->rx[B_HEAD] = 0.15f; ps->rx[B_TORSO] = 0.2f; s_bossEye = 2; break;
    case BS_ROAR: ps->sy[B_MOUTH] = 4; ps->rz[B_ARMP] = 1.1f; ps->rz[B_ARMN] = -1.1f; ps->rx[B_HEAD] = -0.25f; ps->rx[B_TORSO] = -0.15f; break;
    case BS_SLAMW: { float w = minf(1, t / (B.phase2 ? 0.65f : 0.85f)); ps->rx[B_ARMP] = ps->rx[B_ARMN] = -2.9f * w; ps->rx[B_TORSO] = -0.25f * w; break; }
    case BS_RECOVER:
        if (B.last == 1 && t < 0.5f) { ps->rx[B_ARMP] = ps->rx[B_ARMN] = -0.9f; ps->rx[B_TORSO] = 0.4f; }
        if (B.last == 3 && t < 0.3f) ps->sy[B_MOUTH] = 3;
        break;
    case BS_CHARGEW: ps->rx[B_TORSO] = 0.45f; ps->oy[B_HIPS] = -0.3f; ps->rz[B_HEAD] = sinf(t * 30) * 0.06f; s_bossEye = 1; break;
    case BS_CHARGE: ps->rx[B_TORSO] = 0.55f; ps->rx[B_ARMP] = ps->rx[B_ARMN] = 0.9f; s_bossEye = 1; ps->rx[B_LEGL] = sinf(G.time * 18) * 0.7f; ps->rx[B_LEGR] = -ps->rx[B_LEGL]; break;
    case BS_FIREW: { float w = minf(1, t / 0.6f); ps->rx[B_HEAD] = -0.3f * w; ps->sy[B_MOUTH] = 1 + 3 * w; s_bossEye = 1; break; }
    case BS_STUNNED: ps->rz[B_HEAD] = sinf(t * 6) * 0.3f; ps->rz[B_TORSO] = sinf(t * 3) * 0.12f; ps->rx[B_TORSO] = 0.3f; s_bossEye = 2; break;
    case BS_DYING: { float w = smoothf(minf(1, t / 1.8f)); s_bossRootRx = -w * 1.45f; ps->sy[B_MOUTH] = 3; s_bossEye = t > 1.8f ? 2 : 0; break; }
    case BS_DEAD: s_bossRootRx = -1.45f; s_bossEye = 2; break;
    }
}

static void projectiles_update(float dt) {
    for (int i = 0; i < MAX_FB; i++) {
        Fireball *f = &s_fb[i]; if (!f->used) continue;
        f->life -= dt; f->pos.x += f->vel.x * dt; f->pos.z += f->vel.z * dt; f->pos.y = height_at(f->pos.x, f->pos.z) + 1.2f; f->rot += dt * 8;
        bool dead = f->life <= 0;
        if (!f->reflected && !dead && dist2d(f->pos, P.pos) < 0.95f) {
            int r = hurt_player(1, f->pos, false);
            if (r == 2) { f->reflected = true; float tx = B.pos.x - f->pos.x, tz = B.pos.z - f->pos.z, tl = sqrtf(tx * tx + tz * tz) + 1e-3f; f->vel = v3(tx / tl * 15, 0, tz / tl * 15); f->life = 3; sfx(SFX_REFLECT); }
            else if (r == 1) dead = true;
        } else if (f->reflected && B.active && dist2d(f->pos, B.pos) < BOSS_RADIUS + 0.6f) {
            boss_hit(3); if (B.state != BS_DYING) { B.state = BS_STUNNED; B.t = 0.4f; }
            dead = true;
        }
        V3 before = f->pos;
        if (!dead && resolve(&f->pos, 0.3f) && (before.x != f->pos.x || before.z != f->pos.z)) dead = true;
        if (dead) { particles_burst(f->pos, 6, PC_FIRE, 3, 2, 0); f->used = false; }
    }
    for (int i = 0; i < 3; i++) {
        Shock *s = &s_sw[i]; if (!s->used) continue;
        s->r += dt * 9;
        float pd = sqrtf((P.pos.x - s->x) * (P.pos.x - s->x) + (P.pos.z - s->z) * (P.pos.z - s->z));
        if (!s->hit && fabsf(pd - s->r) < 0.55f) { if (hurt_player(2, v3(s->x, 0, s->z), true) == 1) s->hit = true; }
        if (s->r >= 10) s->used = false;
    }
}

// ===================================================================== lifecycle
void actors_init(void) {
    build_gilgamesh(); build_enkidu(); build_scorpion(); build_lion(); build_humbaba(); build_misc();
    P = (Player){0}; P.pos = v3(0, 0, 37); P.facing = PI_F; pose_reset(&P.pose); P.pos.y = height_at(0, 37);
    E = (Companion){0}; E.pos = v3(2.2f, 0, 38.5f); E.facing = PI_F; pose_reset(&E.pose);
    B = (Boss){0}; B.pos = B.home = v3(0, height_at(0, -57), -57); B.hp = B.maxHp = BOSS_HP; B.state = BS_DORMANT; pose_reset(&B.pose);
    // seals that are present from the start
    spawn_seal(0, 14.4f, 19.5f);
    spawn_seal(1, -25.5f, -1.5f);
    static const float scorps[][2] = {{-18, 0}, {-22, 4}, {-21.5f, -4}, {7, 6}, {-7, -15}};
    for (int i = 0; i < 5; i++) spawn_enemy(0, scorps[i][0], scorps[i][1], false);
    static const float lions[][2] = {{12.5f, -8}, {16, -12}, {12, -12.5f}};
    for (int i = 0; i < 3; i++) spawn_enemy(1, lions[i][0], lions[i][1], true);
    spawn_enemy(1, -10, -40, false);
    spawn_enemy(1, 14, -34, false);
    static const float pots[][2] = {{-3, 35.5f}, {-3.9f, 36.6f}, {4.5f, 38}, {17.2f, 23.6f}, {23.4f, 16.6f}, {24, 24}, {-17, 4.5f}, {-24.5f, 3}, {12, -50}, {-12, -48}, {11, -61}, {-11, -62}, {2, -24.5f}, {-2.2f, -25}};
    for (int i = 0; i < 14; i++) add_pot(pots[i][0], pots[i][1]);
}

void actors_continue(void) {
    P.state = P_NORMAL; P.inv = 1.5f; P.vel = v3(0, 0, 0); P.pos = G.checkpoint; P.pos.y = height_at(P.pos.x, P.pos.z); P.facing = PI_F;
    E.pos = v3(P.pos.x + 1.5f, 0, P.pos.z + 1.5f); s_lock = -1;
    for (int i = 0; i < s_nen; i++) { Enemy *e = &s_en[i]; if (e->state == E_GONE) continue; e->hp = EDEF[e->kind].hp; e->pos = e->home; e->state = E_IDLE; e->vel = v3(0, 0, 0); }
    for (int i = 0; i < MAX_FB; i++) s_fb[i].used = false;
    for (int i = 0; i < 3; i++) s_sw[i].used = false;
    if (G.bossStarted && !G.bossDead) { B.hp = B.maxHp; B.phase2 = false; B.pos = B.home; B.state = BS_IDLE; B.t = 0; B.facing = 0; G.bossHpFrac = 1; }
}

void actors_update(float dt) {
    if (G.hitstop > 0) { G.hitstop -= dt; return; }
    player_update(dt);
    for (int i = 0; i < s_nen; i++) if (s_en[i].state != E_GONE) enemy_update(&s_en[i], i, dt);
    boss_update(dt);
    enkidu_update(dt);
    pickups_update(dt);
    projectiles_update(dt);
}

void actors_title_anim(float dt) {
    animate_player(dt); animate_enkidu(dt);
    for (int i = 0; i < s_nen; i++) animate_enemy(&s_en[i], dt);
    animate_boss(dt);
    particles_update(dt);
}

void actors_draw(void) {
    // hero, blinking while invulnerable
    if (!(P.inv > 0 && P.state != P_DEAD && ((int)(P.inv * 20) & 1) == 0)) model_draw(&M_GIL, &P.pose, P.pos, P.facing, 0, 0, 0);
    model_draw(&M_ENK, &E.pose, E.pos, E.facing, 0, 0, 0);
    for (int i = 0; i < s_nen; i++) {
        Enemy *e = &s_en[i]; if (e->state == E_GONE || dist2d(e->pos, P.pos) > 50) continue;
        float rz = 0, flash = e->flash > 0 ? 1 : 0;
        if (e->state == E_DEAD) { rz = minf(1, e->t / 0.5f) * 1.4f; flash = 1; }
        else if (e->state == E_WINDUP && ((int)(e->t * 14) & 1)) flash = 0.35f;
        model_draw(e->kind == 0 ? &M_SCORP : &M_LION, &e->pose, e->pos, e->facing, 0, rz, flash);
    }
    s_override[B_EYES] = &MESH_EYES[s_bossEye];
    if (dist2d(B.pos, P.pos) < 60 || G.state == ST_TITLE) model_draw(&M_HUMB, &B.pose, B.pos, B.facing, s_bossRootRx, 0, B.flash > 0 ? 1 : 0);
    s_override[B_EYES] = NULL;
    for (int i = 0; i < s_npots; i++) if (s_pots[i].alive && dist2d(s_pots[i].pos, P.pos) < 45) draw_simple(&MESH_POT, s_pots[i].pos, s_pots[i].rot, 0, 1);
    pickups_draw();
    for (int i = 0; i < MAX_FB; i++) if (s_fb[i].used) draw_simple(s_fb[i].reflected ? &MESH_FIRE_R : &MESH_FIRE, s_fb[i].pos, s_fb[i].rot, s_fb[i].rot * 0.7f, 1);
    particles_draw();
    if (s_lock >= 0) { float ch; bool ib; V3 tp = lock_target_pos(&ch, &ib); draw_simple(&MESH_RETICLE, v3(tp.x, tp.y + ch + 0.6f + sinf(G.time * 6) * 0.1f, tp.z), G.time * 4, 0, 1); }
}

// shadows (alpha blended) and glowing effects (additive) are drawn after opaque geometry
static void draw_shadow(V3 p, float r) { M4 m; m4_trs(&m, p.x * RS, (height_at(p.x, p.z) + 0.05f) * RS, p.z * RS, 0, 0, 0, r * RS * 0.01f); draw_mesh_at(&MESH_SHADOW, &m); }
void actors_draw_shadows(void) {
    draw_shadow(P.pos, 0.55f); draw_shadow(E.pos, 0.55f);
    for (int i = 0; i < s_nen; i++) if (s_en[i].state != E_GONE && dist2d(s_en[i].pos, P.pos) < 40) draw_shadow(s_en[i].pos, 0.8f);
    if (B.state != BS_DEAD) draw_shadow(B.pos, 2.4f);
}
void actors_draw_additive(void) {
    pickups_draw_additive();
    if (s_arcAlpha > 0.15f) {
        M4 root, tilt, m;
        m4_trs(&root, P.pos.x * RS, (P.pos.y + 1.25f) * RS, P.pos.z * RS, 0, P.facing, 0, RS * 0.01f);
        m4_trs(&tilt, 0, 0, 0, 0, 0, 0, 1); if (s_arcSign < 0) tilt.m[0][0] = -1;
        m4_mul(&m, &root, &tilt);
        draw_mesh_at(&MESH_ARC, &m);
    }
    for (int i = 0; i < 3; i++) if (s_sw[i].used) { Shock *s = &s_sw[i]; M4 m; m4_trs(&m, s->x * RS, (height_at(s->x, s->z) + 0.15f) * RS, s->z * RS, 0, 0, 0, s->r * RS * 0.01f); draw_mesh_at(&MESH_SHOCK, &m); }
}

void actors_animate(float dt) {
    animate_player(dt); animate_enkidu(dt);
    for (int i = 0; i < s_nen; i++) if (s_en[i].state != E_GONE) animate_enemy(&s_en[i], dt);
    animate_boss(dt);
    particles_update(dt);
}
float actors_cam_recenter(float dt) { float r = s_camRecenter; s_camRecenter = maxf(0, s_camRecenter - dt); return r; }
