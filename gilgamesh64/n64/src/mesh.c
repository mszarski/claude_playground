// Procedural mesh builder and draw-matrix helpers.
#include "game.h"

typedef struct { float p[3]; uint16_t n; uint32_t c; } BVert;

struct MeshB {
    float unit;                 // multiplier from builder units to int16 mesh units
    float r[3][3], t[3];        // current transform (rotation*scale, translation)
    BVert *v; int nv, cv;
    uint16_t *ti; int nt, ct;   // triangle indices (3 per tri)
    bool boxBottoms;            // emit the underside of boxes (characters need them, scenery does not)
};

// ------------------------------------------------------------------ random + angles + colors
static uint32_t s_seed = 1337u;
float srand01(void) { s_seed = s_seed * 1664525u + 1013904223u; return (s_seed >> 8) / 16777216.0f; }
static uint32_t g_seed = 99991u;
float frand(void) { g_seed = g_seed * 1103515245u + 12345u; return (g_seed >> 8) / 16777216.0f; }
float ang_diff(float a, float b) {
    float d = fmodf(b - a, TAU_F);
    if (d > PI_F) d -= TAU_F;
    if (d < -PI_F) d += TAU_F;
    return d;
}
float turn_toward(float a, float b, float step) { return a + clampf(ang_diff(a, b), -step, step); }
uint32_t col_lerp(uint32_t a, uint32_t b, float t) {
    uint32_t out = 0;
    for (int s = 8; s < 32; s += 8) {
        float ca = (a >> s) & 0xFF, cb = (b >> s) & 0xFF;
        out |= ((uint32_t)clampf(lerpf(ca, cb, t), 0, 255)) << s;
    }
    return out | (a & 0xFF);
}
uint32_t col_shade(uint32_t c, float k) {
    uint32_t out = 0;
    for (int s = 8; s < 32; s += 8) out |= ((uint32_t)clampf(((c >> s) & 0xFF) * k, 0, 255)) << s;
    return out | (c & 0xFF);
}

// ------------------------------------------------------------------ builder
MeshB *mb_new(float unit) {
    MeshB *mb = calloc(1, sizeof(MeshB));
    mb->unit = unit;
    mb->cv = 256; mb->v = malloc(sizeof(BVert) * mb->cv);
    mb->ct = 256; mb->ti = malloc(sizeof(uint16_t) * 3 * mb->ct);
    mb_identity(mb);
    return mb;
}
void mb_box_bottoms(MeshB *mb, bool on) { mb->boxBottoms = on; }
void mb_free(MeshB *mb) { if (!mb) return; free(mb->v); free(mb->ti); free(mb); }
int mb_tris(MeshB *mb) { return mb->nt; }
void mb_identity(MeshB *mb) { mb_xform(mb, 0, 0, 0, 0, 0, 0, 1, 1, 1); }

static void rot_xyz(float out[3][3], float rx, float ry, float rz) {
    // R = Rx * Ry * Rz (same as three.js 'XYZ' Euler order), row-major out[row][col]
    float cx = cosf(rx), sx = sinf(rx), cy = cosf(ry), sy = sinf(ry), cz = cosf(rz), sz = sinf(rz);
    float Rx[3][3] = {{1, 0, 0}, {0, cx, -sx}, {0, sx, cx}};
    float Ry[3][3] = {{cy, 0, sy}, {0, 1, 0}, {-sy, 0, cy}};
    float Rz[3][3] = {{cz, -sz, 0}, {sz, cz, 0}, {0, 0, 1}};
    float t[3][3];
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) { t[i][j] = 0; for (int k = 0; k < 3; k++) t[i][j] += Ry[i][k] * Rz[k][j]; }
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) { out[i][j] = 0; for (int k = 0; k < 3; k++) out[i][j] += Rx[i][k] * t[k][j]; }
}
void mb_xform(MeshB *mb, float px, float py, float pz, float rx, float ry, float rz, float sx, float sy, float sz) {
    float R[3][3]; rot_xyz(R, rx, ry, rz);
    for (int i = 0; i < 3; i++) { mb->r[i][0] = R[i][0] * sx; mb->r[i][1] = R[i][1] * sy; mb->r[i][2] = R[i][2] * sz; }
    mb->t[0] = px; mb->t[1] = py; mb->t[2] = pz;
}
int mb_vert(MeshB *mb, float x, float y, float z, float nx, float ny, float nz, uint32_t rgba) {
    if (mb->nv == mb->cv) { mb->cv *= 2; mb->v = realloc(mb->v, sizeof(BVert) * mb->cv); }
    BVert *v = &mb->v[mb->nv];
    for (int i = 0; i < 3; i++) v->p[i] = mb->r[i][0] * x + mb->r[i][1] * y + mb->r[i][2] * z + mb->t[i];
    float n[3];
    for (int i = 0; i < 3; i++) n[i] = mb->r[i][0] * nx + mb->r[i][1] * ny + mb->r[i][2] * nz;
    float l = sqrtf(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]); if (l < 1e-6f) l = 1;
    v->n = t3d_vert_pack_normal(&(T3DVec3){{n[0] / l, n[1] / l, n[2] / l}});
    v->c = rgba;
    return mb->nv++;
}
void mb_tri(MeshB *mb, int a, int b, int c) {
    if (mb->nt == mb->ct) { mb->ct *= 2; mb->ti = realloc(mb->ti, sizeof(uint16_t) * 3 * mb->ct); }
    mb->ti[mb->nt * 3] = a; mb->ti[mb->nt * 3 + 1] = b; mb->ti[mb->nt * 3 + 2] = c; mb->nt++;
}
// Quad with a flat normal computed from its corners (counter-clockwise seen from the front)
void mb_quad_raw(MeshB *mb, const float p[4][3], uint32_t col) {
    float ux = p[1][0] - p[0][0], uy = p[1][1] - p[0][1], uz = p[1][2] - p[0][2];
    float vx = p[2][0] - p[0][0], vy = p[2][1] - p[0][1], vz = p[2][2] - p[0][2];
    float nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    int i0 = mb_vert(mb, p[0][0], p[0][1], p[0][2], nx, ny, nz, col);
    int i1 = mb_vert(mb, p[1][0], p[1][1], p[1][2], nx, ny, nz, col);
    int i2 = mb_vert(mb, p[2][0], p[2][1], p[2][2], nx, ny, nz, col);
    int i3 = mb_vert(mb, p[3][0], p[3][1], p[3][2], nx, ny, nz, col);
    mb_tri(mb, i0, i1, i2); mb_tri(mb, i2, i3, i0);
}
void mb_box(MeshB *mb, float cx, float cy, float cz, float w, float h, float d, uint32_t col) {
    float x0 = cx - w / 2, x1 = cx + w / 2, y0 = cy - h / 2, y1 = cy + h / 2, z0 = cz - d / 2, z1 = cz + d / 2;
    uint32_t top = col_shade(col, 1.06f), side = col, bot = col_shade(col, 0.8f);
    mb_quad_raw(mb, (const float[4][3]){{x0, y1, z1}, {x1, y1, z1}, {x1, y1, z0}, {x0, y1, z0}}, top);   // +y
    if (mb->boxBottoms) mb_quad_raw(mb, (const float[4][3]){{x0, y0, z0}, {x1, y0, z0}, {x1, y0, z1}, {x0, y0, z1}}, bot);   // -y (rarely visible)
    mb_quad_raw(mb, (const float[4][3]){{x0, y0, z1}, {x1, y0, z1}, {x1, y1, z1}, {x0, y1, z1}}, side);  // +z
    mb_quad_raw(mb, (const float[4][3]){{x1, y0, z0}, {x0, y0, z0}, {x0, y1, z0}, {x1, y1, z0}}, side);  // -z
    mb_quad_raw(mb, (const float[4][3]){{x1, y0, z1}, {x1, y0, z0}, {x1, y1, z0}, {x1, y1, z1}}, side);  // +x
    mb_quad_raw(mb, (const float[4][3]){{x0, y0, z0}, {x0, y0, z1}, {x0, y1, z1}, {x0, y1, z0}}, side);  // -x
}
void mb_cyl(MeshB *mb, float cx, float cy, float cz, float rTop, float rBot, float h, int seg, uint32_t col, bool caps) {
    float y0 = cy - h / 2, y1 = cy + h / 2;
    for (int i = 0; i < seg; i++) {
        float a0 = (float)i / seg * TAU_F, a1 = (float)(i + 1) / seg * TAU_F;
        float c0 = cosf(a0), s0 = sinf(a0), c1 = cosf(a1), s1 = sinf(a1);
        uint32_t c = col_shade(col, 0.92f + 0.12f * ((i & 1) ? 1 : 0));
        if (rTop < 1e-4f) {
            int v0 = mb_vert(mb, cx + c0 * rBot, y0, cz + s0 * rBot, c0, 0.5f, s0, c);
            int v1 = mb_vert(mb, cx + c1 * rBot, y0, cz + s1 * rBot, c1, 0.5f, s1, c);
            int v2 = mb_vert(mb, cx, y1, cz, (c0 + c1) / 2, 0.6f, (s0 + s1) / 2, col_shade(col, 1.1f));
            mb_tri(mb, v0, v2, v1);
        } else {
            mb_quad_raw(mb, (const float[4][3]){{cx + c0 * rBot, y0, cz + s0 * rBot}, {cx + c0 * rTop, y1, cz + s0 * rTop},
                                                {cx + c1 * rTop, y1, cz + s1 * rTop}, {cx + c1 * rBot, y0, cz + s1 * rBot}}, c);
        }
    }
    if (caps && rTop > 1e-4f) {
        uint32_t tc = col_shade(col, 1.08f);
        int ctr = mb_vert(mb, cx, y1, cz, 0, 1, 0, tc);
        for (int i = 0; i < seg; i++) {
            float a0 = (float)i / seg * TAU_F, a1 = (float)(i + 1) / seg * TAU_F;
            int v0 = mb_vert(mb, cx + cosf(a0) * rTop, y1, cz + sinf(a0) * rTop, 0, 1, 0, tc);
            int v1 = mb_vert(mb, cx + cosf(a1) * rTop, y1, cz + sinf(a1) * rTop, 0, 1, 0, tc);
            mb_tri(mb, ctr, v1, v0);
        }
    }
}
void mb_disc(MeshB *mb, float cx, float cy, float cz, float r, int seg, uint32_t col) {
    int ctr = mb_vert(mb, cx, cy, cz, 0, 1, 0, col);
    for (int i = 0; i < seg; i++) {
        float a0 = (float)i / seg * TAU_F, a1 = (float)(i + 1) / seg * TAU_F;
        int v0 = mb_vert(mb, cx + cosf(a0) * r, cy, cz + sinf(a0) * r, 0, 1, 0, col);
        int v1 = mb_vert(mb, cx + cosf(a1) * r, cy, cz + sinf(a1) * r, 0, 1, 0, col);
        mb_tri(mb, ctr, v1, v0);
    }
}

// ------------------------------------------------------------------ baking
#define BATCH_MAX 64
Mesh mb_bake(MeshB *mb) {
    Mesh out = {0};
    out.tris = mb->nt;
    if (mb->nt == 0) return out;
    int *map = malloc(sizeof(int) * mb->nv);
    for (int i = 0; i < mb->nv; i++) map[i] = -1;
    // pass 1: split into batches, assign local indices
    int cap = mb->nt * 3 + 64;
    int *bvGlobal = malloc(sizeof(int) * cap);   // global vertex per output slot
    uint8_t *loc = malloc(mb->nt * 3);
    int *batchStart = malloc(sizeof(int) * (mb->nt + 1)), *batchCount = malloc(sizeof(int) * (mb->nt + 1)), *batchTri0 = malloc(sizeof(int) * (mb->nt + 1));
    int nb = 0, slot = 0, cur = 0, curStart = 0;
    batchTri0[0] = 0;
    for (int t = 0; t < mb->nt; t++) {
        int need = 0;
        for (int k = 0; k < 3; k++) if (map[mb->ti[t * 3 + k]] < 0) need++;
        if (cur + need > BATCH_MAX) {
            if (cur & 1) { bvGlobal[slot] = bvGlobal[slot - 1]; slot++; cur++; }
            for (int s = curStart; s < curStart + cur; s++) map[bvGlobal[s]] = -1;
            batchStart[nb] = curStart; batchCount[nb] = cur; nb++;
            batchTri0[nb] = t; curStart = slot; cur = 0;
        }
        for (int k = 0; k < 3; k++) {
            int g = mb->ti[t * 3 + k];
            if (map[g] < 0) { map[g] = cur; bvGlobal[slot++] = g; cur++; }
            loc[t * 3 + k] = map[g];
        }
    }
    if (cur & 1) { bvGlobal[slot] = bvGlobal[slot - 1]; slot++; cur++; }
    batchStart[nb] = curStart; batchCount[nb] = cur; nb++;
    batchTri0[nb] = mb->nt;
    // pass 2: pack vertices
    out.verts = malloc_uncached(sizeof(T3DVertPacked) * (slot / 2));
    for (int s = 0; s < slot; s += 2) {
        BVert *a = &mb->v[bvGlobal[s]], *b = &mb->v[bvGlobal[s + 1]];
        T3DVertPacked *p = &out.verts[s / 2];
        for (int i = 0; i < 3; i++) { p->posA[i] = (int16_t)lrintf(a->p[i] * mb->unit); p->posB[i] = (int16_t)lrintf(b->p[i] * mb->unit); }
        p->normA = a->n; p->normB = b->n; p->rgbaA = a->c; p->rgbaB = b->c;
        p->stA[0] = p->stA[1] = p->stB[0] = p->stB[1] = 0;
    }
    // pass 3: record display list
    rspq_block_begin();
    for (int b = 0; b < nb; b++) {
        t3d_vert_load(&out.verts[batchStart[b] / 2], 0, batchCount[b]);
        for (int t = batchTri0[b]; t < batchTri0[b + 1]; t++) t3d_tri_draw(loc[t * 3], loc[t * 3 + 1], loc[t * 3 + 2]);
        t3d_tri_sync();
    }
    out.block = rspq_block_end();
    free(map); free(bvGlobal); free(loc); free(batchStart); free(batchCount); free(batchTri0);
    return out;
}
void mesh_draw(const Mesh *m) { if (m->block) rspq_block_run(m->block); }

// ------------------------------------------------------------------ draw matrices
// Column-major 4x4: m[col][row]
void m4_trs(M4 *m, float px, float py, float pz, float rx, float ry, float rz, float s) {
    float R[3][3]; rot_xyz(R, rx, ry, rz);
    for (int c = 0; c < 3; c++) { for (int r = 0; r < 3; r++) m->m[c][r] = R[r][c] * s; m->m[c][3] = 0; }
    m->m[3][0] = px; m->m[3][1] = py; m->m[3][2] = pz; m->m[3][3] = 1;
}
void m4_mul(M4 *out, const M4 *a, const M4 *b) { M4 t; t3d_mat4_mul(&t, a, b); *out = t; }
void m4_scale_cols(M4 *m, float s) { for (int c = 0; c < 3; c++) for (int r = 0; r < 3; r++) m->m[c][r] *= s; }

#define MAT_POOL 640
static T3DMat4FP *s_pool[FB_COUNT];
static int s_poolIdx, s_frame;
T3DMat4FP *mat_alloc(void) {
    if (!s_pool[0]) for (int i = 0; i < FB_COUNT; i++) s_pool[i] = malloc_uncached(sizeof(T3DMat4FP) * MAT_POOL);
    if (s_poolIdx >= MAT_POOL) s_poolIdx = MAT_POOL - 1;  // overflow guard: reuse last slot
    return &s_pool[s_frame][s_poolIdx++];
}
void mat_frame_begin(int frameIdx) { if (!s_pool[0]) mat_alloc(); s_frame = frameIdx % FB_COUNT; s_poolIdx = 0; }
void draw_mesh_at(const Mesh *mesh, const M4 *m) {
    if (!mesh->block) return;
    T3DMat4FP *fp = mat_alloc();
    t3d_mat4_to_fixed(fp, m);
    t3d_matrix_push(fp);
    rspq_block_run(mesh->block);
    t3d_matrix_pop(1);
}
