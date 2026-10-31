// Level geometry, terrain and collision for Tablet V.
// +z is south (the city of Uruk), -z is north (the Cedar Forest).
#include "game.h"

// ------------------------------------------------------------------ terrain height
static const float FLAT_ZONES[][3] = {
    {20, 20, 9}, {-20, 1, 9}, {14, -10, 10}, {ARENA_X, ARENA_Z, 21}, {0, 38, 9}, {-20, 32, 10}, {0, GATE_Z, 7},
};
static float path_x(float z) { return sinf(z * 0.08f) * 4.0f; }

float height_at(float x, float z) {
    float h = 0.45f * sinf(x * 0.13f + 1.3f) * cosf(z * 0.11f) + 0.25f * sinf(z * 0.21f + x * 0.05f);
    float f = 1;
    for (unsigned i = 0; i < sizeof(FLAT_ZONES) / sizeof(FLAT_ZONES[0]); i++) {
        float dx = x - FLAT_ZONES[i][0], dz = z - FLAT_ZONES[i][1];
        float d = sqrtf(dx * dx + dz * dz);
        f = minf(f, smoothf(clampf(d / FLAT_ZONES[i][2], 0, 1)));
    }
    h *= f;
    float dOx = x - 20, dOz = z - 20, dOas = sqrtf(dOx * dOx + dOz * dOz);
    if (dOas < 3.8f) h -= 0.7f * (1 - dOas / 3.8f);
    float dx = maxf(maxf(BX0 - x, 0), x - BX1), dz = maxf(maxf(BZ0 - z, 0), z - BZ1);
    float out = sqrtf(dx * dx + dz * dz);
    if (out > 0) h += out * out * 0.07f + out * 0.5f + sinf(x * 0.4f) * cosf(z * 0.37f) * minf(out, 4) * 0.4f;
    return h;
}

// ------------------------------------------------------------------ collision
typedef struct { bool box; float x, z, r; float x0, x1, z0, z1; bool on; float canopy; } Coll;
#define MAX_COLL 360
static Coll s_coll[MAX_COLL];
static int s_ncoll;
static int s_doorColl = -1;
static void add_circle(float x, float z, float r) { if (s_ncoll < MAX_COLL) s_coll[s_ncoll++] = (Coll){false, x, z, r, 0, 0, 0, 0, true, 0}; }
static int add_box(float x0, float x1, float z0, float z1) { if (s_ncoll < MAX_COLL) s_coll[s_ncoll++] = (Coll){true, 0, 0, 0, x0, x1, z0, z1, true, 0}; return s_ncoll - 1; }
void world_set_door_collider(bool on) { if (s_doorColl >= 0) s_coll[s_doorColl].on = on; }

extern void pots_resolve(V3 *pos, float r, bool *moved);
bool resolve(V3 *pos, float r) {
    bool moved = false;
    for (int i = 0; i < s_ncoll; i++) {
        Coll *c = &s_coll[i];
        if (!c->on) continue;
        if (!c->box) {
            float dx = pos->x - c->x, dz = pos->z - c->z, R = c->r + r, q = dx * dx + dz * dz;
            if (q < R * R) { float d = sqrtf(q); if (d < 1e-4f) { d = 1e-4f; dx = 1e-4f; } pos->x = c->x + dx / d * R; pos->z = c->z + dz / d * R; moved = true; }
            continue;
        }
        float cx = clampf(pos->x, c->x0, c->x1), cz = clampf(pos->z, c->z0, c->z1);
        float dx = pos->x - cx, dz = pos->z - cz, q = dx * dx + dz * dz;
        if (q < r * r) {
            if (q > 1e-8f) { float d = sqrtf(q); pos->x = cx + dx / d * r; pos->z = cz + dz / d * r; }
            else {
                float l = pos->x - c->x0, rr = c->x1 - pos->x, b = pos->z - c->z0, t = c->z1 - pos->z;
                float m = minf(minf(l, rr), minf(b, t));
                if (m == l) pos->x = c->x0 - r; else if (m == rr) pos->x = c->x1 + r; else if (m == b) pos->z = c->z0 - r; else pos->z = c->z1 + r;
            }
            moved = true;
        }
    }
    pots_resolve(pos, r, &moved);
    float ox = pos->x, oz = pos->z;
    pos->x = clampf(pos->x, BX0 + 0.5f, BX1 - 0.5f);
    pos->z = clampf(pos->z, BZ0 + 0.5f, BZ1 - 1.0f);
    return moved || ox != pos->x || oz != pos->z;
}

// Keep the camera out of tree trunks and, above trunk height, out of the cedar canopies.
void camera_push(V3 *c) {
    for (int i = 0; i < s_ncoll; i++) {
        Coll *k = &s_coll[i];
        if (k->box || !k->on) continue;
        float r = k->r + 0.6f;
        if (k->canopy > 0 && c->y > height_at(k->x, k->z) + 2.0f) r = k->canopy;
        float dx = c->x - k->x, dz = c->z - k->z, q = dx * dx + dz * dz;
        if (q < r * r) { float d = sqrtf(q); if (d < 1e-3f) { d = 1e-3f; dx = 1e-3f; } c->x = k->x + dx / d * r; c->z = k->z + dz / d * r; }
    }
}

// ------------------------------------------------------------------ chunked static geometry
#define CH_SIZE 20.0f
#define CH_X0 (-60.0f)
#define CH_Z0 (-100.0f)
#define CH_NX 6
#define CH_NZ 9
static MeshB *s_cb[CH_NX][CH_NZ];
static Mesh s_chunk[CH_NX][CH_NZ];
static Mesh s_sky, s_doorMesh[2];
static float s_doorAngle;

static MeshB *cb(float x, float z) {
    int ix = (int)floorf((x - CH_X0) / CH_SIZE), iz = (int)floorf((z - CH_Z0) / CH_SIZE);
    ix = ix < 0 ? 0 : (ix >= CH_NX ? CH_NX - 1 : ix);
    iz = iz < 0 ? 0 : (iz >= CH_NZ ? CH_NZ - 1 : iz);
    if (!s_cb[ix][iz]) s_cb[ix][iz] = mb_new(RS);
    return s_cb[ix][iz];
}
// Set the builder transform for an object placed at (x, y, z) with yaw ry
static MeshB *at(float x, float y, float z, float ry, float s) { MeshB *m = cb(x, z); mb_xform(m, x, y, z, 0, ry, 0, s, s, s); return m; }

// palette
#define C_SAND     RGB(0xD9B27C)
#define C_SAND2    RGB(0xC99E66)
#define C_GRASS    RGB(0x6F9A45)
#define C_FOREST   RGB(0x46703A)
#define C_FOREST2  RGB(0x37592C)
#define C_ROCK     RGB(0xA47E58)
#define C_DIRT     RGB(0xB98A5A)
#define C_CLEAR    RGB(0x6A8C42)
#define C_BRICK    RGB(0xC49A6A)
#define C_BRICK_D  RGB(0x9A7448)
#define C_LAPIS    RGB(0x2346B0)
#define C_GOLD     RGB(0xE2B54A)
#define C_STONE    RGB(0xC8B8A0)
#define C_STONE_D  RGB(0x8F8270)
#define C_TRUNK    RGB(0x8A6A44)
#define C_FROND    RGB(0x4F8A2E)
#define C_FROND2   RGB(0x6D9A35)
#define C_CEDAR_T  RGB(0x6A4A30)
#define C_WOOD     RGB(0x8A5A30)
#define C_BRONZE   RGB(0xC98D3A)
#define C_CLOTH_R  RGB(0xA8331F)
#define C_CLOTH_W  RGB(0xEFE0BD)
#define C_CLOTH_B  RGB(0x2B4A8A)
#define C_WATER    RGB(0x3A78B8)
#define C_CUNE     RGB(0xCDBB98)

static uint32_t ground_color(float x, float z) {
    uint32_t c = col_lerp(C_SAND, C_SAND2, 0.5f + 0.5f * sinf(x * 0.3f) * cosf(z * 0.23f));
    float ft = smoothf(clampf((-z - 18) / 8, 0, 1));
    c = col_lerp(c, ((int)(x * 7 + z * 3) & 1) ? C_FOREST : C_FOREST2, ft);
    float dO = sqrtf((x - 20) * (x - 20) + (z - 20) * (z - 20));
    if (dO < 10) c = col_lerp(c, C_GRASS, smoothf(1 - dO / 10));
    float dA = sqrtf((x - ARENA_X) * (x - ARENA_X) + (z - ARENA_Z) * (z - ARENA_Z));
    if (dA < ARENA_R) c = col_lerp(c, C_CLEAR, 0.6f * smoothf(1 - dA / ARENA_R));
    if (z > -24 && z < 40 && fabsf(x - path_x(z)) < 2.2f) c = col_lerp(c, C_DIRT, 0.7f);
    float out = maxf(maxf(BX0 - x, x - BX1), maxf(BZ0 - z, z - BZ1));
    if (out > 0) c = col_lerp(c, ft > 0.5f ? C_FOREST2 : C_ROCK, smoothf(clampf(out / 6, 0, 1)) * (ft > 0.5f ? 0.6f : 1));
    return c;
}

static void build_terrain(void) {
    const float cell = 4.0f;
    for (int ix = 0; ix < CH_NX; ix++) for (int iz = 0; iz < CH_NZ; iz++) {
        float x0 = CH_X0 + ix * CH_SIZE, z0 = CH_Z0 + iz * CH_SIZE;
        MeshB *m = cb(x0 + 1, z0 + 1); mb_identity(m);
        int n = (int)(CH_SIZE / cell);
        int idx[6][6];
        for (int i = 0; i <= n; i++) for (int j = 0; j <= n; j++) {
            float x = x0 + i * cell, z = z0 + j * cell, h = height_at(x, z);
            float e = 0.5f, hx = height_at(x + e, z) - height_at(x - e, z), hz = height_at(x, z + e) - height_at(x, z - e);
            idx[i][j] = mb_vert(m, x, h, z, -hx, 2 * e, -hz, ground_color(x, z));
        }
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
            mb_tri(m, idx[i][j], idx[i][j + 1], idx[i + 1][j + 1]);
            mb_tri(m, idx[i + 1][j + 1], idx[i + 1][j], idx[i][j]);
        }
    }
}

// ------------------------------------------------------------------ props
static void palm(float x, float z, float s) {
    float y = height_at(x, z) - 0.1f, dir = srand01() * TAU_F, lean = 0.25f + srand01() * 0.3f;
    MeshB *m = at(x, y, z, 0, s);
    float tx = 0, tz = 0, ty = 0;
    for (int i = 0; i < 3; i++) {
        float off = lean * i * i * 0.18f;
        tx = sinf(dir) * off; tz = cosf(dir) * off; ty = 0.95f + i * 1.9f;
        mb_cyl(m, tx, ty, tz, 0.17f, 0.22f, 1.95f, 4, col_shade(C_TRUNK, 0.9f + 0.1f * (i & 1)), false);
    }
    float top = ty + 0.6f;
    for (int i = 0; i < 7; i++) {
        float a = i / 7.0f * TAU_F + srand01() * 0.3f, ca = cosf(a), sa = sinf(a), droop = 0.9f + srand01() * 0.5f;
        float bx = tx + sa * 0.2f, bz = tz + ca * 0.2f;
        float ex = tx + sa * 2.4f, ez = tz + ca * 2.4f, ey = top - droop;
        float wx = ca * 0.35f, wz = -sa * 0.35f;
        uint32_t c = (i & 1) ? C_FROND : C_FROND2;
        float p[4][3] = {{bx - wx * 0.4f, top, bz - wz * 0.4f}, {ex - wx, ey, ez - wz}, {ex + wx, ey, ez + wz}, {bx + wx * 0.4f, top, bz + wz * 0.4f}};
        mb_quad_raw(m, p, c);
        float q[4][3] = {{p[3][0], p[3][1], p[3][2]}, {p[2][0], p[2][1], p[2][2]}, {p[1][0], p[1][1], p[1][2]}, {p[0][0], p[0][1], p[0][2]}};
        mb_quad_raw(m, q, c);
    }
    add_circle(x, z, 0.4f * s);
}
static void cedar(float x, float z, float s, bool collide) {
    float y = height_at(x, z) - 0.2f;
    MeshB *m = at(x, y, z, srand01() * TAU_F, s);
    mb_cyl(m, 0, 2.5f, 0, 0.28f, 0.45f, 5.0f, 4, C_CEDAR_T, false);
    static const uint32_t tiers[3] = {RGB(0x2F5F2C), RGB(0x3A6E33), RGB(0x284F27)};
    for (int i = 0; i < 2; i++) {   // two flat tiers: the cedar-of-Lebanon silhouette at low cost
        float r = 3.2f - i * 1.3f + srand01() * 0.4f;
        mb_cyl(m, (srand01() - 0.5f) * 0.5f, 2.8f + i * 1.9f, (srand01() - 0.5f) * 0.5f, r * 0.85f, r, 0.7f, 5, tiers[(i + (int)(srand01() * 3)) % 3], true);
    }
    mb_cyl(m, 0, 2.8f + 2 * 1.9f + 0.3f, 0, 0, 0.9f, 1.4f, 4, tiers[0], false);
    if (collide) { add_circle(x, z, 0.55f * s); s_coll[s_ncoll - 1].canopy = 2.9f * s; }
}
static void rock(float x, float z, float s, bool collide) {
    MeshB *m = cb(x, z);
    mb_xform(m, x, height_at(x, z) + 0.1f * s, z, srand01(), srand01() * TAU_F, srand01() * 0.5f, s * (0.8f + srand01() * 0.5f), s * (0.5f + srand01() * 0.4f), s * (0.8f + srand01() * 0.5f));
    mb_cyl(m, 0, 0, 0, 0.35f, 0.6f, 0.8f, 4, srand01() < 0.5f ? C_STONE : C_STONE_D, true);
    if (collide && s > 0.9f) add_circle(x, z, 0.5f * s);
}

Stele g_steles[2];
static void stele(int i, float x, float z, float ry) {
    MeshB *m = at(x, height_at(x, z), z, ry, 1);
    mb_box(m, 0, 1.0f, 0, 1.1f, 2.0f, 0.35f, C_CUNE);
    for (int r = 0; r < 6; r++) mb_box(m, 0, 0.4f + r * 0.25f, 0.18f, 0.8f, 0.05f, 0.02f, RGB(0x6E5A3E));  // carved lines
    mb_xform(m, x, height_at(x, z) + 2.0f, z, PI_F / 2, ry, 0, 1, 1, 1);
    mb_cyl(m, 0, 0, 0, 0.55f, 0.55f, 0.35f, 8, C_CUNE, true);
    m = at(x, height_at(x, z), z, ry, 1);
    mb_box(m, 0, 0.12f, 0, 1.4f, 0.25f, 0.6f, C_STONE_D);
    add_circle(x, z, 0.6f);
    g_steles[i].pos = v3(x, 0, z);
}
static float pedestal(float x, float z) {
    MeshB *m = at(x, height_at(x, z), z, 0, 1);
    mb_box(m, 0, 0.1f, 0, 0.9f, 0.2f, 0.9f, C_STONE_D);
    mb_box(m, 0, 0.6f, 0, 0.65f, 1.0f, 0.65f, C_STONE);
    mb_box(m, 0, 1.17f, 0, 0.95f, 0.15f, 0.95f, C_STONE_D);
    add_circle(x, z, 0.5f);
    return 1.25f;
}

static void wall_run(float x0, float x1, float zc, float h, float d, float step) {
    for (float x = x0; x < x1 - 0.01f; x += step) {
        float w = minf(step, x1 - x);
        MeshB *m = at(x + w / 2, 0, zc, 0, 1);
        mb_box(m, 0, h / 2 - 0.6f, 0, w, h + 1.2f, d, C_BRICK);
        for (float bx = -w / 2 + 1.2f; bx < w / 2; bx += 5.0f) mb_box(m, bx, h + 0.4f, 0, 1.8f, 1.0f, d, C_BRICK);
    }
}

static void build_props(void) {
    // ---- the walls of Uruk with a glazed lapis gate
    wall_run(-46, -7, 45, 7.5f, 3, 13);
    wall_run(7, 46, 45, 7.5f, 3, 13);
    for (int i = 0; i < 4; i++) { float x = (float[]){-38, -24, 24, 38}[i]; MeshB *m = at(x, 0, 45, 0, 1); mb_box(m, 0, 4.4f, 0, 5, 9.5f, 4.5f, C_BRICK); mb_box(m, 0, 9.6f, 0, 5.4f, 0.8f, 4.9f, C_BRICK_D); }
    for (int s = -1; s <= 1; s += 2) {
        MeshB *m = at(s * 5.0f, 0, 44.2f, 0, 1);
        mb_box(m, 0, 5.5f, 0, 4, 12, 5, C_LAPIS);
        for (int r = 0; r < 4; r++) mb_box(m, 0, 2.5f + r * 2.6f, -2.52f, 3.0f, 0.3f, 0.05f, C_GOLD);
        mb_box(m, 0, 11.7f, 0, 4.6f, 1, 5.4f, C_GOLD);
    }
    { MeshB *m = at(0, 0, 44.2f, 0, 1); mb_box(m, 0, 9.5f, 0, 6.2f, 3, 4.6f, C_LAPIS); mb_box(m, 0, 4, -1.6f, 6, 8, 0.5f, C_WOOD); mb_box(m, 0, 8.1f, -1.7f, 6.4f, 0.5f, 0.6f, C_GOLD); }
    add_box(-46, 46, 42.6f, 48);

    // ---- ziggurat of Anu
    {
        float zx = -20, zz = 32;
        MeshB *m = at(zx, 0, zz, 0, 1);
        mb_box(m, 0, 1.4f, 0, 12, 3, 10, C_BRICK);
        mb_box(m, 0, 4.4f, 0, 9, 3, 7.5f, col_shade(C_BRICK, 0.95f));
        mb_box(m, 0, 7.4f, 0, 6, 3, 5, C_BRICK);
        mb_box(m, 0, 10.1f, 0, 3, 2.4f, 2.6f, col_shade(C_BRICK, 1.05f));
        mb_box(m, 0, 11.5f, 0, 3.4f, 0.5f, 3, C_LAPIS);
        mb_xform(m, zx + 7.8f, 1.5f, zz, 0, 0, 0.55f, 1, 1, 1); mb_box(m, 0, 0, 0, 4.5f, 0.6f, 2.2f, C_BRICK_D);
        mb_xform(m, zx + 5.4f, 4.5f, zz, 0, 0, 0.55f, 1, 1, 1); mb_box(m, 0, 0, 0, 3.2f, 0.6f, 2.2f, C_BRICK_D);
        add_box(zx - 6, zx + 6, zz - 5, zz + 5); add_box(zx + 6, zx + 9.5f, zz - 1.2f, zz + 1.2f);
    }

    // ---- palms
    static const float palms[][3] = {{8, 30, 1.1f}, {12, 35, 0.9f}, {-8, 24, 1}, {25, 29, 1.2f}, {26, 12, 1}, {15, 27, 0.9f}, {25, 23, 1.1f}, {17, 14, 1},
                                     {-27, 19, 1.1f}, {-11, 40, 1}, {10, 40, 1.2f}, {27, -4, 1}, {-27, -14, 1}, {-2, 18, 0.9f}, {27, 38, 1.1f}, {-28, 40, 1}};
    for (unsigned i = 0; i < sizeof(palms) / sizeof(palms[0]); i++) palm(palms[i][0], palms[i][1], palms[i][2]);
    for (int i = 0; i < 18; i++) { float x = BX0 + srand01() * 60, z = -18 + srand01() * 58; if (fabsf(x - path_x(z)) > 3 && sqrtf((x - 20) * (x - 20) + (z - 20) * (z - 20)) > 5) rock(x, z, 0.5f + srand01() * 0.8f, true); }
    for (int i = 0; i < 12; i++) { float side = srand01() < 0.5f ? -1 : 1; rock(side * (31 + srand01() * 4), -20 + srand01() * 64, 1.6f + srand01() * 2, false); }

    // ---- oasis well (Seal of Dawn)
    { MeshB *m = cb(20, 20); mb_identity(m); mb_disc(m, 20, -0.25f, 20, 3.5f, 12, C_WATER); }
    for (int i = 0; i < 12; i++) { float a = i / 12.0f * TAU_F; MeshB *m = at(20 + cosf(a) * 3.7f, 0.1f, 20 + sinf(a) * 3.7f, -a + PI_F / 2, 1); mb_box(m, 0, 0, 0, 1.8f, 0.5f, 0.7f, C_STONE); }
    add_circle(20, 20, 3.9f);

    // ---- scorpion-men camp (Seal of Noon)
    for (int t = 0; t < 2; t++) {
        float x = t ? -16.5f : -24.5f, z = t ? -4.5f : 6.5f, y = height_at(x, z);
        MeshB *m = at(x, y, z, PI_F / 4 + t, 1);
        for (int band = 0; band < 3; band++) {
            float r0 = 2.1f * (1 - band / 3.0f), r1 = 2.1f * (1 - (band + 1) / 3.0f);
            mb_cyl(m, 0, 0.47f + band * 0.93f, 0, r1 < 0.05f ? 0 : r1, r0, 0.93f, 4, band == 0 ? C_CLOTH_R : band == 1 ? C_CLOTH_W : C_CLOTH_B, false);
        }
        mb_cyl(m, 0, 1.8f, 0, 0.05f, 0.05f, 3.6f, 4, C_TRUNK, false);
        add_circle(x, z, 1.7f);
    }
    {
        float fx = -20, fz = 1; MeshB *m = at(fx, 0, fz, 0, 1);
        for (int i = 0; i < 3; i++) { mb_xform(m, fx, 0.12f, fz, 0, i * 1.05f, 0, 1, 1, 1); mb_box(m, 0, 0, 0, 1.2f, 0.18f, 0.18f, C_TRUNK); }
        mb_xform(m, fx, 0, fz, 0, 0, 0, 1, 1, 1);
        for (int i = 0; i < 8; i++) { float a = i / 8.0f * TAU_F; mb_box(m, cosf(a) * 0.75f, 0.1f, sinf(a) * 0.75f, 0.35f, 0.25f, 0.35f, C_STONE_D); }
        mb_cyl(m, 0, 0.65f, 0, 0, 0.42f, 1.1f, 5, RGB(0xFF7A20), false);
        mb_cyl(m, 0, 0.55f, 0, 0, 0.24f, 0.8f, 5, RGB(0xFFE060), false);
        add_circle(fx, fz, 0.8f);
    }

    // ---- ruined shrine of Shamash
    static const float cols[][3] = {{-6, 12, 3.5f}, {-9.5f, 14, 2.2f}, {-6, 16.5f, 4.2f}, {-11, 10, 1.2f}};
    for (int i = 0; i < 4; i++) { MeshB *m = at(cols[i][0], 0, cols[i][1], 0, 1); mb_cyl(m, 0, cols[i][2] / 2, 0, 0.5f, 0.55f, cols[i][2], 7, C_STONE, true); mb_box(m, 0, 0.15f, 0, 1.3f, 0.3f, 1.3f, C_STONE_D); add_circle(cols[i][0], cols[i][1], 0.65f); }
    { MeshB *m = cb(-8.5f, 8); mb_xform(m, -8.5f, 0.5f, 8, 0, 0.6f, PI_F / 2, 1, 1, 1); mb_cyl(m, 0, 0, 0, 0.5f, 0.5f, 4, 7, C_STONE, true); add_circle(-9.4f, 7.4f, 0.6f); add_circle(-7.6f, 8.6f, 0.6f); }

    // ---- ring of standing stones (Seal of Dusk)
    for (int i = 0; i < 9; i++) {
        if (i == 3) continue;
        float a = i / 9.0f * TAU_F, x = 14 + cosf(a) * 7, z = -10 + sinf(a) * 7;
        MeshB *m = at(x, height_at(x, z), z, -a + PI_F / 2, 1);
        mb_box(m, 0, 1.3f, 0, 1.1f, 2.8f, 0.8f, C_STONE);
        if (i % 3 == 0) mb_box(m, 0, 2.95f, 0, 1.3f, 0.45f, 0.95f, C_STONE_D);
        add_circle(x, z, 0.7f);
    }
    { MeshB *m = at(14, 0, -10, 0, 1); mb_box(m, 0, 0.45f, 0, 1.5f, 0.9f, 1.5f, C_CUNE); add_circle(14, -10, 0.9f); }

    // ---- wall and cedar door of the forest
    wall_run(-46, -4.8f, GATE_Z, 9, 2.2f, 10.3f);
    wall_run(4.8f, 46, GATE_Z, 9, 2.2f, 10.3f);
    for (int s = -1; s <= 1; s += 2) { MeshB *m = at(s * 4.1f, 0, GATE_Z, 0, 1); mb_box(m, 0, 5.5f, 0, 1.8f, 11.5f, 2.8f, C_LAPIS); mb_box(m, 0, 11.4f, 0, 2.2f, 0.8f, 3.2f, C_GOLD); }
    {
        MeshB *m = at(0, 0, GATE_Z, 0, 1);
        mb_box(m, 0, 9.3f, 0, 8.4f, 2.2f, 2.4f, C_BRICK);
        // Humbaba's face carved above the door, as a warning
        mb_box(m, 0, 9.3f, 1.3f, 1.6f, 1.4f, 0.3f, RGB(0x9A7448));
        mb_box(m, -0.38f, 9.5f, 1.47f, 0.3f, 0.18f, 0.06f, RGB(0xFFCC33));
        mb_box(m, 0.38f, 9.5f, 1.47f, 0.3f, 0.18f, 0.06f, RGB(0xFFCC33));
        mb_box(m, 0, 8.95f, 1.47f, 0.9f, 0.25f, 0.06f, RGB(0x2A0A05));
    }
    add_box(-46, -3.2f, GATE_Z - 1.2f, GATE_Z + 1.2f);
    add_box(3.2f, 46, GATE_Z - 1.2f, GATE_Z + 1.2f);
    s_doorColl = add_box(-3.2f, 3.2f, GATE_Z - 0.5f, GATE_Z + 0.5f);
    for (int d = 0; d < 2; d++) {
        float side = d ? 1 : -1;
        MeshB *m = mb_new(RS);
        mb_box(m, -side * 1.6f, 3.9f, 0, 3.2f, 7.8f, 0.45f, C_WOOD);
        for (int r = 0; r < 4; r++) for (int c = 0; c < 3; c++) mb_box(m, -side * (0.6f + c * 1.0f), 1.2f + r * 1.8f, 0.27f, 0.18f, 0.18f, 0.12f, C_BRONZE);
        for (int k = 0; k < 3; k++) mb_box(m, -side * 1.6f, 1.0f + k * 2.9f, 0.25f, 3.2f, 0.22f, 0.06f, C_BRONZE);
        s_doorMesh[d] = mb_bake(m); mb_free(m);
    }

    // ---- the Cedar Forest
    for (int i = 0; i < 160; i++) {
        float x = BX0 + 1 + srand01() * 58, z = -74 + srand01() * 48;
        float ax = x - ARENA_X, az = z - ARENA_Z;
        if (sqrtf(ax * ax + az * az) < ARENA_R + 7.0f) continue;   // keep canopies out of the clearing
        if (fabsf(x) < 5 && z > ARENA_Z) continue;
        if (z > -26) continue;
        cedar(x, z, 0.85f + srand01() * 0.5f, true);
    }
    for (int i = 0; i < 24; i++) { float side = srand01() < 0.5f ? -1 : 1; cedar(side * (32 + srand01() * 14), -20 - srand01() * 60, 1 + srand01() * 0.8f, false); }
    for (int i = 0; i < 12; i++) cedar(-40 + srand01() * 80, -80 - srand01() * 14, 1.2f + srand01() * 0.8f, false);
    for (int i = 0; i < 14; i++) { float a = i / 14.0f * TAU_F; if (sinf(a) > 0.9f) continue; cedar(ARENA_X + cosf(a) * (ARENA_R + 7), ARENA_Z + sinf(a) * (ARENA_R + 7), 1.4f + srand01() * 0.3f, true); }
    cedar(0, -72.5f, 2.4f, true);  // the great cedar

    // ---- seal pedestals (Dawn by the oasis, Noon at the camp)
    pedestal(14.4f, 19.5f);
    pedestal(-25.5f, -1.5f);

    // ---- steles
    stele(0, 3.5f, 34, -0.3f);
    stele(1, 6.5f, -17, 0.25f);
}

static void build_sky(void) {
    MeshB *m = mb_new(1.0f);  // mountains are far away: 1 unit = 1 decimeter is plenty
    for (int i = 0; i < 26; i++) {
        float a = (i / 26.0f) * TAU_F + srand01() * 0.1f, r = 150 + srand01() * 20, h = 32 + srand01() * 40;
        bool north = sinf(a) < -0.3f;
        uint32_t c = north ? RGB(0x5A7656) : RGB(0xB59478);
        mb_xform(m, cosf(a) * r * RS, (h / 2 - 8) * RS, (sinf(a) * r - 15) * RS, 0, srand01() * TAU_F, 0, RS, RS, RS);
        mb_cyl(m, 0, 0, 0, 0, 18 + srand01() * 16, h, 5, c, false);
    }
    s_sky = mb_bake(m); mb_free(m);
}

void world_build(void) {
    build_terrain();
    build_props();
    build_sky();
    int total = 0;
    for (int ix = 0; ix < CH_NX; ix++) for (int iz = 0; iz < CH_NZ; iz++) {
        if (!s_cb[ix][iz]) continue;
        s_chunk[ix][iz] = mb_bake(s_cb[ix][iz]);
        total += s_chunk[ix][iz].tris;
        mb_free(s_cb[ix][iz]); s_cb[ix][iz] = NULL;
    }
    debugf("world: %d triangles in chunks, %d colliders\n", total, s_ncoll);
}

void world_open_doors(float k) { s_doorAngle = k * 1.45f; }
void world_update(float dt) { (void)dt; }

void world_draw_sky(V3 cam) {
    (void)cam;
    t3d_fog_set_enabled(false);
    T3DMat4FP *fp = mat_alloc();
    t3d_mat4fp_identity(fp);
    t3d_matrix_push(fp);
    mesh_draw(&s_sky);
    t3d_matrix_pop(1);
}

static int s_drawnTris, s_frameNo;
void world_draw(V3 cam, float camYaw, float fogFar) {
    if (++s_frameNo % 120 == 0) debugf("[perf] world triangles drawn: %d, fps %.1f\n", s_drawnTris, display_get_fps());
    s_drawnTris = 0;
    float fx = sinf(camYaw), fz = cosf(camYaw);
    T3DMat4FP *fp = mat_alloc();
    t3d_mat4fp_identity(fp);
    t3d_matrix_push(fp);
    for (int ix = 0; ix < CH_NX; ix++) for (int iz = 0; iz < CH_NZ; iz++) {
        if (!s_chunk[ix][iz].block) continue;
        float cx = CH_X0 + (ix + 0.5f) * CH_SIZE, cz = CH_Z0 + (iz + 0.5f) * CH_SIZE;
        float dx = cx - cam.x, dz = cz - cam.z, d = sqrtf(dx * dx + dz * dz);
        if (d > fogFar + CH_SIZE * 0.7f) continue;
        // horizontal view cone: ~38 degrees half-angle plus the chunk's bounding radius
        if (d > CH_SIZE * 0.8f) {
            float ang = acosf(clampf((dx * fx + dz * fz) / d, -1, 1));
            if (ang > 0.70f + asinf(minf(1, CH_SIZE * 0.75f / d))) continue;
        }
        mesh_draw(&s_chunk[ix][iz]);
        s_drawnTris += s_chunk[ix][iz].tris;
    }
    t3d_matrix_pop(1);
    // doors
    for (int d = 0; d < 2; d++) {
        float side = d ? 1 : -1;
        M4 m; m4_trs(&m, side * 3.2f * RS, height_at(0, GATE_Z) * RS, GATE_Z * RS, 0, -side * s_doorAngle, 0, 1);
        draw_mesh_at(&s_doorMesh[d], &m);
    }
}
