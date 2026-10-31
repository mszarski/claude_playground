// GILGAMESH 64 - demo level "Tablet V: The Cedar Forest"
// Built with libdragon + tiny3d. Everything (models, level, music) is generated in code.
#include "game.h"

Game G;
Input IN;
float g_camYaw;

// actors.c extras
void actors_animate(float dt);
void actors_draw_additive(void);
float actors_cam_recenter(float dt);
void spawn_seal(int id, float x, float z);

// ===================================================================== story text
static const Line INTRO[] = {
    {"", "Hear the tale of Gilgamesh, king of Uruk of the great walls: two-thirds god and one-third man."},
    {"", "The gods made the wild man Enkidu to be his equal. The two wrestled in the streets of Uruk, then embraced as brothers."},
    {"", "Now the king wants a name that will outlast him. Beyond the steppe lies the Cedar Forest and its guardian, Humbaba."},
    {"Enkidu", "Brother, the forest door is bound by three seals of Shamash, the sun god. Find all three and the way will open."},
    {"Enkidu", "If you lose your way, come and talk to me."},
};
static const Line STELE0[] = {
    {"Stele of Uruk", "Climb the wall of Uruk and walk upon it. Study its brickwork: were its bricks not fired in the kiln?"},
    {"Stele of Uruk", "A soldier has scratched notes beneath: Stick to walk. B to swing. A to roll, talk or read. Hold Z to lock on, R to raise your shield. C to turn the view."},
};
static const Line STELE1[] = {
    {"Boundary stone", "Here ends the land of men. Beyond lies the Cedar Forest, dwelling of the gods, given by Enlil into the keeping of Humbaba."},
    {"Boundary stone", "His roar is the Flood. His mouth is fire. His breath is death. Let no mortal pass this door."},
};
static const char *SEAL_NAME[3] = {"You got the Seal of Dawn!", "You got the Seal of Noon!", "You got the Seal of Dusk!"};
static const char *SEAL_DESC[3] = {
    "A cylinder seal of lapis and gold. Shamash rises between two mountains, rays streaming from his shoulders.",
    "Shamash sits enthroned at the top of the sky, holding the saw that cuts judgement.",
    "Shamash steps through the gate of the west, into the dark under the earth.",
};
static const char *SEAL_COUNT[4] = {"", "Seals of Shamash: 1 of 3.", "Seals of Shamash: 2 of 3.", "Seals of Shamash: 3 of 3. The way is open!"};
static Line s_sealLines[3];
static const Line GATE_OPEN[] = {
    {"Enkidu", "The three seals have opened the door! Beyond it lies the Cedar Forest."},
    {"Enkidu", "My hands tremble, brother. I saw Humbaba once, when I ran with the beasts. Let us go together."},
};
static const Line RING[] = {{"Enkidu", "The lions are slain! Look, brother, the altar stone is glowing."}};
static const Line BOSS_INTRO[] = {
    {"Humbaba", "Gilgamesh! A fool's counsel has brought you to my forest, and the wild man Enkidu with you."},
    {"Humbaba", "I will feed your flesh to the screaming vultures and the eagles!"},
    {"Enkidu", "Do not be afraid, brother! Raise your shield to his fire, and strike him while he reels!"},
};
static const Line BOSS_END[] = {
    {"Humbaba", "Mercy, Gilgamesh, son of Ninsun! Spare me, and the cedars will be yours to cut, as many as you desire."},
    {"Enkidu", "Do not listen, my friend! Finish him before the great gods hear of this and Enlil grows angry."},
    {"", "So Gilgamesh struck down the guardian, and the two friends felled the tallest cedar, whose crown had scraped the sky."},
    {"", "From its trunk Enkidu made a great door for the temple of Enlil, and they floated it down the Euphrates to Nippur."},
};
static const char *FLAVOUR[4] = {
    "Before the gods made me your brother, I ran with the gazelles on this steppe and drank with them at the water.",
    "Your mother Ninsun prayed to Shamash for us. The sun is on our side, brother.",
    "Hold Z to fix your eyes on a foe. Your blade will follow your gaze.",
    "The elders of Uruk say we are fools. Let them say it after we return with cedar.",
};
static Line s_enkLines[2];
static Line s_gateLines[2];
static char s_gateBuf[120];

// ===================================================================== story hooks
static void after_seal(void) {
    int n = G.seals[0] + G.seals[1] + G.seals[2];
    if (n == 3) cutscene_gate();
}
void on_seal_collected(int id) {
    int n = G.seals[0] + G.seals[1] + G.seals[2];
    s_sealLines[0] = (Line){"", SEAL_NAME[id]};
    s_sealLines[1] = (Line){"", SEAL_DESC[id]};
    s_sealLines[2] = (Line){"", SEAL_COUNT[n]};
    ui_say(s_sealLines, 3, after_seal);
}
static float s_ringTimer = -1;
void on_ring_cleared(void) { s_ringTimer = 0.5f; }
static void show_ending(void) { G.state = ST_ENDING; sfx(SFX_WIN); synth_set_track(1); }
void on_boss_defeated(void) { G.bossDead = true; ui_say(BOSS_END, 4, show_ending); }
void hud_dirty(void) {}

const char *context_label(V3 p) {
    extern V3 enkidu_pos(void);
    if (dist2d(p, enkidu_pos()) < 2.8f) return "Talk";
    for (int i = 0; i < 2; i++) if (dist2d(p, g_steles[i].pos) < 2.4f) return "Read";
    if (!G.gateOpen && fabsf(p.x) < 4 && p.z > GATE_Z && p.z < GATE_Z + 4) return "Check";
    return NULL;
}
bool context_run(V3 p) {
    extern V3 enkidu_pos(void);
    if (dist2d(p, enkidu_pos()) < 2.8f) {
        enkidu_face_player();
        if (G.gateOpen) {
            s_enkLines[0] = (Line){"Enkidu", "The cedar door stands open. Humbaba waits in the clearing beyond."};
            s_enkLines[1] = (Line){"Enkidu", "Raise your shield when he spits fire, and it flies back at him. Roll through his shockwave. Make him charge into a tree."};
        } else {
            s_enkLines[0] = (Line){"Enkidu", FLAVOUR[(int)(frand() * 4) & 3]};
            const char *hint = !G.seals[0] ? "The Seal of Dawn rests beside the oasis well, east of the city."
                             : !G.seals[1] ? "Scorpion-men guard the Seal of Noon at their camp in the west."
                                           : "The Seal of Dusk sleeps in the ring of standing stones to the north-east, where lions prowl.";
            s_enkLines[1] = (Line){"Enkidu", hint};
        }
        ui_say(s_enkLines, 2, NULL);
        return true;
    }
    if (dist2d(p, g_steles[0].pos) < 2.4f) { ui_say(STELE0, 2, NULL); return true; }
    if (dist2d(p, g_steles[1].pos) < 2.4f) { ui_say(STELE1, 2, NULL); return true; }
    if (!G.gateOpen && fabsf(p.x) < 4 && p.z > GATE_Z && p.z < GATE_Z + 4) {
        int n = G.seals[0] + G.seals[1] + G.seals[2];
        snprintf(s_gateBuf, sizeof(s_gateBuf), "Three hollows are cut into the bronze, each shaped for a cylinder seal. You hold %d of them.", n);
        s_gateLines[0] = (Line){"", "A great door of cedar, bound shut with three bands of bronze."};
        s_gateLines[1] = (Line){"", s_gateBuf};
        ui_say(s_gateLines, 2, NULL);
        return true;
    }
    return false;
}

// ===================================================================== camera and cutscenes
static V3 s_camPos = {0, 4, 42.2f}, s_camLook = {0, 1.5f, 37};
typedef struct { V3 pos, pos2, look; float dur; int kind; } Shot;
static Shot s_shots[3];
static int s_nShots, s_shot;
static float s_shotT;
static void (*s_cutDone)(void);
static bool s_cutActive;

static void play_cut(const Shot *shots, int n, void (*done)(void)) { memcpy(s_shots, shots, sizeof(Shot) * n); s_nShots = n; s_shot = 0; s_shotT = 0; s_cutDone = done; s_cutActive = true; }

static void gate_done(void) {
    G.gateOpen = true; world_set_door_collider(false); G.checkpoint = v3(0, 0, GATE_Z + 4); G.shake = 0;
    ui_say(GATE_OPEN, 2, NULL);
}
void cutscene_gate(void) {
    Shot s[1] = {{v3(10, 5, -9), v3(6, 4, -12), v3(0, 4.5f, GATE_Z), 3.4f, 1}};
    sfx(SFX_DOOR);
    play_cut(s, 1, gate_done);
}
static void boss_intro_done(void) { ui_say(BOSS_INTRO, 3, boss_begin_fight); ui_area_card("Guardian of the Cedars", "HUMBABA"); }
void cutscene_boss(void) {
    G.bossStarted = true;
    Shot s[2] = {{v3(6, 3, -40), v3(4, 2.5f, -46), v3(0, 4, -57), 2.2f, 0}, {v3(0, 2.2f, -49), v3(0, 2.6f, -50), v3(0, 5, -57), 2.0f, 2}};
    play_cut(s, 2, boss_intro_done);
}
static void cut_update(float dt, V3 *eye, V3 *look) {
    Shot *s = &s_shots[s_shot];
    if (s_shotT == 0 && s->kind == 2) { boss_set_state_roar(); sfx(SFX_ROAR); G.shake = 0.5f; }
    s_shotT += dt;
    float k = minf(1, s_shotT / s->dur), e = smoothf(k);
    if (s->kind == 1) { float d = smoothf(clampf((k - 0.15f) / 0.75f, 0, 1)); world_open_doors(d); G.shake = k < 0.9f ? 0.12f : 0; }
    *eye = v3(lerpf(s->pos.x, s->pos2.x, e), lerpf(s->pos.y, s->pos2.y, e), lerpf(s->pos.z, s->pos2.z, e));
    *look = s->look;
    if (s_shotT >= s->dur) {
        s_shot++; s_shotT = 0;
        if (s_shot >= s_nShots) { s_cutActive = false; s_camPos = *eye; void (*f)(void) = s_cutDone; s_cutDone = NULL; if (f) f(); }
    }
}

static void camera_update(float dt, V3 *eye, V3 *look) {
    V3 p = player_pos(), head = v3(p.x, p.y + 1.5f, p.z), desired, lk;
    if (lock_target()) {
        float camH; bool isB; V3 tp = lock_target_pos(&camH, &isB);
        float dx = p.x - tp.x, dz = p.z - tp.z, d = sqrtf(dx * dx + dz * dz) + 1e-3f; dx /= d; dz /= d;
        float dist = isB ? 8.5f : 6.2f;
        desired = v3(p.x + dx * dist - dz * 1.4f, p.y + (isB ? 4.2f : 3), p.z + dz * dist + dx * 1.4f);
        float th = isB ? 3.2f : 1.0f;
        lk = v3(lerpf(head.x, tp.x, 0.4f), lerpf(head.y, tp.y + th, 0.4f), lerpf(head.z, tp.z, 0.4f));
    } else {
        float ox = s_camPos.x - p.x, oz = s_camPos.z - p.z, d = sqrtf(ox * ox + oz * oz) + 1e-3f; ox /= d; oz /= d;
        float rot = 0;
        if (IN.cl) rot += 2.2f * dt;
        if (IN.cr) rot -= 2.2f * dt;
        rot -= IN.cx * 2.6f * dt;
        if (actors_cam_recenter(dt) > 0) { float behind = player_facing() + PI_F; rot += ang_diff(atan2f(ox, oz), behind) * minf(1, 12 * dt); }
        if (rot != 0) { float c = cosf(rot), s = sinf(rot), nx = ox * c + oz * s, nz = oz * c - ox * s; ox = nx; oz = nz; }
        bool forest = p.z < GATE_Z;
        float dist = forest ? 7.2f : 6.6f;
        desired = v3(p.x + ox * dist, p.y + (forest ? 3.3f : 2.9f), p.z + oz * dist);
        lk = head;
    }
    desired.x = clampf(desired.x, BX0 + 0.8f, BX1 - 0.8f); desired.z = clampf(desired.z, BZ0 + 0.8f, 42.2f);
    if (!(G.gateOpen && fabsf(desired.x) < 2.6f)) {
        if (p.z > GATE_Z && desired.z < GATE_Z + 1.5f) desired.z = GATE_Z + 1.5f;
        if (p.z < GATE_Z && desired.z > GATE_Z - 1.5f) desired.z = GATE_Z - 1.5f;
    }
    if (desired.x > -27 && desired.x < -13 && desired.z > 26 && desired.z < 38) desired.y = maxf(desired.y, 6);
    float k = 1 - expf(-(lock_target() ? 5 : 8) * dt);
    s_camPos.x += (desired.x - s_camPos.x) * k; s_camPos.y += (desired.y - s_camPos.y) * k; s_camPos.z += (desired.z - s_camPos.z) * k;
    s_camPos.y = maxf(s_camPos.y, height_at(s_camPos.x, s_camPos.z) + 0.9f);
    float kl = 1 - expf(-12 * dt);
    s_camLook.x += (lk.x - s_camLook.x) * kl; s_camLook.y += (lk.y - s_camLook.y) * kl; s_camLook.z += (lk.z - s_camLook.z) * kl;
    *eye = s_camPos; *look = s_camLook;
}

// ===================================================================== state transitions
static void start_game(void) {
    sfx(SFX_SELECT);
    G.state = ST_PLAY; synth_set_track(1);
    s_camPos = v3(0, 4, 42.2f);
#if defined(DEBUG_GATE)
    // debug: all seals collected, door open, start just north of the gate
    G.seals[0] = G.seals[1] = G.seals[2] = true; G.duskSpawned = true;
    G.gateOpen = true; world_set_door_collider(false); world_open_doors(1);
    player_set(v3(DEBUG_GATE_X, 0, DEBUG_GATE_Z), PI_F);
    s_camPos = v3(DEBUG_GATE_X, 4, DEBUG_GATE_Z + 6);
#elif defined(DEBUG_POS)
    player_set(v3(DEBUG_POS_X, 0, DEBUG_POS_Z), PI_F);
    s_camPos = v3(DEBUG_POS_X, 4, DEBUG_POS_Z + 6);
#else
    ui_say(INTRO, 5, NULL);
#endif
    ui_area_card("Tablet V", "THE STEPPE OF URUK");
    G.region = 0;
}
void game_over(void) { G.state = ST_GAMEOVER; }
static void continue_game(void) {
    G.state = ST_PLAY; G.hp = G.maxHp;
    actors_continue();
    V3 p = player_pos();
    s_camPos = v3(p.x, p.y + 3, minf(p.z + 6.5f, 42.2f));
    sfx(SFX_SELECT);
}
void game_ending(void) { show_ending(); }

static void reset_state(void) {
    memset(&G, 0, sizeof(G));
    G.state = ST_TITLE; G.hp = G.maxHp = 6; G.checkpoint = v3(0, 0, 37); G.aLabel = "Roll";
}

// ===================================================================== input
static void read_input(void) {
    joypad_poll();
    joypad_inputs_t in = joypad_get_inputs(JOYPAD_PORT_1);
    joypad_buttons_t pr = joypad_get_buttons_pressed(JOYPAD_PORT_1);
    float sx = in.stick_x / 80.0f, sy = in.stick_y / 80.0f;
    if (fabsf(sx) < 0.12f) sx = 0;
    if (fabsf(sy) < 0.12f) sy = 0;
    if (in.btn.d_left) sx = -1;
    if (in.btn.d_right) sx = 1;
    if (in.btn.d_up) sy = 1;
    if (in.btn.d_down) sy = -1;
    float l = sqrtf(sx * sx + sy * sy); if (l > 1) { sx /= l; sy /= l; }
    IN.mx = sx; IN.my = sy;
    IN.a = pr.a; IN.b = pr.b; IN.z = pr.z || pr.l; IN.r = pr.r; IN.start = pr.start;
    IN.zHeld = in.btn.z || in.btn.l; IN.rHeld = in.btn.r;
    IN.cl = in.btn.c_left; IN.cr = in.btn.c_right; IN.cx = 0;
}

// ===================================================================== main
int main(void) {
    debug_init_isviewer();
    debug_init_usblog();
    asset_init_compression(2);
    dfs_init(DFS_DEFAULT_LOCATION);
    display_init(RESOLUTION_320x240, DEPTH_16_BPP, FB_COUNT, GAMMA_NONE, FILTERS_RESAMPLE_ANTIALIAS_DEDITHER);
    rdpq_init();
    joypad_init();
    synth_init();
    t3d_init((T3DInitParams){});
    ui_init();
    T3DViewport viewport = t3d_viewport_create_buffered(FB_COUNT);

    reset_state();
    world_build();
    actors_init();
    debugf("Gilgamesh 64 ready\n");

    uint64_t last = get_ticks();
    int frameIdx = 0;
    const color_t FOG_STEPPE = RGBA32(0xE8, 0xCF, 0xA4, 0xFF), FOG_FOREST = RGBA32(0x30, 0x47, 0x2F, 0xFF);

    for (;;) {
        uint64_t now = get_ticks();
        float dt = (float)TICKS_DISTANCE(last, now) / TICKS_PER_SECOND; last = now;
        dt = clampf(dt, 0.001f, 1.0f / 15);
        G.time += dt;
        G.shake = maxf(0, G.shake - dt * 1.5f);
        read_input();
        synth_update();

        V3 eye, look;
        switch (G.state) {
        case ST_TITLE: {
            float t = G.time * 0.08f;
            eye = v3(sinf(t) * 26, 9 + sinf(t * 0.7f) * 2, 14 + cosf(t) * 22); look = v3(0, 3, 12);
            actors_animate(dt);
            if (IN.start || IN.a) start_game();
            break;
        }
        case ST_PLAY:
            if (ui_dialog_active()) { ui_dialog_update(dt, IN.a || IN.b || IN.start); actors_animate(dt); if (!s_cutActive) camera_update(dt, &eye, &look); else cut_update(0, &eye, &look); }
            else if (s_cutActive) { cut_update(dt, &eye, &look); actors_animate(dt); }
            else if (IN.start) { G.state = ST_PAUSE; sfx(SFX_SELECT); camera_update(0, &eye, &look); }
            else {
                G.playTime += dt;
                actors_update(dt);
                actors_animate(dt);
                camera_update(dt, &eye, &look);
                // triggers
                V3 p = player_pos();
                int region = p.z < GATE_Z - 0.5f ? 1 : 0;
                if (region != G.region) { G.region = region; if (region) ui_area_card("Dwelling of the gods", "THE CEDAR FOREST"); else ui_area_card("Tablet V", "THE STEPPE OF URUK"); }
                if (G.gateOpen && !G.bossStarted && p.z < -35) cutscene_boss();
                float ax = p.x - ARENA_X, az = p.z - ARENA_Z;
                bool inArena = boss_active() && sqrtf(ax * ax + az * az) < ARENA_R + 4;
                if (!player_dead()) synth_set_track(inArena ? 2 : 1);
                if (s_ringTimer >= 0) { s_ringTimer -= dt; if (s_ringTimer < 0) { sfx(SFX_SECRET); spawn_seal(2, 14, -10); particles_burst(v3(14, 0, -10), 20, 0, 3, 7, 1); ui_say(RING, 1, NULL); } }
            }
            break;
        case ST_PAUSE:
            camera_update(0, &eye, &look);
            if (IN.start) { G.state = ST_PLAY; sfx(SFX_SELECT); }
            break;
        case ST_GAMEOVER:
            actors_animate(dt); camera_update(dt, &eye, &look);
            if (IN.start || IN.a) continue_game();
            break;
        case ST_ENDING:
        default:
            actors_animate(dt); camera_update(dt, &eye, &look);
            if (IN.start) { /* soft reset: reboot the console */ rdpq_detach_wait(); assertf(0, "Thanks for playing Gilgamesh 64!"); }
            break;
        }
        ui_update(dt);
        if (G.shake > 0) { eye.x += (frand() - 0.5f) * G.shake; eye.y += (frand() - 0.5f) * G.shake; eye.z += (frand() - 0.5f) * G.shake; }
        g_camYaw = atan2f(look.x - eye.x, look.z - eye.z);

        // ---------------------------------------------------------- atmosphere
        V3 ref = G.state == ST_TITLE ? eye : player_pos();
        float ft = smoothf(clampf((-ref.z - 16) / 14, 0, 1));
        color_t fog = {
            (uint8_t)lerpf(FOG_STEPPE.r, FOG_FOREST.r, ft), (uint8_t)lerpf(FOG_STEPPE.g, FOG_FOREST.g, ft), (uint8_t)lerpf(FOG_STEPPE.b, FOG_FOREST.b, ft), 0xFF};
        float fogNear = lerpf(30, 12, ft), fogFar = lerpf(80, 48, ft);

        // ---------------------------------------------------------- 3D
        frameIdx = (frameIdx + 1) % FB_COUNT;
        mat_frame_begin(frameIdx);
        t3d_viewport_set_projection(&viewport, T3D_DEG_TO_RAD(60.0f), 5.0f, 900.0f);
        t3d_viewport_look_at(&viewport, &(T3DVec3){{eye.x * RS, eye.y * RS, eye.z * RS}}, &(T3DVec3){{look.x * RS, look.y * RS, look.z * RS}}, &(T3DVec3){{0, 1, 0}});

        rdpq_attach(display_get(), display_get_zbuf());
        t3d_frame_start();
        t3d_viewport_attach(&viewport);
        t3d_screen_clear_color(fog);
        t3d_screen_clear_depth();
        rdpq_mode_combiner(RDPQ_COMBINER1((PRIM, 0, PRIM_ALPHA, SHADE), (0, 0, 0, SHADE)));
        rdpq_set_prim_color(RGBA32(0, 0, 0, 0));
        rdpq_mode_dithering(DITHER_SQUARE_SQUARE);

        uint8_t amb[4] = {(uint8_t)lerpf(120, 70, ft), (uint8_t)lerpf(105, 85, ft), (uint8_t)lerpf(90, 70, ft), 0xFF};
        uint8_t dir[4] = {(uint8_t)lerpf(240, 150, ft), (uint8_t)lerpf(220, 160, ft), (uint8_t)lerpf(185, 120, ft), 0xFF};
        T3DVec3 ldir = {{0.45f, 0.8f, 0.4f}}; t3d_vec3_norm(&ldir);
        t3d_light_set_ambient(amb);
        t3d_light_set_directional(0, dir, &ldir);
        t3d_light_set_count(1);
        t3d_state_set_drawflags(T3D_FLAG_SHADED | T3D_FLAG_DEPTH);

#ifndef NO_SKY
        world_draw_sky(eye);
#endif
        rdpq_mode_fog(RDPQ_FOG_STANDARD);
        rdpq_set_fog_color(fog);
        t3d_fog_set_range(fogNear * RS, fogFar * RS);
        t3d_fog_set_enabled(true);
        world_draw(eye, g_camYaw, fogFar);
        actors_draw();

        // translucent passes: blob shadows, then glowing effects
        t3d_fog_set_enabled(false);
        rdpq_mode_fog(0);
        t3d_state_set_drawflags(T3D_FLAG_SHADED | T3D_FLAG_DEPTH | T3D_FLAG_NO_LIGHT);
        rdpq_mode_zbuf(true, false);
        rdpq_mode_blender(RDPQ_BLENDER_MULTIPLY);
        actors_draw_shadows();
        rdpq_mode_blender(RDPQ_BLENDER_ADDITIVE);
        actors_draw_additive();
        rdpq_mode_blender(0);
        rdpq_mode_zbuf(true, true);

        // ---------------------------------------------------------- 2D
        if (G.state == ST_PLAY || G.state == ST_PAUSE) { if (!s_cutActive) ui_draw_hud(); ui_draw_dialog(); }
        ui_draw_overlays();
        rdpq_detach_show();
    }
}
