// HUD, dialogue box, area cards and full-screen overlays (all drawn with rdpq in 2D).
#include "game.h"

static rdpq_font_t *s_font, *s_title;
enum { STY_WHITE, STY_GOLD, STY_LAPIS, STY_DIM, STY_DARK };

void ui_init(void) {
    s_font = rdpq_font_load("rom:/monogram.font64");
    s_title = rdpq_font_load("rom:/abaddon.font64");
    rdpq_text_register_font(FONT_TEXT, s_font);
    rdpq_text_register_font(FONT_TITLE, s_title);
    const rdpq_fontstyle_t st[] = {
        {.color = RGBA32(0xFF, 0xF8, 0xE8, 0xFF)}, {.color = RGBA32(0xFF, 0xD8, 0x6A, 0xFF)},
        {.color = RGBA32(0x7F, 0xA2, 0xFF, 0xFF)}, {.color = RGBA32(0xB0, 0xA8, 0x98, 0xFF)}, {.color = RGBA32(0x2A, 0x1A, 0x00, 0xFF)},
    };
    for (int i = 0; i < 5; i++) { rdpq_font_style(s_font, i, &st[i]); rdpq_font_style(s_title, i, &st[i]); }
}

// ------------------------------------------------------------------ primitives
static void mode_flat(color_t c) {
    rdpq_set_mode_standard();
    rdpq_mode_combiner(RDPQ_COMBINER_FLAT);
    if (c.a < 255) rdpq_mode_blender(RDPQ_BLENDER_MULTIPLY);
    rdpq_set_prim_color(c);
}
static void rect(float x, float y, float w, float h, color_t c) { mode_flat(c); rdpq_fill_rectangle(x, y, x + w, y + h); }
static void tri(float ax, float ay, float bx, float by, float cx, float cy) {
    rdpq_triangle(&TRIFMT_FILL, (float[]){ax, ay}, (float[]){bx, by}, (float[]){cx, cy});
}
static void star(float cx, float cy, float r, color_t fill, float frac) {
    // 8-pointed star of Ishtar; frac is how much of it is filled (0, 0.5, 1)
    mode_flat(RGBA32(0x3A, 0x1D, 0x00, 0xFF));
    for (int i = 0; i < 16; i++) {
        float a0 = i / 16.0f * TAU_F - PI_F / 2, a1 = (i + 1) / 16.0f * TAU_F - PI_F / 2;
        float r0 = (i & 1) ? r * 0.48f : r + 1.2f, r1 = (i & 1) ? r + 1.2f : r * 0.48f;
        tri(cx, cy, cx + cosf(a0) * r0, cy + sinf(a0) * r0, cx + cosf(a1) * r1, cy + sinf(a1) * r1);
    }
    for (int pass = 0; pass < 2; pass++) {
        color_t c = pass == 0 ? RGBA32(0x2A, 0x1A, 0x2E, 0xFF) : fill;
        if (pass == 1 && frac <= 0) break;
        mode_flat(c);
        if (pass == 1 && frac < 1) rdpq_set_scissor(0, 0, (int)cx, 240);
        for (int i = 0; i < 16; i++) {
            float a0 = i / 16.0f * TAU_F - PI_F / 2, a1 = (i + 1) / 16.0f * TAU_F - PI_F / 2;
            float r0 = (i & 1) ? r * 0.42f : r, r1 = (i & 1) ? r : r * 0.42f;
            tri(cx, cy, cx + cosf(a0) * r0, cy + sinf(a0) * r0, cx + cosf(a1) * r1, cy + sinf(a1) * r1);
        }
        if (pass == 1 && frac < 1) rdpq_set_scissor(0, 0, 320, 240);
    }
}
static void disc(float cx, float cy, float r, color_t c) {
    mode_flat(c);
    for (int i = 0; i < 10; i++) {
        float a0 = i / 10.0f * TAU_F, a1 = (i + 1) / 10.0f * TAU_F;
        tri(cx, cy, cx + cosf(a0) * r, cy + sinf(a0) * r, cx + cosf(a1) * r, cy + sinf(a1) * r);
    }
}
static void text(int font, int style, float x, float y, int width, int align, const char *s) {
    rdpq_set_mode_standard();
    rdpq_text_print(&(rdpq_textparms_t){.style_id = style, .width = width, .align = width ? align : ALIGN_LEFT, .wrap = width ? WRAP_WORD : WRAP_NONE}, font, x, y, s);
}
static void textf(int font, int style, float x, float y, int width, int align, const char *fmt, ...) {
    char buf[160]; va_list ap; va_start(ap, fmt); vsnprintf(buf, sizeof(buf), fmt, ap); va_end(ap);
    text(font, style, x, y, width, align, buf);
}
// text with a 1px drop shadow for readability over the 3D scene
static void stext(int font, int style, float x, float y, int width, int align, const char *s) {
    text(font, STY_DARK, x + 1, y + 1, width, align, s);
    text(font, style, x, y, width, align, s);
}

// ------------------------------------------------------------------ dialogue
static struct { bool active; const Line *lines; int n, i; float shown, blip; void (*onDone)(void); } D;
void ui_say(const Line *lines, int n, void (*onDone)(void)) { D.active = true; D.lines = lines; D.n = n; D.i = 0; D.shown = 0; D.onDone = onDone; }
bool ui_dialog_active(void) { return D.active; }
void ui_dialog_update(float dt, bool adv) {
    if (!D.active) return;
    int len = strlen(D.lines[D.i].text);
    if (D.shown < len) {
        int prev = (int)D.shown;
        D.shown = adv ? len : minf(len, D.shown + dt * 55);
        if ((int)D.shown / 3 != prev / 3) sfx(SFX_TEXT);
    } else if (adv) {
        sfx(SFX_SELECT);
        if (++D.i >= D.n) { D.active = false; void (*f)(void) = D.onDone; D.onDone = NULL; if (f) f(); }
        else D.shown = 0;
    }
}
void ui_draw_dialog(void) {
    if (!D.active) return;
    const Line *L = &D.lines[D.i];
    float x = 24, y = 160, w = 272, h = 70;
    rect(x, y, w, h, RGBA32(12, 10, 30, 210));
    rect(x, y, w, 2, RGBA32(0xE8, 0xC2, 0x6A, 0xFF)); rect(x, y + h - 2, w, 2, RGBA32(0xE8, 0xC2, 0x6A, 0xFF));
    rect(x, y, 2, h, RGBA32(0xE8, 0xC2, 0x6A, 0xFF)); rect(x + w - 2, y, 2, h, RGBA32(0xE8, 0xC2, 0x6A, 0xFF));
    if (L->who && L->who[0]) {
        int tw = strlen(L->who) * 6 + 12;
        rect(x + 10, y - 9, tw, 13, RGBA32(0xE8, 0xC2, 0x6A, 0xFF));
        text(FONT_TEXT, STY_DARK, x + 16, y + 1, 0, ALIGN_LEFT, L->who);
    }
    rdpq_set_mode_standard();
    rdpq_text_print(&(rdpq_textparms_t){.style_id = STY_WHITE, .width = w - 20, .wrap = WRAP_WORD, .max_chars = (int16_t)D.shown}, FONT_TEXT, x + 10, y + 17, L->text);
    if (D.shown >= strlen(L->text) && ((int)(G.time * 3) & 1)) { mode_flat(RGBA32(0xE8, 0xC2, 0x6A, 0xFF)); tri(x + w - 18, y + h - 12, x + w - 10, y + h - 12, x + w - 14, y + h - 7); }
}

// ------------------------------------------------------------------ area card
static struct { const char *top, *name; float t; } A;
void ui_area_card(const char *top, const char *name) { A.top = top; A.name = name; A.t = 3.0f; }
void ui_update(float dt) { if (A.t > 0) A.t -= dt; }

// ------------------------------------------------------------------ HUD
void ui_draw_hud(void) {
    for (int i = 0; i < G.maxHp / 2; i++) {
        int v = G.hp - i * 2;
        float pulse = (G.hp <= 2 && G.hp > 0) ? 1 + 0.12f * sinf(G.time * 7) : 1;
        star(26 + i * 19, 24, 8 * pulse, RGBA32(0xF2, 0xB3, 0x3A, 0xFF), v >= 2 ? 1 : v == 1 ? 0.5f : 0);
    }
    for (int i = 0; i < 3; i++) {
        color_t c = G.seals[i] ? RGBA32(0xF6, 0xD2, 0x72, 0xFF) : RGBA32(0x2C, 0x24, 0x18, 0xC0);
        rect(19 + i * 11, 38, 7, 13, RGBA32(0x12, 0x0C, 0x04, 0xFF));
        rect(20 + i * 11, 39, 5, 11, c);
    }
    // lapis counter
    mode_flat(RGBA32(0x3A, 0x64, 0xFF, 0xFF));
    tri(24, 212, 30, 218, 24, 224); tri(24, 212, 18, 218, 24, 224);
    char buf[8]; snprintf(buf, sizeof(buf), "%03d", G.lapis);
    stext(FONT_TEXT, STY_WHITE, 34, 223, 0, ALIGN_LEFT, buf);
    // action buttons: B (sword) and A (context action)
    disc(258, 26, 11, RGBA32(0x5C, 0x24, 0x10, 0xFF)); disc(258, 26, 10, RGBA32(0xB5, 0x56, 0x2B, 0xFF));
    text(FONT_TEXT, STY_WHITE, 255, 30, 0, ALIGN_LEFT, "B");
    stext(FONT_TEXT, STY_WHITE, 240, 50, 36, ALIGN_CENTER, "Sword");
    disc(292, 28, 13, RGBA32(0x13, 0x24, 0x6B, 0xFF)); disc(292, 28, 12, RGBA32(0x2A, 0x4D, 0xBF, 0xFF));
    text(FONT_TEXT, STY_WHITE, 289, 32, 0, ALIGN_LEFT, "A");
    stext(FONT_TEXT, STY_WHITE, 272, 54, 40, ALIGN_CENTER, G.aLabel ? G.aLabel : "Roll");
    if (G.bossBar) {
        stext(FONT_TEXT, STY_GOLD, 60, 212, 200, ALIGN_CENTER, "HUMBABA, GUARDIAN OF THE CEDARS");
        rect(70, 218, 180, 7, RGBA32(0, 0, 0, 0xFF));
        rect(71, 219, 178, 5, RGBA32(0x30, 0x08, 0x06, 0xFF));
        rect(71, 219, 178 * G.bossHpFrac, 5, RGBA32(0xD8, 0x40, 0x28, 0xFF));
    }
}

// ------------------------------------------------------------------ overlays
static void draw_logo(float y) {
    text(FONT_TITLE, STY_DARK, 2, y + 2, 320, ALIGN_CENTER, "GILGAMESH");
    text(FONT_TITLE, STY_GOLD, 0, y, 320, ALIGN_CENTER, "GILGAMESH");
    text(FONT_TITLE, STY_DARK, 2, y + 30, 320, ALIGN_CENTER, "64");
    text(FONT_TITLE, STY_LAPIS, 0, y + 28, 320, ALIGN_CENTER, "64");
}
extern int g_titleBlink;
void ui_draw_overlays(void) {
    if (A.t > 0 && G.state == ST_PLAY) {
        float a = A.t > 2.4f ? (3 - A.t) / 0.6f : A.t < 0.6f ? A.t / 0.6f : 1;
        if (a > 0.3f) {
            stext(FONT_TEXT, STY_GOLD, 0, 92, 320, ALIGN_CENTER, A.top);
            stext(FONT_TITLE, STY_WHITE, 0, 112, 320, ALIGN_CENTER, A.name);
        }
    }
    switch (G.state) {
    case ST_TITLE:
        draw_logo(62);
        stext(FONT_TEXT, STY_WHITE, 0, 128, 320, ALIGN_CENTER, "Tablet V - The Cedar Forest");
        if ((int)(G.time * 2) & 1) stext(FONT_TEXT, STY_WHITE, 0, 176, 320, ALIGN_CENTER, "PRESS START");
        stext(FONT_TEXT, STY_DIM, 0, 222, 320, ALIGN_CENTER, "A homebrew demo built with libdragon + tiny3d");
        break;
    case ST_PAUSE: {
        rect(28, 30, 264, 180, RGBA32(12, 10, 30, 225));
        rect(28, 30, 264, 2, RGBA32(0xE8, 0xC2, 0x6A, 0xFF)); rect(28, 208, 264, 2, RGBA32(0xE8, 0xC2, 0x6A, 0xFF));
        text(FONT_TEXT, STY_GOLD, 40, 48, 0, ALIGN_LEFT, "TABLET V - THE CEDAR FOREST");
        const char *q[5] = {"Find the Seal of Dawn at the oasis", "Take the Seal of Noon from the scorpion-men", "Win the Seal of Dusk at the standing stones", "Open the cedar door", "Defeat Humbaba"};
        bool done[5] = {G.seals[0], G.seals[1], G.seals[2], G.gateOpen, G.bossDead};
        for (int i = 0; i < 5; i++) { textf(FONT_TEXT, done[i] ? STY_GOLD : STY_DIM, 44, 68 + i * 13, 0, ALIGN_LEFT, "%s %s", done[i] ? "*" : "-", q[i]); }
        text(FONT_TEXT, STY_WHITE, 40, 142, 240, ALIGN_LEFT, "Stick: move   B: sword   A: roll / talk");
        text(FONT_TEXT, STY_WHITE, 40, 155, 240, ALIGN_LEFT, "Z: lock on   R: shield   C: turn camera");
        text(FONT_TEXT, STY_DIM, 40, 190, 240, ALIGN_LEFT, "START: resume");
        break;
    }
    case ST_GAMEOVER:
        rect(0, 0, 320, 240, RGBA32(40, 0, 0, 150));
        stext(FONT_TITLE, STY_WHITE, 0, 100, 320, ALIGN_CENTER, "THE SUN SETS");
        stext(FONT_TEXT, STY_DIM, 0, 122, 320, ALIGN_CENTER, "Even two-thirds of a god can fall.");
        if ((int)(G.time * 2) & 1) stext(FONT_TEXT, STY_WHITE, 0, 160, 320, ALIGN_CENTER, "PRESS START TO RISE AGAIN");
        break;
    case ST_ENDING: {
        rect(0, 0, 320, 240, RGBA32(5, 4, 14, 190));
        draw_logo(40);
        stext(FONT_TITLE, STY_WHITE, 0, 112, 320, ALIGN_CENTER, "TABLET V COMPLETE");
        int t = (int)G.playTime;
        char buf[64]; snprintf(buf, sizeof(buf), "Time on the steppe  %d:%02d", t / 60, t % 60);
        stext(FONT_TEXT, STY_WHITE, 0, 136, 320, ALIGN_CENTER, buf);
        snprintf(buf, sizeof(buf), "Lapis gathered  %d", G.lapis);
        stext(FONT_TEXT, STY_WHITE, 0, 149, 320, ALIGN_CENTER, buf);
        stext(FONT_TEXT, STY_GOLD, 0, 172, 320, ALIGN_CENTER, "Next: Tablet VI - The Bull of Heaven");
        if ((int)(G.time * 2) & 1) stext(FONT_TEXT, STY_WHITE, 0, 204, 320, ALIGN_CENTER, "PRESS START TO PLAY AGAIN");
        break;
    }
    default: break;
    }
}
