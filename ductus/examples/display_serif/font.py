"""Ductus Display -- an original high-contrast display serif (M2).

Built from the serif kit with our own values and the Didone terminal set
(flat hairline serifs, ball terminals) -- every value is a design decision
made by eye, never fitted to a reference. Iteration notes:
docs/display_serif/README.md.
"""
from ductus import Font
from ductus.kits import serif as kit
from ductus.terminals import ball

font = Font("Ductus Display", upm=1000)

P = kit.params(
    # weight and contrast: Didone end of the genre
    stem=96, cap_stem=104, hairline=22, round_gain=1.06, overshoot=11,
    arch_angle=25, arch_contrast=0.86,
    sb_straight=36, sb_round=30,
    # serifs: flat hairlines, a whisper of bracket
    serif_len=74, serif_bracket=18, serif_ca=0.5, serif_cb=0.7, foot_tip=26,
    cap_serif_len=84, cap_serif_bracket=20, head_len=70, head_top=0,
    top_serif_len=58, top_serif_bracket=10, spur_len=60,
    beak_len=95, beak_depth=14, ball_r=48, ball_drop=-22,
    # rounds (iteration 02 o, slight oblique counter)
    o_width=500, o_full_h=0.60, o_full_v=0.56, o_in_full_h=0.56, o_in_full_v=0.62, o_stress=6,
    # arches: hairline join, thick inside
    n_width=400, n_join_at=0.84, n_join_thin=16, n_join_angle=30, n_shoulder_x=0.48, n_out_y=0.62,
    n_taper_len=0.32, n_bias=0.42, n_tension=1.05,
    m_width=360, u_width=400, u_bias=0.3, b_width=500,
    u_join_at=0.84, u_join_thin=16, u_join_angle=30, u_shoulder_x=0.48, u_out_y=0.62, u_taper_len=0.32,
    u_tension=1.05, u_angle=25,
    r_join_at=0.74, r_join_thin=16, r_join_angle=50, r_taper_len=0.32, r_end_w=1.0, t_turn_w=1.0,
    t_angle=0, t_contrast=0.8541667, t_bar_t=32, f_bar_t=32,
    # a: the bowl is a lobe on the stem -- hairline top join, heavier bottom join
    a_hood_angle=25, a_far=46, a_top_at=292, a_top_thin=15, a_top_angle=18, a_bot_at=60, a_bot_thin=30, a_bot_angle=25,
    a_lobe_thick=98, a_lobe_thin_b=16, a_out_mid=0.48, a_in_mid=0.5, a_bx_out=0.6, a_bx_in=0.8,

    H_width=560, H_bar_y=0.5, H_bar_t=16,
    # s / S: vertical terminal faces with hairline beards, spine swells mid-way
    s_width=360, s_t1_x=0.9, s_t1_y=0.8, s_t1_dir=100, s_cut1=0,
    s_t2_x=0.08, s_t2_y=0.2, s_t2_dir=100, s_cut2=0,
    s_top_x=0.5, s_left_y=0.73, s_mid_x=0.5, s_mid_y=0.48, s_mid_dir=-24, s_mid_free=1, s_right_y=0.25, s_bot_x=0.5,
    s_ends=0.5, s_swell=2.2, s_spread=0.24, s_tension=1.1, S_width=470,
    G_width=680, G_term_y=0.80, G_bar_y=0.42, G_serif=50, bar_t=22,
    G_beard_dw=70, G_beard_k=0.35, G_spur_w=0.55,   # the G's arm ends flat; lighter jaw (iteration 12)
    # e: aperture edge bowed so the terminal tapers to an upturned hairline
    e_width=460, e_term_y=160, e_cut_angle=180, e_cut_y=70, e_cut_x=0.5, e_cut_h1=0.45, e_cut_h2=0.4,
    e_bar_y=0.56, e_bar_t=16,
    e_full_h=0.6, e_full_v=0.56, e_in_full_h=0.56, e_in_full_v=0.6, e_stress=6,
    # g: hairline link, lower loop with thick sides and hairline top/bottom, thin ear with a ball
    g_bl=40, g_br=330, g_bb=205, g_bt=541, g_thick=0.95, g_full=0.58, g_in_full=0.58,
    g_ear_x=420, g_ear_y=545, g_ear_w=0.45,
    g_link_x0=0.85, g_link_x=46, g_link_y=128, g_band_x=300, g_band_y=46, g_link_w=0.35,
    g_loop_l=8, g_loop_r=455, g_loop_b=-212, g_loop_t=60, g_loop_thick=0.6, g_loop_tr=0.72,
    g_loop_top_t=18, g_loop_bot_t=14, g_loop_full=0.58, g_loop_in_full=0.6, g_loop_stress=6,
    # the nib turns along curved strokes (iteration 09): toward the horizontal
    # at the tops of curves, so arch shoulders, hook tops and the s's arcs go
    # to true hairlines, and back into the descents, so the swell arrives late
    # and full -- the Bodoni arch, rather than a pen held at one angle
    n_shoulder_pa=-10, n_leg_pa=5,   # the u's cup takes no turns: they hooked its hairline
    f_top_pa=-30, f_end_pa=15, r_end_pa=-30, t_end_pa=0,
    a_top_pa=-30, a_term_pa=15,
    # b (iteration 10): the bowl hangs on the stem at hairline joins, as the a's
    # does -- the old separate bowl laid a thick wall beside the stem ("lb")
    b_far=540, b_top_at=420, b_top_thin=14, b_top_angle=-35, b_bot_at=70, b_bot_thin=22, b_bot_angle=35,
    b_lobe_thick=100, b_lobe_thin_b=16, b_crown_x=0.45, b_crown_t=16, b_top=0, b_foot=0,
    b_out_mid=0.48, b_in_mid=0.5, b_bx_out=0.8, b_bx_in=0.85,
    # the leg stems stop where the arch / cup has turned vertical, so their flat
    # ends never show beside it (a step on the n's leg and the u's left stem)
    n_leg_top=0.55, u_leg_top=0.55,
    r_rsb=-20,   # the r's ball overhangs: close up r-g, r-a (r-n stays clear)
    # s/S (iteration 11): arms rise into ball terminals (the face's other
    # terminals are balls too), hairline arm ends; the spine's middle direction
    # is solved for continuous curvature, so it never kinks
    # arch: the hairline leaves the stem high and curves away (it left low on a
    # straight 50-degree diagonal, which read italic); gentler nib turns
    # t: a longer tail rising gently into a ball, like the face's other terminals
    t_tail=260, t_tail_y=35, t_end_dir=10, t_end_w=0.2, t_turn=220, t_bottom_pa=0,
    # iteration 12: heavier hairlines and serifs (hold up at text sizes and when
    # proofs are scaled), tighter spacing; the s's spine carries its weight;
    # the t's tail ends in a plain hairline (a ball there read as a full stop)
    # fi ligature (iteration 15): the hook reaches over the i, whose dot is the ball
    fi_dx=175,
)
terms = kit.didone_set(P)
terms.by_tag["s_top"] = ball(radius=P.ball_r * 0.8, side="below", drop=P.ball_drop)
terms.by_tag["s_bottom"] = ball(radius=P.ball_r * 0.8, side="above", drop=P.ball_drop)
kit.define(font, P, glyphs="hamburgefonstivHOAGES ", terminals=terms, b_style="lobe")

# spacing (iteration 14): set between n's and o's, judged by eye at 72 px
font.spacing.update({
    "f": (0, -60),     # hook and bar overhang: closes f+n, f+o
})
font.kern("r", "g", -20)
# the f's ball overhangs: keep it clear of dots, ascenders and the next f
for right, v in (("i", 25), ("f", 20), ("b", 30), ("h", 30)):
    font.kern("f", right, v)
