"""Ductus Display -- an original high-contrast display serif (M2).

Built from the serif kit with our own values and the Didone terminal set
(flat hairline serifs, ball terminals) -- every value is a design decision
made by eye, never fitted to a reference. Iteration notes:
docs/display_serif/README.md.
"""
from ductus import Font
from ductus.kits import serif as kit

font = Font("Ductus Display", upm=1000)

P = kit.params(
    # weight and contrast: Didone end of the genre
    stem=96, cap_stem=104, hairline=14, round_gain=1.06, overshoot=11,
    arch_angle=25, arch_contrast=0.86,
    sb_straight=46, sb_round=34,
    # serifs: flat hairlines, a whisper of bracket
    serif_len=62, serif_bracket=12, serif_ca=0.5, serif_cb=0.7, foot_tip=14,
    cap_serif_len=72, cap_serif_bracket=14, head_len=62, head_top=0,
    top_serif_len=58, top_serif_bracket=10, spur_len=60,
    beak_len=95, beak_depth=14, ball_r=48, ball_drop=-22,
    # rounds (iteration 02 o, slight oblique counter)
    o_width=500, o_full_h=0.60, o_full_v=0.56, o_in_full_h=0.56, o_in_full_v=0.62, o_stress=6,
    # arches: hairline join, thick inside
    n_width=400, n_join_at=0.74, n_join_thin=16, n_join_angle=50, n_shoulder_x=0.48, n_out_y=0.62,
    n_taper_len=0.32, n_bias=0.42, n_tension=1.05,
    m_width=360, u_width=400, u_bias=0.3, b_width=500,
    u_join_at=0.74, u_join_thin=16, u_join_angle=50, u_shoulder_x=0.48, u_out_y=0.62, u_taper_len=0.32,
    u_tension=1.05, u_angle=25,
    r_join_at=0.74, r_join_thin=16, r_join_angle=50, r_taper_len=0.32, r_end_w=1.0, t_turn_w=1.0, t_end_w=1.0,
    t_angle=0, t_contrast=0.8541667, t_bar_t=32, f_bar_t=32,
    # a: the bowl is a lobe on the stem -- hairline top join, heavier bottom join
    a_hood_angle=25, a_far=46, a_top_at=292, a_top_thin=15, a_top_angle=18, a_bot_at=115, a_bot_thin=48, a_bot_angle=55,
    a_lobe_thick=98, a_lobe_thin_b=16, a_out_mid=0.48, a_in_mid=0.5, a_bx_out=0.6, a_bx_in=0.8,

    H_width=560, H_bar_y=0.5, H_bar_t=16,
    # s / S: vertical terminal faces with hairline beards, spine swells mid-way
    s_width=360, s_t1_x=0.94, s_t1_y=0.74, s_t1_dir=180, s_cut1=0,
    s_t2_x=0.05, s_t2_y=0.27, s_t2_dir=180, s_cut2=0,
    s_top_x=0.5, s_left_y=0.73, s_mid_x=0.5, s_mid_y=0.48, s_mid_dir=-48, s_right_y=0.25, s_bot_x=0.5,
    s_ends=0.85, s_swell=1.35, s_spread=0.3, s_tension=1.1, S_width=470,
    G_width=680, G_term_y=0.80, G_bar_y=0.42, G_serif=120,
    # e: aperture edge bowed so the terminal tapers to an upturned hairline
    e_width=460, e_term_y=150, e_cut_angle=205, e_cut_y=70, e_cut_x=0.5, e_cut_h1=0.45, e_cut_h2=0.4,
    e_bar_y=0.56, e_bar_t=16,
    e_full_h=0.6, e_full_v=0.56, e_in_full_h=0.56, e_in_full_v=0.6, e_stress=6,
    # g: hairline link, lower loop with thick sides and hairline top/bottom, thin ear with a ball
    g_bl=40, g_br=330, g_bb=205, g_bt=541, g_thick=0.95, g_full=0.58, g_in_full=0.58,
    g_ear_x=420, g_ear_y=545, g_ear_w=0.45,
    g_link_x0=0.85, g_link_x=46, g_link_y=128, g_band_x=300, g_band_y=46, g_link_w=0.35,
    g_loop_l=8, g_loop_r=455, g_loop_b=-212, g_loop_t=60, g_loop_thick=0.75, g_loop_tr=0.9,
    g_loop_top_t=18, g_loop_bot_t=20, g_loop_full=0.58, g_loop_in_full=0.6, g_loop_stress=6,
    # the nib turns along curved strokes (iteration 09): toward the horizontal
    # at the tops of curves, so arch shoulders, hook tops and the s's arcs go
    # to true hairlines, and back into the descents, so the swell arrives late
    # and full -- the Bodoni arch, rather than a pen held at one angle
    n_shoulder_pa=-30, n_leg_pa=15, u_shoulder_pa=-30, u_leg_pa=15,
    s_top_pa=-30, s_left_pa=15, s_right_pa=15, s_bot_pa=-30,
    f_top_pa=-30, f_end_pa=15, r_end_pa=-30, t_bottom_pa=-30, t_end_pa=-15,
    a_top_pa=-30, a_term_pa=15,
)
kit.define(font, P, glyphs="hamburgefonstivHOAGES ", terminals=kit.didone_set(P))
