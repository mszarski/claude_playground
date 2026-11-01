"""Ductus Display -- a transitional display serif (M2), built from the serif kit.

Iterations 01-17 made it a high-contrast Didone by eye. From iteration 18 it is
tuned toward Geist Serif (OFL-1.1, Vercel): the same kit, with parameter values
chosen from side-by-side overlays and the bench's IoU, then rounded. No
outlines are copied -- every glyph is still drawn by the kit's strokes, pens and
terminals. Iteration notes: docs/display_serif/README.md.
"""
from ductus import Font
from ductus.kits import serif as kit

font = Font("Ductus Display", upm=1000)

P = kit.params(
    # weight, contrast, spacing
    hairline=32, round_gain=1.18, overshoot=11, arch_angle=40, arch_contrast=0.3, sb_straight=15, sb_round=42,
    t_contrast=0.3, bar_t=50, bar_taper=0.87, bar_tip_dx=3, bar_tip_y=0.42,
    # serifs and stroke ends
    serif_len=83, serif_bracket=70, serif_ca=0.41, serif_cb=0.85, foot_tip=60, cap_serif_len=85,
    cap_serif_bracket=64, head_len=71, head_top=15, top_serif_bracket=58, spur_len=79, beak_len=95,
    beak_depth=14, ball_r=48, ball_drop=-22, G_spur_w=1.0, head_rise=55, head_drop=34, head_hollow=0.62,
    head_tip=13, head_flat=8, head_bracket=0.5, head_corner=19, head_ca=0.76, head_cb=0.59, head_inset=4,
    head_inset_y=24, serif_cup=-0.67, serif_slope=32, serif_tip=3, top_serif_slope=30, top_serif_tip=7,
    cap_serif_slope=35, cap_serif_tip=5, foot_wide_dr=20, spur_rise=33, spur_drop=48,
    spur_flat=31, spur_corner=26, spur_ca=0.24, spur_cb=0.76, spur_sweep=1.5, spur_tip=1,
    # o
    o_full_h=0.71, o_full_v=0.42, o_in_full_h=0.46, o_in_full_v=0.79,
    # n m h
    n_width=325, n_join_at=0.8, n_join_thin=26, n_join_angle=47, n_shoulder_x=0.59, n_out_y=0.41,
    n_taper_len=0.26, n_bias=0.53, n_tension=1.05, m_width=300, n_shoulder_pa=-5, n_leg_pa=7, n_leg_top=0.94,
    n_leg_t=0.75, n_notch=0.6, n_notch_w=24, h_dwidth=-5, h_top=10,
    # u
    u_width=330, u_join_at=0.77, u_join_thin=26, u_join_angle=50, u_shoulder_x=0.51, u_out_y=0.33,
    u_tension=0.96, u_head_dlen=4.0, u_head_dtop=0.47, u_leg_t=0.95,
    # r
    r_join_at=0.68, r_join_thin=10, r_join_angle=75, r_taper_len=0.64, r_end_w=1.27, r_end_pa=4, r_rsb=10,
    r_width=210, r_beak_depth=190, r_beak_len=110, r_cut_angle=4, r_dir=-5, r_y=0.92,
    # a
    a_hood_angle=-5, a_far=50, a_top_thin=36, a_top_angle=14, a_bot_at=157, a_bot_thin=12, a_bot_angle=71,
    a_lobe_thick=93, a_lobe_thin_b=62, a_out_mid=0.39, a_in_mid=0.41, a_bx_out=0.53, a_bx_in=1.0, a_top_pa=3,
    a_term_pa=7, a_hood_end=0.68, a_hood_thin=0.87, a_stem_top=363, a_stem_up=21, a_stem_x=426, a_taper=0.97,
    a_term_dir=104, a_term_x=118, a_term_y=372, a_out_y=0.8,
    # b
    b_far=410, b_top_at=442, b_top_thin=26, b_top_angle=-40, b_bot_at=113, b_bot_thin=52, b_bot_angle=49,
    b_lobe_thick=84, b_crown_x=0.36, b_crown_t=59, b_top=14, b_foot=12, b_out_mid=0.52, b_in_mid=0.49,
    b_bx_out=0.74, b_bx_in=0.8, b_foot_angle=-57, b_spur_drop=100, b_spur_len=100, b_spur_rise=98,
    # e
    e_width=475, e_term_y=157, e_cut_angle=188, e_cut_x=0.54, e_cut_h1=0.25, e_cut_h2=0.68, e_full_h=0.74,
    e_full_v=0.48, e_in_full_h=0.4, e_in_full_v=0.78, e_stress=4.0, e_thick_dl=-8.0, e_thick_dr=-4.0,
    e_thin_bottom=2.4,   # the bowl's bottom: 2.4 x hairline
    # s S
    s_width=395, s_t1_x=0.9, s_t1_y=0.8, s_t1_dir=115, s_cut1=-6.0, s_t2_x=0.03, s_t2_y=0.2, s_t2_dir=123,
    s_cut2=3.0, s_top_x=0.44, s_left_y=0.75, s_mid_x=0.33, s_mid_y=0.54, s_mid_dir=-16, s_right_y=0.29,
    s_bot_x=0.53, s_ends=0.81, s_swell=1.55, s_spread=0.23, s_tension=0.87, S_swell_k=1.05, s_angle=12.0,
    S_t1_dx=0.02, S_t2_dx=-0.06, S_t2_dy=0.04,
    s_bias=0.54, s_t1_dw=-3.0, s_t2_dw=6.0, s_mid_dw=12.0, s_left_dw=-12.0, s_right_dw=-8.0, s_top_dw=3.0,
    s_bot_dw=3.0, s_t1_pa=-3.0, s_left_t=0.95, s_right_t=0.95, s_t2_pa=3.0, s_mid_t=1.05,
    # g
    g_br=440, g_bt=532, g_thick=1.1, g_full=0.55, g_in_full=0.62, g_ear_y=620, g_ear_w=0.45, g_link_x0=0.42,
    g_link_y=106, g_band_x=188, g_band_y=42, g_link_w=1.5, g_loop_l=-2, g_loop_r=508, g_loop_t=96,
    g_loop_thick=0.94, g_loop_tr=1.1, g_loop_top_t=48, g_loop_bot_t=26, g_loop_full=0.5, g_loop_in_full=0.53,
    g_loop_stress=5, g_rot=4.0, g_loop_rot=15.0, g_ear_h=90,
    g_link_root_dir=190, g_link_root_dy=5.0, g_link_w0=0.7, g_link_w1=1.9, g_low_y=0.47,
    g_link_x=60, g_link_tp=0.2, g_low_t=0.3, g_ear_root_x=0.4, g_ear_root_dx=5, g_ear_root_y=510,
    # the notch where the link meets the band
    g_notch_len=90, g_notch_x=120, g_notch_y=20, g_notch_up=23, g_notch_dn=19,
    # t
    t_turn_w=1.05, t_angle=30, t_bar_t=40, t_tail=246, t_tail_y=120, t_end_dir=0, t_end_w=0.28, t_turn=133,
    t_bar_l=126, t_bar_r=238, t_top=654, t_wedge_sweep=45, t_wedge_len=5.0, t_wedge_rise=88, t_wedge_drop=0.0,
    t_wedge_flat=35,
    # f
    f_bar_t=40, fi_dx=175, f_end_w=0.96, f_end_dir=-7, f_hook=211, f_bar_l=132, f_bar_r=202, f_beak_depth=173,
    f_beak_len=106, f_end_y=0.94, f_top=750, f_top_dw=-27, f_bar_dy=-0.23,
    # i v
    v_sb=-88, v_width=440, i_dot_y=662, i_dot_r=56, v_thick=88, v_thin=35, v_thin_dx=-20, v_vertex=0.53,
    # capitals
    H_width=450, H_bar_t=36, G_width=720, G_term_y=0.68, G_bar_y=0.47, G_serif=75,
    A_width=550, A_sb=-100, E_width=396, A_thick=105, A_thin=34, A_apex=20, O_width=690,
    E_mid=0.77, E_arm_t=35, E_serif=100, E_serif_w=30, E_wedge=80, A_bar_y=0.36, G_jaw_in=0.4, G_jaw_dx=-95,
    G_spur_bot=0.04, G_head_bracket=30, G_head_slope=45, G_face=0.86,
)
kit.define(font, P, glyphs="hamburgefonstivHOAGES ", terminals=kit.bracketed_set(P), g_ear="flag",
           g_band="flat", b_style="lobe", E_serifs="wedge", G_jaw="serif")

# side bearings (left, right) on top of sb_straight / sb_round, set from the
# glyphs' measured side bearings in the overlays
font.spacing.update({
    "h": (8, -6),
    "a": (4, -6),
    "m": (8, -6),
    "b": (-6, -2),
    "u": (-6, 4),
    "r": (8, -48),
    "g": (-42, -26),
    "e": (8, 22),
    "f": (-28, -42),
    "o": (8, 8),
    "n": (8, -6),
    "s": (6, -4),
    "t": (-40, -36),
    "i": (-20, -24),
    "v": (0, 22),
    "H": (4, 4),
    "O": (10, 10),
    "A": (24, 24),
    "G": (10, -38),
    "E": (4, 12),
    "S": (10, 10),
})
# the r's flag overhangs round letters, the f's hook overhangs o
for left, right, v in (("r", "o", -25), ("r", "e", -25), ("r", "s", -20), ("f", "o", -40)):
    font.kern(left, right, v)
