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
    t_contrast=0.3, bar_t=38, bar_taper=0.87, bar_tip_dx=3, bar_tip_y=0.42,
    # serifs and stroke ends
    serif_len=83, serif_bracket=70, serif_ca=0.41, serif_cb=0.85, foot_tip=60, cap_serif_len=85,
    cap_serif_bracket=64, head_len=73, head_top=15, top_serif_bracket=58, spur_len=79, beak_len=95,
    beak_depth=14, ball_r=48, ball_drop=-22, G_spur_w=1.0, head_rise=55, head_drop=36, head_hollow=0.62,
    head_tip=13, head_flat=8, head_bracket=0.5, head_corner=19, head_ca=0.76, head_cb=0.59, head_inset=4,
    head_inset_y=24, serif_cup=-0.67, serif_slope=32, serif_tip=3, top_serif_slope=30, top_serif_tip=7,
    top_serif_len=52, top_serif_in=1.8,
    cap_serif_slope=35, cap_serif_tip=5, foot_wide_dr=20, spur_rise=33, spur_drop=48,
    cap_stem=92,
    spur_flat=31, spur_corner=26, spur_ca=0.24, spur_cb=0.76, spur_sweep=1.5, spur_tip=1,
    # o
    o_full_h=0.72, o_full_v=0.42, o_in_full_h=0.44, o_in_full_v=0.79,
    # n m h
    n_width=325, n_join_at=0.8, n_join_thin=25, n_join_angle=53, n_shoulder_x=0.59, n_out_y=0.41,
    n_taper_len=0.27, n_bias=0.52, n_tension=1.03, m_width=300, n_shoulder_pa=-5, n_leg_pa=7, n_leg_top=0.94,
    n_leg_t=0.75, n_notch=0.6, n_notch_w=24, h_dwidth=-5, h_top=13.0,
    # u
    u_width=310, u_join_at=0.77, u_join_thin=29, u_join_angle=50, u_shoulder_x=0.55, u_out_y=0.35,
    u_tension=0.9, u_head_dlen=8.0, u_head_dtop=0.57, u_leg_t=0.82,
    u_bias=0.51, u_angle=40, u_taper_len=0.32,
    # r
    r_join_at=0.68, r_join_thin=8.0, r_join_angle=75, r_taper_len=0.58, r_end_w=1.19, r_end_pa=4, r_rsb=10,
    r_width=210, r_beak_depth=190, r_beak_len=110, r_cut_angle=4, r_dir=-7.0, r_y=0.92,
    r_head_dinset=4.0,
    # a
    a_hood_angle=-8.0, a_far=50, a_top_thin=39, a_top_angle=14, a_bot_at=150, a_bot_thin=12, a_bot_angle=71,
    a_lobe_thick=93, a_lobe_thin_b=60, a_out_mid=0.41, a_in_mid=0.45, a_bx_out=0.53, a_bx_in=1.0, a_top_pa=3,
    a_term_pa=7, a_hood_end=0.71, a_hood_thin=0.87, a_stem_top=358, a_stem_up=21, a_stem_x=426, a_taper=0.97,
    a_term_dir=100, a_term_x=118, a_term_y=372, a_out_y=0.8,
    a_top_at=280,
    # b
    b_far=415, b_top_at=442, b_top_thin=29, b_top_angle=-40, b_bot_at=108, b_bot_thin=52, b_bot_angle=49,
    b_lobe_thick=90, b_crown_x=0.39, b_crown_t=59, b_top=14, b_foot=12, b_out_mid=0.52, b_in_mid=0.5,
    b_bx_out=0.68, b_bx_in=0.88, b_foot_angle=-59, b_spur_drop=100, b_spur_len=100, b_spur_rise=98,
    b_lobe_thin_b=34,
    # e
    e_width=475, e_term_y=157, e_cut_angle=185, e_cut_x=0.54, e_cut_h1=0.25, e_cut_h2=0.68, e_full_h=0.72,
    e_full_v=0.48, e_in_full_h=0.4, e_in_full_v=0.78, e_stress=4.0, e_thick_dl=-8.0, e_thick_dr=-2.0,
    e_thin_bottom=2.45,   # the bowl's bottom: 2.4 x hairline
    e_bar_t=33, e_bar_rin=4.0,
    # s S
    s_width=395, s_t1_x=0.9, s_t1_y=0.805, s_t1_dir=115, s_cut1=-6.0, s_t2_x=0.03, s_t2_y=0.21, s_t2_dir=123,
    s_cut2=3.0, s_top_x=0.45, s_left_y=0.75, s_mid_x=0.355, s_mid_y=0.54, s_mid_dir=-16, s_right_y=0.29,
    s_bot_x=0.53, s_ends=0.81, s_swell=1.55, s_spread=0.23, s_tension=0.87, S_swell_k=0.97, s_angle=12.0,
    S_t1_dx=0.03, S_t2_dx=-0.055, S_t2_dy=0.04,
    S_mid_dx=0.015, S_right_dy=-0.05, S_bot_dx=-0.06, S_top_dx=0.09, S_t1_dy=-0.035,
    S_left_dy=0.035, S_t1_dw=15, S_mid_dw=-4, S_right_dw=-4, S_bot_dw=-5, S_t2_dw=11.0, S_width=505,
    S_top_dw=6, S_mid_dy=-0.005,
    s_bias=0.54, s_t1_dw=-3.0, s_t2_dw=6.0, s_mid_dw=12.0, s_left_dw=-9.0, s_right_dw=-8.0, s_top_dw=3.0,
    s_bot_dw=3.0, s_t1_pa=-3.0, s_left_t=0.95, s_right_t=0.95, s_t2_pa=3.0, s_mid_t=1.05,
    s_top_dir=182, s_right_dir=272, s_top_pa=-4.0, s_bot_pa=-4.0, s_left_pa=2.0, s_right_pa=2.0, s_bot_dt=0.03,
    # g
    g_br=443, g_bt=532, g_thick=1.12, g_full=0.55, g_in_full=0.62, g_ear_y=620, g_ear_w=0.45, g_link_x0=0.42,
    g_link_y=109, g_band_x=184, g_band_y=42, g_link_w=1.5, g_loop_l=-2, g_loop_r=508, g_loop_t=96,
    g_loop_thick=0.94, g_loop_tr=1.1, g_loop_top_t=48, g_loop_bot_t=26, g_loop_full=0.46, g_loop_in_full=0.55,
    g_loop_stress=5, g_rot=4.0, g_loop_rot=15.0, g_ear_h=90,
    g_link_root_dir=190, g_link_root_dy=5.0, g_link_w0=0.7, g_link_w1=1.9, g_low_y=0.47,
    g_link_x=57, g_link_tp=0.2, g_low_t=0.3, g_ear_root_x=0.4, g_ear_root_dx=5, g_ear_root_y=510,
    # the notch where the link meets the band
    g_notch_len=90, g_notch_x=120, g_notch_y=20, g_notch_up=23, g_notch_dn=19,
    g_ear_x=505, g_tr_dh=0.0, g_tr_dv=-0.21,
    g_full_dv=0.06,
    # t
    t_turn_w=1.0, t_angle=29, t_bar_t=40, t_tail=255, t_tail_y=99, t_end_dir=0, t_end_w=0.38, t_turn=133,
    t_bar_l=126, t_bar_r=238, t_top=654, t_wedge_sweep=45, t_wedge_len=5.0, t_wedge_rise=94, t_wedge_drop=0.0,
    t_wedge_flat=35, t_tail_cut=-3.0,
    # f
    f_bar_t=40, fi_dx=175, f_end_w=0.91, f_end_dir=-9.0, f_hook=211, f_bar_l=132, f_bar_r=202, f_beak_depth=173,
    f_beak_len=106, f_end_y=0.94, f_top=750, f_top_dw=-23, f_bar_dy=-0.23,
    # i v
    v_sb=-88, v_width=438, i_dot_y=662, i_dot_r=56, v_thick=85, v_thin=37, v_thin_dx=-20, v_vertex=0.53,
    # capitals
    H_width=450, H_bar_t=36, G_width=717, G_term_y=0.67, G_bar_y=0.475, G_serif=81,
    H_bar_y=0.515,
    A_width=560, A_sb=-100, E_width=393, A_thick=100, A_thin=35, A_apex=20, O_width=687,
    O_thick_k=1.08, O_thin_k=1.1, O_full_dh=0.02,
    O_in_full_dh=-0.01,
    E_mid=0.77, E_arm_t=35, E_serif=144, E_serif_w=15.0, E_wedge=110, A_bar_y=0.34, G_jaw_in=0.35, G_jaw_dx=-95,
    A_bracket_k=1.8, A_foot_dl=-20, E_mid_y=0.51,
    E_bracket_k=1.95, E_mid_sk=0.57, E_bot_sk=0.96, E_bot_k=1.03, A_os_k=0.0, A_foot_in_dl=80,
    G_spur_bot=0.035, G_head_bracket=33, G_head_slope=48, G_face=0.855,
    G_full_dh=0.03, G_in_full_dh=-0.03, G_stress=3, G_thick_k=0.955, stem=84,
    G_in_full_dv=0.01, G_thin_k=1.05,
    slash_run=300, paren_w=84, paren_top=745, paren_bot=-145, paren_bow=190, paren_end_w=0.3, paren_end_dir=29,
    paren_tension=1.2, comma_bend=62, quote_gap=176, quest_width=455, quest_w=95, quest_neck_x=0.52,
    quest_neck_y=266, quest_bot_w=0.4, quest_tension=0.85, quest_thin=46, quest_end_y=463, quest_top_x=0.53,
    quest_right_y=506, quest_neck_dir=250, quest_bot_dx=-15, semi_dx=12,
    d_far=420, d_top_at=460, d_top_thin=58, d_bot_at=100, d_bot_thin=26, d_lobe_thick=90, d_lobe_thin_b=58,
    d_out_mid=0.44, d_in_mid=0.56, d_bx_out=0.7, d_bx_in=1.0, d_crown_x=0.35, d_crown_t=26, d_top_angle=45,
    q_top=-20, c_width=485, c_face=0.94, c_top_y=350, c_term_y=140, c_cut_x=0.62, c_cut_angle=165, j_bot=-28,
    j_hook=168, j_end_y=-172, j_end_w=1.4,
    j_end_dir=142,
   
   
   
   
   
   
   
    w_width=690, w_thick=84, w_thin=40, w_v1=0.28,
    w_mid3_dx=-8.0, w_v2=0.74,
    w_a2_top=0.86, w_thin_dx=-5.0,
   
   
    y_width=414, y_vertex=0.47, y_tail_y=-120, y_end_x=-54, y_end_y=-173, y_end_w=2.0, z_width=390, z_thick=98,
    z_serif=86, z_wedge=104, z_bar_t=37,
    k_arm_x=326, k_join_y=161, x_width=368, x_thick=84, x_thin=34, x_thin_dx=-48,
    diag_foot_len=41, diag_foot_in=2.3,
    B_flat_t=208, B_flat_b=212, D_flat=160, D_th=100, P_bot=292, R_leg_x0=211, R_leg_x=409, R_leg_w=94,
    B_hair=34, B_th=95,
    B_mid=0.512, B_flat_m=158, B_in_full=0.66,
    C_width=679, C_face=0.9, C_top_y=470, C_term_y=160, C_cut_x=0.53, C_cut_y=68, C_cut_angle=150, C_cut_h2=0.4,
    C_thick_k=0.94, C_stress=6.0, C_in_full_dh=0.04, C_in_full_dv=-0.08, C_right_k=0.6, C_bottom_k=1.6,
    Q_tip_x=620, F_width=371, F_mid=0.81, F_mid_y=0.5, L_width=390, T_width=582, T_wedge=108, T_serif=144,
    J_bot=-58, J_end_y=-202, J_end_w=1.6, J_end_dir=181, K_arm_x=428, K_join_y=282, K_leg_x=426, K_leg_y0=400,
    K_leg_x0=103, K_thin=34, K_thick=94, M_width=576, M_thin=36, M_thick=95, N_width=512, U_join=220,
    N_dx0=-50, N_dx1=62, N_thick=95, M_fall_dx=40, M_rise_dx=-50, M_vdx=-40, M_vertex=0.54,
    U_end_w=0.22, V_width=512, V_thick=101, V_thin=34, V_vertex=0.52, V_thin_dx=-20, W_width=880, W_v1=0.29,
    W_mid_dx=0, W_mid3_dx=-18, W_v2=0.72, X_width=464, X_thin=42, X_thin_dx=-36, Y_width=520, Y_join=0.39,
    W_a2_top=0.85, W_thin_dx=-8,
    Y_thick=96, Z_thick=113, Z_serif=134, Z_wedge=136, Z_width=508,
    fig_stem=88, fig_thin_k=1.1, zero_w=557, one_fy=634, one_flag=166, two_w=472, two_right_y=535,
    two_mid_x=0.41, two_mid_y=266, two_mid_dir=210, two_bar_l=-25, two_bar_r=478, two_serif=186, two_wedge=96,
    three_w=472, three_mid_y=388, three_mid_x=0.33, three_lr_y=203, three_end_x=0.06, three_end2_y=118,
    three_end_dir=122, three_bot_x=0.47, three_ur_y=542, four_x=392, four_bar_r=562, four_thin=36, four_top=693,
    five_w=536, five_stem_bot=371, five_stem_w=48, five_end_y=62, five_end_dir=159, six_w=499, six_hook_x=0.91,
    six_face_y=561, six_wedge=108, six_hook_w=0.75, six_bowl_h=458, seven_w=500, seven_foot_l=147,
    seven_foot_r=271, seven_serif=211, seven_wedge=120, eight_top_w=446, eight_waist=372, eight_thick_k=0.9,
    eight_sp_x0=0.08, eight_sp_x1=0.92, eight_sp_w=0.65, eight_sp_y0=530, nine_bowl_h=466, nine_face_y=561,
    nine_hook_w=0.75, nine_wedge=108, nine_w=499, nine_hook_x=0.93,
    acute_dx=60, circ_in=45, tilde_len=286, tilde_w=54, tilde_lo=622, tilde_dir=60, ring_thick=47, ced_len=165,
    ced_drop=216, ced_neck=-45, ced_w=70, ced_thin=38, acc_dx_c=20, acc_dx_y=-6.0, acc_dx_a=-8.0,
    acc_dx_A=-18.0, acc_dx_U=40, acc_dx_E=45, acc_dx_C=35, math_t=34, times_w=364, lt_w=444, lt_h=426, lt_t=54,
    lt_dy=8.0, lt_end_k=1.2,
    ellipsis_gap=200, brk_w=245, brk_stem=82, brk_arm=34, brace_w=350, brace_pen=82, brace_x=0.47, hash_w=419,
    brace_end_w=1.4, brace_beak_w=1.7, brace_beak_x=22,
    hash_slant=132, hash_t=59, hash_gap=206, hash_bar_t=53, pct_w=696, pct_ring_w=288, pct_ring_h=354,
    pct_slash_w=59, deg_w=274, deg_t=47, dollar_bar=36, ast_r=171, ast_y=551, ast_tip_w=55, ast_root_w=14.0,
    pct_s0=80, pct_s1=70, dollar_h=660, dollar_y0=24,
    at_w=899, at_bot=-190, at_ring_t=58, at_in_w=340, at_in_h=450, at_in_y=42, at_h=945, at_in_x=0.27,
    at_slant=14.0, at_root_dir=348, at_end_x=0.82, at_end_y=0.09, at_end_dir=22, at_pen_k=1.4,
    amp_w=560, amp_arm_x=0.98, amp_arm_y=378, amp_loop_x=0.47, amp_loop_w=300, amp_waist_y=382, amp_leg_x=0.9,
    amp_pen=98, euro_x0=62, euro_w=635, euro_bar_l=-13.0, euro_y0=272, euro_y1=424,
    amp_loop_lx=-40, amp_left_y=200, amp_waist_dir=15.0, amp_leg_dir=294,
    eq_dy=-4.0, eq_t=37, eq_gap=150,
    oslash_w=56, oslash_x0=-6.0, oslash_os=8.0, Oslash_x0=14.0, AE_x=522, AE_top_x=469, AE_thick=62,
    AE_top_l=56, AE_top_w=392, AE_bot_w=411, OE_x=497, OE_dw=0, OE_top_w=392, OE_mid_w=306, OE_bot_w=406,
    OE_right_k=0.2,
    ae_dx=333, oe_dx=417, ss_top=494, ss_top_x=192, ss_lr_y=128, ss_bot_x=305, ss_end_y=112, ss_waist_x=160,
    ss_waist_y=372, tl_len=262, tl_amp=70, tl_w=58, tl_dir=80,
)
kit.define(font, P, glyphs="hamburgefonstivcdjklpqwxyz HOAGESBCDFIJKLMNPQRTUVWXYZ0123456789ÀÁÂÃÄÅÇÈÉÊËÌÍÎÏÑÒÓÔÕÖÙÚÛÜÝàáâãäåçèéêëìíîïñòóôõöùúûüýÿı´`¨¸˚ÆŒØæœøß&*%·#$+<=>@[\\]^_{|}~…•°×€£.,:;-–—'\"‘’“”!?()/", terminals=kit.bracketed_set(P), g_ear="flag",
           g_band="flat", b_style="lobe", E_serifs="wedge", G_jaw="serif")

# side bearings (left, right) on top of sb_straight / sb_round, set from the
# glyphs' measured side bearings in the overlays
font.spacing.update({
    "germandbls": (-2, -16),
    "oslash": (8, -8),
    "Oslash": (10, 10),
    "OE": (10, 12),
    "AE": (-76, 12),
    "Euro": (-8, 20),
    "multiply": (60, 60),
    "degree": (-20, -20),
    "bullet": (-10, -10),
    "ellipsis": (-18, -18),
    "asciitilde": (76, 76),
    "braceright": (-32, -6),
    "bar": (26, 28),
    "braceleft": (-6, -32),
    "underscore": (40, 40),
    "bracketright": (-4, 76),
    "backslash": (-22, -84),
    "bracketleft": (76, -4),
    "at": (10, 10),
    "greater": (34, 36),
    "equal": (80, 80),
    "less": (36, 34),
    "plus": (80, 80),
    "dollar": (16, 10),
    "numbersign": (16, -16),
    "periodcentered": (-8, -8),
    "percent": (-28, -28),
    "asterisk": (12, 12),
    "ampersand": (-2, -36),
    "nine": (0, -2),
    "eight": (-2, -8),
    "seven": (8, -20),
    "six": (-2, 0),
    "five": (8, 0),
    "four": (-4, -8),
    "three": (28, 16),
    "two": (40, 26),
    "one": (-12, 34),
    "zero": (-22, -22),
    "Z": (-12, -18),
    "Y": (-60, -60),
    "X": (-38, -62),
    "W": (28, 28),
    "V": (28, 28),
    "R": (4, -36),
    "Q": (10, -24),
    "P": (4, -16),
    "N": (-22, -26),
    "M": (4, 4),
    "L": (4, 12),
    "K": (4, -36),
    "J": (-116, 0),
    "I": (4, 4),
    "F": (4, 12),
    "D": (4, 10),
    "C": (10, -2),
    "B": (4, -22),
    "z": (12, 12),
    "y": (0, 22),
    "x": (-40, -40),
    "w": (0, 8),
    "q": (-2, -36),
    "p": (4, -2),
    "l": (2, 4),
    "k": (8, -38),
    "j": (-76, 20),
    "d": (0, 4),
    "c": (8, -4),
    "slash": (-28, -80),
    "parenright": (-20, 18),
    "parenleft": (18, -20),
    "question": (20, 20),
    "exclam": (-8, -6),
    "quotedblright": (-6, -8),
    "quotedblleft": (-8, -8),
    "quoteright": (-10, -14),
    "quoteleft": (-14, -12),
    "quotedbl": (-12, -12),
    "quotesingle": (-12, -12),
    "emdash": (28, 28),
    "endash": (40, 40),
    "hyphen": (36, 36),
    "semicolon": (-8, -8),
    "colon": (-8, -8),
    "comma": (-8, -8),
    "period": (-8, -8),
    "space": (0, 44),   # a 250 u word space
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
    "s": (6, -6),
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
# kerning: measured from the outlines by `ductus kern` (optical auto-kerning;
# regenerate after changing shapes or spacing). Pairs set by hand below it
# would override it.
import json as _json
from pathlib import Path as _Path

_kern = _Path(__file__).with_name("kerning.json")
if _kern.exists():
    for _l, _r, _v in _json.loads(_kern.read_text())["pairs"]:
        font.kern(_l, _r, _v)
