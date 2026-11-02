"""Ductus Sans -- a grotesque sans-serif from the same compiler and kit as
Ductus Display.

The serif kit's constructions (arches, lobes, bowls, spines, diagonals,
composites) are style-neutral; the serifs live in the terminal set. This
face swaps in ``kit.sans_set`` (every stroke end cut flat), drops the
contrast to Geist Sans's (stems 84, horizontals ~76), and takes a
single-storey g. Values are tuned toward Geist Sans (OFL-1.1, Vercel) from
overlays and the bench's IoU, then rounded; no outlines are copied.
"""
from ductus import Font
from ductus.kits import serif as kit

font = Font("Ductus Sans", upm=1000)

P = kit.params(
    # weight and contrast: near monoline
    stem=84, cap_stem=84, hairline=79, round_gain=1.03, arch_contrast=0.1, t_contrast=0.1,
    sb_straight=80, sb_round=44, overshoot=12,
    # no serifs: the E's arm "serifs" sit inside the arms
    E_serif=34, E_serif_w=10, E_arm_t=84, H_bar_t=108, bar_t=76, t_bar_t=76, f_bar_t=76, e_bar_t=74,
    E_width=405,
    E_mid=0.965,
    E_bot_k=1.02,
    # widths (stem centres), measured off Geist Sans
    n_width=337, u_width=337, m_width=300, o_width=485, e_width=473, c_width=470, s_width=432,
    H_width=445, O_width=649,
    # monoline: the serif kit's hairline parts at near-stem weight
    v_thick=88, v_thin=80, w_thick=88, w_thin=80, x_thick=88, x_thin=80, k_thick=88, k_thin=80,
    A_thick=84, A_thin=84, V_thick=90, V_thin=82, X_thick=90, X_thin=82, Y_thick=90, Y_thin=82,
    A_width=536,
    A_apex=46,
    A_bar_y=0.34,
    A_os_k=0.9,
    K_thick=90, K_thin=82, M_thin=84, M_thick=88, N_thin=84, N_thick=90, U_thin=84, U_thick=88,
    z_thick=88, Z_thick=90, four_thin=84, R_leg_w=90,
    n_join_thin=70, u_join_thin=70, r_join_thin=60, a_top_thin=70, a_bot_thin=70, b_top_thin=70,
    b_bot_thin=70, d_top_thin=70, d_bot_thin=70, a_lobe_thin_b=76, b_lobe_thin_b=76, d_lobe_thin_b=76,
    a_lobe_thick=88, b_lobe_thick=88, d_lobe_thick=88, b_crown_t=76, d_crown_t=76,
    s_ends=1.0, s_swell=1.05, z_bar_t=76, S_swell_k=1.0, e_thin_bottom=1.0, C_right_k=1.0, C_bottom_k=1.0,
    O_thin_k=1.0, fig_thin_k=1.0, B_hair=76, B_th=88, D_th=88,
)
kit.define(font, P, glyphs="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,:;-'\"!?()/",
           terminals=kit.sans_set(P), g_style="single", g_ear="flag", b_style="lobe")
