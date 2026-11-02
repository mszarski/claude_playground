"""High-contrast display serif kit (SPEC M2).

Constructions for the benchmark glyphs (o n H e a s g v) and the test word
(hamburgefonstiv, H O A G E S), written once against a parameter namespace
``P``. A font calls :func:`params` with its own values, picks a terminal set
(:func:`bracketed_set`, :func:`didone_set` or its own) and calls
:func:`define`.

Glyph-specific parameters are prefixed with the glyph name (``n_width``),
shared ones are not (``stem``, ``serif_len``). Every construction keeps a
fixed point structure: parameters only move points.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

from ..params import Param, lazy_max, lazy_min, val
from ..pen import Pen
from ..profiles import Profile, constant
from ..terminals import TerminalSet, beak, bracketed, flat, none, wedge

# name: (default, (lo, hi))  -- generic genre defaults, not fitted to anything
DEFAULTS: dict[str, tuple] = {
    # metrics
    "xheight": (530, (480, 580)), "capheight": (710, (660, 760)), "ascender": (750, (700, 800)),
    "overshoot": (12, (0, 25)), "head_top": (14, (0, 40)),
    # weight / contrast
    "stem": (86, (50, 140)), "cap_stem": (94, (50, 150)), "hairline": (30, (6, 70)),
    "round_gain": (1.15, (0.9, 1.4)),
    "arch_angle": (30, (-20, 70)), "arch_contrast": (0.6, (0.0, 0.95)),
    # spacing
    "sb_straight": (25, (0, 120)), "sb_round": (45, (0, 120)),
    # serifs
    "serif_len": (75, (10, 160)), "serif_bracket": (60, (0, 140)), "serif_tip": (4, (0, 40)),
    "serif_ca": (0.6, (0.0, 1.0)), "serif_cb": (0.6, (0.0, 1.0)), "serif_cup": (0, (-10, 20)),
    "serif_slope": (0, (-10, 50)), "cap_serif_slope": (0, (-10, 50)),
    "cap_serif_len": (80, (10, 180)), "cap_serif_bracket": (60, (0, 160)), "cap_serif_tip": (4, (0, 40)),
    "head_len": (80, (10, 160)), "head_rise": (60, (0, 140)), "head_drop": (55, (0, 140)), "head_hollow": (0.3, (0, 1)),
    # head wedge fine shape (see terminals.wedge): flat tip, flat top, bracketed underside
    "head_tip": (0, (0, 20)), "head_flat": (0, (0, 40)), "head_bracket": (0, (0, 1)),
    "head_corner": (20, (0, 80)), "head_ca": (0.7, (0.0, 1.0)), "head_cb": (0.5, (0.0, 1.0)),
    "g_full_dv": (0, (-0.3, 0.3)), "g_in_full_dv": (0, (-0.3, 0.3)), "g_rot": (0, (-15, 15)),  # upper bowl shape
    "G_spur_bot": (0, (0, 0.4)), "G_jaw_in": (0, (0, 1)), "G_jaw_dx": (0, (-120, 60)),
    # G_jaw="serif": the jaw's head serif (bracket, slope) and its foot, cut off
    # on a diagonal (G_cut: the cut's normal, degrees) down to a point
    "G_face": (0.92, (0.7, 1.0)),   # x of the terminal's vertical face, share of the width
    "G_head_bracket": (40, (0, 120)), "G_head_slope": (30, (0, 60)), "G_cut": (45, (15, 75)),
    # the S's arm ends, relative to the s's (shares of the width / height):
    # reaching further round, its near-vertical end cuts become tall faces
    "S_t1_dx": (0, (-0.2, 0.2)), "S_t1_dy": (0, (-0.2, 0.2)), "S_t2_dx": (0, (-0.2, 0.2)), "S_t2_dy": (0, (-0.2, 0.2)),
    "S_mid_dx": (0, (-0.15, 0.15)), "S_mid_dy": (0, (-0.15, 0.15)), "S_left_dy": (0, (-0.15, 0.15)),
    "S_right_dy": (0, (-0.15, 0.15)), "S_top_dx": (0, (-0.15, 0.15)), "S_bot_dx": (0, (-0.15, 0.15)),
    # the S's stroke width at each node, added to the s's (font units)
    "S_t1_dw": (0, (-40, 40)), "S_top_dw": (0, (-40, 40)), "S_left_dw": (0, (-40, 40)), "S_mid_dw": (0, (-40, 40)),
    "S_right_dw": (0, (-40, 40)), "S_bot_dw": (0, (-40, 40)), "S_t2_dw": (0, (-40, 40)),
    "G_beard_dw": (0, (-10, 80)), "G_beard_k": (0.6, (0.1, 3.0)), "G_spur_w": (1.0, (0.5, 1.2)),
    "a_stem_w": (1.0, (0.85, 1.15)), "a_stem_bot": (0, (-20, 20)),
    "e_thick_dl": (0, (-40, 30)), "e_thick_dr": (0, (-40, 30)), "e_bar_rin": (0, (-10, 40)),
    "g_notch_x": (170, (100, 260)), "g_notch_y": (22, (-40, 80)), "g_notch_len": (0, (0, 160)),
    "g_notch_up": (19, (0, 50)), "g_notch_dn": (23, (0, 60)),
    "g_tr_dh": (0, (-0.4, 0.3)), "g_tr_dv": (0, (-0.4, 0.3)),
    "g_ear_root_dx": (0, (-60, 30)),   # the ear's sloped top starts on the bowl, left of its foot
    "g_link_root_dy": (0, (-40, 80)), "g_link_root_dir": (210, (170, 260)),   # where the link leaves the bowl
    "t_end_dir": (60, (10, 90)),   # direction the t's tail leaves at its end
    "t_tail_cut": (0, (-30, 85)),   # normal of the cut across the t's tail end (0: vertical cut)
    "t_wedge_flat": (0, (0, 60)), "t_wedge_sweep": (0, (-20, 60)),   # the t's flat top and concave flank
    "h_dwidth": (0, (-40, 40)),   # the h's arch relative to the n's width
    # where a leg stem stops, as a fraction of n_out_y: 1 = as before; ~0.55 hides
    # the stem's top corner under the arch, which is not yet vertical higher up
    "f_angle_d": (0, (-40, 40)),
    "fi_dx": (0, (-80, 260)),   # the fi ligature's i stem, from under the f's hook end
    "r_rsb": (0, (-80, 40)),   # the r's right sidebearing, added to the side's default (its ear overhangs)
    "n_leg_top": (1.0, (0.5, 1.0)), "u_leg_top": (1.0, (0.5, 1.0)),   # (the u's left stem, from below)
    "n_shoulder_dw": (0, (-30, 30)), "n_leg_t": (1.0, (0.7, 1.6)),   # arch: weight at the shoulder, turn into the leg
    "u_head_dlen": (0, (-40, 60)), "u_head_dtop": (0, (-30, 30)),   # the u's heads relative to n's
    # Geist's stem is set in on the right between an arch/ear join and the
    # head: a strip `head_inset` wide is cut from `head_inset_y` above the
    # join's crotch to the top (0: no inset)
    "head_inset": (0, (0, 20)), "head_inset_y": (40, (0, 140)), "r_head_dinset": (0, (-10, 15)),
    "spur_len": (70, (0, 160)), "spur_rise": (60, (0, 140)), "spur_drop": (40, (0, 120)),
    # spur fine shape, as for the head wedge (all degenerate at their defaults)
    "spur_tip": (0, (0, 20)), "spur_flat": (0, (0, 40)), "spur_bracket": (0, (0, 1)),
    "spur_corner": (20, (0, 80)), "spur_ca": (0.7, (0.0, 1.0)), "spur_cb": (0.5, (0.0, 1.0)),
    "spur_sweep": (0, (-20, 40)),
    # the G's spur, apart from the u's / b's
    "G_spur_len": (70, (0, 160)), "G_spur_rise": (60, (0, 140)), "G_spur_drop": (40, (0, 160)),
    "beak_len": (25, (0, 120)), "beak_depth": (70, (5, 200)),
    "top_serif_len": (60, (0, 160)), "top_serif_bracket": (40, (0, 140)), "top_serif_tip": (10, (0, 60)),
    "top_serif_in": (1.0, (0.0, 1.5)), "top_serif_slope": (0, (-10, 50)),
    # o
    "o_width": (505, (380, 640)), "o_full_h": (0.6, (0.4, 0.85)), "o_full_v": (0.6, (0.4, 0.85)),
    "o_in_full_h": (0.6, (0.4, 0.85)), "o_in_full_v": (0.6, (0.4, 0.85)), "o_stress": (0, (-20, 20)),
    # n
    "n_width": (410, (300, 520)),
    "n_join_at": (0.78, (0.45, 0.95)), "n_join_thin": (30, (4, 90)), "n_join_angle": (45, (0, 85)),
    "n_notch": (0, (0, 30)), "n_notch_w": (20, (4, 60)),
    "n_shoulder_x": (0.45, (0.2, 0.75)), "n_out_y": (0.62, (0.3, 0.9)),
    "n_taper_len": (0.3, (0.02, 0.8)), "n_bias": (0.5, (0.1, 0.9)),
    "n_tension": (1.0, (0.6, 1.8)),
    # H
    "H_width": (550, (420, 700)), "H_bar_y": (0.52, (0.35, 0.65)), "H_bar_t": (35, (5, 80)),
    # e
    "e_width": (470, (380, 600)), "e_full_h": (0.6, (0.4, 0.85)), "e_full_v": (0.6, (0.4, 0.85)),
    "e_in_full_h": (0.6, (0.4, 0.85)), "e_in_full_v": (0.6, (0.4, 0.85)), "e_stress": (0, (-25, 25)),
    "e_thin_bottom": (1.0, (0.5, 2.5)),
    "e_bar_y": (0.57, (0.4, 0.7)), "e_bar_t": (35, (5, 80)),
    "e_term_y": (150, (40, 260)), "e_cut_angle": (200, (150, 260)), "e_cut_y": (60, (-20, 200)),
    "e_cut_h1": (0.4, (0.05, 1.2)), "e_cut_h2": (0.4, (0.05, 1.2)), "e_cut_x": (0.5, (0.25, 0.85)),
    # a
    "a_stem_x": (420, (300, 520)), "a_stem_top": (360, (250, 500)),
    "a_term_x": (110, (40, 220)), "a_term_y": (400, (300, 500)), "a_term_dir": (80, (30, 120)),
    "a_top_x": (0.55, (0.3, 0.8)), "a_out_y": (0.6, (0.3, 0.95)),
    "a_bowl_l": (40, (0, 150)), "a_bowl_top": (300, (200, 400)), "a_bowl_thick": (1.0, (0.5, 1.5)),
    "a_bowl_full_h": (0.6, (0.4, 0.85)), "a_bowl_full_v": (0.6, (0.4, 0.85)), "a_bowl_in": (0.6, (0.4, 0.85)),
    "a_bowl_stress": (0, (-30, 30)), "a_bowl_thin": (1.0, (0.4, 3.0)), "a_bowl_rot": (0, (-10, 35)),
    "a_bowl_r": (0.0, (-60, 60)), "a_bowl_tr": (0.4, (0.05, 1.5)), "a_stem_up": (0, (-40, 120)),
    # a's bowl as a lobe on the stem (two joins)
    "a_far": (40, (0, 150)), "a_top_at": (280, (150, 380)), "a_top_thin": (35, (6, 100)),
    "a_top_angle": (20, (-40, 75)), "a_bot_at": (200, (40, 320)), "a_bot_thin": (70, (6, 180)),
    "a_bot_angle": (30, (-40, 85)), "a_lobe_thick": (95, (30, 170)), "a_lobe_thin_b": (34, (6, 100)),
    "a_out_mid": (0.45, (0.05, 0.95)), "a_in_mid": (0.5, (0.05, 0.95)), "a_bx_out": (0.5, (0.1, 0.95)),
    "a_bx_in": (0.5, (0.1, 1.0)), "a_lobe_t": (1.0, (0.6, 1.8)), "a_lobe_ti": (1.0, (0.6, 1.8)),
    "a_hood_end": (0.25, (0.0, 1.0)), "a_hood_angle": (30, (-30, 80)),
    "a_taper": (0.6, (0.05, 1.2)), "a_bias": (0.5, (0.1, 0.9)), "a_hood_thin": (0.6, (0.2, 1.5)),
    # s
    "s_width": (390, (300, 500)), "s_top_x": (0.45, (0.25, 0.7)), "s_left_y": (0.72, (0.55, 0.9)),
    "s_mid_x": (0.5, (0.3, 0.7)), "s_mid_y": (0.45, (0.3, 0.6)), "s_mid_dir": (-25, (-70, 10)),
    "s_right_y": (0.28, (0.1, 0.45)), "s_bot_x": (0.5, (0.25, 0.75)),
    "s_t1_x": (0.95, (0.7, 1.1)), "s_t1_y": (0.75, (0.55, 0.95)), "s_t2_x": (0.05, (-0.1, 0.3)),
    "s_t2_y": (0.22, (0.05, 0.45)), "s_swell": (1.0, (0.5, 1.6)), "s_ends": (0.6, (0.2, 1.4)),
    "s_angle": (0, (-40, 40)), "s_tension": (1.0, (0.6, 1.6)),
    "s_spread": (0.5, (0.08, 0.5)), "s_bias": (0.5, (0.15, 0.85)),
    # local weight at the s's nodes (width added there, interpolated between)
    # the s's extremes may lean (Geist's are not vertical), and every node
    # has its own tension (the top's is s_tension, the bottom's s_tension + s_bot_dt)
    # the nib turns along the s: pen-angle offset at each node (degrees)
    # the nib turns along the arch (n h m), the a's hood, the t's tail
    "n_join_pa": (0, (-35, 35)), "n_shoulder_pa": (0, (-35, 35)), "n_leg_pa": (0, (-35, 35)),
    "a_term_pa": (0, (-35, 35)), "a_top_pa": (0, (-35, 35)),
    "t_bottom_pa": (0, (-35, 35)), "t_end_pa": (0, (-35, 35)),
    "u_join_pa": (0, (-35, 35)), "u_shoulder_pa": (0, (-35, 35)), "u_leg_pa": (0, (-35, 35)),
    "r_join_pa": (0, (-35, 35)), "r_end_pa": (0, (-35, 35)),
    "f_top_pa": (0, (-35, 35)), "f_end_pa": (0, (-35, 35)),
    "s_t1_pa": (0, (-35, 35)), "s_top_pa": (0, (-35, 35)), "s_left_pa": (0, (-35, 35)),
    "s_mid_pa": (0, (-35, 35)), "s_right_pa": (0, (-35, 35)), "s_bot_pa": (0, (-35, 35)),
    "s_t2_pa": (0, (-35, 35)),
    "s_top_dir": (180, (160, 200)), "s_bot_dir": (180, (160, 200)), "s_mid_free": (0, (0, 1)),   # >= 0.5: solve the mid direction
    "s_left_dir": (270, (235, 305)), "s_right_dir": (270, (235, 305)),
    "s_left_t": (1.0, (0.7, 1.6)), "s_mid_t": (1.0, (0.7, 1.6)), "s_right_t": (1.0, (0.7, 1.6)),
    "s_bot_dt": (0, (-0.5, 0.5)), "s_t1_dw": (0, (-30, 40)), "s_t2_dw": (0, (-30, 40)),   # each end's own weight
    "s_top_dw": (0, (-40, 40)), "s_left_dw": (0, (-50, 50)), "s_mid_dw": (0, (-50, 50)),
    "s_right_dw": (0, (-50, 50)), "s_bot_dw": (0, (-40, 40)), "s_beak_len": (40, (0, 160)), "s_beak_depth": (60, (5, 200)), "s_t1_dir": (100, (20, 170)), "s_t2_dir": (100, (20, 170)), "s_cut1": (0, (-60, 60)), "s_cut2": (0, (-60, 60)),
    # g
    "g_bl": (30, (0, 120)), "g_br": (350, (250, 450)), "g_bb": (180, (120, 260)), "g_bt": (530, (480, 560)),
    "g_thick": (1.0, (0.5, 1.4)), "g_full": (0.6, (0.4, 0.85)), "g_in_full": (0.6, (0.4, 0.85)),
    "g_ear_x": (500, (400, 600)), "g_ear_y": (530, (460, 620)), "g_ear_w": (1.0, (0.3, 2.5)),
    "g_link_x": (40, (0, 150)), "g_link_y": (90, (20, 160)),
    "g_band_y": (55, (0, 140)), "g_band_x": (300, (150, 450)),
    "g_loop_l": (10, (-40, 100)), "g_loop_r": (480, (380, 560)), "g_loop_b": (-205, (-240, -150)),
    "g_loop_t": (40, (-40, 120)), "g_loop_w": (1.0, (0.4, 1.6)),
    "g_link_x0": (0.3, (0.0, 0.9)), "g_link_w": (1.0, (0.15, 1.5)),
    "g_ear_h": (80, (20, 160)), "g_ear_root_x": (0.75, (0.4, 1.1)), "g_ear_root_y": (500, (420, 560)),
    "g_ear_bot": (490, (420, 560)), "g_link_turn": (0.5, (0.1, 0.9)),
    "g_link_w0": (0.4, (0.05, 1.5)), "g_link_w1": (1.0, (0.2, 2.0)), "g_link_tp": (0.5, (0.0, 0.9)),
    "g_bar_top": (95, (40, 140)), "g_bar_bot": (12, (-40, 80)), "g_bar_l": (170, (60, 320)),
    "g_bar_r": (450, (300, 560)),
    "g_low_t": (0.35, (0.05, 0.95)), "g_low_y": (0.5, (0.0, 1.0)), "g_low_dir": (-20, (-80, 40)), "g_loop_thick": (0.7, (0.2, 1.4)), "g_loop_tr": (0.9, (0.2, 1.6)),
    "g_loop_top_t": (80, (10, 140)), "g_loop_bot_t": (40, (5, 120)), "g_loop_full": (0.6, (0.4, 0.98)),
    "g_loop_in_full": (0.6, (0.4, 0.98)), "g_loop_rot": (0, (-15, 30)), "g_loop_stress": (0, (-30, 30)),
    # up to ~1: nearly square quadrants, for the g's flat-topped loop band
    "g_loop_full_v": (0.6, (0.4, 0.98)), "g_loop_in_full_v": (0.6, (0.4, 0.98)),
    # v
    "v_width": (560, (420, 680)), "v_thick": (90, (40, 160)), "v_thin": (25, (5, 80)),
    "v_vertex": (0.5, (0.3, 0.7)), "v_os": (12, (0, 30)), "v_sb": (0, (-150, 80)),
    "v_taper": (1.0, (0.3, 1.3)), "v_thin_dx": (0, (-80, 40)),
    "A_sb": (0, (-150, 80)),
    "A_bracket_k": (0.7, (0.2, 3.0)),   # the A's foot brackets, relative to the cap serifs'
    "A_foot_dl": (0, (-60, 60)),        # the A's foot serifs' length, added to the cap serifs'
    "A_foot_in_dl": (0, (-60, 80)),     # ... and the inner serifs' (between the feet), added again
    "A_os_k": (1.0, (0.0, 1.5)),        # the apex's overshoot above the cap height, x overshoot
    # display / test-word glyphs
    "m_width": (330, (250, 450)), "b_width": (500, (400, 620)), "b_bowl_l": (0.3, (0.0, 0.6)),
    "u_width": (410, (300, 520)), "u_bias": (0.5, (0.1, 0.9)),
    # the u's cup has its own join and shoulder (it is not just the n rotated)
    "u_join_at": (0.78, (0.45, 0.95)), "u_join_thin": (30, (4, 90)), "u_join_angle": (45, (0, 85)),
    "u_shoulder_x": (0.45, (0.2, 0.75)), "u_out_y": (0.62, (0.3, 0.9)), "u_taper_len": (0.3, (0.02, 0.8)),
    "u_tension": (1.0, (0.6, 1.8)), "u_angle": (30, (-20, 70)),
    "h_top": (0, (-40, 80)), "h_stem_w": (1.0, (0.85, 1.15)),
    # b as a lobe hanging on the stem's right edge (b_style="lobe")
    "b_far": (500, (380, 640)), "b_top_at": (430, (300, 520)), "b_top_thin": (30, (6, 100)),
    "b_top_angle": (-30, (-75, 40)), "b_bot_at": (110, (20, 260)), "b_bot_thin": (60, (6, 180)),
    "b_bot_angle": (30, (-40, 85)), "b_lobe_thick": (100, (30, 170)), "b_lobe_thin_b": (34, (6, 100)),
    "b_out_mid": (0.5, (0.05, 0.95)), "b_in_mid": (0.5, (0.05, 0.95)), "b_bx_out": (0.5, (0.1, 0.95)),
    "b_bx_in": (0.5, (0.1, 0.95)), "b_lobe_t": (1.0, (0.6, 1.8)), "b_lobe_ti": (1.0, (0.6, 1.8)),
    "b_top": (0, (-40, 80)), "b_foot": (0, (-40, 60)),
    "b_crown_x": (0.4, (0.05, 0.9)), "b_crown_t": (34, (6, 100)),
    "b_spur_len": (20, (0, 100)), "b_spur_rise": (20, (0, 100)), "b_spur_drop": (20, (0, 100)),
    "b_foot_angle": (-52, (-89, -10)),
    "r_width": (230, (150, 330)), "r_y": (0.82, (0.6, 0.98)), "r_dir": (-30, (-80, 20)),
    "f_hook": (230, (120, 330)), "f_top": (720, (640, 790)), "f_bar_l": (90, (30, 160)), "f_bar_r": (150, (60, 240)),
    "f_end_y": (0.92, (0.8, 1.0)), "f_beak_len": (25, (0, 120)), "f_beak_depth": (70, (5, 200)),
    # crossbars of f and t: the left end can come to a point (at bar_tip_y of
    # the bar's height, bar_tip_dx further out) with the top edge sloping in to
    # the stem (bar_taper of the way). 0 / 1 / 0 is a plain rectangle.
    "bar_taper": (0, (0, 1)), "bar_tip_y": (1, (0, 1)), "bar_tip_dx": (0, (-20, 40)),
    "f_top_dw": (0, (-50, 30)),   # the f hook's weight across its top
    "foot_wide_dl": (0, (-50, 30)), "foot_wide_dr": (0, (-20, 60)),   # r and f: lopsided feet
    "foot_wide_ca": (0.6, (0.0, 1.0)), "foot_wide_cb": (0.6, (0.0, 1.0)),  # ... with straighter brackets
    "r_cut_angle": (0, (-45, 45)),     # normal of the cut
    "r_beak_len": (25, (0, 120)), "r_beak_depth": (70, (5, 200)),
    # the r's ear meets the stem through its own join (lower and deeper than the n's)
    "r_join_at": (0.72, (0.4, 0.95)), "r_join_thin": (40, (4, 160)), "r_join_angle": (40, (-20, 85)),
    "r_taper_len": (0.3, (0.02, 0.9)), "r_end_w": (1.0, (0.3, 2.0)),
    # the t's tail thins along the bottom into a fine terminal
    "t_turn_w": (1.0, (0.3, 1.6)), "t_end_w": (0.6, (0.1, 1.6)),
    "t_wedge_len": (40, (0, 140)), "t_wedge_rise": (100, (0, 200)), "t_wedge_drop": (10, (0, 100)),
    "t_angle": (0, (-60, 60)), "t_contrast": (None, (0.0, 0.95)), "t_bar_t": (None, (5, 80)), "t_bar_dy": (0, (-40, 40)),
    "f_end_dir": (-60, (-120, 0)), "f_end_w": (1.0, (0.2, 1.6)), "f_bar_t": (None, (5, 80)), "f_bar_dy": (0, (-40, 40)),
    "u_leg_t": (1.0, (0.6, 2.5)),
    "t_top": (660, (560, 740)), "t_turn": (110, (40, 200)), "t_tail": (170, (80, 360)), "t_tail_y": (60, (20, 150)),
    "t_bar_l": (90, (30, 160)), "t_bar_r": (140, (60, 320)), "t_cut": (30, (-10, 70)),
    "i_dot_y": (690, (620, 760)), "i_dot_r": (55, (30, 90)),
    "bar_t": (32, (5, 80)),
    # the cap rounds' (O, G) own weight and shape: thick x round_gain x cap stem,
    # thin x hairline, fullness offsets from the o's
    "O_thick_k": (1.0, (0.7, 1.4)), "O_thin_k": (1.0, (0.5, 2.5)), "O_full_dh": (0, (-0.2, 0.2)),
    "O_full_dv": (0, (-0.2, 0.2)), "O_in_full_dh": (0, (-0.2, 0.2)), "O_in_full_dv": (0, (-0.2, 0.2)),
    # the G's bowl, relative to the O's
    "G_thick_k": (1.0, (0.8, 1.25)), "G_thin_k": (1.0, (0.6, 1.6)), "G_full_dh": (0, (-0.15, 0.15)),
    "G_full_dv": (0, (-0.15, 0.15)), "G_in_full_dh": (0, (-0.15, 0.15)), "G_in_full_dv": (0, (-0.15, 0.15)),
    "G_stress": (0, (-20, 20)),
    "O_width": (700, (560, 820)), "A_width": (680, (520, 820)), "A_bar_y": (0.32, (0.2, 0.45)),
    "A_apex": (40, (0, 120)), "A_thick": (115, (60, 180)), "A_thin": (28, (8, 80)),
    "E_width": (480, (380, 600)), "E_mid": (0.86, (0.6, 1.0)), "E_arm_t": (32, (8, 80)),
    "E_serif": (90, (20, 180)), "E_serif_w": (26, (8, 60)),
    "G_width": (690, (560, 820)), "G_bar_y": (0.42, (0.3, 0.55)), "G_term_y": (0.72, (0.6, 0.9)),
    "G_serif": (110, (30, 220)),
    "S_width": (500, (380, 620)),
    # ---- the rest of the lowercase (starting values measured off Geist's bounds)
    "desc_stem": (-200, (-240, -160)),    # where descending stems (p q) end
    # d / q: the lobe on the stem's left (like the a's), rising to a crown
    "d_far": (450, (360, 540)), "d_top_at": (440, (300, 520)), "d_top_thin": (26, (8, 80)),
    "d_top_angle": (40, (0, 80)), "d_bot_at": (110, (40, 200)), "d_bot_thin": (50, (8, 90)),
    "d_bot_angle": (50, (0, 80)), "d_lobe_thick": (90, (50, 130)), "d_lobe_thin_b": (34, (10, 80)),
    "d_out_mid": (0.5, (0.3, 0.7)), "d_in_mid": (0.5, (0.3, 0.7)), "d_bx_out": (0.75, (0.3, 1.0)),
    "d_bx_in": (0.85, (0.3, 1.0)), "d_crown_x": (0.4, (0.1, 0.8)), "d_crown_t": (50, (15, 90)),
    "q_dfar": (0, (-60, 60)), "q_top": (0, (-40, 60)),
    "p_dfar": (0, (-60, 60)),
    # c: the e's bowl and lower terminal; the upper terminal a vertical face
    "c_width": (465, (380, 540)), "c_face": (0.9, (0.7, 1.0)), "c_top_y": (420, (330, 500)),
    "c_term_y": (150, (80, 240)), "c_cut_x": (0.5, (0.3, 0.7)), "c_cut_y": (70, (20, 140)),
    "c_cut_angle": (185, (150, 220)), "c_cut_h1": (0.3, (0.0, 0.9)), "c_cut_h2": (0.6, (0.0, 0.9)),
    # j: the i's stem running below the baseline into a hook to the left
    "j_bot": (-60, (-150, 60)), "j_hook": (190, (100, 280)), "j_end_y": (-150, (-200, -60)),
    "j_end_w": (0.9, (0.3, 1.4)), "j_end_dir": (150, (100, 200)),
    # k: thin arm with a top serif, thick leg with a foot
    "k_arm_x": (356, (220, 520)), "k_join_y": (181, (80, 320)), "k_leg_x": (336, (220, 560)),
    "k_leg_y0": (288, (180, 420)), "k_leg_x0": (135, (60, 260)), "k_thick": (87, (50, 130)),
    "k_thin": (41, (10, 80)),
    # w: two v's, sharing the middle apex
    "w_width": (700, (520, 920)), "w_v1": (0.27, (0.15, 0.4)), "w_v2": (0.73, (0.6, 0.85)),
    "w_thick": (88, (50, 130)), "w_thin": (34, (10, 70)), "w_mid_dx": (0, (-40, 40)), "w_mid3_dx": (0, (-60, 60)),
    "w_a2_top": (1.0, (0.6, 1.0)), "w_thin_dx": (0, (-30, 60)),
    # x: thick falling stroke, thin rising one
    "x_width": (310, (220, 600)), "x_thick": (88, (50, 130)), "x_thin": (34, (10, 70)),
    "x_thin_dx": (0, (-60, 60)),
    "diag_foot_len": (50, (10, 120)), "diag_foot_in": (1.0, (0.3, 3.0)),   # feet of diagonals   # the thin stroke's bottom, right of the thick one's top
    # y: the v's thick arm; the thin arm runs on below the baseline into a tail
    "y_width": (420, (300, 600)), "y_vertex": (0.5, (0.3, 0.7)), "y_tail_y": (-120, (-180, -40)),
    "y_end_x": (60, (-40, 200)), "y_end_y": (-190, (-220, -120)), "y_end_w": (2.0, (0.5, 4.0)),
    # z: bars and a thick diagonal; wedges at the top left and bottom right
    "z_width": (430, (340, 520)), "z_bar_t": (34, (15, 60)), "z_thick": (90, (50, 130)),
    "z_serif": (110, (40, 200)), "z_serif_w": (20, (8, 50)), "z_wedge": (80, (0, 160)),
    # ---- the rest of the capitals (starting values measured off Geist's bounds)
    # D-bowls (B D P R): flat hairline arms off the stem, then a curve round
    # to the far side. far: outer extreme from the stem's centre; flat: where
    # the outer arms start curving; th: far wall; full: curve fullness
    "B_mid": (0.5, (0.4, 0.6)), "B_far_t": (419, (320, 520)), "B_far_b": (443, (340, 540)),
    "B_flat_t": (184, (60, 300)), "B_flat_b": (214, (60, 320)), "B_th": (94, (50, 140)),
    "B_hair": (35, (15, 70)), "B_flat_m": (223, (60, 340)), "B_full": (0.6, (0.4, 0.85)), "B_in_full": (0.6, (0.4, 0.85)),   # B_flat_m: the flats either side of the B's waist
    "D_far": (514, (400, 620)), "D_flat": (190, (60, 360)), "D_th": (110, (60, 160)),
    "P_bot": (332, (260, 420)), "P_far": (434, (340, 540)), "P_flat": (184, (60, 300)),
    "R_far": (423, (340, 540)), "R_bot": (332, (260, 420)), "R_leg_x0": (171, (80, 280)), "R_leg_x": (401, (300, 500)),
    "R_leg_w": (90, (50, 130)),
    "C_width": (623, (540, 720)), "C_face": (0.88, (0.7, 1.0)), "C_top_y": (480, (380, 600)),
    "C_term_y": (200, (100, 300)), "C_cut_x": (0.5, (0.3, 0.7)), "C_cut_y": (100, (40, 200)),
    "C_cut_angle": (185, (150, 220)), "C_cut_h1": (0.3, (0.0, 0.9)), "C_cut_h2": (0.6, (0.0, 0.9)),
    "C_thick_k": (1.0, (0.8, 1.25)), "C_right_k": (1.0, (0.2, 1.2)), "C_bottom_k": (1.0, (0.5, 3.0)), "C_stress": (0, (-20, 20)), "C_in_full_dh": (0, (-0.15, 0.15)),
    "C_in_full_dv": (0, (-0.15, 0.15)),
    "F_width": (389, (300, 480)), "F_mid": (0.85, (0.6, 1.0)), "F_mid_y": (0.48, (0.38, 0.6)),
    "L_width": (408, (320, 500)), "T_width": (614, (520, 720)), "T_wedge": (90, (0, 160)),
    "T_serif": (120, (60, 200)),
    "J_bot": (14, (-60, 120)), "J_hook": (233, (140, 320)), "J_end_y": (-190, (-240, -120)),
    "J_end_w": (1.0, (0.4, 1.6)), "J_end_dir": (175, (120, 210)),
    "K_arm_x": (420, (300, 540)), "K_join_y": (250, (120, 380)), "K_leg_x": (440, (340, 560)),
    "K_leg_y0": (380, (250, 500)), "K_leg_x0": (230, (100, 340)), "K_thick": (100, (60, 140)),
    "K_thin": (40, (15, 80)),
    "M_width": (640, (540, 760)), "M_thin": (36, (15, 70)), "M_thick": (100, (60, 140)),
    "M_vertex": (0.5, (0.4, 0.6)), "M_fall_dx": (0, (-60, 120)), "M_rise_dx": (0, (-120, 60)),
    "M_vdx": (0, (-60, 60)),
    "N_width": (560, (460, 660)), "N_thin": (36, (15, 70)), "N_thick": (100, (60, 140)),
    "N_dx0": (0, (-60, 120)), "N_dx1": (0, (-120, 160)),   # the diagonal's ends, off the stems' centres
    "U_width": (505, (420, 600)), "U_thin": (36, (15, 70)), "U_thick": (111, (70, 150)),
    "U_join": (260, (150, 360)), "U_end_w": (0.32, (0.15, 0.8)),
    "V_width": (560, (460, 660)), "V_thick": (105, (60, 150)), "V_thin": (34, (15, 70)),
    "V_vertex": (0.5, (0.4, 0.6)), "V_thin_dx": (0, (-40, 40)),
    "W_width": (950, (820, 1080)), "W_v1": (0.27, (0.15, 0.4)), "W_v2": (0.73, (0.6, 0.85)),
    "W_mid_dx": (0, (-50, 50)), "W_mid3_dx": (0, (-60, 60)), "W_a2_top": (1.0, (0.6, 1.0)),
    "W_thin_dx": (0, (-30, 60)),   # the thin strokes' feet, right of the thick ones' (as on v)
    "X_width": (520, (400, 640)), "X_thick": (100, (60, 140)), "X_thin": (36, (15, 70)),
    "X_thin_dx": (0, (-60, 60)),
    "Y_width": (560, (460, 680)), "Y_join": (0.45, (0.3, 0.6)), "Y_thick": (100, (60, 140)),
    "Y_thin": (34, (15, 70)),
    "Z_width": (500, (400, 600)), "Z_thick": (105, (60, 150)), "Z_serif": (140, (60, 220)),
    "Z_wedge": (100, (0, 180)),
    # Q's tail: a wedge from the bowl's bottom to a point below right
    "Q_t0": (233, (150, 320)), "Q_t1": (380, (300, 460)), "Q_tip_x": (628, (520, 740)),
    "Q_tip_y": (-184, (-240, -120)), "Q_end_x": (722, (620, 820)), "Q_end_y": (-107, (-180, -40)),
    # ---- digits (lining; starting values measured off Geist) ---------------
    "fig_h": (722, (680, 760)),           # height of the round figures' tops
    "fig_stem": (92, (60, 120)), "fig_thin_k": (1.0, (0.5, 2.0)),
    "zero_w": (549, (440, 620)), "zero_thick_k": (1.0, (0.7, 1.3)),
    "one_top": (725, (690, 760)), "one_fy": (626, (560, 690)), "one_flag": (182, (100, 260)),
    "one_flag_t": (5, (0, 40)),
    "two_w": (440, (360, 520)), "two_end_y": (474, (380, 560)), "two_right_y": (535, (420, 640)),
    "two_mid_x": (0.5, (0.3, 0.7)), "two_mid_y": (290, (180, 380)), "two_mid_dir": (225, (190, 260)),
    "two_bar_l": (0, (-40, 60)), "two_bar_r": (470, (380, 540)), "two_bar_t": (84, (30, 110)),
    "two_serif": (202, (120, 280)), "two_wedge": (80, (0, 160)), "two_end_w": (1.0, (0.5, 1.6)),
    "three_w": (440, (360, 520)), "three_end_y": (516, (420, 600)), "three_mid_y": (382, (320, 440)),
    "three_mid_x": (0.33, (0.15, 0.5)), "three_ur_y": (550, (480, 640)), "three_lr_y": (195, (120, 280)),
    "three_ur_dx": (-20, (-80, 40)), "three_bot_x": (0.5, (0.3, 0.7)), "three_end_x": (0.12, (0.0, 0.3)),
    "three_end2_y": (110, (40, 200)), "three_end_dir": (110, (60, 160)),
    "four_x": (410, (340, 480)), "four_bar_y": (200, (140, 260)), "four_bar_t": (40, (20, 70)),
    "four_bar_r": (580, (500, 640)), "four_thin": (36, (15, 70)), "four_top": (725, (690, 760)),
    "five_w": (480, (400, 560)), "five_top_l": (130, (80, 200)), "five_top_r": (501, (420, 560)),
    "five_bar_t": (85, (40, 120)), "five_stem_bot": (395, (330, 460)), "five_stem_w": (44, (20, 80)),
    "five_bowl_top": (430, (380, 500)), "five_end_x": (60, (20, 140)), "five_end_y": (80, (30, 160)),
    "five_end_dir": (135, (100, 170)),
    "six_w": (507, (440, 580)), "six_bowl_h": (458, (380, 520)), "six_hook_x": (0.85, (0.6, 1.0)),
    "six_face_y": (537, (460, 640)), "six_hook_w": (0.4, (0.2, 1.0)), "six_wedge": (60, (0, 140)),
    "six_bowl_thick_k": (1.0, (0.7, 1.3)),
    "nine_w": (507, (440, 580)), "nine_bowl_h": (458, (380, 520)), "nine_hook_x": (0.85, (0.6, 1.0)),
    "nine_face_y": (537, (460, 640)), "nine_hook_w": (0.4, (0.2, 1.0)), "nine_wedge": (60, (0, 140)),
    "nine_bowl_thick_k": (1.0, (0.7, 1.3)),
    "seven_w": (460, (380, 540)), "seven_bar_t": (84, (40, 120)), "seven_foot_l": (153, (60, 260)),
    "seven_foot_r": (277, (180, 360)), "seven_top_w": (57, (30, 100)), "seven_serif": (203, (100, 280)),
    "seven_wedge": (80, (0, 160)),
    "eight_w": (514, (440, 580)), "eight_top_w": (430, (360, 500)), "eight_waist": (390, (320, 440)),
    "eight_thick_k": (0.9, (0.6, 1.2)),
    # the 8's spine: a stroke across the waist, from the upper bowl's left
    # down to the lower bowl's right (shares of the width; heights)
    "eight_sp_x0": (0.2, (0.0, 0.5)), "eight_sp_y0": (540, (440, 640)), "eight_sp_x1": (0.8, (0.5, 1.0)),
    "eight_sp_y1": (230, (130, 330)), "eight_sp_w": (0.8, (0.3, 1.3)),
    # ---- accents (drawn at the lowercase position) ---------------------------
    "acc_y0": (583, (540, 620)), "acc_y1": (745, (690, 800)),        # bottom / top of acute, grave, circumflex
    "acute_bot_w": (30, (10, 60)), "acute_top_w": (97, (50, 140)), "acute_run": (76, (20, 140)),
    "circ_w": (306, (220, 380)), "circ_top_w": (100, (50, 150)), "circ_leg_w": (25, (10, 50)),
    "circ_in": (40, (10, 90)),
    "tilde_len": (302, (220, 380)), "tilde_lo": (627, (580, 680)), "tilde_hi": (691, (640, 740)),
    "tilde_w": (46, (20, 80)), "tilde_dir": (55, (20, 80)),
    "diaer_r": (56, (35, 75)), "diaer_gap": (180, (130, 240)), "diaer_y": (662, (620, 700)),
    "ring_w": (166, (120, 220)), "ring_h": (160, (120, 220)), "ring_y0": (577, (540, 620)),
    "ring_thick": (38, (20, 60)), "ring_thin": (22, (10, 50)),
    "ced_w": (60, (30, 90)), "ced_thin": (20, (8, 40)), "ced_len": (125, (80, 180)), "ced_drop": (200, (150, 240)),
    "ced_neck": (-60, (-100, -20)),
    # placement: acute and grave sit by their foot, not their centre, so their
    # centres stand off the base's (acute right, grave left); some bases
    # centre their marks off their ink's centre (E and U on their arms, ...)
    "acute_dx": (65, (0, 120)), "grave_dx": (-80, (-140, 0)),
    "acc_dx_E": (40, (-40, 100)), "acc_dx_U": (45, (-40, 100)), "acc_dx_Y": (20, (-40, 100)),
    "acc_dx_y": (0, (-60, 100)), "acc_dx_a": (0, (-60, 60)), "acc_dx_A": (-6, (-60, 60)),
    "acc_dx_c": (0, (-60, 60)), "acc_dx_C": (0, (-60, 60)),
    # ---- symbols (starting values off Geist; ^ and £ have no reference) -----
    "math_t": (40, (20, 70)), "math_y": (279, (220, 340)), "math_w": (408, (320, 480)),
    "eq_gap": (145, (90, 200)), "eq_dy": (0, (-40, 40)), "lt_dy": (0, (-40, 40)), "eq_t": (40, (20, 70)),
    "lt_w": (446, (360, 520)), "lt_h": (508, (420, 580)), "lt_t": (42, (20, 90)), "lt_end_k": (1.0, (0.3, 1.2)),
    "times_w": (396, (320, 460)),
    "mid_dot_y": (297, (240, 360)), "bullet_r": (112, (70, 150)), "bullet_y": (311, (250, 380)),
    "ellipsis_gap": (190, (140, 240)),
    "under_w": (476, (380, 560)), "under_y": (-47, (-100, -10)), "under_t": (52, (25, 80)),
    "bar_w": (56, (30, 90)), "bar_top": (745, (680, 800)), "bar_bot": (-110, (-200, -40)),
    "brk_w": (263, (180, 340)), "brk_stem": (62, (35, 100)), "brk_arm": (40, (20, 70)),
    "brk_top": (745, (680, 800)), "brk_bot": (-134, (-200, -60)),
    "brace_w": (318, (240, 400)), "brace_pen": (62, (35, 100)), "brace_x": (0.5, (0.3, 0.7)),
    "hash_w": (491, (400, 580)), "hash_slant": (60, (0, 140)), "hash_t": (50, (25, 80)),
    "hash_gap": (230, (160, 300)), "hash_bar_t": (44, (20, 70)), "hash_y0": (236, (160, 320)),
    "hash_y1": (474, (400, 560)),
    "pct_w": (688, (600, 780)), "pct_ring_w": (240, (180, 300)), "pct_ring_h": (330, (260, 400)),
    "pct_thick": (60, (30, 90)), "pct_slash_w": (44, (20, 80)), "pct_s0": (130, (0, 260)), "pct_s1": (110, (0, 260)),
    "deg_w": (280, (200, 340)), "deg_t": (50, (25, 80)),
    "dollar_bar": (30, (15, 50)), "dollar_h": (710, (640, 820)), "dollar_y0": (0, (-60, 40)),
    "dollar_bar_dx": (0, (-40, 40)), "dollar_top": (800, (740, 860)), "dollar_bot": (-90, (-150, -40)),
    "ast_r": (177, (130, 220)), "ast_y": (551, (480, 620)), "ast_tip_w": (70, (30, 110)),
    "ast_root_w": (14, (0, 40)), "ast_rot": (0, (-30, 30)),
    "at_w": (909, (800, 1000)), "at_h": (915, (820, 1000)), "at_bot": (-190, (-240, -120)),
    "at_ring_t": (50, (25, 80)), "at_in_w": (330, (260, 420)), "at_in_h": (430, (360, 500)),
    "at_in_x": (0.33, (0.25, 0.45)), "at_in_y": (90, (40, 160)), "at_slant": (14, (0, 25)),
    "amp_w": (620, (540, 700)), "amp_arm_x": (0.9, (0.75, 1.0)), "amp_arm_y": (330, (250, 420)),
    "amp_loop_x": (0.45, (0.3, 0.6)), "amp_loop_w": (220, (160, 300)), "amp_waist_y": (430, (360, 500)),
    "amp_leg_x": (1.0, (0.85, 1.1)), "amp_pen": (90, (60, 120)),
    "euro_x0": (110, (60, 180)), "euro_w": (555, (480, 640)), "euro_bar_l": (7, (-20, 60)),
    "euro_bar_r": (470, (380, 560)), "euro_y0": (290, (230, 350)), "euro_y1": (430, (370, 500)),
    "pound_w": (520, (440, 600)), "pound_bar_y": (330, (260, 400)),
    "caret_w": (420, (320, 500)), "caret_h": (330, (260, 400)), "caret_y0": (380, (320, 440)),
    "caret_t": (50, (25, 80)),
    "atilde_dy": (-391, (-460, -320)),
    # punctuation. dot: radius (the period's, colon's, ! and ? dots)
    "dot_r": (68, (40, 100)),
    # comma: tail depth below the baseline, tip offset from the ball's centre,
    # tip width, and how far below the ball's equator its outer curve bends
    "comma_tail": (157, (80, 240)), "comma_dx": (-15, (-60, 40)), "comma_tip_w": (50, (10, 100)),
    "comma_bend": (52, (0, 120)),
    "semi_dx": (0, (-40, 40)),   # the semicolon's comma, right of its dot
    # curly quotes: the comma shape scaled by quote_k; ’ tops out at quote_top,
    # ‘ (turned) bottoms out at quote_bot; pairs are quote_gap apart (centres)
    "quote_k": (0.9, (0.6, 1.2)), "quote_top": (724, (650, 800)), "quote_bot": (530, (450, 620)),
    "quote_gap": (186, (120, 260)),
    # straight quotes: a wedge from quote_w at the cap height to quote_tip_w,
    # quote_h long; " pairs them quote_dbl_gap apart
    "quote_w": (94, (40, 140)), "quote_tip_w": (4, (0, 40)), "quote_h": (210, (120, 300)),
    "quote_dbl_gap": (140, (80, 220)),
    # dashes: thickness, centre height; lengths of the hyphen, en and em dash
    "dash_t": (52, (20, 90)), "dash_y": (322, (250, 400)), "hyphen_len": (319, (200, 420)),
    "endash_len": (476, (380, 600)), "emdash_len": (798, (700, 960)),
    # !: a wedge excl_w wide down to excl_shoulder, tapering to excl_tip_w at excl_bot
    "excl_w": (97, (50, 140)), "excl_shoulder": (530, (400, 700)), "excl_tip_w": (35, (10, 80)),
    "excl_bot": (170, (120, 260)),
    # ?: a stroke (thick upright, thin across), from its left end round the top
    # and down into the neck; quest_w: the pen's width, quest_thin: across the top
    "quest_width": (465, (360, 560)), "quest_w": (100, (60, 140)), "quest_thin": (43, (15, 80)),
    "quest_end_y": (487, (400, 600)), "quest_neck_x": (0.62, (0.4, 0.8)), "quest_neck_y": (330, (250, 420)),
    "quest_bot": (170, (120, 260)), "quest_bot_w": (0.35, (0.15, 1.0)), "quest_tension": (1.0, (0.7, 1.4)),
    "quest_top_x": (0.5, (0.3, 0.7)), "quest_right_y": (606, (450, 700)), "quest_neck_dir": (240, (200, 270)),
    "quest_bot_dx": (-25, (-80, 30)),
    # parentheses: crescent strokes; paren_w at the middle, thin to paren_end_w
    # (share) at the ends; paren_top / paren_bot the ends' heights, paren_bow how
    # far the middle bows out from the ends
    "paren_w": (80, (40, 120)), "paren_end_w": (0.15, (0.05, 0.6)), "paren_top": (740, (680, 800)),
    "paren_bot": (-126, (-200, -60)), "paren_bow": (200, (120, 300)),
    "paren_end_dir": (33, (10, 80)),   # the ends' direction, degrees below the horizontal
    "paren_tension": (1.0, (0.7, 1.4)),
    # slash: horizontal width; from the descent to above the cap height
    "slash_w": (56, (30, 100)), "slash_top": (745, (680, 800)), "slash_bot": (-110, (-200, -40)),
    "slash_run": (356, (200, 480)),
    # the S's spine weight relative to the s's (a cap S often carries more)
    "S_swell_k": (1.0, (0.6, 1.8)),
    # E arm serifs with E_serifs="wedge": how far the wedge slopes back along the arm
    "E_wedge": (60, (0, 160)),
    # the middle arm's height (share of the cap height) and the bottom arm's
    # length (relative to the top arm's)
    "E_mid_y": (0.5, (0.4, 0.6)), "E_bot_k": (1.04, (0.9, 1.15)),
    # the middle and bottom serifs' heights relative to E_serif; the stem's
    # serif brackets relative to the cap serifs'
    "E_mid_sk": (0.45, (0.2, 0.8)), "E_bot_sk": (1.1, (0.6, 1.6)), "E_bracket_k": (1.0, (0.3, 2.5)),
    "ball_r": (42, (20, 80)), "ball_drop": (0, (-40, 60)),
    "foot_tip": (26, (4, 60)),
}


def params(**values) -> SimpleNamespace:
    """A namespace of parameters: kit defaults overridden by ``values``.
    ``values`` may be numbers (change the default) or ``(default, (lo, hi))``."""
    unknown = set(values) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"unknown kit params: {sorted(unknown)}")
    ns = {}
    inherit = {"t_contrast": 0.6, "t_bar_t": 32, "f_bar_t": 32}   # defaults for None entries
    for name, (dflt, rng) in DEFAULTS.items():
        if dflt is None:
            dflt = inherit[name]
        v = values.get(name, dflt)
        if isinstance(v, tuple):
            dflt, rng = v
        else:
            dflt = v
        lo, hi = rng
        ns[name] = Param(dflt, (min(lo, dflt), max(hi, dflt)), name=name)
    return SimpleNamespace(**ns)


# --------------------------------------------------------------------------
# terminal sets

def _vtops(P, tip) -> dict:
    """Top serifs of diagonals: outer side full length, inner side scaled by
    ``top_serif_in``, both with ``top_serif_slope``."""
    def side(scale):
        return dict(length=P.top_serif_len * scale, bracket=P.top_serif_bracket, tip=tip,
                    slope=P.top_serif_slope, curve=(P.serif_ca, P.serif_cb))
    def foot(scale):
        return dict(length=P.diag_foot_len * scale, bracket=P.top_serif_bracket, tip=tip,
                    slope=P.top_serif_slope, curve=(P.serif_ca, P.serif_cb))
    return {"vtop_l": bracketed(left=side(1.0), right=side(P.top_serif_in)),
            "vtop_r": bracketed(left=side(P.top_serif_in), right=side(1.0)),
            # feet of diagonals (x, X, K): the outer side diag_foot_len, the inner scaled
            "vfoot_l": bracketed(left=foot(1.0), right=foot(P.diag_foot_in)),
            "vfoot_r": bracketed(left=foot(P.diag_foot_in), right=foot(1.0))}


def bracketed_set(P) -> TerminalSet:
    """Sharp bracketed serifs, wedge head, pointed spur, beaks."""
    foot = bracketed(length=P.serif_len, bracket=P.serif_bracket, tip=P.serif_tip,
                     curve=(P.serif_ca, P.serif_cb), cup=P.serif_cup, slope=P.serif_slope)
    cap = bracketed(length=P.cap_serif_len, bracket=P.cap_serif_bracket, tip=P.cap_serif_tip,
                    curve=(P.serif_ca, P.serif_cb), cup=P.serif_cup, slope=P.cap_serif_slope)
    head_shape = dict(tip=P.head_tip, flat=P.head_flat, bracket=P.head_bracket, corner=P.head_corner,
                      ca=P.head_ca, cb=P.head_cb)
    wide = bracketed(left=dict(length=lazy_max(P.serif_len + P.foot_wide_dl, 0)),
                     right=dict(length=lazy_max(P.serif_len + P.foot_wide_dr, 0)),
                     bracket=P.serif_bracket, tip=P.serif_tip, curve=(P.foot_wide_ca, P.foot_wide_cb),
                     cup=P.serif_cup, slope=P.serif_slope)
    return TerminalSet(
        none,
        foot=foot, foot_wide=wide, cap_foot=cap, cap_head=cap,
        head=wedge(side="left", length=P.head_len, rise=P.head_rise, drop=P.head_drop, hollow=P.head_hollow,
                   **head_shape),
        u_head=wedge(side="left", length=lazy_max(P.head_len + P.u_head_dlen, 0), rise=P.head_rise, drop=P.head_drop,
                     hollow=P.head_hollow, **head_shape),
        spur=wedge(side="right", length=P.spur_len, rise=P.spur_rise, drop=P.spur_drop, tip=P.spur_tip,
                   flat=P.spur_flat, bracket=P.spur_bracket, corner=P.spur_corner, ca=P.spur_ca, cb=P.spur_cb,
                   sweep=P.spur_sweep),
        G_spur=wedge(side="right", length=P.G_spur_len, rise=P.G_spur_rise, drop=P.G_spur_drop, tip=P.spur_tip,
                   flat=P.spur_flat, bracket=P.spur_bracket, corner=P.spur_corner, ca=P.spur_ca, cb=P.spur_cb,
                   sweep=P.spur_sweep),
        top=bracketed(length=P.top_serif_len, bracket=P.top_serif_bracket, tip=P.top_serif_tip,
                      curve=(P.serif_ca, P.serif_cb)),
        **_vtops(P, P.top_serif_tip),
        # s has its own beaks (sharing a's made fitting a silently change s); both
        # rise above the stroke -- the top one past its upper edge, the bottom one
        # up the vertical face of the lower terminal
        # Geist's s ends are plain angled cuts (a sharp corner, no beak: the
        # fit had shrunk the beaks to nothing, leaving a 2 u flap)
        s_top=flat(P.s_cut1), s_bottom=flat(P.s_cut2),
        G_head=bracketed(length=P.G_serif, bracket=P.G_head_bracket, tip=P.cap_serif_tip,
                         curve=(P.serif_ca, P.serif_cb), slope=P.G_head_slope),
        G_foot=flat(P.G_cut),
        hood=flat("vertical"),   # Geist's a: the hood curves down to a plain horizontal cut
        ear=flat("horizontal"), vertex=flat("vertical"), apex=flat("vertical"),
        cap_foot_l=bracketed(length=P.cap_serif_len, bracket=P.cap_serif_bracket * P.E_bracket_k, tip=P.cap_serif_tip,
                             curve=(P.serif_ca, P.serif_cb), right=None),
        cap_vfoot=bracketed(length=P.cap_serif_len + P.A_foot_dl, bracket=P.cap_serif_bracket * P.A_bracket_k, tip=P.cap_serif_tip,
                            curve=(P.serif_ca, P.serif_cb)),
        # the A's feet: the inner serifs (between the feet) may run longer
        A_lfoot=bracketed(length=P.cap_serif_len + P.A_foot_dl, bracket=P.cap_serif_bracket * P.A_bracket_k, tip=P.cap_serif_tip,
                          curve=(P.serif_ca, P.serif_cb),
                          right=dict(length=lazy_max(P.cap_serif_len + P.A_foot_dl + P.A_foot_in_dl, 0))),
        A_rfoot=bracketed(length=P.cap_serif_len + P.A_foot_dl, bracket=P.cap_serif_bracket * P.A_bracket_k, tip=P.cap_serif_tip,
                          curve=(P.serif_ca, P.serif_cb),
                          left=dict(length=lazy_max(P.cap_serif_len + P.A_foot_dl + P.A_foot_in_dl, 0))),
        # the b's stem has no foot: it is chiselled off on a diagonal down to a point
        b_foot=flat(P.b_foot_angle),
        # r and f have their own beaks (shared ones would couple them to a's hood)
        # Geist's r ear is not beaked: a thick stroke sheared off almost vertically
        r_ear=flat(P.r_cut_angle),
        # the f's hook ends in a plain vertical face; the t's top is a flat top
        # with a long flag down the left of the stem (a wedge)
        f_hook=flat("horizontal"),
        t_tail=flat(P.t_tail_cut), t_top=wedge(side="left", length=P.t_wedge_len, rise=P.t_wedge_rise, drop=P.t_wedge_drop,
                                          flat=P.t_wedge_flat, sweep=P.t_wedge_sweep),
    )


def didone_set(P) -> TerminalSet:
    """Flat hairline serifs with a minimal bracket, ball terminals, vertical
    hairline beaks."""
    from ..terminals import ball
    foot = bracketed(length=P.serif_len, bracket=P.serif_bracket, tip=P.foot_tip,
                     curve=(P.serif_ca, P.serif_cb), cup=P.serif_cup)
    cap = bracketed(length=P.cap_serif_len, bracket=P.cap_serif_bracket, tip=P.foot_tip,
                    curve=(P.serif_ca, P.serif_cb), cup=P.serif_cup)
    head = bracketed(left=dict(length=P.head_len, bracket=P.serif_bracket, tip=P.foot_tip), right=None,
                     curve=(P.serif_ca, P.serif_cb))
    vbeak = lambda side: beak(side=side, length=P.beak_len, depth=P.beak_depth, axis="horizontal")
    balls = ball(radius=P.ball_r, side="left", drop=P.ball_drop)
    return TerminalSet(
        none,
        foot=foot, foot_wide=foot, cap_foot=cap, cap_head=cap,
        cap_foot_l=bracketed(length=P.cap_serif_len, bracket=P.cap_serif_bracket * P.E_bracket_k, tip=P.foot_tip,
                             curve=(P.serif_ca, P.serif_cb), right=None),
        head=head,
        u_head=bracketed(left=dict(length=lazy_max(P.head_len + P.u_head_dlen, 0), bracket=P.serif_bracket, tip=P.foot_tip),
                         right=None, curve=(P.serif_ca, P.serif_cb)),
        spur=bracketed(right=dict(length=P.spur_len * 0.6, bracket=P.serif_bracket * 0.6, tip=P.foot_tip),
                                  left=None, curve=(P.serif_ca, P.serif_cb)),
        G_spur=bracketed(right=dict(length=P.spur_len * 0.6, bracket=P.serif_bracket * 0.6, tip=P.foot_tip),
                         left=None, curve=(P.serif_ca, P.serif_cb)),
        top=bracketed(length=P.top_serif_len, bracket=P.top_serif_bracket, tip=P.foot_tip,
                      curve=(P.serif_ca, P.serif_cb)),
        **_vtops(P, P.foot_tip),
        cap_vfoot=bracketed(length=P.cap_serif_len + P.A_foot_dl, bracket=P.cap_serif_bracket * P.A_bracket_k, tip=P.foot_tip,
                            curve=(P.serif_ca, P.serif_cb)),
        # the A's feet: the inner serifs (between the feet) may run longer
        A_lfoot=bracketed(length=P.cap_serif_len + P.A_foot_dl, bracket=P.cap_serif_bracket * P.A_bracket_k, tip=P.foot_tip,
                          curve=(P.serif_ca, P.serif_cb),
                          right=dict(length=lazy_max(P.cap_serif_len + P.A_foot_dl + P.A_foot_in_dl, 0))),
        A_rfoot=bracketed(length=P.cap_serif_len + P.A_foot_dl, bracket=P.cap_serif_bracket * P.A_bracket_k, tip=P.foot_tip,
                          curve=(P.serif_ca, P.serif_cb),
                          left=dict(length=lazy_max(P.cap_serif_len + P.A_foot_dl + P.A_foot_in_dl, 0))),
        # s/S: vertical faces with hairline beards (top one hangs down, bottom one stands up)
        s_top=beak(side="below", length=P.s_beak_len, depth=P.s_beak_depth, axis=P.s_cut1),
        s_bottom=beak(side="above", length=P.s_beak_len, depth=P.s_beak_depth, axis=P.s_cut2),
        hood=ball(radius=P.ball_r, side="left", drop=P.ball_drop),
        r_ear=ball(radius=P.ball_r * 0.9, side="below", drop=P.ball_drop),
        f_hook=ball(radius=P.ball_r * 0.9, side="below", drop=P.ball_drop),
        ear=ball(radius=P.ball_r * 0.7, side="above"), vertex=flat("vertical"), apex=flat("vertical"),
        t_tail=flat(0), t_top=flat(P.t_cut + 90),
        b_foot=flat("vertical"),   # the lobe-style b: its stem simply ends at the baseline
    )


# --------------------------------------------------------------------------

def define(font, P, glyphs: str | None = None, terminals=None, g_ear: str = "stroke", g_band: str = "none",
           b_style: str = "bowl", E_serifs: str = "rect",
           G_jaw: str = "bar"):
    """Register the kit's glyphs on ``font``. ``g_ear``: ``"stroke"`` (a
    short stroke taking the ``ear`` terminal, e.g. a ball) or ``"flag"`` (a
    four-sided flag). ``g_band``: ``"flat"`` puts a flat bar across the top of
    the g's lower loop (and trims the loop above it) -- a tilted loop with a
    flat top, which one contour bowl can't make. ``b_style``: ``"bowl"`` (a
    free bowl overlapping the stem) or ``"lobe"`` (a lobe hanging on the
    stem at two joins, like the a's bowl mirrored). ``E_serifs``: ``"rect"``
    (upright bars at the arm ends) or ``"wedge"`` (vertical faces whose inner
    edges slope back E_wedge along the arm). ``G_jaw``: ``"bar"`` (a stem
    under a flat bar) or ``"serif"`` (a stem with a bracketed head serif for
    the bar, standing to the baseline and cut off there on a diagonal; the
    bowl is trimmed at the stem's outer edge; the upper terminal is the bowl's
    wall cut off by a vertical face at G_face of the width, with no beard)."""
    from ..handles import handle
    from ..junctions import contact, join, join_handles, notch_cut
    from ..parts import part

    font.pen = Pen(angle=0, width=P.stem, contrast=1 - P.hairline / P.stem)
    cap_pen = Pen(angle=0, width=P.cap_stem, contrast=1 - P.hairline / P.cap_stem)
    arch_pen = Pen(angle=P.arch_angle, width=P.stem, contrast=P.arch_contrast)
    font.xheight, font.capheight, font.ascender, font.descender = P.xheight, P.capheight, P.ascender, -200
    font.sidebearings = {"straight": P.sb_straight, "round": P.sb_round}
    font.terminals = terminals or bracketed_set(P)
    xh, os_ = P.xheight, P.overshoot
    want = set(glyphs) if glyphs else None

    def g(name, sb=None, **kw):
        def deco(fn):
            from ..parts import _unicode_for
            u = _unicode_for(name)
            ch = chr(u) if u is not None else None
            lig = "_" in name and want is not None and all(c in want for c in name.split("_"))
            if want is None or (ch and ch in want) or name in (want or ()) or lig:
                font.glyph(name, sb=sb, **kw)(fn)
            return fn
        return deco

    # ---- parts ----------------------------------------------------------
    @part
    def stem_(p, x, top, bottom=0, pen=None, foot="foot", head="foot"):
        p.line((x, bottom), (x, top), name="stem", names=("bottom", "top"),
               terminal=(foot, head), pen=pen)

    @part
    def bowl(p, x, width, bottom, top, thick, thin, stress=0, full=0.6, in_full=0.6, thin_bottom=None, rotate=0,
             thick_right=None):
        p.bowl(x, x + width, bottom, top, thick=thick, thin=thin, stress=stress,
               fullness=full, inner_fullness=in_full, thin_bottom=thin_bottom, rotate=rotate,
               thick_right=thick_right)

    def head_inset_cut(p, edge_x, crotch_y, inset, name="head_inset"):
        """Cut a strip `inset` wide off the right edge of a headed stem, from
        `head_inset_y` above the join's crotch to well above the head."""
        y0 = crotch_y + P.head_inset_y
        (p.cut(name).move((edge_x - inset, y0), "a").line((edge_x + 1, y0), "b")
            .line((edge_x + 1, xh + 400), "c").line((edge_x - inset, xh + 400), "d").close())

    def n_join(top):
        # val(top): the join height must be (number x param) to get a handle
        return join(val(top) * P.n_join_at, P.n_join_thin, P.n_join_angle, P.n_notch, P.n_notch_w)

    def taper_handle(stroke, t_expr):
        """Handle at the point ``t`` along a stroke (by control-polygon length),
        moving along the stroke: the end of a taper profile."""
        import numpy as np
        from ..geometry import bez
        cub = stroke.cubics()
        lens = [float(np.linalg.norm(np.diff(C, axis=0), axis=1).sum()) for C, _ in cub]
        total = sum(lens) or 1.0
        t, acc = val(t_expr) * total, 0.0
        for (C, _), L in zip(cub, lens):
            if acc + L >= t or (C is cub[-1][0]):
                u = min(max((t - acc) / max(L, 1e-9), 0.0), 1.0)
                pt_ = bez(C, np.array([u]))[0]
                from ..geometry import bez_d1
                d = bez_d1(C, np.array([u]))[0]
                # t is measured along the control polygon: dt moves u by
                # dt*total/L, i.e. the point by |B'(u)|*total/L font units
                rate = float(np.linalg.norm(d)) * total / max(L, 1e-9)
                handle(t_expr, pt_, d, rate=rate, label="taper length")
                return
            acc += L

    @part
    def arch(p, x, width, top=xh, flip=False, edge_half=None, inset=False):
        """Leaves the stem at a :class:`Join` (hairline thinning at the crotch),
        over the shoulder, down into the leg. Ends inside the leg so the leg
        can be an exact stem. ``flip`` rotates it 180 degrees (the cup of u)."""
        if flip:
            return _cup(p, x, width, top)
        bias = P.n_bias
        hair = arch_pen.hairline
        J = n_join(top)
        # the join sits on the stem's actual edge (the h's stem is a little
        # heavier: on n's edge its notch cut a pinhole inside the stem)
        edge = x + (P.stem / 2 if edge_half is None else edge_half)
        _, _, st, d0, thin = contact(J, val(edge), +1, rising=True)
        m0 = thin / max(arch_pen.width_at(d0), 1.0)
        left_half = hair / 2 + (val(P.stem) - hair) * val(bias)
        leg_x = x + width
        sh_y = top + os_ - arch_pen.width_at(0) * (1 - val(bias))
        (p.path(name="arch", pen=arch_pen, bias=bias,
                profile=Profile((0, m0), (P.n_taper_len, 1), (1, 1)))
            .start(st, name="join", dir=d0, terminal=False, pen_angle=P.n_join_pa)
            .curve_to((x + width * P.n_shoulder_x, sh_y), name="shoulder", dir=0, tension=P.n_tension,
                      width=P.n_shoulder_dw, pen_angle=P.n_shoulder_pa)
            .curve_to((leg_x + P.stem / 2 - left_half, top * P.n_out_y * 0.5), name="leg", dir=-90,
                      tension=P.n_leg_t, pen_angle=P.n_leg_pa))
        notch_cut(p, J, val(edge), +1)
        if inset:
            head_inset_cut(p, edge, val(top) * P.n_join_at, P.head_inset)
        join_handles(J, val(edge), +1, rising=True)
        taper_handle(p._strokes[-1], P.n_taper_len)

    def _cup(p, x, width, top):
        """The u's cup: leaves the right stem low through a Join (the arch's
        geometry rotated 180 degrees, with its own u_* values -- the u isn't
        simply an n turned round), under the bowl, up into the left stem."""
        bias = P.u_bias
        cup_pen = Pen(angle=P.u_angle, width=P.stem, contrast=arch_pen.contrast)
        hair = cup_pen.hairline
        R = x + width
        J = join(val(top) - val(top) * P.u_join_at, P.u_join_thin, P.u_join_angle, P.n_notch, P.n_notch_w)
        edge = R - P.stem / 2
        _, _, st, d0, thin = contact(J, val(edge), -1, rising=False)
        m0 = thin / max(cup_pen.width_at(d0), 1.0)
        left_half = hair / 2 + (val(P.stem) - hair) * val(bias)
        sh_y = -os_ + cup_pen.width_at(180) * (1 - val(bias))
        (p.path(name="cup", pen=cup_pen, bias=bias,
                profile=Profile((0, m0), (P.u_taper_len, 1), (1, 1)))
            .start(st, name="join", dir=d0, terminal=False, pen_angle=P.u_join_pa)
            .curve_to((R - width * P.u_shoulder_x, sh_y), name="shoulder", dir=180, tension=P.u_tension,
                      pen_angle=P.u_shoulder_pa)
            .curve_to((x - P.stem / 2 + left_half, top - top * P.u_out_y * 0.5), name="leg", dir=90,
                      tension=P.u_leg_t, pen_angle=P.u_leg_pa))       # >1 stops the curve swinging out past the stem
        notch_cut(p, J, val(edge), -1)

    @part
    def spine(p, width, height, stem_w, swell_k=1.0, t1=(0, 0), t2=(0, 0), mid=(0, 0), sides=(0, 0), tops=(0, 0),
              dw=(0, 0, 0, 0, 0, 0, 0), y0=0):
        """The s/S spine: one stroke, width peaking mid-spine. ``t1``/``t2``,
        ``mid`` (x, y), ``sides`` (left y, right y) and ``tops`` (top x,
        bottom x) offset the s's nodes: the S's own proportions. ``dw`` adds to
        the s's width offsets at t1, top, left, mid, right, bottom, t2."""
        W = width
        sp = Pen(angle=P.s_angle, width=stem_w, contrast=1 - P.hairline / stem_w)
        top = y0 + height + os_ - sp.width_at(180) / 2
        bot = y0 - os_ + sp.width_at(180) / 2
        e_, m_ = P.s_ends * swell_k, P.s_swell * swell_k
        sw = Profile((0, e_), (0.5 - P.s_spread, e_), (0.5, m_), (0.5 + P.s_spread, e_), (1, e_))
        (p.path(name="spine", pen=sp, profile=sw, bias=P.s_bias, smooth=val(P.s_mid_free) >= 0.5)
            .start((W * (P.s_t1_x + t1[0]), y0 + height * (P.s_t1_y + t1[1])), name="t1", pen_angle=P.s_t1_pa, dir=P.s_t1_dir, terminal="s_top", width=P.s_t1_dw + dw[0])
            .curve_to((W * (P.s_top_x + tops[0]), top), name="top", pen_angle=P.s_top_pa, dir=P.s_top_dir, tension=P.s_tension, width=P.s_top_dw + dw[1])
            .curve_to((sp.width_at(90) / 2, y0 + height * (P.s_left_y + sides[0])), name="left", pen_angle=P.s_left_pa, dir=P.s_left_dir,
                      tension=P.s_left_t, width=P.s_left_dw + dw[2])
            # s_mid_free: the spine's middle direction is solved (Hobby's mock-
            # curvature continuity) instead of imposed -- an imposed direction
            # makes both halves curl to meet it, and high contrast turns that
            # dip in direction into a kink in the width
            .curve_to((W * (P.s_mid_x + mid[0]), y0 + height * (P.s_mid_y + mid[1])), name="mid", pen_angle=P.s_mid_pa,
                      dir=None if val(P.s_mid_free) >= 0.5 else P.s_mid_dir, tension=P.s_mid_t,
                      width=P.s_mid_dw + dw[3])
            .curve_to((W - sp.width_at(90) / 2, y0 + height * (P.s_right_y + sides[1])), name="right", pen_angle=P.s_right_pa, dir=P.s_right_dir,
                      tension=P.s_right_t, width=P.s_right_dw + dw[4])
            .curve_to((W * (P.s_bot_x + tops[1]), bot), name="bottom", pen_angle=P.s_bot_pa, dir=P.s_bot_dir, tension=P.s_tension + P.s_bot_dt,
                      width=P.s_bot_dw + dw[5])
            .curve_to((W * (P.s_t2_x + t2[0]), y0 + height * (P.s_t2_y + t2[1])), name="t2", dir=P.s_t2_dir, terminal="s_bottom",
                      width=P.s_t2_dw + dw[6], pen_angle=P.s_t2_pa))

    def tapered_bar(gl, x0, y0, x1, y1, stem_l):
        """A crossbar whose left end is a point: a pentagon, degenerate (a plain
        rectangle) at the defaults; always five points (principle 4)."""
        xa = x0 + (stem_l - x0) * P.bar_taper
        (gl.contour("bar").move((x0, y0), "bl").line((x1, y0), "br").line((x1, y1), "tr").line((xa, y1), "tl")
            .line((x0 - P.bar_tip_dx, y0 + (y1 - y0) * P.bar_tip_y), "tip").close())

    def cap_bowl(width, thick_k=1.0, thin_k=1.0, dfull=(0, 0, 0, 0), stress=0, right_k=None, bottom_k=None):
        """The O's bowl (also the G's and C's, with their own offsets; the C's
        right wall and bottom may differ: ``right_k`` x the thick wall,
        ``bottom_k`` x the thin one)."""
        ch = P.capheight
        th = P.cap_stem * P.round_gain * P.O_thick_k * thick_k
        tn = P.hairline * P.O_thin_k * thin_k
        return bowl(x=0, width=width, bottom=-os_, top=ch + os_, thick=th, thin=tn, stress=P.o_stress + stress,
                    thick_right=None if right_k is None else th * right_k,
                    thin_bottom=None if bottom_k is None else tn * bottom_k,
                    full=(P.o_full_h + P.O_full_dh + dfull[0], P.o_full_v + P.O_full_dv + dfull[1]),
                    in_full=(P.o_in_full_h + P.O_in_full_dh + dfull[2], P.o_in_full_v + P.O_in_full_dv + dfull[3]))

    def rect(gl, name, x0, y0, x1, y1):
        gl.contour(name).move((x0, y0), "bl").line((x1, y0), "br").line((x1, y1), "tr").line((x0, y1), "tl").close()

    # ---- glyphs ---------------------------------------------------------
    @g("o")
    def o(gl):
        gl += bowl(x=0, width=P.o_width, bottom=-os_, top=xh + os_, thick=P.stem * P.round_gain,
                   thin=P.hairline, stress=P.o_stress, full=(P.o_full_h, P.o_full_v),
                   in_full=(P.o_in_full_h, P.o_in_full_v))

    @g("n")
    def n(gl):
        gl += stem_(x=0, top=xh + P.head_top, head="head")
        gl += arch(x=0, width=P.n_width, inset=True)
        gl += stem_(x=P.n_width, top=xh * P.n_out_y * P.n_leg_top, name="leg", head=False)

    @g("H")
    def H(gl):
        ch = P.capheight
        gl += stem_(x=0, top=ch, pen=cap_pen, foot="cap_foot", head="cap_head", name="left")
        gl += stem_(x=P.H_width, top=ch, pen=cap_pen, foot="cap_foot", head="cap_head", name="right")
        y = ch * P.H_bar_y
        gl.contour("bar").move((0, y - P.H_bar_t / 2), "bl").line((P.H_width, y - P.H_bar_t / 2), "br") \
            .line((P.H_width, y + P.H_bar_t / 2), "tr").line((0, y + P.H_bar_t / 2), "tl").close()

    @g("e")
    def e(gl):
        w = P.e_width
        th = P.stem * P.round_gain
        # the e's walls may differ from the o's (it shares round_gain)
        gl += bowl(x=0, width=w, bottom=-os_, top=xh + os_, thick=th + P.e_thick_dl, thin=P.hairline,
                   stress=P.e_stress, full=(P.e_full_h, P.e_full_v), in_full=(P.e_in_full_h, P.e_in_full_v),
                   thin_bottom=P.hairline * P.e_thin_bottom, thick_right=th + P.e_thick_dr)
        y = xh * P.e_bar_y
        t = P.e_bar_t
        bx = w - P.e_bar_rin            # the bar stops short of the bowl's curved outer edge
        gl.contour("bar").move((th * 0.5, y - t / 2), "bl").line((bx, y - t / 2), "br") \
            .line((bx, y + t / 2), "tr").line((th * 0.5, y + t / 2), "tl").close()
        # aperture: right of the counter, between the bar and the terminal.
        # Its lower edge *is* the terminal's upper edge: it leaves the
        # terminal tip at ``e_cut_angle`` and runs into the counter.
        R = w + 120
        tip = (w, P.e_term_y)
        d_ = (w * P.e_cut_x, P.e_cut_y)
        a1 = math.radians(val(P.e_cut_angle))
        L = math.hypot(val(d_[0]) - val(tip[0]), val(d_[1]) - val(tip[1]))
        c1 = (tip[0] + math.cos(a1) * L * P.e_cut_h1, tip[1] + math.sin(a1) * L * P.e_cut_h1)
        c2 = (d_[0] + L * P.e_cut_h2, d_[1])        # arrives at the counter travelling left
        (gl.cut("aperture").move((w * P.e_cut_x, y - t / 2), "a").line((R, y - t / 2), "b")
            .line((R, P.e_term_y), "c").line(tip, "tip").curve(c1, c2, d_, "d").close())

    @g("a")
    def a(gl):
        sx = P.a_stem_x
        stem_top = P.a_stem_top + P.a_stem_up
        # the a's stem may be a little lighter than the n's, and dip below the baseline
        a_pen = Pen(angle=0, width=P.stem * P.a_stem_w, contrast=1 - P.hairline / (P.stem * P.a_stem_w))
        gl += stem_(x=sx, top=stem_top, bottom=P.a_stem_bot, foot="spur", head=False, pen=a_pen)
        hood_pen = Pen(angle=P.a_hood_angle, width=P.stem, contrast=1 - P.hairline * P.a_hood_thin / P.stem)
        top_y = xh + os_ - hood_pen.width_at(0) * (1 - val(P.a_bias))
        (gl.path(name="hood", pen=hood_pen, bias=P.a_bias,
                 profile=Profile((0, P.a_taper), (P.a_hood_end, 1), (1, 1)))
            .start((P.a_term_x, P.a_term_y), name="terminal", dir=P.a_term_dir, terminal="hood",
                   pen_angle=P.a_term_pa)
            .curve_to((P.a_term_x + (sx - P.a_term_x) * P.a_top_x, top_y), name="top", dir=0, pen_angle=P.a_top_pa)
            # always end inside the stem, so hood and stem overlap
            .curve_to((sx, min(val(P.a_stem_top + (xh - P.a_stem_top) * (1 - P.a_out_y) * 0.5),
                               val(stem_top) - 6)), name="out", dir=-90))
        # the bowl hangs on the stem's left edge at two joins
        gl.lobe(sx - P.stem * P.a_stem_w / 2, -1, P.a_far, -os_,
                join(P.a_top_at, P.a_top_thin, P.a_top_angle),
                # the bottom join never rises past the top one (the lobe would fold)
                join(lazy_min(P.a_bot_at, P.a_top_at), P.a_bot_thin, P.a_bot_angle),
                thick=P.a_lobe_thick, thin_bottom=P.a_lobe_thin_b, out_mid=P.a_out_mid, in_mid=P.a_in_mid,
                bx_out=P.a_bx_out, bx_in=P.a_bx_in, tension=P.a_lobe_t, in_tension=P.a_lobe_ti, prefix="bowl.")

    @g("s")
    def s(gl):
        gl += spine(width=P.s_width, height=xh, stem_w=P.stem)

    @g("g")
    def g_(gl):
        gl += bowl(x=P.g_bl, width=P.g_br - P.g_bl, bottom=P.g_bb, top=P.g_bt,
                   thick=P.stem * P.g_thick, thin=P.hairline, full=((P.g_full, P.g_full + P.g_full_dv),       # br, tr, tl, bl: the
                         (P.g_full + P.g_tr_dh, P.g_full + P.g_full_dv + P.g_tr_dv),   # shoulder
                         (P.g_full, P.g_full + P.g_full_dv),       # under the ear
                         (P.g_full, P.g_full + P.g_full_dv)),      # sits lower
                   in_full=(P.g_in_full, P.g_in_full + P.g_in_full_dv), rotate=P.g_rot, name="bowl")
        cx = (P.g_bl + P.g_br) / 2
        if g_ear == "flag":
            # four-sided flag: flat bottom, vertical outer edge, sloped top
            rx = cx + (P.g_br - cx) * P.g_ear_root_x
            (gl.contour("ear").move((rx, P.g_ear_bot), "root_bottom").line((P.g_ear_x, P.g_ear_bot), "bottom")
                .line((P.g_ear_x, P.g_ear_bot + P.g_ear_h), "top").line((rx + P.g_ear_root_dx, P.g_ear_root_y), "root_top")
                .close())
        else:
            (gl.path(name="ear", profile=constant(P.g_ear_w))
                .start((cx + (P.g_br - cx) * 0.6, P.g_bt - 30), name="root", terminal=False)
                .curve_to((P.g_ear_x, P.g_ear_y), name="tip", dir=10, terminal="ear"))
        # link: leaves the bowl's bottom, swings out left, turns into the loop's top band
        # hairline through the turn, heavy where it runs into the loop's top band
        (gl.path(name="link", profile=Profile((0, P.g_link_w0), (P.g_link_tp, P.g_link_w0),
                                              (1, P.g_link_w1)))
            .start((P.g_bl + (cx - P.g_bl) * P.g_link_x0, P.g_bb + 12 + P.g_link_root_dy), name="root",
                   dir=P.g_link_root_dir, terminal=False)
            .curve_to((P.g_link_x, P.g_link_y), name="turn", dir=270)
            # where the link bottoms out and swings into the loop's top band
            .curve_to((P.g_link_x + (P.g_band_x - P.g_link_x) * P.g_low_t,
                       P.g_band_y + (P.g_link_y - P.g_band_y) * P.g_low_y), name="low", dir=P.g_low_dir)
            .curve_to((P.g_band_x, P.g_band_y), name="band", dir=0))
        # lower loop: a contour bowl with its own walls (heavy top band)
        gl += bowl(x=P.g_loop_l, width=P.g_loop_r - P.g_loop_l, bottom=P.g_loop_b, top=P.g_loop_t,
                   thick=P.stem * P.g_loop_thick, thick_right=P.stem * P.g_loop_tr, thin=P.g_loop_top_t,
                   thin_bottom=P.g_loop_bot_t, full=(P.g_loop_full, P.g_loop_full_v),
                   in_full=(P.g_loop_in_full, P.g_loop_in_full_v),
                   rotate=P.g_loop_rot, stress=P.g_loop_stress, name="loop")
        # the notch between the link and the lower loop's top-left arm: a
        # wedge opening to the left, tip at (g_notch_x, g_notch_y), edges
        # g_notch_up / g_notch_dn degrees above/below the horizontal,
        # g_notch_len long (0: none -- the old g)
        nx, ny, nl = val(P.g_notch_x), val(P.g_notch_y), max(val(P.g_notch_len), 0.0)
        au, ad = math.radians(val(P.g_notch_up)), math.radians(val(P.g_notch_dn))
        (gl.cut("link_notch").move((nx, ny), "tip")
            .line((nx - nl * math.cos(au), ny + nl * math.sin(au)), "upper")
            .line((nx - nl * math.cos(ad), ny - nl * math.sin(ad)), "lower").close())
        if g_band == "flat":
            rect(gl, "bar", P.g_bar_l, P.g_bar_bot, P.g_bar_r, P.g_bar_top)
            # trim whatever of the tilted loop pokes above the bar (not the link)
            (gl.cut("bar_trim").move((P.g_bar_l + 10, P.g_bar_top), "a").line((P.g_loop_r + 40, P.g_bar_top), "b")
                .line((P.g_loop_r + 40, P.g_bar_top + 60), "c").line((P.g_bar_l + 10, P.g_bar_top + 60), "d").close())

    @g("v", sb=(P.v_sb, P.v_sb))
    def v(gl):
        W, vx = P.v_width, P.v_width * P.v_vertex
        thick = Pen(angle=0, width=P.v_thick, contrast=0)
        thin = Pen(angle=0, width=P.v_thin, contrast=0)
        # the thick arm tapers towards the vertex (``v_taper`` = width there / at the top)
        gl.line((0, xh), (vx, -P.v_os), name="left", names=("top", "vertex"), terminal=("vtop_l", "vertex"),
                pen=thick, profile=Profile((0, 1), (1, P.v_taper)))
        # both arms run to the vertex; below the crotch the thick arm is
        # clipped by the thin arm's outer edge, leaving a narrow flat point
        tx = vx + P.v_thin_dx
        gl.line((W, xh), (tx, -P.v_os), name="right", names=("top", "vertex"), terminal=("vtop_r", "vertex"),
                pen=thin)
        dx, dy = val(W) - val(tx), val(xh) + val(P.v_os)
        ln = math.hypot(dx, dy)
        off = val(P.v_thin) / 2 * ln / dy                   # horizontal half-width of the thin arm
        x_at = lambda y: val(tx) + off + (y + val(P.v_os)) * dx / dy   # thin arm's outer edge
        y0, y1 = -val(P.v_os) - 30, val(xh) * 0.35
        (gl.cut("vertex_clip").move((x_at(y0), y0), "a").line((x_at(y0) + 200, y0), "b")
            .line((x_at(y1) + 200, y1), "c").line((x_at(y1), y1), "d").close())

    # ---- the rest of the test word and capitals -------------------------
    asc = P.ascender

    @g("h")
    def h(gl):
        # the h's ascender stem may carry a slightly different weight than the n's
        h_pen = Pen(angle=0, width=P.stem * P.h_stem_w, contrast=1 - P.hairline / (P.stem * P.h_stem_w))
        gl += stem_(x=0, top=asc + P.head_top * 0.5 + P.h_top, head="head", pen=h_pen)
        w = P.n_width + P.h_dwidth
        gl += arch(x=0, width=w, edge_half=P.stem * P.h_stem_w / 2)   # no inset: the join is far below the head
        gl += stem_(x=w, top=xh * P.n_out_y * P.n_leg_top, name="leg", head=False)

    @g("m")
    def m(gl):
        w = P.m_width
        gl += stem_(x=0, top=xh + P.head_top, head="head")
        gl += arch(x=0, width=w, name="arch1", inset=True)
        gl += stem_(x=w, top=xh * P.n_out_y * P.n_leg_top, name="leg1", head=False)
        gl += arch(x=w, width=w, name="arch2")
        gl += stem_(x=2 * w, top=xh * P.n_out_y * P.n_leg_top, name="leg2", head=False)

    @g("u")
    def u(gl):
        w = P.u_width
        top = xh + P.head_top + P.u_head_dtop
        gl += stem_(x=0, bottom=xh * (1 - P.u_out_y * P.u_leg_top), top=top, head="u_head", foot=False, name="left")
        gl += arch(x=0, width=w, flip=True)
        gl += stem_(x=w, top=top, head="u_head", foot="spur", name="right")

    @g("b")
    def b(gl):
        if b_style == "lobe":
            gl += stem_(x=0, bottom=P.b_foot, top=asc + P.head_top * 0.5 + P.b_top, head="head", foot="b_foot")
            gl.lobe(P.stem / 2, +1, P.b_far, -os_,
                    join(P.b_top_at, P.b_top_thin, P.b_top_angle), join(P.b_bot_at, P.b_bot_thin, P.b_bot_angle),
                    thick=P.b_lobe_thick, thin_bottom=P.b_lobe_thin_b, out_mid=P.b_out_mid, in_mid=P.b_in_mid,
                    bx_out=P.b_bx_out, bx_in=P.b_bx_in, tension=P.b_lobe_t, in_tension=P.b_lobe_ti, prefix="bowl.",
                    crown=dict(x=P.b_crown_x, y=xh + os_, thick=P.b_crown_t))
            return
        gl += stem_(x=0, top=asc + P.head_top * 0.5, head="head", foot=False)
        x0 = P.stem * P.b_bowl_l
        gl += bowl(x=x0, width=P.b_width - x0, bottom=-os_, top=xh + os_, thick=P.stem * P.round_gain,
                   thin=P.hairline, stress=P.o_stress, full=(P.o_full_h, P.o_full_v),
                   in_full=(P.o_in_full_h, P.o_in_full_v))

    @g("r", sb=(0, P.r_rsb))
    def r(gl):
        gl += stem_(x=0, top=xh + P.head_top, head="head", foot="foot_wide")
        J = join(val(xh) * P.r_join_at, P.r_join_thin, P.r_join_angle)
        _, _, st, d0, thin = contact(J, val(P.stem / 2), +1, rising=True)
        join_handles(J, val(P.stem / 2), +1, rising=True)
        head_inset_cut(gl, P.stem / 2, val(xh) * P.r_join_at, lazy_max(P.head_inset + P.r_head_dinset, 0))
        (gl.path(name="ear", pen=arch_pen, bias=P.n_bias,
                 profile=Profile((0, thin / max(arch_pen.width_at(d0), 1.0)), (P.r_taper_len, 1), (1, P.r_end_w)))
            .start(st, name="join", dir=d0, terminal=False, pen_angle=P.r_join_pa)
            .curve_to((P.r_width, xh * P.r_y), name="end", dir=P.r_dir, terminal="r_ear", pen_angle=P.r_end_pa))

    def f_body(gl, bar_r=None, hook=None):
        """The f: stem, hook, bar (shared by f and the f_i ligature, whose hook
        reaches further: ``hook`` = the hook end's x)."""
        top = P.f_top
        hook = P.f_hook if hook is None else hook
        # the hook's nib: the arch's, turned by f_angle_d (a Didone f wants it
        # vertical -- at 25 degrees the hook went thick-thin-thick, a hump)
        hook_pen = Pen(angle=P.arch_angle + P.f_angle_d, width=P.stem, contrast=P.arch_contrast)
        y0 = xh + (top - xh) * 0.35
        gl += stem_(x=0, top=y0, head=False, foot="foot_wide")
        (gl.path(name="hook", pen=hook_pen, bias=P.n_bias, profile=Profile((0, 1), (0.6, 1), (1, P.f_end_w)))
            .start((0, y0 - 2), name="root", dir=90, terminal=False)
            .curve_to((hook * 0.45, top + os_ - hook_pen.width_at(0) / 2), name="top", dir=0,
                      width=P.f_top_dw, pen_angle=P.f_top_pa)
            .curve_to((hook, top * P.f_end_y), name="end", dir=P.f_end_dir, terminal="f_hook",
                      pen_angle=P.f_end_pa))
        tapered_bar(gl, -P.f_bar_l, xh - P.f_bar_t + P.f_bar_dy, P.f_bar_r if bar_r is None else bar_r,
                    xh + P.f_bar_dy, -P.stem / 2)

    @g("f")
    def f(gl):
        f_body(gl)

    # fi: the f's ball is the i's dot -- a dotless i stands under the hook's end,
    # and the f's bar runs across to it; the i stem stops at the bar's top
    if want is None or {"f", "i"} <= set(want):
        @g("f_i")
        def f_i(gl):
            xi = P.f_hook + P.fi_dx
            f_body(gl, bar_r=xi, hook=xi)
            gl += stem_(x=xi, top=xh + P.f_bar_dy, head=False, name="i_stem")

        font.ligature("f_i", ("f", "i"))

    @g("t")
    def t(gl):
        tail_pen = Pen(angle=P.arch_angle, width=P.stem, contrast=P.arch_contrast)
        # the t has its own pen: its tail is a thick crescent in some designs,
        # a hairline in others (normalised, so the vertical stem stays exact)
        t_pen = Pen(angle=P.t_angle, width=P.stem, contrast=P.t_contrast)
        (gl.path(name="stem", bias=0.5, pen=t_pen,
                 profile=Profile((0, 1), (0.55, 1), (0.8, P.t_turn_w), (1, P.t_end_w)))
            .start((0, P.t_top), name="top", terminal="t_top")
            .line_to((0, P.t_turn), name="turn")
            .curve_to((P.t_tail * 0.55, -os_ + t_pen.width_at(0) / 2), name="bottom", dir=0, pen_angle=P.t_bottom_pa)
            .curve_to((P.t_tail, P.t_tail_y), name="end", dir=P.t_end_dir, terminal="t_tail", pen_angle=P.t_end_pa))
        tapered_bar(gl, -P.t_bar_l, xh - P.t_bar_t + P.t_bar_dy, P.t_bar_r, xh + P.t_bar_dy, -P.stem / 2)

    @g("i")
    def i(gl):
        gl += stem_(x=0, top=xh + P.head_top, head="head")
        r = P.i_dot_r
        k = 0.5523 * r
        y = P.i_dot_y
        (gl.contour("dot").move((0, y - r), "bottom").curve((k, y - r), (r, y - k), (r, y), "right")
            .curve((r, y + k), (k, y + r), (0, y + r), "top").curve((-k, y + r), (-r, y + k), (-r, y), "left")
            .curve_close((-r, y - k), (-k, y - r)).close())

    @g("O")
    def O(gl):
        ch = P.capheight
        gl += cap_bowl(P.O_width)

    @g("A", sb=(P.A_sb, P.A_sb))
    def A(gl):
        ch, W = P.capheight, P.A_width
        thick = Pen(angle=0, width=P.A_thick, contrast=0)
        thin = Pen(angle=0, width=P.A_thin, contrast=0)
        ax = W / 2
        top = ch + os_ * P.A_os_k
        gl.line((ax - P.A_apex * 0.5, top), (0, 0), name="left", names=("apex", "foot"),
                terminal=("apex", "A_lfoot"), pen=thin)
        gl.line((ax + P.A_apex * 0.5 - P.A_thick * 0.3, top), (W, 0), name="right", names=("apex", "foot"),
                terminal=("apex", "A_rfoot"), pen=thick)
        y = ch * P.A_bar_y
        k = top / (W / 2)
        rect(gl, "bar", y / k, y - P.bar_t / 2, W - y / k, y + P.bar_t / 2)

    @g("E")
    def E(gl):
        ch, W = P.capheight, P.E_width
        gl += stem_(x=0, top=ch, pen=cap_pen, foot="cap_foot_l", head="cap_foot_l")
        t, sw, sh = P.E_arm_t, P.E_serif_w, P.E_serif
        rect(gl, "arm_top", 0, ch - t, W, ch)
        rect(gl, "arm_bottom", 0, 0, W * P.E_bot_k, t)
        y = ch * P.E_mid_y
        rect(gl, "arm_mid", 0, y - t / 2, W * P.E_mid, y + t / 2)
        if E_serifs == "rect":
            rect(gl, "serif_top", W - sw, ch - sh, W, ch)
            rect(gl, "serif_bottom", W * P.E_bot_k - sw, 0, W * P.E_bot_k, sh * P.E_bot_sk)
            rect(gl, "serif_mid", W * P.E_mid - sw, y - sh * P.E_mid_sk, W * P.E_mid, y + sh * P.E_mid_sk)
        else:
            # wedges: a vertical face sw wide at the arm's end, its inner edge
            # sloping back to meet the arm E_wedge further in
            L = P.E_wedge
            def wedge_(name, x, ya, yb):        # arm edge at ya, serif tip at yb
                pts = [((x - L, ya), "root"), ((x, ya), "a"), ((x, yb), "tip"), ((x - sw, yb), "tip_in")]
                if val(yb) > val(ya):           # counter-clockwise, as rect() draws
                    pts = pts[::-1]
                c = gl.contour(name).move(*pts[0])
                for pt in pts[1:]:
                    c = c.line(*pt)
                c.close()
            wedge_("serif_top", W, ch - t + 1, ch - sh)
            wedge_("serif_bottom", W * P.E_bot_k, t - 1, sh * P.E_bot_sk)
            Wm = W * P.E_mid
            wedge_("serif_mid_up", Wm, y, y + sh * P.E_mid_sk)
            wedge_("serif_mid_dn", Wm, y, y - sh * P.E_mid_sk)

    @g("G")
    def G(gl):
        ch, W = P.capheight, P.G_width
        gl += cap_bowl(W, P.G_thick_k, P.G_thin_k, (P.G_full_dh, P.G_full_dv, P.G_in_full_dh, P.G_in_full_dv),
                       P.G_stress)
        yb, yt = ch * P.G_bar_y, ch * P.G_term_y
        R = W + 100
        ry, cy = (ch + 2 * os_) / 2, ch / 2
        q = max(0.0, 1 - ((val(yt) - val(cy)) / val(ry)) ** 2)
        ex = W / 2 + (W / 2) * q ** 0.42          # outer edge of the bowl at the terminal
        bh = P.G_serif * P.G_beard_k              # beard length
        bw = P.E_serif_w + P.G_beard_dw           # beard width
        if G_jaw == "serif":
            # the terminal is the bowl's own wall cut off vertically at
            # G_face of the width: the opening below yt, the face above it
            xf = W * P.G_face
            (gl.cut("aperture").move((W * 0.5, yb), "a").line((R, yb), "b").line((R, ch + 200), "c")
                .line((xf, ch + 200), "d").line((xf, yt), "e").line((W * 0.5, yt), "f").close())
        else:
            # aperture steps around the beard so the beard survives the cut
            (gl.cut("aperture").move((W * 0.5, yb), "a").line((R, yb), "b").line((R, yt - bh), "c")
                .line((ex - bw, yt - bh), "d").line((ex - bw, yt), "e").line((W * 0.5, yt), "f").close())
        # the jaw: a stem whose outer edge stays at the bowl's right, weight G_spur_w x cap stem
        sw_ = P.cap_stem * P.G_spur_w
        # G_jaw_in: the jaw's outer edge moves from the bowl's widest point (0)
        # to the bowl's own outer edge at the bar's height (1), so the curve
        # below continues it rather than the jaw standing outside the bowl
        qb = max(0.0, 1 - ((val(cy) - val(yb)) / val(ry)) ** 2)
        exb = W / 2 + (W / 2) * qb ** 0.42
        sx = W - (W - exb) * P.G_jaw_in - sw_ * 0.5
        sx = sx + P.G_jaw_dx                     # the jaw may stand inside the bowl's edge
        spur_pen = Pen(angle=0, width=sw_, contrast=1 - P.hairline / sw_)
        # the jaw may stop inside the bowl's curve (G_spur_bot, a share of the cap
        # height) instead of standing on the baseline; its foot is the G_spur tag
        if G_jaw == "serif":
            # the bowl stops at the jaw's outer edge (it would bulge past it),
            # up to where the head serif's bracket takes over (the trim must
            # not reach the serif, whose right half stands over the bowl)
            ytr = yb - P.G_head_bracket
            (gl.cut("jaw_trim").move((sx + sw_ / 2, -os_ - 60), "a").line((R, -os_ - 60), "b")
                .line((R, ytr), "c").line((sx + sw_ / 2, ytr), "d").close())
            gl += stem_(x=sx, bottom=ch * P.G_spur_bot, top=yb, pen=spur_pen, foot="G_foot", head="G_head",
                        name="spur")
        else:
            gl += stem_(x=sx, bottom=ch * P.G_spur_bot, top=yb, pen=spur_pen, foot="G_spur", head=False,
                        name="spur")
            rect(gl, "bar", sx - sw_ / 2 - P.G_serif, yb - P.bar_t, sx + sw_ / 2, yb)
            rect(gl, "beard", ex - bw, yt - bh, ex, yt + 2)

    @g("S")
    def S(gl):
        gl += spine(width=P.S_width, height=P.capheight, stem_w=P.cap_stem, swell_k=P.S_swell_k,
                    t1=(P.S_t1_dx, P.S_t1_dy), t2=(P.S_t2_dx, P.S_t2_dy), mid=(P.S_mid_dx, P.S_mid_dy),
                    sides=(P.S_left_dy, P.S_right_dy), tops=(P.S_top_dx, P.S_bot_dx),
                    dw=(P.S_t1_dw, P.S_top_dw, P.S_left_dw, P.S_mid_dw, P.S_right_dw, P.S_bot_dw, P.S_t2_dw))

    # ---- the rest of the lowercase --------------------------------------
    def face_wedge(gl, name, x, ya, yb, L, sw, opens=-1):
        """A wedge serif: a vertical face sw wide at x from the arm edge ya to
        the tip yb, its inner edge sloping back L along the arm (opens=-1:
        the arm runs left of x, as on an E; +1: right)."""
        pts = [((x + opens * L, ya), "root"), ((x, ya), "a"), ((x, yb), "tip"), ((x + opens * sw, yb), "tip_in")]
        if (val(yb) > val(ya)) == (opens < 0):     # counter-clockwise either way
            pts = pts[::-1]
        poly_(gl, name, pts)

    @g("l")
    def l(gl):
        gl += stem_(x=0, top=asc + P.head_top * 0.5 + P.h_top, head="head")

    def d_lobe(gl, far):
        gl.lobe(-P.stem / 2, -1, far, -os_,
                join(P.d_top_at, P.d_top_thin, P.d_top_angle), join(P.d_bot_at, P.d_bot_thin, P.d_bot_angle),
                thick=P.d_lobe_thick, thin_bottom=P.d_lobe_thin_b, out_mid=P.d_out_mid, in_mid=P.d_in_mid,
                bx_out=P.d_bx_out, bx_in=P.d_bx_in, prefix="bowl.",
                crown=dict(x=P.d_crown_x, y=xh + os_, thick=P.d_crown_t))

    @g("d")
    def d(gl):
        gl += stem_(x=0, top=asc + P.head_top * 0.5 + P.h_top, head="head", foot="spur")
        d_lobe(gl, -P.d_far)

    @g("q")
    def q(gl):
        gl += stem_(x=0, bottom=P.desc_stem, top=xh + os_ + P.q_top, head="apex", foot="foot")
        d_lobe(gl, -P.d_far - P.q_dfar)

    @g("p")
    def p_(gl):
        gl += stem_(x=0, bottom=P.desc_stem, top=xh + P.head_top, head="head", foot="foot")
        gl.lobe(P.stem / 2, +1, P.b_far + P.p_dfar, -os_,
                join(P.b_top_at, P.b_top_thin, P.b_top_angle), join(P.b_bot_at, P.b_bot_thin, P.b_bot_angle),
                thick=P.b_lobe_thick, thin_bottom=P.b_lobe_thin_b, out_mid=P.b_out_mid, in_mid=P.b_in_mid,
                bx_out=P.b_bx_out, bx_in=P.b_bx_in, tension=P.b_lobe_t, in_tension=P.b_lobe_ti, prefix="bowl.",
                crown=dict(x=P.b_crown_x, y=xh + os_, thick=P.b_crown_t))

    @g("c")
    def c(gl):
        w = P.c_width
        th = P.stem * P.round_gain
        gl += bowl(x=0, width=w, bottom=-os_, top=xh + os_, thick=th + P.e_thick_dl, thin=P.hairline,
                   stress=P.e_stress, full=(P.e_full_h, P.e_full_v), in_full=(P.e_in_full_h, P.e_in_full_v),
                   thin_bottom=P.hairline * P.e_thin_bottom, thick_right=th + P.e_thick_dr)
        R = w + 120
        xf = w * P.c_face
        tip = (w, P.c_term_y)
        d_ = (w * P.c_cut_x, P.c_cut_y)
        a1 = math.radians(val(P.c_cut_angle))
        L = math.hypot(val(d_[0]) - val(tip[0]), val(d_[1]) - val(tip[1]))
        c1 = (tip[0] + math.cos(a1) * L * P.c_cut_h1, tip[1] + math.sin(a1) * L * P.c_cut_h1)
        c2 = (d_[0] + L * P.c_cut_h2, d_[1])
        (gl.cut("aperture").move((w * P.c_cut_x, P.c_top_y), "a").line((xf, P.c_top_y), "face")
            .line((xf, xh + 200), "b").line((R, xh + 200), "c").line((R, P.c_term_y), "d")
            .line(tip, "tip").curve(c1, c2, d_, "e").close())

    @g("j")
    def j(gl):
        gl += stem_(x=0, bottom=P.j_bot, top=xh + P.head_top, head="head", foot=False)
        hook_pen = Pen(angle=0, width=P.stem, contrast=1 - P.hairline / P.stem)
        (gl.path(name="hook", pen=hook_pen, profile=Profile((0, 1), (0.5, 1), (1, P.j_end_w)))
            .start((0, P.j_bot + 2), name="root", dir=270, terminal=False)
            .curve_to((-P.j_hook, P.j_end_y), name="end", dir=P.j_end_dir, terminal="ear"))   # a blunt flag
        r, k = P.i_dot_r, 0.5523 * P.i_dot_r
        dot(gl, "dot", 0, P.i_dot_y, P.i_dot_r)

    @g("k")
    def k_(gl):
        gl += stem_(x=0, top=asc + P.head_top * 0.5 + P.h_top, head="head", foot="foot")
        thin = Pen(angle=0, width=P.k_thin, contrast=0)
        thick = Pen(angle=0, width=P.k_thick, contrast=0)
        gl.line((P.k_arm_x, xh), (0, P.k_join_y), name="arm", names=("top", "join"),
                terminal=("vtop_r", False), pen=thin)
        gl.line((P.k_leg_x0, P.k_leg_y0), (P.k_leg_x, 0), name="leg", names=("top", "foot"),
                terminal=(False, "vertex"), pen=thick)     # Geist's k leg: no foot, a flat end

    @g("w", sb=(P.v_sb, P.v_sb))
    def w(gl):
        W = P.w_width
        thick = Pen(angle=0, width=P.w_thick, contrast=0)
        thin = Pen(angle=0, width=P.w_thin, contrast=0)
        v1, v2, mx = W * P.w_v1, W * P.w_v2, W * 0.5 + P.w_mid_dx
        gl.line((0, xh), (v1, -P.v_os), name="a1", names=("top", "vertex"), terminal=("vtop_l", "vertex"), pen=thick)
        t1, t2 = v1 + P.w_thin_dx, v2 + P.w_thin_dx
        a2t = (t1 + (mx - t1) * P.w_a2_top, -P.v_os + (xh + P.v_os) * P.w_a2_top)
        gl.line(a2t, (t1, -P.v_os), name="a2", names=("top", "vertex"), terminal=(False, "vertex"), pen=thin)
        gl.line((mx + P.w_mid3_dx, xh), (v2, -P.v_os), name="a3", names=("top", "vertex"), terminal=("apex", "vertex"), pen=thick)
        gl.line((W, xh), (t2, -P.v_os), name="a4", names=("top", "vertex"), terminal=("vtop_r", "vertex"), pen=thin)

    @g("x")
    def x_(gl):
        W = P.x_width
        thick = Pen(angle=0, width=P.x_thick, contrast=0)
        thin = Pen(angle=0, width=P.x_thin, contrast=0)
        gl.line((0, xh), (W, 0), name="fall", names=("top", "foot"), terminal=("vtop_l", "vfoot_r"), pen=thick)
        gl.line((W, xh), (P.x_thin_dx, 0), name="rise", names=("top", "foot"), terminal=("vtop_r", "vfoot_l"), pen=thin)

    @g("y", sb=(P.v_sb, P.v_sb))
    def y_(gl):
        W, vx = P.y_width, P.y_width * P.y_vertex
        thick = Pen(angle=0, width=P.v_thick, contrast=0)
        thin = Pen(angle=0, width=P.v_thin, contrast=0)
        gl.line((0, xh), (vx, -P.v_os), name="left", names=("top", "vertex"), terminal=("vtop_l", "vertex"), pen=thick)
        # the thin arm, straight on past the vertex into the tail
        dx, dy = val(W) - val(vx), val(xh) + val(P.v_os)
        tx = val(vx) - (val(P.y_tail_y) + val(P.v_os)) * -dx / dy
        ang = math.degrees(math.atan2(-dy, -dx))
        (gl.path(name="tail", pen=thin, profile=Profile((0, 1), (0.6, 1), (1, P.y_end_w)))
            .start((W, xh), name="top", dir=ang, terminal="vtop_r")
            .line_to((tx, P.y_tail_y), name="bend", dir=ang, terminal=False)
            .curve_to((P.y_end_x, P.y_end_y), name="end", dir=180, terminal="ear"))   # a blunt, vertically cut flag

    @g("z")
    def z(gl):
        W, t = P.z_width, P.z_bar_t
        rect(gl, "top", 0, xh - t, W, xh)
        rect(gl, "bottom", 0, 0, W, t)
        hw = P.z_thick / 2
        poly_(gl, "diag", [((0, t - 1), "bl"), ((hw * 2, t - 1), "br"), ((W, xh - t + 1), "tr"),
                           ((W - hw * 2, xh - t + 1), "tl")])
        face_wedge(gl, "serif_top", 0, xh - t + 1, xh - P.z_serif, P.z_wedge, P.z_serif_w, opens=+1)
        face_wedge(gl, "serif_bottom", W, t - 1, P.z_serif, P.z_wedge, P.z_serif_w, opens=-1)

    # ---- the rest of the capitals ------------------------------------------
    def dbowl(gl, name, sx, y0, y1, far, flat, th, tt, tb, full, in_full, flat_bot=None):
        """A D-shaped bowl right of a stem whose right edge is at sx: flat arms
        from the stem to ``flat`` (from the stem's centre), curving round to
        the outer far extreme ``far``. Walls: far ``th``, top ``tt``, bottom
        ``tb``. The counter is cut out."""
        fb = flat if flat_bot is None else flat_bot     # the bottom arm's flat may differ

        def shape(c, x0, ya, yb, fxa, fxb, rx, k):
            ym = (ya + yb) / 2
            c.move((x0, ya), "bl").line((fxa, ya), "b").curve((fxa + (rx - fxa) * k, ya), (rx, ym - (ym - ya) * k), (rx, ym), "far") \
             .curve((rx, ym + (yb - ym) * k), (fxb + (rx - fxb) * k, yb), (fxb, yb), "t").line((x0, yb), "tl").close()
        shape(gl.contour(name), sx - 1, y0, y1, fb, flat, far, full)
        shape(gl.cut(name + "_counter"), sx, y0 + tb, y1 - tt, fb - th * 0.3, flat - th * 0.3, far - th, in_full)

    def aperture_(gl, w, h, face, top_y, term_y, cut_x, cut_y, cut_angle, h1, h2):
        """The c / C opening: a vertical face above, a curved lower terminal."""
        R = w + 140
        xf = w * face
        tip = (w, term_y)
        d_ = (w * cut_x, cut_y)
        a1 = math.radians(val(cut_angle))
        L = math.hypot(val(d_[0]) - val(tip[0]), val(d_[1]) - val(tip[1]))
        c1 = (tip[0] + math.cos(a1) * L * h1, tip[1] + math.sin(a1) * L * h1)
        c2 = (d_[0] + L * h2, d_[1])
        (gl.cut("aperture").move((w * cut_x, top_y), "a").line((xf, top_y), "face")
            .line((xf, h + 200), "b").line((R, h + 200), "c").line((R, term_y), "d")
            .line(tip, "tip").curve(c1, c2, d_, "e").close())

    def cap_stem(gl, x=0, head="cap_head", foot="cap_foot", name="stem", bottom=0, top=None, pen=None):
        gl += stem_(x=x, bottom=bottom, top=P.capheight if top is None else top, pen=pen or cap_pen,
                    foot=foot, head=head, name=name)

    @g("I")
    def I(gl):
        cap_stem(gl)

    @g("L")
    def L_(gl):
        ch, W, t = P.capheight, P.L_width, P.E_arm_t
        cap_stem(gl, foot="cap_foot_l")
        rect(gl, "arm", 0, 0, W, t)
        face_wedge(gl, "serif", W, t - 1, P.E_serif * P.E_bot_sk, P.E_wedge, P.E_serif_w)

    @g("F")
    def F(gl):
        ch, W, t = P.capheight, P.F_width, P.E_arm_t
        cap_stem(gl, head="cap_foot_l")
        rect(gl, "arm_top", 0, ch - t, W, ch)
        face_wedge(gl, "serif_top", W, ch - t + 1, ch - P.E_serif, P.E_wedge, P.E_serif_w)
        y, Wm = ch * P.F_mid_y, W * P.F_mid
        rect(gl, "arm_mid", 0, y - t / 2, Wm, y + t / 2)
        face_wedge(gl, "serif_mid_up", Wm, y, y + P.E_serif * P.E_mid_sk, P.E_wedge, P.E_serif_w)
        face_wedge(gl, "serif_mid_dn", Wm, y, y - P.E_serif * P.E_mid_sk, P.E_wedge, P.E_serif_w)

    @g("T")
    def T(gl):
        ch, W, t = P.capheight, P.T_width, P.E_arm_t
        cap_stem(gl, head=False)
        rect(gl, "bar", -W / 2, ch - t, W / 2, ch)
        face_wedge(gl, "serif_l", -W / 2, ch - t + 1, ch - P.T_serif, P.T_wedge, P.E_serif_w, opens=+1)
        face_wedge(gl, "serif_r", W / 2, ch - t + 1, ch - P.T_serif, P.T_wedge, P.E_serif_w, opens=-1)

    @g("B")
    def B(gl):
        ch = P.capheight
        cap_stem(gl, head="cap_foot_l", foot="cap_foot_l")
        ym = ch * P.B_mid
        sx = P.cap_stem / 2
        dbowl(gl, "upper", sx, ym - P.B_hair / 2, ch, P.B_far_t, P.B_flat_t, P.B_th, P.B_hair, P.B_hair / 2,
              P.B_full, P.B_in_full, flat_bot=P.B_flat_m)
        dbowl(gl, "lower", sx, 0, ym + P.B_hair / 2, P.B_far_b, P.B_flat_m, P.B_th, P.B_hair / 2, P.B_hair,
              P.B_full, P.B_in_full, flat_bot=P.B_flat_b)

    @g("D")
    def D(gl):
        ch = P.capheight
        cap_stem(gl, head="cap_foot_l", foot="cap_foot_l")
        dbowl(gl, "bowl", P.cap_stem / 2, 0, ch, P.D_far, P.D_flat, P.D_th, P.B_hair, P.B_hair,
              P.B_full, P.B_in_full)

    def p_bowl(gl, far, bot):
        ch = P.capheight
        dbowl(gl, "bowl", P.cap_stem / 2, bot, ch, far, P.P_flat, P.B_th, P.B_hair, P.B_hair,
              P.B_full, P.B_in_full)

    @g("P")
    def P_(gl):
        cap_stem(gl, head="cap_foot_l")
        p_bowl(gl, P.P_far, P.P_bot)

    @g("R")
    def R(gl):
        cap_stem(gl, head="cap_foot_l")
        p_bowl(gl, P.R_far, P.R_bot)
        leg = Pen(angle=0, width=P.R_leg_w, contrast=0)
        gl.line((P.R_leg_x0, P.R_bot + P.B_hair / 2), (P.R_leg_x, 0), name="leg", names=("top", "foot"),
                terminal=(False, "vertex"), pen=leg)

    @g("C")
    def C(gl):
        gl += cap_bowl(P.C_width, P.C_thick_k, 1.0, (0, 0, P.C_in_full_dh, P.C_in_full_dv), P.C_stress,
                       right_k=P.C_right_k, bottom_k=P.C_bottom_k)
        aperture_(gl, P.C_width, P.capheight, P.C_face, P.C_top_y, P.C_term_y, P.C_cut_x, P.C_cut_y,
                  P.C_cut_angle, P.C_cut_h1, P.C_cut_h2)

    @g("Q")
    def Q(gl):
        gl += cap_bowl(P.O_width)
        poly_(gl, "tail", [((P.Q_t0, 6), "t0"), ((P.Q_tip_x, P.Q_tip_y), "tip"),
                           ((P.Q_end_x, P.Q_end_y), "end"), ((P.Q_t1, -17), "t1")])

    @g("J")
    def J(gl):
        cap_stem(gl, bottom=P.J_bot, foot=False)
        (gl.path(name="hook", pen=cap_pen, profile=Profile((0, 1), (0.5, 1), (1, P.J_end_w)))
            .start((0, P.J_bot + 2), name="root", dir=270, terminal=False)
            .curve_to((-P.J_hook, P.J_end_y), name="end", dir=P.J_end_dir, terminal="ear"))

    @g("K")
    def K(gl):
        ch = P.capheight
        cap_stem(gl)
        thin = Pen(angle=0, width=P.K_thin, contrast=0)
        thick = Pen(angle=0, width=P.K_thick, contrast=0)
        gl.line((P.K_arm_x, ch), (0, P.K_join_y), name="arm", names=("top", "join"),
                terminal=("vtop_r", False), pen=thin)
        gl.line((P.K_leg_x0, P.K_leg_y0), (P.K_leg_x, 0), name="leg", names=("top", "foot"),
                terminal=(False, "vertex"), pen=thick)    # Geist's K leg ends flat, as its k's does

    @g("M")
    def M(gl):
        ch, W = P.capheight, P.M_width
        thin = Pen(angle=0, width=P.M_thin, contrast=0)
        thick = Pen(angle=0, width=P.M_thick, contrast=0)
        cap_stem(gl, x=0, head="cap_foot_l", pen=thin, name="left")
        cap_stem(gl, x=W, name="right")
        vx = W * P.M_vertex
        # the diagonals' tops may stand off the stems' centres (M_fall_dx,
        # M_rise_dx), and the thin one's foot off the thick one's (M_vdx)
        gl.line((P.M_fall_dx, ch), (vx, 0), name="fall", names=("top", "vertex"), terminal=("apex", "vertex"), pen=thick)
        gl.line((W + P.M_rise_dx, ch), (vx + P.M_vdx, 0), name="rise", names=("top", "vertex"),
                terminal=("apex", "vertex"), pen=thin)

    @g("N")
    def N_(gl):
        ch, W = P.capheight, P.N_width
        thin = Pen(angle=0, width=P.N_thin, contrast=0)
        thick = Pen(angle=0, width=P.N_thick, contrast=0)
        cap_stem(gl, x=0, head="cap_foot_l", pen=thin, name="left")
        cap_stem(gl, x=W, bottom=-os_, foot="vertex", pen=thin, name="right")
        gl.line((P.N_dx0, ch), (W + P.N_dx1, -os_), name="diag", names=("top", "foot"), terminal=("apex", "vertex"),
                pen=thick)
        # nothing of the diagonal stands outside the right stem, or below the
        # stem's foot (its flat-cut corner would poke through as a tick)
        xr = W + P.N_thin / 2
        (gl.cut("diag_clip").move((W - 250, -os_ - 60), "a").line((xr + 300, -os_ - 60), "b")
            .line((xr + 300, ch * 0.5), "c").line((xr, ch * 0.5), "d").line((xr, -os_), "e")
            .line((W - 250, -os_), "f").close())

    @g("U")
    def U(gl):
        ch, W = P.capheight, P.U_width
        thickp = Pen(angle=0, width=P.U_thick, contrast=1 - P.hairline / P.U_thick)
        thinp = Pen(angle=0, width=P.U_thin, contrast=0)
        cap_stem(gl, x=0, bottom=P.U_join, foot=False, pen=thickp, name="left")
        cap_stem(gl, x=W, bottom=P.U_join, foot=False, pen=thinp, name="right")
        (gl.path(name="cup", pen=thickp, profile=Profile((0, 1), (0.45, 1), (1, P.U_end_w)))
            .start((0, P.U_join + 2), name="left", dir=270, terminal=False)
            .curve_to((W * 0.48, -os_ + P.hairline / 2), name="bottom", dir=0)
            .curve_to((W, P.U_join + 2), name="right", dir=90, terminal=False))

    @g("V", sb=(P.v_sb, P.v_sb))
    def V(gl):
        ch, W, vx = P.capheight, P.V_width, P.V_width * P.V_vertex
        thick = Pen(angle=0, width=P.V_thick, contrast=0)
        thin = Pen(angle=0, width=P.V_thin, contrast=0)
        gl.line((0, ch), (vx, -os_), name="left", names=("top", "vertex"), terminal=("vtop_l", "vertex"), pen=thick)
        gl.line((W, ch), (vx + P.V_thin_dx, -os_), name="right", names=("top", "vertex"), terminal=("vtop_r", "vertex"),
                pen=thin)

    @g("W", sb=(P.v_sb, P.v_sb))
    def W_(gl):
        ch, W = P.capheight, P.W_width
        thick = Pen(angle=0, width=P.V_thick, contrast=0)
        thin = Pen(angle=0, width=P.V_thin, contrast=0)
        v1, v2, mx = W * P.W_v1, W * P.W_v2, W * 0.5 + P.W_mid_dx
        gl.line((0, ch), (v1, -os_), name="a1", names=("top", "vertex"), terminal=("vtop_l", "vertex"), pen=thick)
        # the thin middle stroke may stop short of the cap height (W_a2_top),
        # where it runs into the thick one -- Geist's does, at about 0.8
        t1, t2 = v1 + P.W_thin_dx, v2 + P.W_thin_dx
        a2t = (t1 + (mx - t1) * P.W_a2_top, -os_ + (ch + os_) * P.W_a2_top)
        gl.line(a2t, (t1, -os_), name="a2", names=("top", "vertex"), terminal=(False, "vertex"), pen=thin)
        gl.line((mx + P.W_mid3_dx, ch), (v2, -os_), name="a3", names=("top", "vertex"), terminal=("apex", "vertex"),
                pen=thick)
        gl.line((W, ch), (t2, -os_), name="a4", names=("top", "vertex"), terminal=("vtop_r", "vertex"), pen=thin)

    @g("X")
    def X(gl):
        ch, W = P.capheight, P.X_width
        thick = Pen(angle=0, width=P.X_thick, contrast=0)
        thin = Pen(angle=0, width=P.X_thin, contrast=0)
        gl.line((0, ch), (W, 0), name="fall", names=("top", "foot"), terminal=("vtop_l", "vfoot_r"), pen=thick)
        gl.line((W, ch), (P.X_thin_dx, 0), name="rise", names=("top", "foot"), terminal=("vtop_r", "vfoot_l"), pen=thin)

    @g("Y")
    def Y(gl):
        ch, W = P.capheight, P.Y_width
        thick = Pen(angle=0, width=P.Y_thick, contrast=0)
        thin = Pen(angle=0, width=P.Y_thin, contrast=0)
        jy = ch * P.Y_join
        cap_stem(gl, x=W / 2, top=jy + 20, head=False)
        gl.line((0, ch), (W / 2, jy), name="left", names=("top", "join"), terminal=("vtop_l", False), pen=thick)
        gl.line((W, ch), (W / 2, jy), name="right", names=("top", "join"), terminal=("vtop_r", False), pen=thin)

    @g("Z")
    def Z(gl):
        ch, W, t = P.capheight, P.Z_width, P.E_arm_t
        rect(gl, "top", 0, ch - t, W, ch)
        rect(gl, "bottom", 0, 0, W, t)
        hw = P.Z_thick / 2
        poly_(gl, "diag", [((0, t - 1), "bl"), ((hw * 2, t - 1), "br"), ((W, ch - t + 1), "tr"),
                           ((W - hw * 2, ch - t + 1), "tl")])
        face_wedge(gl, "serif_top", 0, ch - t + 1, ch - P.Z_serif, P.Z_wedge, P.E_serif_w, opens=+1)
        face_wedge(gl, "serif_bottom", W, t - 1, P.Z_serif, P.Z_wedge, P.E_serif_w, opens=-1)

    # ---- digits -----------------------------------------------------------
    fig_pen = Pen(angle=0, width=P.fig_stem, contrast=1 - P.hairline * P.fig_thin_k / P.fig_stem)

    def fig_bowl(x, width, bottom, top, thick_k=1.0, name="bowl"):
        return bowl(name=name, x=x, width=width, bottom=bottom, top=top, thick=P.fig_stem * P.round_gain * thick_k,
                    thin=P.hairline * P.fig_thin_k, stress=P.o_stress, full=(P.o_full_h, P.o_full_v),
                    in_full=(P.o_in_full_h, P.o_in_full_v))

    @g("zero", sb=(0, 0))
    def zero(gl):
        gl += fig_bowl(0, P.zero_w, -os_, P.fig_h, P.zero_thick_k)

    @g("one", sb=(0, 0))
    def one(gl):
        w = P.fig_stem
        gl += stem_(x=0, top=P.one_fy, foot="cap_foot", head=False,
                    pen=Pen(angle=0, width=w, contrast=1 - P.hairline / w))
        # the flag: the stem's top, cut off on a diagonal running down-left
        poly_(gl, "flag", [((-w / 2, P.one_fy - 40), "a"), ((w / 2, P.one_fy - 40), "b"), ((w / 2, P.one_top), "c"),
                           ((w / 2 - 10, P.one_top), "d"), ((-P.one_flag, P.one_fy + P.one_flag_t), "e"),
                           ((-P.one_flag, P.one_fy), "f")])

    @g("two", sb=(0, 0))
    def two(gl):
        W, h = P.two_w, P.fig_h
        top = h - P.hairline * P.fig_thin_k / 2
        (gl.path(name="hook", pen=fig_pen, profile=Profile((0, P.two_end_w), (0.15, 1), (1, 1)))
            .start((P.fig_stem / 2, P.two_end_y), name="end", dir=90, terminal="apex")
            .curve_to((W * 0.5, top), name="top", dir=0)
            .curve_to((W - P.fig_stem / 2, P.two_right_y), name="right", dir=270)
            .curve_to((W * P.two_mid_x, P.two_mid_y), name="mid", dir=P.two_mid_dir)
            .curve_to((P.two_bar_l + P.fig_stem * 0.4, P.two_bar_t * 0.5), name="foot", dir=250, terminal=False))
        rect(gl, "bar", P.two_bar_l, 0, P.two_bar_r, P.two_bar_t)
        face_wedge(gl, "serif", P.two_bar_r, P.two_bar_t - 1, P.two_serif, P.two_wedge, P.E_serif_w)

    @g("three", sb=(0, 0))
    def three(gl):
        W, h = P.three_w, P.fig_h
        top = h - P.hairline * P.fig_thin_k / 2
        bot = -os_ + P.hairline * P.fig_thin_k / 2
        mx = W * P.three_mid_x
        (gl.path(name="upper", pen=fig_pen)
            .start((P.fig_stem / 2, P.three_end_y), name="end", dir=90, terminal="apex")
            .curve_to((W * 0.5, top), name="top", dir=0)
            .curve_to((W - P.fig_stem / 2 + P.three_ur_dx, P.three_ur_y), name="right", dir=270)
            .curve_to((mx, P.three_mid_y), name="mid", dir=180, terminal=False))
        (gl.path(name="lower", pen=fig_pen)
            .start((mx, P.three_mid_y), name="mid", dir=0, terminal=False)
            .curve_to((W - P.fig_stem / 2, P.three_lr_y), name="right", dir=270)
            .curve_to((W * P.three_bot_x, bot), name="bottom", dir=180)
            .curve_to((W * P.three_end_x, P.three_end2_y), name="end", dir=P.three_end_dir, terminal="ear"))

    @g("four", sb=(0, 0))
    def four(gl):
        x, w = P.four_x, P.fig_stem
        gl += stem_(x=x, top=P.four_top, foot="cap_foot", head=False,
                    pen=Pen(angle=0, width=w, contrast=1 - P.hairline / w))
        rect(gl, "bar", 0, P.four_bar_y - P.four_bar_t / 2, P.four_bar_r, P.four_bar_y + P.four_bar_t / 2)
        thin = Pen(angle=0, width=P.four_thin, contrast=0)
        gl.line((x + w / 2 - P.four_thin * 0.7, P.four_top), (P.four_thin * 0.7, P.four_bar_y), name="diag",
                names=("top", "foot"), terminal=("apex", False), pen=thin)

    @g("five", sb=(0, 0))
    def five(gl):
        W = P.five_w
        bt = P.five_bar_t
        poly_(gl, "bar", [((P.five_top_l, P.capheight), "tl"), ((P.five_top_r, P.capheight + os_), "tr"),
                          ((P.five_top_r - 18, P.capheight - bt), "br"), ((P.five_top_l + P.five_stem_w, P.capheight - bt), "bl")])
        poly_(gl, "stem", [((P.five_top_l - 28, P.five_stem_bot), "bl"), ((P.five_top_l - 28 + P.five_stem_w, P.five_stem_bot + 20), "br"),
                           ((P.five_top_l + P.five_stem_w, P.capheight - 1), "tr"), ((P.five_top_l, P.capheight - 1), "tl")])
        top = P.five_bowl_top
        bot = -os_ * 1.5 + P.hairline * P.fig_thin_k / 2
        (gl.path(name="bowl", pen=fig_pen)
            .start((P.five_top_l - 28 + P.five_stem_w * 0.5, P.five_stem_bot + 10), name="start", dir=40, terminal=False)
            .curve_to((W * 0.55, top), name="top", dir=0)
            .curve_to((W - P.fig_stem / 2, top * 0.48), name="right", dir=270)
            .curve_to((W * 0.5, bot), name="bottom", dir=180)
            .curve_to((P.five_end_x, P.five_end_y), name="end", dir=P.five_end_dir, terminal="ear"))

    def six_like(gl, turn):
        """The 6 (and, turned 180 degrees about its centre, the 9, with its own
        values): a closed bowl below, and a hook rising from its left wall over
        to a vertical face."""
        k = (lambda n: getattr(P, ("nine_" if turn else "six_") + n))
        W, h = k("w"), P.fig_h
        cx, cy = W / 2, (h - os_) / 2
        T = (lambda q: (2 * cx - q[0], 2 * cy - q[1])) if turn else (lambda q: q)
        D = (lambda d: d + 180) if turn else (lambda d: d)
        bt, bb = k("bowl_h"), -os_
        if turn:
            bt, bb = 2 * cy - bb, 2 * cy - bt
        gl += fig_bowl(0, W, bb, bt, k("bowl_thick_k"))
        top = h - P.hairline * P.fig_thin_k / 2
        xf = W * k("hook_x")
        (gl.path(name="hook", pen=fig_pen, profile=Profile((0, 1), (0.6, 1), (1, k("hook_w"))))
            .start(T((P.fig_stem * 0.55, k("bowl_h") * 0.55)), name="root", dir=D(90), terminal=False)
            .curve_to(T((W * 0.55, top)), name="top", dir=D(0))
            .curve_to(T((xf, top - 8)), name="end", dir=D(-10), terminal=False))
        if turn:
            face_wedge(gl, "face", 2 * cx - xf, 2 * cy - (top + 6), 2 * cy - k("face_y"), k("wedge"), P.E_serif_w,
                       opens=+1)
        else:
            face_wedge(gl, "face", xf, top + 6, k("face_y"), k("wedge"), P.E_serif_w)

    @g("six", sb=(0, 0))
    def six(gl):
        six_like(gl, False)

    @g("nine", sb=(0, 0))
    def nine(gl):
        six_like(gl, True)

    @g("seven", sb=(0, 0))
    def seven(gl):
        W, ch, bt = P.seven_w, P.capheight, P.seven_bar_t
        rect(gl, "bar", 0, ch - bt, W, ch)
        poly_(gl, "diag", [((P.seven_foot_l, 0), "bl"), ((P.seven_foot_r, 0), "br"), ((W, ch - bt + 51), "tr"),
                           ((W - P.seven_top_w, ch - bt), "tl")])
        face_wedge(gl, "serif", 0, ch - bt + 1, ch - P.seven_serif, P.seven_wedge, P.E_serif_w, opens=+1)

    @g("eight", sb=(0, 0))
    def eight(gl):
        W, h, Wt = P.eight_w, P.fig_h, P.eight_top_w
        gl += fig_bowl(0, W, -os_, P.eight_waist + P.hairline, P.eight_thick_k)
        gl += fig_bowl((W - Wt) / 2, Wt, P.eight_waist - P.hairline, h, P.eight_thick_k * 0.9, name="upper")
        sp = Pen(angle=0, width=P.fig_stem * P.eight_sp_w, contrast=0)
        gl.line((W * P.eight_sp_x0, P.eight_sp_y0), (W * P.eight_sp_x1, P.eight_sp_y1), name="spine",
                names=("top", "bottom"), terminal=(False, False), pen=sp)

    # ---- accents and accented letters ------------------------------------
    # The marks are glyphs of their own (combining: zero advance; spacing:
    # with side bearings), drawn at the lowercase position; accented letters
    # are font.composite()s of a base and marks, placed by the compiler.
    def acute_(gl, flip=False):
        y0, y1, bw, tw, run = P.acc_y0, P.acc_y1, P.acute_bot_w, P.acute_top_w, P.acute_run
        pts = [(0, y0), (bw, y0), (run + tw, y1 - 4), (run + tw, y1), (run, y1)]
        if flip:                       # the grave: the acute mirrored
            pts = [(-x, y) for x, y in pts][::-1]
        poly_(gl, "mark", [(q, None) for q in pts])

    def circumflex_(gl):
        y0, y1, w, tw, lw = P.acc_y0, P.acc_y1, P.circ_w, P.circ_top_w, P.circ_leg_w
        poly_(gl, "mark", [((0, y0 + 4), "l0"), ((3, y0), "l1"), ((lw, y0), "l2"), ((w / 2, y1 - P.circ_in), "in"),
                           ((w - lw, y0), "r2"), ((w - 3, y0), "r1"), ((w, y0 + 4), "r0"),
                           ((w / 2 + tw / 2, y1), "tr"), ((w / 2 - tw / 2, y1), "tl")])

    def tilde_(gl):
        L, lo, hi, a = P.tilde_len, P.tilde_lo, P.tilde_hi, P.tilde_dir
        pen = Pen(angle=0, width=P.tilde_w, contrast=0)
        (gl.path(name="mark", pen=pen)
            .start((0, lo), name="l", dir=a, terminal=False)
            .curve_to((L * 0.35, hi), name="hump", dir=0)
            .curve_to((L * 0.65, lo + (hi - lo) * 0.2), name="trough", dir=0)
            .curve_to((L, hi + (hi - lo) * 0.3), name="r", dir=a, terminal=False))

    def dieresis_(gl):
        dot(gl, "dot_l", 0, P.diaer_y, P.diaer_r)
        dot(gl, "dot_r", P.diaer_gap, P.diaer_y, P.diaer_r)

    def ring_(gl):
        gl += bowl(name="ring", x=0, width=P.ring_w, bottom=P.ring_y0, top=P.ring_y0 + P.ring_h,
                   thick=P.ring_thick, thin=P.ring_thin)

    def cedilla_(gl):
        pen = Pen(angle=0, width=P.ced_w, contrast=1 - P.ced_thin / P.ced_w)
        L, d = P.ced_len, P.ced_drop
        (gl.path(name="mark", pen=pen, profile=Profile((0, 0.5), (0.3, 1), (1, 0.7)))
            .start((0, 15), name="root", dir=262, terminal=False)
            .curve_to((-8, P.ced_neck), name="neck", dir=0)
            .curve_to((L * 0.75, -(d + P.ced_neck) / 2 + P.ced_neck * 0.3), name="curl", dir=270)
            .curve_to((-L * 0.25, -d + P.ced_thin / 2), name="end", dir=180, terminal="ear"))

    MARKS = {"grave": lambda gl: acute_(gl, True), "acute": acute_, "circumflex": circumflex_, "tilde": tilde_,
             "dieresis": dieresis_, "ring": ring_, "cedilla": cedilla_}
    ACCENTED = {   # base: {accent: composite glyph}
        "A": "grave acute circumflex tilde dieresis ring", "E": "grave acute circumflex dieresis",
        "I": "grave acute circumflex dieresis", "N": "tilde", "O": "grave acute circumflex tilde dieresis",
        "U": "grave acute circumflex dieresis", "Y": "acute", "C": "cedilla",
        "a": "grave acute circumflex tilde dieresis ring", "e": "grave acute circumflex dieresis",
        "dotlessi": "grave acute circumflex dieresis", "n": "tilde", "o": "grave acute circumflex tilde dieresis",
        "u": "grave acute circumflex dieresis", "y": "acute dieresis", "c": "cedilla"}
    from fontTools.agl import toUnicode as _tou
    comp_names = {}
    for base_, accs in ACCENTED.items():
        for acc in accs.split():
            nm = ("i" if base_ == "dotlessi" else base_) + acc
            ch_ = _tou(nm)
            if len(ch_) == 1 and (want is None or ch_ in want) and (want is None or base_ != "dotlessi" or True):
                comp_names[nm] = (base_, acc)
    used_marks = {acc for _, acc in comp_names.values()}
    spacing_marks = {"grave": "grave", "acute": "acute", "dieresis": "dieresis", "cedilla": "cedilla", "ring": "ring"}
    for acc in MARKS:
        if acc in used_marks:
            font.glyph(acc + "comb", advance=0)(lambda gl, f=MARKS[acc]: f(gl))
        sp = spacing_marks.get(acc)
        if sp and (want is None or _tou(sp) in want):
            font.glyph(sp, sb=(0, 0))(lambda gl, f=MARKS[acc]: f(gl))
    if want is None or any(b == "dotlessi" for b, _ in comp_names.values()) or "ı" in (want or ()):
        @font.glyph("dotlessi")
        def dotlessi(gl):
            gl += stem_(x=0, top=xh + P.head_top, head="head")
    mark_dx = {"acute": P.acute_dx, "grave": P.grave_dx}
    for nm, (base_, acc) in comp_names.items():
        dx = mark_dx.get(acc, 0) + (getattr(P, "acc_dx_" + base_) if "acc_dx_" + base_ in DEFAULTS else 0)
        font.composite(nm, base_, [acc + "comb"], dx=dx)

    # ---- symbols ------------------------------------------------------------
    sym_pen = lambda w: Pen(angle=0, width=w, contrast=0)
    cy_math = P.math_y

    @g("plus", sb=(0, 0))
    def plus(gl):
        w, t = P.math_w, P.math_t
        rect(gl, "h", 0, cy_math - t / 2, w, cy_math + t / 2)
        rect(gl, "v", w / 2 - t / 2, cy_math - w / 2, w / 2 + t / 2, cy_math + w / 2)

    @g("equal", sb=(0, 0))
    def equal(gl):
        w, t, gp, cy = P.math_w, P.eq_t, P.eq_gap, cy_math + P.eq_dy
        rect(gl, "lo", 0, cy - gp / 2 - t / 2, w, cy - gp / 2 + t / 2)
        rect(gl, "hi", 0, cy + gp / 2 - t / 2, w, cy + gp / 2 + t / 2)

    def chevron(gl, flip):
        w, h, t, cy = P.lt_w, P.lt_h, P.lt_t, cy_math + P.lt_dy
        tip, a, b = ((w, cy), (0, cy + h / 2), (0, cy - h / 2)) if flip else \
                    ((0, cy), (w, cy + h / 2), (w, cy - h / 2))
        pr = Profile((0, P.lt_end_k), (1, 1))      # heavier at the point
        gl.line(a, tip, name="upper", names=("end", "tip"), terminal=(False, False), pen=sym_pen(t), profile=pr)
        gl.line(b, tip, name="lower", names=("end", "tip"), terminal=(False, False), pen=sym_pen(t), profile=pr)

    @g("less", sb=(0, 0))
    def less(gl):
        chevron(gl, False)

    @g("greater", sb=(0, 0))
    def greater(gl):
        chevron(gl, True)

    @g("multiply", sb=(0, 0))
    def multiply(gl):
        w, t = P.times_w, P.math_t
        gl.line((0, cy_math - w / 2), (w, cy_math + w / 2), name="a", names=("s", "e"), terminal=(False, False),
                pen=sym_pen(t))
        gl.line((0, cy_math + w / 2), (w, cy_math - w / 2), name="b", names=("s", "e"), terminal=(False, False),
                pen=sym_pen(t))

    @g("periodcentered", sb=(0, 0))
    def periodcentered(gl):
        dot(gl, "dot", 0, P.mid_dot_y, P.dot_r)

    @g("bullet", sb=(0, 0))
    def bullet(gl):
        dot(gl, "dot", 0, P.bullet_y, P.bullet_r)

    @g("ellipsis", sb=(0, 0))
    def ellipsis(gl):
        for k in range(3):
            dot(gl, f"dot{k}", k * P.ellipsis_gap, cy_dot, P.dot_r)

    @g("underscore", sb=(0, 0))
    def underscore(gl):
        rect(gl, "bar", 0, P.under_y - P.under_t / 2, P.under_w, P.under_y + P.under_t / 2)

    @g("bar", sb=(0, 0))
    def bar_(gl):
        rect(gl, "bar", 0, P.bar_bot, P.bar_w, P.bar_top)

    def bracket(gl, flip):
        w, st, at, top, bot = P.brk_w, P.brk_stem, P.brk_arm, P.brk_top, P.brk_bot
        x0 = w - st if flip else 0
        rect(gl, "stem", x0, bot, x0 + st, top)
        rect(gl, "top", 0, top - at, w, top)
        rect(gl, "bottom", 0, bot, w, bot + at)

    @g("bracketleft", sb=(0, 0))
    def bracketleft(gl):
        bracket(gl, False)

    @g("bracketright", sb=(0, 0))
    def bracketright(gl):
        bracket(gl, True)

    @g("backslash", sb=(0, 0))
    def backslash(gl):
        w, run = P.slash_w, P.slash_run
        poly_(gl, "bar", [((run, P.slash_bot), "bl"), ((run + w, P.slash_bot), "br"),
                          ((w, P.slash_top), "tr"), ((0, P.slash_top), "tl")])

    def brace(gl, flip):
        W, top, bot = P.brace_w, P.brk_top, P.brk_bot
        mid = (top + bot) / 2
        pen = Pen(angle=0, width=P.brace_pen, contrast=1 - P.hairline / P.brace_pen)
        sx = (lambda x: W - x) if flip else (lambda x: x)
        sd = (lambda d: 180 - d) if flip else (lambda d: d)
        xs = W * P.brace_x
        for nm, ya, yb, da in (("upper", top - P.hairline / 2, mid, 1), ("lower", bot + P.hairline / 2, mid, -1)):
            (gl.path(name=nm, pen=pen, profile=Profile((0, 0.6), (0.3, 1), (0.8, 1), (1, 0.4)))
                .start((sx(W - 10), ya), name="end", dir=sd(180), terminal=False)
                .curve_to((sx(xs), ya - (ya - yb) * 0.3), name="upper", dir=sd(270 if da > 0 else 90))
                .curve_to((sx(xs), yb + (ya - yb) * 0.15), name="lower", dir=sd(270 if da > 0 else 90))
                .curve_to((sx(10), yb), name="beak", dir=sd(180), terminal=False))

    @g("braceleft", sb=(0, 0))
    def braceleft(gl):
        brace(gl, False)

    @g("braceright", sb=(0, 0))
    def braceright(gl):
        brace(gl, True)

    @g("numbersign", sb=(0, 0))
    def numbersign(gl):
        w, sl, t, gp = P.hash_w, P.hash_slant, P.hash_t, P.hash_gap
        x0 = (w - gp) / 2 - sl / 2
        for k, nm in ((0, "v1"), (1, "v2")):
            x = x0 + k * gp
            poly_(gl, nm, [((x, 0), "bl"), ((x + t, 0), "br"), ((x + t + sl, P.capheight), "tr"),
                           ((x + sl, P.capheight), "tl")])
        bt = P.hash_bar_t
        rect(gl, "h1", 0, P.hash_y0 - bt / 2, w, P.hash_y0 + bt / 2)
        rect(gl, "h2", 0, P.hash_y1 - bt / 2, w, P.hash_y1 + bt / 2)

    @g("percent", sb=(0, 0))
    def percent(gl):
        W, rw, rh, th = P.pct_w, P.pct_ring_w, P.pct_ring_h, P.pct_thick
        ch = P.capheight
        gl += bowl(name="ring_top", x=0, width=rw, bottom=ch + os_ - rh, top=ch + os_, thick=th, thin=P.hairline)
        gl += bowl(name="ring_bottom", x=W - rw, width=rw, bottom=-os_, top=-os_ + rh, thick=th, thin=P.hairline)
        sw = P.pct_slash_w
        x0, x1 = P.pct_s0, W - P.pct_s1
        poly_(gl, "slash", [((x0, -os_), "bl"), ((x0 + sw, -os_), "br"), ((x1, ch + os_), "tr"),
                            ((x1 - sw, ch + os_), "tl")])

    @g("degree", sb=(0, 0))
    def degree(gl):
        ch, w = P.capheight, P.deg_w
        gl += bowl(name="ring", x=0, width=w, bottom=ch + os_ - w, top=ch + os_, thick=P.deg_t, thin=P.deg_t * 0.8)

    @g("dollar", sb=(0, 0))
    def dollar(gl):
        gl += spine(width=P.S_width, height=P.dollar_h, stem_w=P.cap_stem, swell_k=P.S_swell_k,
                    t1=(P.S_t1_dx, P.S_t1_dy), t2=(P.S_t2_dx, P.S_t2_dy), mid=(P.S_mid_dx, P.S_mid_dy),
                    sides=(P.S_left_dy, P.S_right_dy), tops=(P.S_top_dx, P.S_bot_dx),
                    dw=(P.S_t1_dw, P.S_top_dw, P.S_left_dw, P.S_mid_dw, P.S_right_dw, P.S_bot_dw, P.S_t2_dw),
                    y0=P.dollar_y0)
        x = P.S_width / 2 + P.dollar_bar_dx
        rect(gl, "bar", x - P.dollar_bar / 2, P.dollar_bot, x + P.dollar_bar / 2, P.dollar_top)

    @g("asterisk", sb=(0, 0))
    def asterisk(gl):
        cy, R, tw, rw = P.ast_y, P.ast_r, P.ast_tip_w, P.ast_root_w
        for k in range(6):                          # six tapered petals (Geist's: none upright)
            a = math.radians(P.ast_rot + 60 * k)
            ux, uy = math.cos(a), math.sin(a)
            px, py = -uy, ux
            tip = (ux * R, cy + uy * R)
            mid = (ux * R * 0.7, cy + uy * R * 0.7)
            poly_(gl, f"petal{k}", [((px * rw / 2, cy + py * rw / 2), "r1"), ((mid[0] + px * tw / 2, mid[1] + py * tw / 2), "m1"),
                                    ((tip[0] + px * tw * 0.3, tip[1] + py * tw * 0.3), "t1"),
                                    ((tip[0] - px * tw * 0.3, tip[1] - py * tw * 0.3), "t2"),
                                    ((mid[0] - px * tw / 2, mid[1] - py * tw / 2), "m2"), ((-px * rw / 2, cy - py * rw / 2), "r2")])

    font.slant("at", P.at_slant)       # Geist's @ is an italic form

    @g("at", sb=(0, 0))
    def at(gl):
        W, H, bot, t = P.at_w, P.at_h, P.at_bot, P.at_ring_t
        iw, ih = P.at_in_w, P.at_in_h
        ix, iy = W * P.at_in_x, P.at_in_y
        pen = Pen(angle=0, width=t * 1.6, contrast=1 - t * 0.6 / (t * 1.6))
        sx = ix + iw + P.stem / 2                   # the inner stem's centre
        gl += bowl(name="inner", x=ix, width=iw, bottom=iy, top=iy + ih, thick=P.stem, thin=P.hairline)
        gl += stem_(x=sx - P.stem / 2, bottom=iy, top=iy + ih, head=False, foot=False, name="stem")
        cy = bot + H / 2
        (gl.path(name="ring", pen=pen)
            .start((sx - P.stem / 2, iy + 20), name="root", dir=300, terminal=False)
            .curve_to((W - t, cy), name="right", dir=90)
            .curve_to((W * 0.5, bot + H - t / 2), name="top", dir=180)
            .curve_to((t, cy), name="left", dir=270)
            .curve_to((W * 0.5, bot + t / 2), name="bottom", dir=0)
            .curve_to((W * 0.85, bot + H * 0.12), name="end", dir=30, terminal=False))

    @g("ampersand", sb=(0, 0))
    def ampersand(gl):
        W, ch = P.amp_w, P.capheight
        pen = Pen(angle=0, width=P.amp_pen, contrast=1 - P.hairline / P.amp_pen)
        lx, lw = W * P.amp_loop_x, P.amp_loop_w
        top = ch + os_ - P.hairline / 2
        (gl.path(name="stroke", pen=pen, smooth=True)
            .start((W * P.amp_arm_x, P.amp_arm_y), name="arm", dir=250, terminal="apex")
            .curve_to((W * 0.42, -os_ + P.hairline), name="bottom", dir=180)
            .curve_to((P.amp_pen / 2, 180), name="left", dir=90)
            .curve_to((lx - lw * 0.1, P.amp_waist_y), name="waist", dir=45)
            .curve_to((lx + lw / 2 - P.amp_pen / 2, 620), name="loop_r", dir=90)
            .curve_to((lx, top), name="top", dir=180)
            .curve_to((lx - lw / 2 + P.amp_pen / 2, 600), name="loop_l", dir=270)
            .curve_to((W * P.amp_leg_x, 0), name="leg", dir=310, terminal="vertex"))

    @g("Euro", sb=(0, 0))
    def euro(gl):
        x0, w = P.euro_x0, P.euro_w
        gl += bowl(name="bowl", x=x0, width=w, bottom=-os_, top=P.capheight + os_,
                   thick=P.cap_stem * P.round_gain * P.O_thick_k * P.C_thick_k, thin=P.hairline * P.O_thin_k,
                   stress=P.o_stress + P.C_stress, thick_right=P.cap_stem * P.round_gain * P.C_right_k,
                   thin_bottom=P.hairline * P.C_bottom_k)
        (gl.cut("aperture").move((x0 + w * 0.6, P.C_top_y), "a").line((x0 + w * P.C_face, P.C_top_y), "f")
            .line((x0 + w * P.C_face, P.capheight + 200), "b").line((x0 + w + 140, P.capheight + 200), "c")
            .line((x0 + w + 140, P.C_term_y), "d").line((x0 + w, P.C_term_y), "tip")
            .line((x0 + w * 0.6, P.C_cut_y), "e").close())
        t = P.math_t
        rect(gl, "bar_lo", P.euro_bar_l, P.euro_y0 - t / 2, P.euro_bar_r, P.euro_y0 + t / 2)
        rect(gl, "bar_hi", P.euro_bar_l + 20, P.euro_y1 - t / 2, P.euro_bar_r + 20, P.euro_y1 + t / 2)

    @g("sterling", sb=(0, 0))
    def sterling(gl):
        W, ch = P.pound_w, P.capheight
        pen = Pen(angle=0, width=P.cap_stem, contrast=1 - P.hairline / P.cap_stem)
        (gl.path(name="hook", pen=pen, profile=Profile((0, 1), (0.7, 1), (1, 0.6)))
            .start((W * 0.3, 40), name="foot", dir=90, terminal=False)
            .curve_to((W * 0.32, ch * 0.62), name="rise", dir=85)
            .curve_to((W * 0.62, ch + os_ - P.hairline / 2), name="top", dir=0)
            .curve_to((W * 0.92, ch * 0.83), name="end", dir=-70, terminal="ear"))
        rect(gl, "cross", W * 0.08, P.pound_bar_y - P.math_t / 2, W * 0.66, P.pound_bar_y + P.math_t / 2)
        rect(gl, "base", 0, 0, W, P.E_arm_t * 1.3)

    @g("asciicircum", sb=(0, 0))
    def asciicircum(gl):
        w, h, y0, t = P.caret_w, P.caret_h, P.caret_y0, P.caret_t
        gl.line((0, y0), (w / 2, y0 + h), name="l", names=("s", "e"), terminal=(False, False), pen=sym_pen(t))
        gl.line((w, y0), (w / 2, y0 + h), name="r", names=("s", "e"), terminal=(False, False), pen=sym_pen(t))

    @g("asciitilde", sb=(0, 0))
    def asciitilde(gl):
        L, lo, hi, a = P.tilde_len, P.tilde_lo + P.atilde_dy, P.tilde_hi + P.atilde_dy, P.tilde_dir
        (gl.path(name="wave", pen=sym_pen(P.tilde_w))
            .start((0, lo), name="l", dir=a, terminal=False)
            .curve_to((L * 0.35, hi), name="hump", dir=0)
            .curve_to((L * 0.65, lo + (hi - lo) * 0.2), name="trough", dir=0)
            .curve_to((L, hi + (hi - lo) * 0.3), name="r", dir=a, terminal=False))

    # ---- punctuation ----------------------------------------------------
    def dot(gl, name, cx, cy, r):
        k = 0.5523 * r
        (gl.contour(name).move((cx, cy - r), "bottom").curve((cx + k, cy - r), (cx + r, cy - k), (cx + r, cy), "right")
            .curve((cx + r, cy + k), (cx + k, cy + r), (cx, cy + r), "top")
            .curve((cx - k, cy + r), (cx - r, cy + k), (cx - r, cy), "left")
            .curve_close((cx - r, cy - k), (cx - k, cy - r)).close())

    def comma_(gl, name, cx, cy, sc=1.0, turn=False):
        """A ball with a tail curling down-left from its right side (turned
        180 degrees about the ball's centre for the opening quote)."""
        r, k = P.dot_r * sc, 0.5523 * P.dot_r * sc
        tail = cy - (P.comma_tail + P.dot_r - os_) * sc
        tx, tw = cx + P.comma_dx * sc, P.comma_tip_w * sc
        tip_l, tip, tip_r = (tx - tw * 0.55, tail + 10 * sc), (tx, tail), (tx + tw * 0.45, tail + 45 * sc)
        ctrl = (cx + r, cy - P.comma_bend * sc)
        c1 = (tip_r[0] + (ctrl[0] - tip_r[0]) * 0.67, tip_r[1] + (ctrl[1] - tip_r[1]) * 0.67)
        c2 = (cx + r + (ctrl[0] - cx - r) * 0.67, cy + (ctrl[1] - cy) * 0.67)
        pts = [("p", tip_l, "tip_l"), ("p", tip, "tip"), ("p", tip_r, "tip_r"),
               ("c", c1, c2, (cx + r, cy), "right"),
               ("c", (cx + r, cy + k), (cx + k, cy + r), (cx, cy + r), "top"),
               ("c", (cx - k, cy + r), (cx - r, cy + k), (cx - r, cy), "left"),
               ("c", (cx - r, cy - k), (cx - k, cy - r), (cx, cy - r), "bottom")]
        T = (lambda q: (2 * cx - q[0], 2 * cy - q[1])) if turn else (lambda q: q)
        c = gl.contour(name).move(T(pts[0][1]), pts[0][2])
        for it in pts[1:]:
            c = c.line(T(it[1]), it[2]) if it[0] == "p" else c.curve(T(it[1]), T(it[2]), T(it[3]), it[4])
        c.close()                               # closes with a line back to tip_l

    def poly_(gl, name, pts):
        c = gl.contour(name).move(*pts[0])
        for q in pts[1:]:
            c = c.line(*q)
        c.close()

    def quote_wedge(gl, name, cx):
        top, w, tw = P.capheight, P.quote_w, P.quote_tip_w
        poly_(gl, name, [((cx - tw / 2, top - P.quote_h), "bl"), ((cx + tw / 2, top - P.quote_h), "br"),
                         ((cx + w / 2, top), "tr"), ((cx - w / 2, top), "tl")])

    cy_dot = P.dot_r - os_

    @g("period", sb=(0, 0))
    def period(gl):
        dot(gl, "dot", 0, cy_dot, P.dot_r)

    @g("comma", sb=(0, 0))
    def comma(gl):
        comma_(gl, "comma", 0, cy_dot)

    @g("colon", sb=(0, 0))
    def colon(gl):
        dot(gl, "dot", 0, cy_dot, P.dot_r)
        dot(gl, "dot_top", 0, xh + os_ - P.dot_r, P.dot_r)

    @g("semicolon", sb=(0, 0))
    def semicolon(gl):
        comma_(gl, "comma", P.semi_dx, cy_dot)
        dot(gl, "dot_top", 0, xh + os_ - P.dot_r, P.dot_r)

    @g("quoteright", sb=(0, 0))
    def quoteright(gl):
        r = P.dot_r * P.quote_k
        comma_(gl, "quote", 0, P.quote_top - r, P.quote_k)

    @g("quoteleft", sb=(0, 0))
    def quoteleft(gl):
        r = P.dot_r * P.quote_k
        comma_(gl, "quote", 0, P.quote_bot + r, P.quote_k, turn=True)

    @g("quotedblright", sb=(0, 0))
    def quotedblright(gl):
        r = P.dot_r * P.quote_k
        comma_(gl, "quote_l", 0, P.quote_top - r, P.quote_k)
        comma_(gl, "quote_r", P.quote_gap, P.quote_top - r, P.quote_k)

    @g("quotedblleft", sb=(0, 0))
    def quotedblleft(gl):
        r = P.dot_r * P.quote_k
        comma_(gl, "quote_l", 0, P.quote_bot + r, P.quote_k, turn=True)
        comma_(gl, "quote_r", P.quote_gap, P.quote_bot + r, P.quote_k, turn=True)

    @g("quotesingle", sb=(0, 0))
    def quotesingle(gl):
        quote_wedge(gl, "quote", 0)

    @g("quotedbl", sb=(0, 0))
    def quotedbl(gl):
        quote_wedge(gl, "quote_l", 0)
        quote_wedge(gl, "quote_r", P.quote_dbl_gap)

    def dash(gl, length):
        rect(gl, "bar", 0, P.dash_y - P.dash_t / 2, length, P.dash_y + P.dash_t / 2)

    @g("hyphen", sb=(0, 0))
    def hyphen(gl):
        dash(gl, P.hyphen_len)

    @g("endash", sb=(0, 0))
    def endash(gl):
        dash(gl, P.endash_len)

    @g("emdash", sb=(0, 0))
    def emdash(gl):
        dash(gl, P.emdash_len)

    @g("exclam", sb=(0, 0))
    def exclam(gl):
        w, tw, top = P.excl_w, P.excl_tip_w, P.capheight
        poly_(gl, "stem", [((-tw / 2, P.excl_bot), "bl"), ((tw / 2, P.excl_bot), "br"),
                           ((w / 2, P.excl_shoulder), "sr"), ((w / 2, top), "tr"), ((-w / 2, top), "tl"),
                           ((-w / 2, P.excl_shoulder), "sl")])
        dot(gl, "dot", 0, cy_dot, P.dot_r)

    @g("question", sb=(0, 0))
    def question(gl):
        W, w = P.quest_width, P.quest_w
        pen = Pen(angle=0, width=w, contrast=1 - P.quest_thin / w)
        top = P.capheight + os_ - P.quest_thin / 2
        nx = W * P.quest_neck_x
        (gl.path(name="hook", pen=pen, profile=Profile((0, 1), (0.75, 1), (1, P.quest_bot_w)))
            .start((w / 2, P.quest_end_y), name="end", dir=90, terminal="apex")
            .curve_to((W * P.quest_top_x, top), name="top", dir=0, tension=P.quest_tension)
            .curve_to((W - w / 2, P.quest_right_y), name="right", dir=270, tension=P.quest_tension)
            .curve_to((nx, P.quest_neck_y), name="neck", dir=P.quest_neck_dir)
            .curve_to((nx + P.quest_bot_dx, P.quest_bot), name="bottom", dir=270, terminal="apex"))
        dot(gl, "dot", nx + P.quest_bot_dx, cy_dot, P.dot_r)

    def paren(gl, flip):
        pen = Pen(angle=0, width=P.paren_w, contrast=0)
        mid_y = (P.paren_top + P.paren_bot) / 2
        x_end, x_mid = (0, P.paren_bow) if flip else (P.paren_bow, 0)
        a_ = P.paren_end_dir
        d_top, d_bot = (360 - a_, 180 + a_) if flip else (180 + a_, 360 - a_)
        (gl.path(name="arc", pen=pen, profile=Profile((0, P.paren_end_w), (0.5, 1), (1, P.paren_end_w)))
            .start((x_end, P.paren_top), name="top", dir=d_top, terminal=False)
            .curve_to((x_mid, mid_y), name="mid", dir=270, tension=P.paren_tension)
            .curve_to((x_end, P.paren_bot), name="bottom", dir=d_bot, terminal=False, tension=P.paren_tension))

    @g("parenleft", sb=(0, 0))
    def parenleft(gl):
        paren(gl, False)

    @g("parenright", sb=(0, 0))
    def parenright(gl):
        paren(gl, True)

    @g("slash", sb=(0, 0))
    def slash(gl):
        w, run = P.slash_w, P.slash_run
        poly_(gl, "bar", [((0, P.slash_bot), "bl"), ((w, P.slash_bot), "br"),
                          ((run + w, P.slash_top), "tr"), ((run, P.slash_top), "tl")])

    @g("space", advance=P.stem * 2.4)
    def space(gl):
        pass

