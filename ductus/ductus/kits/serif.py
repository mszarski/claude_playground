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
              dw=(0, 0, 0, 0, 0, 0, 0)):
        """The s/S spine: one stroke, width peaking mid-spine. ``t1``/``t2``,
        ``mid`` (x, y), ``sides`` (left y, right y) and ``tops`` (top x,
        bottom x) offset the s's nodes: the S's own proportions. ``dw`` adds to
        the s's width offsets at t1, top, left, mid, right, bottom, t2."""
        W = width
        sp = Pen(angle=P.s_angle, width=stem_w, contrast=1 - P.hairline / stem_w)
        top = height + os_ - sp.width_at(180) / 2
        bot = -os_ + sp.width_at(180) / 2
        e_, m_ = P.s_ends * swell_k, P.s_swell * swell_k
        sw = Profile((0, e_), (0.5 - P.s_spread, e_), (0.5, m_), (0.5 + P.s_spread, e_), (1, e_))
        (p.path(name="spine", pen=sp, profile=sw, bias=P.s_bias, smooth=val(P.s_mid_free) >= 0.5)
            .start((W * (P.s_t1_x + t1[0]), height * (P.s_t1_y + t1[1])), name="t1", pen_angle=P.s_t1_pa, dir=P.s_t1_dir, terminal="s_top", width=P.s_t1_dw + dw[0])
            .curve_to((W * (P.s_top_x + tops[0]), top), name="top", pen_angle=P.s_top_pa, dir=P.s_top_dir, tension=P.s_tension, width=P.s_top_dw + dw[1])
            .curve_to((sp.width_at(90) / 2, height * (P.s_left_y + sides[0])), name="left", pen_angle=P.s_left_pa, dir=P.s_left_dir,
                      tension=P.s_left_t, width=P.s_left_dw + dw[2])
            # s_mid_free: the spine's middle direction is solved (Hobby's mock-
            # curvature continuity) instead of imposed -- an imposed direction
            # makes both halves curl to meet it, and high contrast turns that
            # dip in direction into a kink in the width
            .curve_to((W * (P.s_mid_x + mid[0]), height * (P.s_mid_y + mid[1])), name="mid", pen_angle=P.s_mid_pa,
                      dir=None if val(P.s_mid_free) >= 0.5 else P.s_mid_dir, tension=P.s_mid_t,
                      width=P.s_mid_dw + dw[3])
            .curve_to((W - sp.width_at(90) / 2, height * (P.s_right_y + sides[1])), name="right", pen_angle=P.s_right_pa, dir=P.s_right_dir,
                      tension=P.s_right_t, width=P.s_right_dw + dw[4])
            .curve_to((W * (P.s_bot_x + tops[1]), bot), name="bottom", pen_angle=P.s_bot_pa, dir=P.s_bot_dir, tension=P.s_tension + P.s_bot_dt,
                      width=P.s_bot_dw + dw[5])
            .curve_to((W * (P.s_t2_x + t2[0]), height * (P.s_t2_y + t2[1])), name="t2", dir=P.s_t2_dir, terminal="s_bottom",
                      width=P.s_t2_dw + dw[6], pen_angle=P.s_t2_pa))

    def tapered_bar(gl, x0, y0, x1, y1, stem_l):
        """A crossbar whose left end is a point: a pentagon, degenerate (a plain
        rectangle) at the defaults; always five points (principle 4)."""
        xa = x0 + (stem_l - x0) * P.bar_taper
        (gl.contour("bar").move((x0, y0), "bl").line((x1, y0), "br").line((x1, y1), "tr").line((xa, y1), "tl")
            .line((x0 - P.bar_tip_dx, y0 + (y1 - y0) * P.bar_tip_y), "tip").close())

    def cap_bowl(width, thick_k=1.0, thin_k=1.0, dfull=(0, 0, 0, 0), stress=0):
        """The O's bowl (also the G's, with its own offsets)."""
        ch = P.capheight
        return bowl(x=0, width=width, bottom=-os_, top=ch + os_, thick=P.cap_stem * P.round_gain * P.O_thick_k * thick_k,
                    thin=P.hairline * P.O_thin_k * thin_k, stress=P.o_stress + stress,
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
        gl.line((mx, xh), (v1, -P.v_os), name="a2", names=("top", "vertex"), terminal=("apex", "vertex"), pen=thin)
        gl.line((mx + P.w_mid3_dx, xh), (v2, -P.v_os), name="a3", names=("top", "vertex"), terminal=("apex", "vertex"), pen=thick)
        gl.line((W, xh), (v2, -P.v_os), name="a4", names=("top", "vertex"), terminal=("vtop_r", "vertex"), pen=thin)

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

