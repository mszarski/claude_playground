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
    "g_tr_dh": (0, (-0.4, 0.3)), "g_tr_dv": (0, (-0.4, 0.3)),
    "g_ear_root_dx": (0, (-60, 30)),   # the ear's sloped top starts on the bowl, left of its foot
    "g_link_root_dy": (0, (-40, 80)), "g_link_root_dir": (210, (170, 260)),   # where the link leaves the bowl
    "t_tail_cut": (0, (-30, 85)),   # normal of the cut across the t's tail end (0: vertical cut)
    "t_wedge_flat": (0, (0, 60)), "t_wedge_sweep": (0, (-20, 60)),   # the t's flat top and concave flank
    "h_dwidth": (0, (-40, 40)),   # the h's arch relative to the n's width
    "n_shoulder_dw": (0, (-30, 30)), "n_leg_t": (1.0, (0.7, 1.6)),   # arch: weight at the shoulder, turn into the leg
    "u_head_dlen": (0, (-40, 60)), "u_head_dtop": (0, (-30, 30)),   # the u's heads relative to n's
    "spur_len": (70, (0, 160)), "spur_rise": (60, (0, 140)), "spur_drop": (40, (0, 120)),
    # spur fine shape, as for the head wedge (all degenerate at their defaults)
    "spur_tip": (0, (0, 20)), "spur_flat": (0, (0, 40)), "spur_bracket": (0, (0, 1)),
    "spur_corner": (20, (0, 80)), "spur_ca": (0.7, (0.0, 1.0)), "spur_cb": (0.5, (0.0, 1.0)),
    "spur_sweep": (0, (-20, 40)),
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
    "a_bx_in": (0.5, (0.1, 0.95)), "a_lobe_t": (1.0, (0.6, 1.8)), "a_lobe_ti": (1.0, (0.6, 1.8)),
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
    "g_link_w0": (0.4, (0.05, 1.5)), "g_link_w1": (1.0, (0.2, 2.0)), "g_link_tp": (0.5, (0.1, 0.9)),
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
    "t_top": (660, (560, 740)), "t_turn": (110, (40, 200)), "t_tail": (170, (80, 260)), "t_tail_y": (60, (20, 150)),
    "t_bar_l": (90, (30, 160)), "t_bar_r": (140, (60, 220)), "t_cut": (30, (-10, 70)),
    "i_dot_y": (690, (620, 760)), "i_dot_r": (55, (30, 90)),
    "bar_t": (32, (5, 80)),
    "O_width": (700, (560, 820)), "A_width": (680, (520, 820)), "A_bar_y": (0.32, (0.2, 0.45)),
    "A_apex": (40, (0, 120)), "A_thick": (115, (60, 180)), "A_thin": (28, (8, 80)),
    "E_width": (480, (380, 600)), "E_mid": (0.86, (0.6, 1.0)), "E_arm_t": (32, (8, 80)),
    "E_serif": (90, (20, 180)), "E_serif_w": (26, (8, 60)),
    "G_width": (690, (560, 820)), "G_bar_y": (0.42, (0.3, 0.55)), "G_term_y": (0.72, (0.6, 0.9)),
    "G_serif": (110, (30, 220)),
    "S_width": (500, (380, 620)),
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
    return {"vtop_l": bracketed(left=side(1.0), right=side(P.top_serif_in)),
            "vtop_r": bracketed(left=side(P.top_serif_in), right=side(1.0))}


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
        top=bracketed(length=P.top_serif_len, bracket=P.top_serif_bracket, tip=P.top_serif_tip,
                      curve=(P.serif_ca, P.serif_cb)),
        **_vtops(P, P.top_serif_tip),
        # s has its own beaks (sharing a's made fitting a silently change s); both
        # rise above the stroke -- the top one past its upper edge, the bottom one
        # up the vertical face of the lower terminal
        s_top=beak(side="above", length=P.s_beak_len, depth=P.s_beak_depth, axis=P.s_cut1),
        s_bottom=beak(side="above", length=P.s_beak_len, depth=P.s_beak_depth, axis=P.s_cut2),
        hood=beak(side="left", length=P.beak_len, depth=P.beak_depth, axis="vertical"),
        ear=flat("horizontal"), vertex=flat("vertical"), apex=flat("vertical"),
        cap_foot_l=bracketed(length=P.cap_serif_len, bracket=P.cap_serif_bracket, tip=P.cap_serif_tip,
                             curve=(P.serif_ca, P.serif_cb), right=None),
        cap_vfoot=bracketed(length=P.cap_serif_len, bracket=P.cap_serif_bracket * 0.7, tip=P.cap_serif_tip,
                            curve=(P.serif_ca, P.serif_cb)),
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
        cap_foot_l=bracketed(length=P.cap_serif_len, bracket=P.cap_serif_bracket, tip=P.foot_tip,
                             curve=(P.serif_ca, P.serif_cb), right=None),
        head=head,
        u_head=bracketed(left=dict(length=lazy_max(P.head_len + P.u_head_dlen, 0), bracket=P.serif_bracket, tip=P.foot_tip),
                         right=None, curve=(P.serif_ca, P.serif_cb)),
        spur=bracketed(right=dict(length=P.spur_len * 0.6, bracket=P.serif_bracket * 0.6, tip=P.foot_tip),
                                  left=None, curve=(P.serif_ca, P.serif_cb)),
        top=bracketed(length=P.top_serif_len, bracket=P.top_serif_bracket, tip=P.foot_tip,
                      curve=(P.serif_ca, P.serif_cb)),
        **_vtops(P, P.foot_tip),
        cap_vfoot=bracketed(length=P.cap_serif_len, bracket=P.cap_serif_bracket * 0.7, tip=P.foot_tip,
                            curve=(P.serif_ca, P.serif_cb)),
        # s/S: vertical faces with hairline beards (top one hangs down, bottom one stands up)
        s_top=beak(side="below", length=P.s_beak_len, depth=P.s_beak_depth, axis=P.s_cut1),
        s_bottom=beak(side="above", length=P.s_beak_len, depth=P.s_beak_depth, axis=P.s_cut2),
        hood=ball(radius=P.ball_r, side="left", drop=P.ball_drop),
        r_ear=ball(radius=P.ball_r * 0.9, side="below", drop=P.ball_drop),
        f_hook=ball(radius=P.ball_r * 0.9, side="below", drop=P.ball_drop),
        ear=ball(radius=P.ball_r * 0.7, side="above"), vertex=flat("vertical"), apex=flat("vertical"),
        t_tail=flat(0), t_top=flat(P.t_cut + 90),
    )


# --------------------------------------------------------------------------

def define(font, P, glyphs: str | None = None, terminals=None, g_ear: str = "stroke", g_band: str = "none",
           b_style: str = "bowl"):
    """Register the kit's glyphs on ``font``. ``g_ear``: ``"stroke"`` (a
    short stroke taking the ``ear`` terminal, e.g. a ball) or ``"flag"`` (a
    four-sided flag). ``g_band``: ``"flat"`` puts a flat bar across the top of
    the g's lower loop (and trims the loop above it) -- a tilted loop with a
    flat top, which one contour bowl can't make. ``b_style``: ``"bowl"`` (a
    free bowl overlapping the stem) or ``"lobe"`` (a lobe hanging on the
    stem at two joins, like the a's bowl mirrored)."""
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
            ch = name if len(name) == 1 else (" " if name == "space" else None)
            if want is None or (ch and ch in want) or name in (want or ()):
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
    def arch(p, x, width, top=xh, flip=False, edge_half=None):
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
            .start(st, name="join", dir=d0, terminal=False)
            .curve_to((x + width * P.n_shoulder_x, sh_y), name="shoulder", dir=0, tension=P.n_tension,
                      width=P.n_shoulder_dw)
            .curve_to((leg_x + P.stem / 2 - left_half, top * P.n_out_y * 0.5), name="leg", dir=-90,
                      tension=P.n_leg_t))
        notch_cut(p, J, val(edge), +1)
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
            .start(st, name="join", dir=d0, terminal=False)
            .curve_to((R - width * P.u_shoulder_x, sh_y), name="shoulder", dir=180, tension=P.u_tension)
            .curve_to((x - P.stem / 2 + left_half, top - top * P.u_out_y * 0.5), name="leg", dir=90,
                      tension=P.u_leg_t))       # >1 stops the curve swinging out past the stem
        notch_cut(p, J, val(edge), -1)

    @part
    def spine(p, width, height, stem_w):
        """The s/S spine: one stroke, width peaking mid-spine."""
        W = width
        sp = Pen(angle=P.s_angle, width=stem_w, contrast=1 - P.hairline / stem_w)
        top = height + os_ - sp.width_at(180) / 2
        bot = -os_ + sp.width_at(180) / 2
        sw = Profile((0, P.s_ends), (0.5 - P.s_spread, P.s_ends), (0.5, P.s_swell), (0.5 + P.s_spread, P.s_ends),
                     (1, P.s_ends))
        (p.path(name="spine", pen=sp, profile=sw, bias=P.s_bias)
            .start((W * P.s_t1_x, height * P.s_t1_y), name="t1", dir=P.s_t1_dir, terminal="s_top")
            .curve_to((W * P.s_top_x, top), name="top", dir=180, tension=P.s_tension, width=P.s_top_dw)
            .curve_to((sp.width_at(90) / 2, height * P.s_left_y), name="left", dir=270, width=P.s_left_dw)
            .curve_to((W * P.s_mid_x, height * P.s_mid_y), name="mid", dir=P.s_mid_dir, width=P.s_mid_dw)
            .curve_to((W - sp.width_at(90) / 2, height * P.s_right_y), name="right", dir=270, width=P.s_right_dw)
            .curve_to((W * P.s_bot_x, bot), name="bottom", dir=180, tension=P.s_tension, width=P.s_bot_dw)
            .curve_to((W * P.s_t2_x, height * P.s_t2_y), name="t2", dir=P.s_t2_dir, terminal="s_bottom"))

    def tapered_bar(gl, x0, y0, x1, y1, stem_l):
        """A crossbar whose left end is a point: a pentagon, degenerate (a plain
        rectangle) at the defaults; always five points (principle 4)."""
        xa = x0 + (stem_l - x0) * P.bar_taper
        (gl.contour("bar").move((x0, y0), "bl").line((x1, y0), "br").line((x1, y1), "tr").line((xa, y1), "tl")
            .line((x0 - P.bar_tip_dx, y0 + (y1 - y0) * P.bar_tip_y), "tip").close())

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
        gl += arch(x=0, width=P.n_width)
        gl += stem_(x=P.n_width, top=xh * P.n_out_y, name="leg", head=False)

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
        gl += bowl(x=0, width=w, bottom=-os_, top=xh + os_, thick=th, thin=P.hairline,
                   stress=P.e_stress, full=(P.e_full_h, P.e_full_v), in_full=(P.e_in_full_h, P.e_in_full_v),
                   thin_bottom=P.hairline * P.e_thin_bottom)
        y = xh * P.e_bar_y
        t = P.e_bar_t
        gl.contour("bar").move((th * 0.5, y - t / 2), "bl").line((w, y - t / 2), "br") \
            .line((w, y + t / 2), "tr").line((th * 0.5, y + t / 2), "tl").close()
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
        gl += stem_(x=sx, top=stem_top, foot="spur", head=False)
        hood_pen = Pen(angle=P.a_hood_angle, width=P.stem, contrast=1 - P.hairline * P.a_hood_thin / P.stem)
        top_y = xh + os_ - hood_pen.width_at(0) * (1 - val(P.a_bias))
        (gl.path(name="hood", pen=hood_pen, bias=P.a_bias,
                 profile=Profile((0, P.a_taper), (P.a_hood_end, 1), (1, 1)))
            .start((P.a_term_x, P.a_term_y), name="terminal", dir=P.a_term_dir, terminal="hood")
            .curve_to((P.a_term_x + (sx - P.a_term_x) * P.a_top_x, top_y), name="top", dir=0)
            # always end inside the stem, so hood and stem overlap
            .curve_to((sx, min(val(P.a_stem_top + (xh - P.a_stem_top) * (1 - P.a_out_y) * 0.5),
                               val(stem_top) - 6)), name="out", dir=-90))
        # the bowl hangs on the stem's left edge at two joins
        gl.lobe(sx - P.stem / 2, -1, P.a_far, -os_,
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
        gl += arch(x=0, width=w, edge_half=P.stem * P.h_stem_w / 2)
        gl += stem_(x=w, top=xh * P.n_out_y, name="leg", head=False)

    @g("m")
    def m(gl):
        w = P.m_width
        gl += stem_(x=0, top=xh + P.head_top, head="head")
        gl += arch(x=0, width=w, name="arch1")
        gl += stem_(x=w, top=xh * P.n_out_y, name="leg1", head=False)
        gl += arch(x=w, width=w, name="arch2")
        gl += stem_(x=2 * w, top=xh * P.n_out_y, name="leg2", head=False)

    @g("u")
    def u(gl):
        w = P.u_width
        top = xh + P.head_top + P.u_head_dtop
        gl += stem_(x=0, bottom=xh * (1 - P.u_out_y), top=top, head="u_head", foot=False, name="left")
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

    @g("r")
    def r(gl):
        gl += stem_(x=0, top=xh + P.head_top, head="head", foot="foot_wide")
        J = join(val(xh) * P.r_join_at, P.r_join_thin, P.r_join_angle)
        _, _, st, d0, thin = contact(J, val(P.stem / 2), +1, rising=True)
        join_handles(J, val(P.stem / 2), +1, rising=True)
        (gl.path(name="ear", pen=arch_pen, bias=P.n_bias,
                 profile=Profile((0, thin / max(arch_pen.width_at(d0), 1.0)), (P.r_taper_len, 1), (1, P.r_end_w)))
            .start(st, name="join", dir=d0, terminal=False)
            .curve_to((P.r_width, xh * P.r_y), name="end", dir=P.r_dir, terminal="r_ear"))

    @g("f")
    def f(gl):
        top = P.f_top
        hook_pen = Pen(angle=P.arch_angle, width=P.stem, contrast=P.arch_contrast)
        y0 = xh + (top - xh) * 0.35
        gl += stem_(x=0, top=y0, head=False, foot="foot_wide")
        (gl.path(name="hook", pen=hook_pen, bias=P.n_bias, profile=Profile((0, 1), (0.6, 1), (1, P.f_end_w)))
            .start((0, y0 - 2), name="root", dir=90, terminal=False)
            .curve_to((P.f_hook * 0.45, top + os_ - hook_pen.width_at(0) / 2), name="top", dir=0,
                      width=P.f_top_dw)
            .curve_to((P.f_hook, top * P.f_end_y), name="end", dir=P.f_end_dir, terminal="f_hook"))
        tapered_bar(gl, -P.f_bar_l, xh - P.f_bar_t + P.f_bar_dy, P.f_bar_r, xh + P.f_bar_dy, -P.stem / 2)

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
            .curve_to((P.t_tail * 0.55, -os_ + t_pen.width_at(0) / 2), name="bottom", dir=0)
            .curve_to((P.t_tail, P.t_tail_y), name="end", dir=60, terminal="t_tail"))
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
        gl += bowl(x=0, width=P.O_width, bottom=-os_, top=ch + os_, thick=P.cap_stem * P.round_gain,
                   thin=P.hairline, stress=P.o_stress, full=(P.o_full_h, P.o_full_v),
                   in_full=(P.o_in_full_h, P.o_in_full_v))

    @g("A", sb=(P.A_sb, P.A_sb))
    def A(gl):
        ch, W = P.capheight, P.A_width
        thick = Pen(angle=0, width=P.A_thick, contrast=0)
        thin = Pen(angle=0, width=P.A_thin, contrast=0)
        ax = W / 2
        gl.line((ax - P.A_apex * 0.5, ch + os_), (0, 0), name="left", names=("apex", "foot"),
                terminal=("apex", "cap_vfoot"), pen=thin)
        gl.line((ax + P.A_apex * 0.5 - P.A_thick * 0.3, ch + os_), (W, 0), name="right", names=("apex", "foot"),
                terminal=("apex", "cap_vfoot"), pen=thick)
        y = ch * P.A_bar_y
        k = (ch + os_) / (W / 2)
        rect(gl, "bar", y / k, y - P.bar_t / 2, W - y / k, y + P.bar_t / 2)

    @g("E")
    def E(gl):
        ch, W = P.capheight, P.E_width
        gl += stem_(x=0, top=ch, pen=cap_pen, foot="cap_foot_l", head="cap_foot_l")
        t, sw, sh = P.E_arm_t, P.E_serif_w, P.E_serif
        rect(gl, "arm_top", 0, ch - t, W, ch)
        rect(gl, "serif_top", W - sw, ch - sh, W, ch)
        rect(gl, "arm_bottom", 0, 0, W * 1.04, t)
        rect(gl, "serif_bottom", W * 1.04 - sw, 0, W * 1.04, sh * 1.1)
        y = ch * 0.5
        rect(gl, "arm_mid", 0, y - t / 2, W * P.E_mid, y + t / 2)
        rect(gl, "serif_mid", W * P.E_mid - sw, y - sh * 0.45, W * P.E_mid, y + sh * 0.45)

    @g("G")
    def G(gl):
        ch, W = P.capheight, P.G_width
        th = P.cap_stem * P.round_gain
        gl += bowl(x=0, width=W, bottom=-os_, top=ch + os_, thick=th, thin=P.hairline,
                   stress=P.o_stress, full=(P.o_full_h, P.o_full_v), in_full=(P.o_in_full_h, P.o_in_full_v))
        yb, yt = ch * P.G_bar_y, ch * P.G_term_y
        R = W + 100
        ry, cy = (ch + 2 * os_) / 2, ch / 2
        q = max(0.0, 1 - ((val(yt) - val(cy)) / val(ry)) ** 2)
        ex = W / 2 + (W / 2) * q ** 0.42          # outer edge of the bowl at the terminal
        bh = P.G_serif * 0.6                      # beard length
        # aperture steps around the beard so the beard survives the cut
        (gl.cut("aperture").move((W * 0.5, yb), "a").line((R, yb), "b").line((R, yt - bh), "c")
            .line((ex - P.E_serif_w, yt - bh), "d").line((ex - P.E_serif_w, yt), "e").line((W * 0.5, yt), "f").close())
        sx = W - P.cap_stem * 0.5
        gl += stem_(x=sx, top=yb, pen=cap_pen, foot="spur", head=False, name="spur")
        rect(gl, "bar", sx - P.cap_stem / 2 - P.G_serif, yb - P.bar_t, sx + P.cap_stem / 2, yb)
        rect(gl, "beard", ex - P.E_serif_w, yt - bh, ex, yt + 2)

    @g("S")
    def S(gl):
        gl += spine(width=P.S_width, height=P.capheight, stem_w=P.cap_stem)

    @g("space", advance=P.stem * 2.4)
    def space(gl):
        pass

