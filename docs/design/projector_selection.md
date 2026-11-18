# Projector selection

The owner: "Can you help with a projector selection". This document says what the projector and its amplifier
must do, where each requirement comes from, what kinds of transducer do it, which products could be checked,
and which coherent choices exist. It ends with a decision table, the questions for a manufacturer and the
build-spec fields each choice would set.

**What is and is not established here.**

- The requirements are computed from the build files by `tools/projector_requirements.py` (new), which calls the
  repository's own link budget (`gpusonar.buildspec.performance`) and direct-sound arithmetic
  (`buildspec.describe`). Every table marked *tool* below is that tool's output, pasted by a script, not retyped.
  Full output for all six builds: `docs/design/projector_requirements.json`.
- **No manufacturer data sheet could be opened in this session.** Every manufacturer site tried was refused by
  the session's network policy (Sec. 6 lists the hosts and what was wanted from each). The product section
  therefore contains: a few figures that open reports and theses quote, each with its source; and a "to check"
  list with **no numbers**. No product figure in this document comes from memory.
- No projector is chosen. The recommendation is a decision table.

**Convention.** Source level is **RMS**, dB re 1 µPa at 1 m (the owner's decision on finding P-5 of
`docs/HARDCODED_AUDIT.md`). Three consequences:
(1) the link budget is an energy equation and takes an rms level as it stands, so the SNR tables of
`docs/DESIGN.md` hold; (2) the overload tables of `docs/DESIGN.md` and `direct_sound_study.json` treat the same
number as a *peak* pressure, so with an rms specification **every direct-sound overload figure is 3 dB higher**:
on the full face +10.6 to +22.1 dB, not +7.6 to +19.1 dB; (3) cavitation is set by peak pressure, rms + 3 dB.
Both conventions are printed wherever it matters.

## 0. Summary

| Requirement | Value for the 12 km point (`popoto_face_3x32`) | Origin |
| --- | --- | --- |
| Source level | 197 dB re 1 µPa at 1 m rms; each dB is a dB of SNR. 194 dB still detects the ship at 12 km with 2 or 3 rows (+1.0 / +2.4 dB of margin), 190 dB does not | link budget, Sec. 1.1 |
| Band | 4–6 kHz, 40% of the centre; response within 3 dB across it (mechanical Q of 2.4 or less), or voltage headroom to equalise | Sec. 1.2 |
| Pulse, duty | 1 s every 17.9 s, 5.6%; 201 pings an hour | build file |
| Acoustic power | 410 W omnidirectional (417 W with the rounded constant 170.8 the design text uses); less by the directivity index | Sec. 1.3 |
| Electrical | 0.6–1.4 kW into the transducer at 70–30% efficiency, about 1 kW from the supply at 50%; 964 J a ping, 54 Wh an hour; 23 W of average heat in the transducer | Sec. 1.3 |
| Drive voltage | 10^((197 − TVR) / 20) V rms: 708 V at a TVR of 140 dB, 1259 V at 135 dB | Sec. 1.7 |
| Cavitation | 197 dB needs 0.13 m² of wet radiating area to be free at the surface; a 0.03 m² radiator (a 10 cm sphere) is free only below 10 m, 19 m if half its face does the work | Sec. 1.4 |
| Depth rating | a parameter. The design's own surface-image formula puts the best own depth for a target at 4 m at 12 km at 225 m: the rating should then be 300 m or more | Sec. 1.5 |
| Direct sound | +22.1 dB over the inputs' full scale at the nearest hydrophone (rms convention), +10.6 dB at the farthest; a response 22 dB down towards the array (28 dB with margin) would remove it; full scale is reached 21.5 m away | Sec. 1.6 |
| Prototype (test AUV, 0.1 s pulse) | 170 dB reaches the ship-sized target at 2.0 km and the small one at 0.7 km; 180 dB: 3.4 / 1.2 km. Less than 10 W acoustic | Sec. 1.9 |

The four options (Sec. 4): **A** one source at 197 dB; **B** a smaller source at 191–194 dB with a 2–4 s pulse;
**C** a short line of small projectors, broadside, with its null along the face; **D** a small source at
170–185 dB for the test AUV now, the full-scale choice deferred. D is compatible with each of A, B and C and
costs none of them anything; the direct-sound and input-recovery measurements that decide between A, B and C
can only be made once D exists.

## 1. Requirements

### 1.1 Source level: what it buys

Output SNR = SL − 2 TL + TS − NL + array gain + 10 log₁₀ T (`docs/DESIGN.md` Sec. 5): spherical spreading,
Ainslie-McColm absorption (12 °C, 50 m), ambient noise at 5 m/s wind (51.5 dB re 1 µPa²/Hz at 5 kHz), 13 dB at
the map for detection, ship +25 dB, small target +5 dB. Noise-limited: reverberation and the channel are not in
it, and `docs/DESIGN.md` ("The 12 km design point") shows the water moves the answer by tens of dB while a row
moves it by 2–4 dB. Long rows (32 at 0.15 m), 1 s pulse. *Tool:*

| Rows | Source level (dB) | Ship at 6 km (dB) | Ship at 12 km (dB) | margin at 12 km (dB) | Small at 6 km (dB) | Small at 12 km (dB) | Reach ship / small (km) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 190 | 23.6 | 7.6 | -5.4 | 3.6 | -12.4 | 9.6 / 3.8 |
| 1 | 194 | 27.6 | 11.6 | -1.4 | 7.6 | -8.4 | 11.4 / 4.6 |
| 1 | 197 | 30.6 | 14.6 | +1.6 | 10.6 | -5.4 | 12.8 / 5.3 |
| 1 | 200 | 33.6 | 17.6 | +4.6 | 13.6 | -2.4 | 14.4 / 6.2 |
| 1 | 203 | 36.6 | 20.6 | +7.6 | 16.6 | 0.6 | 16.0 / 7.1 |
| 2 | 190 | 26.0 | 10.0 | -3.0 | 6.0 | -10.0 | 10.6 / 4.3 |
| 2 | 194 | 30.0 | 14.0 | +1.0 | 10.0 | -6.0 | 12.5 / 5.2 |
| 2 | 197 | 33.0 | 17.0 | +4.0 | 13.0 | -3.0 | 14.0 / 6.0 |
| 2 | 200 | 36.0 | 20.0 | +7.0 | 16.0 | 0.0 | 15.7 / 6.9 |
| 2 | 203 | 39.0 | 23.0 | +10.0 | 19.0 | 3.0 | 17.4 / 7.9 |
| 3 | 190 | 27.3 | 11.4 | -1.6 | 7.3 | -8.6 | 11.2 / 4.6 |
| 3 | 194 | 31.3 | 15.4 | +2.4 | 11.3 | -4.6 | 13.2 / 5.5 |
| 3 | 197 | 34.3 | 18.4 | +5.4 | 14.3 | -1.6 | 14.7 / 6.4 |
| 3 | 200 | 37.3 | 21.4 | +8.4 | 17.3 | 1.4 | 16.4 / 7.3 |
| 3 | 203 | 40.3 | 24.4 | +11.4 | 20.3 | 4.4 | 18.3 / 8.4 |

Reading it:

- **dB for dB.** Between 6 and 12 km a dB of source level is worth about 0.5 km of reach on the ship-sized
  target and 0.3 km on the small one.
- **197 dB** gives +1.6 / +4.0 / +5.4 dB of margin at 12 km with 1 / 2 / 3 rows on the ship. **194 dB** leaves
  −1.4 / +1.0 / +2.4 dB: with two or three rows the 12 km point still closes on this budget, with almost no
  margin. **190 dB** does not close it (−1.6 dB with three rows); it reaches 11.2 km.
- **200 and 203 dB** buy margin on the ship (+8.4, +11.4 dB with three rows) but do not bring the small target to
  12 km (1.4 and 4.4 dB against 13 needed); they bring it to 7.3 and 8.4 km. `docs/DESIGN.md` gives 212 dB for
  that, which Sec. 1.4 shows is a large radiator run deep.
- Margin matters more than the table suggests: the budget omits the surface image (−17 dB each way for a
  target at 4 m seen from 10 m at 12 km, before bottom bounces return energy; Sec. 1.5) and the time spreading
  of the echo. An NPS study of a shallow site at 3.5 kHz found the loss of peak level to time spreading
  "appreciable (> 6 dB one way)" once pulses were shorter than 1 s [N9]; this argues for pulses of 1 s or more
  and against buying range with a short, loud pulse.

### 1.2 Band, bandwidth and flatness

4–6 kHz is 40% of the centre frequency. A single mechanical resonance gives a −3 dB band of f₀ / Q, so the
whole band inside 3 dB needs **Q ≤ 2.4**; most compact resonant projectors are sharper than that, and the usual
routes to a wide band are two coupled resonances (the ring and cavity modes of a free-flooded ring; the
flexural and breathing modes of a slotted cylinder) or an element that is simply run off resonance at a lower
response. What a peaked response costs when the amplifier holds the voltage constant across the sweep, and what
equalising it costs in voltage (the amplifier must then be rated for the higher voltage, and the transducer
must stand it at the band edges). *Tool:*

| Response at the band edges | Mechanical Q | Energy lost over the sweep | Voltage to equalise |
| --- | --- | --- | --- |
| −1 dB | 1.2 | 0.3 dB | ×1.12 |
| −3 dB | 2.4 | 1.0 dB | ×1.41 |
| −6 dB | 4.2 | 2.2 dB | ×2.00 |
| −10 dB | 7.3 | 3.8 dB | ×3.16 |

A response 3 dB down at the edges costs 1.0 dB of pulse energy and acts as a mild taper on the sweep (the file's
`window_alpha: 0.1` already tapers it); 6 dB costs 2.2 dB. The matched filter must use the *transmitted*
spectrum: record the pulse on a reference hydrophone and use it as the replica (the repository takes a sampled
replica: `transmit.waveform.type: sampled`). Requirement for the data sheet: **the TVR curve from 3.5 to 6.5 kHz,
not the value at resonance**, and the voltage and power limits at 4, 5 and 6 kHz separately. A source level
quoted "at resonance" is the best point of the curve.

A narrower sweep is the fallback if no projector covers the band: range resolution is c / 2B (0.375 m at 2 kHz,
0.75 m at 1 kHz), and the noise-limited budget does not depend on B at all (it depends on energy). A 1 kHz band
(20%) needs Q ≤ 5.

### 1.3 Power, energy and heat

Acoustic power for an rms source level: SL = 10 log₁₀(ρc P / 4π) + 120 + DI, which is 170.8 + 10 log₁₀ P + DI
for round sea-water values (the tool uses ρ = 1025 kg/m³, c from the file: 410 W at 197 dB where the design text
says 417 W). Electrical power into the transducer is P / η; from the supply P / (η η_amp). Efficiency is a
data-sheet item; reports opened here quote 90% at resonance and 65% at the −6 dB points for one depth-compensated
ring-shell projector at 610 Hz [N2], which shows both how good a resonant projector can be and that efficiency
falls away from resonance: across a 40% band the mean will be well under the peak. The plan's "1–2 kW amplifier
and a capacitor bank" is consistent with 197 dB at 30–50%. *Tool* (omnidirectional; amplifier 85% efficient,
an assumption; pulse and interval of the full face):

| Source level (dB) | Acoustic (W) | Into the transducer at 30 / 50 / 70% (W) | From the supply at 50% (W) | Energy a ping at 50% (J) | Average at 50% (W = Wh per hour) | Heat in the transducer, average at 50% (W) |
| --- | --- | --- | --- | --- | --- | --- |
| 190 | 82 | 272 / 163 / 117 | 192 | 192 | 10.7 | 4.6 |
| 194 | 205 | 684 / 411 / 293 | 483 | 483 | 26.9 | 11.4 |
| 197 | 410 | 1365 / 819 / 585 | 964 | 964 | 53.7 | 22.8 |
| 200 | 817 | 2724 / 1635 / 1168 | 1923 | 1923 | 107.2 | 45.6 |
| 203 | 1631 | 5436 / 3262 / 2330 | 3837 | 3837 | 214.0 | 90.9 |

- **Battery.** At 197 dB and 50%: 964 J a ping, 201 pings an hour, **54 Wh an hour** (54 W average) for the
  transmit chain. At 30% it is 90 Wh an hour. The amplifier's quiescent draw between pings is not in this and can
  exceed it: ask for the standby power and whether the output stage can be shut down between pings.
- **Peak current.** About 1 kW for 1 s: 20 A from a 48 V bus. A battery that allows it needs no capacitor bank;
  a bank that carries the whole pulse with the bus sagging from 48 to 40 V would be 2.7 F (2 × 964 J / (48² − 40²) V²).
- **Heat.** (1 − η) of the electrical power stays in the transducer: 410 W during the pulse at 50%, 23 W on
  average at this duty. Whether that is acceptable depends on the transducer's rated duty at full power, which
  data sheets state as a duty cycle and a maximum pulse length; one free-flooded ring quoted in an open paper is
  rated "300 W with 2% duty cycle" [A3]: **5.6% at full power for 1 s is not to be assumed** and is an RFQ item.
  A longer pulse at lower level (Sec. 1.8) raises the duty towards 10–40% at proportionally lower power: the
  average heat is the same or less, the peak is lower.
- **Directivity.** A projector with a directivity index of DI dB in the direction of the targets needs DI dB
  less power for the same source level (3 dB halves every number in the table).

### 1.4 Cavitation

A 197 dB source near the surface is a real question. The relation used:

    p_c(h)  = p_atm + ρ g h − p_vapour                 peak acoustic pressure at which cavitation begins, Pa
    I_c(h)  = γ p_c² / (2 ρ c)                         W/m²   (0.32 W/cm² at the surface, γ = 1)
    P_c     = I_c A                                    W radiated; A the wet radiating area
    SL_c    = 10 log₁₀(ρ c P_c / 4π) + 120 + DI        dB re 1 µPa at 1 m, rms

For a uniformly pulsating sphere of radius a this is SL_c = 20 log₁₀(p_c a / √2): the pressure at the surface of
the sphere may not exceed the static pressure.

**Where it comes from.** It is the statement that water at these frequencies cannot be pulled below about zero
absolute pressure: cavitation begins when the peak of the sound pressure at the radiating face reaches the
static pressure there. Open sources read in this session: a 2023 study of the acoustic cavitation threshold
[A1] reproduces Urick's chart of threshold against frequency and reports "the average value of 1 atm" at low
frequency (its own measurements at 0.48–1.43 kHz agree: tension of 0.99 bar relative to atmospheric, i.e. the
vapour pressure, separates cavitating from non-cavitating cases), rising with frequency (2.5–6.5 atm at 25 kHz,
3.5–4.6 atm at 60 kHz in the measurements it cites), with the hydrostatic term ρgh added to the threshold. An NPS
report on tracking-range transducers [N6] applies the same reasoning: it compares average radiated intensity
(W/cm²) with "data on cavitation thresholds for various frequencies, pulse lengths and hydrostatic pressures"
from Urick and concludes that cavitation "could be a limiting factor ... particularly at shallow depths".
A sonar-waveform dissertation [A2] states the system consequence: "Too high a source level induces cavitation
on the head of the sonar's transducers. These physical constraints are typically countered by transmitting a
long duration waveform at a lower source level". The textbook form of the rule (Urick, *Principles of Underwater
Sound*, 3rd ed., the cavitation-limit section; about 0.3 W/cm² at the surface, rising with the square of the
absolute pressure in atmospheres, times a near-field factor) is the same relation with rounded constants;
**the book itself and the report that restates it (DTIC AD0743199) could not be opened** (Sec. 6), so the
constant used here is the one computed above, not one quoted from them.

**Uncertainty.** Treat the limits as ±3 dB, and the cautious side as the real one:

- *The threshold.* 1 atm at the surface is an average over measurements with a wide spread [A1]; dissolved gas
  and bubbles near the surface lower it, clean deep water raises it. At 5 kHz some tensile strength above the
  low-frequency value may exist; none is counted. Short pulses raise the threshold; a 1 s pulse is long and
  gets no credit.
- *The radiating face.* γ ≤ 1 is the ratio of the mean intensity over the face to the largest. A real
  transducer's near field is not uniform (edges, the ends of a ring, the nodes of a flexing shell): the table
  gives γ = 0.5 (3 dB) as a second column. Area means *wet, moving* area: both faces of a free-flooded ring
  count, a housing does not.
- *What happens beyond it.* Onset is not a cliff: output stops rising linearly with drive, the pulse distorts
  (the replica no longer matches) and the face erodes over time. Running at the limit costs waveform fidelity
  before it costs level.

*Tool* (full face, 197 dB):

| Wet radiating area | sphere of that area | at 5 m | at 10 m | at 30 m | at 100 m | at 200 m | least depth for 197 dB | the same with half the face doing the work |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.01 m² | 0.06 m dia. | 189.5 dB | 192.0 dB | 198.1 dB | 206.9 dB | 212.5 dB | 25 m | 40 m |
| 0.03 m² | 0.10 m dia. | 194.3 dB | 196.8 dB | 202.8 dB | 211.6 dB | 217.3 dB | 10 m | 19 m |
| 0.1 m² | 0.18 m dia. | 199.5 dB | 202.0 dB | 208.1 dB | 216.9 dB | 222.5 dB | 1 m | 6 m |
| 0.3 m² | 0.31 m dia. | 204.3 dB | 206.8 dB | 212.8 dB | 221.6 dB | 227.3 dB | 0 m | 0 m |

Threshold: 0.73 W/cm² at 5 m, 1.30 W/cm² at 10 m, 5.24 W/cm² at 30 m, 39.72 W/cm² at 100 m, 144.88 W/cm² at 200 m. 197 dB is free of cavitation at the surface from 0.127 m² of wet radiating area (a sphere 0.20 m across).

Reading it:

- **197 dB from a compact source (0.03 m², a 10 cm sphere) is at the cavitation limit at 10 m depth and 2.7 dB
  over it at 5 m.** It needs either area (0.13 m², a 20 cm sphere or a ring of comparable wet area) or depth
  (10–20 m), or both.
- Depth helps fast: +8.5 dB from 5 m to 30 m and +17 dB to 100 m. A vehicle
  that runs at 100–200 m for the propagation reasons of Sec. 1.5 has 15–20 dB of cavitation margin with any of
  these sizes.
- **The static frame at 10 m and the test AUV in a 30 m trial site are the cavitation-limited cases**, not the
  deep-running vehicle. A full-level test of a compact projector hung at 5 m from a quay would cavitate.
- An array of N small projectors has N times the area for the same total power: each element of a four-element
  line runs at 185 dB (Sec. 1.6), which a 0.01 m² element holds from the surface down (189.5 dB limit at 5 m).

### 1.5 Operating depth and depth rating

The owner has not stated an operating depth: it is a parameter. What the design's own logic implies: a
ship-sized target is near the surface (4 m in the scenario), and `docs/DESIGN.md` ("A target near the surface is
the hard case for straight rays") gives the surface-image factor on each leg as 20 log₁₀(2 sin(2π z₁ z₂ / λR)).
It is largest when z₁ = λR / 4z₂: **225 m of own depth for a target at 4 m at 12 km**; at the 10 m the scenario
uses it is −17 dB each way. *Tool:*

| Own depth (m) | 5 | 10 | 30 | 100 | 200 |
| --- | --- | --- | --- | --- | --- |
| surface image, one way (dB) | -23.1 | -17.1 | -7.6 | +2.2 | +5.9 |
| absolute pressure (bar) | 1.5 | 2.0 | 4.0 | 11.1 | 21.1 |
| cavitation limit, 0.03 m² (dB) | 194.3 | 196.8 | 202.8 | 211.6 | 217.3 |

This is the first estimate the task statement mentions (straight rays, isovelocity; another study is refining it
with profiles and bottom bounces, which put energy back at shallow own depth). For the projector it means:

- **Depth rating: at least 1.3 times the deepest planned depth**, so 300 m if the vehicle is to use 225 m; the
  test AUV and the static frame need only tens of metres.
- **Depth dependence of the response matters as much as the rating.** Reports opened here: a free-flooded ring
  keeps "the same hydrostatic pressure inside and outside" and one model was measured "quite stable with depth"
  in a hyperbaric tank campaign to 440 bar [A3]; a manufacturer's figure for a 2.5 kHz free-flooded ring quoted in
  an NPS thesis is operational depth "unlimited" [N3]. In contrast "the operating depth of the ring-shell
  projector is limited to a few meters without a depth compensating system", in common with other
  flextensionals, and compensation (a water bladder, gas, compliant tubes) is what gives one such projector
  "exceeds 400 m" [N1, N2]. A miniature flextensional measured at NPS lost more than 30 dB of sensitivity with
  static pressure before compensation was added [N4]. So: **ask for TVR and maximum drive at the operating
  depths, not only at the test tank's depth.**
- Cavitation margin grows with depth (Sec. 1.4), so the electrical and thermal limits, not cavitation, set the
  level at depth.

### 1.6 Directivity, placement and the direct sound

From `docs/design/direct_sound_study.json` and `buildspec.describe`: the inputs' full scale is 173 dB re 1 µPa
(peak) at the hydrophone with the files' sensitivity (−208 dB re 1 V/µPa, a placeholder) and gain (43 dB). *Tool*
(full face, projector at the placeholder [4.0, 0, 0] m in the forward end margin):

|  | level taken as a peak (the files' tables) | level taken as rms (the specification) |
| --- | --- | --- |
| over full scale at the nearest | +19.1 dB | +22.1 dB |
| over full scale at the farthest | +7.6 dB | +10.6 dB |
| distance at which it is at full scale | 15.3 m | 21.5 m |

Reduction towards the array for full scale at the nearest hydrophone: 22.1 dB (28.1 dB with 6 dB of margin); the rms source level that just reaches full scale there: 174.9 dB. Against distance (rms level): 0.3 m: +37 dB, 0.5 m: +33 dB, 1 m: +27 dB, 2 m: +21 dB, 4 m: +15 dB, 8 m: +9 dB, 16 m: +3 dB, 32 m: -3 dB.

The same for the other builds (their placeholders are closer): test AUV +34.0 dB (peak convention) / +37.0 dB
(rms) at 0.30 m; static frame +35.3 / +38.3 dB at 0.26 m.

What a projector choice can do about it:

- **Level.** dB for dB. 174.9 dB rms just reaches full scale at the face's placeholder position; 160 dB on the
  test AUV. This is the strongest argument for a low-level prototype source (option D).
- **Distance.** 6 dB per doubling; full scale at 21.5 m. No position on a 9 m face clears it at 197 dB. On the
  static frame the projector can be hung apart: 16 m away leaves +3 dB, 32 m is clear (the direct sound then
  ends 11–21 ms later, and the bistatic geometry must be in the file: `transmit.lever_arm` carries it).
- **A null or reduced response towards the array.** 22 dB towards the nearest hydrophone to reach full scale,
  28 dB for 6 dB of margin. Because the hydrophones span 4.6 m and sit 1.7–6.4 m from the projector, they
  subtend a wide angle and are in the near field of any projector array: a far-field null of 30 dB does not
  give 30 dB there. Three ways to get directivity at a 30 cm wavelength, (a) to (c):

(a) *A broadside line of small projectors along the face* (in phase, half a wavelength apart). Its far-field
pattern has its maximum out from the face, towards the targets, and its minimum along the face, towards
the array. Computed exactly (spherical waves from each element to each hydrophone, across 4–6 kHz, worst
hydrophone at the worst frequency of the sweep; same far-field source level as one projector). *Tool:*

| Line | each element (dB) | directivity index (dB) | acoustic power (W) | half beam width at −3 dB | worst hydrophone, worst frequency (dB over full scale, rms) | change from one projector (dB) |
| --- | --- | --- | --- | --- | --- | --- |
| 2 along | 191.0 | 3.0 | 205 | 30° | +13.7 | -8.4 |
| 3 along | 187.5 | 4.8 | 137 | 18° | +12.7 | -9.4 |
| 4 along | 185.0 | 6.0 | 102 | 13° | +10.5 | -11.7 |
| 2 vertical | 191.0 | 3.0 | 205 | 30° | +22.0 | -0.1 |
| 3 vertical | 187.5 | 4.8 | 137 | 18° | +21.8 | -0.3 |
| 4 vertical | 185.0 | 6.0 | 102 | 13° | +21.6 | -0.6 |

Two elements along the face take 8.4 dB off the worst case, four take 11.7 dB: half of what is needed, not
all of it, because the null is exact only at the centre frequency and only along the line. The price is
the transmit sector: ±30° at −3 dB for two elements, ±13° for four, where one projector covers the whole
side. Steering the transmit beam by phase moves the null off the array. A *vertical* line does nothing for
the direct sound (−0.1 to −0.6 dB) but gives 3–6 dB of directivity index (half to a quarter of the power)
and puts less sound into the surface and the bottom, which is what limits detection in shallow water.

(b) *A multimode (cardioid) cylinder.* One ceramic cylinder driven in its omni, dipole and quadrupole modes
together synthesises a cardioid or sharper pattern, steerable electrically; an NPS thesis evaluated such a
prototype for 15–20 kHz (the requirement study behind it works with a directivity index of 9 dB) [N5]. At 4–6 kHz the cylinder is three to
four times larger; whether a product exists there is a "to check" item (Sec. 3.2). A cardioid's rear null
pointed at the array is the single-transducer form of what is wanted.

(c) *A baffle or the hull.* A pressure-release or mass baffle between projector and array, or mounting the
projector on the other side of the vehicle from the array. At a 30 cm wavelength a baffle must be of the
order of a metre across to shadow by more than a few dB; nothing was found to quantify it, and it is a
measurement item, not a design value.

- **Depth-bias in shallow water.** `docs/DESIGN.md` records that the projector's place moved the multipath bias
  on depth (below the array's centre read closer to the truth than beside it, in one geometry). A position
  below or above the rows also puts the array broadside to a *vertical* projector line, where that line gives
  no reduction. The two aims (null towards the array; projector on the array's vertical axis) conflict; the
  elevation bench should be rerun for the position each option implies.
- **None of this removes the need for the front-end measures** of `docs/DESIGN.md` (protection, gain reduction
  or a known recovery time, blanking, sync bursts clear of the transmit). The projector choice can supply 8–12 dB
  of the 22 dB (option C), or 3–6 dB (option B), or all of it at prototype level (option D).

### 1.7 Electrical drive: amplifier voltage, power, tuning, cable

For any transducer with a transmitting voltage response TVR(f) (dB re 1 µPa/V at 1 m) and impedance
Z(f) = |Z| e^{jφ}:

    V_rms = 10^((SL − TVR) / 20)          volts rms at the transducer terminals (peak: × √2)
    S     = V_rms² / |Z|                  volt-amperes the amplifier must supply
    P     = S cos φ = V_rms² G            real power (G the conductance); acoustic power = η P
    check: SL = 170.8 + 10 log₁₀(η P) + DI must give the same level; if not, the TVR and the efficiency disagree

*Tool* (`drive()` in the tool takes TVR, |Z| and φ; the table is voltage only because no impedance is known):

| TVR (dB re 1 µPa/V at 1 m) | V rms | V peak |
| --- | --- | --- |
| 130 | 2239 | 3166 |
| 135 | 1259 | 1780 |
| 140 | 708 | 1001 |
| 145 | 398 | 563 |
| 150 | 224 | 317 |

- **Voltage is set at the worst point of the band**, where the TVR is lowest: a transducer with 140 dB at
  resonance and 134 dB at the band edge needs 1.4 kV rms there for a flat 197 dB, or delivers 191 dB there at
  constant voltage. The transducer's maximum voltage (ceramic field, insulation, cable and connector rating)
  and its maximum power are separate limits; a data sheet's "maximum source level" is whichever is reached first
  and usually only at resonance.
- **Reactive power.** A piezoelectric projector is a capacitor with a motional branch; away from resonance φ
  approaches −90° and S greatly exceeds P. A tuning inductor (series or parallel, L = 1 / ((2πf₀)² C₀)) cancels the
  clamped capacitance at one frequency; across a 40% band a single inductor leaves a power factor well below 1
  at the edges. The amplifier is sized in VA at the worst frequency, not in watts. Ask for the admittance
  (G and B) across the band, in water, at depth.
- **Matching.** A transformer brings the amplifier's output voltage (tens to a couple of hundred volts for a
  class-D stage on a battery bus) up to the hundreds or thousands the ceramic needs; it is normally part of the
  amplifier or supplied with the transducer and tuned to it. The custom line array of Sec. 3.1 sends about
  200 V AC down its cable and has one tuning inductor per element in the wet end [N8].
- **Cable.** Rated for the peak voltage with margin (2 kV class at a TVR near 135 dB), with wet-mateable
  connectors of the same rating. Its capacitance (of the order of 100 pF/m) adds to C₀ and belongs inside the
  tuning; at 10 m it is small against a power projector's own capacitance. Keep the drive cable away from the
  hydrophone harness: `docs/DESIGN.md` already lists electrical coupling from the amplifier into the inputs as
  unmodelled and possibly larger than the water path.
- **Waveform fidelity.** A class-D stage switching at hundreds of kHz puts ripple into the cable; its harmonics
  must stay clear of the 38–48 kHz synchronisation burst and of the ADC's alias bands (102.4 kHz sampling).
  A transmit-current or voltage monitor output is worth asking for: it gives the replica and the trigger instant
  the build needs ("a trigger whose instant is in the board stream", `popoto_parts.md`).

### 1.8 The trade that reduces all of the above: lower level, longer or coded pulse

Detection in noise depends on pulse energy, so each halving of power can be bought back with a doubling of the
pulse: processing gain is 10 log₁₀(BT). *Tool* (full face; constant energy; transducer 50% efficient):

| Pulse (s) | Source level (dB) | 10 log BT (dB) | Blind inside (m) | Ping interval (s) | Duty | Nearest hydrophone over full scale (dB, rms) | Acoustic (W) | From the supply (W) | Average (W) | Cavitation-free below (m), 0.03 m² | Linear sweep tolerates (m/s) | Doppler channels for ±10 m/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 207.0 | 23.0 | 75 | 17.0 | 0.6% | +32.1 | 4096 | 9638 | 57 | 55 | ±3.75 | 7 |
| 0.5 | 200.0 | 30.0 | 375 | 17.4 | 2.9% | +25.1 | 819 | 1928 | 55 | 19 | ±0.75 | 29 |
| 1 | 197.0 | 33.0 | 750 | 17.9 | 5.6% | +22.1 | 410 | 964 | 54 | 10 | ±0.38 | 55 |
| 2 | 194.0 | 36.0 | 1500 | 18.9 | 10.6% | +19.1 | 205 | 482 | 51 | 5 | ±0.19 | 109 |
| 4 | 191.0 | 39.0 | 3000 | 20.9 | 19.1% | +16.1 | 102 | 241 | 46 | 0 | ±0.09 | 215 |
| 10 | 187.0 | 43.0 | 7500 | 26.9 | 37.1% | +12.1 | 41 | 96 | 36 | 0 | ±0.04 | 535 |

What moves with the pulse length:

- **Peak power, voltage, cavitation**: all down dB for dB. At 191 dB (4 s) a 0.03 m² source is free of
  cavitation at the surface, the amplifier is a 250 W unit and the drive voltage is half that of 197 dB.
- **Direct sound**: down dB for dB, but 187 dB (10 s) is still +12 dB over full scale at the nearest
  hydrophone. Reaching full scale by this route alone needs 174.9 dB, a 162 s pulse: it is not a way out by
  itself (as `docs/DESIGN.md` says), it shrinks the gain reduction needed.
- **Blind range**: c T / 2, 750 m per second of pulse, **as long as the receiver cannot listen while it
  transmits**. With these inputs it cannot: during the transmit the direct sound is 100 dB and more above a 12 km
  echo (170–193 dB re 1 µPa at the hydrophones for source levels of 187–197 dB, against about 52 dB for the ship's echo, SL − 2 TL + TS), which no
  fixed-gain 24-bit input holds even where it does not clip.
- **Ping interval and update rate**: the window still takes 16.9 s after the pulse, so 2 s gives a ping every
  18.9 s, 10 s every 26.9 s.
- **Doppler.** A linear sweep of 2 kHz at 5 kHz is not Doppler tolerant when long. An echo from a target closing
  at v is compressed by 1 + 2v/c; the phase error against the replica at the ends of the pulse is π B T v / c,
  and the estimate used in the table (error under π/2: v ≤ c / 2BT) gives ±0.38 m/s for 1 s, ±0.19 m/s for 2 s,
  ±0.04 m/s for 10 s. The vehicle's own 1.5 m/s resolved along the beam already exceeds the 1 s figure off
  broadside. This is an estimate of where loss begins (about 1 dB), not a hard limit, and the remedy is in the
  repository: a bank of replicas (`gpusonar.processing.doppler_bank`), at 55 channels for ±10 m/s at 1 s and
  proportionally more for longer pulses, or a Doppler-tolerant sweep: the hyperbolic FM that the sampled-waveform
  path provides (`transmit.waveform.source: hfm`); the dissertation read here calls the HFM "optimally Doppler
  tolerant" [A2]. Own-ship Doppler is known from navigation and can be removed per beam first.
- **Coded and continuous schemes.** The repository's sampled waveforms include phase codes (`barker`, `mseq`,
  `code`) and stepped FM. A code spreads the same energy with a thumbtack ambiguity (range and speed together,
  but Doppler-sensitive: it needs the bank). Continuous active sonar transmits a long sweep (the dissertation
  cites 20 s sweeps of 500–1000 Hz) and splits the echo into sub-bands to revisit the scene every T/P seconds
  at B/P of bandwidth [A2]. It would bring the level down to about 185 dB for the same energy per 18 s, but it
  requires receiving during the transmit, i.e. a projector tens of metres from the array or 30 dB and more of
  suppression towards it plus an input that stays linear: **for the static frame with the projector hung apart it
  is a real option; for a projector on the same 9 m face it is not**, with these boards.
- **Time spreading** favours long pulses in shallow water [N9] (Sec. 1.1).

So the useful part of the trade is the middle: **2–4 s at 194–191 dB**: half to a quarter of the power, voltage
down by 1.4–2, cavitation-free near the surface with a compact source, 3–6 dB less overload to recover from,
at the price of a blind range of 1.5–3 km (inside which a second, short pulse can be interleaved if the vehicle
needs the near field) and two to four times the Doppler channels.

### 1.9 The prototype: test AUV and static frame, 0.1 s pulse

The short row (32 at 0.09 m) with the 0.1 s pulse of the short-range builds. *Tool*
(`tools/projector_requirements.py popoto_testauv_1x32 --levels 160 ... --ranges 500 1000 2000`):

| Rows | Source level (dB) | Ship at 0.5 km (dB) | Ship at 1 km (dB) | Ship at 2 km (dB) | margin at 2 km (dB) | Small at 0.5 km (dB) | Small at 1 km (dB) | Small at 2 km (dB) | Reach ship / small (km) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 160 | 28.1 | 15.8 | 3.1 | -9.9 | 8.1 | -4.2 | -16.9 | 1.2 / 0.4 |
| 1 | 165 | 33.1 | 20.8 | 8.1 | -4.9 | 13.1 | 0.8 | -11.9 | 1.5 / 0.5 |
| 1 | 170 | 38.1 | 25.8 | 13.1 | +0.1 | 18.1 | 5.8 | -6.9 | 2.0 / 0.7 |
| 1 | 175 | 43.1 | 30.8 | 18.1 | +5.1 | 23.1 | 10.8 | -1.9 | 2.6 / 0.9 |
| 1 | 180 | 48.1 | 35.8 | 23.1 | +10.1 | 28.1 | 15.8 | 3.1 | 3.4 / 1.2 |
| 1 | 185 | 53.1 | 40.8 | 28.1 | +15.1 | 33.1 | 20.8 | 8.1 | 4.4 / 1.5 |
| 1 | 190 | 58.1 | 45.8 | 33.1 | +20.1 | 38.1 | 25.8 | 13.1 | 5.6 / 2.0 |
| 1 | 197 | 65.1 | 52.8 | 40.1 | +27.1 | 45.1 | 32.8 | 20.1 | 7.7 / 2.9 |

| Source level (dB) | Acoustic (W) | Into the transducer at 30 / 50 / 70% (W) | From the supply at 50% (W) | Energy a ping at 50% (J) | Average at 50% (W = Wh per hour) | Heat in the transducer, average at 50% (W) |
| --- | --- | --- | --- | --- | --- | --- |
| 160 | 0 | 0 / 0 / 0 | 0 | 0 | 0.0 | 0.0 |
| 165 | 0 | 1 / 1 / 0 | 1 | 0 | 0.0 | 0.0 |
| 170 | 1 | 3 / 2 / 1 | 2 | 0 | 0.1 | 0.0 |
| 175 | 3 | 9 / 5 / 4 | 6 | 1 | 0.2 | 0.1 |
| 180 | 8 | 27 / 16 / 12 | 19 | 2 | 0.7 | 0.3 |
| 185 | 26 | 86 / 52 / 37 | 61 | 6 | 2.2 | 0.9 |
| 190 | 82 | 272 / 163 / 117 | 192 | 19 | 7.0 | 3.0 |
| 197 | 410 | 1365 / 819 / 585 | 964 | 96 | 34.8 | 14.8 |

- **170–180 dB is a complete short-range trial**: the ship-sized target to 2.0–3.4 km, a +5 dB target to
  0.7–1.2 km, on 1–10 W of acoustic power. 185 dB reaches 4.4 / 1.5 km.
- **Direct sound at the prototype level**: the overload falls dB for dB from +37.0 dB (rms) at 197 dB and 0.30 m:
  +15 dB at 175 dB, 0 dB at 160 dB. Moving the projector from 0.30 m to the end of a 3 m hull (about 1.7 m from
  the nearest hydrophone) is another 15 dB. **A 175 dB source 1.7 m from the row is at the inputs' full scale**:
  the prototype can be the first build in which the direct sound is a measurement and not a problem.
- **Cavitation**: none at these levels at any depth (175 dB from a 0.01 m² element is 14 dB under the 5 m limit).
- **What it does not do**: it does not exercise the full-level amplifier, the thermal duty, cavitation, or the
  recovery of the inputs from +20 dB overloads. The last can be done on the bench electrically.
- The static frame's 2 × 32 and 3 × 32 with the same small source reach the same order of range (the long rows
  gain 2.2 dB over the short one, the stacking 2.4–3.7 dB).

### 1.10 Size, mass, mounting

No data-sheet values are in hand (Sec. 6), so these are requirements and scaling, not product figures.

- **Size follows wavelength** (30 cm at 5 kHz). The one free-flooded ring with dimensions in a report opened
  here resonates at 2.5 kHz (ring mode) with a 0.367 m outside diameter and 0.17 m ceramic height [N3]; the ring
  resonance scales inversely with diameter, so a 5 kHz ring is of the order of **0.18 m across and 0.1 m tall**
  (a scaling estimate, not a product). The custom 3.8 kHz line array of Sec. 3.1 uses flexural discs on 0.152 m
  centres [N8].
- **Mass in water** matters more than mass in air on an AUV (trim and ballast): ask for both. Free-flooded
  designs have no air volume, so their weight in water is close to their weight in air less the ceramic's
  displacement; air-backed and compensated designs are lighter in water but depth-limited or depth-dependent.
- **Mounting.** The radiating surfaces need free water: a ring needs both ends and its bore open and clear of
  structure by a fraction of a wavelength, or its cavity resonance (which supplies the lower half of its band)
  shifts; the NPS ring model found the mounting plate itself "will influence the acoustic cavity mode" [N3].
  Isolation from the hull (compliant mounts) limits structure-borne sound into the hydrophone mounts.
  A fairing must be acoustically transparent at 4–6 kHz.
- **Where it can go.** Along the face in the 1 m end margins (the placeholder), below or above the rows, or on a
  pole or the far side of the hull: Sec. 1.6 for what each does to the direct sound and the depth bias.

## 2. Technology survey for 4–6 kHz at 185–200 dB

What could be supported from documents opened in this session is cited; where a statement is general
engineering knowledge that no opened document states, it is marked *(not verified here)* and should be confirmed
against a transducer text (Sherman and Butler, *Transducers and Arrays for Underwater Sound*; Stansfield,
*Underwater Electroacoustic Transducers*; neither is open) or by the manufacturers.

| Family | How it radiates | Band and Q | Level and efficiency | Depth | Directivity | Verified here |
| --- | --- | --- | --- | --- | --- | --- |
| **Free-flooded ring** (piezoceramic ring, water inside and out) | ring breathing mode plus the resonance of the water column in the bore; the two together give a wide band | two resonances an octave or so apart (1.0 kHz cavity and 2.5 kHz ring in the one quoted [N3]); broad | 132 dB re 1 µPa/V at the ring resonance for the 0.37 m ring [N3]; a 4.4 cm ring: 133 dB re 1 µPa/V at 30 kHz, 300 W at 2% duty [A3] | pressure-balanced: "unlimited" [N3]; measured stable to 440 bar [A3] | omnidirectional around the axis; in the plane containing the axis the beam depends on the ring's height (about 60° for the small ring of [A3]) | yes: [N3], [A3]; the plan names this family for stage 2 |
| **Flextensional, classes I–VII** (ceramic stack or ring driving a flexing shell: barrel-stave I–III, oval shell IV, ring-shell V–VI, "dog-bone" VII) | small extensional motion of the driver becomes large flexural motion of the shell: "large volume velocities ... at low frequencies for the size of the device" [N4] | a ring-shell at 610 Hz: 470–750 Hz at −3 dB (Q about 4) [N2]; "wide bandwidth and low mechanical quality factor" [N1] | that unit: 213 dB re 1 µPa at 1 m at 3000 V rms, 90% efficient at resonance, 65% at −6 dB [N2] | "limited to a few meters without a depth compensating system"; with a water bladder "exceeds 400 m" [N1, N2]; uncompensated miniature units lose tens of dB with pressure [N4] | omnidirectional when small against the wavelength | yes for the class V figures at 0.6 kHz [N1, N2]; nothing opened gives a 4–6 kHz flextensional's figures. Most of the family is built for 0.3–3 kHz *(not verified here)* |
| **Flexural (bender) disc** | a ceramic-metal disc bending like a drumhead; air- or oil-backed | resonant; the 15-element array below is tuned to 3.8 kHz and used over 3.5–3.9 kHz [N8] | measured 181–191 dB re 1 µPa at 1 m per element at 3.8 kHz, 210 dB for 15 in line [N8] | air-backed discs are depth-limited unless compensated *(not verified here)*; the array's rating is not stated in [N8] | each disc near-omnidirectional; used in arrays | yes: [N8] |
| **Tonpilz (piston) and arrays of them** | ceramic stack between a head mass that radiates and a tail mass | Q of 3–7 typical, a 20–40% band with matching *(not verified here)* | the standard element of hull sonars at 3–10 kHz; an array of pistons is directional by construction *(not verified here)* | usually pressure-tolerant with a compliant or fluid backing *(not verified here)* | a single piston is omnidirectional when small; a baffled array has a rear rejection that suits the direct-sound problem | no opened document; to check with manufacturers |
| **Slotted cylinder** | a ceramic-lined tube with a slit, bending like a tuning fork | couples a flexural mode with a breathing mode for a broad band (patent abstract seen only as a search result) | compact for its frequency *(not verified here)* | air-filled: depth-limited unless compensated *(not verified here)* | omnidirectional | no |
| **Spherical and cylindrical ceramic shells** (air-backed sphere, capped cylinder) | breathing mode | resonance set by diameter: a sphere for 5 kHz is large; used below resonance as broadband, lower-level sources and calibration standards | lower TVR below resonance; the reference projectors met in the reports are of this kind [A3, N5] | limited by the shell's collapse depth | omnidirectional | partly: used as lab standards in [A3], [N5] |
| **Multimode cylinder** | one cylinder driven in omni, dipole and quadrupole modes together | as its modes allow; the prototype was for 15–20 kHz [N5] | the requirement study behind it works with a directivity index of 9 dB [N5] | — | **cardioid or sharper, steerable electrically**: the only single-transducer way found to a null towards the array | yes, as a 15–20 kHz prototype [N5] |
| **Moving-coil underwater loudspeaker** | a coil-driven piston, pressure-compensated | very broad (hundreds of Hz to tens of kHz) *(not verified here)* | low: suited to the prototype's 160–180 dB, not to 197 dB *(not verified here)* | shallow to moderate *(not verified here)* | omnidirectional at these frequencies | no; the plan's "underwater speaker" for stage 1 |

What the survey supports:

1. **For a vehicle that changes depth, the free-flooded ring is the family whose response does not.** It is
   also the one with a naturally wide band from two resonances, and is what the owner's plan names. Its weakness
   is duty and heat at high power (the one rating seen is 2% [A3]) and, for a small ring near the surface,
   cavitation (Sec. 1.4).
2. **Flextensionals give the most level per volume and the best efficiency at resonance, at a cost in depth
   behaviour** (compensation) and usually in band; those documented here are low-frequency units.
3. **Arrays of small flexural discs or pistons reach levels well beyond 197 dB at these frequencies**: 210 dB
   from fifteen 181–191 dB elements at 3.8 kHz is a measured, fielded, hand-deployed system [N8]. An array also
   brings directivity, redundancy and cavitation margin.
4. **Directivity towards the targets and away from the array is available in one transducer only as a multimode
   cylinder**, verified here only at 15–20 kHz.

## 3. Candidate products

### 3.1 Figures from documents opened in this session

**Strict rule applied: every number below was read in the cited document in this session.** None of these is a
manufacturer's data sheet; each is a report or thesis quoting or measuring the device. The source level
"computed across 4–6 kHz at rated drive" asked for cannot be given for any of them: no TVR curve across 4–6 kHz
was readable (the two that exist are figures in scanned reports).

| Device | What it is | Figures and where they are stated | Convention | Against the requirement |
| --- | --- | --- | --- | --- |
| Custom 15-element line array (Applied Physical Sciences, built for a marine-mammal exposure study) | 15 flexural-disc transducers on 6-inch (0.152 m) centres on a wire-rope harness, each with its own amplifier channel and tuning inductor; drive "e.g., 200VAC" up the riser cable; steerable in elevation | resonance 3.8 kHz; signals 3.5–3.9 kHz, 1.4 s long, one every 25 s; per-element level at 1 m 181.1–191.1 dB (table of 15, median 187.5 dB); combined 210.3 dB measured, 210.5 dB calculated; 205 dB used for beam patterns, steered ±15° with 204.9–206.1 dB; "easily deployed and retrieved by hand", run from a ship's AC supply [N8, pp. 38–40 and Sec. 2] | "dB re: 1 µPa (RMS)" stated for the transmitted source level; "@ 1m" | **not a catalogue product and 5–25% below the band** (resonant at 3.8 kHz, no figures at 4–6 kHz). It shows that 197 dB needs only **3–4 such elements in line** (187.5 dB + 20 log₁₀ 3 = 197.0; 4 give 199.5), and 185–191 dB **one or two**: the existence proof for option C. Depth rating, mass and efficiency are not stated |
| Depth-compensated ring-shell projector (Sparton of Canada; class V flextensional) | piezoceramic-and-steel ring between two spherical caps, water-bladder compensated | 610 Hz; 213 dB re 1 µPa at 1 m "driven by 3000 Vrms"; efficiency 90% at resonance, 65% at −6 dB; band >140 Hz (470–750 Hz at −3 dB); depth "exceeds 400 m" [N1 Table 3.2; N2] | re 1 µPa at 1 m; rms drive voltage stated, level's rms/peak not stated | **out of band by a factor of 8**: listed for what a flextensional's efficiency, depth and drive voltage look like, not as a candidate |
| Free-flooded ring, 0.367 m (maker not named in the thesis) | segmented ceramic ring, fibreglass-wrapped, steel mounting plate | resonances 1.0 kHz (cavity) and 2.5 kHz (ring); 132 dB re 1 µPa/V at 1 m at ring resonance; operational depth "unlimited"; ring 14.44 in (0.3668 m) outside diameter, staves 6.8 in (0.173 m) high [N3] | re 1 µPa/V at 1 m; rms/peak not stated | **below the band by a factor of 2**; a geometrically similar ring for 5 kHz would be about 0.18 m across (scaling, Sec. 1.10). At 132 dB re 1 µPa/V, 197 dB would need 1.8 kV rms: the reason larger rings are driven through a transformer |
| Sensor Technology SX30 free-flooded ring | small ring, over-moulded | 20–40 kHz; 4.4 cm outside, 2 cm inside diameter, 2.5 cm high; 133 dB re 1 µPa/V at 1 m at 30 kHz; −193 dB re 1 V/µPa; "maximum input power is 300 W with 2% duty cycle"; tested stable to 440 bar [A3] | re 1 µPa/V at 1 m | **out of band by a factor of 5**: listed for the duty rating and the depth behaviour of the family. The same maker's larger rings are a "to check" item |
| USRD G34 (US Navy reference projector, rented) | "7 tuned ceramic transducers ... compressed side-by-side in a bronze housing" | "advertised frequency range of 200 Hz – 5 kHz"; usable range stretched to 1–8 kHz in that experiment; TVR "variable (not flat)", given only as a figure [N7 Sec. 5.1, Fig. 11] | — | a calibration projector, not a sonar source; no level figure readable. Possible for a prototype trial if available on loan; its band stops in the middle of ours |
| ITC-1042 (reference projector in a laboratory) | small omnidirectional sphere | TVR "148 dB re 1 µPa/V @ 1m" quoted without frequency [A3] | re 1 µPa/V at 1 m | a high-frequency laboratory standard; the figure is at its own resonance, far above the band: not a candidate |

### 3.2 To check: relevant by reputation or by a search result, data sheet not opened, **no numbers given**

Model names that came back from generic web searches are marked *(search)*; a search result is not a document
opened, so its figures are deliberately left out. The others are makers known to offer projectors of the
families of Sec. 2; whether each has a 4–6 kHz model at 185–200 dB is exactly what is to be checked.

| Maker | What to look for | For option |
| --- | --- | --- |
| GeoSpectrum Technologies (Canada) | free-flooded ring series; a model *(search: M27-450)* with resonance near the band; bender-type low-frequency sources | A, B |
| Ultra Maritime / Ultra Electronics (UK, Canada) | free-flooded ring data sheet *(search: "FFR 08Fx4000")*; flextensional and barrel-stave projectors | A |
| Neptune Sonar (UK) | free-flooded ring *(search: T170)* described as covering 4–8 kHz; spherical and ring projectors | A, B, C |
| Sensor Technology / SensorTech (Canada) | larger free-flooded rings of the SX series; flextensional models | A, B |
| Gavial ITC (International Transducer Corporation, USA) | deep-water ring, sphere and flextensional projectors in the 3–8 kHz range | A, B, C |
| Benthowave Instrument (Canada) | low-frequency projector series and matching power amplifiers; small spheres and rings for the prototype | B, C, D |
| Massa Products (USA) | mid-frequency sonar transducers and Tonpilz-type projectors | A, C |
| Exail (iXblue Sonar, France) | free-flooded ring and Janus-Helmholtz sources; sub-bottom and tomography projectors | A, B |
| Kongsberg, Thales, Atlas Elektronik, L3Harris | mid-frequency sonar transducer elements; normally sold as systems | A, C |
| BTech Acoustics (USA) | multimode directional cylindrical transducers (the product form of [N5]) | C |
| Applied Physical Sciences (USA) | the flexural-disc line array of Sec. 3.1, or its elements | C |
| Lubell Labs, Electro-Voice, Ocean Engineering Enterprise, Aquarian Audio | underwater loudspeakers and diver-recall speakers for the prototype's 160–180 dB | D |
| US Navy USRD (NUWC Newport) | reference projectors on loan (G34, J-series) for a calibrated prototype trial | D |
| **Amplifiers:** Instruments Inc., AE Techron, Benthowave, Applied Physical Sciences, E&I, and the projector makers' own matched units | pulsed power at 4–6 kHz into a reactive load, with matching transformer; battery (DC bus) versions; standby power | all |

### 3.3 Power amplifiers

No amplifier data sheet could be opened. One laboratory amplifier is named in an opened thesis (a Techron 5507
driving a reference projector [N5]) without figures. The requirement, to be matched against data sheets:

| Item | Option A (197 dB, 1 s) | Option B (191–194 dB, 4–2 s) | Option C (4 × 185 dB, 1 s) | Option D (170–185 dB, 0.1 s) |
| --- | --- | --- | --- | --- |
| Real power into the transducer(s), η = 30–70% | 0.6–1.4 kW | 150–700 W | 4 channels of 40–85 W (150–340 W in all) | 1–90 W |
| Volt-amperes | real power / power factor at the worst band edge: allow 2–3 times | the same rule | the same rule, per channel | the same rule |
| Output voltage at the transducer | 10^((SL − TVR_min) / 20): 0.4–1.3 kV rms for a TVR of 145–135 dB | 0.2–0.9 kV rms | per element at 185 dB: 0.1–0.3 kV rms | under 0.35 kV rms |
| Band | 4–6 kHz at full power, flat or with programmable equalisation; arbitrary waveform input (sweep, HFM, codes) | same | same; per-channel phase control if the line is to be steered | same |
| Pulse and duty | 1 s, 5.6% | 2–4 s, 11–19% | 1 s, 5.6% | 0.1 s, 2.4–3.6% |
| Supply | DC bus of the vehicle; about 1 kW peak, 54 Wh per hour at η = 50% | 240–480 W peak, 46–51 Wh per hour | about 240 W peak, 13 Wh per hour (the line's 6 dB of directivity) | under 200 W peak, under 7 Wh per hour |
| Also | transformer and tuning matched to the chosen projector at depth; protection against open and short (a projector out of water is nearly open-circuit mechanically and can over-travel); current and voltage monitor outputs; a trigger output with known delay; standby power; conducted and radiated noise in 100 Hz–50 kHz while idle (the inputs listen then) | | | |

## 4. Recommendation: a decision table

| | **A. One source at full level** | **B. Lower level, longer pulse** | **C. A line of small projectors** | **D. Small source for the test AUV now** |
| --- | --- | --- | --- | --- |
| What it is | one free-flooded ring or flextensional at 197 dB re 1 µPa at 1 m rms across 4–6 kHz, 1 s | one smaller source at 194 dB with 2 s, or 191 dB with 4 s (HFM or a sweep with a Doppler bank) | 2–4 elements of 185–191 dB each, in phase, half a wavelength apart along the face, each with its own amplifier channel | an off-the-shelf source of 170–185 dB, 0.1 s pulse, on the test AUV and the static frame |
| Link budget (noise-limited, ship at 12 km, 1 / 2 / 3 rows) | +1.6 / +4.0 / +5.4 dB of margin | the same (same energy), if the Doppler processing holds its loss under 1 dB | the same on the line's axis; 3 dB less at ±30° (2 elements) or ±13° (4) | not the 12 km point: 2.0–4.4 km on the ship, 0.7–1.5 km on the small target with one short row |
| Blind range | 750 m | 1.5 km (2 s) or 3 km (4 s) | 750 m | 75 m |
| Direct sound left (nearest hydrophone, rms convention, face placeholder) | +22 dB: all of it for the front end | +19 dB (2 s) or +16 dB (4 s) | +14 dB (2 elements) to +10.5 dB (4), worst frequency | +15 dB at 175 dB and 0.30 m; 0 dB at 175 dB and 1.7 m |
| Cavitation near the surface | needs 0.13 m² of wet area at the surface, or 10–20 m of depth for a 10 cm-class source: **limiting for the static frame at 10 m and quay tests** | free at the surface from 0.06 m² (194 dB) or 0.03 m² (191 dB) | each 185 dB element is free at 5 m from 0.01 m² | none |
| Power (η = 50%) | 410 W acoustic, 0.96 kW from the supply in the pulse, 54 Wh per hour | 205 / 102 W acoustic, 480 / 240 W peak, 51 / 46 Wh per hour | 205 W (2 elements) or 102 W (4) acoustic; 27 / 13 Wh per hour | under 30 W acoustic, under 7 Wh per hour |
| Drive voltage (TVR 140 dB as an example) | 708 V rms | 500 / 355 V rms | 355 V (191 dB) or 178 V (185 dB) per element | 30–180 V rms |
| Size and mass | one unit of the order of 0.2 m (scaling); mass unknown | smaller unit possible; unknown | 2–4 units on a 0.15–0.45 m bar; more cabling, more penetrators | small; unknown |
| Redundancy | none | none | loses 2.5 dB (1 of 4) or 6 dB (1 of 2) with one element out, and its null | — |
| Risk | finding one unit that holds 197 dB over the whole 40% band at 5.6% duty; heat; cavitation shallow; the full 22 dB of overload | Doppler loss and the larger bank; blind range; tracker update slower by 1–3 s; reverberation-limited water gains nothing from the longer pulse | mutual coupling between close elements changes each one's loading (the array's source level is not exactly N² of one: an NPS model of ring-shell lines finds nulls filled and 2 dB differences from the point-source sum [N1]); narrow transmit sector; elevation bias from the position | little: it defers, it does not decide |
| Unknown | whether such a unit exists as a product: no data sheet opened | the same for the smaller unit; time-spread of the site | whether a 185–191 dB element exists for 4–6 kHz with a depth rating (the measured ones are 3.8 kHz, rating unstated) | the TVR of loudspeaker-class sources at 4–6 kHz |
| Sets up | the plan's stage 2 as written ("free-flooded ring or flextensional ... 1–2 kW amplifier") | a 250–500 W amplifier instead of 1–2 kW | the direct-sound and cavitation margins the others lack | the direct-sound, recovery, replica and trigger measurements all others need |

**How to choose.**

1. **Do D first, whatever follows.** It is cheap, it is what the test AUV needs, it removes the direct-sound
   problem from the first hardware (175 dB at 1.7 m is at full scale), and it produces the measurements that
   decide the rest: input recovery, coupling from the amplifier, the replica path, the trigger. Buy it with an
   amplifier that accepts arbitrary waveforms so the 2–4 s HFM of option B can be tried at low level.
2. **Send one RFQ (Sec. 5) for A, B and C at once.** Whether A exists as a single unit at 5.6% duty over the
   whole band is a data-sheet question that could not be answered here. If it does and the vehicle runs deeper
   than 20 m, A is the simplest. If the unit that exists is 3–6 dB short, B closes the gap in software. If the
   front end cannot be made to recover from +22 dB (the bench measurement of `docs/DESIGN.md`), C is the only
   option here that takes 8–12 dB off at the source.
3. **B and C combine** (a pair at 188 dB each, 2 s): 194 dB on axis, about +11 dB of direct sound, free of
   cavitation at the surface with elements of 0.016 m² or more, a 250 W-class amplifier. That is the lowest-stress full-range configuration on
   these numbers, and the one with the most unknowns in processing.
4. For the **static frame**, hanging the projector 16–32 m from the array (above, below or on a separate line)
   is available and removes the overload by distance; it also permits continuous transmission (Sec. 1.8).

## 5. Questions for the manufacturer (RFQ checklist)

State: 4–6 kHz linear or hyperbolic FM, 1 s (alternatively 2–4 s), repeated every 18 s, source level 197 dB re
1 µPa at 1 m rms (alternatively 191–194 dB; alternatively 185–191 dB per element for an array of 2–4), operating
depth 5–300 m (parameter), on a battery-powered vehicle.

1. **TVR curve** 3–7 kHz (dB re 1 µPa/V at 1 m), and whether its levels are rms or peak.
2. **Maximum source level at 4, 5 and 6 kHz** separately, with what limits each (voltage, current, stress, heat,
   cavitation), for a 1 s pulse and for a 4 s pulse.
3. **Maximum drive against depth**: the TVR and the limits at 5, 10, 30, 100, 200, 300 m; depth rating
   (operating and survival); any compensation system and how it behaves on fast depth changes.
4. **Cavitation-limited source level** against depth as the maker measures or computes it, and the wet radiating
   area.
5. **Duty**: maximum pulse length and duty cycle at full power in water at 25 °C; thermal time constant;
   temperature monitoring.
6. **Beam patterns** at 4, 5, 6 kHz in both planes; directivity index; for arrays the pattern of a pair or line
   at half-wavelength spacing and the change of each element's impedance in the array.
7. **Impedance or admittance** (G and B) across 3–7 kHz in water, at depth; clamped capacitance; recommended
   tuning and matching network; maximum voltage at the terminals and on the cable.
8. **Efficiency** across the band (not only at resonance).
9. **Dimensions, mass in air and in water**, mounting points, clearance to structure that the maker requires,
   materials in contact with sea water, cable type and length, connector.
10. **Receive response and recovery**: whether the projector rings after the pulse (mechanical Q), since ringing
    extends the direct sound past the nominal end of the pulse.
11. **Matching amplifier**: model, DC supply range, peak and standby power, output voltage and VA, waveform
    input, monitor and trigger outputs, noise when idle, size and mass, whether it is rated for a pressure
    housing.
12. **Calibration** supplied with the unit (traceable TVR of the delivered serial number), lead time, price,
    export classification.

## 6. What could not be verified, and hosts that were refused

**Not verified:** any manufacturer's data-sheet value; therefore whether a single commercial projector delivers
197 dB rms across 4–6 kHz at 5.6% duty; the mass, size and efficiency of any candidate; any amplifier's figures;
Urick's and Sherman and Butler's own statements of the cavitation limit (the relation is derived here and
checked against an open paper that reproduces Urick's chart); the transducer-family statements marked
*(not verified here)* in Sec. 2; the baffle option.

**Hosts refused by the session's network policy** (one request each; none was retried or worked around). To
complete Sec. 3, allow these hosts or fetch the named document:

| Host | What was wanted |
| --- | --- |
| `www.ultra-hyperspike.com`, `www.ultra.group` | the free-flooded ring data sheet (a PDF titled "free flooded rings datasheet") |
| `geospectrum.ca` | the free-flooded ring product page and its model spec sheets |
| `www.oceansciencetechnology.com` | a reseller page for the same rings |
| `www.neptune-sonar.co.uk` | the T170 free-flooded ring data sheet and the projector catalogue |
| `www.sensortechcanada.com` | SX-series free-flooded ring and flextensional data sheets |
| `www.gavial.com`, `www.itc-transducers.com` | ITC projector data sheets for 3–8 kHz |
| `www.benthowave.com` | low-frequency projector and power-amplifier data sheets |
| `www.massa.com`, `www.exail.com`, `www.ixblue.com`, `www.kongsberg.com`, `www.thalesgroup.com`, `www.l3harris.com`, `www.teledynemarine.com`, `www.airmar.com`, `www.btechacoustics.com`, `www.rtsys.eu`, `www.evologics.de`, `www.chelsea.co.uk`, `www.hydroacoustics.com`, `www.sercel.com` | projector catalogues (BTech: multimode directional cylinders) |
| `lubell.com`, `www.electrovoice.com`, `www.oceanears.com`, `www.aquarianaudio.com` | underwater loudspeaker data sheets for the prototype |
| `www.instrumentsinc.com`, `www.aetechron.com` | power-amplifier data sheets |
| `apps.dtic.mil` (reachable; the site answered 403 to this client), `discover.dtic.mil` | AD0743199 (ARL-TR-71-51, restates Urick's cavitation limit), ADA459091 and ADA233406 (low-frequency projector surveys) |
| `www.ioa.org.uk` (reachable; the document addresses returned "not found") | Institute of Acoustics proceedings papers on high-power free-flooded rings, class IV flextensional limits and cavitation in sonar |
| `files01.core.ac.uk`, `pdf.core.ac.uk` | a thesis on the design of free-flooded ring transducers |
| `theses.hal.science`, `publications.tno.nl`, `cradpdf.drdc-rddc.gc.ca`, `www.dst.defence.gov.au` | theses and laboratory reports on flextensional and ring projectors and on continuous active sonar |
| `image-ppubs.uspto.gov`, `patents.google.com` | patents returned by searches (barrel-stave, slotted-cylinder, free-flooded and multimode projectors) |
| `dosits.org`, `rdrr.io`, `en.wikipedia.org`, `www.ncbi.nlm.nih.gov`, `www.researchgate.net`, `www.semanticscholar.org`, `www.sciencedirect.com`, `archive.org`, `www.osti.gov`, `www.npl.co.uk`, `www.nrl.navy.mil` and other publisher and library hosts | background on cavitation limits and transducer families; open copies of papers |

Also: `link.springer.com` was reachable but returned no article text to this client; `repository.tudelft.nl`
was reachable but the two records searched for (a TNO report on low-frequency continuous active sonar; a thesis)
were not found at the addresses returned.

## 7. Sources opened in this session

Open papers (arXiv):

- [A1] "Towards a general description of the cavitation threshold in acoustic systems", arXiv:2311.13425,
  https://arxiv.org/abs/2311.13425 (Sec. 1 and Fig. 1: Urick's chart of threshold against frequency and the
  measurements behind it; Eq. 15 and Fig. 12: the hydrostatic term and the 0.99 bar result at 0.48–1.43 kHz).
- [A2] "The Generalized Sinusoidal Frequency Modulated Waveform for Active Sonar Systems", PhD dissertation,
  University of Massachusetts Dartmouth, arXiv:1809.11009, https://arxiv.org/abs/1809.11009 (Ch. 1: peak-power
  and cavitation limits and long waveforms; Ch. 2: Doppler tolerance, HFM; Sec. 6.1: pulsed against continuous
  active sonar).
- [A3] "Acoustic Transmitters for Underwater Neutrino Telescopes", arXiv:1204.0809,
  https://arxiv.org/abs/1204.0809 (Sec. 2: the SX30 free-flooded ring's figures, pressure tests, the ITC-1042
  reference projector). Also read: arXiv:1204.0900 and arXiv:1210.7936 (the same transducer).

Naval Postgraduate School archive (Calhoun):

- [N1] E. Ozer, "Application of the T-matrix method to the numerical modeling of a linear active sonar array",
  thesis, 2013, https://calhoun.nps.edu/items/141a6a14-5454-454c-aa13-299c236a99e6 (Sec. 2.3 and Table 3.2: the
  ring-shell projector; Ch. 5: line arrays of 3–8 projectors against the point-source sum).
- [N2] R. N. C. Pinto, "Three-dimensional finite element model of a high power, low frequency ring-shell
  flextensional sonar transducer", thesis, 1992, https://hdl.handle.net/10945/24073.
- [N3] T. B. Tay, "Numerical modeling of a free-flooded piezoelectric ring sonar transducer", thesis, 1993,
  https://hdl.handle.net/10945/24204 (Ch. II: the ring's dimensions and the manufacturer's characteristics).
- [N4] K. E. Jenne, "Acoustic cymbal transducers: design, hydrostatic pressure compensation, and acoustic
  performance", thesis, 2004, https://hdl.handle.net/10945/1670 (Ch. I: the flextensional classes; pressure
  dependence and compensation of miniature class V units).
- [N5] G. S. Sineiro, "Underwater multimode directional transducer evaluation", thesis, 2003,
  https://hdl.handle.net/10945/6143.
- [N6] O. B. Wilson, "Development work on acoustic transducers for underwater range tracking: summary report",
  1979, https://hdl.handle.net/10945/29251 (pp. 8–9: intensity against cavitation thresholds at shallow depth).
- [N7] C. W. Miller and A. Kumar, "San Clemente Island Undersea Range Acoustic Experiment, July 2002",
  NPS-OC-04-001, 2003, https://hdl.handle.net/10945/685 (Sec. 5.1: the G34 projector).
- [N8] "Biological and Behavioral Response Studies of Marine Mammals in Southern California, 2011 (SOCAL-11):
  Final Project Report", NPS-OC-12-005CR, 2012,
  https://calhoun.nps.edu/items/f16aacbb-df4d-4cea-975b-9f5886f646f0 (Sec. 2 and the sound-source appendix,
  pp. 38–40: the 15-element source, its calibration table and beam steering).
- [N9] B. S. Adams, "An analysis of the effects of energy spreading loss and transmission loss on low frequency
  active sonar operations in shallow water", thesis, 1997,
  https://calhoun.nps.edu/items/7b583fbe-ec21-4d14-8524-3bbf343aebaf (abstract).

Repository files read: `docs/DESIGN.md` (Sec. 3 and 5), `docs/design/popoto_parts.md`,
`docs/design/direct_sound_study.json`, `docs/design/popoto_12km.json`, `configs/builds/*.yaml`,
`tools/popoto_range_study.py`, `python/gpusonar/buildspec.py`, `docs/HARDCODED_AUDIT.md` (P-5),
`docs/REFERENCE_DESIGN.md`.

## Appendix A. Feeding back into the build spec

Nothing under `configs/builds/`, `buildspec.py` or the plan tools was edited. Proposed values per option, for
their owner to apply; every one stays `unconfirmed` until a data sheet or a measurement backs it.

**Existing `projector` fields** (full face unless stated):

| Field | A | B | C | D (test AUV / static frame) |
| --- | --- | --- | --- | --- |
| `model`, `amplifier` | null until chosen | null | null | null until bought |
| `source_level_db` | 197.0 | 194.0 (2 s) or 191.0 (4 s) | 197.0 on axis (4 × 185.0 or 2 × 191.0) | 175.0 (range 170–185) |
| `band_hz` | [4000, 6000] | [4000, 6000] | [4000, 6000] | [4000, 6000], or the source's usable part of it |
| `waveform` | `{type: lfm, duration_s: 1.0, window_alpha: 0.1}` | `{type: sampled, source: hfm, duration_s: 2.0}` (or 4.0); `lfm` with a Doppler bank as the alternative | as A | `{type: lfm, duration_s: 0.1, window_alpha: 0.1}` |
| `position_m` | [4.0, 0, 0] (placeholder; as far from the rows as the platform allows) | as A | centre of the line at [4.0, 0, 0], elements along the first axis | test AUV: at the end of the hull, about [1.7 + half the row, 0, 0] instead of [0, 0, 0.3]; static: hung apart if the frame allows |
| `blanking_after_s` | measured on the bench, expected largest here | smaller | smaller still | first measured here |
| `operation.range_m[0]` (blind range) | null (750 m) | null (1500 or 3000 m) | null (750 m) | null (75 m) |

**Fields worth adding to the `projector` section** (all optional, all "leave out: unknown"):

| Field | Unit | Meaning | Used by |
| --- | --- | --- | --- |
| `source_level_reference` | `rms` or `peak` | the convention of `source_level_db` (the owner: rms); finding P-5 | direct-sound overload (+3 dB when rms), simulator amplitude |
| `usable_band_hz`, `tvr_db` (a list of [Hz, dB re 1 µPa/V at 1 m]) | Hz, dB | the data sheet's response | a check that `band_hz` lies inside it; band-edge droop; drive voltage |
| `max_voltage_v_rms`, `max_power_w`, `duty_max`, `pulse_max_s` | V, W, –, s | the data sheet's limits | a check against the level, pulse and ping interval of the build |
| `efficiency` | – | electroacoustic, in band | power and energy lines of the parts list |
| `radiating_area_m2` | m² | wet moving area | the cavitation check below |
| `depth_rating_m` | m | operating depth rating | a check against `trial.own_depth_m` and the vehicle's depth |
| `directivity_index_db`, `response_towards_array_db` | dB | gain towards the targets; response in the direction of the rows relative to the axis | acoustic power; direct-sound level |
| `elements`, `element_spacing_m`, `element_axis` | –, m, `along` / `vertical` | a line of projectors | the exact near-field direct sound (`line_direct` in the tool), transmit sector |
| `mass_air_kg`, `mass_water_kg`, `size_m` | kg, m | integration | parts list |

**Validator checks worth adding** (`buildspec.validate`; warnings, not errors):

1. *Cavitation.* With `radiating_area_m2` and the depth the build runs at (`trial.own_depth_m`):
   limit = 10 log₁₀(ρc · [(p_atm + ρ g h − p_v)² / 2ρc] · A / 4π) + 120 + DI. Warn when
   `source_level_db` (rms) is within 3 dB of it, and print the depth below which it is clear with 3 dB of margin.
   Implemented as `cavitation_warning(source_level, depth_m, area_m2)` in `tools/projector_requirements.py`;
   with no area given, warn above 190 dB at depths under 20 m that a compact source may cavitate
   (0.03 m² holds 194.3 dB at 5 m, 196.8 dB at 10 m, 202.8 dB at 30 m).
2. *Convention.* When `source_level_reference` is `rms` (or absent, by the owner's decision), add 3 dB to
   every direct-sound overload figure and drop the "3 dB more if the source level is an rms figure" wording.
3. *Band.* Warn when `band_hz` is not inside `usable_band_hz`, and when the fractional band exceeds 1 / Q of a
   stated Q.
4. *Duty and pulse.* Warn when pulse / ping interval exceeds `duty_max` or the pulse exceeds `pulse_max_s`.
5. *Depth.* Warn when `trial.own_depth_m` exceeds `depth_rating_m`.
6. *Doppler.* For `type: lfm`, warn when c / (2 B T) is under the platform's speed (`trial.speed_m_s`): 0.38 m/s
   for 1 s and 2 kHz against 1.5 m/s on the face builds today. The chain needs a Doppler bank or an HFM pulse.
7. *Energy.* Print acoustic power, energy per ping and per hour for an assumed or stated efficiency in
   `describe` (`energy()` in the tool), so the parts list's power line stops saying "no draw is known" for
   the transmit chain's part of it.

## Appendix B. Reproducing the numbers

    PYTHONPATH=python python3 tools/projector_requirements.py                      # all builds; writes docs/design/projector_requirements.json
    PYTHONPATH=python python3 tools/projector_requirements.py popoto_face_3x32
    PYTHONPATH=python python3 tools/projector_requirements.py popoto_testauv_1x32 --levels 160 165 170 175 180 185 190 197 --ranges 500 1000 2000 --no-json

The tool reads the build spec, so a changed source level, band, pulse, projector position, hydrophone
sensitivity or gain changes every table here. Its link table is `buildspec.performance` with the level and the
row count replaced; its direct-sound figures are `buildspec.describe`'s plus 3 dB for the rms convention; the
cavitation, power, pulse-trade, flatness and line-array arithmetic is its own and is stated in its docstring.
