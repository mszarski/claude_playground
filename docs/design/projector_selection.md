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
- **Manufacturer data sheets were opened for four makers** (Neptune Sonar, GeoSpectrum, Massa, Benthowave) after
  the owner allowed their hosts; Teledyne, Exail and BTech were reachable and list nothing for 4–6 kHz at this
  level; six further hosts redirect to addresses that are still refused, and one maker's catalogue sits on a host
  that is not allowed (Sec. 6). Sec. 3 gives every candidate's
  figures with the sheet and page they came from. Curves were read by eye off the sheets' plots: TVR to about
  ±1 dB, admittance and impedance to about ±15 %, beam patterns to about ±2 dB. No product figure in this
  document comes from memory.
- **The owner's depth: "Depth no further than 500m"** (owner-stated). It is used as the required depth rating
  throughout, with the convention *rated operating depth ≥ 500 m* (no extra factor is added here: the sheets'
  own margins are unknown, and none of the sheets read distinguishes an operating from a survival depth; Sec. 1.5).
- **Answer to the main question: yes, single listed units meet 197 dB rms across 4–6 kHz at 5.6 % duty by their
  own sheets**: the Neptune T335 (198.5 dB at its 10 % duty rating) and the Massa TR-1075A / D (197–202 dB at
  their 30 % rating). Both are directional pistons rated to 600 m. What the sheets leave open: the longest pulse
  at that drive (the build's is 1 s), the cavitation depth (Massa: about 30 m at full power; Neptune: not stated
  for the T335), and the level convention. The free-flooded rings read reach 190–197.5 dB each and need a pair.
- No projector is chosen. The recommendation is a decision table, now with named candidates.

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
| Cavitation | by the makers' own statements none of the candidates holds 197 dB at 5 m: about 9–17 m is needed (T170: 750 V from 10 m; TR-1075: full power from about 30 m), 53 m for a single small ring. The uniform-face relation derived here (0.13 m² free at the surface) is an upper bound 0–15 dB above the makers | Sec. 1.4, 3.3 |
| Depth rating | **500 m (owner: "Depth no further than 500m")**; rated operating depth ≥ 500 m. The design's surface-image formula puts the best own depth for a target at 4 m at 12 km at 225 m, inside it | Sec. 1.5 |
| Direct sound | +22.1 dB over the inputs' full scale at the nearest hydrophone (rms convention), +10.6 dB at the farthest; a response 22 dB down towards the array (28 dB with margin) would remove it; full scale is reached 21.5 m away | Sec. 1.6 |
| Prototype (test AUV, 0.1 s pulse) | 170 dB reaches the ship-sized target at 2.0 km and the small one at 0.7 km; 180 dB: 3.4 / 1.2 km. Less than 10 W acoustic; the full-scale candidates do it at 15–300 V | Sec. 1.9, 3.5 |

| A single listed unit? | yes by their sheets: Neptune T335, Massa TR-1075A / D (directional pistons, 600 m, about 11 kg); free-flooded rings (unlimited depth, about 5 kg) as a pair | Sec. 3 |

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
  data sheets state as a duty cycle and sometimes a maximum pulse length. The sheets read rate full drive at
  10 % (Neptune), 30 % (Massa) and 20–40 % with pulses of 30–200 s (Benthowave), against 2 % for one small ring
  in an open paper [A3]: **5.6 % is inside the candidates' ratings; the 1 s pulse length is on no candidate's
  sheet** and is an RFQ item.
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

**What the makers say (read in this session), and what it does to the relation.** Four data sheets state a
cavitation depth, and they are lower than the uniform-face relation, by a little for pistons and by a lot for
free-flooded rings (Sec. 3.3 has the table):

- Neptune T170 ring: "750 Vrms (Minimum 10m depth)", i.e. 197.5 dB from 10 m; the relation gives 202.7 dB there
  for its 0.118 m² of wet area: **5 dB above the maker**.
- GeoSpectrum M27-450 ring: "Cavitation Depth at 193 dB re µPa @ 1 m: 30 Meters"; the relation gives 207.8 dB at
  30 m: **15 dB above the maker**. Their 5 kHz bender: 190 dB from 25 m, 7 dB below the relation.
- Massa TR-1075 piston: "Minimum water depths of approximately 30 and 100 feet are recommended for input power of
  200 and 600 Watts": within 2 dB of the relation before any directivity index is added to it (which would
  raise the relation by the piston's DI).
- Neptune T420 small piston: "2,000 Vrms (Minimum 20m depth)": equal to the relation without DI.

So **the relation is an upper bound**. In a ring the largest pressure is in the bore, not on the outside, and
the maker's limit reflects it. Use the maker's figure where there is one, scaled with depth as 20 log₁₀ of the
static pressure (the one part of the relation all four statements are consistent with: Massa's two points,
200 W at 9 m and 600 W at 30 m, differ by 4.8 dB in power and 6.5 dB in static pressure); where there is none,
take the relation less 5 dB for a piston's face and less 10–15 dB for a ring, until the maker answers.

*Tool* (full face, 197 dB; the uniform-face bound):

| Wet radiating area | sphere of that area | at 5 m | at 10 m | at 30 m | at 100 m | at 200 m | at 500 m | least depth for 197 dB | the same with half the face doing the work |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.01 m² | 0.06 m dia. | 189.5 dB | 192.0 dB | 198.1 dB | 206.9 dB | 212.5 dB | 220.2 dB | 25 m | 40 m |
| 0.03 m² | 0.10 m dia. | 194.3 dB | 196.8 dB | 202.8 dB | 211.6 dB | 217.3 dB | 225.0 dB | 10 m | 19 m |
| 0.1 m² | 0.18 m dia. | 199.5 dB | 202.0 dB | 208.1 dB | 216.9 dB | 222.5 dB | 230.2 dB | 1 m | 6 m |
| 0.3 m² | 0.31 m dia. | 204.3 dB | 206.8 dB | 212.8 dB | 221.6 dB | 227.3 dB | 235.0 dB | 0 m | 0 m |

Threshold: 0.73 W/cm² at 5 m, 1.30 W/cm² at 10 m, 5.24 W/cm² at 30 m, 39.72 W/cm² at 100 m, 144.88 W/cm² at 200 m, 854.92 W/cm² at 500 m. 197 dB is free of cavitation at the surface from 0.127 m² of wet radiating area (a sphere 0.20 m across).

Reading it:

- **Even on the bound, 197 dB from a compact source (0.03 m²) is at the limit at 10 m and 2.7 dB over it at
  5 m.** On the makers' statements no candidate holds 197 dB at 5 m; the least depths are 9 m (T170, at 6 kHz),
  17–18 m (TR-1075A / D) and 53 m (one M27-450 driven to 197 dB, which its voltage rating forbids anyway).
- Depth helps fast: +8.5 dB from 5 m to 30 m, +17 dB to 100 m, +31 dB to the owner's 500 m (51 bar absolute;
  the bound's threshold is 855 W/cm² there). **Below about 30 m cavitation stops being the limit for every
  candidate at 197 dB; from there to 500 m the electrical and thermal ratings set the level.**
- **The static frame at 10 m and the test AUV in a 30 m trial site are the cavitation-limited cases**, not the
  deep-running vehicle. A full-level test of a compact projector hung at 5 m from a quay would cavitate.
- An array of N small projectors has N times the area for the same total power: each element of a four-element
  line runs at 185 dB (Sec. 1.6). On the bound a 0.01 m² element holds that from the surface (189.5 dB at 5 m);
  on GeoSpectrum's statement for its 5 kHz bender the limit is 182.6 dB at 5 m and 185.1 dB at 10 m, so even
  that needs 10 m.

### 1.5 Operating depth and depth rating

**The owner: "Depth no further than 500m".** The vehicle's maximum operating depth is 500 m (owner-stated), and
it is the required depth rating of the projector, its cable and its connectors. Convention applied: **the data
sheet's rated operating depth must be at least 500 m**; no further factor is applied here. None of the sheets
read gives a separate survival (crush) depth: Neptune and Benthowave say "Operating Depth", GeoSpectrum and
Massa "Maximum Operating Depth"; whether those figures already hold a margin, and what the survival depth is,
are RFQ items. The same 500 m applies to the hydrophones, their cables and connectors and any wet-end
electronics in the parts list.

What the design's own logic implies about where in that range the vehicle would run: a
ship-sized target is near the surface (4 m in the scenario), and `docs/DESIGN.md` ("A target near the surface is
the hard case for straight rays") gives the surface-image factor on each leg as 20 log₁₀(2 sin(2π z₁ z₂ / λR)).
It is largest when z₁ = λR / 4z₂: **225 m of own depth for a target at 4 m at 12 km**; at the 10 m the scenario
uses it is −17 dB each way. *Tool:*

| Own depth (m) | 5 | 10 | 30 | 100 | 200 | 500 |
| --- | --- | --- | --- | --- | --- | --- |
| surface image, one way (dB) | -23.1 | -17.1 | -7.6 | +2.2 | +5.9 | -3.3 |
| absolute pressure (bar) | 1.5 | 2.0 | 4.0 | 11.1 | 21.1 | 51.3 |
| cavitation limit, 0.03 m² (dB) | 194.3 | 196.8 | 202.8 | 211.6 | 217.3 | 225.0 |

This is the first estimate the task statement mentions (straight rays, isovelocity; another study is refining it
with profiles and bottom bounces, which put energy back at shallow own depth). For the projector it means:

- **Depth rating: 500 m (owner).** The 225 m optimum of this first estimate is inside it; the test AUV and the
  static frame need only tens of metres. Screening of the candidates (Sec. 3.2): the free-flooded rings pass
  with unlimited or 6000 m ratings; the pistons (T335, T296, T420, TR-1075) and the 5 kHz bender pass with 600 m
  and 700 m (1.2 and 1.4 times); the Benthowave BII7594 is rated exactly 500 m (no margin); the BII7596 (300 m),
  the BII7530 low-frequency series (150 m) and Neptune's T473 flextensional (50 m) fail. No maker reached lists
  pressure compensation as an option for a 4–6 kHz unit; GeoSpectrum lists passive and active compensation only
  for its very-low-frequency C-BASS sources.
- **Which types change their response between 0 and 500 m.** *Free-flooded rings: not expected to* (pressure
  balanced; Neptune: "operate down to full ocean depth at maximum efficiency"; GeoSpectrum: "unlimited depth
  capability"). *Tonpilz pistons: rated, but the change of TVR and impedance with depth is not on any sheet*
  (Neptune says only "tolerant of both dynamic and static pressure"); a piston's backing (air, compliant or
  fluid) decides it: an RFQ item. *Flexural discs (benders): air-backed and depth-limited by construction* (the
  same maker's standard benders are rated 80, 190, 270, 400, 500 and 700 m by model, and it quotes its
  electrodynamic sources' resonance "@ 20 m depth", which says the resonance moves with depth); expect the
  resonance and TVR of a bender to shift with static pressure and ask for the curve at 0, 100, 250 and 500 m.
  *Flextensionals: depth-limited unless compensated* (Neptune's own, at 1.3 kHz, is rated 50 m).
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
four times larger; no such product was found on the maker pages reached (Sec. 3.4). A cardioid's rear null
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
  amplifier or supplied with the transducer and tuned to it. The custom line array of Sec. 3.7 sends about
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

Product sizes and masses from the data sheets are in Sec. 3.1; this section is the reasoning behind them.

- **Size follows wavelength** (30 cm at 5 kHz). The one free-flooded ring with dimensions in a report opened
  here resonates at 2.5 kHz (ring mode) with a 0.367 m outside diameter and 0.17 m ceramic height [N3]; the ring
  resonance scales inversely with diameter, so a 5 kHz ring is of the order of **0.18 m across and 0.1 m tall**
  (a scaling estimate; the sheets since read agree: the 6 kHz T170 ring is 0.191 m across and 0.085 m high, the 4.25 kHz M27-450 has a 0.15 m body 0.137 m high). The custom 3.8 kHz line array of Sec. 3.7 uses flexural discs on 0.152 m
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
| **Free-flooded ring** (piezoceramic ring, water inside and out) | ring breathing mode plus the resonance of the water column in the bore; the two together give a wide band | two resonances an octave or so apart (1.0 kHz cavity and 2.5 kHz ring in the one quoted [N3]); broad | 132 dB re 1 µPa/V at the ring resonance for the 0.37 m ring [N3]; a 4.4 cm ring: 133 dB re 1 µPa/V at 30 kHz, 300 W at 2% duty [A3] | pressure-balanced: "unlimited" [N3]; measured stable to 440 bar [A3] | omnidirectional around the axis; in the plane containing the axis the beam depends on the ring's height (about 60° for the small ring of [A3]) | yes: [N3], [A3], and the Neptune T170 / T160 and GeoSpectrum M27 sheets (Sec. 3): 190–197.5 dB per ring in the band, unlimited or 6000 m, toroidal; the plan names this family for stage 2 |
| **Flextensional, classes I–VII** (ceramic stack or ring driving a flexing shell: barrel-stave I–III, oval shell IV, ring-shell V–VI, "dog-bone" VII) | small extensional motion of the driver becomes large flexural motion of the shell: "large volume velocities ... at low frequencies for the size of the device" [N4] | a ring-shell at 610 Hz: 470–750 Hz at −3 dB (Q about 4) [N2]; "wide bandwidth and low mechanical quality factor" [N1] | that unit: 213 dB re 1 µPa at 1 m at 3000 V rms, 90% efficient at resonance, 65% at −6 dB [N2] | "limited to a few meters without a depth compensating system"; with a water bladder "exceeds 400 m" [N1, N2]; uncompensated miniature units lose tens of dB with pressure [N4] | omnidirectional when small against the wavelength | yes for the class V figures at 0.6 kHz [N1, N2]; nothing opened gives a 4–6 kHz flextensional's figures. Most of the family is built for 0.3–3 kHz *(not verified here)* |
| **Flexural (bender) disc** | a ceramic-metal disc bending like a drumhead; air- or oil-backed | resonant; the 15-element array below is tuned to 3.8 kHz and used over 3.5–3.9 kHz [N8] | measured 181–191 dB re 1 µPa at 1 m per element at 3.8 kHz, 210 dB for 15 in line [N8] | air-backed discs are depth-limited unless compensated *(not verified here)*; the array's rating is not stated in [N8] | each disc near-omnidirectional; used in arrays | yes: [N8], and GeoSpectrum's M21 sheets: a 0.6 kg disc resonant at 5 kHz, 190 dB maximum, about 0.85 kHz wide, rated 700 m; the family's ratings run from 80 m to 700 m by model |
| **Tonpilz (piston) and arrays of them** | ceramic stack between a head mass that radiates and a tail mass | Q of 3–7 typical, a 20–40% band with matching *(not verified here)* | the standard element of hull sonars at 3–10 kHz; an array of pistons is directional by construction *(not verified here)* | usually pressure-tolerant with a compliant or fluid backing *(not verified here)* | a single piston is omnidirectional when small; a baffled array has a rear rejection that suits the direct-sound problem | yes, as products: Neptune T335 (3–8 kHz, 146.5 dB re 1 µPa/V flat, −12 dB at 90° at 5 kHz, 600 m) and Massa TR-1075 (2.5–10 kHz, 80° cone, 600 m), both described by their makers as array elements (Sec. 3) |
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

**Strict rule applied: every figure below was read in the manufacturer's public data sheet or product page in
this session**; the URL and the page or figure are in the tables and in Sec. 7. Values read off a plotted curve
are marked by the precision stated at the top of this document. No contact, quote or download-gate form was used;
no price is printed on any page read.

### 3.1 Data-sheet values

*Tool* (`tools/projector_candidates.py`; the values and their sources are the `CANDIDATES` table in that file):

| Model | Type | Resonance; band | TVR at 4 / 5 / 6 kHz (dB re 1 µPa/V at 1 m) | Rated drive (duty) | Pattern | Depth | Size | Mass | Impedance | Convention; pulse length |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Neptune T335 | Tonpilz piston with internal tuning network, directional | 3.5 / 7.0 kHz; 3-8 kHz | 146.5 / 146.5 / 146.5 (sheet: 146 dB re 1 uPa/V at 1 m) | 750 V rms at 1 %; 400 V rms at 10 %; 100 V rms at 100 % | conical; at 5 kHz about -3 dB at +/-45 deg | Operating Depth 600 m | 183 x 183 mm flange, 257 mm long, body dia. 120 mm | 11.4 kg air / 6.7 kg water | 160 ohm nominal | drive in V rms; level convention not stated; not stated |
| Massa TR-1075A | Tonpilz piston, parallel-tuned with transformer, directional | 3.5 kHz untuned; tuned for 0 deg phase at 4 kHz; 2.5-10 kHz | 152.5 / 150.0 / 149.5 (sheet: curve only) | 600 W at 30 %; 200 W at 100 % | 80 deg conical beam angle; piston half a wavelength across at 4 kHz | Maximum Operating Depth is 600 meters | 7.25 in (184 mm) square flange, about 8.6 in (218 mm) long | 25 lb (11.3 kg) in air; in water not stated | 100 ohm at 4 kHz nominal | not stated (power in watts; curves 'dB re 1 uPa/1V @ 1m'); not stated |
| Massa TR-1075D | Tonpilz piston, series-tuned with shunt resistor (lower Q) | tuned at 3.5 kHz; 2.5-10 kHz | 147.5 / 146.5 / 148.5 (sheet: curve only) | 1000 W at 30 %; 335 W at 100 % | 80 deg conical beam angle | Maximum Operating Depth is 600 meters | as TR-1075A | 25 lb (11.3 kg) in air | 200 ohm at 3.5 kHz nominal | not stated; not stated |
| Neptune T296 | piston projector, directional | 1.7 kHz; 1-6 kHz | 139.5 / 142.0 / 137.5 (sheet: 142 dB re 1 uPa/V at 1 m) | 750 V rms at 10 %; 350 V rms at 100 % | conical; patterns given at 1.7 and 3.5 kHz only | Operating Depth 600 m | 260 x 260 mm flange, 418 mm long, face dia. 264 mm | 27.1 kg air / 12.2 kg water | 400 ohm nominal | drive in V rms; level convention not stated; not stated |
| Neptune T170 | free-flooded ring | 6 kHz; 4-8 kHz | 135.0 / 139.0 / 140.0 (sheet: 140 dB re 1 uPa/V at 1 m) | 750 V rms at 10 %; 250 V rms at 100 % | omni +/-2 dB around the axis; toroidal, nulls on the axis (patterns at 4 and 6 kHz) | Operating Depth Unlimited | ring 191 mm outside, 121 mm inside diameter, 85 mm high | 4.7 kg air / 2.6 kg water with 10 m of cable | 68 nF at 1 kHz with 10 m of cable | drive in V rms; level convention not stated; not stated |
| GeoSpectrum M27-450 | free-flooded ring | 4250 Hz; curve 2.5-6.5 kHz | 136.0 / 135.0 / 134.0 (sheet: 138 dB re 1 uPa/V at 1 m at resonance) | 600 V rms (duty not stated) | toroidal (pattern at 4.25 kHz) | Maximum Operating Depth 6000 Meters | dia. 6.5 in (165 mm) flange, 5.9 in body, 2.71 in bore, 6.7 in (170 mm) high | 5 kg in air; in water not stated | 60 nF | drive in V rms; level convention not stated; not stated; duty not stated |
| Neptune T160 | free-flooded ring | 2.8 kHz; 2.5-6 kHz | 133.5 / 134.0 / 130.0 (sheet: 134 dB re 1 uPa/V at 1 m) | 750 V rms at 10 %; 250 V rms at 100 % | toroidal; lobed at 6 kHz | Operating Depth Unlimited | ring 234 mm outside, 160 mm inside diameter, 106 mm high | 5.7 kg air / 2.6 kg water with 10 m of cable | 38 nF at 1 kHz with 10 m of cable | drive in V rms; level convention not stated; not stated |
| GeoSpectrum M21-070-5000 | flexural disc (bender) | 5 kHz; 1-10 kHz transmitting range | 126.0 / 136.5 / 131.0 (sheet: 137 dB re 1 uPa/V at 1 m maximum) | 600 V rms (duty not stated) | omnidirectional at resonance | Maximum Operating Depth 700 Meters | disc dia. 80 mm, 25 mm thick | 0.6 kg | 35 nF | drive in V rms; level convention not stated; not stated; duty not stated |
| Neptune T420 | small piston projector | 5 kHz; 4.5-14 kHz | 115.0 / 126.5 / 124.0 (sheet: 126 dB re 1 uPa/V at 1 m) | 2000 V rms at 10 %; 600 V rms at 100 % | at 5 kHz about -7 dB at 90 deg, -5 dB behind | Operating Depth 600 m | 102 x 102 mm flange, 165 mm long | 3.8 kg air / 2.1 kg water | 20 kohm nominal | drive in V rms; level convention not stated; not stated |
| Benthowave BII7594 | free-flooded ring, multi-resonance | 4, 7.5, 11 kHz; 1-30 kHz (series) | 130.0 / 117.0 / 118.0 (sheet: graph) | 400 V rms at 20 % | omnidirectional to toroidal | Operating Depth <= 500 m | dia. 114 x 90 mm | >= 1.2 kg in air | 115 nF; power factor 0.36 at fs | 'SL = 20*logVi + TVR ... Vi in Vrms'; 30 s at 480 W, duty <= 20 % |
| Benthowave BII7596 | free-flooded ring, multi-resonance | 3.5 and 6.5 kHz; 1-30 kHz (series) | 131.0 / 124.0 / 130.0 (sheet: graph) | 600 V rms at 40 % | omnidirectional to toroidal | Operating Depth <= 300 m | dia. 168 x 115 mm | >= 2.2 kg in air | 69 nF; power factor 0.38 at fs | 'SL = 20*logVi + TVR ... Vi in Vrms'; 200 s at 600 W, duty <= 40 % |

Sources, page by page:

| Model | Document | Where in it |
| --- | --- | --- |
| Neptune T335, T296, T170, T160, T420 | data-sheet PDFs linked from `https://www.neptune-sonar.co.uk/products/projectors/<model>` (the addresses are in Sec. 7) | two pages each (catalogue pages B19–B30): specification tables, Transmit Sensitivity, Admittance and Beam Pattern graphs, outline drawing |
| GeoSpectrum M27-450 | `https://geospectrum.ca/wp-content/uploads/2022/05/SP-M27-450-R0.pdf` | p. 1 characteristics; p. 2 Figures 1–4 |
| GeoSpectrum M21-070-5000 | `https://geospectrum.ca/wp-content/uploads/2022/05/SP-M21-070-5000-R1.pdf` | p. 1 characteristics; p. 2 Figures 1–5 |
| Massa TR-1075A / D (also E, H) | `https://www.massa.com/wp-content/uploads/2023/03/TR-1075-Datasheet-230327.pdf` | p. 1 text; p. 2 outline and tuning chart; p. 3 Nominal Response Curves |
| Benthowave BII7594, BII7596 | `https://www.benthowave.com/products/Specs/BII7590Datasheet.pdf` | pp. 1–2 specification table; p. 6 and p. 8 graphs |

Not in the table because they do not cover the band, each read on its own page or sheet: Neptune T444 (TVR curve
starts at 5 kHz: 131.5 dB there, 140.5 dB at 6 kHz), T161 (1.3–4 kHz), T406 (8–15 kHz), T589 (1.4 kHz), T473
flextensional (1–2 kHz, 50 m), the D/11 to D/45 spheres (11.5–45 kHz); GeoSpectrum M27-650 ring (resonance
3.05 kHz; its curve reads about 131–131.5 dB re 1 µPa/V from 4 to 5 kHz and ends at 5 kHz; 600 V rms, 193 dB,
cavitation depth 25 m at 193 dB, 4.5 kg in air and 2.5 kg in water, 6000 m), M27-550 / -850 / -890 rings
(0.53–1.85 kHz), the other M21 benders (0.5–3 kHz), M27-93x / -94x communications transducers (7–17 kHz), M18C
spheres (resonances 12–68 kHz), C-BASS (below 3 kHz); Massa TR-1055 (8–16 kHz) and TR-1077A (the TR-1075 for
depths "greater than 20,000 feet", 50 lb, "3.5 kHz region"; its curves were not read); Benthowave BII7530
low-frequency series (150 m maximum depth) and BII7520 spheres (15 kHz and up); Teledyne RESON TC1037 (6–15 kHz).

### 3.2 Against the requirement

*Tool:*

| Model | Level at rated drive, 4 / 5 / 6 kHz (dB) | 197 dB across the band from one unit | Units in phase for 197 dB | For 197 dB at 4 / 5 / 6 kHz | For 194 dB | For 191 dB | Depth rating against 500 m | Direct sound at 197 dB, worst hydrophone (change) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Neptune T335 | 198.5 / 198.5 / 198.5 | **yes** | 1 | 335 / 335 / 335 V; 292 / 247 / 337 W, 368 / 253 / 355 VA | 237 / 237 / 237 V | 168 / 168 / 168 V | 600 m: ok (×1.2) | +12.6 dB (-9.5) |
| Massa TR-1075A | 199.6 / 200.6 / 197.1 | **yes** | 1 | 168 / 224 / 237 V; 332 / 264 / 592 VA | 119 / 158 / 168 V | 84 / 112 / 119 V | 600 m: ok (×1.2) | +22.1 dB (+0.0) |
| Massa TR-1075D | 202.0 / 202.4 / 201.7 | **yes** | 1 | 299 / 335 / 266 V; 318 / 288 / 337 VA | 211 / 237 / 188 V | 150 / 168 / 133 V | 600 m: ok (×1.2) | +22.1 dB (+0.0) |
| Neptune T296 | 197.0 / 199.5 / 195.0 | no, 2.0 dB short | 2 | 750 / 562 / 944 V (over the rating) | 531 / 398 / 668 V | 376 / 282 / 473 V | 600 m: ok (×1.2) | +22.1 dB (+0.0) |
| Neptune T170 | 192.5 / 196.5 / 197.5 | no, 4.5 dB short | 2 | 1259 / 794 / 708 V; 1109 / 694 / 601 W, 3061 / 1251 / 924 VA (over the rating) | 891 / 562 / 501 V (over) | 631 / 398 / 355 V | unlimited: ok | +13.6 dB (-8.5) |
| GeoSpectrum M27-450 | 191.6 / 190.6 / 189.6 | no, 7.4 dB short | 3 | 1122 / 1259 / 1413 V; 441 / 190 / 399 W, 2309 / 3175 / 5004 VA (over the rating) | 794 / 891 / 1000 V (over) | 562 / 631 / 708 V (over) | 6000 m: ok (×12.0) | +13.6 dB (-8.5) |
| Neptune T160 | 191.0 / 191.5 / 187.5 | no, 9.5 dB short | 3 | 1496 / 1413 / 2239 V (over the rating) | 1059 / 1000 / 1585 V (over) | 750 / 708 / 1122 V (over) | unlimited: ok | +13.6 dB (-8.5) |
| GeoSpectrum M21-070-5000 | 181.6 / 192.1 / 186.6 | no, 15.4 dB short | 6 | 3548 / 1059 / 1995 V; 378 / 482 / 199 W, 11337 / 1325 / 4781 VA (over the rating) | 2512 / 750 / 1413 V (over) | 1778 / 531 / 1000 V (over) | 700 m: ok (×1.4) | +22.1 dB (+0.0) |
| Neptune T420 | 181.0 / 192.5 / 190.0 | no, 16.0 dB short | 7 | 12589 / 3350 / 4467 V (over the rating) | 8913 / 2371 / 3162 V (over) | 6310 / 1679 / 2239 V (over) | 600 m: ok (×1.2) | +15.1 dB (-7.0) |
| Benthowave BII7594 | 182.0 / 169.0 / 170.0 | no, 28.0 dB short | 25 | 2239 / 10000 / 8913 V (over the rating) | 1585 / 7079 / 6310 V (over) | 1122 / 5012 / 4467 V (over) | 500 m: ok (×1.0) | +13.6 dB (-8.5) |
| Benthowave BII7596 | 186.6 / 179.6 / 185.6 | no, 17.4 dB short | 8 | 1995 / 4467 / 2239 V (over the rating) | 1413 / 3162 / 1585 V (over) | 1000 / 2239 / 1122 V (over) | 300 m: **fails** (×0.6) | +13.6 dB (-8.5) |

How to read it, and what it does not say:

- **Level at rated drive** uses the largest voltage (or power) the sheet rates for a duty of at least 5.6 %:
  400 V at 10 % for the T335, 750 V at 10 % for the Neptune rings and the T296, 600 W at 30 % for the TR-1075A,
  1000 W at 30 % for the TR-1075D; GeoSpectrum states one maximum voltage and no duty. **No sheet states the
  longest pulse at that drive**; Benthowave does (30 s and 200 s at full pulse power for its two rings), which
  shows a 1 s pulse is not exotic, but the question has to be put to Neptune, Massa and GeoSpectrum.
- **Massa's figures are converted from watts to volts as if the load were resistive** (V = √(P·|Z|)); the sheet
  gives the impedance magnitude across the band and the phase only where the model is tuned. Off that frequency
  the real power for a given voltage is lower, so the level at "600 W" is at least what is shown, while the
  voltage limit is unknown. At 6 kHz the TR-1075A needs 592 VA for 197 dB: at its rating.
- **Convention.** Every sheet gives the drive in V rms and a TVR per volt, so the levels computed here are rms
  levels **if** the TVR curves are rms pressure per rms volt, which is the normal practice; Benthowave says so
  in words ("SL = 20·log Vi + TVR ... Vi in Vrms"), the others do not state it. GeoSpectrum's "Maximum SPL" and
  Massa's curves carry no rms/peak label.
- **Units in phase** is 20 log₁₀ N on the weakest of the three frequencies, with mutual coupling ignored
  (an NPS model of close-spaced projector lines found the real sum 2 dB off in places [N1]).
- **Direct sound.** For a ring the figure assumes its axis is laid along the face, pointing at the rows: a ring
  does not radiate along its axis (the sheets' vertical patterns pinch to the plot's −20 dB floor there), and
  the rows of the face lie within 26° of that axis. With the far-field sin pattern applied at each hydrophone
  the worst one is 8.5 dB better off, most are 15–20 dB better off. The price is the same as a piston's: the
  horizontal transmit pattern becomes a figure of eight, −3 dB at ±45° from broadside. For the T335 the rows
  are 90° off the piston's axis, where its sheet shows −12 dB at 5 kHz and −7 dB at 3 kHz (−9.5 dB taken at
  4 kHz by interpolation). **All of these are far-field patterns used 1.7–6.4 m away with a hull beside the
  source: they are estimates of what a measurement would find, good to several dB at best.**
- **Mass.** 11.4 kg in air and 6.7 kg in water (T335); 11.3 kg in air (TR-1075; in water not stated);
  4.7 / 2.6 kg with 10 m of cable (T170); 5 kg in air (M27-450); 0.6 kg (M21 bender).

### 3.3 Cavitation: the makers' statements against the bound

*Tool* (the maker's statement scaled with depth as 20 log₁₀ of static pressure; the bound is Sec. 1.4's relation
for the wet area measured off the sheet's drawing, with no directivity index added):

| Model | Wet radiating area (from the drawing) | Maker's statement | at 5 m: maker / bound (dB) | at 10 m: maker / bound (dB) | at 30 m: maker / bound (dB) | at 100 m: maker / bound (dB) | at 200 m: maker / bound (dB) | at 500 m: maker / bound (dB) | Least depth for 197 dB (maker) | Bound above the maker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Neptune T335 | 0.018 m² | none on the sheet | — / 192.1 | — / 194.6 | — / 200.7 | — / 209.5 | — / 215.1 | — / 222.8 | — | — |
| Massa TR-1075A | 0.028 m² | 200.6 dB at 30 m | 192.0 / 193.9 | 194.5 / 196.5 | 200.6 / 202.5 | 209.4 / 211.3 | 215.0 / 216.9 | 222.7 / 224.6 | 17 m | 1.9 dB |
| Massa TR-1075D | 0.028 m² | 200.2 dB at 30 m | 191.6 / 193.9 | 194.2 / 196.5 | 200.2 / 202.5 | 209.0 / 211.3 | 214.6 / 216.9 | 222.3 / 224.6 | 18 m | 2.3 dB |
| Neptune T296 | 0.055 m² | none on the sheet | — / 196.9 | — / 199.4 | — / 205.4 | — / 214.2 | — / 219.9 | — / 227.6 | — | — |
| Neptune T170 | 0.118 m² | 197.5 dB at 10 m | 195.0 / 200.2 | 197.5 / 202.7 | 203.5 / 208.8 | 212.3 / 217.6 | 218.0 / 223.2 | 225.7 / 230.9 | 9 m | 5.2 dB |
| GeoSpectrum M27-450 | 0.094 m² | 193.0 dB at 30 m | 184.4 / 199.3 | 187.0 / 201.8 | 193.0 / 207.8 | 201.8 / 216.6 | 207.4 / 222.2 | 215.1 / 229.9 | 53 m | 14.8 dB |
| Neptune T160 | 0.177 m² | 191.5 dB at 10 m | 189.0 / 202.0 | 191.5 / 204.5 | 197.5 / 210.5 | 206.3 / 219.3 | 212.0 / 225.0 | 219.7 / 232.7 | 28 m | 13.0 dB |
| GeoSpectrum M21-070-5000 | 0.010 m² | 190.0 dB at 25 m | 182.6 / 189.5 | 185.1 / 192.0 | 191.2 / 198.1 | 200.0 / 206.9 | 205.6 / 212.5 | 213.3 / 220.2 | 68 m | 6.9 dB |
| Neptune T420 | 0.005 m² | 192.5 dB at 20 m | 186.5 / 186.5 | 189.0 / 189.0 | 195.0 / 195.1 | 203.8 / 203.9 | 209.4 / 209.5 | 217.2 / 217.2 | 40 m | 0.1 dB |
| Benthowave BII7594 | 0.052 m² | none on the sheet | — / 196.6 | — / 199.2 | — / 205.2 | — / 214.0 | — / 219.6 | — / 227.3 | — | — |
| Benthowave BII7596 | 0.097 m² | none on the sheet | — / 199.4 | — / 201.9 | — / 207.9 | — / 216.7 | — / 222.4 | — / 230.1 | — | — |

- For the pistons the bound and the maker agree within 2 dB (and a piston's DI would raise the bound further);
  for the rings the makers are 5–15 dB below it. **Use the "maker" column.**
- **At 197 dB the least depths are 9 m (T170 at 6 kHz), 17–18 m (TR-1075A / D).** Neptune states nothing for the
  T335; with the same face loading as the Massa piston it would be of the same order: assume 20 m until
  Neptune answers.
- **At the owner's 500 m every candidate has more than 15 dB of cavitation margin**; from about 30 m down the
  drive ratings, not cavitation, set the level.
- A single ring driven beyond its sheet is not an option: the M27-450's own maximum (193 dB at resonance)
  already needs 30 m.

### 3.4 What was verified, maker by maker

| Maker (host) | Read | Result for 4–6 kHz |
| --- | --- | --- |
| Neptune Sonar (`www.neptune-sonar.co.uk`) | the projector listing and all 17 projector pages; data-sheet PDFs of T335, T296, T170, T160, T420, T444 | **T335: 198.5 dB flat across 4–6 kHz at 400 V rms and 10 % duty; directional; 600 m.** T170 ring: 192.5–197.5 dB at 750 V and 10 % (minimum 10 m), unlimited depth. T296: 195–199.5 dB, 27 kg. T160 ring and T420: lower. A 2022 catalogue PDF is linked from every page; it was not opened (the per-model sheets are its pages) |
| GeoSpectrum (`geospectrum.ca`) | transducer category; ring, bender, sphere, communications and C-BASS pages; sheets of M27-450, M27-650, M21-070-5000 | **M27-450 ring: 189.6–191.6 dB at 600 V rms; 6000 m; cavitation depth 30 m at 193 dB.** A 0.6 kg bender resonant at 5 kHz (190 dB maximum, 700 m, sharp). The maker states rings and ring-based sources "from 500 Hz to 20 kHz and source levels from 190 to 220 dB" to order, and that ring elements "can be combined together to produce custom high-power underwater sources". An amplifier (M620-110) is named on the sphere page; no page for it was found |
| Massa (`www.massa.com`) | oceanographic and military pages; TR-1075, TR-1055, TR-1077A sheets; two capability PDFs | **TR-1075A / D / H: 197–202 dB across 4–6 kHz at 600–1000 W and 30 % duty** (volts inferred from watts); 600 m; about 9 m and 30 m minimum depth at 200 and 600 W; "easily assembled into arrays". TR-1077A is the deep (>6000 m) version at 22.7 kg |
| Benthowave (`www.benthowave.com`) | product index; ring, low-frequency, directional, sphere, amplifier and matching pages; sheets BII7590, BII7530, BII5060, BII5100, BII5110, BII5160 (BII7620, BII5120, BII6000 downloaded, not read in full) | rings BII7594 / BII7596: 169–187 dB with 7–13 dB of ripple across the band; rated 500 m and 300 m; the maker says the series "is not designed for underwater acoustic detection". Low-frequency series: 150 m. **Amplifiers with full sheets** (Sec. 3.6) |
| Teledyne Marine (`www.teledynemarine.com`) | sonar, hydrophones-and-transducers and sound-sources listings | no projector listed for 4–6 kHz: TC1037 starts at 6 kHz; the rest are hydrophones, echo-sounder transducers and a 200–300 Hz tomography source |
| Exail (`www.exail.com`) | home page and its product-range links | no component transducers among the product ranges linked; nothing read for this band |
| BTech Acoustics (`www.btechacoustics.com`) | products and specialty pages; two model pages | modem transducers and custom work; the model pages carry no figures in their text (embedded files); no listed multimode or 4–6 kHz projector found |
| Gavial (`www.gavial.com`) | home page | a holding-company page; the transducer catalogue is on another host (Sec. 6) |

### 3.5 Prototype sources (test AUV and static frame: 170–185 dB, 0.1 s, 4–6 kHz)

Volts for the level from the same TVR readings (`drive()` of the requirements tool); every one is within the
unit's continuous (100 %) or 10 % rating unless marked:

| Unit | 175 dB at 4 / 5 / 6 kHz | 185 dB at 4 / 5 / 6 kHz | Mass in air | Depth | Remark |
| --- | --- | --- | --- | --- | --- |
| Neptune T335 | 27 / 27 / 27 V | 84 / 84 / 84 V | 11.4 kg | 600 m | flat; rated 100 V rms continuous; the same unit serves option A |
| Massa TR-1075A | 13 / 18 / 19 V | 42 / 56 / 60 V | 11.3 kg | 600 m | the same unit serves option A |
| Neptune T170 | 100 / 63 / 56 V | 316 / 200 / 178 V | 4.7 kg with cable | unlimited | 250 V continuous, 750 V at 10 %; the same unit serves options B and C |
| GeoSpectrum M27-450 | 89 / 100 / 112 V | 282 / 316 / 355 V | 5 kg | 6000 m | within 600 V |
| GeoSpectrum M21-070-5000 | 282 / 84 / 158 V | 891 (over) / 266 / 501 V | 0.6 kg | 700 m | the lightest; 185 dB only above about 4.4 kHz; 175 dB everywhere |
| Neptune T420 | 1000 / 266 / 355 V | over at 4 kHz | 3.8 kg | 600 m | needs kilovolts at 4 kHz |
| Benthowave BII7594 | 178 / 794 (over) / 708 V (over) | over | 1.2 kg | 500 m | 13 dB of ripple: not suited to a sweep |

Cavitation is no issue at these levels at any depth for the rings and pistons (the T170's limit scales to
195 dB at 5 m); the bender's limit at 5 m is 182.6 dB on its maker's statement. **The prototype does not need a
different product: the option-A or option-C unit, driven at 15–300 V by a small amplifier, is the prototype
source**, and it removes one integration step later. The 0.6 kg bender is the fallback if the test AUV cannot
carry 5–11 kg. Underwater loudspeakers could not be read (their hosts redirect to refused addresses, Sec. 6).

### 3.6 Power amplifiers

Read: Benthowave's amplifier page and the sheets BII5060, BII5100, BII5110, BII5160
(`https://www.benthowave.com/products/Specs/BII5060DataSheet.pdf` and siblings). The two other amplifier makers'
hosts redirect to refused addresses (Sec. 6).

| Amplifier | Kind | Output | Band | Supply | Size, mass | Idle | Note from the sheet |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BII5060 series | linear, arbitrary waveform | 315 W rms at +48 V DC (158 W versions) | 0.1–70 kHz (−3 dB) | up to +60 V DC; 14.2 A at maximum power (7.1 A versions) | boards of 0.2–0.4 kg; enclosed versions 1.5–2.2 kg, about 254 × 148 × 94 mm | 59–104 mA | output voltage "≤ 2·Vs − 16" V peak-to-peak, so a step-up transformer (their BII6000) is needed for hundreds of volts; efficiency into a tuned load 64 % at 48 V, times cos θ into an untuned one |
| BII5160 | linear, for low-frequency transducers, with step-up | 208 W rms | 100 Hz–20 kHz | up to +60 V DC | portable case 0.28 × 0.26 × 0.18 m, 5.0–5.2 kg | 59 mA active, 24 mA shut down | output up to "(2·Vs − 10)·14.5" V peak-to-peak: 441 V rms at 48 V |
| BII5100 | switching (square or pulse input) | 309 W at +56 V, 261 W at +48 V | 1–500 kHz | +10.5 to +56 V DC | 0.16 kg board; 0.8–1 kg enclosed, 181 × 110 × 75 mm | 13 mA | 91 % efficient at 48 V; "the transducer should be tuned ... as resistive load"; not for arbitrary waveforms |
| BII5110 | switching, small | 108 or 216 W | 1 kHz–10 MHz | +30 V DC | 25 g to 0.7 kg | 1–2 mA | tuned loads only; duty ≤ 25–50 % |

Against the candidates (requirement from Sec. 3.2):

- **T335 at 197 dB**: 335 V rms, 247–337 W real and 253–368 VA across the band. That is just beyond one 315 W
  linear unit at the band edges, on a load that is not resistive there (the sheet's susceptance is −2 mS at
  4 kHz against 2.6 mS of conductance). **At 194 dB it needs 124–169 W: inside one BII5060 or one BII5160**, and
  the BII5160's 441 V rms covers the voltage. A T335 on one BII5160 at its 208 W gives about 195 dB.
- **TR-1075A at 197 dB**: 168–237 V rms and 264–592 VA: a 600 VA amplifier; its 100 Ω class impedance suits a
  low-voltage stage with a modest step-up.
- **A pair of T170 at 191 dB each**: 355–631 V rms and, per ring, 151–279 W real but 232–768 VA untuned (the
  rings are capacitive: 68 nF): tuning inductors are needed, then about 300 W per ring.
- **Prototype**: any of the above at 1–30 W.
- So the amplifier for the full level is **0.4–0.6 kVA at 250–350 V rms (pistons) or two channels of about
  300 W at up to 630 V rms (ring pair)**; no sheet read offers that in one unit, two BII5060-class units do,
  and the makers whose pages could not be reached remain to be checked. Standby draw of the units read is
  1–5 W, small against the 16–30 Wh per hour of transmitting.

### 3.7 Figures from reports and theses (first pass, kept for the technology they document)

| Device | Figures and where they are stated | Use here |
| --- | --- | --- |
| Custom 15-element flexural-disc line array (Applied Physical Sciences) | resonance 3.8 kHz; 181.1–191.1 dB re 1 µPa at 1 m per element, 210.3 dB combined ("RMS"); 0.152 m spacing; about 200 V AC drive; 1.4 s signals every 25 s; hand-deployed [N8, pp. 38–40] | existence proof for a line of small elements; GeoSpectrum's 5 kHz bender is a catalogue element of the same kind |
| Sparton ring-shell (class V flextensional) | 610 Hz; 213 dB at 3000 V rms; 90 % efficient at resonance, 65 % at −6 dB; "exceeds 400 m" compensated [N1, N2] | what a compensated flextensional does, far below the band |
| Free-flooded ring, 0.367 m | 1.0 and 2.5 kHz; 132 dB re 1 µPa/V; depth "unlimited" [N3] | scaling: confirmed by the T170 (6 kHz ring, 0.191 m across) and the M27-450 (4.25 kHz, 0.15 m body) |
| Sensor Technology SX30 ring | 20–40 kHz; 133 dB re 1 µPa/V; 300 W at 2 % duty; stable to 440 bar [A3] | duty and depth behaviour of the family |
| USRD G34 | advertised 0.2–5 kHz, used 1–8 kHz; TVR only as a figure [N7] | a loan projector for calibration |

### 3.8 Still to check (no numbers given)

| Maker | Why unread | What to look for |
| --- | --- | --- |
| Gavial ITC | catalogue on `gavialitc.com`, not allowed | ring, sphere and flextensional projectors for 3–8 kHz |
| Sensor Technology | `www.sensortechcanada.com` redirects to `sensortechcanada.com`, not allowed | larger SX-series free-flooded rings |
| Ultra | `www.ultra.group` redirects to `www.cobhamultra.com`, not allowed | the free-flooded ring data sheet |
| Lubell Labs; Ocean Engineering Enterprise | redirect to `www.lubell.com`; to `oceanears.com`, not allowed | underwater loudspeakers for the prototype |
| Instruments Inc.; AE Techron | redirect to `instrumentsinc.com`; to `aetechron.com`, not allowed | 0.5–1 kVA sonar amplifiers on a DC bus |
| Neptune Sonar | not opened | the 2022 catalogue PDF (accessories, arrays of T335); "T400" and array-mounting accessories |
| GeoSpectrum | not found on the site | the M620-110 amplifier; custom ring sources for 197–200 dB |
| Massa | not read | the TR-1077A curves; a TR-1075 TVR in arrays |
| Kongsberg, Thales, Atlas Elektronik, L3Harris, Applied Physical Sciences | hosts not allowed or not tried | system-level transducers; the flexural-disc line array |

## 4. Recommendation: a decision table

| | **A. One source at full level** | **B. Lower level, longer pulse** | **C. A pair (or line) of sources** | **D. The prototype, now** |
| --- | --- | --- | --- | --- |
| What it is | one directional piston at 197 dB rms across 4–6 kHz, 1 s | the same unit, or one ring, at 194 dB with 2 s or 191 dB with 4 s (HFM, or a sweep with a Doppler bank) | two free-flooded rings in phase, half a wavelength apart, axes along the face; or two pistons | the unit of A or C at 170–185 dB, 0.1 s, on the test AUV and the static frame |
| **Named candidates (by their sheets)** | **Neptune T335** (198.5 dB flat at 400 V rms, rated 10 % duty); **Massa TR-1075A** (197.1–200.6 dB at 600 W, 30 %) or **TR-1075D** (about 202 dB at 1000 W, 30 %); Neptune T296 (195–199.5 dB, 27 kg) falls 2 dB short at 6 kHz | T335 at 237 or 168 V rms; TR-1075A at 119–168 V or 84–119 V; **one Neptune T170** reaches 191 dB across the band at 355–631 V rms but a 4 s pulse is 19 % duty against its 10 % rating at 750 V | **two Neptune T170** at 191 dB each (355–631 V rms, inside 750 V at 10 %); two GeoSpectrum M27-450 give 195.6–197.6 dB at their 600 V limit; three give 199 dB | T335 at 27–84 V rms; TR-1075A at 13–60 V; T170 at 56–316 V; GeoSpectrum M21-070-5000 bender (0.6 kg) for 175 dB |
| Depth rating against the owner's 500 m | 600 m (T335, TR-1075): passes with a factor 1.2; response change with depth not on the sheets | as A; T170 unlimited | T170 unlimited, M27-450 6000 m: pass, and the type whose response does not change with depth | as the unit chosen |
| Link budget (noise-limited, ship at 12 km, 1 / 2 / 3 rows) | +1.6 / +4.0 / +5.4 dB of margin on the piston's axis; 3 dB less at about ±45° (sheet: "conical 90°" at 5 kHz, 80° for the Massa) | the same (same energy), if the Doppler processing holds its loss under 1 dB | the same at broadside; the pair along the face narrows the sector to about ±30° at −3 dB, and each ring's own figure-of-eight to ±45° | 2.0–4.4 km on the ship, 0.7–1.5 km on the small target with one short row |
| Blind range | 750 m | 1.5 km (2 s) or 3 km (4 s) | 750 m | 75 m |
| Direct sound left (nearest hydrophone, rms convention, face placeholder) | about +12.6 dB with the T335 facing out (its sheet: −9.5 to −14 dB at 90°); +22 dB with the TR-1075 until its side response is known | 3 or 6 dB less than A | about +5 to +14 dB: the rings' axial null (−8.5 dB at the worst hydrophone) and the pair's null (−8.4 dB at the worst frequency) do not simply add in the near field; not computed jointly | at 175 dB: +15 dB at 0.30 m, 0 dB at 1.7 m, before any help from the unit's pattern |
| Cavitation (makers' statements) | 17–18 m for the TR-1075 at 197 dB; not stated for the T335 (assume about 20 m); **not usable at full level on the static frame at 10 m** | TR-1075A: 194 dB from about 10 m, 191 dB from about 4 m | each T170 at 191 dB: from the surface (its limit is 195 dB at 5 m); each M27-450 at 191 dB: from about 22 m | none |
| Power | T335: 247–337 W into the transducer (from its conductance), about 290–400 W from an 85 % amplifier, **16–22 Wh per hour**; TR-1075A: up to 592 VA | half or a quarter of A in the pulse; about the same per hour | two rings: 151–279 W each real (300–560 W together); tuning inductors needed | 1–30 W |
| Drive voltage | 335 V rms (T335); 168–237 V rms (TR-1075A) | 119–237 V rms | 355–631 V rms per ring | 13–320 V rms |
| Size and mass | T335: 183 mm square × 257 mm, 11.4 kg in air, 6.7 kg in water; TR-1075: 184 mm square × about 220 mm, 11.3 kg in air | as A, or one ring of 4.7 kg | two rings 191 mm across × 85 mm, 4.7 kg each in air and 2.6 kg in water with cable, on a 0.15 m spacing bar | as the unit chosen; 0.6 kg for the bender |
| Redundancy | none | none | 6 dB less with one ring out | — |
| Risk | pulse length at rated drive not on the sheets; piston response against depth not on the sheets; cavitation depth of the T335 not stated; Massa's volts inferred from watts | Doppler loss and the larger bank; blind range; a ring's duty rating between 10 % and 100 % is not given | mutual coupling of two close rings (each sees the other's field; the makers say rings "can be combined" but give no array data); narrower sector; more wet connections; cavity-mode detuning by the mounting | little |
| What the sheets leave unknown | Sec. 5, per unit | the same | the same, plus array data | the same |

**How to choose, with the data sheets in hand.**

1. **A single listed unit does meet 197 dB rms across 4–6 kHz at 5.6 % duty according to its own sheet**: the
   Neptune T335 (with 1.5 dB to spare at its 10 % rating) and the Massa TR-1075A / D (0–5 dB to spare at their
   30 % rating). Option A is therefore real, at about 11 kg, 600 m, and a transmit sector of about ±45°. It
   was the unknown of the first pass.
2. **For a vehicle that goes to 500 m, the pistons pass by a factor 1.2 and the rings by a wide margin.** If the
   vehicle team wants the type whose response does not depend on depth and half the mass, the pair of T170
   rings (option C) reaches the same level at the same duty rating, is free of cavitation from the surface at
   191 dB per ring, and puts two nulls towards the rows.
3. **Do D first, with the unit of A or C.** A T335 or a T170 at tens to a few hundred volts on a 200 W
   amplifier is the prototype source; the same hardware then runs option B (a 2–4 s HFM at 191–194 dB) before
   the full-level amplifier exists. One BII5160-class amplifier with a T335 gives about 195 dB: two rows
   already close the 12 km budget at 194 dB (+1.0 dB of margin).
4. **The direct sound is still not solved by the projector alone**: the best case on these sheets is about
   +5 to +13 dB over the inputs' full scale at the face's placeholder. The front-end measures of
   `docs/DESIGN.md` stay necessary; the projector choice reduces what they must absorb from 22 dB to about 10 dB.
5. **Near-surface full-level operation is not available from any candidate** (9–18 m minimum by the makers'
   statements): the static frame at 10 m should plan on 191–194 dB, or hang the projector deeper.
6. Between the T335 and the TR-1075A: the T335's sheet shows the flatter response, a measured side response and
   the weight in water; the TR-1075's shows the higher duty rating (30 %), states its cavitation depths, has the
   lower drive voltage, and comes in tuning variants (the D model trades efficiency for a lower Q). Both makers
   describe the unit as an array element, so two of either is also option C with pistons.

## 5. Questions for the manufacturer (RFQ checklist)

State: 4–6 kHz linear or hyperbolic FM, 1 s (alternatively 2–4 s), repeated every 18 s, source level 197 dB re
1 µPa at 1 m rms (alternatively 191–194 dB; alternatively 185–191 dB per element for an array of 2–4), operating
depth 5–500 m (500 m is the vehicle's maximum, owner-stated), on a battery-powered vehicle.

1. **TVR curve** 3–7 kHz (dB re 1 µPa/V at 1 m), and whether its levels are rms or peak.
2. **Maximum source level at 4, 5 and 6 kHz** separately, with what limits each (voltage, current, stress, heat,
   cavitation), for a 1 s pulse and for a 4 s pulse.
3. **Maximum drive against depth**: the TVR and the limits at 5, 10, 30, 100, 250, 500 m; depth rating
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

**Gaps in each candidate's own sheet** (ask these first):

| Unit | Not on its sheet |
| --- | --- |
| Neptune T335 | longest pulse at 400 V rms and at 335 V rms (1 s? 4 s?); duty between the 10 % and 100 % points; **minimum depth for cavitation**; TVR, impedance and limits at 100, 250 and 500 m; whether levels are rms; efficiency; directivity index; side and rear response at 4 and 6 kHz (the pattern is plotted at 3, 5, 7 kHz); the tuning network's behaviour with a long cable; array data at half-wavelength spacing; whether 600 m is an operating or a survival figure; price and lead time |
| Massa TR-1075A / D / H | maximum voltage and the impedance phase across 4–6 kHz (only |Z| is plotted); longest pulse at 600 W; rms or peak; **beam pattern and side response** (only "80° conical"); weight in water; response against depth to 500 m; efficiency by model; which tuning suits a flat 4–6 kHz band (the standard models are tuned at 3.5–4 kHz); price and lead time |
| Neptune T170 | duty between 10 % (750 V) and 100 % (250 V) and the longest pulse; cavitation depth below 750 V (only "minimum 10 m" at the maximum); horizontal pattern of a pair; mounting clearances for the bore; tuning inductor; ring weight without the cable; a matched pair's TVR tolerance |
| GeoSpectrum M27-450 | duty and pulse length at 600 V rms; weight in water; cavitation depth at levels below 193 dB; rms or peak; data for two or three rings combined, which the maker offers as custom sources |
| GeoSpectrum M21-070-5000 | duty and pulse length; **TVR and resonance against depth to 500 m** (a bender); efficiency; array data |
| Benthowave amplifiers | the BII5060's behaviour into a load that is reactive at the band edges; a transformer for 335–630 V rms at 0.3–0.6 kVA; pressure-housing suitability; conducted noise while idle |

## 6. What could not be verified, and hosts that were refused

**Not verified:** anything beyond what the sheets print: no maker was contacted, so every gap of Sec. 5 is open;
the curves were read by eye; Massa's volts are inferred from watts; the far-field patterns were applied in the
near field; mutual coupling in a pair is not modelled; Urick's and Sherman and Butler's own statements of the
cavitation limit were not opened (the relation is derived here and is now checked against four makers' own
cavitation depths, Sec. 3.3); the transducer-family statements marked *(not verified here)* in Sec. 2; the
baffle option.

**Hosts allowed by the owner for the second pass and what happened** (one request each; a redirect to a host
that is not allowed was not followed further):

| Host allowed | Result |
| --- | --- |
| `www.neptune-sonar.co.uk`, `geospectrum.ca`, `www.massa.com`, `www.benthowave.com` | read; data sheets served from the same hosts |
| `www.teledynemarine.com`, `www.exail.com`, `www.btechacoustics.com` | read; nothing for this band |
| `www.gavial.com` | read; its transducer catalogue is on **`gavialitc.com`** (refused: not on the list) |
| `www.sensortechcanada.com` | redirects to **`sensortechcanada.com`** (refused) |
| `www.ultra.group` | redirects to **`www.cobhamultra.com`** (refused) |
| `lubell.com` | redirects to **`www.lubell.com`** (refused) |
| `www.oceanears.com` | redirects to **`oceanears.com`** (refused) |
| `www.instrumentsinc.com` | redirects to **`instrumentsinc.com`** (refused) |
| `www.aetechron.com` | redirects to **`aetechron.com`** (refused) |

To finish Sec. 3.8, allow the seven hosts in bold. No data sheet read sat behind a form or a login; Neptune's
pages carry a "Request Quote" button and GeoSpectrum's an enquiry cart, neither of which was used.

**Hosts refused in the first pass and still not allowed** (what was wanted from each):

| Host | What was wanted |
| --- | --- |
| `www.ultra-hyperspike.com` | the free-flooded ring data sheet (a PDF titled "free flooded rings datasheet") |
| `www.oceansciencetechnology.com` | a reseller page for free-flooded rings |
| `www.itc-transducers.com` | ITC projector data sheets for 3–8 kHz |
| `www.ixblue.com`, `www.kongsberg.com`, `www.thalesgroup.com`, `www.l3harris.com`, `www.airmar.com`, `www.rtsys.eu`, `www.evologics.de`, `www.chelsea.co.uk`, `www.hydroacoustics.com`, `www.sercel.com` | projector catalogues |
| `www.electrovoice.com`, `www.aquarianaudio.com` | underwater loudspeaker data sheets for the prototype |
| `apps.dtic.mil` (reachable; the site answered 403 to this client), `discover.dtic.mil` | AD0743199 (ARL-TR-71-51, restates Urick's cavitation limit), ADA459091 and ADA233406 (low-frequency projector surveys) |
| `www.ioa.org.uk` (reachable; the document addresses returned "not found") | Institute of Acoustics proceedings papers on high-power free-flooded rings, class IV flextensional limits and cavitation in sonar |
| `files01.core.ac.uk`, `pdf.core.ac.uk` | a thesis on the design of free-flooded ring transducers |
| `theses.hal.science`, `publications.tno.nl`, `cradpdf.drdc-rddc.gc.ca`, `www.dst.defence.gov.au` | theses and laboratory reports on flextensional and ring projectors and on continuous active sonar |
| `image-ppubs.uspto.gov`, `patents.google.com` | patents returned by searches |
| `dosits.org`, `rdrr.io`, `en.wikipedia.org`, `www.ncbi.nlm.nih.gov`, `www.researchgate.net`, `www.semanticscholar.org`, `www.sciencedirect.com`, `archive.org`, `www.osti.gov`, `www.npl.co.uk`, `www.nrl.navy.mil` and other publisher and library hosts | background on cavitation limits and transducer families; open copies of papers |

Also: `link.springer.com` was reachable but returned no article text to this client; `repository.tudelft.nl`
was reachable but the two records searched for were not found at the addresses returned.

## 7. Sources opened in this session

Manufacturers' data sheets and pages (opened 2026-11-18; copies were kept outside the repository and are not
committed):

- Neptune Sonar: `https://www.neptune-sonar.co.uk/products/projectors` and the 17 model pages under it; PDFs
  `.../downloads/3aced3db-b5b4-4f6e-8dd1-d0ed3e1bcbd6/T335.pdf`, `.../8682c070-8807-4d6c-bc6c-a37dcb3d849e/T296.pdf`,
  `.../b76a71d6-dd89-48b1-8026-63b45c5761d3/T170.pdf`, `.../8df0fff1-08ad-40c2-a577-90a755c087ca/T160.pdf`,
  `.../fbeb4025-8243-4084-b976-7a59740bfbfa/T420.pdf`, `.../32ad6dd7-c2e7-45f2-a1f5-c2eba446b5ea/T444.pdf`
  (all under `https://www.neptune-sonar.co.uk/downloads/`).
- GeoSpectrum: `https://geospectrum.ca/product/ffr-free-flooded-rings/`, `.../product/benders/`,
  `.../product/spherical-transducers/`, `.../product/comms/`, `.../product/c-bass/`; PDFs
  `https://geospectrum.ca/wp-content/uploads/2022/05/SP-M27-450-R0.pdf`, `.../SP-M27-650-R0.pdf`,
  `.../SP-M21-070-5000-R1.pdf`.
- Massa: `https://www.massa.com/industrial/oceanographic/`, `https://www.massa.com/military/`; PDFs
  `https://www.massa.com/wp-content/uploads/2023/03/TR-1075-Datasheet-230327.pdf`,
  `.../2023/03/TR-1055-Datasheet-230327.pdf`, `.../2019/06/TR-1077A-Specification-190612.pdf`.
- Benthowave: `https://www.benthowave.com/products/BII-7590freefloodedtransducer.html`,
  `.../BII-7530lowfrequencytransducer.html`, `.../BII-7520omnidirectionaltransducer.html`,
  `.../poweramplifier.html`; PDFs `https://www.benthowave.com/products/Specs/BII7590Datasheet.pdf`,
  `.../BII7530Datasheet.pdf`, `.../BII5060DataSheet.pdf`, `.../BII5100DataSheet.pdf`, `.../BII5110DataSheet.pdf`,
  `.../BII5160Datasheet.pdf`.
- Teledyne Marine: `https://www.teledynemarine.com/products/product-line/hydrophones-and-transducers`,
  `.../product-line/sonar/singlebeam-and-transducers`, `.../product-line/acoustic-sound-sources`.
- BTech Acoustics: `https://www.btechacoustics.com/products`, `.../products/specialty-products`.
  Exail: `https://www.exail.com/`. Gavial: `https://www.gavial.com/`.

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
| `model`, `amplifier` | `Neptune T335` or `Massa TR-1075A` once chosen (candidates, not a choice) | the same, or `Neptune T170` | `2 x Neptune T170` | the unit of A or C |
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
| `depth_rating_m` | m | the data sheet's rated operating depth | a check against the platform's maximum depth: **500 m, owner-stated** (a `platform.max_depth_m` field would carry it; the hydrophones', cables' and connectors' ratings need the same check) and against `trial.own_depth_m` |
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
5. *Depth.* Error when the platform's maximum depth (500 m, owner-stated) exceeds `depth_rating_m`; warn when
   `trial.own_depth_m` does. The same for `hydrophone.depth_rating_m` and the connectors.
   With a maker's cavitation statement (`cavitation_level_db`, `cavitation_depth_m`) use it in check 1 instead
   of the area: limit(h) = level + 20 log₁₀(p_c(h) / p_c(depth)).
6. *Doppler.* For `type: lfm`, warn when c / (2 B T) is under the platform's speed (`trial.speed_m_s`): 0.38 m/s
   for 1 s and 2 kHz against 1.5 m/s on the face builds today. The chain needs a Doppler bank or an HFM pulse.
7. *Energy.* Print acoustic power, energy per ping and per hour for an assumed or stated efficiency in
   `describe` (`energy()` in the tool), so the parts list's power line stops saying "no draw is known" for
   the transmit chain's part of it.

## Appendix B. Reproducing the numbers

    PYTHONPATH=python python3 tools/projector_requirements.py                      # all builds; writes docs/design/projector_requirements.json
    PYTHONPATH=python python3 tools/projector_requirements.py popoto_face_3x32
    PYTHONPATH=python python3 tools/projector_requirements.py popoto_testauv_1x32 --levels 160 165 170 175 180 185 190 197 --ranges 500 1000 2000 --no-json
    PYTHONPATH=python python3 tools/projector_candidates.py                        # Sec. 3 tables; writes docs/design/projector_candidates.json

The tool reads the build spec, so a changed source level, band, pulse, projector position, hydrophone
sensitivity or gain changes every table here. Its link table is `buildspec.performance` with the level and the
row count replaced; its direct-sound figures are `buildspec.describe`'s plus 3 dB for the rms convention; the
cavitation, power, pulse-trade, flatness and line-array arithmetic is its own and is stated in its docstring.
