# The short-range mode of the long-range sonar

2026-11-28, branch `shortmode` (issue `sonar-9ed5`). Opt-in; no default, record field or FPGA code changed. Every number
here is **ours**: the suite's link budget and depth planner (`gpusonar.buildspec`, `gpusonar.depthplan`, the functions the
trade study and POP-3 use), the suite's own hyperbolic sweep for the cross-talk, arithmetic for the timing, and the simulator
and both live-chain engines for the scenes. Nothing was measured on water. The figures are in
`docs/design/short_range_mode.json`, made by `python3 tools/short_range_study.py --json docs/design/short_range_mode.json`
(Bellhop for the three sites with a profile).

## What is wrong, uncertain or not built (read first)

1. **The simulator makes each ping on its own.** One ping's echoes arriving in the next ping's window are not modelled.
   With the mode, the short ping is sent when the long ping's window closes. Its echoes from beyond its own window, from
   3 km out to the long range, then arrive during the next long ping's window. The scenes cannot show this. Section 2(a)
   estimates it from the model: under the long ping's own background where that background is noise. The same is true
   today of the long ping's echoes from beyond 12.7 km.
2. **Two hardware facts are not known** (owner questions, section 5). Can the projector and its drive board change the
   pulse length from one ping to the next? How soon after a transmission can the receiver listen? The
   `projector.blanking_after_s` of a build spec is null. The mode needs both, and so does the baseline.
3. **The tracker counts a ping that cannot see a track as a miss.** One short ping after every long one works: a track
   misses one ping, then is seen again, and `max_misses` is 2. With `every` of 3 or more, a track seen only by the short
   pings is dropped between them. Not changed; a follow-up. The coverage picture (`chain.coverage`) takes its detection
   table from the first ping only. With the mode that is the long pulse's table. Not changed; a follow-up.
4. **For the smallest class the overlap is thin.** `xl_auv` at its low +5 dB is held by the 0.25 s pulse at 1.5 km at every
   site, by +0.8 to +19.4 dB. It is not held at 2.25 km except at the deep site at sea state 3. The long pulse holds it at
   2.25 km by only +0.1 to +1.5 dB on the shelf, slope and winter sites, and not at all in summer mud. For that class the
   two modes barely meet at the hand-over. For the ship and the large submerged class the overlap is wide (section 1).
5. **The two kinds read a mover's range differently.** An up-sweep's range bias is f_stop T / B: 6 m per m/s for the 2 s
   pulse and 0.75 m per m/s for the 0.25 s pulse (`gpusonar.doppler.hfm_range_factor`). A mover at 4 m/s therefore jumps
   by about 21 m between the two kinds. That is inside the tracker's gate (derived range sigma 15 m, gate 4 sigma), and on
   the scene it stays one track. It adds scatter to the track's velocity; the classifier's speed estimate was not
   measured for it.
6. **Not built:** a build-spec key for the mode (`projector.short_range`) and its generated files. Today the mode is a block
   in the sonar file, or in a scene's `sonar_overrides`. Also not built: the simulator's survey legs, raster and motion
   runs with the mode (refused by name), and a GPU run.

## 1. What the short pulse holds (the planner; one row of 32 unless said; each class at its low target strength)

The long pulse is the D-256 baseline: a 2 s hyperbolic sweep, 4-6 kHz, 197 dB rms, 200.0 dB re 1 µPa² s, blind inside
c T / 2 = 1500 m. The short pulses are the same sweep at the same level, cut to 0.25 s or 0.1 s:

| Pulse | Energy at 1 m | Less than the long pulse | Blind inside | Window to | Ping interval |
| --- | --- | --- | --- | --- | --- |
| 2 s (baseline) | 200.0 dB | | 1500 m | 12.7 km | 18.93 s |
| 0.25 s | 191.0 dB | 9.0 dB | 187.5 m | 3 km | 4.25 s |
| 0.1 s | 187.0 dB | 13.0 dB | 75 m | 3 km | 4.10 s |

Margin over the detector's need at 1.5 km, the long pulse's blind range: 0.25 s / 0.1 s, in dB. The 2 s column is the long
pulse at 2.25 km:

| Site | ship | large_submerged | xl_auv | xl_auv at 2.25 km: 0.25 s / long 2 s |
| --- | --- | --- | --- | --- |
| deep, no bottom return | +40.4 / +36.4 | +36.9 / +32.9 | +19.4 / +15.4 | +3.8 / +21.8 |
| deep, sea state 5 | +26.1 / +22.1 | +25.9 / +22.0 | +5.7 / +1.9 | -1.9 / +6.6 |
| shelf 100 m sand | +26.9 / +26.8 | +23.5 / +23.4 | +3.3 / +3.2 | -0.5 / +1.5 |
| slope 1000 m silt | +20.5 / +20.3 | +20.7 / +20.5 | +0.7 / +0.5 | -0.9 / +0.1 |
| winter 200 m sand | +25.1 / +25.0 | +23.7 / +23.6 | +3.7 / +3.6 | -0.3 / +1.1 |
| summer 60 m mud | +30.4 / +26.4 | +38.8 / +35.5 | +0.8 / **-2.5** | -35.5 / -9.0 |

- With 0.25 s, the ship and the large submerged class are held from the short pulse's blind range to 3 km at every site.
  There are two exceptions: summer mud's ship at 3 km (+3.6 dB, a shadow), and the slope site at 200 to 700 m at the own
  depth the plan picks for 1.5 km. So the short mode overlaps the long mode's blind range by a factor of two for those
  classes.
- Where reverberation limits (shelf, slope, winter), the 0.25 s and 0.1 s pulses give the same margin within 0.4 dB. For a
  swept pulse of one band, echo over reverberation does not depend on the pulse length. Where noise limits (the deep site,
  summer), 0.25 s is 4 dB better. **0.25 s is the length chosen**: it holds `xl_auv` at 1.5 km in summer mud, where 0.1 s
  does not, and its 187.5 m blind range is near enough.
- Three rows add about 4 dB where noise limits and nothing where reverberation limits (bound A of the planner).

## 2. The options, compared

### (a) Alternating pings, each in its own slot (recommended, built)

A long ping. When its window closes, a short ping. When that window closes, the next long ping. The cycle is 18.93 + 4.25 =
23.2 s (23.3 s at the scene's 1497 m/s). Against 18.93 s for the long mode alone, the long mode keeps **82 %** of its update
rate (a cost of 18 %). The short mode updates every 23.2 s as well. With one short ping after every three long ones,
the long mode keeps 93 %, but tracks seen only by the short pings are lost between them (item 3 above).

What it does not cost: the chain needs no new map, size or setting. Both pulses sweep the same band, so the range
resolution c / 2B (0.375 m) and every size the chain derives from it are the same. Each ping is mapped over its own window
and tested with a detector planned for its own pulse. One tracker takes both kinds.

What it may cost and the scenes cannot show (item 1): the short ping's echoes from beyond 3 km during the next long ping's
window. At the long range R they carry the short ping's reverberation from about R - 6 km, 9 dB down in energy. The model
was run only for the opposite case, a short ping inside the long window (next option). Where the long ping is
reverberation-limited at near range, this needs counting before the mode is trusted on a reverberating site.

### (a') The short ping sent inside the long ping's listening time

Here the short ping is sent t_s seconds after the long one. Its transmission blanks the long receiver for 0.25 s. Every
long echo from c (t_s - 2) / 2 to c (t_s + 0.25) / 2 loses up to 0.25 / 2 of its energy (0.58 dB), a 1.7 km band of
ranges. That part is mild. The real cost is the short ping's own reverberation, received through the long filter. For one
band, a diffuse field passes another pulse's matched filter with exactly its energy ratio (crosstalk below). Shelf site,
one row, t_s after the long ping starts:

| t_s | long echoes partly blanked | long background raised > 3 dB | > 1 dB |
| --- | --- | --- | --- |
| 2 s (right after the long pulse) | 1500-1688 m | 1.6-2.75 km | 1.6-7.2 km |
| 4 s | 1500-3188 m | 3.1-6.55 km | 3.1-10.15 km |
| 8 s | 4500-6188 m | 6.1-11.4 km | 6.1-12.7 km |
| 12 s | 7500-9188 m | 9.1-12.7 km | 9.1-12.7 km |

So the notch is several kilometres wide, not 190 m. It can be moved from ping to ping, but on a reverberating site it costs
more of the long range than the slot of option (a) costs in time. Not chosen.

### (b) One ping, two pulses (long then short, the same band, an up- and a down-sweep)

Through the other pulse's matched filter (`tools/short_range_study.py crosstalk`, the 2 s up-sweep against the 0.25 s
sweep):

| | short up | short down |
| --- | --- | --- |
| a point echo of the long pulse in the short filter (peak, over the short pulse's own) | -23.5 dB | -24.7 dB |
| a point echo of the short pulse in the long filter | -41.6 dB | -42.8 dB |
| the long pulse's reverberation in the short filter (mean, over the short's own) | **+9.0 dB** | **+9.0 dB** |
| the short pulse's reverberation in the long filter | -9.0 dB | -9.0 dB |

A point echo is rejected by about the time-bandwidth product, 24 to 43 dB, for either slope. Reverberation is not rejected
at all: two pulses of one band share their spectrum, so the mean power is the energy ratio, whatever the sweep's direction.
Near echoes and near reverberation are tens of dB stronger than far ones: 34.5 dB from 1 km to 6 km, 50.5 dB from 1 km to
12 km (spreading and absorption). So the short pulse's near reverberation, 9 dB down in the long filter, lies 20 to 40 dB
over the long pulse's far echoes for the seconds after it is sent (the t_s = 2 s row above). The long pulse's reverberation
raises the short channel's background by 9 dB where reverberation limits it. Sub-bands (for example long 4-5 kHz, short
5-6 kHz) would separate the two pulses, but that halves the long pulse's band, doubles its range cell and changes the
baseline the owner chose on measurements (D-256). Not chosen.

### (c) A selectable mode

The operator or the mission picks the long pulse or the short pulse; nothing is interleaved. This needs no code: a sonar or
deployment file with `transmit.waveform.duration: 0.25` and a near window (today's scenes do it by `sonar_overrides`).
Simplest of all, but it never gives both pictures at once: a vehicle on the long mode stays blind inside 1.5 km.

### (d) Listening as soon as the long pulse ends (partial echoes)

A target at r inside c T / 2 returns its echo over [2r/c, 2r/c + T]. The part after the transmission ends is a fraction
2r / (cT) of the echo: at 800 m 53 % (-2.7 dB), at 500 m -4.8 dB, at 300 m -7.0 dB. It is the top of the sweep, so only
63 % of the band is left at 800 m (about 4.74 to 6 kHz) and 27 % at 300 m, which coarsens the range cell. It needs no
second ping. It does need the receiver to recover from its own 2 s blast at once, and the chain to map ranges whose echoes
began before the window opened. Today the chain refuses this on purpose: blanking keeps it from reporting nearer than whole
echoes. Promising, but it rests on two unknowns and changes the chain's map. Not built; listed as a follow-up.

### Practice elsewhere

`docs/REFERENCES.md` holds nothing on how sonars of this class schedule pulse lengths. A generic web search (generic terms
only) found radar practice, not sonar: a marine radar manual that ties pulse length to the range scale, and a patent that
sets the transmitted pulse length by the search range. Neither was opened beyond its summary, so they are **unconfirmed**
and are not cited as evidence.

### Recommendation

**(a): one short ping (the same sweep, 0.25 s) after every long one, in its own slot.** It is simple to explain: "every
cycle the sonar sends its long pulse for 12 km, then a short one for the first 3 km". It costs 18 % of the long mode's
update rate, and the short pulse has 9 dB less energy, which it can spare inside 3 km for the ship and large submerged
classes. It needs nothing new from the record or the chain's sizes.

## 3. What was built (opt-in)

- **The sonar's `short_range` block** (`docs/CONFIGURATION.md`): `waveform` (laid over `transmit.waveform`; normally only
  `duration`), `window`, `every`, `interval_s`. The pieces are `gpusonar.sonar` `short_range_block`, `short_range_config`
  (the short pings' sonar: everything else is the sonar's own), `ping_schedule` and `window_end_s`.
- **The simulator** (`sim/simulator.py run_scenario`, the straight run): each ping takes its kind and time from the
  schedule. A short ping is made by the short pings' sonar and draws its noise from a generator of its own. A run with the
  mode therefore gives the long pings of a run without it at the same transmit times (to 4e-8 of their peak: the times'
  last bits). A schedule whose windows overlap is refused, and so are legs, raster and motion. `ScenarioResult.kinds`
  names each record's kind. The records gain nothing: each carries its own replica and window.
- **The chain, both engines:** one change. The multiscale detector's plan is keyed on the ping's pulse as well (type,
  band, length, taper), so it is planned for each ping's own pulse (`python/gpusonar/live.py`,
  `libs/live/src/chain.cpp`). Before, two pings whose maps had the same number of cells kept the first ping's plan; for
  these two pulses the thresholds were 0.002 dB apart. Everything else (the map over each record's window, blanking from
  each record's own pulse, one tracker, the classifier) already worked per record. There is no report field: the ping's
  kind is its pulse length, which the record carries.
- **Scenes:** `popoto_short_range_mode` (a ship at 12 km, a still object at 800 m, a mover opening through 1.5 km; free
  field; ten pings). `popoto_elevation_short_range` (popoto_elevation_free's three targets near 1 km with the shipped 2 s
  pulse and the mode). There are 102 scenario files.
- **The study:** `tools/short_range_study.py`.

## 4. Measured on the scenes (both engines, the same report lines; `tests/python/test_short_range_mode.py`)

`popoto_short_range_mode`, the chain's defaults, one thread:

- **Without the mode the near object is not seen.** The window opens at 1500 m. No detection is nearer than 1.5 km and no
  track is within 300 m of 800 m, in either engine.
- **With it**, the object at 800 m is detected on 5 of 5 short pings (snr about 44 dB) and tracked, and on no long ping.
- **Hand-over:** the mover opening from 1300 m is detected by the short pings from the first, and by the long pings once it
  is beyond 1.5 km (1514, 1610, 1705 m). All its detections go to one track, which keeps its id to the last ping (1764 m).
- **The 12 km ship:** detected on 5 of 5 long pings (snr 14.4 to 16.3 dB) and tracked. Its detections equal those of the
  baseline alone at the same transmit times, within 1e-4 dB and 1e-4 degrees. So the derived cost per ping is 0 dB in this
  simulation (item 1). The cost is the update rate: one long ping every 23.3 s instead of every 19.0 s.
- One false detection in ten pings (ping 0, 11 307 m, 10.7 dB), the same in both engines.
- Cost on this CPU: simulation 10 s for the ten pings, each chain about 70 s.
- **The rows' elevation** (`popoto_elevation_short_range`): the long pings see nothing (the targets are inside 1.5 km). The
  short pings give the three tracks the targets' depths (10, 60, 160 m) within 2 m in both engines, as
  `popoto_elevation_free` does with the former 1 s sweep.

## 5. Questions for the owner

1. Can the chosen projector and drive board send a 2 s and a 0.25 s pulse in turn, every 19 and 4 s? Neither the
   projector nor the board is chosen (`docs/design/projector_selection.md`); no sheet states it.
2. How soon after a transmission does the receiver recover? This sets the true blind range of either pulse
   (`projector.blanking_after_s`, a bench measurement), and it decides whether option (d) is worth building.
3. Is one short ping after every long one (the long mode at 82 % of its update rate) acceptable, or should the short ping
   come less often? Less often needs the tracker change of item 3.
4. Is `xl_auv` at its low +5 dB a class the hand-over must hold? If it is, neither pulse holds it well between 1.5 and
   2.25 km on the shelf-type sites (item 4).

## 6. The GPU job (not run)

The scenes are free field and run on a CPU. On the device, the same scenes with Bellhop water and reverberation are what
would show item 1 and the near-range reverberation: `GPUSONAR_JOB=doppler` has no sub-run for it, so it is a follow-up.
The full GPU suite runs the new test file like any other.
