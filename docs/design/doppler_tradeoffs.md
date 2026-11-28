# Doppler in the live chain: the plain sweep, the sweep bank and the tone path

For the owner, 2026-11-27. Every number is **ours** (simulation; nothing has run on a vehicle or on real data) and names
where it is recorded. "GPU 2026-11-27" marks numbers from that day's L4 runs that the coordinator records in
`docs/DECISIONS.md` (not yet in the repository when this was written). The detail is in `docs/tasks/DOPPLER.md`.

## What is worse or not known, first

- **(Added by the document sweep after D-263.) Since D-262 the 12 km baseline is the 2 s hyperbolic sweep at 197 dB
  with the plain filter** (`configs/sonars/popoto_long_range.yaml`; `docs/tasks/DOPPLER.md` HB.1 to HB.4). What it cost,
  first: nothing is seen inside 1.5 km (0.75 km before); a mover's range reads 6 m per m/s of range rate off, which the
  plain filter cannot correct; no range rate from the sweep (the bank takes a linear sweep only, and `range_doppler:
  ownship | full` is now refused on a sampled pulse in both engines, where it applied 0 silently); the elevation scenes
  keep the 1 s linear sweep, so no depth at 12 km was measured then (measured since, D-266: as deployed the stacked rows'
  depth cannot tell a submerged target from a ship at 12 km, and in 100 m of water at 1 to 2 km it is dishonest;
  `docs/tasks/DEPTH_RANGE.md`); two face self-checks fail (one passes with the opt-in range correction of D-264). Measured from its own
  configuration (one seed, 6 pings, L4): at the deep site at sea state 3 every target 6 of 6, the resolved hulls by
  +1.28 / +0.73 dB (thin); sea state 4 not held; sea state 5 nothing held; on the 12 km shelf 9.84 / 10.0 / 12.61 dB,
  all tracked as movers (the 1 s linear sweep 7.98 / 9.22 / 11.41 dB). **The "Plain sweep (default)" column below
  was written for the 1 s linear sweep shipped until D-262**; the bank and the tone path in it still take a linear
  sweep (the bank's scene `popoto_12km_large_submerged` and the trains state their own pulses).
- **(Added by the document sweep after D-261.) At the noise-limited deep site the plain linear sweep, the shipped
  baseline, does not hold the resolved hull at 12 km** (D-260; `docs/tasks/DOPPLER.md` DM.1 to DM.7;
  `docs/tasks/doppler_deep12_eval.json`; one seed, 6 pings, L4): at sea state 3 the 1 s, 197 dB linear sweep with the
  plain filter holds neither the closing nor the oblique `large_submerged` hull nor the ship (0 of 6 each). Two causes,
  measured: the hull's highlights resolve in range (no Doppler path helps that: hull minus point -3.8 to -6.0 dB with a
  filter that takes no Doppler loss) and the linear sweep's plain-filter loss on movers (at 4 s against the hyperbolic
  sweep 9.3 dB on the ship, 6.0 dB on the closing hull; the bank recovers it within 0.1 to 0.8 dB of the hyperbolic
  sweep, at 25.6 s a ping with three rows on the L4). With the bank (`combine: cfar`) the ship is held at the shipped
  energy, the two hulls are not. The owner's decision (D-256): the baseline's waveform becomes a **2 s hyperbolic sweep
  at 197 dB with the plain filter, no bank** (measured: every target held on 6 of 6 at sea state 3, the hulls by +1.28 /
  +0.73 dB on one seed; not at sea state 4; costs: blind range 1.5 km, a range bias of 6 m per m/s on movers the plain
  filter cannot correct, no range rate, the bank refuses it). (Since D-262 merged, D-263; the item above.)
- **The tone path is the only one that measures a range rate, and it has three blind spots**: targets slow against the
  water (its notch: +-0.38 m/s at 6.5 kHz, 12 % of headings at 2 m/s, `DOPPLER.md` LS.3), every crossing target
  (on its beam's ridge: 0 of 3 tone pings on `popoto_12km_large_submerged`, LS.3), and, under way, targets coming nearly
  straight in or out on beams beyond 24 degrees (the grating-lobe notch: 7 to 16 % of headings at 2 m/s, `DOPPLER.md`
  GL.3). The sweep path detects all three; only the rate is missing.
- **The tone path's false-alarm rate is counted on one train scene, over 4 target-free pings only** (amended by the
  document sweep after D-254; first written: "is not known at the default budget"). Before the grating-lobe notch, the
  train scene at 12 km gave 22 false tone detections in 4 target-free pings at the default budget (GPU 2026-11-27,
  against 0 with the tone path off). (Added by the document sweep of 2026-11-27: D-246 and `DOPPLER.md` GL.8 found those
  22, and the 25 with the notch, mostly an artefact of the simulator's reverberation Doppler at the train's centre
  frequency, D-249, not a property of the tone path.) With the notch a synthetic field gives 15 -> 0 (GL.1); a stronger field still leaves some through
  the beams' far sidelobes (GL.1 item 2). The GPU rerun is the acceptance. (Added after D-254: the simulator was fixed
  (D-251) and the rerun made (D-252, job 6b0947c354072e8b71697a5a, `popoto_12km_large_submerged_tone`, 4 target-free
  pings, both engines agreeing): **0 false detections at the default budget with the notch (auto), 17 with it off, all
  17 on a lobe's ridge; 1 with the tone path off.** At 3 a ping: 25 with the notch, 3 of them the tone's and 22 the
  sweep path's; 16 with the tone path off. Not shown: more than 4 pings, the far-sidelobe case on a real scene. The tone
  path stays off by default, D-252.)
- **No path gives the tracker a rate**: the tracker takes the tone's coarse range sigma only (D5 not done). With
  alternating pings a 2 m/s target was called `static` (LS.1 item 4); with `combine: cfar` the crossing target was too.
  (Added after D-261: D5 is done, opt-in, D-258: `tracker.range_rate: update` takes the tone path's rate; on the device
  the flank small vehicle is a mover from ping 3 and the 12 km oblique target too; the oblique target's class flickers
  on alternating pings, and the crossing target, which the tone cannot see, is `uncertain`.)
- **The sweep bank's rate is not a measurement** (LS.1 item 5: -3.4 to +7.9 m/s on a target at -0.50).

## The three, side by side

| | Plain sweep (default) | Sweep bank, `doppler: {mode: sweep_bank, combine: cfar}` | Tone path, `doppler: {tone: bank}` (tone above the band in one ping, or alternating pings) |
| --- | --- | --- | --- |
| What it adds | nothing: one matched filter | each beam compressed at 2K + 1 range-rate hypotheses, each normalised by its own background (`max`, the bank's default, loses 3.4 to 5.3 dB of margin on every scene, D-232, DOPPLER.md D3.2) | a Doppler filter bank on a tone per beam: a range rate within 0.06 m/s (8 looks, LS.4) and detection of movers the sweep loses in reverberation |
| Still targets | detected | detected (wreck 14.63 -> 14.15 dB margin, -0.48 dB, D-237) | not reported by the tone (the notch); the sweep part of the ping keeps them |
| Movers | detected; a long sweep loses gain on fast ones (11.7 dB at 2 m/s on a 1 s, 4 kHz sweep, D4) | gains on every mover measured: +0.52 to +1.51 dB margin (D-237), +0.6 to +0.9 dB on the 12 km `large_submerged` (LS.4); small vehicle 3 of 6 pings against 1 | outside the notches: 6 of 6 tone pings with rates 1.964 to 1.997 m/s (true 1.996) on the flank scene (GPU 2026-11-27) |
| Crossing targets | detected | detected; may be called `static` (LS.1 item 4) | in the notch: not reported by the tone (0 of 3, LS.3) |
| Grating-lobe bearings (tone above the band, under way) | none in the sweep's band within +-40 degrees (from 41.5 degrees at 6 kHz, LS.1 item 0) | as the plain sweep | notched since 2026-11-27 (`tone_lobe_notch: auto`): before, 3 to 5 false detections every tone ping at 32 to 40 degrees (LS.1); now the cost of GL.3 |
| Range rate on the report | none (no field: a contract) | the winning hypothesis, coarse, with `report: true` | the tone's, with `report: true`; `doppler_path` says which path |
| False detections on noise at the default budget, long-range array | 39 in 40 000 pings: 0.97 of the budget (95 % 0.69 to 1.33; GPU 2026-11-27) | 8 in 10 296: 0.78 (0.34 to 1.53); `max` 0 in 16 557 (0 to 0.22) (GPU 2026-11-27) | not counted at the default budget; within the budget at 3 and 30 a ping on noise (`test_doppler_tone.py`); flank scene background 1 in 4 pings (GPU 2026-11-27) |
| Cost a ping on an L4, C++ chain (GPU 2026-11-27) | flank_patrol 0.116 s, popoto_far_mover 0.062 s, popoto_12km 0.020 s | `cfar` 0.453, 0.403, 0.109 s (`max` 0.424, 0.379, 0.097; before the device batching 3.935, 2.766, 0.532) | popoto_12km_doppler 0.033 -> 0.048 s |
| What the tracker does with it | positions only | positions only (rate unused; `range_rate_from: any` would take it, not recommended, D-258) | positions, and the tone's coarse range sigma (c T / 8); since D-258 opt-in the rate too (`tracker.range_rate`) |
| The ping | the shipped sweep | the shipped sweep (refused on a train, LS.1 item 6) | a tone 0.5 kHz above the sweep in the same ping (half the energy each, 2 s ping, unconfirmed for the projector), or tone and sweep pings in turn (half the update rate each) |

## When to turn each on

- **Plain sweep**: always (it is the default and every other path keeps it beside itself).
- **`combine: cfar`**: when the margin on movers matters more than about 0.5 dB on still targets and the classifier's
  calls on crossing targets are not relied on. Never `max` (it loses on every scene).
- **Tone path**: when a measured range rate is wanted (for D5, the tracker and classifier) and the transmit can carry
  the tone. It adds no detection at 12 km on the reverberation-limited shelf, where the sweep holds all three
  `large_submerged` targets (LS.4; at the noise-limited deep site the 1 s sweep does not, D-260, above); it adds a
  rate on targets outside its notches. Keep `tone_lobe_notch: auto` (the default with the tone path); `off` only to
  compare. Do not rely on it for crossing targets or for targets within +-0.4 m/s of the water.
