# Doppler in the live chain: the plain sweep, the sweep bank and the tone path

For the owner, 2026-11-27. Every number is **ours** (simulation; nothing has run on a vehicle or on real data) and names
where it is recorded. "GPU 2026-11-27" marks numbers from that day's L4 runs that the coordinator records in
`docs/DECISIONS.md` (not yet in the repository when this was written). The detail is in `docs/tasks/DOPPLER.md`.

## What is worse or not known, first

- **The tone path is the only one that measures a range rate, and it has three blind spots**: targets slow against the
  water (its notch: +-0.38 m/s at 6.5 kHz, 12 % of headings at 2 m/s, `DOPPLER.md` LS.3), every crossing target
  (on its beam's ridge: 0 of 3 tone pings on `popoto_12km_large_submerged`, LS.3), and, under way, targets coming nearly
  straight in or out on beams beyond 24 degrees (the grating-lobe notch: 7 to 16 % of headings at 2 m/s, `DOPPLER.md`
  GL.3). The sweep path detects all three; only the rate is missing.
- **The tone path's false-alarm rate is not known at the default budget.** Before the grating-lobe notch, the train scene
  at 12 km gave 22 false tone detections in 4 target-free pings at the default budget (GPU 2026-11-27, against 0 with
  the tone path off). With the notch a synthetic field gives 15 -> 0 (GL.1); a stronger field still leaves some through
  the beams' far sidelobes (GL.1 item 2). The GPU rerun is the acceptance.
- **No path gives the tracker a rate**: the tracker takes the tone's coarse range sigma only (D5 not done). With
  alternating pings a 2 m/s target was called `static` (LS.1 item 4); with `combine: cfar` the crossing target was too.
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
| What the tracker does with it | positions only | positions only (rate unused) | positions, and the tone's coarse range sigma (c T / 8); rate unused (D5) |
| The ping | the shipped sweep | the shipped sweep (refused on a train, LS.1 item 6) | a tone 0.5 kHz above the sweep in the same ping (half the energy each, 2 s ping, unconfirmed for the projector), or tone and sweep pings in turn (half the update rate each) |

## When to turn each on

- **Plain sweep**: always (it is the default and every other path keeps it beside itself).
- **`combine: cfar`**: when the margin on movers matters more than about 0.5 dB on still targets and the classifier's
  calls on crossing targets are not relied on. Never `max` (it loses on every scene).
- **Tone path**: when a measured range rate is wanted (for D5, the tracker and classifier) and the transmit can carry
  the tone. It adds no detection at 12 km where the sweep holds all three `large_submerged` targets (LS.4); it adds a
  rate on targets outside its notches. Keep `tone_lobe_notch: auto` (the default with the tone path); `off` only to
  compare. Do not rely on it for crossing targets or for targets within +-0.4 m/s of the water.
