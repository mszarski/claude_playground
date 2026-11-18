# Own depth against a sound-speed maximum: the depth planner in fjord-type water

Made by `tools/fjord_depth_plan.py` (the planner of `tools/popoto_depth_planner.py` with the waters named below; flat seabed). ILLUSTRATIVE water; placeholder targets.

## popoto_testauv_1x32, target of 10 dB between 5 and 40 m at 500 m

```
Build popoto_testauv_1x32 (1 row); environment fjord_profile: 70 m, sound-speed maximum at 19 m (illustrative), calm
  Bellhop arrivals over the profile (flat bottom, range-independent); the projector taken at the array's depth; wind 1.9 m/s, rms wave height 0.01 m; own depths 5-59 m; range of interest 0.5 km; threshold 13 dB

| Class | TS used (dB) | Depths (m) | Free-field margin | Best own depth | Margin there | Own depths that hold it | Within 3 dB of best | Reach there (km) | Target depths not held there |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| trial_target | +10 (assumed) | 5-40 | +43.6 dB | 56 m | +48.6 dB | 5-59 m | 5-59 m | 0.8 | none |

  Rests on: the sound-speed profile supplied with the environment.
  trial_target: at own depth 5 m: +46.7 dB, reach 0.8 km; 11 m: +46.9 dB, reach 0.8 km; 29 m: +46.9 dB, reach 0.8 km. With the high end of its target strength add 0.0 dB.
    depth diversity (every target depth to 69 m): 5 m -> 100%
```

Two-way propagation factor re free field at the range, worst over the target's depths, by own depth, dB: 5 m: +3.2; 8 m: +3.5; 11 m: +3.4; 14 m: +3.9; 17 m: +4.5; 20 m: +3.6; 23 m: +3.4; 26 m: +2.8; 29 m: +3.3; 32 m: +3.0; 35 m: +3.2; 38 m: +3.8; 41 m: +3.7; 44 m: +4.1; 47 m: +2.7; 50 m: +4.7; 53 m: +4.2; 56 m: +5.1; 59 m: +4.9

```
Build popoto_testauv_1x32 (1 row); environment same_isovelocity: the same water at one sound speed
  image series of an isovelocity layer over a fluid bottom (Rayleigh reflection), to 30 degrees of grazing; wind 1.9 m/s, rms wave height 0.01 m; own depths 5-59 m; range of interest 0.5 km; threshold 13 dB

| Class | TS used (dB) | Depths (m) | Free-field margin | Best own depth | Margin there | Own depths that hold it | Within 3 dB of best | Reach there (km) | Target depths not held there |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| trial_target | +10 (assumed) | 5-40 | +43.6 dB | 23 m | +48.6 dB | 5-59 m | 5-59 m | 0.8 | none |

  Rests on: isovelocity water: no profile was supplied (the bottom rung: nothing known of the profile).
  trial_target: at own depth 5 m: +47.3 dB, reach 0.8 km; 11 m: +47.2 dB, reach 0.8 km; 29 m: +46.8 dB, reach 0.8 km. With the high end of its target strength add 0.0 dB.
    depth diversity (every target depth to 69 m): 5 m -> 100%
```

Two-way propagation factor re free field at the range, worst over the target's depths, by own depth, dB: 5 m: +3.8; 8 m: +3.5; 11 m: +3.7; 14 m: +3.3; 17 m: +3.5; 20 m: +4.4; 23 m: +5.0; 26 m: +4.3; 29 m: +3.3; 32 m: +4.0; 35 m: +4.7; 38 m: +3.2; 41 m: +4.9; 44 m: +4.5; 47 m: +4.4; 50 m: +4.4; 53 m: +4.2; 56 m: +4.6; 59 m: +4.9

## popoto_static_3x32, target of 10 dB between 5 and 40 m at 150 m

```
Build popoto_static_3x32 (3 rows); environment fjord_profile: 70 m, sound-speed maximum at 19 m (illustrative), calm
  Bellhop arrivals over the profile (flat bottom, range-independent); the projector taken at the array's depth; wind 1.9 m/s, rms wave height 0.01 m; own depths 5-59 m; range of interest 0.1 km; threshold 13 dB

| Class | TS used (dB) | Depths (m) | Free-field margin | Best own depth | Margin there | Own depths that hold it | Within 3 dB of best | Reach there (km) | Target depths not held there |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| trial_target | +10 (assumed) | 5-40 | +70.6 dB | 11 m | +75.7 dB | 5-59 m | 5-59 m | 0.2 | none |

  Rests on: the sound-speed profile supplied with the environment.
  trial_target: at own depth 5 m: +74.5 dB, reach 0.2 km; 11 m: +75.7 dB, reach 0.2 km; 29 m: +75.6 dB, reach 0.2 km. With the high end of its target strength add 0.0 dB.
    depth diversity (every target depth to 69 m): 5 m -> 100%
```

Two-way propagation factor re free field at the range, worst over the target's depths, by own depth, dB: 5 m: +3.9; 8 m: +4.4; 11 m: +5.1; 14 m: +5.1; 17 m: +5.0; 20 m: +2.6; 23 m: +4.3; 26 m: +4.6; 29 m: +5.0; 32 m: +4.5; 35 m: +4.2; 38 m: +4.6; 41 m: +4.4; 44 m: +3.6; 47 m: +2.3; 50 m: +4.2; 53 m: +4.1; 56 m: +4.3; 59 m: +4.1

```
Build popoto_static_3x32 (3 rows); environment same_isovelocity: the same water at one sound speed
  image series of an isovelocity layer over a fluid bottom (Rayleigh reflection), to 30 degrees of grazing; wind 1.9 m/s, rms wave height 0.01 m; own depths 5-59 m; range of interest 0.1 km; threshold 13 dB

| Class | TS used (dB) | Depths (m) | Free-field margin | Best own depth | Margin there | Own depths that hold it | Within 3 dB of best | Reach there (km) | Target depths not held there |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| trial_target | +10 (assumed) | 5-40 | +70.6 dB | 56 m | +70.7 dB | 5-59 m | 5-59 m | 0.2 | none |

  Rests on: isovelocity water: no profile was supplied (the bottom rung: nothing known of the profile).
  trial_target: at own depth 5 m: +69.7 dB, reach 0.2 km; 11 m: +69.1 dB, reach 0.2 km; 29 m: +70.3 dB, reach 0.2 km. With the high end of its target strength add 0.0 dB.
    depth diversity (every target depth to 69 m): 5 m -> 100%
```

Two-way propagation factor re free field at the range, worst over the target's depths, by own depth, dB: 5 m: -0.9; 8 m: -0.4; 11 m: -1.5; 14 m: -0.7; 17 m: +0.0; 20 m: -0.2; 23 m: -0.4; 26 m: -0.3; 29 m: -0.3; 32 m: +0.1; 35 m: -0.1; 38 m: -0.0; 41 m: -0.1; 44 m: -0.2; 47 m: -0.2; 50 m: -0.1; 53 m: -0.2; 56 m: +0.1; 59 m: -0.4

## popoto_static_2x32, target of 15 dB between 5 and 40 m at 2600 m

```
Build popoto_static_2x32 (2 rows); environment fjord_profile: 70 m, sound-speed maximum at 19 m (illustrative), calm
  Bellhop arrivals over the profile (flat bottom, range-independent); the projector taken at the array's depth; wind 1.9 m/s, rms wave height 0.01 m; own depths 5-59 m; range of interest 2.6 km; threshold 13 dB

| Class | TS used (dB) | Depths (m) | Free-field margin | Best own depth | Margin there | Own depths that hold it | Within 3 dB of best | Reach there (km) | Target depths not held there |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| trial_target | +15 (assumed) | 5-40 | +23.1 dB | 20 m | +32.7 dB | 5-59 m | 5-29 m, 38-41 m, 50-53 m | 3.9 | none |

  Rests on: the sound-speed profile supplied with the environment.
  trial_target: at own depth 5 m: +30.5 dB, reach 3.9 km; 11 m: +32.2 dB, reach 3.9 km; 29 m: +31.0 dB, reach 3.9 km. With the high end of its target strength add 0.0 dB.
    depth diversity (every target depth to 69 m): 5 m -> 100%
```

Two-way propagation factor re free field at the range, worst over the target's depths, by own depth, dB: 5 m: +7.3; 8 m: +8.8; 11 m: +9.1; 14 m: +9.1; 17 m: +9.0; 20 m: +9.6; 23 m: +8.7; 26 m: +8.8; 29 m: +7.9; 32 m: +5.8; 35 m: +5.8; 38 m: +6.7; 41 m: +7.1; 44 m: +6.1; 47 m: +6.1; 50 m: +7.7; 53 m: +7.2; 56 m: +6.4; 59 m: +6.0

```
Build popoto_static_2x32 (2 rows); environment same_isovelocity: the same water at one sound speed
  image series of an isovelocity layer over a fluid bottom (Rayleigh reflection), to 30 degrees of grazing; wind 1.9 m/s, rms wave height 0.01 m; own depths 5-59 m; range of interest 2.6 km; threshold 13 dB

| Class | TS used (dB) | Depths (m) | Free-field margin | Best own depth | Margin there | Own depths that hold it | Within 3 dB of best | Reach there (km) | Target depths not held there |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| trial_target | +15 (assumed) | 5-40 | +23.1 dB | 41 m | +31.7 dB | 5-59 m | 5-59 m | 3.9 | none |

  Rests on: isovelocity water: no profile was supplied (the bottom rung: nothing known of the profile).
  trial_target: at own depth 5 m: +30.2 dB, reach 3.9 km; 11 m: +31.2 dB, reach 3.9 km; 29 m: +31.3 dB, reach 3.9 km. With the high end of its target strength add 0.0 dB.
    depth diversity (every target depth to 69 m): 5 m -> 100%
```

Two-way propagation factor re free field at the range, worst over the target's depths, by own depth, dB: 5 m: +7.1; 8 m: +7.8; 11 m: +8.1; 14 m: +6.9; 17 m: +6.9; 20 m: +7.9; 23 m: +8.1; 26 m: +8.1; 29 m: +8.2; 32 m: +8.4; 35 m: +8.5; 38 m: +7.8; 41 m: +8.5; 44 m: +6.7; 47 m: +7.7; 50 m: +8.0; 53 m: +8.2; 56 m: +7.8; 59 m: +8.5

