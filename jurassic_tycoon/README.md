# Jurassic Tycoon — Containment Edition

A pixel-art park builder in the spirit of SimCity, set on a dinosaur island where *everything* is trying to go wrong.

Build paddocks, power the electric fences, hatch dinosaurs, lay paths for guests — then survive storms,
grid sabotage, earthquakes, eruptions, outbreaks and very clever raptors.

## Features

- **Paddocks & power** — electric fences only work inside a power plant's coverage (extend it with pylons). Concrete walls need no power but cost more.
- **11 species** from Gallimimus to T. rex, plus Pteranodons in an Aviary and an endgame Indominus hybrid that camouflages itself.
- **Dinosaur needs** — space, tree cover, water, herd size, food, and fear of predators all feed a comfort score. Stressed dinos test the fences; raptors look for the weakest spot.
- **Guests** walk paths, eat, shop, ride the electric **jeep tour**, watch **Mosasaur** feeding shows, and flee to shelters when the alarm sounds.
- **Disasters** — tropical storms with lightning, grid sabotage, earthquakes, volcanic eruptions, outbreaks, rampages, raptor probes, safety inspections. Trigger them yourself from the ☄ menu.
- **Response tools** — evacuation alarm, siren towers, rangers with tranquilizer rifles, engineers, ACU helicopter airlifts and strikes, emergency fence repair, backup generators, vet clinics.
- **Scenarios** — Nedry's Night, Isla Sorna Cleanup, The Lost World, Opening Day, Raptor Siege and Storm Season: prebuilt crises with objectives and deadlines.
- Goals, an advisor, star rating, finances with history graphs, a news ticker, a breach cam, save/load, three difficulty levels, sandbox mode, synthesized sound effects and an original chiptune soundtrack.

## Play

Open `index.html` in any modern browser. No build step, no dependencies.

## Controls

| Input | Action |
| --- | --- |
| Left click | Use tool / select |
| Drag | Build lines (paths, fences, walls) and areas (paddock, demolish, trees) |
| Right-drag, or drag with Inspect | Pan |
| Mouse wheel / pinch | Zoom |
| WASD / arrows | Pan |
| Space, 1, 2, 3 | Pause / speed |
| E | Evacuation alarm |
| O / K | Power / paddock overlays |
| L | Jump to loose dinosaur |
| G | Dinosaur roster |
| U or Ctrl+Z | Undo last construction (refunded) |
| Q P N J F R V X T C H | Tools: inspect, path, smart path, tour track, fence, paddock, wall, demolish, trees, clear, hatch |

## Development

All code is plain JavaScript in `src/` loaded by `index.html`:

- `sprites.js` — all pixel art (dinosaurs are hand-drawn character grids, buildings are procedural)
- `world.js` — terrain generation, paddock region detection, power grid, pathfinding
- `entities/` — `dino.js`, `ptera.js`, `guest.js`, `staff.js` (rangers, engineers), `vehicles.js` (tour jeeps, ACU helicopter)
- `events.js` — disasters
- `scenarios.js` — scenario setups and win/lose checks
- `game.js` — economy, goals, simulation loop
- `render.js`, `ui.js` (input, toolbar, HUD), `ui-panels.js` (info panel, dialogs), `audio.js`, `main.js`

`tools/` holds headless Playwright smoke tests (`node tools/smoke.js out.png tools/play1.js`).
