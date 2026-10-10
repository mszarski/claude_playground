# Jurassic Tycoon — Containment Edition

A pixel-art park builder in the spirit of SimCity, set on a dinosaur island where *everything* is trying to go wrong.

Build paddocks, power the electric fences, hatch dinosaurs, lay paths for guests — then survive storms,
grid sabotage, earthquakes, eruptions, outbreaks and very clever raptors.

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
| Q P F R V X T C H | Tools: inspect, path, fence, paddock, wall, demolish, trees, clear, hatch |

## Development

All code is plain JavaScript in `src/` loaded by `index.html`:

- `sprites.js` — all pixel art (dinosaurs are hand-drawn character grids, buildings are procedural)
- `world.js` — terrain generation, paddock region detection, power grid, pathfinding
- `entities.js` — dinosaurs, guests, rangers, engineers, the ACU helicopter
- `events.js` — disasters
- `game.js` — economy, goals, simulation loop
- `render.js`, `ui.js`, `audio.js`, `main.js`

`tools/` holds headless Playwright smoke tests (`node tools/smoke.js out.png tools/play1.js`).
