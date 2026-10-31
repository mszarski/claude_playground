# Jurassic Tycoon — Containment Edition

A pixel-art park builder in the spirit of SimCity, set on a dinosaur island where *everything* is trying to go wrong.

Build paddocks, power the electric fences, hatch dinosaurs, lay paths for guests — then survive storms,
grid sabotage, earthquakes, eruptions, outbreaks and very clever raptors.

## Features

- **Paddocks & power** — electric fences only work inside a power plant's coverage (extend it with pylons). Concrete walls need no power but cost more.
- **11 species** from Gallimimus to T. rex, plus Pteranodons in an Aviary and an endgame Indominus hybrid that camouflages itself.
- **Dinosaur needs** — space, tree cover, water, herd size, food, and fear of predators all feed a comfort score. Stressed dinos test the fences; raptors look for the weakest spot; when the grid dies, predators go for the dead fences; and the Indominus schemes against her walls every few days. Plant trees and **dig ponds** to keep them content, and top up an exhibit with **Hatch another** from its paddock panel.
- **Guests** (each with a name and a hometown) walk paths, eat, shop, stay in the **Lodge Hotel** (40 rooms), ride the **jeep tour** and **gyrospheres**, crowd the **Feeding Grandstand** at feeding time, visit the **Mosasaur Lagoon** and **Raptor Arena**, and flee to shelters when the alarm sounds. Narrow paths jam in a panic: lay **wide paths** where the crowds are.
- **Disasters** — tropical storms with lightning (and winds that tear at the Aviary mesh), grid sabotage, earthquakes, volcanic eruptions, outbreaks, rampages, raptor probes, safety inspections. Trigger them yourself from the ☄ menu. When power runs short, fences, staff and shelters stay lit first and a banner says how much more you need.
- **Response tools** — evacuation alarm, siren towers, rangers with tranquilizer rifles, engineers, ACU helicopter airlifts and strikes, emergency fence repair, backup generators, vet clinics, and closing the park from the Main Gate.
- **Campaign** — five chapters (Opening Season, Clever Girls, Monsoon, Site B, Jurassic World), each unlocking a perk for the ones after it.
- **InGen calls** — investors, lawyers, the board and Dr. Wu phone in with deals that pay now and bite later.
- **RTS controls** — select rangers, engineers and helicopters and give them orders, or hit RESPOND on a breach; everyone also works on their own.
- **Scenarios** — Nedry's Night, Isla Sorna Cleanup, The Lost World, Opening Day, Raptor Siege and Storm Season: prebuilt crises with objectives and deadlines.
- **Research Lab** — upgrades (reinforced fences, high-voltage grid, faster darts, ranger training, turbines, medicine, feeders, motion sensors, ACU gunship, marketing) and genome research that unlocks species without stars.
- **Seasons** — Spring, Summer (tourist peak), Monsoon (storms) and Cool Season; adjustable day length; a Calm mode with no disasters.
- **Island codes** to share a map, and **medals** for each scenario with your best result saved.
- Goals (with Show me / How? help), an advisor, a **Park Rating** breakdown (click the rating), finances with a daily net and history graphs, a news ticker, a breach cam, save/load, three difficulty levels, sandbox mode, synthesized sound effects and an original chiptune soundtrack.

## Play

Open `index.html` in any modern browser. No build step, no dependencies.

## Controls

| Input | Action |
| --- | --- |
| Left click | Use tool / select |
| Drag | Build lines (paths, fences, walls) and areas (paddock, demolish, trees, ponds) |
| Right-drag, or drag with Inspect | Pan |
| Mouse wheel / pinch | Zoom |
| WASD / arrows | Pan |
| Space, 1–4 | Pause / speed |
| E | Evacuation alarm |
| O / K | Power / paddock overlays |
| L | Jump to loose dinosaur |
| B / M | Respond to a breach / select all rangers |
| G | Dinosaur roster |
| U or Ctrl+Z | Undo last construction, demolition or egg |
| Y | Research |
| Q P I N J F R V X T C H | Tools: inspect, path, wide path, smart path, tour track, fence, paddock, wall, demolish, trees, clear, hatch |

## Development

All code is plain JavaScript in `src/` loaded by `index.html`:

- `sprites.js` — pixel art (dinosaurs are hand-drawn character grids, buildings are procedural)
- `sprites-views.js` — hand-drawn front/back dinosaur views
- `world.js` — terrain generation, paddock region detection, power grid, pathfinding
- `entities/` — `dino.js`, `ptera.js`, `guest.js`, `staff.js` (rangers, engineers), `vehicles.js` (tour jeeps, ACU helicopter)
- `events.js` — disasters
- `scenarios.js` — scenario setups and win/lose checks
- `tutorial.js` — guided first session
- `research.js` — research projects and their effects
- `game.js` — economy, goals, simulation loop
- `render.js`, `ui.js` (input, toolbar, HUD), `ui-panels.js` (info panel, dialogs), `audio.js`, `main.js`

`tools/` holds headless Playwright tests. `node tools/ci.js` runs the suite that GitHub Actions runs on every push (boot every mode, complete the tutorial, fuzz with save/load round trips, open every panel). Other scripts are scripted playthroughs, e.g. `node tools/smoke.js out.png tools/long.js`.
