# Session handoff: Jurassic Tycoon

Notes for a fresh session picking up this project. Most of the context below came out of a long run of
"play more" playtesting rounds, not the code, so read this before changing balance or behaviour.

## 1. Goal

**What it is:** *Jurassic Tycoon: Containment Edition*, a pixel-art dinosaur theme-park builder (SimCity /
Jurassic Park Builder spirit) that runs in the browser with no build step: plain JavaScript and canvas.

You build paddocks, power the electric fences, hatch dinosaurs, lay paths, run attractions and keep guests
happy, while storms, grid failures, quakes, eruptions and clever predators try to undo it.

**Why:** a fun, polished, playable game. The owner's standing direction throughout has been:

- **More drama:** breaches should be frightening and cost lives. They explicitly said "seems like need more
  deaths though? of guests?".
- **Less busywork:** one-click upkeep, templates, sensible automation.
- **Optional RTS-style control:** you can command rangers, but things also happen on their own.
- **A campaign.**

**What "done" looks like:** there is no fixed finish line. The working loop is:

1. Playtest.
2. Find what's broken, confusing, unfair or dull.
3. Fix or rebalance it.
4. Verify with automated runs and screenshots.
5. Ship: commit, push, sync the PR, publish the artifact.
6. Report.

Success per round means something real improved, every scenario and chapter stays winnable with sensible
play and losable with poor play, and CI is green.

## 2. Current state (as of commit `d883867`, 2026-10-31)

### Everything below works and is shipped

**Modes**
- Tutorial: 12 steps, a fixed friendly island (seed 51), slow days with `secPerHour` 7.5.
- Free play: Calm, Relaxed, Standard and Chaos Theory difficulties; Small, Medium or Large islands; three day
  lengths; shareable island codes.
- Sandbox: $5M to start, everything unlocked, no bankruptcy, random disasters still on.
- Six scenarios: Nedry's Night, Isla Sorna Cleanup, The Lost World, Opening Day, Raptor Siege and Storm
  Season.
- A five-chapter campaign with perks that carry forward.

**Systems**
- **Power and fences:** a power grid with pylons, electric fences, concrete walls and track gates.
- **Paddocks and dinosaurs:**
  - Paddock regions are detected automatically.
  - 12 species, including Pteranodons (Aviary) and the Indominus.
  - Dinosaurs have comfort (space, cover, water, social, food, fear), age and breed.
- **Guests:** pathing on flow fields, needs, crowding, shelters, fleeing and freezing, and names.
- **Staff:** rangers with darts, engineers, ACU helicopters and ground-crew trucks.
- **Disasters, decisions, research:** disasters, InGen decision calls, a Research Lab and seasons.
- **Advice:** an advisor, a Park Rating breakdown panel and finances with a Net row.
- **Presentation:** a news ticker, a breach cam, a save/load system, synthesized audio and chiptune music.
- **Platforms:** touch and phone layout, and an installable offline PWA.

**Recent additions, last ~2 weeks (see `git log`)**
- **Tools:**
  - Wide Path (key I): a 3-wide path in one drag.
  - Dig Pond: water in a paddock, $150 per tile.
  - "Hatch another <species>" button in the paddock panel.
  - Paddock templates now copy ponds.
- **Help and info:**
  - Park Rating panel: click the Rating stat, or "How?" on star, crowd or cash goals.
  - Power-shortage banner with a Build Power Plant button. Essentials stay lit first: staff, shelters,
    sirens, then hatchery, restrooms, restaurants and the Visitor Center, then lamps, then attractions.
- **Danger:**
  - Carnivores attack dead or unpowered fences during an outage.
  - The Indominus "schemes": every 2–4 days she does a long bout on her weakest wall, powering through zaps.
  - Storms damage the Aviary mesh, and lightning favours the dome.
  - Loose predators hunt the nearest *reachable* person.
  - Darts miss moving targets: 45% for clever species, 20% for others, ×0.6 with darts research.
  - Panicking crowds jam on packed tiles, and fleeing guests trip.
  - Opening Day's Indominus actually steps out through the hole she makes.
  - Rangers rescue dinosaurs stranded in tiny pockets between paddocks.
  - A relocation needs a paddock of at least 30% of the species' space.
- **Economy:**
  - Lodge Hotel has 40 rooms.
  - Mosasaur Lagoon entry is $60 and upkeep $1,000 a day; guests now visit it.
  - Gyrosphere rides take 8–12 seconds and cost $70.
  - Feeding Grandstand: guests gather in the hour before the 12:00 and 18:00 shows, which pay $20 per
    spectator, up to 80.
  - Guests pick only among attractions the park actually has.
- **Saves:** they keep eggs, storms, outages, quakes, volcanoes, lava timers, queued fences, the alarm, a
  locked gate, and building flags (Lagoon escaped, doors broken, plant offline).
- **UX:**
  - Goals tick while paused, and the egg goal shows incubation time.
  - InGen calls wait until a crisis is over.
  - Bad news takes over the ticker.
  - The failure screen shows the goal and where you stood.
  - The New Island menu puts options before difficulty.
  - Locked campaign chapters are readable.
  - Species-themed names (raptors Blue and Charlie, the T. rex Rexy or Roberta), and named guests in death
    messages.
  - Sandbox opening message.

### Latest numbers

All measured with bots on the play harness (§8).

- **Scenarios:** sensible prep or response wins, no prep loses.

| Scenario | No prep | With prep |
|---|---|---|
| Nedry's Night | loses 2 of 3 | wins 4 of 4 (walls, backup generator, bunkers) |
| Raptor Siege | loses by 13:00 on Day 1 | 2+ bunkers win 4 of 4, 0–2 deaths |
| Opening Day | — | wins 3 of 3 (cheaper ticket, amenities, prompt response), 3–5 deaths, limit 6 |
| Isla Sorna Cleanup | — | wins Day 4, 11 of 11 contained |
| The Lost World | — | wins Day 2–3 with a 20- or 55-tile holding pen |
| Storm Season | idle park earns $248k (one seed) and $301k (another), goal $300k | — |

- **Campaign:**
  - Chapter 2: wins 4 of 4.
  - Chapter 3: an idle park earns $276k or $308k against $275k. One seed won; the other failed on
    reputation, 48 of 50.
  - Chapter 5 with my planner bot plus power management: one seed won on Day 50 (silver; it was Day 38
    before the hotel cap); the other fell just short at score 319 of 330, because the bot builds only one
    restaurant and one restroom for 400+ guests.
- **Difficulty gradient** (goal bot, 30 days):

| Difficulty | Disasters | Escapes | Deaths |
|---|---|---|---|
| Calm | 0 | 0 | 0 |
| Relaxed | 5 | 8 | 0 |
| Standard (60 days) | regular | 9–29 | 1–9 per game, mostly lava from a late eruption |
| Chaos | daily | dozens | 0 guest deaths when the bot sounds the alarm instantly; rangers die instead |

- **Attraction payback** in a ~420-guest park:

| Attraction | Payback |
|---|---|
| Restaurant | ~3–4 days |
| Gift shop | ~4–7 days |
| Lodge Hotel | ~11–13 days |
| Tour | ~31 days |
| Feeding Grandstand | ~39 days |
| Gyrosphere | ~41 days |
| Raptor Arena | ~43 days |
| Mosasaur Lagoon | ~250 days; its value is appeal 30, the biggest single rating boost |

- **Performance:** a large island with ~420 guests and 25 dinosaurs runs at a steady 60 fps at both zooms
  and at 4× and 8× speed, even in headless software rendering. Simulation takes ~0.28 ms per update. Drawing
  takes ~0.4 ms of script per frame because terrain is cached.
- **Robustness:** an extended fuzzer ran 16 rounds × 400 random actions on Chaos, with save/load mixed in,
  and found no errors.

### Half-finished or open

- **Guest deaths on Chaos with a perfect responder are ~0.** I offered the owner more levers but got no
  answer yet (§7).
- **The Chapter 5 gold medal (Day 45)** now needs tight play. No bot has re-confirmed it since the hotel cap.
- **The Build menu on phones scrolls** to reach the last item (Clear Land). That's fine, but there's no
  scroll hint.

## 3. Decisions and reasons

- **No build step, plain JS.** Mixins are applied to `Game.prototype` across `src/*.js`.
  - `tools/bundle.js` makes a single-file HTML for the claude.ai artifact.
  - `tools/build-pages.js` makes the PWA site.
  - Rejected: a framework or bundler, to keep it openable straight from `index.html`.
- **The simulation is deterministic per seed** (`RNG = mulberry32(seed)`), so island codes reproduce maps
  and tests are repeatable.
  - Purely cosmetic randomness, such as guest names, uses `Math.random`, so it doesn't shift the seeded
    simulation.
  - Species names use the seeded RNG. Adding that shifted every seeded outcome, so the scenario sweep was
    re-run and was still balanced.
- **The star rating is** `score = attraction + reputation × 0.9`.
  - Attraction is the sum of appeal × comfort, plus 8 per species, plus building appeal.
  - `STAR_THRESH = [0, 55, 110, 165, 235, 330]`.
  - Reputation drifts nightly toward `avgHappy × 0.55 + safety × 0.45`.
- **Money goals count "earned from guests"** (`stats.guestIncome`, tickets plus shops), not the bank
  balance, so building isn't punished.
- **Drama over safety, but player skill matters.** A prepared, alert park (alarm, bunkers, rangers, walls,
  backup generator) can keep casualties near zero. An unprepared or slow one loses people.
  - Example: blackouts let carnivores out unless you have a Backup Generator. With a backup: 0 of 3
    Dilophosaurus escaped. Without: 3 of 3.
- **Power shortage triage:** fences first, then staff, then shelters and sirens, then hatchery, restrooms,
  restaurants and Visitor Center, then lamps, then everything else. Guests starving in the dark was the
  worst failure mode in playtests.
- **The relocation room rule is 30% of species space.** 60% made The Lost World unwinnable with a small
  holding pen. Below ~18 tiles, a T. rex was being dropped into 15-tile scraps of land.
- **Hotel capacity of 40 rooms**, enforced both when guests choose and at the door. Uncapped, it earned
  ~$24k a day and paid back in under 4 days, which dominated everything.
- **Escapes in Chapter 2 count only raptors.** Storm-flattened herbivore fences were failing a raptor
  chapter.
- **Calm has no disasters, but power overload can still release carnivores.** That's a consequence, not a
  disaster. Bots never trip it in Calm.
- **InGen calls are deferred during crises.** A modal call paused the game mid-breakout.
- **Do-nothing parks may only barely win money goals.** When the uncapped hotel made idle wins easy, that
  was treated as a bug.

## 4. Dead ends (don't repeat)

- **Opening Day's original breakout:** two wall tiles broken, with the Indominus somewhere in her paddock.
  Maintenance patched the hole before she noticed, so the headline event never happened. Fix: she is moved
  out through the hole.
- **"Rampage on buildings" with the nearest building as target:** loose giants chewed their own feeder for
  8+ hours. Feeders are excluded now.
- **The weakest-fence search initialised its best score at -1:** walls score -50, so walls were never
  chosen even when forced. It's `-Infinity` now.
- **A Lagoon added to the guest goal list without filtering to existing buildings:** wasted guest choices in
  small parks and cut shop income 15–30%. The list is filtered now.
- **Hotel room cap checked only when guests choose:** many chose at once and occupancy hit 127 of 40. It's
  enforced at the door too.
- **Making Chaos deadlier for perfect responders** with crowd crush, trips, darts missing, shorter raptor
  idle time and reachable-prey targeting. All are good changes, but Chaos guest deaths stayed ~0, because
  escapes happen at night or the alarm empties the paths first. Bigger levers are needed (§7).
- **Earlier sessions, from commit messages:**
  - Hotel at $200 a night paid back in 4 days, so it went to $150 (then rooms were added).
  - Gyrosphere originally lost ~$200 a day.
  - Clever Girls at 10 shows was done by Day 3, so it went to 24.
  - Storm Season's original $650k target.

## 5. Gotchas

- **Save format:** `serialize` and `loadGame` in `src/main.js` are hand-maintained.
  - Any new persistent field must be added to both, and to the CI test "save/load keeps dinosaurs and park
    state" in `tools/ci.js`.
  - A `//` comment inside a one-line statement once silently dropped dinosaur growth and age on load.
- **Regions and power are recomputed from dirty flags.** After scripted edits, call `w.computeRegions()`
  and `w.computePower(g)` before relying on them.
- **`findRect` in the play helpers searches near the start**, so big parks run out of room. That's a bot
  limit, not a game bug.
- **The hatchery needs power,** and `canHatchAt` returns the reason. Bot "hatch failures" were usually an
  overloaded grid.
- **Scripted mouse clicks:** the first click right after `location.reload()` can land before the title is
  ready. Click by selector (`#btnSandbox`), not by coordinates.
- **Toasts use real-time `setTimeout`,** so screenshots after fast-forwarding in a tight loop show stale
  toasts. That's an artefact, not a bug.
- **Screen coordinates:** `sx = (tx * 16 - cam.x) * zoom`, with `TILE = 16`. The camera is at
  `UIX.r.cam`, not `UIX.cam`.
- **Bottom-left panel layout:** the CSS vars `--goalR` and `--goalH` position the squad bar and the ticker.
  They follow whichever panel is visible, the goal panel or the tutorial panel (`#tut`).
- **Killing processes:** `pkill -f` with a pattern that matches your own shell kills the shell (exit 144).
  Find the PID with `ps -eo pid,args | grep "[m]server.js"` and kill that.
- **Shell edits:** Python `str.replace` scripts are safer than `sed` for multi-line edits. Assert the old
  string exists, because a mis-sliced replace once duplicated half of `campaign.js`.
- **Ignored paths:** the repo `.gitignore` ignores `dist/`. The PR-sync recipe force-adds `icons/`, because
  a broader ignore applies on the PR branch.
- **Browser rendering:** headless Chromium renders in software, so per-frame timings in JS can mislead.
  Measure real fps with `requestAnimationFrame` intervals.
- **Background servers:** the cloud container kills them after 2 hours. Restart the play server as needed.

## 6. Owner preferences

- **"Play more"** is the standing request. Each round: pick an angle, playtest, find concrete issues, fix,
  verify with numbers and screenshots, ship, report. Rounds with nothing to fix are fine, but say so
  honestly and ship nothing.
- **Wants drama and deaths.** "need more deaths though? of guests?" Breaches should be scary.
- **Wants less busywork,** RTS-style control that is optional, and a campaign.
- **Report format:** a short summary per round. What was wrong, the fix, the measured before and after
  numbers, then "pushed, synced to PR #6, live in the artifact".
- **Ship every round:**
  1. Commit.
  2. Push the game branch.
  3. Sync it into PR #6's `jurassic_tycoon/` folder.
  4. Rebuild the bundle.
  5. Republish the artifact.
- **CI notifications:** on a CI-green notification for PR #6, just say so briefly.
- **Commit attribution:** commits end with the co-author/session trailer lines (see `git log`). Never put
  model names in commits or code.
- **Don't open PRs or take outward actions** unless asked. The PR already exists.
- **Not yet answered:** whether to make Chaos deadlier for a well-prepared park (§7). Ask before doing it.

## 7. Next steps (priority order)

1. **Ask the owner about Chaos lethality.** Options:
   - more escapes in daylight hours;
   - shelters admit people gradually, so queues form at the doors;
   - a short reaction delay before guests respond to the alarm.

   Currently a perfectly alert park on Chaos takes ~0 guest deaths.
2. **Re-confirm Chapter 5 gold (Day 45)** is reachable after the hotel cap.
   - Improve the planner bot (`c5b.js`) to scale restaurants and restrooms with guests and to manage power.
   - If gold is out of reach for competent play, loosen it slightly.
3. **Re-run the scenario and chapter sweep after any economy or behaviour change.** Re-checks that went
   stale are what caught the guest-goal and hotel regressions.
4. **Optional polish:**
   - a scroll hint on the phone Build flyout;
   - a "rooms" hint on the hotel build card;
   - the paddock template preview could show feeders by icon.
5. **Consider moving the play harness into the repo** under `tools/play/`. Today it lives only in the
   session scratchpad (§8) and is lost between sessions.

Open questions:
- Should a well-run park still lose guests on Chaos?
- Is Chapter 5 gold at Day 45 the right bar?
- Should Calm mode prevent overload breakouts entirely?

## 8. Pointers

### Branches and URLs
- **`jurassic-park-tycoon`:** all game work, on remote `origin` (`https://github.com/mszarski/claude_playground`).
- **`jurassic-tycoon-pr`:** a mirror of the game under `jurassic_tycoon/`, the head of **draft PR #6**.
  Sync it after each push:

  ```bash
  git checkout -q jurassic-tycoon-pr && git pull -q origin jurassic-tycoon-pr && git rm -rq jurassic_tycoon && \
  mkdir -p jurassic_tycoon && git --work-tree=jurassic_tycoon checkout jurassic-park-tycoon -- . && git reset -q && \
  rm -rf jurassic_tycoon/.github && git add -A jurassic_tycoon && git add -f jurassic_tycoon/icons && \
  git commit -qm "Sync: ..." && git push -q origin jurassic-tycoon-pr; git checkout -q jurassic-park-tycoon
  ```

- **GitHub Pages (PWA):** https://mszarski.github.io/claude_playground/. Deployed by
  `.github/workflows/jurassic-tycoon.yml` on every push to `jurassic-park-tycoon`: the `test` job, then the
  `deploy` job, which pushes `tools/build-pages.js` output to `gh-pages`.
- **claude.ai artifact (single-file build):** https://claude.ai/artifact/FGPyGT92AZStNCUThgALdD.
  - Build: `node tools/bundle.js` → `dist/jurassic-tycoon.html`.
  - Publish that file to the same artifact URL.

### Code map (`src/`)
- **Core:**
  - `data.js`: species, buildings, tools, `FENCE_DEF`, `TOOL_GROUPS`.
  - `game.js`: the `Game` class, economy, goals, rating, advisor (`collectIssues`), relocation
    (`findPaddockFor`, `moveStrays`), power and saves glue.
  - `world.js`: terrain, regions, the power grid (`computePower`, triage order), BFS, flow fields.
  - `main.js`: start, save/load, menus, the service worker.
- **Entities (`entities/`):**
  - `dino.js`: AI; `decide`, fence attacks, scheming, hunting, `findPrey`.
  - `guest.js`: needs, `pickGoal`, fleeing, crowd crush, names.
  - `staff.js`: rangers and engineers.
  - `vehicles.js`: helicopters, jeeps, trucks.
  - `ptera.js`, plus `common.js` for name pools.
- **Rules and content:**
  - `attractions.js`: gyrospheres, arena, feeding shows, Lagoon.
  - `events.js`: disasters, storms, outages, lava.
  - `scenarios.js` and `campaign.js`: `SCENARIOS`, `CHAPTERS`, perks; `spawnIn` fixes the starter
    paddock's feeder diet.
  - `decisions.js`: InGen calls. `research.js`. `blueprints.js`: paddock templates, repair, cover, feeders.
- **Interface:**
  - `ui.js`: input, tools, drag, HUD, alerts, ticker, goal panel.
  - `ui-panels.js`: inspect panels, Park Rating, finances, research, How to Play.
  - `render.js`, `sprites.js` and `sprites-views.js`: drawing.
  - `tutorial.js`, `audio.js`, `command.js` (RTS orders).

### Tests (`tools/`)
- `node tools/ci.js`: what CI runs. Boots every mode, completes the tutorial, fuzz plus save/load, the
  save/load state test, every panel renders.
- `node tools/rts.js`: RTS controls with a real mouse.
- `node tools/smoke.js <out.png> <script.js>`: runs any browser script with `tools/helpers.js` (the `T`
  helpers) prepended. For example:
  - `tools/tutorial.js`: walks the tutorial.
  - `tools/decisions.js`: every InGen choice.
  - `tools/sorna.js`, `tools/attractions.js`, `tools/tour.js`, `tools/long.js`, `tools/fuzz.js`.
- Playwright: `tools/pw.js` resolves it. CI installs `playwright@1.56.1`.

### Play harness (scratchpad only, not in the repo)
A Playwright server, `server.js`, on port 8777 loads `index.html` and a helper library, `lib.js`, which
defines `window.PL`:

- `PL.line`, `PL.paddock`, `PL.pylons`, `PL.near`, `PL.reg`, `PL.hatchIn`, `PL.route`, `PL.findRect`,
  `PL.starter`, `PL.containPrep`, `PL.state`.
- `do.sh '<json actions>'` posts actions such as `eval`, `click`, `drag`, `key`, `wait`, `shot`.
- A phone variant, `mserver.js`, on port 8778 adds the `tap`, `tdrag` and `pinch` actions.
- Bot scripts:
  - `goalbot.js`: follows the free-play goals.
  - `late.js` / `late2.js`: a late-game park with ROI.
  - `att3.js`: attraction payback.
  - `sweep.js`: scenarios idle or responding.
  - `siege.js`, `pack.js`, `rexout.js`, `lw.js`, `c5b.js` (Chapter 5 planner), `req5.json` (the scenario
    balance sweep).

Recreating this harness in `tools/play/` is next step 5.

### Key data
- `STAR_THRESH = [0, 55, 110, 165, 235, 330]`; 8 rating points per species.
- **Scenario objectives:**
  - Nedry: survive to Day 3 08:00 with fewer than 4 casualties.
  - Sorna: contain all dinosaurs by Day 10.
  - Lost World: ship 6 including the T. rex by Day 6.
  - Opening Day: 1,400 guests by Day 5 with fewer than 6 casualties.
  - Siege: lose fewer than 8 before Day 3 12:00.
  - Storm Season: earn $300k from guests by Day 12.
- **Chapters:**
  - C1: 2 stars, 120 guests at once and $150k from guests by Day 15.
  - C2: 3 raptors and 24 shows by Day 12, with 4 or fewer raptor escapes.
  - C3: $275k from guests and reputation 50+ by Day 26.
  - C4: contain all and 3 stars by Day 18.
  - C5: 5 stars, a T. rex, a Lagoon and 400 guests at once (gold by Day 45, silver by Day 60).
- **localStorage keys:** `jt_save`, `jt_campaign`, `jt_medals`, `jt_tutorial_done`, `jt_daylen`,
  `jt_mapsize`, `jt_audio_levels`, `jt_wheel`.
