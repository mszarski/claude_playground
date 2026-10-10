// ---------- Guided tutorial: highlights UI and waits for the player to do each step ----------
'use strict';

const TUTORIAL_STEPS = [
  {
    text: 'Welcome, park director! This short tutorial builds a tiny park and then lets something escape. Drag the map (or use WASD) to look around, and scroll to zoom.',
    next: true,
  },
  {
    text: 'Electric fences need power. Open the <b>Power</b> menu and place a <b>Power Plant</b> on open grass near the gate.',
    target: '.cat[data-cat="infra"]', tool: 'power',
    check: (g) => g.hasBuilding('power'),
  },
  {
    text: 'The blue area is power coverage. Choose the <b>Paddock</b> tool and drag a rectangle (about 10×8) <b>inside</b> the blue area. Dead fences blink red.',
    target: '.cat[data-cat="build"]', tool: 'paddock',
    check: (g) => g.world.regions.some((r) => !r.public && r.size >= 20 && r.size < 1200 && tutPaddockPowered(g, r)),
    hint: 'Make sure every side of the fence sits inside the blue coverage. You can press U to undo.',
  },
  {
    text: 'Dinosaurs get hungry. Open <b>Dinos</b> and place a <b>Herbivore Feeder</b> inside your paddock.',
    target: '.cat[data-cat="dino"]', tool: 'feeder_h',
    check: (g) => g.world.regions.some((r) => !r.public && r.size < 1200 && r.feeders.herb.length > 0),
  },
  {
    text: 'Now build a <b>Hatchery</b> (also in Dinos). It must be inside power coverage too.',
    target: '.cat[data-cat="dino"]', tool: 'hatchery',
    check: (g) => g.buildingsOfType('hatchery').some((b) => b.powered),
    hint: 'If it shows a yellow "!" it has no power: move it closer or add a Power Pylon.',
  },
  {
    text: 'Hatch two <b>Gallimimus</b>: press <b>H</b> (or Dinos → Hatch Dinosaur), pick Gallimimus, then click inside the paddock twice. They like company.',
    target: '.cat[data-cat="dino"]', tool: 'hatch',
    check: (g) => g.dinos.filter((d) => d.species === 'galli').length + g.eggs.filter((e) => e.species === 'galli').length >= 2,
  },
  {
    text: 'Guests arrive at the gate and only walk on paths. Build a <b>Visitor Center</b> (Guests menu) and connect it to the gate with <b>Smart Path</b>: drag from the gate path to the building.',
    target: '.cat[data-cat="guest"]', tool: 'visitor',
    check: (g) => g.guests.length >= 5,
    hint: 'Guests only come once there is something connected to the gate by paths. Run a path past the paddock so they can see the dinosaurs.',
  },
  {
    text: 'Every park needs a <b>Ranger Station</b> (Staff menu). Rangers tranquilize anything that gets out.',
    target: '.cat[data-cat="staff"]', tool: 'ranger',
    check: (g) => g.hasBuilding('ranger'),
  },
  {
    text: 'Uh oh. A fence just failed and a dinosaur is loose! Press <b>⚠ ALARM</b> (or E) to send guests to shelter.',
    target: '#alarmBtn',
    enter: (g) => tutStageBreach(g),
    check: (g) => g.alarm,
  },
  {
    text: 'Good. Now watch the breach cam: your rangers will dart the escapee. Then fix the fence: click the broken (red rubble) tile with Inspect and choose <b>Emergency repair</b>, or wait for engineers if you have a Maintenance Shed.',
    target: '#pip',
    check: (g) => !g.looseCount && g.countBrokenFences() === 0,
    hint: 'Sedated escapees are trucked back to a paddock by the ground crew (a Helipad is faster). Click the alert at the top to find the loose dinosaur.',
  },
  {
    text: 'Contained! Turn off the alarm so guests come back out (press ⚠ ALARM again).',
    target: '#alarmBtn',
    check: (g) => !g.alarm,
  },
  {
    text: 'That is the job: build, power, feed, and react fast when things go wrong. Follow the goals in the bottom-left to grow your park. Good luck!',
    next: true, last: true,
  },
];

function tutPaddockPowered(g, r) {
  const w = g.world;
  for (const t of r.tiles) {
    const x = t % w.W, y = (t / w.W) | 0;
    for (const [dx, dy] of DIRS4) {
      const nx = x + dx, ny = y + dy;
      if (!w.inb(nx, ny)) continue;
      const j = w.idx(nx, ny);
      if (w.fence[j] === F_ELECTRIC && !w.fencePowered[j]) return false;
    }
  }
  return true;
}

function tutStageBreach(g) {
  const w = g.world;
  const d = g.dinos.find((x) => x.species === 'galli' && !x.loose) || g.dinos[0];
  if (!d) return;
  // knock out the fence tile nearest to the dinosaur and nudge it through
  const p = w.bfs(d.tx, d.ty, w.dinoPass, (x, y) => isSolidFence(w.fence[w.idx(x, y)]), 3000);
  if (!p || !p.length) return;
  const [fx, fy] = p[p.length - 1];
  g.damageFence(fx, fy, 99999, null);
  d.stress = 80; d.x = fx + 0.5; d.y = fy + 0.5;
  w.computeRegions(); g.updateLoose();
}

class Tutorial {
  constructor(ui) {
    this.ui = ui; this.i = -1; this.done = false; this.t = 0;
    this.panel = document.getElementById('tut');
    this.hl = document.getElementById('tutHl');
    this.go(0);
  }
  get step() { return TUTORIAL_STEPS[this.i]; }
  go(i) {
    this.i = i; this.t = 0; this.hintShown = false;
    const s = this.step, g = this.ui.game;
    if (!s) { this.finish(); return; }
    if (s.enter) s.enter(g);
    this.render();
    this.ui.sfx.play(i ? 'fanfare' : 'click');
  }
  render() {
    const s = this.step;
    const n = TUTORIAL_STEPS.length;
    this.panel.hidden = false;
    this.panel.innerHTML = `<b>TUTORIAL ${this.i + 1}/${n}</b><div>${s.text}</div><div id="tutHint" class="sub"></div>
      <div class="btns">${s.next ? `<button id="tutNext">${s.last ? 'Start playing' : 'Next'}</button>` : ''}${s.tool ? '<button id="tutTool">Select it for me</button>' : ''}<button id="tutSkip">Skip tutorial</button></div>`;
    const nb = document.getElementById('tutNext'); if (nb) nb.onclick = () => (s.last ? this.finish() : this.go(this.i + 1));
    const tb = document.getElementById('tutTool'); if (tb) tb.onclick = () => this.selectTool(s.tool);
    document.getElementById('tutSkip').onclick = () => this.finish();
  }
  selectTool(t) {
    const ui = this.ui;
    if (t === 'hatch') { ui.species = 'galli'; ui.setTool('hatch'); ui.toast('Gallimimus selected: click inside your paddock.', 'info'); return; }
    ui.setTool(t);
    const grp = TOOL_GROUPS.find((gr) => gr.tools.includes(t));
    if (grp && window.innerWidth >= 760) ui.openFlyout(grp);
  }
  update(dt) {
    if (this.done) return;
    const s = this.step; if (!s) return;
    this.t += dt;
    if (s.check && s.check(this.ui.game)) { this.go(this.i + 1); return; }
    if (s.hint && !this.hintShown && this.t > 25) { this.hintShown = true; const h = document.getElementById('tutHint'); if (h) h.innerHTML = '💡 ' + s.hint; }
    // highlight target element
    const el = s.target && document.querySelector(s.target);
    if (el && el.offsetParent !== null && !el.hidden) {
      const r = el.getBoundingClientRect();
      Object.assign(this.hl.style, { left: r.left - 5 + 'px', top: r.top - 5 + 'px', width: r.width + 10 + 'px', height: r.height + 10 + 'px' });
      this.hl.hidden = false;
    } else this.hl.hidden = true;
  }
  finish() {
    this.done = true;
    this.panel.hidden = true; this.hl.hidden = true;
    const g = this.ui.game;
    g.tutorial = false;
    try { localStorage.setItem('jt_tutorial_done', '1'); } catch (e) { /* storage unavailable */ }
    document.getElementById('goal').hidden = false;
    this.ui.tutorial = null;
  }
}
