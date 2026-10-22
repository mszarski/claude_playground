// ---------- Game state & simulation ----------
'use strict';

const GOALS = [
  { id: 'hatchery', tool: 'hatchery', text: 'Build a Hatchery', reward: 10000, check: (g) => g.hasBuilding('hatchery') },
  { id: 'power', tool: 'power', text: 'Build a Power Plant', reward: 10000, check: (g) => g.hasBuilding('power') },
  { id: 'paddock', tool: 'paddock', text: 'Fence a paddock (use the Paddock tool) and add a feeder inside', reward: 15000,
    check: (g) => g.world.regions.some((r) => !r.public && r.size >= 20 && (r.feeders.herb.length + r.feeders.carn.length) > 0) },
  { id: 'hatch1', tool: 'hatch', text: 'Hatch your first dinosaur', reward: 20000, check: (g) => g.dinos.length > 0 },
  { id: 'guests20', tool: 'visitor', text: 'Connect paths to attractions and welcome 20 guests', reward: 15000, check: (g) => g.guests.length >= 20 },
  { id: 'ranger', tool: 'ranger', text: 'Build a Ranger Station (rangers tranquilize escapees)', reward: 10000, check: (g) => g.hasBuilding('ranger') },
  { id: 'maint', tool: 'maint', text: 'Build a Maintenance Shed (engineers repair fences)', reward: 10000, check: (g) => g.hasBuilding('maint') },
  { id: 'species3', tool: 'hatch', text: 'Exhibit 3 different species', reward: 30000, check: (g) => g.speciesCount() >= 3 },
  { id: 'star2', text: 'Reach a 2-star park rating', reward: 40000, check: (g) => g.stars >= 2 },
  { id: 'carn', tool: 'hatch', text: 'Hatch a carnivore (keep it away from herbivores!)', reward: 30000, check: (g) => g.dinos.some((d) => d.sp.diet === 'carn') },
  { id: 'helipad', tool: 'helipad', text: 'Build an ACU Helipad', reward: 25000, check: (g) => g.hasBuilding('helipad') },
  { id: 'guests150', text: 'Have 150 guests in the park at once', reward: 60000, check: (g) => g.guests.length >= 150 },
  { id: 'trex', tool: 'hatch', text: 'Exhibit a Tyrannosaurus rex', reward: 80000, check: (g) => g.dinos.some((d) => d.species === 'trex' && !d.loose) },
  { id: 'star4', text: 'Reach a 4-star rating', reward: 100000, check: (g) => g.stars >= 4 },
  { id: 'million', text: 'Have $2,000,000 in the bank', reward: 0, check: (g) => g.money >= 2000000 },
  { id: 'star5', text: 'Reach a 5-star rating: the greatest park on Earth', reward: 250000, check: (g) => g.stars >= 5 },
];

const STAR_THRESH = [0, 55, 110, 165, 245, 350];

class Game {
  constructor(seed, opts = {}) {
    this.seed = seed;
    RNG = mulberry32(seed ^ 0x9e3779b9);
    // scenarios and the tutorial keep the island size they were balanced on
    const scDef = opts.scenario && typeof SCENARIOS !== 'undefined' ? SCENARIOS[opts.scenario] : null;
    this.mapSize = MAP_SIZES[opts.size] ? opts.size : scDef && scDef.size ? scDef.size : opts.scenario || opts.tutorial ? 'small' : 'medium';
    const ms = MAP_SIZES[this.mapSize];
    this.world = new World(seed, ms.w, ms.h);
    this.world.generate();
    this.events = new Events(this);
    this.research = new ResearchState(this);
    this.research.apply();
    this.money = 450000;
    this.time = 7.5; // hours since start (day 1 07:30)
    this.secPerHour = 2.5;
    this.dinos = []; this.guests = []; this.staff = []; this.helis = []; this.eggs = []; this.jeeps = []; this.pteros = []; this.gyros = [];
    this.particles = []; this.floats = []; this.darts = [];
    this.logs = [];
    this.reputation = 50; this.stars = 0; this.attraction = 0; this.safety = 100;
    this.ticket = 100; this.priceMul = 1;
    this.alarm = false; this.sirenAlarm = false;
    this.shake = 0;
    this.goalIdx = 0;
    this.deathLog = []; // days of guest deaths
    this.stats = { guestsTotal: 0, deaths: 0, staffDeaths: 0, escapes: 0, dinoDeaths: 0, fledToday: 0, hatched: 0 };
    this.ledger = this.newLedger();
    this.history = [];
    this.lastDay = 1; this.lastIncidentDay = 1;
    this.spawnAcc = 0;
    this.comfortT = 0; this.powerT = 0; this.secondT = 0;
    this.typeCache = null;
    this.gameOver = null;
    this.victory = false;
    this.listeners = {};
    this.soundHook = null;
    this.looseCount = 0;
    this.difficulty = opts.difficulty || 'normal';
    const DIFF = { calm: { money: 650000, evMul: 0 }, easy: { money: 650000, evMul: 0.5 }, normal: { money: 450000, evMul: 1 }, chaos: { money: 400000, evMul: 1.8 } }[this.difficulty] || { money: 450000, evMul: 1 };
    this.money = DIFF.money; this.events.evMul = DIFF.evMul;
    if (this.difficulty === 'calm') this.events.auto = false;
    this.secPerHour = { short: 1.67, normal: 2.5, long: 3.75 }[opts.dayLength || 'normal'] || 2.5;
    if (opts.skipSetup) this.start = this.world.findStartSpot();
    else this.setupStart();
    this.world.computeRegions();
    this.world.computePower(this);
    if (opts.scenario && SCENARIOS[opts.scenario]) {
      this.scenario = opts.scenario;
      SCENARIOS[opts.scenario].setup(this);
      this.lastIncidentDay = this.day;
      this.goalIdx = GOALS.length; // scenarios replace the goal chain
    }
  }

  on(ev, fn) { (this.listeners[ev] = this.listeners[ev] || []).push(fn); }
  emit(ev, ...a) { for (const f of this.listeners[ev] || []) f(...a); }

  newLedger() { return { tickets: 0, shops: 0, grants: 0, upkeep: 0, food: 0, repairs: 0, construction: 0, dinos: 0, fines: 0, ops: 0 }; }

  get day() { return Math.floor(this.time / 24) + 1; }
  get hour() { return this.time % 24; }
  get seasonIdx() { return Math.floor((this.day - 1) / 7) % 4; }
  get season() { return SEASONS[this.seasonIdx]; }
  get isNight() { const h = this.hour; return h >= 20 || h < 6; }
  get darkness() {
    const h = this.hour;
    if (h >= 6 && h < 18) return 0;
    if (h >= 18 && h < 21) return (h - 18) / 3;
    if (h >= 21 || h < 4) return 1;
    return 1 - (h - 4) / 2;
  }

  setupStart() {
    const w = this.world;
    const s = w.findStartSpot();
    this.start = s;
    // clear an area
    for (let dy = -3; dy <= 3; dy++) for (let dx = -4; dx <= 4; dx++) {
      const i = w.idx(s.x + dx, s.y + dy);
      if (w.terrain[i] === T_FOREST || w.terrain[i] === T_SAND) w.terrain[i] = T_GRASS;
    }
    // gate at s, path heading north
    const gx = s.x - 1, gy = s.y;
    for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 3; dx++) {
      const i = w.idx(gx + dx, gy + dy); if (!w.isLand(w.terrain[i])) w.terrain[i] = T_GRASS;
    }
    const gate = w.addBuilding('gate', gx, gy); gate.built = 1;
    // the starting path must not run into rock: clear the ground under it
    const startPath = [];
    for (let y = s.y - 1; y >= s.y - 5; y--) startPath.push([s.x, y]);
    for (let x = s.x - 3; x <= s.x + 3; x++) startPath.push([x, s.y - 5]);
    for (const [x, y] of startPath) {
      const i = w.idx(x, y);
      if (w.terrain[i] === T_ROCK || w.terrain[i] === T_FOREST || w.terrain[i] === T_SAND) w.terrain[i] = T_GRASS;
    }
    for (const [x, y] of startPath) this.placePath(x, y, true);
  }

  // ------------- money -------------
  spend(v, cat = 'construction') { this.money -= v; this.ledger[cat] = (this.ledger[cat] || 0) + v; }
  earn(v, cat = 'shops') {
    this.money += v; this.ledger[cat] = (this.ledger[cat] || 0) + v;
    if (cat === 'tickets' || cat === 'shops') this.stats.guestIncome = (this.stats.guestIncome || 0) + v;
  }
  canAfford(v) { return this.money - v >= -20000; }

  // ------------- logging & fx -------------
  log(msg, type = 'info', ref = null, big = false) {
    const entry = { msg, type, day: this.day, hour: this.hour, ref: ref ? { x: ref.x !== undefined ? ref.x : 0, y: ref.y !== undefined ? ref.y : 0, entity: ref } : null, big, t: performance.now() };
    this.logs.push(entry);
    if (this.logs.length > 200) this.logs.shift();
    this.emit('log', entry);
  }
  sound(name) { if (this.soundHook) this.soundHook(name, 1); }
  soundAt(name, x, y, vol) { if (this.soundHook) this.soundHook(name, vol, x, y); }
  burst(x, y, color, n) {
    for (let i = 0; i < n; i++) this.particles.push({ x, y, vx: randf(-2, 2), vy: randf(-3, -0.5), life: randf(0.4, 0.9), max: 0.9, color, size: 1, grav: 8 });
  }
  sparks(x, y, color = '#f8f070', n = 6) {
    for (let i = 0; i < n; i++) this.particles.push({ x, y, vx: randf(-3, 3), vy: randf(-3, 1), life: randf(0.15, 0.4), max: 0.4, color, size: 1, grav: 2, glow: true });
  }
  addFloat(x, y, text, color) { this.floats.push({ x, y, text, color, life: 1.6 }); }

  // ------------- queries -------------
  buildingsOfType(type) {
    if (!this.typeCache || this.typeCacheV !== this.world.pathVersion) {
      this.typeCache = {};
      for (const b of this.world.buildings.values()) (this.typeCache[b.type] = this.typeCache[b.type] || []).push(b);
      this.typeCacheV = this.world.pathVersion;
    }
    return this.typeCache[type] || [];
  }
  hasBuilding(type) { return this.buildingsOfType(type).length > 0; }
  speciesCount() { return new Set(this.dinos.filter((d) => !d.loose).map((d) => d.species)).size; }
  creatures() { return this.pteros.length ? this.dinos.concat(this.pteros) : this.dinos; }
  *humans() {
    for (const g of this.guests) if (!g.hidden && !g.dead) yield g;
    for (const s of this.staff) if (!s.dead) yield s;
  }
  threatNear(x, y, r) {
    for (const d of this.creatures()) {
      if (!d.loose || d.carried || d.sedatedT > 0) continue;
      if (d.sp.danger < 2) continue;
      const sens = this.research.has('sensors');
      // raptors stalk: you don't see them until they are close (motion sensors spoil the ambush)
      const rr = d.camouflaged && !sens ? Math.min(r, 2.5) : sens ? r * 1.4 : d.sp.clever ? r * 0.6 : r;
      if (dist2(x, y, d.x, d.y) < rr * rr) return d;
    }
    return null;
  }
  litNear(x, y) {
    for (const l of this.buildingsOfType('lamp')) if (dist2(x, y, l.x + 0.5, l.y + 0.5) < 25) return true;
    return false;
  }
  nearBuilding(cx, cy, type, r) {
    for (const b of this.buildingsOfType(type)) {
      if (cx >= b.x - r && cx < b.x + b.w + r && cy >= b.y - r && cy < b.y + b.h + r) return b;
    }
    return null;
  }
  inSirenRange(x, y) {
    for (const s of this.buildingsOfType('siren')) if (s.powered && dist2(x, y, s.x + 0.5, s.y + 0.5) < BUILDINGS.siren.range ** 2) return true;
    return false;
  }
  countBrokenFences() { let n = 0; const f = this.world.fence; for (let i = 0; i < f.length; i++) if (f[i] === F_BROKEN) n++; return n; }
  recentDeaths(days = 7) { return this.deathLog.filter((d) => d > this.day - days).length; }

  buildingField(b) { const w = this.world; return w.flowField('b' + b.id, () => w.accessTiles(b)); }
  gateField() {
    const w = this.world;
    return w.flowField('gate', () => { const out = []; for (const b of this.buildingsOfType('gate')) out.push(...w.accessTiles(b)); return out; });
  }
  shelters() {
    // a shelter whose doors were broken in is no refuge until it is repaired
    return this.buildingsOfType('visitor').concat(this.buildingsOfType('hotel'), this.buildingsOfType('shelter')).filter((b) => !b.doorsBroken);
  }
  shelterField() {
    const w = this.world;
    return w.flowField('shelter', () => { const out = []; for (const b of this.shelters()) out.push(...w.accessTiles(b)); return out; });
  }
  anyShelterSpace() { return this.shelters().some((b) => b.inside < BUILDINGS[b.type].shelter); }
  nearShelterWithSpace(cx, cy) {
    const w = this.world, here = w.idx(cx, cy);
    for (const b of this.shelters()) {
      if (b.inside >= BUILDINGS[b.type].shelter) continue;
      if (w.accessTiles(b).includes(here)) return b;
    }
    return null;
  }

  // ------------- undo -------------
  startBatch() { this.curBatch = []; }
  recordUndo(e) { if (this.curBatch) this.curBatch.push(e); }
  endBatch(label) {
    if (this.curBatch && this.curBatch.length) {
      (this.undoStack = this.undoStack || []).push({ label, items: this.curBatch });
      if (this.undoStack.length > 25) this.undoStack.shift();
    }
    this.curBatch = null;
  }
  undo() {
    const w = this.world;
    const batch = (this.undoStack || []).pop();
    if (!batch) return null;
    let refund = 0;
    for (const e of batch.items.slice().reverse()) {
      const i = e.x !== undefined ? w.idx(e.x, e.y) : -1;
      let ok = false;
      if (e.kind === 'path' && w.path[i]) { w.path[i] = 0; w.terrain[i] = e.prevT; ok = true; }
      else if (e.kind === 'track' && w.track[i] && w.fence[i] !== F_GATE) { w.track[i] = 0; w.terrain[i] = e.prevT; ok = true; }
      else if ((e.kind === 'fence' || e.kind === 'gate') && w.fence[i] === e.now) {
        w.fence[i] = e.prev; w.fenceHp[i] = e.prevHp; w.fenceOrig[i] = e.prevOrig;
        if (e.kind === 'gate') w.track[i] = 0;
        if (e.prevT !== undefined) w.terrain[i] = e.prevT;
        ok = true;
      }
      else if (e.kind === 'terrain' && w.terrain[i] === e.now) { w.terrain[i] = e.prev; ok = true; }
      else if (e.kind === 'bld') { const b = w.buildings.get(e.id); if (b) { this.removeBuildingFx(b); ok = true; } }
      else if (e.kind === 'unbld') {
        // put a demolished building back (and take back the demolition refund)
        if (w.canPlaceBuilding(e.type, e.bx, e.by)) {
          const b = w.addBuilding(e.type, e.bx, e.by); b.hp = e.hp; b.built = 1;
          const def = BUILDINGS[e.type];
          if (def.staff) for (let k = 0; k < def.staffN; k++) this.staff.push(new Staff(this, def.staff, b));
          if (e.type === 'helipad') this.helis.push(new Helicopter(this, b));
          if (e.type === 'tour' || e.type === 'gyro') { b.queue = []; b.jeepT = 0; }
          ok = true;
        }
      }
      else if (e.kind === 'unfence' && !w.fence[i] && !w.bld[i] && !w.path[i] && !this.dinos.some((d) => d.tx === e.x && d.ty === e.y)) { w.fence[i] = e.f; w.fenceHp[i] = e.hp; w.fenceOrig[i] = e.orig; ok = true; }
      else if (e.kind === 'unpath' && !w.path[i] && w.canBuildAt(e.x, e.y) && !w.track[i]) { w.path[i] = 1; ok = true; }
      else if (e.kind === 'untrack' && !w.track[i] && !w.path[i]) { w.track[i] = 1; ok = true; }
      else if (e.kind === 'egg') { const k = this.eggs.indexOf(e.egg); if (k >= 0) { this.eggs.splice(k, 1); ok = true; } }
      if (ok) { if (e.kind === 'egg') { this.money += e.cost; this.ledger.dinos -= e.cost; } else refund += e.cost; if (i >= 0) w.invalidate(e.x, e.y); }
    }
    this.money += refund; this.ledger.construction -= refund;
    for (const e of batch.items) if (e.kind === 'egg') refund += e.cost;
    return { label: batch.label, refund };
  }

  // ------------- construction -------------
  placePath(x, y, free = false) {
    const w = this.world;
    if (!w.canBuildAt(x, y)) return false;
    const i = w.idx(x, y);
    if (w.path[i] || w.track[i]) return false;
    if (w.fence[i] === F_BROKEN) return false;
    let c = 0;
    if (!free) { c = TOOL_INFO.path.cost + (w.terrain[i] === T_FOREST ? 40 : 0); if (!this.canAfford(c)) return false; this.spend(c); }
    this.recordUndo({ kind: 'path', x, y, prevT: w.terrain[i], cost: c });
    w.path[i] = 1;
    if (w.terrain[i] === T_FOREST) w.terrain[i] = T_GRASS;
    w.invalidate(x, y);
    return true;
  }
  placeTrack(x, y) {
    const w = this.world;
    if (!w.inb(x, y)) return false;
    const i = w.idx(x, y);
    if ((w.fence[i] === F_ELECTRIC || w.fence[i] === F_WALL) && !w.track[i] && !w.bld[i]) {
      // crossing a fence: build a gate jeeps can drive through
      const c = FENCE_DEF[F_GATE].cost + TOOL_INFO.track.cost;
      if (!this.canAfford(c)) return false;
      this.spend(c);
      this.recordUndo({ kind: 'gate', x, y, prev: w.fence[i], prevHp: w.fenceHp[i], prevOrig: w.fenceOrig[i], now: F_GATE, cost: c });
      w.track[i] = 1; w.fence[i] = F_GATE; w.fenceHp[i] = FENCE_DEF[F_GATE].hp; w.fenceOrig[i] = F_GATE;
      w.invalidate(x, y);
      return true;
    }
    if (!w.canBuildAt(x, y)) return false;
    if (w.track[i] || w.path[i] || w.fence[i] === F_BROKEN) return false;
    const c = TOOL_INFO.track.cost + (w.terrain[i] === T_FOREST ? 40 : 0);
    if (!this.canAfford(c)) return false;
    this.spend(c);
    this.recordUndo({ kind: 'track', x, y, prevT: w.terrain[i], cost: c });
    w.track[i] = 1;
    if (w.terrain[i] === T_FOREST) w.terrain[i] = T_GRASS;
    w.invalidate(x, y);
    return true;
  }
  // Fence tiles a dinosaur was standing on: built as soon as it steps off
  queueFence(x, y, type, hours = 72) {
    this.pendingFences = this.pendingFences || [];
    if (!this.pendingFences.some((p) => p.x === x && p.y === y)) this.pendingFences.push({ x, y, type, t: hours * this.secPerHour });
  }
  buildPendingFences(dt) {
    const w = this.world;
    this.pendingFences = this.pendingFences.filter((p) => {
      p.t -= dt;
      const i = w.idx(p.x, p.y);
      if (w.fence[i] === p.type || w.bld[i] || w.path[i]) return false;
      if (w.terrain[i] === T_LAVA || this.dinos.some((d) => d.tx === p.x && d.ty === p.y && !d.carried)) return p.t > 0;
      this.placeFence(p.x, p.y, p.type);
      return false;
    });
    if (!this.pendingFences.length) { w.computeRegions(); this.updateLoose(); w.powerDirty = true; }
  }
  // something that will clear by itself: a dinosaur standing there, or lava that will cool
  // Hungry animals in a paddock whose feeder was destroyed (lava, quakes, a rampage)
  hungryWithoutFood() {
    const w = this.world, out = [];
    for (const d of this.dinos) {
      if (d.loose || d.carried || d.hunger < 60) continue;
      const reg = w.regionAt(d.tx, d.ty);
      if (!reg || reg.public || reg.feeders[d.sp.diet].length) continue;
      if (d.sp.diet === 'herb' && reg.forest >= 4) continue; // can graze
      if (reg.feeders[d.sp.diet === 'carn' ? 'herb' : 'carn'].length) continue; // a stray in the other diet's paddock: it gets moved, not fed
      out.push({ d, reg });
    }
    return out;
  }
  // Animals fenced in with the other diet by accident (the paddock's feeders say what it is for): move them out
  moveStrays() {
    if (!this.hasBuilding('ranger')) return;
    const w = this.world;
    for (const d of this.dinos) {
      if (d.loose || d.carried || d.relocate || d.isPtera) continue;
      const reg = w.regionAt(d.tx, d.ty);
      const mine = d.sp.diet === 'carn' ? 'carn' : 'herb', other = mine === 'carn' ? 'herb' : 'carn';
      if (!reg || reg.public || !reg.feeders[other].length || reg.feeders[mine].length) continue;
      if (!this.findPaddockFor(d)) continue; // nowhere better to go yet
      d.relocate = true; d.orderSedate = true;
      this.log(`${d.name} the ${d.sp.name} is in the ${other === 'carn' ? 'carnivore' : 'herbivore'} paddock. Rangers will move it to a better one.`, 'info', d);
    }
  }
  replaceLostFeeders(force = false) {
    if (this.autoFeeders === false && !force) return 0;
    let n = 0;
    const done = new Set();
    for (const { d, reg } of this.hungryWithoutFood()) {
      const key = reg.id + d.sp.diet;
      if (done.has(key)) continue;
      done.add(key);
      const f = this.addFeeder(reg, d.sp.diet);
      if (f) { n++; this.log(`Ground crew installed a ${BUILDINGS[f.type].name} for hungry ${d.name} the ${d.sp.name} (${fmtMoney(BUILDINGS[f.type].cost)}).`, 'info', f); }
    }
    return n;
  }
  fenceBlockedByDino(x, y) { return this.world.terrain[this.world.idx(x, y)] === T_LAVA || this.dinos.some((d) => d.tx === x && d.ty === y && !d.carried); }

  placeFence(x, y, type) {
    const w = this.world;
    if (!w.inb(x, y)) return false;
    const i = w.idx(x, y);
    const t = w.terrain[i];
    if (!(t === T_SAND || t === T_GRASS || t === T_FOREST || t === T_BASALT)) return false;
    if (w.bld[i] || w.path[i]) return false;
    if (w.track[i]) {
      // fence across the tour track becomes a gate
      if (w.fence[i] === F_GATE) return false;
      type = F_GATE;
    }
    if (w.fence[i] === type) return false;
    if (this.dinos.some((d) => d.tx === x && d.ty === y && !d.carried)) return false;
    const c = FENCE_DEF[type].cost + (t === T_FOREST ? 40 : 0);
    if (!this.canAfford(c)) return false;
    this.spend(c);
    this.recordUndo({ kind: 'fence', x, y, prev: w.fence[i], prevHp: w.fenceHp[i], prevOrig: w.fenceOrig[i], prevT: w.terrain[i], now: type, cost: c });
    w.fence[i] = type; w.fenceHp[i] = FENCE_DEF[type].hp; w.fenceOrig[i] = type;
    if (t === T_FOREST) w.terrain[i] = T_GRASS;
    w.invalidate(x, y);
    return true;
  }
  demolishAt(x, y) {
    const w = this.world;
    if (!w.inb(x, y)) return false;
    const i = w.idx(x, y);
    const b = w.buildings.get(w.bld[i]);
    if (b) {
      if (b.type === 'gate' && this.buildingsOfType('gate').length <= 1) { this.log('You need at least one Main Gate.', 'warn'); return false; }
      const recent = b.placedAt !== undefined && this.time - b.placedAt < 2; // within ~5s at normal speed
      const refund = BUILDINGS[b.type].cost * (recent ? 1 : 0.25);
      this.money += refund; this.ledger.construction -= refund;
      this.recordUndo({ kind: 'unbld', type: b.type, bx: b.x, by: b.y, hp: b.hp, cost: -refund });
      this.removeBuildingFx(b);
      return true;
    }
    if (w.fence[i]) { this.recordUndo({ kind: 'unfence', x, y, f: w.fence[i], hp: w.fenceHp[i], orig: w.fenceOrig[i], cost: 0 }); w.fence[i] = F_NONE; w.fenceHp[i] = 0; w.fenceOrig[i] = 0; w.invalidate(x, y); return true; }
    if (w.path[i]) { this.recordUndo({ kind: 'unpath', x, y, cost: 0 }); w.path[i] = 0; w.invalidate(x, y); return true; }
    if (w.track[i]) { this.recordUndo({ kind: 'untrack', x, y, cost: 0 }); w.track[i] = 0; w.invalidate(x, y); return true; }
    return false;
  }
  removeBuildingFx(b) {
    const w = this.world;
    // eject guests inside
    for (const gu of this.guests) if (gu.inBuilding === b) { gu.inBuilding = null; gu.hidden = false; gu.state = 'walk'; }
    w.removeBuilding(b);
    this.burst(b.x + b.w / 2, b.y + b.h / 2, '#8a8a84', 14);
    for (const h of this.helis) if (h.pad === b) h.dead = true;
    for (const j of this.jeeps) if (j.station === b) j.remove();
    for (const gy of this.gyros || []) if (gy.station === b) gy.remove();
  }
  placeBuilding(type, x, y) {
    const w = this.world, def = BUILDINGS[type];
    if (!w.canPlaceBuilding(type, x, y)) return null;
    if (def.unique && this.hasBuilding(type) && type !== 'gate') { this.log(`You can only have one ${def.name}.`, 'warn'); return null; }
    if (def.unlockStars && this.stars < def.unlockStars && !this.sandbox) return null;
    for (const d of this.dinos) if (!d.carried && d.tx >= x && d.tx < x + def.w && d.ty >= y && d.ty < y + def.h) return null;
    if (!this.canAfford(def.cost)) { this.log('Not enough money!', 'warn'); return null; }
    this.spend(def.cost);
    const b = w.addBuilding(type, x, y);
    b.placedAt = this.time;
    this.recordUndo({ kind: 'bld', id: b.id, cost: def.cost });
    if (def.staff) for (let k = 0; k < def.staffN; k++) this.staff.push(new Staff(this, def.staff, b));
    if (type === 'helipad') this.helis.push(new Helicopter(this, b));
    if (type === 'tour' || type === 'gyro') { b.queue = []; b.jeepT = 0; }
    this.burst(x + def.w / 2, y + def.h / 2, '#e8d8b0', 12);
    return b;
  }
  plantTree(x, y) {
    const w = this.world;
    if (!w.inb(x, y)) return false;
    const i = w.idx(x, y);
    if (w.terrain[i] !== T_GRASS || w.path[i] || w.bld[i] || w.fence[i]) return false;
    if (!this.canAfford(TOOL_INFO.trees.cost)) return false;
    this.spend(TOOL_INFO.trees.cost);
    this.recordUndo({ kind: 'terrain', x, y, prev: w.terrain[i], now: T_FOREST, cost: TOOL_INFO.trees.cost });
    w.terrain[i] = T_FOREST; w.invalidate(x, y);
    return true;
  }
  clearLand(x, y) {
    const w = this.world;
    if (!w.inb(x, y)) return false;
    const i = w.idx(x, y);
    if (w.terrain[i] !== T_FOREST) return false;
    if (!this.canAfford(TOOL_INFO.clear.cost)) return false;
    this.spend(TOOL_INFO.clear.cost);
    this.recordUndo({ kind: 'terrain', x, y, prev: w.terrain[i], now: T_GRASS, cost: TOOL_INFO.clear.cost });
    w.terrain[i] = T_GRASS; w.invalidate(x, y);
    return true;
  }

  isUnlocked(species) { return this.stars >= SPECIES[species].unlock || this.sandbox || this.research.has('gen_' + species) || this.freeEgg === species; }
  hatchCost(species) { return this.freeEgg === species ? 0 : SPECIES[species].cost; }

  canHatchAt(species, x, y) {
    const w = this.world;
    const sp = SPECIES[species];
    if (!this.hasBuilding('hatchery')) return 'Build a Hatchery first.';
    if (!this.buildingsOfType('hatchery').some((b) => b.powered)) return 'The Hatchery has no power.';
    if (!this.isUnlocked(species)) return `Requires a ${sp.unlock}-star rating (or genome research).`;
    if (sp.flying) {
      const b = w.buildingAt(x, y);
      if (!b || b.type !== 'aviary') return 'Pteranodons hatch inside an Aviary. Click on one.';
      if (b.hp < b.maxHp * 0.6) return 'The Aviary needs repairs first.';
      const n = this.pteros.filter((p) => !p.dead && p.x >= b.x && p.x < b.x + b.w && p.y >= b.y && p.y < b.y + b.h).length + this.eggs.filter((e) => e.species === 'ptera').length;
      if (n >= 6) return 'This Aviary is full (6 Pteranodons).';
      if (!this.canAfford(this.hatchCost(species))) return 'Not enough money.';
      return null;
    }
    if (!w.dinoPass(x, y)) return 'Pick an open tile inside a paddock.';
    const reg = w.regionAt(x, y);
    if (!reg || reg.public) return 'Dinosaurs must hatch inside a fenced paddock with no paths.';
    if (reg.size > 1200) return 'That area is not enclosed.';
    if (!this.canAfford(this.hatchCost(species))) return 'Not enough money.';
    return null;
  }
  hatch(species, x, y) {
    const err = this.canHatchAt(species, x, y);
    if (err) { this.log(err, 'warn'); return false; }
    const sp = SPECIES[species], cost = this.hatchCost(species);
    this.spend(cost, 'dinos');
    const egg = { species, x, y, t: 8, max: 8 };
    if (this.freeEgg === species) { egg.unstable = true; this.freeEgg = null; }
    this.eggs.push(egg);
    this.recordUndo({ kind: 'egg', egg, cost });
    // no food in this paddock yet? put a feeder in for the new arrival
    const reg = sp.flying ? null : this.world.regionAt(x, y);
    if (reg && !reg.feeders[sp.diet].length && (sp.diet === 'carn' || reg.forest < 4) && this.autoFeeders !== false) {
      const f = this.addFeeder(reg, sp.diet, true);
      if (f) this.log(`A ${BUILDINGS[f.type].name} was added to the paddock (${fmtMoney(BUILDINGS[f.type].cost)}).`, 'info', f);
    }
    this.sound('build');
    return true;
  }

  // ------------- damage & death -------------
  damageFence(x, y, dmg, dino) {
    const w = this.world, i = w.idx(x, y);
    const f = w.fence[i];
    if (!isSolidFence(f)) return;
    w.fenceHp[i] -= dmg * this.mod('fenceDmgMul');
    w.markDirty(x, y);
    if (w.fenceHp[i] <= 0) {
      w.fence[i] = F_BROKEN; w.fenceHp[i] = 0;
      w.invalidate(x, y);
      this.burst(x + 0.5, y + 0.5, '#9a9a90', 12);
      this.soundAt('crash', x, y, 1);
      this.shake = Math.max(this.shake, 3);
      const who = dino ? `${dino.name} the ${dino.sp.name} broke through a fence!` : 'A fence has collapsed!';
      this.log(who, dino ? 'bad' : 'warn', { x: x + 0.5, y: y + 0.5, kind: 'fence', fx: x, fy: y }, !!dino);
    }
  }
  damageBuilding(b, dmg, dino) {
    if (!this.world.buildings.has(b.id)) return;
    b.hp -= dmg;
    // a shelter's doors give way long before its walls do
    if (dino && BUILDINGS[b.type].shelter && b.inside > 0 && b.hp < b.maxHp * 0.5 && !b.doorsBroken) {
      b.doorsBroken = true; this.world.flowCache.clear();
      this.log(`${dino.name} the ${dino.sp.name} broke into the ${BUILDINGS[b.type].name}! The guests inside are running for it.`, 'bad', dino, true);
      const acc = this.world.accessTiles(b);
      for (const gu of this.guests) if (gu.inBuilding === b) {
        gu.inBuilding = null; gu.hidden = false; gu.state = 'flee'; gu.fleeGate = true;
        if (acc.length) { const t = pick(acc); gu.cx = gu.nx = t % this.world.W; gu.cy = gu.ny = (t / this.world.W) | 0; gu.x = gu.cx + 0.5; gu.y = gu.cy + 0.5; }
      }
      b.inside = 0;
      this.soundAt('crash', b.x, b.y, 1);
    }
    if (b.hp <= 0) {
      this.log(`The ${BUILDINGS[b.type].name} was destroyed${dino ? ' by ' + dino.name + ' the ' + dino.sp.name : ''}!`, 'bad', { x: b.x + b.w / 2, y: b.y + b.h / 2 }, true);
      // guests inside are hurt? they flee
      for (const gu of this.guests) if (gu.inBuilding === b) { gu.inBuilding = null; gu.hidden = false; gu.state = 'flee'; }
      this.soundAt('crash', b.x, b.y, 1);
      this.removeBuildingFx(b);
    }
  }
  sellDino(d) {
    if (d.carried) return 'It is being airlifted.';
    if (d.loose && d.sedatedT <= 0) return 'Sedate it first. Nobody ships an angry dinosaur.';
    if (this.scenario === 'lostworld' && d.loose) return 'The buyers want it delivered from a paddock. Airlift it into one first.';
    const v = d.sellValue || Math.round(d.sp.cost * 0.35);
    this.earn(v, 'grants'); d.dead = true;
    this.stats.sold = (this.stats.sold || 0) + 1;
    (this.stats.soldSpecies = this.stats.soldSpecies || []).push(d.species);
    this.log(`${d.name} the ${d.sp.name} was shipped to the mainland (+${fmtMoney(v)}).`, 'info');
    return null;
  }

  // Player-ordered helicopter tranquilizer run
  acuStrike(d, heli = null) {
    const h = heli || this.helis.find((x) => x.state === 'parked' && !x.cargo) || this.helis.find((x) => x.state === 'return' && !x.cargo);
    if (!h) return 'All helicopters are busy.';
    const cost = this.research.has('acu') ? 3000 : 8000;
    if (!this.canAfford(cost)) return `Not enough money (${fmtMoney(cost)}).`;
    this.spend(cost, 'ops');
    h.target = d; h.state = 'strike'; h.shotT = 0.5;
    this.soundAt('heli', h.x, h.y, 1);
    this.log(`ACU helicopter scrambled to tranquilize ${d.name} the ${d.sp.name}.`, 'info', d, true);
    return null;
  }
  // Drag any sedated dinosaur out of a fence gap so it can be rebuilt. Returns true if the gap is clear.
  clearGap(x, y) {
    const w = this.world;
    for (const d of this.dinos) {
      if (d.carried || d.tx !== x || d.ty !== y) continue;
      if (d.sedatedT <= 0) return false;
      const spot = DIRS4.map(([dx, dy]) => [x + dx, y + dy]).find(([nx, ny]) => w.dinoPass(nx, ny) && w.fence[w.idx(nx, ny)] !== F_BROKEN);
      if (!spot) return false;
      d.x = spot[0] + 0.5; d.y = spot[1] + 0.5;
    }
    return true;
  }

  // Instant fence rebuild at triple cost
  emergencyRepair(x, y) {
    const w = this.world, i = w.idx(x, y);
    if (w.fence[i] !== F_BROKEN && !((isSolidFence(w.fence[i])) && w.fenceHp[i] < FENCE_DEF[w.fence[i]].hp)) return 'Nothing to repair.';
    if (!this.clearGap(x, y)) return 'A dinosaur is standing in the gap! Sedate it first.';
    const orig = w.fence[i] === F_BROKEN ? (w.fenceOrig[i] || F_ELECTRIC) : w.fence[i];
    const cost = FENCE_DEF[orig].cost * 3;
    if (!this.canAfford(cost)) return 'Not enough money.';
    this.spend(cost, 'repairs');
    const wasBroken = w.fence[i] === F_BROKEN;
    w.fence[i] = orig; w.fenceHp[i] = FENCE_DEF[orig].hp;
    if (wasBroken) w.invalidate(x, y); else w.markDirty(x, y);
    this.sparks(x + 0.5, y + 0.5, '#f8e080', 12);
    return null;
  }

  onFenceRebuilt(x, y) { this.burst(x + 0.5, y + 0.5, '#f8e080', 6); }

  humanKilled(h, dino) {
    if (h.dead) return;
    h.dead = true;
    this.burst(h.x, h.y, '#c82020', 14);
    this.soundAt('scream', h.x, h.y, 1);
    if (h.kind === 'guest') {
      this.stats.deaths++; this.deathLog.push(this.day);
      this.incident();
      this.reputation = Math.max(0, this.reputation - 5);
      this.spend(30000, 'fines');
      this.removeGuest(h, false);
      this.log(dino ? `A guest was ${dino.isPtera ? 'carried off' : 'eaten'} by ${dino.name} the ${dino.sp.name}! Lawsuit: -$30k` : 'A guest was killed! Lawsuit: -$30k', 'bad', { x: h.x, y: h.y }, true);
    } else {
      this.stats.staffDeaths++;
      this.reputation = Math.max(0, this.reputation - 2);
      const role = h.role === 'ranger' ? 'ranger' : 'engineer';
      this.log(dino ? `${role === 'ranger' ? 'A ranger' : 'An engineer'} was killed by ${dino.name} the ${dino.sp.name}!` : `${role === 'ranger' ? 'A ranger' : 'An engineer'} was killed!`, 'bad', { x: h.x, y: h.y }, true);
      // rehire after a delay
      const home = h.home;
      setTimeoutSim(this, 30, () => { if (this.world.buildings.has(home.id)) { this.staff.push(new Staff(this, h.role, home)); this.spend(5000, 'ops'); } });
    }
  }
  // reset the board by the gate
  incident() {
    const streak = this.day - (this.lastIncidentDay || 1);
    this.lastIncidentDay = this.day;
    this.incidentFlash = 4;
    if (streak >= 3) this.log(`The "days without incident" sign is back to zero after ${streak} days.`, 'warn');
  }

  killDino(d, cause, natural = false) {
    if (d.dead) return;
    d.dead = true;
    this.stats.dinoDeaths++;
    if (natural) {
      this.burst(d.x, d.y, '#e8e0c0', 10);
      this.log(`${d.name} the ${d.sp.name} died peacefully of old age after ${Math.floor(d.ageDays)} days.`, 'warn', { x: d.x, y: d.y }, true);
      return;
    }
    this.burst(d.x, d.y, '#8a2a20', 16);
    this.log(`${d.name} the ${d.sp.name} has died (${cause}).`, 'bad', { x: d.x, y: d.y }, true);
    this.reputation = Math.max(0, this.reputation - 2);
  }

  // Life finds a way: comfortable adult pairs sometimes produce a hatchling
  breed() {
    if (this.breedingOff) return;
    const w = this.world;
    if (this.dinos.length >= 70) return;
    const byReg = new Map();
    for (const d of this.dinos) {
      if (d.loose || d.carried || d.sedatedT > 0 || d.growth < 1 || d.elderly) continue;
      const r = w.region[w.idx(d.tx, d.ty)];
      if (r < 0) continue;
      if (!byReg.has(r)) byReg.set(r, []);
      byReg.get(r).push(d);
    }
    for (const [rid, list] of byReg) {
      const reg = w.regions[rid];
      const all = this.dinos.filter((d) => !d.carried && w.region[w.idx(d.tx, d.ty)] === rid);
      // the old are making way: a herd can replace itself in a paddock it already fills
      const used = all.reduce((a, d) => a + (d.elderly ? 0 : d.sp.space), 0);
      const bySp = {};
      for (const d of list) (bySp[d.species] = bySp[d.species] || []).push(d);
      for (const [sp, ds] of Object.entries(bySp)) {
        if (ds.length < 2) continue;
        const S = SPECIES[sp];
        if (used + S.space > reg.size) continue;
        const comfy = ds.filter((d) => d.comfort >= 62 && d.hunger < 65).length;
        if (comfy < 2) continue;
        if (!chance(S.diet === 'carn' ? 0.2 : 0.35)) continue;
        const parent = ds[0];
        const baby = new Dino(this, sp, parent.tx, parent.ty);
        baby.growth = 0.5;
        this.dinos.push(baby);
        this.stats.born = (this.stats.born || 0) + 1;
        this.burst(parent.x, parent.y, '#f0e8c8', 12);
        this.log(`Life finds a way: ${parent.name} the ${S.name} had a hatchling, ${baby.name}!`, 'good', baby, true);
        this.sound('hatch');
        break;
      }
    }
  }
  removeGuest(gu, leftNormally) {
    gu.dead = true;
    if (gu.inBuilding) gu.inBuilding.inside = Math.max(0, gu.inBuilding.inside - 1);
    if (leftNormally) {
      this.happySamples = this.happySamples || [];
      this.happySamples.push(gu.happy);
      if (this.happySamples.length > 200) this.happySamples.shift();
    }
  }

  findPaddockFor(d) {
    const w = this.world;
    if (d.isPtera) {
      const av = this.buildingsOfType('aviary').filter((a) => a.hp >= a.maxHp * 0.5);
      if (!av.length) return null;
      const a = pick(av);
      return { x: a.x + 2, y: a.y + 2 };
    }
    const mine = d.sp.diet === 'carn' ? 'carn' : 'herb', other = mine === 'carn' ? 'herb' : 'carn';
    const ok = (reg) => {
      if (!reg || reg.public || reg.size < 12 || reg.size > 1200) return false;
      // the player marked this paddock for the other diet by giving it only that kind of feeder
      if (reg.feeders && reg.feeders[other].length && !reg.feeders[mine].length) return false;
      for (const o of this.dinos) {
        if (o === d || o.carried || o.dead) continue;
        if (w.region[w.idx(o.tx, o.ty)] !== reg.id) continue;
        if (o.sp.diet !== d.sp.diet) return false;
        if (o.sp.territorial && d.sp.territorial) return false;
      }
      return true;
    };
    const pickTile = (reg) => {
      for (let k = 0; k < 40; k++) {
        const t = reg.tiles[Math.floor(rand() * reg.tiles.length)];
        const x = t % w.W, y = (t / w.W) | 0;
        if (w.dinoPass(x, y)) return { x, y };
      }
      return null;
    };
    if (d.relocateTo) {
      const r = w.regionAt(d.relocateTo.x, d.relocateTo.y);
      if (r && !r.public && r.size < 1200) { const t = d.relocateTo; d.relocateTo = null; return t; }
      d.relocateTo = null;
    }
    const home = w.regionAt(d.home.x, d.home.y);
    if (ok(home)) return pickTile(home);
    const fed = (reg) => (reg.feeders && reg.feeders[mine].length ? 1 : 0);
    const cands = w.regions.filter(ok).sort((a, b) => fed(b) - fed(a) || b.size - a.size);
    return cands.length ? pickTile(cands[0]) : null;
  }

  // A paddock marked for this diet (by its feeder) is blocked by animals of the other diet, fenced in by accident:
  // have the rangers move them out. Returns a message, or null if there is no such paddock.
  makeRoomFor(d) {
    const w = this.world, mine = d.sp.diet === 'carn' ? 'carn' : 'herb', other = mine === 'carn' ? 'herb' : 'carn';
    for (const reg of w.regions) {
      if (reg.public || reg.size > 1200 || !reg.feeders[mine].length || reg.feeders[other].length) continue;
      const strays = this.dinos.filter((o) => !o.carried && !o.dead && o.sp.diet === other && w.region[w.idx(o.tx, o.ty)] === reg.id);
      if (!strays.length) continue;
      for (const o of strays) { o.relocate = true; o.orderSedate = true; }
      const o = strays[0];
      return `${d.name} the ${d.sp.name} has to wait: ${o.name} the ${o.sp.name}${strays.length > 1 ? ` and ${strays.length - 1} more are` : ' is'} in the ${mine === 'carn' ? 'carnivore' : 'herbivore'} paddock. Rangers will move ${strays.length > 1 ? 'them' : 'it'} out.`;
    }
    return null;
  }

  // ------------- simulation -------------
  update(dt) {
    if (this.gameOver) return;
    const w = this.world;
    const prevHour = this.hour;
    this.time += dt / this.secPerHour;
    if (this.shake > 0) this.shake = Math.max(0, this.shake - dt * 6);
    if (this.incidentFlash > 0) this.incidentFlash -= dt;
    this.events.update(dt);
    runSimTimers(this, dt);

    for (const b of w.buildings.values()) if (b.offline > 0) { b.offline -= dt; if (b.offline <= 0) w.powerDirty = true; }

    this.powerT -= dt;
    if (w.powerDirty || this.powerT <= 0) { w.computePower(this); this.powerT = 1; }
    if (w.regionsDirty) { w.computeRegions(); this.updateLoose(); }
    if (this.pendingFences && this.pendingFences.length) this.buildPendingFences(dt);
    this.feedCheckT = (this.feedCheckT || 0) - dt;
    if (this.feedCheckT <= 0) {
      this.feedCheckT = 3; this.moveStrays(); this.replaceLostFeeders();
      // engineers patched up a broken-into shelter: open its doors again
      for (const b of w.buildings.values()) if (b.doorsBroken && b.hp >= b.maxHp * 0.8) { b.doorsBroken = false; w.flowCache.clear(); }
    }

    // eggs
    for (const e of this.eggs) {
      e.t -= dt;
      if (e.t <= 0) {
        const reg = w.regionAt(e.x, e.y);
        const d = SPECIES[e.species].flying ? new Ptera(this, e.x, e.y) : new Dino(this, e.species, e.x, e.y);
        if (e.unstable) { d.unstable = true; d.hp = d.sp.hp * 1.3; }
        if (d.isPtera) this.pteros.push(d); else this.dinos.push(d);
        this.stats.hatched++;
        this.burst(e.x + 0.5, e.y + 0.5, '#f0e8c8', 14);
        this.log(`${d.name} the ${d.sp.name} has hatched!`, 'good', d);
        this.sound('hatch');
        if (reg && reg.public) this.updateLoose();
      }
    }
    this.eggs = this.eggs.filter((e) => e.t > 0);

    // comfort
    this.comfortT -= dt;
    if (this.comfortT <= 0) { this.comfortT = 2; this.updateComfort(); }

    for (const d of this.dinos) d.update(dt);
    if (this.dinos.some((d) => d.dead)) { this.dinos = this.dinos.filter((d) => !d.dead); }
    for (const p of this.pteros) p.update(dt);
    if (this.pteros.some((d) => d.dead)) this.pteros = this.pteros.filter((d) => !d.dead);
    // rage timers
    for (const d of this.dinos) if (d.rageT > 0) { d.rageT -= dt; d.stress = 100; }

    for (const gu of this.guests) gu.update(dt);
    this.guests = this.guests.filter((gu) => !gu.dead);
    for (const s of this.staff) s.update(dt);
    this.staff = this.staff.filter((s) => !s.dead);
    for (const h of this.helis) h.update(dt);
    this.helis = this.helis.filter((h) => !h.dead);
    this.updateTours(dt);
    this.updateAttractions(dt);
    this.research.update(dt);
    this.updateLagoons(dt);

    // darts
    for (const dt_ of this.darts) {
      const t = dt_.t;
      dt_.life -= dt;
      const dx = t.x - dt_.x, dy = t.y - 0.3 - dt_.y, dd = Math.sqrt(dx * dx + dy * dy);
      const sp = 14 * dt;
      if (dd <= sp) {
        dt_.life = 0;
        if (!t.dead && !t.carried && t.sedatedT <= 0) {
          t.sedation += 1; t.flash = 0.15;
          if (t.sedation >= Math.max(1, t.sp.tranq - (this.research.has('darts') ? 1 : 0))) t.sedate();
          else t.stress = Math.min(100, t.stress + 10);
        }
      } else { dt_.x += dx / dd * sp; dt_.y += dy / dd * sp; dt_.a = Math.atan2(dy, dx); }
    }
    this.darts = this.darts.filter((d) => d.life > 0);

    // particles
    for (const p of this.particles) { p.life -= dt; p.x += p.vx * dt; p.y += p.vy * dt; p.vy += (p.grav || 0) * dt; }
    this.particles = this.particles.filter((p) => p.life > 0);
    if (this.particles.length > 1500) this.particles.splice(0, this.particles.length - 1500);
    for (const f of this.floats) { f.life -= dt; f.y -= dt * 0.8; }
    this.floats = this.floats.filter((f) => f.life > 0);

    this.spawnGuests(dt);
    this.crowdT = (this.crowdT || 0) - dt;
    if (this.crowdT <= 0) {
      this.crowdT = 0.5;
      const w = this.world;
      if (!this.crowd) this.crowd = new Uint16Array(w.W * w.H);
      this.crowd.fill(0);
      for (const gu of this.guests) if (!gu.hidden) this.crowd[w.idx(gu.tx, gu.ty)]++;
    }
    if (this.guests.length > (this.peakGuests || 0)) this.peakGuests = this.guests.length;

    // per-second bookkeeping
    this.secondT -= dt;
    if (this.secondT <= 0) {
      this.secondT = 1;
      this.updateRating();
      if (!this.scenario) this.checkGoals();
      else {
        const sc = SCENARIOS[this.scenario];
        if (sc.tick) sc.tick(this);
        const r = sc.check(this);
        if (r && !this.scenarioDone) { this.scenarioDone = r; this.emit('scenario', r); }
      }
      this.updateDecisions();
      this.adviseT = (this.adviseT === undefined ? 20 : this.adviseT) - 1;
      if (this.adviseT <= 0) { this.adviseT = 45; if (this.advisorOn !== false) this.advise(); }
      this.looseCount = this.creatures().filter((d) => d.loose && !d.carried).length;
    }

    if (Math.floor(this.time / 24) + 1 !== this.lastDay) this.newDay();
  }

  updateTours(dt) {
    const w = this.world;
    for (const st of this.buildingsOfType('tour')) {
      if (!st.queue) st.queue = [];
      st.queue = st.queue.filter((gu) => !gu.dead && gu.state === 'queue');
      const mine = this.jeeps.filter((j) => j.station === st && !j.dead);
      // keep two jeeps per station when track is connected
      st.jeepT = (st.jeepT || 0) - dt;
      if (mine.length < 3 && st.jeepT <= 0 && w.trackAccess(st).length) {
        st.jeepT = mine.length ? 20 : 0.5;
        this.jeeps.push(new Jeep(this, st));
      }
    }
    for (const j of this.jeeps) j.update(dt);
    this.jeeps = this.jeeps.filter((j) => !j.dead);
  }

  updateLagoons(dt) {
    for (const b of this.buildingsOfType('lagoon')) {
      b.showT = (b.showT === undefined ? randf(5, 15) : b.showT) - dt;
      if (b.leap > 0) b.leap -= dt;
      if (b.showT <= 0 && b.powered) {
        b.showT = randf(25, 40);
        b.leap = 2.2;
        this.soundAt('bellow', b.x + 3, b.y + 3, 1);
        setTimeoutSim(this, 1.1, () => { this.burst(b.x + 3, b.y + 3.2, '#a8d8f0', 30); this.soundAt('crash', b.x + 3, b.y + 3, 0.6); this.shake = Math.max(this.shake, 1); });
        // nearby guests love it
        // the splash zone: everyone watching pays for the privilege, like a feeding show
        let crowd = 0;
        for (const gu of this.guests) if (!gu.hidden && dist2(gu.x, gu.y, b.x + 3, b.y + 2.5) < 81) { gu.happy = Math.min(100, gu.happy + 15); if (chance(0.2)) { gu.thought = 'WOW!'; gu.thoughtT = 2; } crowd++; }
        const pay = Math.min(crowd, 60) * 22 * this.priceMul;
        if (pay) { this.earn(pay, 'shops'); b.revenue = (b.revenue || 0) + pay; b.visitors = (b.visitors || 0) + crowd; }
      }
      if (b.hp < b.maxHp * 0.25 && !b.escaped) {
        b.escaped = true;
        this.reputation = Math.max(0, this.reputation - 10);
        this.log('The Mosasaur Lagoon cracked open and the Mosasaurus escaped into the ocean!', 'bad', { x: b.x + 3, y: b.y + 2 }, true);
      }
    }
  }

  jeepAttacked(j, dino) {
    if (j.wrecked) return;
    j.wrecked = 8;
    const n = j.riders.length;
    for (const r of j.riders) { r.hidden = false; r.x = j.x; r.y = j.y; this.humanKilled(r, null); }
    j.riders = [];
    this.burst(j.x, j.y, '#3a8a3a', 14); this.burst(j.x, j.y, '#c82020', n * 4);
    this.shake = Math.max(this.shake, 4);
    this.soundAt('crash', j.x, j.y, 1);
    this.log(`${dino.name} the ${dino.sp.name} flipped a tour jeep!${n ? ' ' + n + ' guest' + (n > 1 ? 's' : '') + ' lost.' : ''}`, 'bad', { x: j.x, y: j.y }, true);
  }

  // Advisor: find the most pressing issue the player can act on
  advise() {
    const w = this.world;
    const issues = [];
    const cre = this.creatures();
    const contained = this.dinos.filter((d) => !d.loose && !d.carried);
    const carn = cre.some((d) => d.sp.danger >= 4);
    let unpowered = null;
    for (let i = 0; i < w.fence.length; i++) if (w.fence[i] === F_ELECTRIC && !w.fencePowered[i]) {
      unpowered = { x: i % w.W, y: (i / w.W) | 0 }; break;
    }
    const offline = this.buildingsOfType('power').filter((b) => b.offline > 0);
    if (unpowered && cre.length) {
      if (w.power.outage > 0) issues.push([90, `The grid is down: fences come back on when it restarts${this.hasBuilding('backup') ? '' : '. A Backup Generator keeps them live meanwhile'}.`, unpowered]);
      else if (offline.length) issues.push([90, `A Power Plant is knocked offline, so some fences have no power until it restarts${this.hasBuilding('backup') ? '' : '. A Backup Generator or a second plant covers this'}.`, offline[0]]);
      else issues.push([90, 'Some electric fences have no power. Add a Pylon or Power Plant nearby.', unpowered]);
    }
    const dark = [...w.buildings.values()].filter((b) => BUILDINGS[b.type].power && !b.powered && w.covered[w.idx(b.x, b.y)]);
    if (w.power.outage <= 0 && w.power.demand > w.power.supply && dark.length) issues.push([88, `Not enough power: ${Math.round(w.power.demand)} MW needed, ${Math.round(w.power.supply)} MW made. ${dark.length} building${dark.length > 1 ? 's are' : ' is'} dark (fences get power first). Build another Power Plant.`, dark[0]]);
    if (this.alarm && !cre.some((d) => d.loose && !d.carried && d.sedatedT <= 0 && d.sp.danger >= 4)) issues.push([86, 'Nothing dangerous is loose any more. Sound the all-clear (alarm button) so visitors can come back.', null]);
    if (carn && !this.hasBuilding('ranger')) issues.push([85, 'You have dangerous carnivores but no Ranger Station. Nobody can tranquilize an escapee.', null]);
    if (carn && !this.shelters().length) issues.push([70, 'There is nowhere for guests to shelter. A Visitor Center, Hotel or Bunker saves lives during a breach.', null]);
    if (cre.length && !this.hasBuilding('maint')) issues.push([65, 'Without a Maintenance Shed, damaged fences are never repaired.', null]);
    if (this.hasBuilding('power') && !this.hasBuilding('backup') && cre.some((d) => d.sp.strength >= 6)) issues.push([45, 'A Backup Generator keeps fences live during a grid failure.', null]);
    const bySp = {};
    for (const d of contained) (bySp[d.species] = bySp[d.species] || []).push(d);
    for (const [sp, ds] of Object.entries(bySp)) {
      if (ds.every((d) => d.elderly) && !this.eggs.some((e) => e.species === sp)) {
        const left = Math.max(1, Math.round(Math.min(...ds.map((d) => d.lifespan - d.ageDays))));
        issues.push([72, `Your ${SPECIES[sp].name} ${ds.length > 1 ? 'are all' : 'is'} elderly: the first will die of old age in about ${left} day${left > 1 ? 's' : ''}. Hatch young ones to keep the exhibit going.`, ds[0]]);
      }
    }
    for (const d of contained) {
      const reg = w.regionAt(d.tx, d.ty);
      if (!reg) continue;
      const p = d.comfortParts || {};
      if (d.hunger > 75) issues.push([80, `${d.name} the ${d.sp.name} is starving. Put a ${d.sp.diet === 'carn' ? 'Carnivore' : 'Herbivore'} Feeder in its paddock.`, d]);
      else if (p.fear < 0) issues.push([75, `${d.name} the ${d.sp.name} is terrified: it shares a paddock with a carnivore. Use Move… to separate them.`, d]);
      else if (p.social < 0) issues.push([75, `${d.name} the ${d.sp.name} is fighting a rival. Move one of them to its own paddock.`, d]);
      else if (d.sp.social > 1 && p.social < 10) issues.push([40, `${d.name} the ${d.sp.name} is lonely. ${d.sp.name} like groups of ${d.sp.social}.`, d]);
      else if (p.space < 18) issues.push([50, `${d.name}'s paddock is crowded. Enlarge it or move some dinosaurs out.`, d]);
      else if (p.forest < 8 && d.sp.forest > 0.2) issues.push([30, `${d.name} the ${d.sp.name} wants more tree cover. Plant trees in its paddock.`, d]);
      else if (d.sick > 0 && !this.hasBuilding('vet')) issues.push([60, `${d.name} is sick. A Vet Clinic would cure it quickly.`, d]);
    }
    const gb = this.buildingsOfType('gate')[0];
    if (this.guests.length > 30 && !this.hasBuilding('restroom')) issues.push([35, 'Guests are desperate for a Restroom.', null]);
    if (this.guests.length > 30 && !this.hasBuilding('restaurant')) issues.push([35, 'Hungry guests have nowhere to eat. Build a Restaurant.', null]);
    if (this.guests.length > 60 && !this.hasBuilding('tour') && this.stars >= 1) issues.push([20, 'A Tour Station with a loop of Tour Track past the paddocks is the best crowd-pleaser.', null]);
    const crowded = this.guests.filter((gu) => gu.crowded);
    if (crowded.length > 25) issues.push([45, 'Paths are overcrowded and guests are unhappy. Widen busy paths or add a second route.', crowded[0]]);
    if (this.money < 30000) issues.push([55, 'Funds are low. Raise ticket prices a little or add shops along busy paths.', gb]);
    this.adviceSaid = this.adviceSaid || {};
    issues.sort((a, b) => b[0] - a[0]);
    for (const [, msg, ref] of issues) {
      const last = this.adviceSaid[msg];
      if (last !== undefined && this.time - last < 48) continue;
      this.adviceSaid[msg] = this.time;
      this.log('Advisor: ' + msg, 'warn', ref, true);
      return msg;
    }
    return null;
  }

  updateLoose() {
    const w = this.world;
    let newly = [];
    for (const d of this.dinos) {
      if (d.carried) continue;
      const reg = w.regionAt(d.tx, d.ty);
      const wasLoose = d.loose;
      d.loose = !!(reg && reg.public);
      if (d.loose && !wasLoose) newly.push(d);
    }
    if (newly.length) {
      this.stats.escapes += newly.length;
      this.reputation = Math.max(0, this.reputation - newly.length * 1.5);
      const d = newly[0];
      const what = newly.length > 1 ? `${newly.length} dinosaurs including ${d.name} the ${d.sp.name}` : `${d.name} the ${d.sp.name}`;
      this.log(`CONTAINMENT BREACH! ${what} ${newly.length > 1 ? 'are' : 'is'} loose!`, 'bad', d, true);
      this.incident();
      this.sound('alarm');
      this.emit('breach', newly);
    }
    for (const d of this.dinos) {
      if (!d.loose && d.wasLooseFlag) { this.log(`${d.name} the ${d.sp.name} is contained again.`, 'good', d); }
      d.wasLooseFlag = d.loose;
    }
    this.looseCount = this.dinos.filter((d) => d.loose).length;
  }

  updateComfort() {
    const w = this.world;
    const byReg = new Map();
    for (const d of this.dinos) {
      if (d.carried) continue;
      const r = w.region[w.idx(d.tx, d.ty)];
      if (!byReg.has(r)) byReg.set(r, []);
      byReg.get(r).push(d);
    }
    for (const [rid, list] of byReg) {
      const reg = w.regions[rid];
      let spaceNeed = 0;
      const counts = {};
      let carn = 0, herb = 0, terr = 0;
      for (const d of list) { spaceNeed += d.sp.space; counts[d.species] = (counts[d.species] || 0) + 1; if (d.sp.diet === 'carn') carn++; else herb++; if (d.sp.territorial) terr++; }
      for (const d of list) {
        const parts = {};
        if (!reg) { d.comfort = 20; continue; }
        const sz = Math.min(reg.size, 900);
        parts.space = clamp(sz / spaceNeed, 0, 1) * 30;
        const fr = reg.forest / reg.size;
        parts.forest = clamp(fr / Math.max(0.05, d.sp.forest), 0, 1) * 18;
        parts.water = reg.water > 0 ? 12 : (d.species === 'spino' ? -10 : 4);
        const n = counts[d.species];
        parts.social = d.sp.social <= 1 ? (n === 1 ? 15 : 10) : clamp(n / d.sp.social, 0, 1) * 15;
        if (d.sp.territorial && terr > 1) parts.social = -25;
        const hasFood = reg.feeders[d.sp.diet].length > 0 || (d.sp.diet === 'herb' && reg.forest > 3);
        parts.food = hasFood ? 15 : 0;
        parts.fear = d.sp.diet === 'herb' && carn > 0 ? -25 : 0;
        parts.base = 10;
        if (d.loose) parts.loose = -10;
        let c = 0; for (const k in parts) c += parts[k];
        d.comfort = clamp(c, 0, 100);
        if (d.rageT > 0) d.comfort = 0;
        d.comfortParts = parts;
      }
    }
  }

  spawnGuests(dt) {
    const w = this.world;
    const h = this.hour;
    const gates = this.buildingsOfType('gate');
    if (!gates.length || this.alarm || this.gateLocked || this.parkClosed || (this.events.storm && this.events.storm.phase === 'active')) return;
    if ((h < 7 || h > 20) && !this.mod('nightGuests', 0)) return;
    const desired = this.desiredGuests();
    if (this.guests.length >= desired) return;
    const perHour = Math.max(2, desired / 5);
    this.spawnAcc += perHour * dt / this.secPerHour;
    while (this.spawnAcc >= 1) {
      this.spawnAcc -= 1;
      const gate = pick(gates);
      const acc = w.accessTiles(gate);
      if (!acc.length) return;
      const gu = new Guest(this, pick(acc));
      this.guests.push(gu);
      this.stats.guestsTotal++;
      this.earn(this.ticket * this.mod('ticketMul'), 'tickets');
      this.floatT = (this.floatT || 0) - 1;
      if (this.floatT <= 0) { this.floatT = 4; this.addFloat(gate.x + 1.5, gate.y - 0.5, '+$' + this.ticket, '#f8d040'); }
    }
  }

  desiredGuests() {
    const attractions = this.attraction;
    // Need something to see / do beyond the gate
    const hasThings = this.world.buildings.size > 1;
    if (!hasThings) return 0;
    const priceF = clamp(1.7 - this.ticket / 150, 0.1, 1.5);
    const repF = 0.4 + this.reputation / 80;
    const night = this.isNight && !this.mod('nightGuests', 0) ? 0.5 : 1;
    const mkt = this.research.has('marketing') ? 1.15 : 1;
    return Math.min(450, Math.floor((10 + attractions * 2.8) * priceF * repF * night * mkt * this.season.guests * this.mod('guestMul')));
  }

  updateRating() {
    let att = 0;
    const species = new Set();
    for (const d of this.creatures()) {
      if (d.loose || d.carried) continue;
      att += d.sp.appeal * (d.sick > 0 ? 0.5 : 1) * (0.6 + d.comfort / 250);
      species.add(d.species);
    }
    att += species.size * 5;
    for (const b of this.world.buildings.values()) att += (BUILDINGS[b.type].appeal || 0) * (b.powered ? 1 : 0.5) * (b.escaped ? 0 : 1);
    this.attraction = att;
    const score = att + this.reputation * 0.9;
    let s = 0;
    for (let i = 0; i < STAR_THRESH.length; i++) if (score >= STAR_THRESH[i]) s = i;
    if (s > this.stars) { this.log(`Your park is now rated ${s} star${s > 1 ? 's' : ''}! New species may be unlocked.`, 'good', null, true); this.sound('fanfare'); }
    this.stars = s;
    this.score = score;
  }

  checkGoals() {
    while (this.goalIdx < GOALS.length && GOALS[this.goalIdx].check(this)) {
      const g = GOALS[this.goalIdx];
      if (g.reward) this.earn(g.reward, 'grants');
      this.goalIdx++;
      this.log(`Goal complete: ${g.text}${g.reward ? ' (+' + fmtMoney(g.reward) + ')' : ''}`, 'goal', null, true);
      this.sound('fanfare');
      if (this.goalIdx >= GOALS.length && !this.victory) { this.victory = true; this.emit('victory'); }
    }
  }

  newDay() {
    const day = this.day;
    this.lastDay = day;
    // upkeep
    let up = 0;
    for (const b of this.world.buildings.values()) up += BUILDINGS[b.type].upkeep;
    up += this.mod('upkeepAdd', 0);
    this.spend(up, 'upkeep');
    // reputation drift
    const avgHappy = this.happySamples && this.happySamples.length ? this.happySamples.reduce((a, b) => a + b, 0) / this.happySamples.length : 55;
    this.safety = clamp(100 - this.recentDeaths(10) * 12 - this.looseCount * 10 - this.countBrokenFences() * 1.5, 0, 100);
    const target = avgHappy * 0.55 + this.safety * 0.45;
    this.reputation += (target - this.reputation) * 0.25;
    this.reputation = clamp(this.reputation, 0, 100);
    const inc = this.ledger.tickets + this.ledger.shops + this.ledger.grants;
    const exp = this.ledger.upkeep + this.ledger.food + this.ledger.repairs + this.ledger.construction + this.ledger.dinos + this.ledger.fines + this.ledger.ops;
    this.history.push({ day: day - 1, money: this.money, guests: this.guests.length, peak: this.peakGuests || 0, income: inc, expense: exp, rep: this.reputation, stars: this.stars, ledger: this.ledger });
    if (this.history.length > 120) this.history.shift();
    this.ledger = this.newLedger();
    this.peakGuests = this.guests.length;
    this.stats.fledToday = 0;
    if ((day - 1) % 7 === 0 && day > 1) { const s = this.season; this.log(`${s.name} has arrived. ${s.desc}`, 'info', null, true); }
    this.events.onNewDay(day);
    this.breed();
    this.maybeOfferDecision();
    this.emit('day', day);
    // game over checks
    if (this.money < -100000) {
      this.brokeDays = (this.brokeDays || 0) + 1;
      if (this.brokeDays === 1) this.log('WARNING: The park is deeply in debt. 5 days until the investors pull out!', 'bad', null, true);
      if (this.brokeDays >= 5) this.endGame('Bankrupt! The investors have pulled their funding.');
    } else this.brokeDays = 0;
    if (this.scenario && SCENARIOS[this.scenario].tick && this.scenarioDone) { /* finished */ }
    if (this.recentDeaths(14) >= 12) this.endGame('Too many casualties. The board has shut the park down.');
  }

  endGame(reason) {
    if (this.sandbox) { this.log(reason + ' (sandbox: continuing)', 'bad', null, true); return; }
    this.gameOver = reason;
    this.emit('gameover', reason);
  }
}

// Simple timers in sim-time
function setTimeoutSim(game, secs, fn) { (game.simTimers = game.simTimers || []).push({ t: secs, fn }); }
function runSimTimers(game, dt) {
  if (!game.simTimers || !game.simTimers.length) return;
  for (const t of game.simTimers) { t.t -= dt; if (t.t <= 0) t.fn(); }
  game.simTimers = game.simTimers.filter((t) => t.t > 0);
}
