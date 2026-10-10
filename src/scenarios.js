// ---------- Scenarios: prebuilt crises with objectives and deadlines ----------
'use strict';

// Lay out a working park around the gate, flattening terrain where needed.
function buildStarterPark(g, opts = {}) {
  const w = g.world, s = g.start;
  const flat = (x0, y0, x1, y1) => {
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
      if (!w.inb(x, y)) continue;
      const i = w.idx(x, y);
      if (w.terrain[i] !== T_GRASS && w.terrain[i] !== T_FOREST) w.terrain[i] = T_GRASS;
    }
  };
  flat(s.x - 24, s.y - 30, s.x + 24, s.y - 1);
  const money = g.money; g.money = 1e9;
  const put = (t, x, y) => { const b = g.placeBuilding(t, x, y); return b; };
  const fenceRect = (x0, y0, x1, y1, type = F_ELECTRIC) => {
    for (let x = x0; x <= x1; x++) { g.placeFence(x, y0, type); g.placeFence(x, y1, type); }
    for (let y = y0 + 1; y < y1; y++) { g.placeFence(x0, y, type); g.placeFence(x1, y, type); }
  };
  const forest = (x0, y0, x1, y1) => { for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) w.terrain[w.idx(x, y)] = T_FOREST; };
  // main boulevard
  for (let x = s.x - 22; x <= s.x + 22; x++) g.placePath(x, s.y - 9, true);
  for (let y = s.y - 9; y <= s.y - 5; y++) g.placePath(s.x, y, true);
  put('power', s.x - 14, s.y - 6); put('power', s.x + 12, s.y - 6);
  put('pylon', s.x - 1, s.y - 20);
  put('hatchery', s.x + 5, s.y - 4);
  put('visitor', s.x + 2, s.y - 8 + 0); put('restaurant', s.x - 4, s.y - 7); put('restroom', s.x - 6, s.y - 7);
  put('shop', s.x - 9, s.y - 7);
  // paddocks north of the boulevard
  const P = [[s.x - 22, s.y - 26, s.x - 9, s.y - 11], [s.x - 7, s.y - 26, s.x + 7, s.y - 11], [s.x + 9, s.y - 26, s.x + 22, s.y - 11]];
  P.forEach(([x0, y0, x1, y1], k) => {
    fenceRect(x0, y0, x1, y1, k === 1 && opts.rexWall ? F_WALL : F_ELECTRIC);
    forest(x0 + 1, y0 + 1, x0 + 4, y0 + 4);
  });
  put('feeder_h', s.x - 15, s.y - 12); put('feeder_c', s.x, s.y - 12); put('feeder_c', s.x + 15, s.y - 12);
  put('viewing', s.x - 12, s.y - 11 + 0);
  if (opts.staff) { put('ranger', s.x + 15, s.y - 7); put('maint', s.x - 20, s.y - 7); }
  w.invalidate(); w.computeRegions(); w.computePower(g);
  g.money = money;
  // reset ledgers so setup costs don't show
  g.ledger = g.newLedger();
  return P;
}

function spawnIn(g, sp, box, n = 1) {
  const w = g.world;
  for (let k = 0; k < n; k++) {
    for (let t = 0; t < 40; t++) {
      const x = randi(box[0] + 2, box[2] - 2), y = randi(box[1] + 2, box[3] - 2);
      if (w.dinoPass(x, y)) { const d = new Dino(g, sp, x, y); d.growth = 1; g.dinos.push(d); break; }
    }
  }
}

const SCENARIOS = {
  nedry: {
    name: "Nedry's Night",
    blurb: 'Your park is finished and full of guests. At nightfall a programmer shuts down the grid and walks off with the embryos. Keep casualties low until the morning of Day 3.',
    objective: 'Survive to Day 3, 08:00 with fewer than 4 guest casualties',
    setup(g) {
      g.money = 180000;
      const P = buildStarterPark(g, { staff: true });
      spawnIn(g, 'galli', P[0], 3); spawnIn(g, 'para', P[0], 2);
      spawnIn(g, 'trex', P[1], 1);
      spawnIn(g, 'raptor', P[2], 3);
      g.time = 15;
      g.events.auto = false;
      g.scenarioState = { struck: false };
    },
    tick(g) {
      if (!g.scenarioState.struck && g.time >= 20) {
        g.scenarioState.struck = true;
        g.spend(50000, 'fines');
        g.events.trigger('outage');
        g.world.power.outage = 150;
        g.log('Dennis Nedry has stolen the embryos (-$50k) and shut down the park. "Ah ah ah!"', 'bad', null, true);
      }
    },
    check(g) {
      if (g.stats.deaths >= 4) return 'lose';
      if (g.time >= 24 * 2 + 8) return 'win';
      return null;
    },
    progress(g) { return `Casualties ${g.stats.deaths}/3 · ${g.time < 20 ? 'Grid fails at 20:00' : g.world.power.outage > 0 ? 'Grid down ' + Math.ceil(g.world.power.outage) + 's' : 'Grid restored'}`; },
  },
  sorna: {
    name: 'Isla Sorna Cleanup',
    blurb: 'A rival company abandoned this island. Dinosaurs roam free. Build paddocks and get every animal contained.',
    objective: 'Contain every dinosaur by the end of Day 10',
    setup(g) {
      g.money = 520000;
      const w = g.world, s = g.start;
      g.placeBuilding('helipad', s.x + 4, s.y - 4) || g.placeBuilding('helipad', s.x - 6, s.y - 4);
      g.money = 520000;
      const roam = (sp, n) => {
        for (let k = 0; k < n; k++) for (let t = 0; t < 200; t++) {
          const x = randi(4, w.W - 5), y = randi(4, w.H - 5);
          if (w.dinoPass(x, y) && dist(x, y, s.x, s.y) > 14) { const d = new Dino(g, sp, x, y); d.growth = 1; d.stress = 20; g.dinos.push(d); break; }
        }
      };
      roam('galli', 4); roam('para', 3); roam('stego', 2); roam('trike', 1); roam('dilo', 2);
      w.computeRegions(); g.updateLoose();
      g.events.auto = false;
    },
    check(g) {
      if (!g.dinos.length) return 'lose';
      if (g.dinos.every((d) => !d.loose && !d.carried && d.sedatedT <= 0)) return 'win';
      if (g.day > 10) return 'lose';
      return null;
    },
    progress(g) { const n = g.dinos.filter((d) => d.loose || d.carried).length; return `${g.dinos.length - n}/${g.dinos.length} contained · Day ${g.day}/10`; },
  },
  storms: {
    name: 'Storm Season',
    blurb: 'Monsoon season. A storm hits every single day, knocking out power and fences. Keep the park profitable anyway.',
    objective: 'Reach $900,000 in the bank by Day 12',
    setup(g) {
      g.money = 250000;
      const P = buildStarterPark(g, { staff: true });
      spawnIn(g, 'galli', P[0], 3); spawnIn(g, 'trike', P[0], 1);
      spawnIn(g, 'stego', P[1], 2);
      spawnIn(g, 'dilo', P[2], 2);
      g.events.auto = false;
      g.scenarioState = { lastStorm: 0 };
    },
    tick(g) {
      if (g.day > g.scenarioState.lastStorm && g.hour >= 11) { g.scenarioState.lastStorm = g.day; g.events.trigger('storm'); }
    },
    check(g) {
      if (g.money >= 900000) return 'win';
      if (g.day > 12) return 'lose';
      return null;
    },
    progress(g) { return `${fmtMoney(g.money)} / $900k · Day ${g.day}/12`; },
  },
};
