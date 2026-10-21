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
  // the starter park needs this ground: move the volcano out of the way if it sits here
  const v = w.volcano;
  if (v && v.x >= s.x - 28 && v.x <= s.x + 28 && v.y >= s.y - 34) {
    let best = null, bd = -1;
    for (let y = 4; y < w.H - 4; y++) for (let x = 4; x < w.W - 4; x++) {
      if (x >= s.x - 28 && x <= s.x + 28 && y >= s.y - 34) continue;
      if (!w.isLand(w.terrain[w.idx(x, y)]) && w.terrain[w.idx(x, y)] !== T_ROCK) continue;
      const d = dist(x, y, s.x, s.y) + (w.terrain[w.idx(x, y)] === T_ROCK ? 6 : 0);
      if (d > bd) { bd = d; best = { x, y }; }
    }
    if (best) {
      for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) {
        const x = best.x + dx, y = best.y + dy;
        if (!w.inb(x, y)) continue;
        const d = Math.sqrt(dx * dx + dy * dy);
        if (d <= 1.5) w.terrain[w.idx(x, y)] = T_VOLCANO; else if (d <= 3.2) w.terrain[w.idx(x, y)] = T_ROCK;
      }
      w.volcano = best;
    }
  }
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
  put('visitor', s.x + 2, s.y - 8); put('restaurant', s.x - 4, s.y - 8); put('restroom', s.x - 6, s.y - 8);
  put('shop', s.x - 9, s.y - 8);
  // paddocks north of the boulevard
  const P = [[s.x - 22, s.y - 26, s.x - 9, s.y - 12], [s.x - 7, s.y - 26, s.x + 7, s.y - 12], [s.x + 9, s.y - 26, s.x + 22, s.y - 12]];
  P.forEach(([x0, y0, x1, y1], k) => {
    fenceRect(x0, y0, x1, y1, k === 1 && opts.rexWall ? F_WALL : F_ELECTRIC);
    forest(x0 + 1, y0 + 1, x0 + 4, y0 + 4);
  });
  // chain pylons (outside the paddocks) until every electric fence has power
  const inBox = (x, y) => P.some(([x0, y0, x1, y1]) => x >= x0 && x <= x1 && y >= y0 && y <= y1);
  for (let k = 0; k < 12; k++) {
    w.computePower(g);
    const dead = [];
    for (let i = 0; i < w.fence.length; i++) if (w.fence[i] === F_ELECTRIC && !w.covered[i]) dead.push([i % w.W, (i / w.W) | 0]);
    if (!dead.length) break;
    let best = null, bs = 0;
    for (let y = s.y - 34; y <= s.y; y++) for (let x = s.x - 30; x <= s.x + 30; x++) {
      if (!w.inb(x, y) || !w.covered[w.idx(x, y)] || inBox(x, y) || !w.canPlaceBuilding('pylon', x, y)) continue;
      let n = 0;
      for (const [fx, fy] of dead) if (dist2(x, y, fx, fy) <= 64) n++;
      if (n > bs) { bs = n; best = { x, y }; }
    }
    if (!best) break;
    put('pylon', best.x, best.y);
  }
  put('feeder_h', s.x - 15, s.y - 13); put('feeder_c', s.x, s.y - 13); put('feeder_c', s.x + 15, s.y - 13);
  put('viewing', s.x - 12, s.y - 11);
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

function mainRegion(g) {
  const w = g.world, s = g.start;
  w.computeRegions();
  const r = w.regionAt(s.x, s.y - 2) || w.regionAt(s.x, s.y - 3);
  return r ? r.id : -1;
}
function roamSpawn(g, sp, n, minDist = 14) {
  const w = g.world, s = g.start, main = mainRegion(g);
  for (let k = 0; k < n; k++) for (let t = 0; t < 400; t++) {
    const x = randi(4, w.W - 5), y = randi(4, w.H - 5);
    if (w.dinoPass(x, y) && dist(x, y, s.x, s.y) > minDist && w.region[w.idx(x, y)] === main) { const d = new Dino(g, sp, x, y); d.growth = 1; d.stress = 20; g.dinos.push(d); break; }
  }
}

const MEDAL_ICON = { gold: '🥇', silver: '🥈', bronze: '🥉' };
const MEDAL_RANK = { bronze: 1, silver: 2, gold: 3 };
function loadMedals() { try { return JSON.parse(localStorage.getItem('jt_medals') || '{}'); } catch (e) { return {}; } }
function saveMedal(id, medal, day) {
  const m = loadMedals();
  const prev = m[id];
  const better = !prev || MEDAL_RANK[medal] > MEDAL_RANK[prev.medal] || (MEDAL_RANK[medal] === MEDAL_RANK[prev.medal] && day < prev.day);
  if (better) { m[id] = { medal, day }; try { localStorage.setItem('jt_medals', JSON.stringify(m)); } catch (e) { /* storage unavailable */ } }
  return better;
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
      spawnIn(g, 'raptor', P[2], 4);
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
        g.gateLocked = true;
        g.log('Dennis Nedry has stolen the embryos (-$50k), shut down the grid and sealed the main gate. Guests cannot leave: build Bunkers! "Ah ah ah!"', 'bad', null, true);
      }
    },
    medal(g) { const d = g.stats.deaths; return d === 0 ? 'gold' : d <= 1 ? 'silver' : 'bronze'; },
    medalText: 'Gold: no casualties · Silver: 1 casualty',
    check(g) {
      if (g.gateLocked && g.world.power.outage <= 0 && g.looseCount === 0) { g.gateLocked = false; g.log('The main gate is unlocked again.', 'good', null, true); }
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
      const w = g.world, s = g.start;
      for (let y = s.y - 10; y <= s.y - 6; y++) for (let x = s.x + 3; x <= s.x + 7; x++) { const i = w.idx(x, y); w.terrain[i] = T_GRASS; w.path[i] = 0; w.fence[i] = 0; }
      w.invalidate();
      g.money = 1e9; g.placeBuilding('helipad', s.x + 4, s.y - 9);
      g.money = 520000;
      roamSpawn(g, 'galli', 4); roamSpawn(g, 'para', 3); roamSpawn(g, 'stego', 2); roamSpawn(g, 'trike', 1); roamSpawn(g, 'dilo', 2, 20);
      w.computeRegions(); g.updateLoose();
      g.events.auto = false;
    },
    medal(g) { return g.day <= 5 ? 'gold' : g.day <= 7 ? 'silver' : 'bronze'; },
    medalText: 'Gold: by Day 5 · Silver: by Day 7',
    check(g) {
      if (!g.dinos.length) return 'lose';
      if (g.dinos.every((d) => !d.loose && !d.carried && d.sedatedT <= 0)) return 'win';
      if (g.day > 10) return 'lose';
      return null;
    },
    progress(g) { const n = g.dinos.filter((d) => d.loose || d.carried).length; return `${g.dinos.length - n}/${g.dinos.length} contained · Day ${g.day}/10`; },
  },
  lostworld: {
    name: 'The Lost World',
    blurb: 'A mainland zoo wants specimens. Build a holding paddock, sedate the wild dinosaurs, airlift them in, then sell them from the paddock.',
    objective: 'Ship 6 dinosaurs, including the T. rex, by the end of Day 6',
    setup(g) {
      g.money = 220000;
      const s = g.start, w = g.world;
      for (let y = s.y - 10; y <= s.y - 6; y++) for (let x = s.x + 3; x <= s.x + 7; x++) { const i = w.idx(x, y); w.terrain[i] = T_GRASS; w.path[i] = 0; w.fence[i] = 0; }
      w.invalidate();
      g.money = 1e9; g.placeBuilding('helipad', s.x + 4, s.y - 9);
      g.money = 260000;
      roamSpawn(g, 'galli', 3); roamSpawn(g, 'para', 2); roamSpawn(g, 'stego', 2); roamSpawn(g, 'trike', 1); roamSpawn(g, 'trex', 1, 22);
      g.world.computeRegions(); g.updateLoose();
      g.events.auto = false;
    },
    medal(g) { return g.day <= 3 ? 'gold' : g.day <= 4 ? 'silver' : 'bronze'; },
    medalText: 'Gold: by Day 3 · Silver: by Day 4',
    check(g) {
      const sold = g.stats.sold || 0, rex = (g.stats.soldSpecies || []).includes('trex');
      if (sold >= 6 && rex) return 'win';
      if (!g.dinos.some((d) => d.species === 'trex') && !rex) return 'lose';
      if (g.day > 6) return 'lose';
      return null;
    },
    progress(g) { return `Shipped ${g.stats.sold || 0}/6 · T. rex ${(g.stats.soldSpecies || []).includes('trex') ? 'shipped' : 'still wild'} · Day ${g.day}/6`; },
  },
  opening: {
    name: 'Opening Day',
    blurb: 'The new park opens to record crowds and your star attraction is an Indominus. On day two she figures out the walls.',
    objective: 'Welcome 1,800 guests by Day 5 with fewer than 6 casualties',
    setup(g) {
      g.money = 300000;
      const P = buildStarterPark(g, { staff: true, rexWall: true });
      spawnIn(g, 'galli', P[0], 3); spawnIn(g, 'para', P[0], 2);
      spawnIn(g, 'indom', P[1], 1);
      spawnIn(g, 'stego', P[2], 2); spawnIn(g, 'trike', P[2], 1);
      const s = g.start;
      g.money = 1e9; g.placeBuilding('hotel', s.x - 17, s.y - 8); g.placeBuilding('shelter', s.x + 18, s.y - 8); g.money = 300000;
      g.reputation = 85; g.events.auto = false;
      g.scenarioState = { broke: false };
    },
    tick(g) {
      if (!g.scenarioState.broke && g.day >= 2 && g.hour >= 14) {
        g.scenarioState.broke = true;
        const d = g.dinos.find((x) => x.species === 'indom');
        if (d) {
          d.rageT = 60; d.stress = 100; d.deterT = 0;
          // she clawed through a weak spot in the wall nearest her
          const w = g.world;
          const p = w.bfs(d.tx, d.ty, w.dinoPass, (x, y) => isSolidFence(w.fence[w.idx(x, y)]), 4000);
          if (p && p.length) { const [fx, fy] = p[p.length - 1]; g.damageFence(fx, fy, 99999, d); }
          g.log(`${d.name} the Indominus vanished from the paddock cameras... then tore through the wall!`, 'bad', d, true);
        }
      }
    },
    medal(g) { const d = g.stats.deaths; return d === 0 && g.day <= 4 ? 'gold' : d <= 2 ? 'silver' : 'bronze'; },
    medalText: 'Gold: no casualties by Day 4 · Silver: 2 or fewer casualties',
    check(g) {
      if (g.stats.deaths >= 6) return 'lose';
      if (g.stats.guestsTotal >= 1800) return 'win';
      if (g.day > 5) return 'lose';
      return null;
    },
    progress(g) { return `Guests ${g.stats.guestsTotal}/1800 · Casualties ${g.stats.deaths}/5 · Day ${g.day}/5`; },
  },
  siege: {
    name: 'Raptor Siege',
    blurb: 'A raptor pack is loose and the gate is sealed. The rescue helicopter lands on Day 3 at noon. Keep the guests alive until then.',
    objective: 'Lose fewer than 8 guests before Day 3, 12:00',
    setup(g) {
      g.money = 200000;
      const P = buildStarterPark(g, { staff: true });
      spawnIn(g, 'galli', P[0], 2); spawnIn(g, 'para', P[2], 2);
      const w = g.world, s = g.start;
      // the raptors are already out, prowling the east side
      for (let k = 0; k < 6; k++) for (let t = 0; t < 200; t++) {
        const x = s.x + randi(10, 24), y = s.y - randi(2, 8);
        if (w.dinoPass(x, y) && !w.path[w.idx(x, y)]) { const d = new Dino(g, 'raptor', x, y); d.growth = 1; d.hunger = 40; g.dinos.push(d); break; }
      }
      w.computeRegions(); g.updateLoose();
      g.time = 9; g.gateLocked = true; g.events.auto = false;
      // pre-fill the park with guests
      const gate = g.buildingsOfType('gate')[0];
      const acc = w.accessTiles(gate);
      for (let k = 0; k < 120 && acc.length; k++) { const gu = new Guest(g, pick(acc)); for (let m = 0; m < 30; m++) gu.update(0.5); g.guests.push(gu); }
    },
    medal(g) { const d = g.stats.deaths; return d === 0 ? 'gold' : d <= 3 ? 'silver' : 'bronze'; },
    medalText: 'Gold: no casualties · Silver: 3 or fewer',
    check(g) {
      if (g.stats.deaths >= 8) return 'lose';
      if (g.time >= 48 + 12) { g.gateLocked = false; return 'win'; }
      return null;
    },
    progress(g) { const h = Math.max(0, 60 - g.time); return `Casualties ${g.stats.deaths}/7 · Rescue in ${Math.floor(h)}h`; },
  },
  storms: {
    name: 'Storm Season',
    blurb: 'Monsoon season. A storm hits every single day, knocking out power and fences. Keep the park profitable anyway.',
    objective: 'Reach $650,000 in the bank by Day 12',
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
    medal(g) { return g.day <= 8 ? 'gold' : g.day <= 10 ? 'silver' : 'bronze'; },
    medalText: 'Gold: by Day 8 · Silver: by Day 10',
    check(g) {
      if (g.money >= 650000) return 'win';
      if (g.day > 12) return 'lose';
      return null;
    },
    progress(g) { return `${fmtMoney(g.money)} / $650k · Day ${g.day}/12`; },
  },
};
