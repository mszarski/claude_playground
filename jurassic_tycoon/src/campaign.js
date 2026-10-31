// ---------- Campaign: five chapters, each unlocking a perk for the ones after it ----------
'use strict';

// only the raptors count: a storm letting the gallimimus wander is not what this chapter is about
const rapEsc = (g) => (g.stats.escBySp && g.stats.escBySp.raptor) || 0;
const CAMPAIGN_ORDER = ['c1', 'c2', 'c3', 'c4', 'c5'];
const CAMPAIGN_PERKS = {
  c1: { name: 'Investor confidence', desc: '+$100k starting money', apply: (g) => { g.money += 100000; } },
  c2: { name: 'Ranger training', desc: 'Ranger Training researched', apply: (g) => g.research.done.add('rangers') },
  c3: { name: 'Storm-tested fences', desc: 'Reinforced Fences researched', apply: (g) => g.research.done.add('fence2') },
  c4: { name: 'ACU veterans', desc: 'Fast tranquilizers and ACU Gunship researched', apply: (g) => { g.research.done.add('darts'); g.research.done.add('acu'); } },
};

function loadCampaign() { try { return JSON.parse(localStorage.getItem('jt_campaign') || '{}'); } catch (e) { return {}; } }
function saveCampaign(id, medal, day) {
  const c = loadCampaign();
  const prev = c[id];
  if (!prev || MEDAL_RANK[medal] > MEDAL_RANK[prev.medal] || (MEDAL_RANK[medal] === MEDAL_RANK[prev.medal] && day < prev.day)) c[id] = { medal, day };
  try { localStorage.setItem('jt_campaign', JSON.stringify(c)); } catch (e) { /* storage unavailable */ }
  return c;
}
function campaignUnlocked(id) {
  const i = CAMPAIGN_ORDER.indexOf(id), done = loadCampaign();
  return i <= 0 || !!done[CAMPAIGN_ORDER[i - 1]];
}
// Perks earned from chapters finished before this one
function campaignPerks(id) {
  const done = loadCampaign();
  return CAMPAIGN_ORDER.slice(0, CAMPAIGN_ORDER.indexOf(id)).filter((k) => done[k] && CAMPAIGN_PERKS[k]).map((k) => CAMPAIGN_PERKS[k]);
}
function applyPerks(g, id) {
  const perks = campaignPerks(id);
  for (const p of perks) p.apply(g);
  g.research.apply();
  if (perks.length) g.log(`Campaign perks: ${perks.map((p) => p.name).join(', ')}.`, 'good', null, true);
}

// A checklist line for the goal panel
const tick = (ok, text) => `${ok ? '✓' : '○'} ${text}`;

const CHAPTERS = {
  c1: {
    endDay: 15,
    chapter: 1, name: 'Opening Season', size: 'small', difficulty: 'normal', hints: true,
    blurb: 'Hammond has the island, the money and a dream. Build the park from nothing and prove people will come.',
    objective: 'Reach 2 stars, have 120 guests in the park at once and earn $150,000 from guests, by Day 15',
    setup(g) { g.money = 500000; applyPerks(g, 'c1'); g.scenarioState = { peak: 0 }; g.stats.guestIncome = 0; },
    tick(g) { g.scenarioState.peak = Math.max(g.scenarioState.peak || 0, g.guests.length); },
    check(g) {
      if (g.stars >= 2 && (g.scenarioState.peak || 0) >= 120 && (g.stats.guestIncome || 0) >= 150000) return 'win';
      if (g.day > 15) return 'lose';
      return null;
    },
    progress(g) { return [tick(g.stars >= 2, `★ ${g.stars}/2`), tick((g.scenarioState.peak || 0) >= 120, `Guests ${Math.max(g.guests.length, g.scenarioState.peak || 0)}/120`), tick((g.stats.guestIncome || 0) >= 150000, `Earned ${fmtMoney(g.stats.guestIncome || 0)} / $150k`), `Day ${g.day}/15`].join(' · '); },
    medal(g) { return g.day <= 8 ? 'gold' : g.day <= 11 ? 'silver' : 'bronze'; },
    medalText: 'Gold: by Day 8 · Silver: by Day 11',
  },
  c2: {
    endDay: 12,
    chapter: 2, name: 'Clever Girls', size: 'medium', difficulty: 'normal',
    blurb: 'The board wants the raptors on show. Muldoon wants them shot. Exhibit a pack and put on raptor shows without losing control.',
    objective: 'Exhibit 3 raptors and hold 24 raptor shows by Day 12, with no more than 4 raptor escapes',
    setup(g) {
      g.money = 380000;
      const P = buildStarterPark(g, { staff: true });
      spawnIn(g, 'galli', P[0], 3); spawnIn(g, 'para', P[0], 2); spawnIn(g, 'dilo', P[2], 2);
      g.research.done.add('gen_trike'); g.research.done.add('gen_raptor');
      applyPerks(g, 'c2');
      g.stats.shows = 0;
    },
    check(g) {
      const raptors = g.dinos.filter((d) => d.species === 'raptor' && !d.loose && !d.carried).length;
      if (raptors >= 3 && (g.stats.shows || 0) >= 24) return 'win';
      if (rapEsc(g) > 4 || g.day > 12) return 'lose';
      return null;
    },
    progress(g) {
      const raptors = g.dinos.filter((d) => d.species === 'raptor' && !d.loose && !d.carried).length;
      return [tick(raptors >= 3, `Raptors ${raptors}/3`), tick((g.stats.shows || 0) >= 24, `Shows ${g.stats.shows || 0}/24`), `Raptor escapes ${rapEsc(g)}/4`, `Day ${g.day}/12`].join(' · ');
    },
    medal(g) { const e = rapEsc(g); return e === 0 ? 'gold' : e <= 2 ? 'silver' : 'bronze'; },
    medalText: 'Gold: no raptor escapes · Silver: 2 or fewer',
  },
  c3: {
    endDay: 26,
    chapter: 3, name: 'Monsoon', size: 'medium', difficulty: 'normal',
    blurb: 'Storm season. A storm rolls in every other day and empties the park. Grow anyway: every dollar guests spend counts, whatever you spend on building.',
    objective: 'Earn $275,000 from guests (tickets, shops and rides) and keep reputation 50+, by Day 26',
    setup(g) {
      g.money = 300000;
      const P = buildStarterPark(g, { staff: true });
      spawnIn(g, 'para', P[0], 3); spawnIn(g, 'stego', P[0], 1);
      spawnIn(g, 'trike', P[1], 2);
      spawnIn(g, 'dilo', P[2], 2);
      // the monsoon starts on Day 15
      g.time = 14 * 24 + 7.5; g.lastDay = g.day; g.events.nextDay = g.day + 2;
      applyPerks(g, 'c3');
      g.scenarioState = { lastStorm: g.day };
      g.stats.guestIncome = 0;
    },
    tick(g) { if (g.day >= g.scenarioState.lastStorm + 2 && g.hour >= 10) { g.scenarioState.lastStorm = g.day; g.events.trigger('storm'); } },
    check(g) {
      if ((g.stats.guestIncome || 0) >= 275000 && g.reputation >= 50) return 'win';
      if (g.day > 26) return 'lose';
      return null;
    },
    progress(g) { return [tick((g.stats.guestIncome || 0) >= 275000, `Earned ${fmtMoney(g.stats.guestIncome || 0)} / $275k`), tick(g.reputation >= 50, `Rep ${Math.round(g.reputation)}/50`), `Day ${g.day}/26`].join(' · '); },
    medal(g) { return g.day <= 23 ? 'gold' : g.day <= 25 ? 'silver' : 'bronze'; },
    medalText: 'Gold: by Day 23 · Silver: by Day 25',
  },
  c4: {
    endDay: 18,
    chapter: 4, name: 'Site B', size: 'medium', difficulty: 'normal',
    blurb: 'InGen\'s abandoned breeding island. The animals have been roaming free for years. Round them up, then open the gates (Main Gate) to a real park.',
    objective: 'Contain every dinosaur and reach 3 stars by Day 18',
    setup(g) {
      const w = g.world, s = g.start;
      for (let y = s.y - 10; y <= s.y - 6; y++) for (let x = s.x + 3; x <= s.x + 7; x++) { const i = w.idx(x, y); w.terrain[i] = T_GRASS; w.path[i] = 0; w.fence[i] = 0; }
      w.invalidate();
      g.money = 1e9; g.placeBuilding('helipad', s.x + 4, s.y - 9);
      g.money = 650000;
      roamSpawn(g, 'galli', 4); roamSpawn(g, 'para', 3); roamSpawn(g, 'stego', 2); roamSpawn(g, 'trike', 2); roamSpawn(g, 'anky', 1, 22); roamSpawn(g, 'dilo', 2, 22); roamSpawn(g, 'raptor', 2, 26);
      w.computeRegions(); g.updateLoose();
      g.parkClosed = true;
      g.log('The park stays closed while the island is overrun. Open it from the Main Gate when it is safe.', 'warn', null, true);
      applyPerks(g, 'c4');
    },
    check(g) {
      if (!g.dinos.length) return 'lose';
      const free = g.dinos.filter((d) => d.loose || d.carried || d.sedatedT > 0).length;
      if (!free && g.stars >= 3) return 'win';
      if (g.day > 18) return 'lose';
      return null;
    },
    progress(g) { const free = g.dinos.filter((d) => d.loose || d.carried).length; return [tick(!free, `Contained ${g.dinos.length - free}/${g.dinos.length}`), tick(g.stars >= 3, `★ ${g.stars}/3`), `Day ${g.day}/18`].concat(g.parkClosed ? ['Park closed: open it at the Main Gate'] : []).join(' · '); },
    medal(g) { return g.day <= 9 ? 'gold' : g.day <= 12 ? 'silver' : 'bronze'; },
    medalText: 'Gold: by Day 9 · Silver: by Day 12',
  },
  c5: {
    chapter: 5, name: 'Jurassic World', size: 'large', difficulty: 'normal',
    blurb: 'The big one. A whole island, every attraction and the whole world watching. Build the greatest park on Earth.',
    objective: 'Reach 5 stars with a T. rex on show, a Mosasaur Lagoon and 400 guests at once',
    setup(g) { g.money = 800000; applyPerks(g, 'c5'); g.scenarioState = { peak: 0 }; },
    tick(g) { g.scenarioState.peak = Math.max(g.scenarioState.peak || 0, g.guests.length); },
    check(g) {
      const rex = g.dinos.some((d) => d.species === 'trex' && !d.loose && !d.carried);
      if (g.stars >= 5 && rex && g.hasBuilding('lagoon') && (g.scenarioState.peak || 0) >= 400) return 'win';
      return null;
    },
    progress(g) {
      const rex = g.dinos.some((d) => d.species === 'trex' && !d.loose && !d.carried);
      return [tick(g.stars >= 5, `★ ${g.stars}/5`), tick(rex, 'T. rex'), tick(g.hasBuilding('lagoon'), 'Lagoon'), tick((g.scenarioState.peak || 0) >= 400, `Guests ${Math.max(g.guests.length, g.scenarioState.peak || 0)}/400`), `Day ${g.day}`].join(' · ');
    },
    medal(g) { return g.day <= 45 ? 'gold' : g.day <= 60 ? 'silver' : 'bronze'; },
    medalText: 'Gold: by Day 45 · Silver: by Day 60',
  },
};
for (const [k, c] of Object.entries(CHAPTERS)) SCENARIOS[k] = Object.assign({ campaign: true, decisions: true }, c);
