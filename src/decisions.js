// ---------- InGen decisions: calls with choices, timed park modifiers, deadline contracts ----------
'use strict';

const DAY_H = 24;

// Timed modifiers a choice can put on the park. Several can stack; numbers multiply.
const MODS = {
  ad_blitz: { label: 'TV ad campaign', icon: '📺', guestMul: 1.4 },
  half_price: { label: 'Half-price tickets', icon: '🎟', guestMul: 1.6, ticketMul: 0.5 },
  night_safari: { label: 'Night safari', icon: '☾', nightGuests: 1, carnStress: 15 },
  cheap_fences: { label: 'Maintenance budget cut', icon: '✂', fenceDmgMul: 1.5 },
  hazard_pay: { label: 'Ranger hazard pay', icon: '$', upkeepAdd: 400, perStation: true },
  ranger_slowdown: { label: 'Ranger work-to-rule', icon: '✊', rangerSpeed: 0.7 },
  stormproof: { label: 'Storm-hardened fences', icon: '⛨', stormproof: 1 },
  goat_shortage: { label: 'Goat shortage', icon: '🐐', feedMul: 2 },
  celebrity_buzz: { label: 'Celebrity buzz', icon: '★', guestMul: 1.4 },
  bad_press: { label: 'Bad press', icon: '📰', guestMul: 0.75 },
};

// Promises with a deadline. check() returns 'win', 'fail' or null; on the deadline a null counts as fail
// unless the contract sets passOnDeadline (the "nothing went wrong" kind).
const CONTRACTS = {
  trex_deal: {
    term: 7, // days until the payback comes due
    label: () => 'Investors: exhibit a T. rex',
    check: (g) => g.dinos.some((d) => d.species === 'trex' && !d.loose && !d.carried) ? 'win' : null,
    win: (g) => { g.earn(60000, 'grants'); g.reputation = Math.min(100, g.reputation + 6); g.log('The investors are thrilled with your T. rex. Bonus: +$60k', 'good', null, true); },
    fail: (g) => { g.spend(250000, 'fines'); g.log('No T. rex by the deadline. The investors claw back $250k.', 'bad', null, true); },
  },
  species_deal: {
    term: 6, // days until the payback comes due
    label: (c) => `Board: exhibit ${c.data.n} species`,
    check: (g, c) => g.speciesCount() >= c.data.n ? 'win' : null,
    win: (g) => { g.earn(60000, 'grants'); g.reputation = Math.min(100, g.reputation + 5); g.log('The board is delighted with the new exhibits: +$60k', 'good', null, true); },
    fail: (g) => { g.spend(120000, 'fines'); g.reputation = Math.max(0, g.reputation - 4); g.log('The board wanted more species. They cut your budget by $120k.', 'bad', null, true); },
  },
  lawyer_visit: {
    label: () => 'Lawyer tour: no incidents',
    passOnDeadline: true,
    check: (g, c) => (g.stats.escapes > c.data.escapes || g.stats.deaths > c.data.deaths) ? 'fail' : null,
    win: (g) => { g.earn(100000, 'grants'); g.reputation = Math.min(100, g.reputation + 8); g.log('The lawyer signed off on the park. "I\'ll admit, I\'m impressed." +$100k', 'good', null, true); },
    fail: (g) => { g.reputation = Math.max(0, g.reputation - 10); g.addMod('bad_press', 3); g.log('The lawyer saw everything and is calling the press. Reputation −10.', 'bad', null, true); },
  },
  welfare_check: {
    label: () => 'Welfare inspection: comfort 65+',
    passOnDeadline: false,
    check: (g, c) => {
      if (g.time < c.until - 0.5) return null;
      const ds = g.dinos.filter((d) => !d.carried);
      if (!ds.length) return 'fail';
      return ds.reduce((a, d) => a + d.comfort, 0) / ds.length >= 65 ? 'win' : 'fail';
    },
    win: (g) => { g.earn(50000, 'grants'); g.reputation = Math.min(100, g.reputation + 12); g.log('Animal-welfare inspectors praised your paddocks! +$50k, reputation +12', 'good', null, true); },
    fail: (g) => { g.reputation = Math.max(0, g.reputation - 8); g.log('The welfare report says your dinosaurs are miserable. Reputation −8.', 'bad', null, true); },
  },
  hammond_stars: {
    label: () => 'Hammond: reach 4 stars',
    check: (g) => g.stars >= 4 ? 'win' : null,
    win: (g) => { g.reputation = Math.min(100, g.reputation + 6); g.log('Hammond is overjoyed: "Spared no expense, and it shows!"', 'good', null, true); },
    fail: (g) => { g.reputation = Math.max(0, g.reputation - 15); g.log('Hammond is very, very disappointed. Reputation −15.', 'bad', null, true); },
  },
  celebrity: {
    label: () => 'Celebrity visit: nothing loose',
    check: (g, c) => {
      if (g.time < c.until - 0.5) return null;
      return g.creatures().some((d) => d.loose && !d.carried) ? 'fail' : 'win';
    },
    win: (g) => { g.reputation = Math.min(100, g.reputation + 15); g.addMod('celebrity_buzz', 3); g.log('The celebrity loved it! Photos everywhere. Reputation +15 and a rush of visitors.', 'good', null, true); },
    fail: (g) => { g.reputation = Math.max(0, g.reputation - 12); g.addMod('bad_press', 3); g.log('The celebrity was chased by a dinosaur on live TV. Reputation −12.', 'bad', null, true); },
  },
};

// The calls themselves. prep() picks parameters when the call comes in (null = not now).
const DECISIONS = {
  trex_deal: {
    from: 'InGen Investors', icon: '💼',
    prep: (g) => (g.stars >= 2 && !g.dinos.some((d) => d.species === 'trex')) ? {} : null,
    text: (g) => `"We want a Tyrannosaurus on exhibit within 7 days. We'll wire <b>$150,000</b> today to help." If there's no T. rex by Day ${g.day + 7}, they want <b>$250,000</b> back.`,
    choices: [
      { label: 'Shake on it', hint: '+$150k now · T. rex by the deadline or −$250k', apply: (g) => { g.earn(150000, 'grants'); g.addContract('trex_deal', 7); } },
      { label: 'Not yet', hint: 'Nothing happens', apply: () => {} },
    ],
  },
  species_deal: {
    from: 'InGen Board', icon: '🏛',
    prep: (g) => { const n = g.speciesCount() + 2; return g.dinos.length && n <= 8 ? { n } : null; },
    text: (g, p) => `"Variety sells tickets." The board offers <b>$80,000</b> now if you exhibit <b>${p.n} species</b> by Day ${g.day + 6}. Miss it and they cut <b>$120,000</b>.`,
    choices: [
      { label: 'Accept', hint: '+$80k now · bonus $60k, or −$120k if you miss', apply: (g, p) => { g.earn(80000, 'grants'); g.addContract('species_deal', 6, { n: p.n }); } },
      { label: 'Decline', hint: 'Nothing happens', apply: () => {} },
    ],
  },
  budget_cut: {
    from: 'Accounting', icon: '🧾',
    prep: (g) => g.countFences() > 20 ? {} : null,
    text: () => `"We could save a fortune on fence maintenance." Accept and you get <b>$90,000</b>, but for 5 days every fence takes <b>50% more damage</b>.`,
    choices: [
      { label: 'Cut costs', hint: '+$90k · fences weaker for 5 days', apply: (g) => { g.earn(90000, 'grants'); g.addMod('cheap_fences', 5); } },
      { label: 'Safety first', hint: '+2 reputation', apply: (g) => { g.reputation = Math.min(100, g.reputation + 2); } },
    ],
  },
  lawyer_visit: {
    from: 'InGen Legal', icon: '⚖',
    prep: (g) => g.dinos.length >= 3 ? {} : null,
    text: () => `The company lawyer wants a <b>2-day tour</b>. If there are no escapes and nobody gets eaten, he signs off: <b>+$100k, +8 reputation</b>. Any incident and he goes straight to the press.`,
    choices: [
      { label: 'Roll out the red carpet', hint: 'Keep everything contained for 2 days', apply: (g) => g.addContract('lawyer_visit', 2, { escapes: g.stats.escapes, deaths: g.stats.deaths }) },
      { label: 'Postpone him', hint: '−3 reputation', apply: (g) => { g.reputation = Math.max(0, g.reputation - 3); } },
    ],
  },
  ad_blitz: {
    from: 'Ad Agency', icon: '📺',
    prep: (g) => g.guests.length > 15 ? {} : null,
    text: () => `"Picture it: a T. rex, a sunset, your logo." <b>$60,000</b> buys 4 days of TV spots and <b>40% more visitors</b>.`,
    choices: [
      { label: 'Run the ads', hint: '−$60k · +40% visitors for 4 days', apply: (g) => { g.spend(60000, 'ops'); g.addMod('ad_blitz', 4); } },
      { label: 'No thanks', hint: 'Nothing happens', apply: () => {} },
    ],
  },
  half_price: {
    from: 'Marketing', icon: '🎟',
    prep: (g) => g.guests.length > 10 ? {} : null,
    text: () => `A half-price weekend? For 3 days tickets earn <b>half as much</b>, but <b>60% more visitors</b> come, and the goodwill lifts your reputation.`,
    choices: [
      { label: 'Half price!', hint: '3 days: ½ ticket income, +60% visitors · +4 reputation', apply: (g) => { g.addMod('half_price', 3); g.reputation = Math.min(100, g.reputation + 4); } },
      { label: 'Full price', hint: 'Nothing happens', apply: () => {} },
    ],
  },
  night_safari: {
    from: 'Guest Services', icon: '☾',
    prep: (g) => g.dinos.some((d) => d.sp.diet === 'carn') ? {} : null,
    text: () => `Guests are begging for night tours. For 4 days visitors keep coming <b>after dark</b>, but the carnivores get <b>restless</b> with flashlights in their faces.`,
    choices: [
      { label: 'Open after dark', hint: 'Full crowds at night · carnivores +15 stress', apply: (g) => g.addMod('night_safari', 4) },
      { label: 'Close at sunset', hint: 'Nothing happens', apply: () => {} },
    ],
  },
  tv_feeding: {
    from: 'Nature Channel', icon: '🎥',
    prep: (g) => g.dinos.some((d) => d.sp.diet === 'carn' && !d.loose) ? {} : null,
    text: () => `A TV crew wants close-ups of feeding time. <b>+$40,000 and +12 reputation</b>, but the lights and cameras will <b>agitate every carnivore</b>.`,
    choices: [
      { label: 'Lights, camera…', hint: '+$40k, +12 rep · carnivores +35 stress now', apply: (g) => { g.earn(40000, 'grants'); g.reputation = Math.min(100, g.reputation + 12); for (const d of g.dinos) if (d.sp.diet === 'carn') d.stress = Math.min(100, d.stress + 35); } },
      { label: 'Not today', hint: 'Nothing happens', apply: () => {} },
    ],
  },
  wu_embryo: {
    from: 'Dr. Wu, Genetics', icon: '🧬',
    prep: (g) => {
      if (!g.hasBuilding('hatchery') || g.stars < 1 || g.freeEgg) return null;
      const opts = ['raptor', 'trex', 'spino', 'anky', 'brachio'].filter((s) => !g.isUnlocked(s) || g.stars >= 2);
      return opts.length ? { species: pick(opts) } : null;
    },
    text: (g, p) => `"We filled the gaps in the genome with frog DNA." Wu offers a <b>free ${SPECIES[p.species].name} egg</b>, even if it isn't unlocked yet. It hatches <b>unstable</b>: stronger, and more prone to stress.`,
    choices: [
      { label: 'Take the embryo', hint: (p) => `Next ${SPECIES[p.species].name} hatch is free · it will be unstable`, apply: (g, p) => { g.freeEgg = p.species; g.log(`Dr. Wu's ${SPECIES[p.species].name} embryo is ready. Use Hatch to place it (free).`, 'info', null, true); } },
      { label: 'Life finds a way. No.', hint: '+1 reputation', apply: (g) => { g.reputation = Math.min(100, g.reputation + 1); } },
    ],
  },
  zoo_offer: {
    from: 'San Diego Zoo', icon: '🦕',
    prep: (g) => {
      const ds = g.dinos.filter((d) => !d.loose && !d.carried && d.growth >= 1 && !d.elderly && d.sp.cost >= 40000);
      if (!ds.length) return null;
      const d = pick(ds);
      return { id: d.id, name: d.name, species: d.species, price: Math.round(d.sp.cost * 0.8 / 1000) * 1000 };
    },
    text: (g, p) => `The mainland zoo wants <b>${p.name} the ${SPECIES[p.species].name}</b> and offers <b>${fmtMoney(p.price)}</b>, more than twice the usual price.`,
    choices: [
      { label: 'Sell', hint: (p) => `+${fmtMoney(p.price)} · ${p.name} leaves the park`, apply: (g, p) => {
        const d = g.dinos.find((x) => x.id === p.id && !x.dead);
        if (!d || d.loose || d.carried) { g.log('The zoo deal fell through: the animal was unavailable.', 'warn'); return; }
        d.dead = true; g.earn(p.price, 'grants'); g.stats.sold = (g.stats.sold || 0) + 1;
        g.log(`${p.name} the ${SPECIES[p.species].name} was shipped to San Diego (+${fmtMoney(p.price)}).`, 'info', null, true);
      } },
      { label: `Keep`, hint: '+2 reputation: the staff love that one', apply: (g) => { g.reputation = Math.min(100, g.reputation + 2); } },
    ],
  },
  hazard_pay: {
    from: 'Rangers\' Union', icon: '✊',
    prep: (g) => g.buildingsOfType('ranger').length ? { n: g.buildingsOfType('ranger').length } : null,
    text: (g, p) => `The rangers want hazard pay: <b>$400 a day per station</b> (${p.n} station${p.n > 1 ? 's' : ''}) for a week. Refuse and they work to rule: <b>30% slower</b> for 7 days.`,
    choices: [
      { label: 'Pay them', hint: (p) => `−${fmtMoney(400 * p.n)}/day for 7 days`, apply: (g) => g.addMod('hazard_pay', 7) },
      { label: 'Refuse', hint: 'Rangers 30% slower for 7 days', apply: (g) => g.addMod('ranger_slowdown', 7) },
    ],
  },
  welfare_check: {
    from: 'Animal Welfare League', icon: '🌿',
    prep: (g) => g.dinos.length >= 4 ? {} : null,
    text: () => `Inspectors want to visit in <b>3 days</b>. If your dinosaurs' average comfort is <b>65% or more</b> on the day: <b>+$50k, +12 reputation</b>. If not, a scathing report.`,
    choices: [
      { label: 'Welcome them', hint: 'Average comfort 65%+ in 3 days', apply: (g) => g.addContract('welfare_check', 3) },
      { label: 'Turn them away', hint: '−4 reputation', apply: (g) => { g.reputation = Math.max(0, g.reputation - 4); } },
    ],
  },
  hammond: {
    term: 10, // days until the payback comes due
    from: 'John Hammond', icon: '🎩',
    prep: (g) => g.stars === 3 ? {} : null,
    text: (g) => `"Spare no expense!" Hammond wires <b>$200,000</b> of his own money. He expects <b>4 stars</b> by Day ${g.day + 10}, and he'll be very disappointed otherwise (<b>−15 reputation</b>).`,
    choices: [
      { label: 'Thank you, John', hint: '+$200k · 4 stars in 10 days', apply: (g) => { g.earn(200000, 'grants'); g.addContract('hammond_stars', 10); } },
      { label: 'We\'re fine', hint: 'Nothing happens', apply: () => {} },
    ],
  },
  celebrity: {
    from: 'Talent Agency', icon: '★',
    prep: (g) => g.stars >= 2 && g.dinos.length >= 3 ? {} : null,
    text: () => `A movie star wants to visit <b>in 2 days</b>. If nothing is loose when they arrive: <b>+15 reputation</b> and a rush of visitors. If something is loose, it's on every channel.`,
    choices: [
      { label: 'Invite them', hint: 'Nothing loose in 2 days', apply: (g) => g.addContract('celebrity', 2) },
      { label: 'Too risky', hint: 'Nothing happens', apply: () => {} },
    ],
  },
  stormproof: {
    from: 'Engineering', icon: '⛨',
    prep: (g) => g.countFences() > 20 ? {} : null,
    text: () => `Engineering can storm-harden the fences: <b>$50,000</b>, and for 10 days lightning can't knock them down.`,
    choices: [
      { label: 'Harden them', hint: '−$50k · no lightning fence damage for 10 days', apply: (g) => { g.spend(50000, 'ops'); g.addMod('stormproof', 10); } },
      { label: 'We\'ll risk it', hint: 'Nothing happens', apply: () => {} },
    ],
  },
  goat_strike: {
    from: 'Feed Supplier', icon: '🐐',
    prep: (g) => g.dinos.some((d) => d.sp.diet === 'carn') ? {} : null,
    text: () => `The goat farmers are on strike. Pay <b>$30,000</b> for an emergency supplier, or feed costs <b>double</b> for 4 days.`,
    choices: [
      { label: 'Emergency supplier', hint: '−$30k now', apply: (g) => g.spend(30000, 'food') },
      { label: 'Pay double', hint: 'Feed costs ×2 for 4 days', apply: (g) => g.addMod('goat_shortage', 4) },
    ],
  },
  lysine: {
    from: 'InGen Board', icon: '💊',
    prep: (g) => !g.breedingOff && (g.stats.born || 0) >= 1 ? {} : null,
    text: () => `Unplanned hatchlings have the board nervous. Enforce the <b>lysine contingency</b>? Breeding stops and regulators relax (<b>+5 reputation</b>). Or let life find a way.`,
    choices: [
      { label: 'Enforce it', hint: 'Breeding stops · +5 reputation', apply: (g) => { g.breedingOff = true; g.reputation = Math.min(100, g.reputation + 5); } },
      { label: 'Life finds a way', hint: 'Free hatchlings keep coming', apply: () => {} },
    ],
  },
};

// ---- Game mixin: modifiers, contracts and the call schedule ----
const DecisionMixin = {
  addMod(id, days) {
    this.mods = (this.mods || []).filter((m) => m.id !== id);
    this.mods.push({ id, until: this.time + days * DAY_H });
  },
  activeMods() { return (this.mods || []).filter((m) => MODS[m.id]); },
  // Product of a numeric modifier over active mods (or the sum for additive keys)
  mod(key, base = 1) {
    let v = base;
    for (const m of this.activeMods()) {
      const def = MODS[m.id];
      if (def[key] === undefined) continue;
      if (key === 'upkeepAdd') v += def[key] * (def.perStation ? this.buildingsOfType('ranger').length : 1);
      else if (key === 'nightGuests' || key === 'stormproof' || key === 'carnStress') v = Math.max(v, def[key]);
      else v *= def[key];
    }
    return v;
  },
  addContract(id, days, data = {}) {
    (this.contracts = this.contracts || []).push({ id, until: this.time + days * DAY_H, data });
    this.emit('contracts');
  },
  countFences() { let n = 0; const f = this.world.fence; for (let i = 0; i < f.length; i++) if (isSolidFence(f[i])) n++; return n; },

  updateDecisions() {
    // expire modifiers
    if (this.mods && this.mods.length) {
      const before = this.mods.length;
      this.mods = this.mods.filter((m) => m.until > this.time);
      if (this.mods.length !== before) this.emit('contracts');
    }
    // resolve contracts
    if (!this.contracts || !this.contracts.length) return;
    const keep = [];
    for (const c of this.contracts) {
      const def = CONTRACTS[c.id];
      if (!def) continue;
      let r = def.check(this, c);
      if (!r && this.time >= c.until) r = def.passOnDeadline ? 'win' : 'fail';
      if (r === 'win') { def.win(this, c); this.sound('fanfare'); }
      else if (r === 'fail') { def.fail(this, c); this.sound('error'); }
      else keep.push(c);
    }
    if (keep.length !== this.contracts.length) { this.contracts = keep; this.emit('contracts'); }
  },

  // Called each morning: maybe ring the director with a decision
  maybeOfferDecision() {
    if (this.decisionsOn === false || this.tutorial || this.pendingDecision) return;
    if (this.scenario && !(SCENARIOS[this.scenario] && SCENARIOS[this.scenario].decisions)) return;
    if (this.day < 3 || this.day < (this.nextDecisionDay || 0)) return;
    if ((this.contracts || []).length >= 2) return;
    this.seenDecisions = this.seenDecisions || {};
    const cands = [];
    for (const [id, def] of Object.entries(DECISIONS)) {
      if (this.seenDecisions[id] !== undefined && this.day - this.seenDecisions[id] < 12) continue;
      if ((this.contracts || []).some((c) => c.id === id)) continue;
      // a deal whose payback would come due after a timed chapter ends is a free lunch: don't offer it
      if (def.term && this.scenario && SCENARIOS[this.scenario].endDay && this.day + def.term > SCENARIOS[this.scenario].endDay) continue;
      const p = def.prep(this);
      if (p) cands.push({ id, params: p });
    }
    if (!cands.length) return;
    const offer = pick(cands);
    this.seenDecisions[offer.id] = this.day;
    this.nextDecisionDay = this.day + randi(2, 3);
    this.pendingDecision = offer;
    this.emit('decision', offer);
  },

  resolveDecision(choiceIdx) {
    const offer = this.pendingDecision;
    if (!offer) return;
    this.pendingDecision = null;
    const def = DECISIONS[offer.id], ch = def.choices[choiceIdx];
    if (!ch) return;
    ch.apply(this, offer.params);
    this.decisionLog = this.decisionLog || [];
    this.decisionLog.push({ day: this.day, id: offer.id, choice: choiceIdx });
    this.emit('contracts');
  },
};

for (const [k, v] of Object.entries(DecisionMixin)) Game.prototype[k] = v;

// Text helpers shared by the UI
function choiceHint(ch, params) { return typeof ch.hint === 'function' ? ch.hint(params) : ch.hint; }
function contractStatus(g) {
  const out = [];
  for (const c of g.contracts || []) {
    const def = CONTRACTS[c.id]; if (!def) continue;
    const left = Math.max(0, c.until - g.time);
    out.push({ text: def.label(c, g), left: left >= 24 ? `${Math.ceil(left / 24)}d` : `${Math.ceil(left)}h`, kind: 'contract' });
  }
  for (const m of g.activeMods()) {
    const def = MODS[m.id], left = Math.max(0, m.until - g.time);
    out.push({ text: `${def.icon} ${def.label}`, left: left >= 24 ? `${Math.ceil(left / 24)}d` : `${Math.ceil(left)}h`, kind: 'mod' });
  }
  return out;
}
