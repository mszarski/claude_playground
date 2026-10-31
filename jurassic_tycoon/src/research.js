// ---------- Research Lab: upgrades and genome unlocks ----------
'use strict';

const RESEARCH = [
  { id: 'fence2', name: 'Reinforced Fences', cost: 30000, hours: 30, desc: 'Electric fences get 60% more integrity.' },
  { id: 'fence3', name: 'High-Voltage Grid', cost: 70000, hours: 48, req: ['fence2'], desc: 'Shocks deter longer, and only the very strongest can push through a live fence.' },
  { id: 'darts', name: 'Fast-Acting Tranquilizers', cost: 40000, hours: 36, desc: 'Every dinosaur needs one dart fewer to go down, and darts miss a running target less often.' },
  { id: 'rangers', name: 'Ranger Training', cost: 35000, hours: 30, desc: 'Rangers run 30% faster and shoot from further away.' },
  { id: 'power', name: 'Efficient Turbines', cost: 45000, hours: 36, desc: 'Power plants produce 40% more electricity.' },
  { id: 'medicine', name: 'Dinosaur Medicine', cost: 30000, hours: 30, desc: 'Illness heals twice as fast, with or without a Vet Clinic.' },
  { id: 'feeders', name: 'Automated Feeders', cost: 35000, hours: 30, desc: 'Each meal costs half as much.' },
  { id: 'sensors', name: 'Motion Sensors', cost: 55000, hours: 40, req: ['rangers'], desc: 'Guests are warned from further away and camouflage no longer hides escapees.' },
  { id: 'acu', name: 'ACU Gunship', cost: 60000, hours: 40, req: ['darts'], desc: 'ACU strikes cost $3k instead of $8k and helicopters fly faster.' },
  { id: 'marketing', name: 'Marketing Campaign', cost: 50000, hours: 24, desc: 'Permanently draws 15% more visitors.' },
  { id: 'gen_trike', name: 'Genome: Triceratops', cost: 30000, hours: 24, genome: 'trike', desc: 'Unlocks Triceratops without needing stars.' },
  { id: 'gen_raptor', name: 'Genome: Velociraptor', cost: 60000, hours: 36, genome: 'raptor', req: ['gen_trike'], desc: 'Unlocks Velociraptors without needing stars.' },
  { id: 'gen_ptera', name: 'Genome: Pteranodon', cost: 50000, hours: 30, genome: 'ptera', desc: 'Unlocks Pteranodons without needing stars.' },
  { id: 'gen_trex', name: 'Genome: Tyrannosaurus', cost: 140000, hours: 60, genome: 'trex', req: ['gen_raptor'], desc: 'Unlocks the T. rex without needing stars.' },
  { id: 'gen_spino', name: 'Genome: Spinosaurus', cost: 160000, hours: 60, genome: 'spino', req: ['gen_trex'], desc: 'Unlocks Spinosaurus without needing stars.' },
  { id: 'gen_indom', name: 'Hybrid: Indominus', cost: 300000, hours: 90, genome: 'indom', req: ['gen_trex', 'gen_raptor', 'fence3'], desc: 'Splice an Indominus without a 5-star rating. What could go wrong?' },
];
const RESEARCH_BY_ID = Object.fromEntries(RESEARCH.map((r) => [r.id, r]));

// Base values that research modifies (restored for every new game)
const RESEARCH_BASE = { elecHp: FENCE_DEF[F_ELECTRIC].hp, supply: BUILDINGS.power.supply };

class ResearchState {
  constructor(game) { this.game = game; this.done = new Set(); this.active = null; }
  has(id) { return this.done.has(id); }
  available(r) { return !this.done.has(r.id) && (!r.req || r.req.every((q) => this.done.has(q))); }
  labs() { return this.game.buildingsOfType('lab').filter((b) => b.powered).length; }
  start(id) {
    const r = RESEARCH_BY_ID[id], g = this.game;
    if (!r || !this.available(r)) return 'Not available yet.';
    if (!g.hasBuilding('lab')) return 'Build a Research Lab first.';
    if (this.active) return 'Another project is already running.';
    if (!g.canAfford(r.cost)) return 'Not enough money.';
    g.spend(r.cost, 'ops');
    this.active = { id, progress: 0 };
    g.log(`Research started: ${r.name}.`, 'info');
    return null;
  }
  update(dt) {
    if (!this.active) return;
    const g = this.game, n = this.labs();
    if (!n) return;
    const r = RESEARCH_BY_ID[this.active.id];
    this.active.progress += (dt / g.secPerHour) * (1 + (n - 1) * 0.6);
    if (this.active.progress >= r.hours) {
      this.done.add(r.id); this.active = null;
      this.apply();
      g.log(`Research complete: ${r.name}! ${r.desc}`, 'goal', null, true);
      g.sound('fanfare');
    }
  }
  // Push numeric effects into the shared definitions
  apply() {
    const g = this.game, w = g.world;
    const oldHp = FENCE_DEF[F_ELECTRIC].hp;
    FENCE_DEF[F_ELECTRIC].hp = RESEARCH_BASE.elecHp * (this.has('fence2') ? 1.6 : 1);
    if (oldHp !== FENCE_DEF[F_ELECTRIC].hp) for (let i = 0; i < w.fence.length; i++) if (w.fence[i] === F_ELECTRIC) w.fenceHp[i] *= FENCE_DEF[F_ELECTRIC].hp / oldHp;
    BUILDINGS.power.supply = RESEARCH_BASE.supply * (this.has('power') ? 1.4 : 1);
    w.powerDirty = true;
  }
}
