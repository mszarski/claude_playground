// ---------- UI panels: selection info and modal dialogs (mixed into UI) ----------
'use strict';

class UIPanels {
  // ---------------- info panel ----------------
  bar(v, max, col) { return `<div class="bar"><i style="width:${clamp(v / max, 0, 1) * 100}%;background:${col}"></i></div>`; }

  renderInfo(force) {
    const el = $('#info');
    const o = this.selected;
    const g = this.game, w = g.world;
    if (!o || o.dead || (o.id && o.type && !w.buildings.has(o.id))) {
      if (el.style.display !== 'none') el.style.display = 'none';
      if (o) this.selected = null;
      return;
    }
    el.style.display = 'block';
    let html = '<button class="close" data-a="close">✕</button>';
    let portrait = null;
    if (o.kind === 'dino') {
      const d = o, sp = d.sp;
      const status = d.carried ? '<span class="tag" style="background:#2a5a9a">AIRLIFT</span>' :
        d.sedatedT > 0 ? '<span class="tag" style="background:#4a4a8a">SEDATED</span>' :
          d.loose ? '<span class="tag" style="background:#b01e10">LOOSE!</span>' : '<span class="tag" style="background:#2a6a2a">CONTAINED</span>';
      const sick = (d.sick > 0 ? '<span class="tag" style="background:#5a8a2a">SICK</span>' : '') + (d.unstable ? '<span class="tag" style="background:#6a2a7a" title="Dr. Wu\'s frog-DNA embryo: stronger and more easily stressed">UNSTABLE</span>' : '') + (d.rageT > 0 ? '<span class="tag" style="background:#c82020" title="Enraged: comfort means nothing until she calms down">ENRAGED</span>' : '');
      const diet = `<span class="tag" style="background:${sp.diet === 'carn' ? '#8a3a1a' : '#3a6a2a'}">${sp.diet === 'carn' ? 'CARNIVORE' : 'HERBIVORE'}</span>`;
      html += `<h2>${d.name}</h2><div class="sub">${sp.name} · <i>${sp.sci}</i></div><canvas class="portrait" id="portrait"></canvas>${status}${sick}${diet}`;
      html += `<div class="row"><span>Health</span><span>${Math.round(d.hp)}/${sp.hp}</span></div>${this.bar(d.hp, sp.hp, '#5ac85a')}`;
      html += `<div class="row"><span>Hunger</span><span>${Math.round(d.hunger)}%</span></div>${this.bar(d.hunger, 100, d.hunger > 70 ? '#e04838' : '#f0a030')}`;
      html += `<div class="row"><span>Stress</span><span>${Math.round(d.stress)}%</span></div>${this.bar(d.stress, 100, d.stress > 65 ? '#e04838' : '#c8a040')}`;
      if (!d.isPtera) html += `<div class="row"><span>Comfort</span><span>${Math.round(d.comfort)}%</span></div>${this.bar(d.comfort, 100, '#70b8f0')}`;
      const p = d.comfortParts || {};
      const lbl = { space: 'Space', forest: 'Cover', water: 'Water', social: 'Social', food: 'Food', fear: 'Predators', loose: 'Loose', base: '' };
      const parts = Object.keys(p).filter((k) => lbl[k]).map((k) => `<span class="tag" style="background:${p[k] < 0 ? '#7a2018' : p[k] >= 10 ? '#2a5a2a' : '#5a5a2a'}">${lbl[k]} ${p[k] >= 0 ? '+' : ''}${Math.round(p[k])}</span>`).join('');
      html += `<div>${parts}</div>`;
      if (d.ageDays !== undefined) html += `<div class="row"><span>Age</span><span>${Math.floor(d.ageDays)} days${d.growth < 1 ? ' · hatchling' : d.elderly ? ' · elderly' : ''}</span></div>`;
      html += `<div class="row"><span>Appeal</span><span>${sp.appeal}</span></div><div class="row"><span>Danger</span><span>${'☠'.repeat(Math.ceil(sp.danger / 2)) || '—'}</span></div>`;
      if (d.kills) html += `<div class="row"><span>Kills</span><span style="color:#e04838">${d.kills}</span></div>`;
      html += `<div class="sub">${sp.desc}</div>`;
      html += `<div class="btns"><button data-a="follow" class="${this.follow ? 'on' : ''}">Follow</button>`;
      html += `<button data-a="sedate" ${d.sedatedT > 0 || d.carried || !g.hasBuilding('ranger') ? 'disabled' : ''} title="Rangers will tranquilize it">${d.orderSedate ? 'Sedating…' : 'Sedate'}</button>`;
      if (d.loose && d.sedatedT <= 0 && !d.carried) html += `<button data-a="strike" ${g.helis.length ? '' : 'disabled'} title="Helicopter darts it from the air ($8k)">ACU Strike $8k</button>`;
      html += `<button data-a="relocate" ${!g.hasBuilding('helipad') || d.carried || d.isPtera ? 'disabled' : ''} title="Pick a destination paddock; rangers sedate it and the ACU flies it there">Move…</button>`;
      html += `<button data-a="sell" class="danger" title="Ship to another facility">Sell ${fmtMoney(d.sellValue || sp.cost * 0.35)}</button></div>`;
      portrait = () => {
        const c = $('#portrait'); if (!c) return;
        const S = DINO_SPRITES[d.species];
        const sc = S.w > 30 ? 3 : 4;
        c.width = S.w * sc + 16; c.height = S.h * sc + 8;
        const x = c.getContext('2d'); x.imageSmoothingEnabled = false;
        x.drawImage(d.sedatedT > 0 ? S.sleepR : S.right[0], 8, 4, S.w * sc, S.h * sc);
      };
    } else if (o.kind === 'guest') {
      html += `<h2>Guest</h2><div class="sub">${o.state === 'flee' ? 'Running for their life!' : o.state === 'shelter' ? 'Hiding in a shelter' : o.state === 'leave' ? 'Heading home' : 'Enjoying the park'}</div>`;
      html += `<div class="row"><span>Happiness</span><span>${Math.round(o.happy)}%</span></div>${this.bar(o.happy, 100, '#5ac85a')}`;
      html += `<div class="row"><span>Hunger</span><span>${Math.round(o.hunger)}%</span></div>${this.bar(o.hunger, 100, '#f0a030')}`;
      html += `<div class="row"><span>Bladder</span><span>${Math.round(o.toilet)}%</span></div>${this.bar(o.toilet, 100, '#70b8f0')}`;
      html += `<div class="row"><span>Species seen</span><span>${o.seenSpecies.size}</span></div><div class="row"><span>Spent</span><span>${fmtMoney(o.spent + g.ticket)}</span></div>`;
    } else if (o.kind === 'ranger' || o.kind === 'worker') {
      html += `<h2>${o.kind === 'ranger' ? 'Park Ranger' : 'Engineer'}</h2>`;
      html += `<div class="sub">${o.kind === 'ranger' ? (o.target ? `Pursuing ${o.target.name} the ${o.target.sp.name}` : 'On patrol') : (o.state === 'flee' ? 'Running from a predator!' : o.state === 'repair' ? 'Repairing' : o.state === 'move' ? 'En route to a job' : 'Idle')}</div>`;
      html += `<div class="row"><span>Orders</span><span>${orderText(o).replace('Ordered: ', '')}</span></div>`;
      html += `<div class="sub">${o.kind === 'ranger' ? 'Tranquilizes loose dinosaurs within 4 tiles.' : 'Repairs fences & buildings, rebuilds breached fences.'} ${this.touchUsed ? 'Use Send… to give orders.' : 'Right-click to give orders · shift-drag to select several.'}</div>`;
      if (o.order) html += `<div class="btns"><button data-a="auto">Back to auto</button></div>`;
    } else if (o.kind === 'heli') {
      const st = { parked: 'On the pad', strike: `Darting ${o.target ? o.target.name : 'a target'} from the air`, fetch: `Fetching ${o.target ? o.target.name : 'a sedated dinosaur'}`, deliver: `Flying ${o.cargo ? o.cargo.name : 'cargo'} to a paddock`, return: 'Returning to base' }[o.state] || o.state;
      html += `<h2>ACU Helicopter</h2><div class="sub">${st}</div>`;
      html += `<div class="sub">Airlifts sedated dinosaurs home on its own. ${this.touchUsed ? 'Select it and use Send…' : 'Right-click'} a loose dinosaur for an air strike (${fmtMoney(g.research.has('acu') ? 3000 : 8000)}).</div>`;
    } else if (o.kind === 'fence') {
      const i = w.idx(o.x, o.y), f = w.fence[i];
      if (!f) { this.selected = null; el.style.display = 'none'; return; }
      const def = FENCE_DEF[f] || { name: 'Breached Fence', hp: 1 };
      html += `<h2>${def.name}</h2>`;
      if (f === F_BROKEN) html += `<div class="sub" style="color:#ff9080">Breached! Engineers will rebuild it.</div>`;
      else {
        html += `<div class="row"><span>Integrity</span><span>${Math.round(w.fenceHp[i])}/${def.hp}</span></div>${this.bar(w.fenceHp[i], def.hp, '#5ac85a')}`;
        if (f === F_ELECTRIC) html += `<div class="row"><span>Power</span><span style="color:${w.fencePowered[i] ? '#f8d040' : '#e04838'}">${w.fencePowered[i] ? 'LIVE' : 'OFFLINE'}</span></div>`;
      }
      const orig = f === F_BROKEN ? (w.fenceOrig[i] || F_ELECTRIC) : f;
      if (f === F_BROKEN || w.fenceHp[i] < def.hp) html += `<div class="btns"><button data-a="erepair" class="danger">Emergency repair ${fmtMoney(FENCE_DEF[orig].cost * 3)}</button></div>`;
    } else if (o.kind === 'paddock') {
      const reg = w.regionAt(o.x, o.y);
      if (!reg || reg.public) { this.selected = null; el.style.display = 'none'; return; }
      const ds = g.dinos.filter((d) => !d.carried && w.region[w.idx(d.tx, d.ty)] === reg.id);
      html += `<h2>Paddock</h2>`;
      html += `<div class="row"><span>Size</span><span>${reg.size} tiles</span></div>`;
      html += `<div class="row"><span>Tree cover</span><span>${Math.round(reg.forest / reg.size * 100)}%</span></div>`;
      html += `<div class="row"><span>Water</span><span>${reg.water ? 'Yes' : 'No'}</span></div>`;
      html += `<div class="row"><span>Feeders</span><span>${reg.feeders.herb.length}H / ${reg.feeders.carn.length}C</span></div>`;
      const sp = ds.reduce((a, d) => a + d.sp.space, 0);
      html += `<div class="row"><span>Space used</span><span>${sp}/${reg.size}</span></div>`;
      const eggs = g.eggs.filter((e) => w.regionAt(e.x, e.y) === reg);
      const eggText = eggs.length ? `${eggs.length} egg${eggs.length > 1 ? 's' : ''} hatching (${[...new Set(eggs.map((e) => SPECIES[e.species].name))].join(', ')})` : '';
      html += `<div class="sub">${ds.length ? ds.map((d) => d.name + ' (' + d.sp.name + ')').join(', ') + (eggText ? ' · ' + eggText : '') : eggText || 'Empty. Use Hatch to add dinosaurs.'}</div>`;
      // one-click upkeep
      const cover = reg.forest / reg.size, needTrees = Math.max(0, Math.ceil(reg.size * 0.3) - reg.forest);
      const fences = g.paddockFences(reg), dmg = fences.filter((j) => g.needsRepair(j % w.W, (j / w.W) | 0)).length;
      const diets = new Set(ds.map((d) => d.sp.diet));
      html += `<div class="btns">`;
      if (cover < 0.28 && needTrees) html += `<button data-a="cover" title="Plant trees until about 30% of the paddock is covered">Tree cover ${fmtMoney(needTrees * TOOL_INFO.trees.cost)}</button>`;
      // an empty paddock offers both: its feeder also tells the ACU which captured animals belong here
      if (!reg.feeders.herb.length && (diets.has('herb') || !ds.length)) html += `<button data-a="feedH" title="Herbivores captured by the ACU will be brought here">+ Herb feeder</button>`;
      if (!reg.feeders.carn.length && (diets.has('carn') || !ds.length)) html += `<button data-a="feedC" title="Carnivores captured by the ACU will be brought here">+ Carn feeder</button>`;
      if (dmg) html += `<button data-a="fixfence" class="danger">Repair ${dmg} fence${dmg > 1 ? 's' : ''}</button>`;
      html += `<button data-a="copy" title="Copy this paddock's fences, trees and feeders, then stamp copies elsewhere">Copy layout</button></div>`;
    } else if (o.type) {
      const def = BUILDINGS[o.type];
      html += `<h2>${def.name}</h2><div class="sub">${def.desc}</div>`;
      html += `<div class="row"><span>Condition</span><span>${Math.round(o.hp)}/${o.maxHp}</span></div>${this.bar(o.hp, o.maxHp, '#5ac85a')}`;
      if (def.power) html += `<div class="row"><span>Power</span><span style="color:${o.powered ? '#f8d040' : '#e04838'}">${o.powered ? 'ON' : 'NO POWER'}</span></div>`;
      if (o.type === 'pylon') html += `<div class="row"><span>Grid</span><span style="color:${w.covered[w.idx(o.x, o.y)] ? '#f8d040' : '#e04838'}">${w.covered[w.idx(o.x, o.y)] ? 'CONNECTED' : 'NOT CONNECTED'}</span></div>${w.covered[w.idx(o.x, o.y)] ? '' : '<div class="sub" style="color:#ff9080">Outside the power area. Move it closer to a power plant or a connected pylon.</div>'}`;
      if (o.type === 'power') html += `<div class="row"><span>Output</span><span>${o.offline > 0 || w.power.outage > 0 ? 'OFFLINE' : def.supply + ' MW'}</span></div>`;
      if (o.type === 'gate') html += `<div class="row"><span>Park</span><span style="color:${g.parkClosed ? '#e04838' : '#5ac85a'}">${g.parkClosed ? 'CLOSED' : 'OPEN'}</span></div><div class="btns"><button data-a="toggleopen">${g.parkClosed ? 'Open the park' : 'Close the park'}</button></div>`;
      if (def.cat === 'guest' && !w.accessTiles(o).length) html += `<div class="sub" style="color:#ff9080">Not connected to a path!</div>`;
      if (o.type === 'tour') {
        html += `<div class="row"><span>Queue</span><span>${(o.queue || []).length}</span></div><div class="row"><span>Jeeps</span><span>${g.jeeps.filter((j) => j.station === o).length}</span></div>`;
        if (!w.trackAccess(o).length) html += `<div class="sub" style="color:#ff9080">Needs Tour Track next to it!</div>`;
      }
      if (o.type === 'gyro') {
        html += `<div class="row"><span>Queue</span><span>${(o.queue || []).length}</span></div><div class="row"><span>Spheres</span><span>${(g.gyros || []).filter((x) => x.station === o).length}</span></div>`;
        if (o.noPaddock) html += `<div class="sub" style="color:#ff9080">Must touch a paddock (build it right against a paddock's fence).</div>`;
      }
      if (o.type === 'arena') {
        const night = g.hour < 8 || g.hour >= 20;
        html += `<div class="row"><span>Next show</span><span>${o.noRaptors ? '—' : night ? 'after 08:00' : 'in ' + Math.max(1, Math.ceil((o.showT || 0) / g.secPerHour)) + 'h'}</span></div>`;
        if (o.noRaptors) html += `<div class="sub" style="color:#ff9080">Needs a paddock with 2+ raptors within 6 tiles.</div>`;
        if (o.revenue) html += `<div class="row"><span>Show takings</span><span>${fmtMoney(o.revenue)}</span></div>`;
      }
      if (o.type === 'lagoon') html += `<div class="row"><span>Next leap</span><span>${o.powered ? 'in ' + Math.max(1, Math.ceil((o.showT || 0) / g.secPerHour)) + 'h' : '—'}</span></div><div class="sub">Guests within 9 tiles watch the Mosasaur leap and pay for the show. Run a path past it.</div>`;
      if (o.type === 'feedshow') {
        html += `<div class="row"><span>Shows</span><span>12:00 · 18:00</span></div>`;
        if (o.noCarnivore) html += `<div class="sub" style="color:#ff9080">No carnivore paddock within 6 tiles.</div>`;
        if (o.revenue) html += `<div class="row"><span>Show takings</span><span>${fmtMoney(o.revenue)}</span></div>`;
      }
      if (def.income) html += `<div class="row"><span>Visitors</span><span>${o.visitors}</span></div><div class="row"><span>Revenue</span><span>${fmtMoney(o.revenue || 0)}</span></div>`;
      if (def.shelter) html += `<div class="row"><span>Sheltering</span><span>${o.inside}/${def.shelter}</span></div>`;
      if (o.doorsBroken) html += `<div class="sub" style="color:#ff9080">DOORS BROKEN: guests won't shelter here until engineers repair it (80% condition).</div>`;
      if (def.upkeep) html += `<div class="row"><span>Upkeep</span><span>${fmtMoney(def.upkeep)}/day</span></div>`;
      const fixCost = Math.round((1 - o.hp / o.maxHp) * def.cost * 0.3 / 100) * 100;
      html += `<div class="btns">${o.hp < o.maxHp ? `<button data-a="fixb" title="Instant repair by contractors">Repair ${fmtMoney(Math.max(100, fixCost))}</button>` : ''}<button data-a="demolish" class="danger">Demolish (+${fmtMoney(def.cost * 0.25)})</button></div>`;
    }
    if (force || html !== this.lastInfo) {
      this.lastInfo = html;
      el.innerHTML = html;
      if (portrait) portrait();
      for (const b of el.querySelectorAll('[data-a]')) b.onclick = () => this.infoAction(b.dataset.a);
    }
  }

  infoAction(a) {
    const g = this.game, o = this.selected;
    this.sfx.play('click');
    if (a === 'close') { this.select(null); return; }
    if (a === 'follow') { this.follow = !this.follow; }
    if (a === 'auto' && o) { g.releaseUnits([o]); }
    const reg = o && o.kind === 'paddock' ? g.world.regionAt(o.x, o.y) : null;
    if (reg && !reg.public) {
      if (a === 'cover') { const n = g.plantCover(reg); this.toast(n ? `Planted ${n} trees.` : 'No room for trees.', n ? 'good' : 'warn'); if (n) this.sfx.play('build'); }
      if (a === 'feedH' || a === 'feedC') { const f = g.addFeeder(reg, a === 'feedC' ? 'carn' : 'herb'); if (f) this.sfx.play('build'); else this.toast('No free tile for a feeder (or not enough money).', 'warn'); }
      if (a === 'fixfence') { const r = g.repairFences(reg); this.toast(r.n ? `Repaired ${r.n} fence tiles (${fmtMoney(r.cost)}).${r.blocked ? ' A dinosaur is blocking a gap.' : ''}` : r.blocked ? 'A dinosaur is standing in the gap! Sedate it first.' : 'Not enough money.', r.n ? 'good' : 'warn'); if (r.n) this.sfx.play('build'); }
      if (a === 'copy') { const bp = g.copyPaddock(reg); if (bp) { this.blueprint = bp; this.setTool('stamp'); this.toast(`Layout copied (${bp.w}×${bp.h}). Click to stamp a copy, Esc to stop.`, 'info'); } }
      // keep the paddock selected on a tile that is still inside it
      const t = reg.tiles.find((i) => g.world.dinoPass(i % g.world.W, (i / g.world.W) | 0));
      if (t !== undefined) { const r2 = g.world.regionAt(t % g.world.W, (t / g.world.W) | 0); if (r2) this.selected = { kind: 'paddock', x: t % g.world.W, y: (t / g.world.W) | 0, reg: r2 }; }
    }
    if (a === 'toggleopen') { g.parkClosed = !g.parkClosed; g.log(g.parkClosed ? 'The park is closed: no new visitors until you reopen the gate.' : 'The park is open!', g.parkClosed ? 'warn' : 'good', null, true); }
    if (a === 'sedate' && o) { o.orderSedate = true; g.log(`Rangers dispatched to sedate ${o.name}.`, 'info', o); }
    if (a === 'strike' && o) { const err = g.acuStrike(o); if (err) this.toast(err, 'warn'); }
    if (a === 'erepair' && o) { const err = g.emergencyRepair(o.x, o.y); if (err) this.toast(err, 'warn'); else this.sfx.play('build'); }
    if (a === 'relocate' && o) { this.moveDino = o; this.setTool('movedest'); this.toast(`Click inside the paddock where ${o.name} should go (Esc to cancel).`, 'info'); }
    if (a === 'sell' && o) {
      const err = g.sellDino(o);
      if (err) { this.toast(err, 'warn'); return; }
      this.select(null); return;
    }
    if (a === 'fixb' && o && o.type) {
      const c = Math.max(100, Math.round((1 - o.hp / o.maxHp) * BUILDINGS[o.type].cost * 0.3 / 100) * 100);
      if (!g.canAfford(c)) { this.toast('Not enough money.', 'warn'); return; }
      g.spend(c, 'repairs'); o.hp = o.maxHp; o.doorsBroken = false; g.world.flowCache.clear();
      g.sparks(o.x + o.w / 2, o.y + o.h / 2, '#f8e080', 8); this.sfx.play('build');
      this.toast(`${BUILDINGS[o.type].name} repaired (${fmtMoney(c)}).`, 'good');
    }
    if (a === 'demolish' && o) { g.startBatch(); const ok = g.demolishAt(o.x, o.y); g.endBatch('demolition'); if (ok) this.sfx.play('demolish'); this.select(null); return; }
    this.renderInfo(true);
  }

  // ---------------- modals ----------------
  showModal(html, locked = false) {
    $('#modalBox').innerHTML = html;
    $('#modal').style.display = 'flex';
    this.modalLocked = locked;
    this.hideTip();
  }
  closeModal() { $('#modal').style.display = 'none'; this.modalLocked = false; }

  showSpeciesPicker() {
    const g = this.game;
    let html = `<h1>HATCHERY</h1><p>Choose a species, then ${IS_TOUCH ? 'tap' : 'click'} inside a fenced paddock (no paths inside!). Carnivores will eat herbivores that share their paddock. Pteranodons hatch inside an Aviary.</p>`;
    if (!g.hasBuilding('hatchery')) html += `<p style="color:#ff9080">You need to build a Hatchery first (Dinos menu).</p>`;
    html += `<div class="cards">`;
    for (const k of SPECIES_ORDER) {
      const sp = SPECIES[k];
      const locked = !g.isUnlocked(k);
      html += `<div class="card ${locked ? 'locked' : ''}" data-sp="${k}"><canvas data-spr="${k}"></canvas><div class="nm">${sp.name.toUpperCase()}</div>
        <div>${sp.diet === 'carn' ? '<span style="color:#f08a5a">Carnivore</span>' : '<span style="color:#8ad06a">Herbivore</span>'} · ${g.freeEgg === k ? '<span style="color:#f8d040">FREE (Dr. Wu)</span>' : fmtMoney(sp.cost)}</div>
        <div style="color:#9ab08a">Appeal ${sp.appeal} · Danger ${sp.danger} · Str ${sp.strength}</div>
        <div style="color:#9ab08a">${sp.flying ? 'Needs an Aviary' : `Needs ${sp.space} tiles${sp.social > 1 ? ' · groups of ' + sp.social : ''}`}</div>
        ${locked ? `<div style="color:#f8d040">Unlocks at ${'★'.repeat(sp.unlock)}</div>` : ''}</div>`;
    }
    html += `</div><div class="btns" style="justify-content:flex-end"><button data-close>Close</button></div>`;
    this.showModal(html);
    for (const c of document.querySelectorAll('canvas[data-spr]')) {
      const S = DINO_SPRITES[c.dataset.spr];
      const sc = S.h > 24 ? 2 : 3;
      c.width = S.w * sc; c.height = S.h * sc;
      const x = c.getContext('2d'); x.imageSmoothingEnabled = false;
      x.drawImage(S.right[0], 0, 0, S.w * sc, S.h * sc);
      c.style.height = Math.min(64, S.h * sc) + 'px';
    }
    for (const card of document.querySelectorAll('.card[data-sp]')) {
      card.onclick = () => {
        if (card.classList.contains('locked')) { this.sfx.play('error'); return; }
        this.species = card.dataset.sp; this.setTool('hatch'); this.closeModal(); this.sfx.play('click');
        this.toast(`Click inside a paddock to hatch a ${SPECIES[this.species].name}.`);
      };
    }
    $('[data-close]').onclick = () => this.closeModal();
  }

  showRoster() {
    const g = this.game;
    const mini = (v, col) => `<div class="mini"><i style="width:${clamp(v, 0, 100)}%;background:${col}"></i></div>`;
    const all = g.creatures();
    let html = `<h1>DINOSAURS (${all.length})</h1>`;
    if (!all.length) html += `<p>No dinosaurs yet. Build a Hatchery, fence a paddock, then use Hatch.</p>`;
    else {
      html += `<div class="rhead"><span></span><span>Name</span><span>Health</span><span>Hunger</span><span class="hide">Stress</span><span>Status</span></div><div class="roster">`;
      const order = [...all].sort((a, b) => (b.loose - a.loose) || (b.stress - a.stress));
      for (const d of order) {
        const st = d.carried ? '<span style="color:#70b8f0">AIRLIFT</span>' : d.sedatedT > 0 ? '<span style="color:#a8a8f0">SEDATED</span>' : d.loose ? '<span style="color:#ff6040">LOOSE!</span>' : d.sick > 0 ? '<span style="color:#9ae060">SICK</span>' : d.stress > 65 ? '<span style="color:#f0a030">AGITATED</span>' : '<span style="color:#8ad06a">OK</span>';
        html += `<div class="rrow" data-id="${d.id}"><canvas data-dspr="${d.species}"></canvas><span><b>${d.name}</b> <span style="color:#9ab08a">${d.sp.name}${d.ageDays !== undefined ? ' · ' + Math.floor(d.ageDays) + 'd' + (d.growth < 1 ? ' baby' : d.elderly ? ' old' : '') : ''}</span></span>${mini(d.hp / d.sp.hp * 100, '#5ac85a')}${mini(d.hunger, d.hunger > 70 ? '#e04838' : '#f0a030')}<span class="hide">${mini(d.stress, d.stress > 65 ? '#e04838' : '#c8a040')}</span><span>${st}</span></div>`;
      }
      html += `</div>`;
    }
    html += `<div class="btns" style="justify-content:flex-end;margin-top:8px"><button data-close>Close</button></div>`;
    this.showModal(html);
    for (const c of document.querySelectorAll('canvas[data-dspr]')) {
      const S = DINO_SPRITES[c.dataset.dspr];
      c.width = S.w; c.height = S.h;
      c.getContext('2d').drawImage(S.right[0], 0, 0);
      c.style.width = Math.min(52, S.w * 28 / S.h) + 'px';
    }
    for (const r of document.querySelectorAll('.rrow')) r.onclick = () => {
      const d = g.creatures().find((x) => x.id === +r.dataset.id);
      if (d) { this.closeModal(); this.centerOn(d.x, d.y); this.select(d); }
    };
    $('[data-close]').onclick = () => this.closeModal();
  }

  showResearch() {
    const g = this.game, rs = g.research;
    const labs = g.buildingsOfType('lab').length;
    let html = `<h1>RESEARCH</h1><p>${labs ? `${labs} lab${labs > 1 ? 's' : ''} (${rs.labs()} powered). Extra labs speed research up.` : '<span style="color:#ff9080">Build a Research Lab (Dinos menu) to start researching.</span>'}</p><div class="rgrid">`;
    for (const r of RESEARCH) {
      const done = rs.has(r.id), active = rs.active && rs.active.id === r.id, avail = rs.available(r);
      const cls = done ? 'done' : active ? 'active' : avail ? '' : 'locked';
      let foot;
      if (done) foot = '<span style="color:#8ad06a">✓ Complete</span>';
      else if (active) foot = `<div class="pbar" style="width:100%"><i style="width:${rs.active.progress / r.hours * 100}%"></i></div>`;
      else if (!avail) foot = `<span style="color:#9ab08a">Needs: ${r.req.filter((q) => !rs.has(q)).map((q) => RESEARCH_BY_ID[q].name).join(', ')}</span>`;
      else foot = `<button data-r="${r.id}" ${!labs || rs.active ? 'disabled' : ''}>Research ${fmtMoney(r.cost)} · ${r.hours}h</button>`;
      html += `<div class="rcard ${cls}"><div class="nm">${r.name.toUpperCase()}</div><div>${r.desc}</div>${foot}</div>`;
    }
    html += `</div><div class="btns" style="justify-content:flex-end;margin-top:8px"><button data-close>Close</button></div>`;
    this.showModal(html);
    for (const b of document.querySelectorAll('[data-r]')) b.onclick = () => { const err = rs.start(b.dataset.r); if (err) this.toast(err, 'warn'); else this.sfx.play('build'); this.showResearch(); };
    $('[data-close]').onclick = () => this.closeModal();
  }

  showAudio() {
    const s = this.sfx;
    const row = (id, label, v) => `<div class="row" style="align-items:center;gap:10px"><label for="${id}" style="min-width:90px">${label}</label><input type="range" id="${id}" min="0" max="100" value="${Math.round(v * 100)}" style="flex:1;accent-color:#f8d040"><span id="${id}V" style="min-width:40px;text-align:right">${Math.round(v * 100)}%</span></div>`;
    const html = `<h1>SOUND</h1>${row('volM', 'Master', s.vol * 2)}${row('volS', 'Effects', s.sfxVol)}${row('volU', 'Music', s.musicVol)}
      <p style="color:#9ab08a">The music turns tense during breaches, power failures and alarms.</p>
      <div class="btns" style="justify-content:flex-end"><button data-close>Done</button></div>`;
    this.showModal(html);
    const upd = () => {
      const m = $('#volM').value / 100, e = $('#volS').value / 100, u = $('#volU').value / 100;
      for (const id of ['volM', 'volS', 'volU']) $('#' + id + 'V').textContent = $('#' + id).value + '%';
      s.setLevels(m / 2, e, u);
    };
    for (const id of ['volM', 'volS', 'volU']) $('#' + id).oninput = upd;
    $('[data-close]').onclick = () => { this.closeModal(); this.sfx.play('click'); };
  }

  showFinances() {
    const g = this.game;
    const L = g.ledger;
    const last = g.history.length ? g.history[g.history.length - 1].ledger : null;
    const row = (k, lbl, sign) => `<tr><td>${lbl}</td><td style="color:${sign > 0 ? '#8ad06a' : '#ff9080'}">${fmtMoneyFull(L[k] * sign)}</td><td style="color:#9ab08a">${last ? fmtMoneyFull(last[k] * sign) : '—'}</td></tr>`;
    let html = `<h1>FINANCES</h1><table class="fin"><tr><td></td><td>Today</td><td>Yesterday</td></tr>`;
    html += row('tickets', 'Tickets', 1) + row('shops', 'Shops & food', 1) + row('grants', 'Grants & sales', 1);
    html += row('upkeep', 'Upkeep & wages', -1) + row('food', 'Dino food', -1) + row('repairs', 'Repairs', -1) + row('construction', 'Construction', -1) + row('dinos', 'Hatching', -1) + row('ops', 'Operations', -1) + row('fines', 'Lawsuits & fines', -1);
    html += `</table>`;
    html += `<h2>TICKET PRICE</h2><div class="btns" style="align-items:center"><button data-t="-10">−10</button><button data-t="-1">−1</button><span style="font-size:26px;color:#f8d040;min-width:70px;text-align:center" id="tp">$${g.ticket}</span><button data-t="1">+1</button><button data-t="10">+10</button><span style="color:#9ab08a">Higher prices = fewer visitors.</span></div>`;
    html += `<h2>HISTORY</h2><canvas id="finChart" width="680" height="150" style="width:100%;height:150px;background:#0a120a;border:2px solid #000"></canvas>`;
    html += `<p style="color:#9ab08a">Guests total: ${g.stats.guestsTotal} · Escapes: ${g.stats.escapes} · Guest casualties: ${g.stats.deaths} · Staff lost: ${g.stats.staffDeaths} · Dinos lost: ${g.stats.dinoDeaths}</p>`;
    html += `<div class="btns" style="justify-content:flex-end"><button data-close>Close</button></div>`;
    this.showModal(html);
    for (const b of document.querySelectorAll('[data-t]')) b.onclick = () => { g.ticket = clamp(g.ticket + +b.dataset.t, 0, 400); $('#tp').textContent = '$' + g.ticket; this.sfx.play('click'); };
    $('[data-close]').onclick = () => this.closeModal();
    const c = $('#finChart'), x = c.getContext('2d');
    const H = g.history.slice(-40);
    if (H.length > 1) {
      const line = (vals, col, labelY) => {
        const mn = Math.min(0, ...vals), mx = Math.max(1, ...vals);
        const sy = (v) => 138 - (v - mn) / Math.max(1, mx - mn) * 112;
        x.strokeStyle = col; x.lineWidth = 2; x.beginPath();
        vals.forEach((v, i) => { const px = i / (vals.length - 1) * 670 + 5; if (i) x.lineTo(px, sy(v)); else x.moveTo(px, sy(v)); });
        x.stroke();
        return [mn, mx, sy];
      };
      const [mn, mx, sy] = line(H.map((h) => h.money), '#f8d040');
      x.strokeStyle = '#3a4a36'; x.lineWidth = 1; x.beginPath(); x.moveTo(0, sy(0)); x.lineTo(680, sy(0)); x.stroke();
      const [, gmx] = line(H.map((h) => h.peak || h.guests), '#70b8f0');
      x.font = '15px VT323';
      x.fillStyle = '#f8d040'; x.fillText('Funds ' + fmtMoney(mx) + ' max', 6, 14);
      x.fillStyle = '#70b8f0'; x.fillText('Peak guests ' + gmx, 160, 14);
      x.fillStyle = '#9ab08a'; x.fillText('Day ' + H[0].day, 6, 148); x.fillText('Day ' + H[H.length - 1].day, 630, 148);
    } else { x.fillStyle = '#9ab08a'; x.font = '18px VT323'; x.fillText('History appears after day 1.', 10, 80); }
  }

  showDisasters() {
    const g = this.game;
    let html = `<h1>DISASTERS</h1><p>Disasters strike on their own every few days. You can also unleash them yourself. Chaos theory in action.</p><div class="cards">`;
    for (const [k, d] of Object.entries(DISASTERS)) html += `<div class="card" data-dis="${k}"><div style="font-size:30px">${d.icon}</div><div class="nm">${d.name.toUpperCase()}</div><div style="color:#9ab08a">${d.desc}</div></div>`;
    html += `</div><div class="btns" style="justify-content:space-between;margin-top:10px"><button id="autoDis" class="${g.events.auto ? 'on' : ''}">Random disasters: ${g.events.auto ? 'ON' : 'OFF'}</button><button data-close>Close</button></div>`;
    this.showModal(html);
    const why = { rampage: 'nothing in the park is big enough to go on a rampage', raptors: 'there are no Velociraptors in a paddock', disease: 'there are no dinosaurs to fall ill', storm: 'a storm is already on the way', volcano: 'the volcano is already erupting' };
    for (const c of document.querySelectorAll('[data-dis]')) c.onclick = () => {
      const n = g.logs.length;
      g.events.trigger(c.dataset.dis);
      this.closeModal();
      // a disaster with nothing to act on does nothing: say so rather than leave the player wondering
      if (g.logs.length === n) this.toast(`Nothing happened: ${why[c.dataset.dis] || 'the conditions are not right'}.`, 'info');
    };
    $('#autoDis').onclick = () => { g.events.auto = !g.events.auto; this.showDisasters(); };
    $('[data-close]').onclick = () => this.closeModal();
  }

  showMenu() {
    let has = false;
    try { has = !!localStorage.getItem('jt_save'); } catch (e) { /* storage unavailable */ }
    let html = `<h1>MENU</h1><div class="btns" style="flex-direction:column;align-items:stretch;gap:6px">
      <button data-m="resume">Resume</button><button data-m="save">Save park</button><button data-m="load" ${has ? '' : 'disabled'}>Load saved park</button>
      <button data-m="advisor">Advisor tips: ${this.game.advisorOn === false ? 'OFF' : 'ON'}</button>
      <button data-m="calls" title="Calls from InGen with choices and consequences">InGen calls: ${this.game.decisionsOn === false ? 'OFF' : 'ON'}</button>
      <button data-m="breed">Breeding: ${this.game.breedingOff ? 'PREVENTED (lysine contingency)' : 'ALLOWED'}</button>
      <button data-m="seed" title="Share this code so a friend can play the same island">Island code: ${seedCode(this.game.seed)}${this.game.mapSize !== 'medium' ? ' · ' + MAP_SIZES[this.game.mapSize].label.toLowerCase() : ''} (copy)</button>
      <button data-m="wheel" title="What scrolling does on the map">Scroll: ${{ auto: 'AUTO (mouse wheel zooms, trackpad pans)', zoom: 'ZOOM', pan: 'PAN (pinch or +/− to zoom)' }[this.wheelMode]}</button>
      <button data-m="audio">Sound settings</button>
      <button data-m="help">How to play</button><button data-m="new" class="danger">New island (lose progress)</button></div>`;
    this.showModal(html);
    for (const b of document.querySelectorAll('[data-m]')) b.onclick = () => {
      const m = b.dataset.m;
      if (m === 'resume') this.closeModal();
      else if (m === 'save') { const ok = saveGame(this.game); this.closeModal(); this.toast(ok ? 'Park saved.' : 'Could not save (storage unavailable).', ok ? 'good' : 'bad'); }
      else if (m === 'load') { this.closeModal(); loadGameFromStorage(); }
      else if (m === 'help') this.showHelp();
      else if (m === 'audio') this.showAudio();
      else if (m === 'seed') {
        const code = seedCode(this.game.seed);
        if (navigator.clipboard) navigator.clipboard.writeText(code).then(() => this.toast(`Copied island code ${code}`, 'good'), () => this.toast(`Island code: ${code}`, 'info'));
        else this.toast(`Island code: ${code}`, 'info');
      }
      else if (m === 'calls') { this.game.decisionsOn = this.game.decisionsOn === false; this.showMenu(); }
      else if (m === 'breed') { this.game.breedingOff = !this.game.breedingOff; this.showMenu(); }
      else if (m === 'wheel') {
        this.wheelMode = { auto: 'pan', pan: 'zoom', zoom: 'auto' }[this.wheelMode];
        try { localStorage.setItem('jt_wheel', this.wheelMode); } catch (e) { /* storage unavailable */ }
        this.showMenu();
      }
      else if (m === 'advisor') { this.game.advisorOn = this.game.advisorOn === false; this.showMenu(); }
      else if (m === 'new') { this.closeModal(); showTitle(); }
    };
  }

  showHelp() {
    const html = `<h1>HOW TO PLAY</h1>
      <p>You run a dinosaur theme park on a remote island. Build attractions, hatch dinosaurs, keep guests happy and — above all — <b style="color:#f8d040">keep the dinosaurs contained</b>.</p>
      <h2>BUILDING A PADDOCK</h2>
      <p>1. Build a <b>Power Plant</b> (Power menu). Electric fences only work inside its blue coverage area; extend it with Pylons.<br>
      2. Use the <b>Paddock</b> tool to drag a fenced rectangle. Trees and water inside make dinos happier.<br>
      3. Put a <b>Feeder</b> inside (herbivore or carnivore).<br>
      4. Build a <b>Hatchery</b>, then <b>Hatch</b> a species inside the paddock.</p>
      <h2>GUESTS</h2>
      <p>Guests arrive at the Main Gate and only walk on paths. Lead paths past paddocks (dinos within ~6 tiles are visible) and add Viewing Platforms, food, shops and restrooms. A <b>Tour Station</b> with a loop of <b>Tour Track</b> running past the paddocks is the biggest crowd-pleaser — but the electric jeeps stall when the power goes out.</p>
      <h2>WHEN THINGS GO WRONG</h2>
      <p>Stressed or hungry dinosaurs attack fences. Unpowered fences fall fast. When a dinosaur escapes: sound the <b>ALARM</b> (guests run to the Visitor Center, Hotel or Bunkers), let <b>Rangers</b> tranquilize it, and the <b>ACU Helipad</b> airlifts it home. <b>Engineers</b> from Maintenance Sheds repair fences. Build a <b>Backup Generator</b> for grid failures and a <b>Vet Clinic</b> for outbreaks.</p>
      <p>Escaped predators head for the crowds. Shops and ride queues are no hiding place, and a determined dinosaur can break a shelter's doors (repair it from its panel). <b>Siren Towers</b> sound by themselves when something dangerous is near, so guests start running before you even notice. <b>Raptors</b> stalk, hunt for sport and keep testing electric fences: <b>Concrete Walls</b> hold them. You can <b>close the park</b> from the Main Gate while you deal with a crisis.</p>
      <h2>LIFE FINDS A WAY</h2>
      <p>Hatchlings grow up in a couple of days. Comfortable adults of the same species breed when their paddock has room, and every dinosaur grows old and dies eventually (its panel shows its age): keep hatching young ones, or the exhibit fades away.</p>
      <h2>CONTROLS</h2>
      <p>Left click: use tool / select · Drag: build lines & areas · Right-drag or drag in Inspect: pan · Mouse wheel or +/−: zoom · WASD/Arrows: pan<br>
      Trackpad: two-finger scroll pans, pinch zooms · Touch: one finger builds (or pans in Inspect), two fingers pan and pinch<br>
      Space: pause · 1–4: speed · E: alarm · B: respond to a breach · M: select all rangers · O: power overlay · K: paddock overlay · L: find loose dinos · G: dinosaur roster · Y: research · U or Ctrl+Z: undo · Esc: cancel</p>
      <h2>CAMPAIGN AND INGEN CALLS</h2>
      <p>The <b>Campaign</b> is five chapters, from opening day to Jurassic World; each one you finish gives a perk to the chapters after it. Every few mornings <b>InGen calls</b> with a choice: investors, lawyers, Dr. Wu, the rangers' union and more. Deals show above your goal with their deadlines. Turn calls off in the menu if you just want to build.</p>
      <h2>ATTRACTIONS</h2>
      <p><b>Gyrosphere Station</b> (1★): build it against a paddock fence; glass balls roll guests among the dinosaurs. Put it on a carnivore paddock for thrills, and the occasional cracked sphere. <b>Feeding Grandstand</b>: at noon and 6pm a nearby carnivore is fed in front of the crowd. <b>Raptor Arena</b> (2★): next to a paddock with 2+ raptors, shows pay well but stress the pack, and sometimes go badly wrong. <b>Mosasaur Lagoon</b> (4★): a huge draw; everyone within splash range pays to watch it leap. <b>Lodge Hotel</b>: guests stay the night, and stay longer.</p>
      <h2>SHORTCUTS</h2>
      <p>Click a paddock for one-click <b>Tree cover</b>, <b>feeders</b> and <b>fence repairs</b>. <b>Copy layout</b> turns a paddock into a blueprint you can stamp anywhere. Hatching into a paddock with no food adds a feeder automatically, and <b>Repair all</b> on the breached-fence alert fixes every fence at once.</p>
      <h2>COMMANDING STAFF (OPTIONAL)</h2>
      <p>Rangers, engineers and helicopters work on their own, but you can take charge. Click one to select it, shift-click or shift-drag to select several, double-click to select every one of that kind, or press <b>M</b> for all rangers. Then <b>right-click</b> (on touch: <b>Send…</b>, then tap) a dinosaur to hunt it, a broken fence to repair it, or open ground to hold a position. Units on orders move faster. <b>Back to auto</b> releases them. When something escapes, <b>RESPOND</b> (or <b>B</b>) sounds the alarm and sends every ranger and helicopter at once.</p>
      <h2>PLAY OFFLINE</h2>
      <p>Open <b>mszarski.github.io/claude_playground</b> on your phone and choose <b>Add to Home Screen</b> (Share menu on iPhone, ⋮ menu on Android). After the first visit it runs without a connection.</p>
      <p style="color:#9ab08a">Lose if you stay deeply in debt or too many guests are eaten. Win by completing all goals.</p>
      <div class="btns" style="justify-content:flex-end"><button data-close>Got it</button></div>`;
    this.showModal(html);
    $('[data-close]').onclick = () => this.closeModal();
  }

  // An InGen call: the game is paused (modal open) until the director picks an option
  showDecision(offer) {
    const g = this.game, def = DECISIONS[offer.id];
    this.decisionShown = offer;
    const html = `<div class="call"><div class="callhead"><span class="callicon">${def.icon}</span><div><div class="callfrom">☎ INCOMING CALL</div><h1>${def.from.toUpperCase()}</h1></div></div>
      <p class="calltext">${def.text(g, offer.params)}</p>
      <div class="choices">${def.choices.map((c, i) => `<button class="choice" data-ch="${i}"><b>${typeof c.label === 'function' ? c.label(offer.params) : c.label}</b><span>${choiceHint(c, offer.params)}</span></button>`).join('')}</div></div>`;
    this.showModal(html, true);
    this.sfx.play('ring');
    for (const b of document.querySelectorAll('[data-ch]')) b.onclick = () => {
      g.resolveDecision(+b.dataset.ch);
      this.decisionShown = null;
      this.closeModal();
      this.sfx.play('click');
      this.renderDeals(true);
    };
  }

  renderDeals(force) {
    const list = contractStatus(this.game);
    const html = list.map((c) => `<div class="deal ${c.kind}"><span>${c.kind === 'contract' ? '📋 ' : ''}${c.text}</span><i>${c.left}</i></div>`).join('');
    if (!force && html === this.lastDeals) return;
    this.lastDeals = html;
    const el = $('#deals');
    el.innerHTML = html;
    el.hidden = !list.length;
    document.documentElement.style.setProperty('--dealsH', (el.hidden ? 0 : el.offsetHeight + 4) + 'px');
  }

  showGameOver(reason) {
    const g = this.game;
    const html = `<div id="gameover"><h1>PARK CLOSED</h1><p style="font-size:22px">${reason}</p>
      <p style="color:#9ab08a">Survived ${g.day} days · ${g.stats.guestsTotal} guests · ${g.stats.deaths} eaten · ${g.stats.escapes} escapes</p>
      <p><i>"John, the kind of control you're attempting simply is... it's not possible."</i></p>
      <div class="btns" style="justify-content:center"><button data-m="new">Try a new island</button></div></div>`;
    this.showModal(html, true);
    $('[data-m="new"]').onclick = () => { this.closeModal(); showTitle(); };
  }

  showChapterIntro(id) {
    const c = SCENARIOS[id], perks = campaignPerks(id);
    const html = `<div id="gameover"><div style="font-family:'Press Start 2P';font-size:9px;color:var(--orange)">CHAPTER ${c.chapter} OF ${CAMPAIGN_ORDER.length}</div><h1>${c.name.toUpperCase()}</h1>
      <p style="font-size:21px">${c.blurb}</p><p style="color:var(--gold);font-size:20px">${c.objective}.</p>
      ${perks.length ? `<p style="color:#9ab08a">Your perks: ${perks.map((p) => `<b style="color:#a8f080">${p.name}</b> (${p.desc})`).join(', ')}</p>` : ''}
      <div class="btns" style="justify-content:center"><button data-close>Let's go</button></div></div>`;
    this.showModal(html);
    $('[data-close]').onclick = () => this.closeModal();
  }

  showScenarioEnd(r) {
    const g = this.game, sc = SCENARIOS[g.scenario];
    this.setSpeed(0);
    const win = r === 'win';
    if (win) this.sfx.play('fanfare');
    const medal = win && sc.medal ? sc.medal(g) : null;
    const best = medal ? saveMedal(g.scenario, medal, g.day) : false;
    if (sc.campaign) {
      if (win) saveCampaign(g.scenario, medal || 'bronze', g.day);
      const next = CAMPAIGN_ORDER[CAMPAIGN_ORDER.indexOf(g.scenario) + 1];
      const perk = win && CAMPAIGN_PERKS[g.scenario];
      const html = `<div id="gameover"><div style="font-family:'Press Start 2P';font-size:9px;color:var(--orange)">CHAPTER ${sc.chapter} OF ${CAMPAIGN_ORDER.length}</div><h1>${win ? (next ? 'CHAPTER COMPLETE' : 'CAMPAIGN COMPLETE') : 'CHAPTER FAILED'}</h1>
        <p style="font-size:22px">${sc.name}: ${win ? (next ? 'on to the next island.' : 'the greatest park on Earth. Welcome to Jurassic World.') : 'the island won this time.'}</p>
        ${medal ? `<p style="font-size:44px;margin:4px 0">${MEDAL_ICON[medal]}</p><p style="font-family:'Press Start 2P';font-size:10px;color:var(--gold)">${medal.toUpperCase()} MEDAL${best ? ' · NEW BEST' : ''}</p>` : ''}
        ${sc.medalText ? `<p style="color:#9ab08a">${sc.medalText}</p>` : ''}
        ${perk ? `<p>New perk for the chapters ahead: <b style="color:#a8f080">${perk.name}</b> (${perk.desc})</p>` : ''}
        <p style="color:#9ab08a">Day ${g.day} · ${g.stats.deaths} guests lost · ${g.stats.escapes} escapes · ${fmtMoney(g.money)}</p>
        <div class="btns" style="justify-content:center">${win && next ? '<button data-m="next">Next chapter</button>' : ''}<button data-m="again">${win ? 'Replay' : 'Try again'}</button><button data-m="title">Main menu</button><button data-close>Keep playing</button></div></div>`;
      this.showModal(html, true);
      const nb = $('[data-m="next"]'); if (nb) nb.onclick = () => { this.closeModal(); startChapter(next); };
      $('[data-m="again"]').onclick = () => { this.closeModal(); startChapter(g.scenario); };
      $('[data-m="title"]').onclick = () => { this.closeModal(); showTitle(); };
      $('[data-close]').onclick = () => { this.closeModal(); this.setSpeed(1); };
      return;
    }
    const html = `<div id="gameover"><h1>${win ? 'SCENARIO COMPLETE' : 'SCENARIO FAILED'}</h1><p style="font-size:22px">${sc.name}: ${win ? 'you did it.' : 'the island won this time.'}</p>
      ${medal ? `<p style="font-size:44px;margin:4px 0">${MEDAL_ICON[medal]}</p><p style="font-family:'Press Start 2P';font-size:10px;color:var(--gold)">${medal.toUpperCase()} MEDAL${best ? ' · NEW BEST' : ''}</p>` : ''}
      ${sc.medalText ? `<p style="color:#9ab08a">${sc.medalText}</p>` : ''}
      <p style="color:#9ab08a">Day ${g.day} · ${g.stats.deaths} guests lost · ${g.stats.escapes} escapes · ${fmtMoney(g.money)}</p>
      <div class="btns" style="justify-content:center"><button data-m="again">Try again</button><button data-m="title">Main menu</button><button data-close>Keep playing</button></div></div>`;
    this.showModal(html, true);
    $('[data-m="again"]').onclick = () => { this.closeModal(); startGame({ scenario: g.scenario, difficulty: 'normal' }); };
    $('[data-m="title"]').onclick = () => { this.closeModal(); showTitle(); };
    $('[data-close]').onclick = () => { this.closeModal(); this.setSpeed(1); };
  }

  showVictory() {
    const g = this.game;
    const html = `<div id="gameover"><h1>WELCOME TO JURASSIC TYCOON</h1><p style="font-size:22px">Every goal complete in ${g.day} days. You spared no expense — and kept (most of) the dinosaurs in their paddocks.</p>
      <p style="color:#9ab08a">${g.stats.guestsTotal} guests · ${g.stats.deaths} eaten · ${g.stats.escapes} escapes</p>
      <div class="btns" style="justify-content:center"><button data-close>Keep playing</button></div></div>`;
    this.showModal(html);
    $('[data-close]').onclick = () => this.closeModal();
  }
}

for (const k of Object.getOwnPropertyNames(UIPanels.prototype)) if (k !== 'constructor') UI.prototype[k] = UIPanels.prototype[k];
