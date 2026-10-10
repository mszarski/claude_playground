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
      const sick = d.sick > 0 ? '<span class="tag" style="background:#5a8a2a">SICK</span>' : '';
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
      html += `<div class="sub">${o.kind === 'ranger' ? (o.target ? `Pursuing ${o.target.name} the ${o.target.sp.name}` : 'On patrol') : (o.state === 'repair' ? 'Repairing' : o.state === 'move' ? 'En route to a job' : 'Idle')}</div>`;
      html += `<div class="sub">${o.kind === 'ranger' ? 'Tranquilizes loose dinosaurs within 4 tiles.' : 'Repairs fences & buildings, rebuilds breached fences.'}</div>`;
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
      html += `<div class="sub">${ds.length ? ds.map((d) => d.name + ' (' + d.sp.name + ')').join(', ') : 'Empty. Use Hatch to add dinosaurs.'}</div>`;
    } else if (o.type) {
      const def = BUILDINGS[o.type];
      html += `<h2>${def.name}</h2><div class="sub">${def.desc}</div>`;
      html += `<div class="row"><span>Condition</span><span>${Math.round(o.hp)}/${o.maxHp}</span></div>${this.bar(o.hp, o.maxHp, '#5ac85a')}`;
      if (def.power) html += `<div class="row"><span>Power</span><span style="color:${o.powered ? '#f8d040' : '#e04838'}">${o.powered ? 'ON' : 'NO POWER'}</span></div>`;
      if (o.type === 'power') html += `<div class="row"><span>Output</span><span>${o.offline > 0 || w.power.outage > 0 ? 'OFFLINE' : def.supply + ' MW'}</span></div>`;
      if (def.cat === 'guest' && !w.accessTiles(o).length) html += `<div class="sub" style="color:#ff9080">Not connected to a path!</div>`;
      if (o.type === 'tour') {
        html += `<div class="row"><span>Queue</span><span>${(o.queue || []).length}</span></div><div class="row"><span>Jeeps</span><span>${g.jeeps.filter((j) => j.station === o).length}</span></div>`;
        if (!w.trackAccess(o).length) html += `<div class="sub" style="color:#ff9080">Needs Tour Track next to it!</div>`;
      }
      if (def.income) html += `<div class="row"><span>Visitors</span><span>${o.visitors}</span></div><div class="row"><span>Revenue</span><span>${fmtMoney(o.revenue || 0)}</span></div>`;
      if (def.shelter) html += `<div class="row"><span>Sheltering</span><span>${o.inside}/${def.shelter}</span></div>`;
      if (def.upkeep) html += `<div class="row"><span>Upkeep</span><span>${fmtMoney(def.upkeep)}/day</span></div>`;
      html += `<div class="btns"><button data-a="demolish" class="danger">Demolish (+${fmtMoney(def.cost * 0.25)})</button></div>`;
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
    if (a === 'sedate' && o) { o.orderSedate = true; g.log(`Rangers dispatched to sedate ${o.name}.`, 'info', o); }
    if (a === 'strike' && o) { const err = g.acuStrike(o); if (err) this.toast(err, 'warn'); }
    if (a === 'erepair' && o) { const err = g.emergencyRepair(o.x, o.y); if (err) this.toast(err, 'warn'); else this.sfx.play('build'); }
    if (a === 'relocate' && o) { this.moveDino = o; this.setTool('movedest'); this.toast(`Click inside the paddock where ${o.name} should go (Esc to cancel).`, 'info'); }
    if (a === 'sell' && o) {
      if (o.carried) return;
      g.earn(o.sellValue || o.sp.cost * 0.35, 'grants'); o.dead = true;
      g.log(`${o.name} the ${o.sp.name} was shipped to another facility.`, 'info');
      this.select(null); return;
    }
    if (a === 'demolish' && o) { if (g.demolishAt(o.x, o.y)) this.sfx.play('demolish'); this.select(null); return; }
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
    let html = `<h1>HATCHERY</h1><p>Choose a species, then click inside a fenced paddock (no paths inside!). Carnivores will eat herbivores that share their paddock. Pteranodons hatch inside an Aviary.</p>`;
    if (!g.hasBuilding('hatchery')) html += `<p style="color:#ff9080">You need to build a Hatchery first (Dinos menu).</p>`;
    html += `<div class="cards">`;
    for (const k of SPECIES_ORDER) {
      const sp = SPECIES[k];
      const locked = !g.isUnlocked(k);
      html += `<div class="card ${locked ? 'locked' : ''}" data-sp="${k}"><canvas data-spr="${k}"></canvas><div class="nm">${sp.name.toUpperCase()}</div>
        <div>${sp.diet === 'carn' ? '<span style="color:#f08a5a">Carnivore</span>' : '<span style="color:#8ad06a">Herbivore</span>'} · ${fmtMoney(sp.cost)}</div>
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
    for (const c of document.querySelectorAll('[data-dis]')) c.onclick = () => { g.events.trigger(c.dataset.dis); this.closeModal(); };
    $('#autoDis').onclick = () => { g.events.auto = !g.events.auto; this.showDisasters(); };
    $('[data-close]').onclick = () => this.closeModal();
  }

  showMenu() {
    let has = false;
    try { has = !!localStorage.getItem('jt_save'); } catch (e) { /* storage unavailable */ }
    let html = `<h1>MENU</h1><div class="btns" style="flex-direction:column;align-items:stretch;gap:6px">
      <button data-m="resume">Resume</button><button data-m="save">Save park</button><button data-m="load" ${has ? '' : 'disabled'}>Load saved park</button>
      <button data-m="advisor">Advisor tips: ${this.game.advisorOn === false ? 'OFF' : 'ON'}</button>
      <button data-m="breed">Breeding: ${this.game.breedingOff ? 'PREVENTED (lysine contingency)' : 'ALLOWED'}</button>
      <button data-m="help">How to play</button><button data-m="new" class="danger">New island (lose progress)</button></div>`;
    this.showModal(html);
    for (const b of document.querySelectorAll('[data-m]')) b.onclick = () => {
      const m = b.dataset.m;
      if (m === 'resume') this.closeModal();
      else if (m === 'save') { const ok = saveGame(this.game); this.closeModal(); this.toast(ok ? 'Park saved.' : 'Could not save (storage unavailable).', ok ? 'good' : 'bad'); }
      else if (m === 'load') { this.closeModal(); loadGameFromStorage(); }
      else if (m === 'help') this.showHelp();
      else if (m === 'breed') { this.game.breedingOff = !this.game.breedingOff; this.showMenu(); }
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
      <h2>CONTROLS</h2>
      <p>Left click: use tool / select · Drag: build lines & areas · Right-drag or drag in Inspect: pan · Wheel: zoom · WASD/Arrows: pan<br>
      Space: pause · 1/2/3: speed · E: alarm · O: power overlay · K: paddock overlay · L: find loose dinos · Esc: cancel</p>
      <p style="color:#9ab08a">Lose if you stay deeply in debt or too many guests are eaten. Win by completing all goals.</p>
      <div class="btns" style="justify-content:flex-end"><button data-close>Got it</button></div>`;
    this.showModal(html);
    $('[data-close]').onclick = () => this.closeModal();
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

  showScenarioEnd(r) {
    const g = this.game, sc = SCENARIOS[g.scenario];
    this.setSpeed(0);
    const win = r === 'win';
    if (win) this.sfx.play('fanfare');
    const html = `<div id="gameover"><h1>${win ? 'SCENARIO COMPLETE' : 'SCENARIO FAILED'}</h1><p style="font-size:22px">${sc.name}: ${win ? 'you did it.' : 'the island won this time.'}</p>
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
