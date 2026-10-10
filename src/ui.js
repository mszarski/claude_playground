// ---------- UI: input, panels, HUD ----------
'use strict';

const $ = (s) => document.querySelector(s);
const LINE_TOOLS = ['path', 'track', 'fence', 'wall'];
const RECT_TOOLS = ['demolish', 'trees', 'clear'];

class UI {
  constructor(game, renderer, sfx) {
    this.game = game; this.r = renderer; this.sfx = sfx;
    this.tool = 'inspect'; this.species = null;
    this.hover = null; this.drag = null; this.dragTiles = null;
    this.selected = null; this.follow = false;
    this.speed = 1; this.prevSpeed = 1;
    this.keys = {};
    this.openCat = null;
    this.mouse = { x: 0, y: 0, down: false, button: 0 };
    this.infoT = 0; this.hudT = 0;
    this.buildToolbar();
    this.bindInput();
    this.bindButtons();
    this.attachGame(game);
  }

  attachGame(game) {
    this.game = game;
    this.selected = null;
    $('#log').innerHTML = '';
    game.on('log', (e) => this.onLog(e));
    game.on('gameover', (reason) => this.showGameOver(reason));
    game.on('victory', () => this.showVictory());
    game.on('scenario', (r) => this.showScenarioEnd(r));
    game.on('breach', () => { if (this.speed > 1) this.setSpeed(1); });
    game.on('day', () => {
      const h = game.history[game.history.length - 1];
      if (!h) return;
      const net = h.income - h.expense;
      this.toast(`Day ${h.day} report: ${net >= 0 ? '+' : '−'}${fmtMoney(Math.abs(net)).replace('-', '')} net · ${h.peak} guests at peak · ${'★'.repeat(game.stars) || 'no stars yet'}`, net >= 0 ? 'good' : 'warn');
    });
    game.soundHook = (name, vol, x, y) => {
      if (x !== undefined) {
        // attenuate by distance from camera center
        const z = this.r.cam.zoom;
        const cx = (this.r.cam.x + this.r.canvas.width / z / 2) / TILE, cy = (this.r.cam.y + this.r.canvas.height / z / 2) / TILE;
        const d = dist(cx, cy, x, y);
        const reach = this.r.canvas.width / z / TILE * 0.8;
        if (d > reach) return;
        vol *= 1 - (d / reach) * 0.8;
      }
      this.sfx.play(name, vol);
    };
    this.renderGoal();
  }

  // ---------------- toolbar ----------------
  buildToolbar() {
    const tb = $('#toolbar');
    tb.innerHTML = '';
    const catIcons = { inspect: 'inspect', build: 'path', guest: 'restaurant', infra: 'power', staff: 'ranger', dino: 'hatch' };
    for (const grp of TOOL_GROUPS) {
      const b = document.createElement('button');
      b.className = 'cat'; b.dataset.cat = grp.id;
      const c = document.createElement('canvas'); c.width = 24; c.height = 24;
      c.getContext('2d').drawImage(ICONS[catIcons[grp.id]], 0, 0);
      b.appendChild(c);
      const l = document.createElement('span'); l.textContent = grp.label; b.appendChild(l);
      b.onclick = () => {
        this.sfx.play('click');
        if (grp.id === 'inspect') { this.setTool('inspect'); this.closeFlyout(); return; }
        if (this.openCat === grp.id) this.closeFlyout(); else this.openFlyout(grp);
      };
      tb.appendChild(b);
    }
  }

  openFlyout(grp) {
    this.openCat = grp.id;
    const f = $('#flyout');
    f.innerHTML = `<h3>${grp.label.toUpperCase()}</h3>`;
    for (const t of grp.tools) {
      const b = document.createElement('button');
      b.className = 'tool' + (this.tool === t ? ' on' : '');
      const c = document.createElement('canvas'); c.width = 24; c.height = 24;
      c.getContext('2d').drawImage(ICONS[t] || ICONS.inspect, 0, 0);
      b.appendChild(c);
      const info = BUILDINGS[t] || TOOL_INFO[t];
      const cost = BUILDINGS[t] ? BUILDINGS[t].cost : (TOOL_INFO[t].cost || 0);
      const nm = document.createElement('span'); nm.className = 'nm'; nm.innerHTML = `${info.name}${TOOL_INFO[t] && TOOL_INFO[t].key ? ` <span class="k">[${TOOL_INFO[t].key}]</span>` : ''}`;
      b.appendChild(nm);
      if (BUILDINGS[t] && BUILDINGS[t].unlockStars && this.game.stars < BUILDINGS[t].unlockStars && !this.game.sandbox) { b.style.opacity = 0.5; nm.innerHTML += ` <span class="k">${'★'.repeat(BUILDINGS[t].unlockStars)}</span>`; }
      if (cost) { const cs = document.createElement('span'); cs.className = 'c'; cs.textContent = fmtMoney(cost) + (LINE_TOOLS.includes(t) || RECT_TOOLS.includes(t) || t === 'paddock' ? '/t' : ''); b.appendChild(cs); }
      b.onmouseenter = (e) => this.showTip(e, `<b>${info.name}</b><br>${info.desc}` + (BUILDINGS[t] && BUILDINGS[t].upkeep ? `<br><span style="color:#9ab08a">Upkeep ${fmtMoney(BUILDINGS[t].upkeep)}/day${BUILDINGS[t].power ? ' · ' + BUILDINGS[t].power + ' MW' : ''}</span>` : ''));
      b.onmouseleave = () => this.hideTip();
      b.onclick = () => {
        this.sfx.play('click');
        if (t === 'hatch') { this.showSpeciesPicker(); return; }
        this.setTool(t);
        if (window.innerWidth < 760) this.closeFlyout(); else this.openFlyout(grp);
      };
      f.appendChild(b);
    }
    f.style.display = 'block';
    for (const c of document.querySelectorAll('.cat')) c.classList.toggle('on', c.dataset.cat === grp.id);
  }
  closeFlyout() {
    this.openCat = null;
    $('#flyout').style.display = 'none';
    for (const c of document.querySelectorAll('.cat')) c.classList.toggle('on', c.dataset.cat === 'inspect' && this.tool === 'inspect');
    this.hideTip();
  }

  setTool(t) {
    this.tool = t;
    this.drag = null; this.dragTiles = null;
    this.r.canvas.style.cursor = t === 'inspect' ? 'default' : 'crosshair';
    if (t === 'inspect') for (const c of document.querySelectorAll('.cat')) c.classList.toggle('on', c.dataset.cat === 'inspect');
  }

  // ---------------- input ----------------
  screenToTile(sx, sy) {
    const z = this.r.cam.zoom;
    const wx = this.r.cam.x + sx / z, wy = this.r.cam.y + sy / z;
    return { x: Math.floor(wx / TILE), y: Math.floor(wy / TILE), wx: wx / TILE, wy: wy / TILE };
  }

  bindInput() {
    const cv = this.r.canvas;
    cv.addEventListener('contextmenu', (e) => e.preventDefault());
    cv.addEventListener('mousedown', (e) => {
      this.sfx.ensure();
      this.mouse.down = true; this.mouse.button = e.button;
      this.mouse.sx = e.clientX; this.mouse.sy = e.clientY; this.mouse.moved = 0;
      this.mouse.camX = this.r.cam.x; this.mouse.camY = this.r.cam.y;
      if (e.button !== 0) { this.panning = true; return; }
      const t = this.screenToTile(e.clientX, e.clientY);
      if (LINE_TOOLS.includes(this.tool) || RECT_TOOLS.includes(this.tool) || this.tool === 'paddock') {
        this.drag = { x: t.x, y: t.y }; this.updateDrag(t);
      }
    });
    window.addEventListener('mousemove', (e) => {
      this.mouse.x = e.clientX; this.mouse.y = e.clientY;
      if (e.target !== cv && !this.mouse.down) { this.hover = null; return; }
      const t = this.screenToTile(e.clientX, e.clientY);
      this.hover = t;
      if (this.mouse.down) {
        this.mouse.moved += Math.abs(e.movementX) + Math.abs(e.movementY);
        if (this.panning || (this.tool === 'inspect' && this.mouse.moved > 6) || (this.tool === 'hatch' && this.mouse.moved > 6)) {
          this.panning = true;
          const z = this.r.cam.zoom;
          this.r.cam.x = this.mouse.camX - (e.clientX - this.mouse.sx) / z;
          this.r.cam.y = this.mouse.camY - (e.clientY - this.mouse.sy) / z;
          this.follow = false;
          this.clampCam();
        } else if (this.drag) this.updateDrag(t);
      }
      if (this.drag && this.dragTiles) {
        const cost = this.dragCost();
        this.showTip(e, `${this.dragTiles.length} tiles${cost ? ' · ' + fmtMoney(cost) : ''}`);
      } else if (e.target === cv && BUILDINGS[this.tool]) {
        this.showTip(e, `${BUILDINGS[this.tool].name} · ${fmtMoney(BUILDINGS[this.tool].cost)}`);
      } else if (e.target === cv && this.tool === 'hatch' && this.species) {
        const err = this.game.canHatchAt(this.species, t.x, t.y);
        this.showTip(e, err ? `<span style="color:#ff9080">${err}</span>` : `Hatch ${SPECIES[this.species].name} · ${fmtMoney(SPECIES[this.species].cost)}`);
      } else if (e.target === cv) this.hideTip();
    });
    window.addEventListener('mouseup', (e) => {
      if (!this.mouse.down) return;
      this.mouse.down = false;
      const wasPan = this.panning; this.panning = false;
      if (e.target !== cv && !this.drag) return;
      if (wasPan && this.mouse.button !== 0) return;
      if (this.mouse.button !== 0) return;
      const t = this.screenToTile(e.clientX, e.clientY);
      if (this.drag) { this.applyDrag(); this.drag = null; this.dragTiles = null; this.hideTip(); return; }
      if (wasPan) return;
      this.click(t);
    });
    cv.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.zoomAt(e.deltaY < 0 ? 1 : -1, e.clientX, e.clientY);
    }, { passive: false });

    // touch: one finger = tool / pan, two = pinch zoom
    let pinch = null;
    cv.addEventListener('touchstart', (e) => {
      e.preventDefault();
      this.sfx.ensure();
      if (e.touches.length === 2) {
        const [a, b] = e.touches;
        pinch = { d: dist(a.clientX, a.clientY, b.clientX, b.clientY), z: this.r.cam.zoom };
        this.drag = null; this.dragTiles = null; this.mouse.down = false;
        return;
      }
      const tt = e.touches[0];
      this.mouse.down = true; this.mouse.button = 0; this.mouse.sx = tt.clientX; this.mouse.sy = tt.clientY; this.mouse.moved = 0;
      this.mouse.camX = this.r.cam.x; this.mouse.camY = this.r.cam.y;
      const t = this.screenToTile(tt.clientX, tt.clientY);
      this.hover = t;
      if (LINE_TOOLS.includes(this.tool) || RECT_TOOLS.includes(this.tool) || this.tool === 'paddock') { this.drag = { x: t.x, y: t.y }; this.updateDrag(t); }
    }, { passive: false });
    cv.addEventListener('touchmove', (e) => {
      e.preventDefault();
      if (pinch && e.touches.length === 2) {
        const [a, b] = e.touches;
        const d = dist(a.clientX, a.clientY, b.clientX, b.clientY);
        const nz = clamp(Math.round(pinch.z * d / pinch.d), 1, 6);
        if (nz !== this.r.cam.zoom) this.zoomAt(nz - this.r.cam.zoom, (a.clientX + b.clientX) / 2, (a.clientY + b.clientY) / 2);
        return;
      }
      const tt = e.touches[0];
      const t = this.screenToTile(tt.clientX, tt.clientY);
      this.hover = t;
      this.mouse.moved += 4;
      if (this.drag) { this.updateDrag(t); return; }
      if (this.mouse.moved > 8) {
        const z = this.r.cam.zoom;
        this.r.cam.x = this.mouse.camX - (tt.clientX - this.mouse.sx) / z;
        this.r.cam.y = this.mouse.camY - (tt.clientY - this.mouse.sy) / z;
        this.clampCam(); this.panning = true;
      }
    }, { passive: false });
    cv.addEventListener('touchend', (e) => {
      e.preventDefault();
      if (pinch) { if (e.touches.length < 2) pinch = null; return; }
      if (!this.mouse.down) return;
      this.mouse.down = false;
      if (this.drag) { this.applyDrag(); this.drag = null; this.dragTiles = null; return; }
      if (!this.panning && this.hover) this.click(this.hover);
      this.panning = false;
    }, { passive: false });

    window.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT') return;
      this.keys[e.key.toLowerCase()] = true;
      const k = e.key.toLowerCase();
      if (k === ' ') { e.preventDefault(); this.setSpeed(this.speed === 0 ? (this.prevSpeed || 1) : 0); }
      else if (k === '1') this.setSpeed(1);
      else if (k === '2') this.setSpeed(2);
      else if (k === '3') this.setSpeed(4);
      else if (k === '4') this.setSpeed(8);
      else if (k === 'escape') { if ($('#modal').style.display === 'flex') this.closeModal(); else if (this.tool !== 'inspect') { this.setTool('inspect'); this.closeFlyout(); } else if (this.selected) this.select(null); else this.showMenu(); }
      else if (k === 'e') this.toggleAlarm();
      else if (k === 'o') this.toggleOverlay('power');
      else if (k === 'k') this.toggleOverlay('paddock');
      else if (k === 'l') this.jumpToLoose();
      else if (k === 'g') this.showRoster();
      else if (k === '=' || k === '+') this.zoomAt(1, window.innerWidth / 2, window.innerHeight / 2);
      else if (k === '-') this.zoomAt(-1, window.innerWidth / 2, window.innerHeight / 2);
      else {
        for (const [t, info] of Object.entries(TOOL_INFO)) {
          if (info.key && info.key.toLowerCase() === k) {
            if (t === 'hatch') this.showSpeciesPicker(); else this.setTool(t);
            return;
          }
        }
      }
    });
    window.addEventListener('keyup', (e) => { this.keys[e.key.toLowerCase()] = false; });
  }

  zoomAt(dir, sx, sy) {
    const cam = this.r.cam;
    const nz = clamp(cam.zoom + dir, 1, 6);
    if (nz === cam.zoom) return;
    const wx = cam.x + sx / cam.zoom, wy = cam.y + sy / cam.zoom;
    cam.zoom = nz;
    cam.x = wx - sx / nz; cam.y = wy - sy / nz;
    this.clampCam();
  }

  clampCam() {
    const cam = this.r.cam, w = this.game.world;
    const vw = this.r.canvas.width / cam.zoom, vh = this.r.canvas.height / cam.zoom;
    const W = w.W * TILE, H = w.H * TILE;
    cam.x = vw >= W ? (W - vw) / 2 : clamp(cam.x, -40, W - vw + 40);
    cam.y = vh >= H ? (H - vh) / 2 : clamp(cam.y, -60, H - vh + 40);
  }

  centerOn(x, y) {
    const cam = this.r.cam;
    cam.x = x * TILE - this.r.canvas.width / cam.zoom / 2;
    cam.y = y * TILE - this.r.canvas.height / cam.zoom / 2;
    this.clampCam();
  }

  updateDrag(t) {
    const d = this.drag;
    const tiles = [];
    if (LINE_TOOLS.includes(this.tool)) {
      // L-shaped: along x first then y
      const sx = Math.sign(t.x - d.x) || 1, sy = Math.sign(t.y - d.y) || 1;
      if (Math.abs(t.x - d.x) >= Math.abs(t.y - d.y)) {
        for (let x = d.x; x !== t.x + sx; x += sx) tiles.push([x, d.y]);
        for (let y = d.y + sy; y !== t.y + sy; y += sy) if (t.y !== d.y) tiles.push([t.x, y]);
      } else {
        for (let y = d.y; y !== t.y + sy; y += sy) tiles.push([d.x, y]);
        for (let x = d.x + sx; x !== t.x + sx; x += sx) if (t.x !== d.x) tiles.push([x, t.y]);
      }
    } else if (this.tool === 'paddock') {
      const x0 = Math.min(d.x, t.x), x1 = Math.max(d.x, t.x), y0 = Math.min(d.y, t.y), y1 = Math.max(d.y, t.y);
      for (let x = x0; x <= x1; x++) { tiles.push([x, y0]); if (y1 !== y0) tiles.push([x, y1]); }
      for (let y = y0 + 1; y < y1; y++) { tiles.push([x0, y]); if (x1 !== x0) tiles.push([x1, y]); }
    } else {
      const x0 = Math.min(d.x, t.x), x1 = Math.max(d.x, t.x), y0 = Math.min(d.y, t.y), y1 = Math.max(d.y, t.y);
      for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) tiles.push([x, y]);
    }
    this.dragTiles = tiles.filter(([x, y]) => this.game.world.inb(x, y));
  }

  dragCost() {
    const tool = this.tool, w = this.game.world;
    let c = 0;
    for (const [x, y] of this.dragTiles) {
      const i = w.idx(x, y);
      if (tool === 'path' && !w.path[i] && !w.track[i] && w.canBuildAt(x, y)) c += TOOL_INFO.path.cost;
      else if (tool === 'track' && !w.path[i] && !w.track[i] && w.canBuildAt(x, y)) c += TOOL_INFO.track.cost;
      else if ((tool === 'fence' || tool === 'paddock') && w.fence[i] !== F_ELECTRIC && w.canBuildAt(x, y) && !w.path[i]) c += FENCE_DEF[F_ELECTRIC].cost;
      else if (tool === 'wall' && w.fence[i] !== F_WALL && w.canBuildAt(x, y) && !w.path[i]) c += FENCE_DEF[F_WALL].cost;
      else if (tool === 'trees' && w.terrain[i] === T_GRASS && !w.path[i] && !w.bld[i] && !w.fence[i]) c += TOOL_INFO.trees.cost;
      else if (tool === 'clear' && w.terrain[i] === T_FOREST) c += TOOL_INFO.clear.cost;
    }
    return c;
  }

  applyDrag() {
    const g = this.game, tool = this.tool;
    if (!this.dragTiles) return;
    let n = 0;
    for (const [x, y] of this.dragTiles) {
      let ok = false;
      if (tool === 'path') ok = g.placePath(x, y);
      else if (tool === 'track') ok = g.placeTrack(x, y);
      else if (tool === 'fence' || tool === 'paddock') ok = g.placeFence(x, y, F_ELECTRIC);
      else if (tool === 'wall') ok = g.placeFence(x, y, F_WALL);
      else if (tool === 'demolish') {
        // demolish only fences/paths in drag, buildings only on single tile
        const w = g.world, i = w.idx(x, y);
        if (w.bld[i] && this.dragTiles.length > 1) continue;
        ok = g.demolishAt(x, y);
      }
      else if (tool === 'trees') ok = g.plantTree(x, y);
      else if (tool === 'clear') ok = g.clearLand(x, y);
      if (ok) n++;
    }
    if (n) this.sfx.play(tool === 'demolish' ? 'demolish' : 'build');
    else this.sfx.play('error');
    if (n && (tool === 'fence' || tool === 'paddock')) {
      const w = g.world;
      w.computePower(g);
      let off = 0, total = 0;
      for (const [x, y] of this.dragTiles) { const i = w.idx(x, y); if (w.fence[i] === F_ELECTRIC) { total++; if (!w.fencePowered[i]) off++; } }
      if (off) this.toast(`${off} of ${total} fence tiles have no power (blinking red). Build a Power Plant or Pylon nearby.`, 'warn');
    }
  }

  click(t) {
    const g = this.game, w = g.world;
    const tool = this.tool;
    if (BUILDINGS[tool]) {
      const def = BUILDINGS[tool];
      const bx = t.x - Math.floor((def.w - 1) / 2), by = t.y - Math.floor((def.h - 1) / 2);
      const b = g.placeBuilding(tool, bx, by);
      if (b) {
        this.hideTip();
        this.sfx.play('build');
        if (def.cat === 'guest' && !w.accessTiles(b).length) this.toast(`Connect the ${def.name} to a path!`, 'warn');
        if (tool === 'tour' && !w.trackAccess(b).length) this.toast('Now lay Tour Track from the station past your paddocks, ideally in a loop.', 'info');
        if (def.feeds) { const reg = w.regionAt(bx + 1, by) || w.regionAt(bx - 1, by) || w.regionAt(bx, by + 1) || w.regionAt(bx, by - 1); if (reg && reg.public) this.toast('Feeders should go inside a fenced paddock.', 'warn'); }
      } else {
        this.sfx.play('error');
        if (def.unlockStars && g.stars < def.unlockStars && !g.sandbox) this.toast(`The ${def.name} unlocks at ${def.unlockStars} stars.`, 'warn');
        else if (!g.canAfford(def.cost)) this.toast(`Not enough money for a ${def.name}.`, 'warn');
        else if (def.unique && g.hasBuilding(tool) && tool !== 'gate') this.toast(`You can only have one ${def.name}.`, 'warn');
        else this.toast(`Can't build there — needs ${def.w}×${def.h} clear land (no water, rock, paths or fences).`, 'warn');
      }
      return;
    }
    if (tool === 'hatch') {
      if (!this.species) { this.showSpeciesPicker(); return; }
      const err = g.canHatchAt(this.species, t.x, t.y);
      if (err) { this.toast(err, 'warn'); this.sfx.play('error'); }
      else if (g.hatch(this.species, t.x, t.y)) this.sfx.play('build');
      return;
    }
    if (tool === 'demolish') { if (g.demolishAt(t.x, t.y)) this.sfx.play('demolish'); return; }
    if (tool === 'movedest') {
      const d = this.moveDino, w2 = g.world;
      const reg = w2.regionAt(t.x, t.y);
      if (!d || d.dead) { this.setTool('inspect'); return; }
      if (!reg || reg.public || reg.size > 1200) { this.toast('Pick an open tile inside a fenced paddock.', 'warn'); this.sfx.play('error'); return; }
      d.relocateTo = { x: t.x, y: t.y }; d.relocate = true;
      if (d.sedatedT <= 0) d.orderSedate = true;
      g.log(`${d.name} the ${d.sp.name} will be sedated and airlifted to the new paddock.`, 'info', d, true);
      this.sfx.play('click'); this.moveDino = null; this.setTool('inspect');
      return;
    }
    // Inspect: pick entity under cursor
    const pick = this.pickAt(t.wx, t.wy);
    this.select(pick);
    this.sfx.play('click');
  }

  pickAt(wx, wy) {
    const g = this.game, w = g.world;
    const px = wx * TILE, py = wy * TILE;
    let best = null, by = -1e9;
    for (const d of g.dinos) {
      if (d.carried) continue;
      const S = DINO_SPRITES[d.species];
      const x0 = d.x * TILE - S.w / 2, y0 = d.y * TILE - S.h + 5;
      if (px >= x0 && px <= x0 + S.w && py >= y0 && py <= y0 + S.h + 2 && d.y > by) { best = d; by = d.y; }
    }
    for (const p of g.pteros) {
      const ay = p.sedatedT > 0 ? 0 : p.alt * TILE;
      if (Math.abs(px - p.x * TILE) < 10 && py > p.y * TILE - ay - 14 && py < p.y * TILE - ay + 4) return p;
    }
    if (best) return best;
    for (const h of g.humans()) {
      const x0 = h.x * TILE - 4, y0 = h.y * TILE - 10;
      if (px >= x0 && px <= x0 + 8 && py >= y0 && py <= y0 + 12 && h.y > by) { best = h; by = h.y; }
    }
    if (best) return best;
    const tx = Math.floor(wx), ty = Math.floor(wy);
    if (!w.inb(tx, ty)) return null;
    // buildings (sprites extend upward; check tile & a couple of tiles below)
    for (let dy = 0; dy <= 2; dy++) {
      const b = w.buildingAt(tx, ty + dy);
      if (b) {
        const spr = BSPR[b.type];
        const top = (b.y + b.h) * TILE - spr.canvas.height;
        if (py >= top && px >= b.x * TILE && px < (b.x + b.w) * TILE) return b;
      }
    }
    const i = w.idx(tx, ty);
    if (w.fence[i]) return { kind: 'fence', x: tx, y: ty };
    const reg = w.regionAt(tx, ty);
    if (reg && !reg.public && reg.size < 1200) return { kind: 'paddock', x: tx, y: ty, reg };
    return null;
  }

  select(o) {
    this.selected = o;
    this.follow = false;
    this.renderInfo(true);
  }

  // ---------------- buttons ----------------
  bindButtons() {
    for (const b of document.querySelectorAll('#speedBtns button')) b.onclick = () => { this.setSpeed(+b.dataset.speed); this.sfx.play('click'); };
    $('#alarmBtn').onclick = () => this.toggleAlarm();
    $('#finBtn').onclick = () => this.showFinances();
    $('#rosterBtn').onclick = () => this.showRoster();
    $('#disBtn').onclick = () => this.showDisasters();
    $('#sndBtn').onclick = () => {
      // Sound + music -> sound only -> mute -> ...
      const s = this.sfx;
      if (!s.muted && s.musicOn) { s.stopMusic(); s.wantMusic = false; this.toast('Music off', 'info'); }
      else if (!s.muted) { s.muted = true; if (s.master) s.master.gain.value = 0; this.toast('All sound off', 'info'); }
      else { s.muted = false; if (s.master) s.master.gain.value = s.vol; s.wantMusic = true; s.startMusic(); this.toast('Sound and music on', 'info'); }
      $('#sndBtn').textContent = s.muted ? '✕' : s.musicOn ? '♫' : '♪';
      try { localStorage.setItem('jt_audio', s.muted ? 'mute' : s.musicOn ? 'all' : 'sfx'); } catch (e) { /* storage unavailable */ }
    };
    $('#menuBtn').onclick = () => this.showMenu();
    for (const b of document.querySelectorAll('#miniBtns button[data-ov]')) b.onclick = () => this.toggleOverlay(b.dataset.ov);
    $('#findBtn').onclick = () => this.jumpToLoose();
    const mm = $('#minimap');
    const mmNav = (e) => {
      const r = mm.getBoundingClientRect();
      const x = (e.clientX - r.left) / r.width * this.game.world.W, y = (e.clientY - r.top) / r.height * this.game.world.H;
      this.centerOn(x, y); this.follow = false;
    };
    mm.addEventListener('mousedown', (e) => { mmNav(e); this.mmDrag = true; });
    window.addEventListener('mousemove', (e) => { if (this.mmDrag) mmNav(e); });
    window.addEventListener('mouseup', () => { this.mmDrag = false; });
    mm.addEventListener('touchstart', (e) => { e.preventDefault(); mmNav(e.touches[0]); }, { passive: false });
    $('#modal').addEventListener('mousedown', (e) => { if (e.target.id === 'modal' && !this.modalLocked) this.closeModal(); });
  }

  setSpeed(s) {
    if (s > 0) this.prevSpeed = s;
    this.speed = s;
    for (const b of document.querySelectorAll('#speedBtns button')) b.classList.toggle('on', +b.dataset.speed === s);
  }

  toggleAlarm() {
    const g = this.game;
    g.alarm = !g.alarm;
    $('#alarmBtn').classList.toggle('on', g.alarm);
    if (g.alarm) { g.log('EVACUATION ALARM: guests are heading to shelters.', 'warn', null, true); this.sfx.play('alarm'); }
    else g.log('All clear. Guests may leave the shelters.', 'good', null, true);
  }

  toggleOverlay(name) {
    this.r.overlay = this.r.overlay === name ? null : name;
    for (const b of document.querySelectorAll('#miniBtns button[data-ov]')) b.classList.toggle('on', b.dataset.ov === this.r.overlay);
  }

  jumpToLoose() {
    const loose = this.game.creatures().filter((d) => d.loose && !d.carried);
    if (!loose.length) { this.toast('No dinosaurs are loose. For now.', 'good'); return; }
    this.looseIdx = ((this.looseIdx || 0) + 1) % loose.length;
    const d = loose[this.looseIdx];
    this.centerOn(d.x, d.y); this.select(d);
  }

  // ---------------- tooltip / toasts / log ----------------
  showTip(e, html) {
    const t = $('#tip');
    t.innerHTML = html; t.style.display = 'block';
    const x = Math.min(window.innerWidth - t.offsetWidth - 6, e.clientX + 16);
    const y = Math.min(window.innerHeight - t.offsetHeight - 6, e.clientY + 18);
    t.style.left = x + 'px'; t.style.top = y + 'px';
  }
  hideTip() { $('#tip').style.display = 'none'; }

  toast(msg, type = 'info') {
    const el = document.createElement('div');
    el.className = 'toast ' + type; el.textContent = msg;
    const box = $('#toasts');
    box.appendChild(el);
    while (box.children.length > 3) box.removeChild(box.firstChild);
    setTimeout(() => { el.style.opacity = '0'; }, 3200);
    setTimeout(() => el.remove(), 3800);
  }

  headline(e) {
    const m = e.msg;
    const H = [
      [/eaten|flipped a tour jeep/, ['ISLAND PARK UNDER FIRE AFTER "UNFORTUNATE INCIDENT"', 'LAWYERS BOOK FLIGHTS TO ISLA NUBLAR', 'CEO INSISTS PARK IS "PERFECTLY SAFE"', 'VISITOR ASKS FOR REFUND, IS INFORMED HE IS LUNCH']],
      [/CONTAINMENT BREACH/, ['WITNESSES REPORT "VERY LARGE CHICKEN" ON THE LOOSE', 'PARK SPOKESMAN: "EVERYTHING IS UNDER CONTROL"', 'CHAOS THEORIST: "TOLD YOU SO"']],
      [/SYSTEM FAILURE/, ['IT DEPARTMENT BLAMES SINGLE UNDERPAID PROGRAMMER', 'POWER OUTAGE: "AH AH AH, YOU DIDN\'T SAY THE MAGIC WORD"']],
      [/storm has hit/, ['TROPICAL STORM BATTERS DINO ISLAND', 'BOAT TO MAINLAND CANCELLED, AGAIN']],
      [/ERUPTION/, ['VOLCANO ERUPTS. EXPERTS ASK WHY THE PARK WAS BUILT NEXT TO A VOLCANO']],
      [/EARTHQUAKE/, ['QUAKE ROCKS ISLAND, RIPPLES SPOTTED IN WATER GLASSES']],
      [/has hatched/, ['NEW ARRIVAL AT THE HATCHERY: "SHE\'S A GIRL. THEY\'RE ALL GIRLS."', 'SCIENTISTS CELEBRATE LATEST HATCHLING', 'PARK SPARES NO EXPENSE ON NEW HATCHLING']],
      [/rated \d star/, ['PARK RATING SOARS: CRITICS CALL IT "WONDERFUL"', 'TRAVEL GUIDES ADD DINO ISLAND TO MUST-SEE LIST']],
      [/sedated/, ['RANGERS RESTORE ORDER WITH TRANQUILIZER DARTS']],
      [/inspection passed/, ['INSPECTOR IMPRESSED: "SURPRISINGLY FEW TEETH MARKS"']],
      [/inspector fined/, ['SAFETY INSPECTOR: "I HAVE NEVER SEEN SO MANY BITE MARKS"']],
    ];
    for (const [re, opts] of H) if (re.test(m)) return pick(opts);
    return null;
  }

  showTicker(text) {
    const box = $('#ticker');
    box.style.display = 'block';
    box.innerHTML = '';
    const s = document.createElement('span');
    s.textContent = '📰 ' + text;
    box.appendChild(s);
    const W = box.clientWidth;
    let x = W;
    const anim = () => {
      x -= 1.6;
      s.style.left = x + 'px';
      if (x > -s.offsetWidth && s.parentNode) requestAnimationFrame(anim);
      else if (s.parentNode) box.style.display = 'none';
    };
    anim();
  }

  onLog(e) {
    if (e.big) this.toast(e.msg, e.type);
    const hl = (e.type === 'bad' || e.type === 'good' || e.type === 'goal') ? this.headline(e) : null;
    if (hl && (!this.tickerT || performance.now() - this.tickerT > 15000)) { this.tickerT = performance.now(); this.showTicker(hl); }
    const box = $('#log');
    const el = document.createElement('div');
    el.className = e.type;
    const hh = String(Math.floor(e.hour)).padStart(2, '0');
    el.textContent = `D${e.day} ${hh}h · ${e.msg}`;
    if (e.ref && e.ref.entity && e.ref.entity.kind === 'fence') el.onclick = () => { this.centerOn(e.ref.x, e.ref.y); this.select({ kind: 'fence', x: e.ref.entity.fx, y: e.ref.entity.fy }); };
    else if (e.ref) el.onclick = () => { const ent = e.ref.entity; if (ent && ent.x !== undefined && !ent.dead) { this.centerOn(ent.kind ? ent.x : ent.x + (ent.w || 0) / 2, ent.kind ? ent.y : ent.y + (ent.h || 0) / 2); if (ent.kind) this.select(ent); } else this.centerOn(e.ref.x, e.ref.y); };
    box.appendChild(el);
    while (box.children.length > 6) box.removeChild(box.firstChild);
    if (e.type === 'goal') this.renderGoal();
  }

  renderGoal() {
    const g = this.game;
    const el = $('#goal');
    if (g.scenario) {
      const sc = SCENARIOS[g.scenario];
      el.innerHTML = `<b>SCENARIO · ${sc.name.toUpperCase()}</b>${sc.objective}<br><span style="color:var(--gold)" id="scProg">${sc.progress(g)}</span>`;
      return;
    }
    if (g.goalIdx >= GOALS.length) { el.innerHTML = `<b>ALL GOALS COMPLETE</b>Spared no expense. Keep building!`; return; }
    const goal = GOALS[g.goalIdx];
    el.innerHTML = `<b>GOAL ${g.goalIdx + 1}/${GOALS.length}</b>${goal.text}${goal.reward ? ` <span style="color:var(--gold)">+${fmtMoney(goal.reward)}</span>` : ''}${goal.tool ? ' <button id="goalBtn" style="font-size:15px;padding:0 6px;margin-left:4px">Show me</button>' : ''}`;
    const gb = $('#goalBtn');
    if (gb) gb.onclick = () => {
      this.sfx.play('click');
      if (goal.tool === 'hatch') { this.showSpeciesPicker(); return; }
      const grp = TOOL_GROUPS.find((gr) => gr.tools.includes(goal.tool));
      this.setTool(goal.tool);
      if (grp && window.innerWidth >= 760) this.openFlyout(grp);
      const tips = { paddock: 'Drag a rectangle inside the blue power coverage, then put a Feeder inside.', power: 'Place it near where your paddocks will go. Fences only work in its blue radius.', visitor: 'Build it next to a path, and connect paths from the gate past your paddocks.' };
      this.toast(tips[goal.tool] || `Selected: ${(BUILDINGS[goal.tool] || TOOL_INFO[goal.tool]).name}. Click on the map to place it.`, 'info');
    };
  }

  // ---------------- HUD ----------------
  updateHUD(dt) {
    const g = this.game, w = g.world;
    this.hudT -= dt;
    if (this.hudT > 0) return;
    this.hudT = 0.2;
    const m = $('#sMoney');
    m.textContent = fmtMoney(g.money); m.classList.toggle('neg', g.money < 0);
    $('#sGuests').textContent = g.guests.length;
    $('#sDay').textContent = 'Day ' + g.day;
    const h = Math.floor(g.hour), mi = Math.floor((g.hour % 1) * 60 / 15) * 15;
    $('#sTime').textContent = `${String(h).padStart(2, '0')}:${String(mi).padStart(2, '0')}${g.isNight ? ' ☾' : ''}`;
    $('#stars').textContent = '★'.repeat(g.stars) + '☆'.repeat(5 - g.stars);
    $('#sRep').textContent = Math.round(g.reputation);
    const p = w.power;
    const pct = p.demand > 0 ? Math.min(1, p.supply / p.demand) : (p.supply > 0 ? 1 : 0);
    const bar = $('#powerbar i');
    bar.style.width = (pct * 100) + '%';
    bar.style.background = p.outage > 0 ? '#e04838' : pct < 1 ? '#f0a030' : '#f8d040';
    $('#sPow').textContent = `${Math.round(p.demand)}/${Math.round(p.supply)} MW`;
    $('#sPowK').textContent = p.outage > 0 ? 'GRID DOWN' : 'Power';
    $('#sPowK').style.color = p.outage > 0 ? '#e04838' : '';
    $('#alarmBtn').classList.toggle('on', g.alarm);
    this.renderAlerts();
    if (g.scenario) { const p = $('#scProg'); if (p) p.textContent = SCENARIOS[g.scenario].progress(g); }
    this.infoT -= 0.2;
    if (this.infoT <= 0) { this.infoT = 0.4; this.renderInfo(false); }
  }

  renderAlerts() {
    const g = this.game, w = g.world;
    const out = [];
    const loose = g.creatures().filter((d) => d.loose && !d.carried);
    if (loose.length) {
      const sed = loose.filter((d) => d.sedatedT > 0).length;
      out.push(`<div class="alert red" data-act="loose">⚠ ${loose.length} DINOSAUR${loose.length > 1 ? 'S' : ''} LOOSE${sed ? ` (${sed} SEDATED)` : ''}${!g.hasBuilding('ranger') ? ' · NO RANGERS!' : ''}</div>`);
    }
    if (w.power.outage > 0) {
      const backup = g.hasBuilding('backup') && w.power.backupFuel > 0 && w.power.backupDelay <= 0;
      out.push(`<div class="alert red" data-act="none">⚡ GRID DOWN ${Math.ceil(w.power.outage)}s${backup ? ' · BACKUP ON' : ''} <button data-act="reboot">Manual restart $15k</button></div>`);
    }
    const s = g.events.storm;
    if (s) out.push(`<div class="alert ${s.phase === 'active' ? 'blue' : 'amber'}">⛈ ${s.phase === 'warn' ? 'STORM INCOMING ' + Math.ceil(s.t) + 's' : 'TROPICAL STORM ' + Math.ceil(s.t) + 's'}</div>`);
    const v = g.events.volcano;
    if (v) out.push(`<div class="alert ${v.phase === 'erupt' ? 'red' : 'amber'}" data-act="volcano">🌋 ${v.phase === 'warn' ? 'ERUPTION IN ' + Math.ceil(v.t) + 's' : 'ERUPTION!'}</div>`);
    if (g.events.quake > 0) out.push(`<div class="alert red">〰 EARTHQUAKE</div>`);
    if (g.alarm) out.push(`<div class="alert amber" data-act="alarm">🔊 EVACUATION IN PROGRESS · click to sound all-clear</div>`);
    const html = out.join('');
    if (html !== this.lastAlerts) {
      this.lastAlerts = html;
      const box = $('#alerts');
      box.innerHTML = html;
      for (const el of box.querySelectorAll('[data-act]')) {
        el.onclick = (e) => {
          e.stopPropagation();
          const a = el.dataset.act;
          if (a === 'loose') this.jumpToLoose();
          else if (a === 'reboot') {
            if (g.world.power.outage > 8 && g.canAfford(15000)) { g.spend(15000, 'ops'); g.world.power.outage = 8; g.log('Engineers are rebooting the grid manually...', 'info', null, true); }
          } else if (a === 'volcano') this.centerOn(w.volcano.x, w.volcano.y);
          else if (a === 'alarm') this.toggleAlarm();
        };
      }
    }
  }

  // ---------------- per-frame ----------------
  frame(dt) {
    const k = this.keys;
    const sp = 420 * dt / this.r.cam.zoom;
    let mx = 0, my = 0;
    if (k['a'] || k['arrowleft']) mx -= 1;
    if (k['d'] || k['arrowright']) mx += 1;
    if (k['w'] || k['arrowup']) my -= 1;
    if (k['s'] || k['arrowdown']) my += 1;
    if (mx || my) { this.r.cam.x += mx * sp; this.r.cam.y += my * sp; this.follow = false; this.clampCam(); }
    if (this.follow && this.selected && this.selected.kind && !this.selected.dead) {
      const z = this.r.cam.zoom;
      const tx = this.selected.x * TILE - this.r.canvas.width / z / 2, ty = this.selected.y * TILE - this.r.canvas.height / z / 2;
      this.r.cam.x += (tx - this.r.cam.x) * Math.min(1, dt * 5);
      this.r.cam.y += (ty - this.r.cam.y) * Math.min(1, dt * 5);
      this.clampCam();
    }
    this.updateHUD(dt);
  }
}
