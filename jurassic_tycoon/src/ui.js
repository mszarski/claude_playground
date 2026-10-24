// ---------- UI: input, panels, HUD ----------
'use strict';

const $ = (s) => document.querySelector(s);
const LINE_TOOLS = ['path', 'widepath', 'route', 'track', 'fence', 'wall'];
const RECT_TOOLS = ['demolish', 'trees', 'pond', 'clear'];
const isUnit = (o) => o && (o.kind === 'ranger' || o.kind === 'worker' || o.kind === 'heli');

class UI {
  constructor(game, renderer, sfx) {
    this.game = game; this.r = renderer; this.sfx = sfx;
    this.tool = 'inspect'; this.species = null;
    this.hover = null; this.drag = null; this.dragTiles = null;
    this.selected = null; this.follow = false;
    this.squad = []; // RTS selection: rangers, engineers and helicopters you can give orders to
    this.speed = 1; this.prevSpeed = 1;
    this.keys = {};
    // 'auto' guesses mouse wheel (zoom) vs trackpad (pan); 'zoom' and 'pan' force one
    this.wheelMode = 'auto';
    try { this.wheelMode = localStorage.getItem('jt_wheel') || 'auto'; } catch (e) { /* storage unavailable */ }
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
    this.selected = null; this.squad = [];
    this.decisionShown = null;
    $('#log').innerHTML = '';
    game.on('log', (e) => this.onLog(e));
    game.on('gameover', (reason) => this.showGameOver(reason));
    game.on('victory', () => this.showVictory());
    game.on('scenario', (r) => this.showScenarioEnd(r));
    game.on('breach', (newly) => {
      if (this.speed > 1) this.setSpeed(1);
      // a dangerous escape gets a moment: slow motion, a roar, a red flash
      const worst = newly.reduce((a, d) => (d.sp.danger > a.sp.danger ? d : a), newly[0]);
      if (worst.sp.danger >= 4) {
        this.slowmo = 1.3;
        this.sfx.play(worst.sp.size >= 3 ? 'roar' : 'screech', 1.2);
        setTimeout(() => this.sfx.play('scream', 0.8), 450);
        const f = $('#flash'); f.classList.remove('go'); void f.offsetWidth; f.classList.add('go');
      }
    });
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
      const nm = document.createElement('span'); nm.className = 'nm'; nm.innerHTML = `${info.name}${TOOL_INFO[t] && TOOL_INFO[t].key ? ` <span class="k key">[${TOOL_INFO[t].key}]</span>` : ''}`;
      b.appendChild(nm);
      if (BUILDINGS[t] && BUILDINGS[t].unlockStars && this.game.stars < BUILDINGS[t].unlockStars && !this.game.sandbox) { b.style.opacity = 0.5; nm.innerHTML += ` <span class="k">${'★'.repeat(BUILDINGS[t].unlockStars)}</span>`; }
      if (cost) { const cs = document.createElement('span'); cs.className = 'c'; cs.textContent = fmtMoney(cost) + (LINE_TOOLS.includes(t) || RECT_TOOLS.includes(t) || t === 'paddock' ? '/t' : ''); b.appendChild(cs); }
      b.onmouseenter = (e) => this.showTip(e, `<b>${info.name}</b><br>${info.desc}` + (BUILDINGS[t] && BUILDINGS[t].upkeep ? `<br><span style="color:#9ab08a">Upkeep ${fmtMoney(BUILDINGS[t].upkeep)}/day${BUILDINGS[t].power ? ' · ' + BUILDINGS[t].power + ' MW' : ''}</span>` : ''));
      b.onmouseleave = () => this.hideTip();
      b.onclick = () => {
        this.sfx.play('click');
        // the picker covers the map; the menu must not be left open under it to catch the next tap
        if (t === 'hatch') { this.closeFlyout(); this.showSpeciesPicker(); return; }
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
    // keep the category of the active tool lit so it's clear what a click will do
    const grp = TOOL_GROUPS.find((gr) => gr.tools.includes(this.tool));
    const cat = this.tool === 'inspect' || !grp ? 'inspect' : grp.id;
    for (const c of document.querySelectorAll('.cat')) c.classList.toggle('on', c.dataset.cat === cat);
    this.hideTip();
  }

  setTool(t) {
    this.tool = t;
    this.drag = null; this.dragTiles = null;
    this.r.canvas.style.cursor = t === 'inspect' ? 'default' : 'crosshair';
    if (this.lastSquad !== undefined && t !== 'order' && this.squad && this.squad.length) this.lastSquad = '';
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
    // double-click a ranger/engineer/helicopter: select every one of that kind on screen
    cv.addEventListener('dblclick', (e) => {
      if (this.tool !== 'inspect') return;
      const t = this.screenToTile(e.clientX, e.clientY);
      const u = this.pickAt(t.wx, t.wy);
      if (u && isUnit(u)) this.selectSquad(this.unitsOnScreen().filter((x) => x.kind === u.kind));
    });
    cv.addEventListener('mousedown', (e) => {
      this.sfx.ensure();
      this.mouse.down = true; this.mouse.button = e.button;
      this.mouse.sx = e.clientX; this.mouse.sy = e.clientY; this.mouse.moved = 0;
      this.mouse.camX = this.r.cam.x; this.mouse.camY = this.r.cam.y;
      // the open menu would hide part of the map: tuck it away once the player starts building
      if (this.openCat && e.button === 0) this.closeFlyout();
      if (e.button !== 0) { this.panning = true; return; }
      if (e.shiftKey && (this.tool === 'inspect' || this.tool === 'order')) { this.box = { x0: e.clientX, y0: e.clientY, x1: e.clientX, y1: e.clientY }; return; }
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
      if (this.mouse.down && this.box) { this.box.x1 = e.clientX; this.box.y1 = e.clientY; return; }
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
        this.showTip(e, this.tool === 'route' && this.routeFail ? '<span style="color:#ff9080">No route between these points</span>' : `${this.dragLabel()}${cost ? ' · ' + fmtMoney(cost) : ''}`);
      } else if (e.target === cv && BUILDINGS[this.tool]) {
        const def = BUILDINGS[this.tool], w = this.game.world;
        const bx = t.x - Math.floor((def.w - 1) / 2), by = t.y - Math.floor((def.h - 1) / 2);
        let why = null;
        for (let dy = 0; dy < def.h && !why; dy++) for (let dx = 0; dx < def.w && !why; dx++) {
          why = w.placementBlock(bx + dx, by + dy);
          if (!why && this.game.dinos.some((d) => d.tx === bx + dx && d.ty === by + dy)) why = 'a dinosaur';
        }
        if (!why && !this.game.canAfford(def.cost)) why = 'not enough money';
        const offGrid = !why && this.tool === 'pylon' && !w.covered[w.idx(bx, by)];
        this.showTip(e, `${def.name} · ${fmtMoney(def.cost)}${why ? `<br><span style="color:#ff9080">Blocked by ${why}</span>` : ''}${offGrid ? '<br><span style="color:#ff9080">Outside the blue power area: it won\'t connect</span>' : ''}`);
      } else if (e.target === cv && this.tool === 'stamp' && this.blueprint) {
        const bp = this.blueprint, plan = this.game.stampPlan(bp, t.x - Math.floor(bp.w / 2), t.y - Math.floor(bp.h / 2));
        this.showTip(e, `Paddock blueprint · ${fmtMoney(plan.cost)}${plan.blocked ? `<br><span style="color:#ff9080">${plan.blocked} fence tiles blocked</span>` : ''}`);
      } else if (e.target === cv && this.tool === 'hatch' && this.species) {
        const err = this.game.canHatchAt(this.species, t.x, t.y);
        this.showTip(e, err ? `<span style="color:#ff9080">${err}</span>` : `Hatch ${SPECIES[this.species].name} · ${this.game.hatchCost(this.species) ? fmtMoney(this.game.hatchCost(this.species)) : 'FREE'}`);
      } else if (e.target === cv) {
        // hover a dinosaur for a quick read on how it's doing
        const d = (this.tool === 'inspect' || !BUILDINGS[this.tool]) && this.pickAt(t.wx, t.wy);
        if (d && d.kind === 'dino') this.showTip(e, this.dinoTip(d));
        else this.hideTip();
      }
    });
    window.addEventListener('mouseup', (e) => {
      if (!this.mouse.down) return;
      this.mouse.down = false;
      const wasPan = this.panning; this.panning = false;
      if (this.box) { this.finishBox(e.shiftKey); return; }
      if (e.target !== cv && !this.drag) return;
      const t = this.screenToTile(e.clientX, e.clientY);
      // right-click (not a right-drag pan) gives the selected units an order
      if (this.mouse.button === 2) { if (this.mouse.moved < 6 && this.squad.length) this.issueOrder(t); return; }
      if (this.mouse.button !== 0) return;
      if (this.drag) { this.applyDrag(); this.drag = null; this.dragTiles = null; this.hideTip(); return; }
      if (wasPan) return;
      this.click(t);
    });
    // Wheel: a mouse wheel zooms; a trackpad's two-finger scroll pans and its pinch zooms.
    // Browsers report a trackpad pinch as a wheel event with ctrlKey set.
    cv.addEventListener('wheel', (e) => {
      e.preventDefault();
      const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? window.innerHeight : 1;
      const dx = e.deltaX * unit, dy = e.deltaY * unit;
      if (e.ctrlKey) { this.stepZoom(-dy, 3, e.clientX, e.clientY); return; }
      if (this.wheelMode === 'auto' && (dx !== 0 || !Number.isInteger(dy) || (dy !== 0 && Math.abs(dy) < 50))) this.sawTrackpad = true;
      const pan = this.wheelMode === 'pan' || (this.wheelMode === 'auto' && this.sawTrackpad);
      if (pan) {
        const z = this.r.cam.zoom;
        this.r.cam.x += (e.shiftKey && !dx ? dy : dx) / z;
        this.r.cam.y += (e.shiftKey && !dx ? 0 : dy) / z;
        this.follow = false; this.clampCam();
        if (!this.panHinted) { this.panHinted = true; this.toast('Scroll pans · pinch or +/− zooms (change in Menu)', 'info'); }
      } else this.stepZoom(-dy, 1, e.clientX, e.clientY);
    }, { passive: false });
    // Safari sends trackpad pinches as gesture events instead
    let gest = null;
    cv.addEventListener('gesturestart', (e) => { e.preventDefault(); gest = { z: this.r.cam.zoom }; });
    cv.addEventListener('gesturechange', (e) => {
      e.preventDefault();
      if (!gest) return;
      const nz = clamp(Math.round(gest.z * e.scale), 1, 6);
      if (nz !== this.r.cam.zoom) this.zoomAt(nz - this.r.cam.zoom, e.clientX || window.innerWidth / 2, e.clientY || window.innerHeight / 2);
    });
    cv.addEventListener('gestureend', (e) => { e.preventDefault(); gest = null; });

    // touch: one finger = tool (or pan in Inspect), two fingers = pan + pinch zoom
    let pinch = null;
    const mid = (a, b) => ({ x: (a.clientX + b.clientX) / 2, y: (a.clientY + b.clientY) / 2 });
    cv.addEventListener('touchstart', (e) => {
      e.preventDefault();
      this.sfx.ensure();
      this.touchUsed = true;
      if (e.touches.length >= 2) {
        const [a, b] = e.touches, m = mid(a, b);
        pinch = { d: Math.max(1, dist(a.clientX, a.clientY, b.clientX, b.clientY)), z: this.r.cam.zoom, mx: m.x, my: m.y, cx: this.r.cam.x, cy: this.r.cam.y };
        this.drag = null; this.dragTiles = null; this.mouse.down = false; this.panning = false; this.hideTip();
        return;
      }
      if (pinch) return;
      const tt = e.touches[0];
      this.mouse.down = true; this.mouse.button = 0; this.mouse.sx = tt.clientX; this.mouse.sy = tt.clientY; this.mouse.moved = 0;
      this.mouse.camX = this.r.cam.x; this.mouse.camY = this.r.cam.y;
      const t = this.screenToTile(tt.clientX, tt.clientY);
      this.hover = t;
      if (LINE_TOOLS.includes(this.tool) || RECT_TOOLS.includes(this.tool) || this.tool === 'paddock') { this.drag = { x: t.x, y: t.y }; this.updateDrag(t); }
    }, { passive: false });
    cv.addEventListener('touchmove', (e) => {
      e.preventDefault();
      if (pinch) {
        if (e.touches.length < 2) return;
        const [a, b] = e.touches, m = mid(a, b), cam = this.r.cam;
        const d = dist(a.clientX, a.clientY, b.clientX, b.clientY);
        const nz = clamp(Math.round(pinch.z * d / pinch.d), 1, 6);
        // keep the world point that was under the fingers' midpoint under it
        const wx = pinch.cx + pinch.mx / pinch.z, wy = pinch.cy + pinch.my / pinch.z;
        cam.zoom = nz;
        cam.x = wx - m.x / nz; cam.y = wy - m.y / nz;
        this.follow = false; this.clampCam();
        return;
      }
      const tt = e.touches[0];
      const t = this.screenToTile(tt.clientX, tt.clientY);
      this.hover = t;
      this.mouse.moved = Math.max(this.mouse.moved, Math.abs(tt.clientX - this.mouse.sx) + Math.abs(tt.clientY - this.mouse.sy));
      if (this.drag) {
        this.updateDrag(t);
        if (this.dragTiles) { const cost = this.dragCost(); this.showTip({ clientX: tt.clientX, clientY: tt.clientY - 60 }, `${this.dragLabel()}${cost ? ' · ' + fmtMoney(cost) : ''}`); }
        return;
      }
      if (this.mouse.moved > 10) {
        const z = this.r.cam.zoom;
        this.r.cam.x = this.mouse.camX - (tt.clientX - this.mouse.sx) / z;
        this.r.cam.y = this.mouse.camY - (tt.clientY - this.mouse.sy) / z;
        this.follow = false; this.clampCam(); this.panning = true;
      }
    }, { passive: false });
    const touchEnd = (e) => {
      e.preventDefault();
      if (pinch) {
        // after a two-finger gesture, wait until every finger is up before a tap can count again
        if (e.touches.length === 0) pinch = null;
        return;
      }
      if (!this.mouse.down) return;
      this.mouse.down = false;
      this.hideTip();
      if (this.drag) { if (e.type === 'touchend') this.applyDrag(); this.drag = null; this.dragTiles = null; this.hover = null; return; }
      if (e.type === 'touchend' && !this.panning && this.hover) this.click(this.snapForTouch(this.hover));
      this.panning = false;
      // no finger, no hover: don't leave a build preview (or a red X) stuck where the last tap was
      this.hover = null;
    };
    cv.addEventListener('touchend', touchEnd, { passive: false });
    cv.addEventListener('touchcancel', touchEnd, { passive: false });

    window.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT') return;
      this.keys[e.key.toLowerCase()] = true;
      const k = e.key.toLowerCase();
      if (k === ' ') { e.preventDefault(); this.setSpeed(this.speed === 0 ? (this.prevSpeed || 1) : 0); }
      else if (k === '1') this.setSpeed(1);
      else if (k === '2') this.setSpeed(2);
      else if (k === '3') this.setSpeed(4);
      else if (k === '4') this.setSpeed(8);
      else if (k === 'escape') { if ($('#modal').style.display === 'flex') this.closeModal(); else if (this.tool !== 'inspect') { this.setTool('inspect'); this.closeFlyout(); } else if (this.squad.length) this.selectSquad([]); else if (this.selected) this.select(null); else this.showMenu(); }
      else if (k === 'm') this.selectSquad(this.game.staff.filter((s) => s.role === 'ranger' && !s.dead));
      else if (k === 'b' && this.game.looseCount) this.respond();
      else if (k === 'e') this.toggleAlarm();
      else if (k === 'o') this.toggleOverlay('power');
      else if (k === 'k') this.toggleOverlay('paddock');
      else if (k === 'l') this.jumpToLoose();
      else if (k === 'g') this.showRoster();
      else if (k === 'z' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); this.doUndo(); }
      else if (k === 'u') this.doUndo();
      else if (k === 'y') this.showResearch();
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
    const trackShift = (e) => { this.shiftDown = e.shiftKey; };
    window.addEventListener('mousedown', trackShift, true); window.addEventListener('mouseup', trackShift, true);
  }

  // Accumulate fine-grained wheel/pinch input into whole zoom steps so pixels stay crisp.
  stepZoom(amount, scale, sx, sy) {
    this.zAcc = (this.zAcc || 0) + amount * scale;
    if (Math.abs(this.zAcc) < 60) return;
    this.zoomAt(Math.sign(this.zAcc), sx, sy);
    this.zAcc = 0;
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
    // let the shore scroll clear of the toolbar (left) and the panels along the bottom (tutorial, minimap, goals)
    const left = Math.max(40, 80 / cam.zoom), bottom = Math.max(40, vh * 0.45);
    cam.x = vw >= W ? (W - vw) / 2 : clamp(cam.x, -left, W - vw + 40);
    cam.y = vh >= H ? (H - vh) / 2 : clamp(cam.y, -60, H - vh + bottom);
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
    if (this.tool === 'route') {
      const r = this.game.world.route(d.x, d.y, t.x, t.y);
      this.dragTiles = r || [[d.x, d.y]];
      this.routeFail = !r;
      return;
    }
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
      if (this.tool === 'widepath') {
        const seen = new Set(), wide = [];
        for (const [x, y] of tiles) for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          const k = (x + dx) + ',' + (y + dy);
          if (!seen.has(k)) { seen.add(k); wide.push([x + dx, y + dy]); }
        }
        tiles.length = 0; tiles.push(...wide);
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
      if ((tool === 'path' || tool === 'widepath' || tool === 'route') && !w.path[i] && !w.track[i] && w.canBuildAt(x, y)) c += TOOL_INFO.path.cost + (w.terrain[i] === T_FOREST ? 40 : 0);
      else if (tool === 'track' && !w.path[i] && !w.track[i] && w.canBuildAt(x, y)) c += TOOL_INFO.track.cost;
      else if ((tool === 'fence' || tool === 'paddock') && w.fence[i] !== F_ELECTRIC && w.canBuildAt(x, y) && !w.path[i]) c += FENCE_DEF[F_ELECTRIC].cost;
      else if (tool === 'wall' && w.fence[i] !== F_WALL && w.canBuildAt(x, y) && !w.path[i]) c += FENCE_DEF[F_WALL].cost;
      else if (tool === 'trees' && w.terrain[i] === T_GRASS && !w.path[i] && !w.bld[i] && !w.fence[i]) c += TOOL_INFO.trees.cost;
      else if (tool === 'pond' && (w.terrain[i] === T_GRASS || w.terrain[i] === T_FOREST || w.terrain[i] === T_SAND) && !w.path[i] && !w.track[i] && !w.bld[i] && !w.fence[i]) c += TOOL_INFO.pond.cost;
      else if (tool === 'clear' && w.terrain[i] === T_FOREST) c += TOOL_INFO.clear.cost;
    }
    return c;
  }

  applyDrag() {
    const g = this.game, tool = this.tool;
    if (!this.dragTiles) return;
    let n = 0, queued = 0, gates = 0;
    g.startBatch();
    for (const [x, y] of this.dragTiles) {
      let ok = false;
      if (tool === 'path' || tool === 'widepath' || tool === 'route') ok = g.placePath(x, y);
      else if (tool === 'track') { const f0 = g.world.fence[g.world.idx(x, y)]; ok = g.placeTrack(x, y); if (ok && f0 && f0 !== F_GATE) gates++; }
      else if (tool === 'fence' || tool === 'paddock' || tool === 'wall') {
        const type = tool === 'wall' ? F_WALL : F_ELECTRIC;
        ok = g.placeFence(x, y, type);
        if (!ok && g.fenceBlockedByDino(x, y) && g.world.fence[g.world.idx(x, y)] !== type) { g.queueFence(x, y, type); queued++; }
      }
      else if (tool === 'demolish') {
        // demolish only fences/paths in drag, buildings only on single tile
        const w = g.world, i = w.idx(x, y);
        if (w.bld[i] && this.dragTiles.length > 1) continue;
        ok = g.demolishAt(x, y);
      }
      else if (tool === 'trees') ok = g.plantTree(x, y);
      else if (tool === 'pond') ok = g.digPond(x, y);
      else if (tool === 'clear') ok = g.clearLand(x, y);
      if (ok) n++;
    }
    g.endBatch(tool === 'demolish' ? 'demolition' : (TOOL_INFO[tool] || {}).name || tool);
    if (n) this.sfx.play(tool === 'demolish' ? 'demolish' : 'build');
    else this.sfx.play('error');
    if (n && (tool === 'fence' || tool === 'paddock')) {
      const w = g.world;
      w.computePower(g);
      let off = 0, total = 0;
      for (const [x, y] of this.dragTiles) { const i = w.idx(x, y); if (w.fence[i] === F_ELECTRIC) { total++; if (!w.fencePowered[i]) off++; } }
      if (off) this.toast(`${off} of ${total} fence tiles have no power (blinking red). Build a Power Plant or Pylon nearby.`, 'warn');
    }
    if (gates > 2) this.toast(`The track runs along a fence: ${gates} fence tiles became Track Gates (${fmtMoney(gates * FENCE_DEF[F_GATE].cost)}). Gates don't zap, so that side of the paddock is weaker. Cross fences at a right angle instead (Undo to take it back).`, 'warn');
    if (queued) this.toast(`A dinosaur or lava is in the way of ${queued} fence tile${queued > 1 ? 's' : ''}. ${queued > 1 ? 'They\'ll' : 'It\'ll'} be built as soon as the way is clear.`, 'info');
    else if (n && tool === 'paddock' && this.dragTiles.length > 4) {
      // a paddock that doesn't close is just an expensive fence: say so
      const w = g.world;
      w.computeRegions();
      const xs = this.dragTiles.map((p) => p[0]), ys = this.dragTiles.map((p) => p[1]);
      const cx = Math.round((Math.min(...xs) + Math.max(...xs)) / 2), cy = Math.round((Math.min(...ys) + Math.max(...ys)) / 2);
      const reg = w.regionAt(cx, cy);
      if (reg && reg.public && Math.max(...xs) - Math.min(...xs) > 1 && Math.max(...ys) - Math.min(...ys) > 1) this.toast('This paddock isn\'t closed: something blocked part of the fence. Look for the gap and fill it with the Fence tool.', 'warn');
    }
  }

  // Paddocks and areas read better as width × height than as a tile count
  dragLabel() {
    const d = this.drag, t = this.dragTiles;
    if (d && (this.tool === 'paddock' || RECT_TOOLS.includes(this.tool)) && t.length) {
      const xs = t.map((p) => p[0]), ys = t.map((p) => p[1]);
      const w = Math.max(...xs) - Math.min(...xs) + 1, h = Math.max(...ys) - Math.min(...ys) + 1;
      return this.tool === 'paddock' ? `${w}×${h} paddock (${Math.max(0, (w - 2) * (h - 2))} tiles inside)` : `${w}×${h}`;
    }
    return `${t.length} tiles`;
  }

  // A fingertip is less precise than a mouse and there is no hover preview: nudge a building to the
  // nearest spot within 2 tiles where it fits
  snapForTouch(t) {
    const def = BUILDINGS[this.tool], w = this.game.world;
    if (!def) return t;
    const ox = Math.floor((def.w - 1) / 2), oy = Math.floor((def.h - 1) / 2);
    if (w.canPlaceBuilding(this.tool, t.x - ox, t.y - oy)) return t;
    let best = null, bd = 99;
    for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
      const d = dx * dx + dy * dy;
      if (d < bd && w.canPlaceBuilding(this.tool, t.x + dx - ox, t.y + dy - oy)) { bd = d; best = { x: t.x + dx, y: t.y + dy }; }
    }
    return best || t;
  }

  click(t) {
    const g = this.game, w = g.world;
    const tool = this.tool;
    if (BUILDINGS[tool]) {
      const def = BUILDINGS[tool];
      const bx = t.x - Math.floor((def.w - 1) / 2), by = t.y - Math.floor((def.h - 1) / 2);
      g.startBatch();
      const b = g.placeBuilding(tool, bx, by);
      g.endBatch(def.name);
      if (b) {
        this.hideTip();
        this.sfx.play('build');
        if (def.cat === 'guest' && !w.accessTiles(b).length) this.toast(`Connect the ${def.name} to a path!`, 'warn');
        if (tool === 'tour' && !w.trackAccess(b).length) this.toast('Now lay Tour Track from the station past your paddocks, ideally in a loop.', 'info');
        if (tool === 'pylon') { w.computePower(g); if (!w.covered[w.idx(b.x, b.y)]) this.toast('This pylon is outside the power area, so it is not connected. Place pylons inside the blue area to extend it.', 'warn'); }
        if (tool === 'gyro' && !g.gyroAccess(b).length) this.toast('A Gyrosphere Station must touch a paddock: build it right against the fence.', 'warn');
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
      else { g.startBatch(); const ok = g.hatch(this.species, t.x, t.y); g.endBatch(SPECIES[this.species].name + ' egg'); if (ok) this.sfx.play('build'); }
      return;
    }
    if (tool === 'demolish') { g.startBatch(); const ok = g.demolishAt(t.x, t.y); g.endBatch('demolition'); if (ok) this.sfx.play('demolish'); return; }
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
    if (tool === 'stamp' && this.blueprint) {
      const bp = this.blueprint;
      const r = g.stamp(bp, t.x - Math.floor(bp.w / 2), t.y - Math.floor(bp.h / 2));
      if (r.n) { this.sfx.play('build'); this.toast(`Paddock stamped (${fmtMoney(r.plan.cost)}).${r.plan.blocked ? ` ${r.plan.blocked} fence tiles were blocked, so it may not be closed: look for gaps.` : ''}`, r.plan.blocked ? 'warn' : 'good'); }
      else { this.sfx.play('error'); this.toast(r.money ? `Not enough money (${fmtMoney(r.plan.cost)}).` : "Can't stamp there.", 'warn'); }
      return;
    }
    if (tool === 'order') {
      this.issueOrder(t);
      this.setTool('inspect');
      return;
    }
    // Inspect: pick entity under cursor
    const pick = this.pickAt(t.wx, t.wy);
    if (this.shiftDown && pick && isUnit(pick)) {
      // shift-click adds or removes a unit from the squad
      const i = this.squad.indexOf(pick);
      if (i >= 0) this.squad.splice(i, 1); else this.squad.push(pick);
      this.renderSquad(true);
      this.sfx.play('click');
      return;
    }
    this.squad = pick && isUnit(pick) ? [pick] : [];
    this.renderSquad(true);
    this.select(pick);
    this.sfx.play('click');
  }

  // ---------------- RTS squad ----------------
  unitsOnScreen() {
    const z = this.r.cam.zoom, cam = this.r.cam;
    const x0 = cam.x / TILE, y0 = cam.y / TILE, x1 = x0 + this.r.canvas.width / z / TILE, y1 = y0 + this.r.canvas.height / z / TILE;
    return this.allUnits().filter((u) => u.x >= x0 && u.x <= x1 && u.y >= y0 && u.y <= y1);
  }
  allUnits() { return this.game.staff.filter((s) => !s.dead && !s.hidden).concat(this.game.helis.filter((h) => !h.dead)); }
  selectSquad(units) {
    this.squad = units.slice();
    if (units.length === 1) this.select(units[0]); else if (units.length) this.select(null);
    this.renderSquad(true);
    if (units.length) this.sfx.play('click');
  }
  finishBox(add) {
    const b = this.box; this.box = null;
    const a = this.screenToTile(Math.min(b.x0, b.x1), Math.min(b.y0, b.y1)), c = this.screenToTile(Math.max(b.x0, b.x1), Math.max(b.y0, b.y1));
    const inBox = this.allUnits().filter((u) => u.x >= a.wx && u.x <= c.wx && u.y >= a.wy - 0.6 && u.y <= c.wy + 0.4);
    this.selectSquad(add ? [...new Set(this.squad.concat(inBox))] : inBox);
  }
  issueOrder(t) {
    const g = this.game, w = g.world;
    this.squad = this.squad.filter((u) => !u.dead);
    if (!this.squad.length) return;
    const p = this.pickAt(t.wx, t.wy);
    let target;
    if (p && (p.kind === 'dino' || p.kind === 'ptera')) target = { dino: p };
    else if (p && p.kind === 'fence') target = { fence: { x: p.x, y: p.y }, x: p.x, y: p.y };
    else if (p && p.type && BUILDINGS[p.type]) target = { building: p, x: t.x, y: t.y };
    else target = { x: t.x, y: t.y };
    if (!target.dino && !w.humanPass(t.x, t.y) && !target.fence && !target.building) { this.toast("Units can't go there.", 'warn'); this.sfx.play('error'); return; }
    const n = g.commandUnits(this.squad, target);
    this.orderPing = { x: target.dino ? target.dino.x : t.x + 0.5, y: target.dino ? target.dino.y : t.y + 0.5, t: 0.8, hunt: !!target.dino };
    if (!n) { this.toast(target.dino ? 'No selected unit can take that order.' : 'Nothing to do there.', 'warn'); this.sfx.play('error'); return; }
    this.sfx.play('click');
    this.renderSquad(true);
  }
  respond() {
    const err = this.game.respondToBreach();
    if (err) this.toast(err, 'info'); else { if (this.game.alarm) this.sfx.play('alarm'); $('#alarmBtn').classList.toggle('on', this.game.alarm); }
  }
  renderSquad(force) {
    const el = $('#squad');
    this.squad = this.squad.filter((u) => !u.dead);
    const sq = this.squad;
    if (!sq.length) { if (!el.hidden) { el.hidden = true; this.lastSquad = ''; } if (this.tool === 'order') this.setTool('inspect'); return; }
    const count = (k) => sq.filter((u) => u.kind === k).length;
    const parts = [['ranger', 'ranger', '🎯'], ['worker', 'engineer', '🔧'], ['heli', 'helicopter', '🚁']].filter(([k]) => count(k)).map(([k, nm, ic]) => `${ic} ${count(k)} ${nm}${count(k) > 1 ? 's' : ''}`);
    const ordered = sq.filter((u) => u.order).length;
    const touch = this.touchUsed;
    const html = `<div class="sqhead">${parts.join(' · ')}${ordered ? ` <span>(${ordered} on orders)</span>` : ''}</div>
      <div class="sqhint">${touch ? 'Tap Send, then tap a dinosaur to hunt it, a fence to repair or ground to hold.' : 'Right-click a dinosaur to hunt it, a fence to repair, or ground to hold there.'}</div>
      <div class="btns"><button data-sq="send" class="${this.tool === 'order' ? 'on' : ''}">Send…</button><button data-sq="auto" ${ordered ? '' : 'disabled'} title="Let them work on their own again">Back to auto</button><button data-sq="clear">✕</button></div>`;
    if (!force && html === this.lastSquad) return;
    this.lastSquad = html;
    el.innerHTML = html; el.hidden = false;
    for (const b of el.querySelectorAll('[data-sq]')) b.onclick = () => {
      const a = b.dataset.sq;
      this.sfx.play('click');
      if (a === 'send') { this.setTool(this.tool === 'order' ? 'inspect' : 'order'); if (this.tool === 'order') this.toast('Tap where to send them.', 'info'); }
      else if (a === 'auto') { this.game.releaseUnits(this.squad); this.toast('Back to working on their own.', 'info'); }
      else if (a === 'clear') this.selectSquad([]);
      this.renderSquad(true);
    };
  }

  pickAt(wx, wy) {
    const g = this.game, w = g.world;
    const px = wx * TILE, py = wy * TILE;
    let best = null, by = -1e9;
    for (const h of g.helis) {
      const hy = (h.y - h.alt) * TILE;
      if (Math.abs(px - h.x * TILE) < 12 && py > hy - 12 && py < hy + 6) return h;
    }
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
    const rat = $('#stars').parentElement; rat.style.cursor = 'pointer'; rat.title = 'What makes the rating, and how to raise it'; rat.onclick = () => this.showRating();
    $('#undoBtn').onclick = () => this.doUndo();
    $('#resBtn').onclick = () => this.showResearch();
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
    $('#zoomIn').onclick = () => this.zoomAt(1, window.innerWidth / 2, window.innerHeight / 2);
    $('#zoomOut').onclick = () => this.zoomAt(-1, window.innerWidth / 2, window.innerHeight / 2);
    // panels sit below the top bar, which wraps to several rows on narrow screens
    const tb = $('#topbar');
    const syncTop = () => document.documentElement.style.setProperty('--top', tb.offsetHeight + 'px');
    const goalEl = $('#goal');
    const syncGoal = () => {
      // the squad bar and ticker make room for whichever bottom-left panel is showing: the goal, or the tutorial
      const tutEl = $('#tut');
      const panel = goalEl.offsetWidth ? goalEl : tutEl && tutEl.offsetWidth ? tutEl : null;
      document.documentElement.style.setProperty('--goalH', (panel ? panel.offsetHeight : 0) + 'px');
      document.documentElement.style.setProperty('--goalR', (panel ? panel.offsetLeft + panel.offsetWidth : 70) + 'px');
    };
    this.syncGoal = syncGoal;
    const dealsEl = $('#deals');
    const syncDeals = () => document.documentElement.style.setProperty('--dealsH', (dealsEl.hidden ? 0 : dealsEl.offsetHeight + 4) + 'px');
    if (window.ResizeObserver) { new ResizeObserver(syncTop).observe(tb); new ResizeObserver(syncGoal).observe(goalEl); if ($('#tut')) new ResizeObserver(syncGoal).observe($('#tut')); new ResizeObserver(syncDeals).observe(dealsEl); }
    syncTop();
    mm.addEventListener('mousedown', (e) => { mmNav(e); this.mmDrag = true; });
    window.addEventListener('mousemove', (e) => { if (this.mmDrag) mmNav(e); });
    window.addEventListener('mouseup', () => { this.mmDrag = false; });
    mm.addEventListener('touchstart', (e) => { e.preventDefault(); mmNav(e.touches[0]); }, { passive: false });
    $('#modal').addEventListener('mousedown', (e) => { if (e.target.id === 'modal' && !this.modalLocked) this.closeModal(); });
  }

  dinoTip(d) {
    const status = d.carried ? 'Being airlifted' : d.sedatedT > 0 ? 'Sedated' : d.loose ? '<span style="color:#ff6040">LOOSE!</span>' : d.sick > 0 ? '<span style="color:#9ae060">Sick</span>' : d.sleeping ? 'Sleeping' : 'Contained';
    const lbl = { space: 'too crowded', forest: 'wants tree cover', water: 'no water nearby (Dig Pond)', social: d.sp.territorial ? 'rival nearby' : 'lonely', food: 'no food source', fear: 'scared of predators', loose: 'out of its paddock' };
    let worst = null, wv = 1e9;
    const max = { space: 30, forest: 18, water: 12, social: 15, food: 15, fear: 0, loose: 0 };
    for (const k in d.comfortParts || {}) { if (!(k in lbl)) continue; const gap = d.comfortParts[k] - (max[k] || 0); if (gap < wv && gap < -3) { wv = gap; worst = k; } }
    const mood = d.stress > 65 ? '<span style="color:#ff9080">agitated</span>' : d.stress > 40 ? 'uneasy' : 'calm';
    return `<b>${d.name}</b> · ${d.sp.name}${d.growth < 1 ? ' (hatchling)' : d.elderly ? ' (elderly)' : ''}<br>${status} · ${mood}<br>Hunger ${Math.round(d.hunger)}% · Health ${Math.round(d.hp / d.sp.hp * 100)}%${worst && !d.isPtera ? `<br><span style="color:#f0c060">Problem: ${lbl[worst]}</span>` : ''}`;
  }

  doUndo() {
    const r = this.game.undo();
    if (!r) { this.toast('Nothing to undo.', 'info'); this.sfx.play('error'); return; }
    this.toast(`Undid ${r.label}${r.refund ? ` (${r.refund > 0 ? '+' : '−'}${fmtMoney(Math.abs(r.refund))})` : ''}`, 'good');
    this.sfx.play('demolish');
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
    const ra = g.research.active;
    $('#resStat').hidden = !ra;
    if (ra) { const r = RESEARCH_BY_ID[ra.id]; $('#resK').textContent = (g.research.labs() ? '' : 'NO LAB POWER · ') + r.name; $('#resbar i').style.width = (ra.progress / r.hours * 100) + '%'; }
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
    msg = forTouch(msg);
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
    if (hl) {
      // bad news bumps whatever is scrolling; good news waits its turn and stays quiet in a crisis
      const g = this.game, now = performance.now(), bad = e.type === 'bad';
      const crisis = g.looseCount > 0 || g.alarm || g.world.power.outage > 0;
      if (bad ? !this.tickerBadT || now - this.tickerBadT > 15000 : (!this.tickerT || now - this.tickerT > 15000) && !crisis) {
        this.tickerT = now; if (bad) this.tickerBadT = now;
        this.showTicker(hl);
      }
    }
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
      // chapter 1 keeps the beginner's step-by-step hints from the classic goal list
      const hint = sc.hints ? GOALS.slice(0, 12).find((x) => !x.check(g)) : null;
      this.hintId = hint ? hint.id : null;
      el.innerHTML = `<b>${sc.campaign ? 'CHAPTER ' + sc.chapter : 'SCENARIO'} · ${sc.name.toUpperCase()}</b>${sc.objective}<br><span style="color:var(--gold)" id="scProg">${sc.progress(g)}</span>${hint ? `<br><span style="color:var(--dim)">Next step: ${hint.text}</span>${hint.tool ? ' <button id="goalBtn" style="font-size:15px;padding:0 6px;margin-left:4px">Show me</button>' : ''}` : ''}`;
      if (hint && hint.tool) $('#goalBtn').onclick = () => this.showMeTool(hint.tool);
      return;
    }
    if (g.goalIdx >= GOALS.length) { el.innerHTML = `<b>ALL GOALS COMPLETE</b>Spared no expense. Keep building!`; return; }
    const goal = GOALS[g.goalIdx];
    // eggs already in: say so, or the Show me button invites buying a second clutch
    if (goal.id === 'hatch1' && g.eggs.length) {
      const e = g.eggs.reduce((a, b) => (b.t < a.t ? b : a));
      const hrs = Math.max(1, Math.ceil(e.t / g.secPerHour));
      this.eggGoalKey = hrs;
      el.innerHTML = `<b>GOAL ${g.goalIdx + 1}/${GOALS.length}</b>${goal.text}<br><span style="color:var(--dim)">${g.eggs.length > 1 ? g.eggs.length + ' eggs' : 'Egg'} incubating: hatches in ~${hrs}h of game time.</span> <button id="goalBtn" style="font-size:15px;padding:0 6px;margin-left:4px">Show me</button>`;
      $('#goalBtn').onclick = () => { this.sfx.play('click'); this.centerOn(e.x, e.y); };
      return;
    }
    this.eggGoalKey = null;
    el.innerHTML = `<b>GOAL ${g.goalIdx + 1}/${GOALS.length}</b>${goal.text}${goal.reward ? ` <span style="color:var(--gold)">+${fmtMoney(goal.reward)}</span>` : ''}${' <button id="goalBtn" style="font-size:15px;padding:0 6px;margin-left:4px">' + (goal.tool ? 'Show me' : 'How?') + '</button>'}`;
    const gb = $('#goalBtn');
    // goals with no single tool (stars, crowds, cash) open the breakdown of what drives them
    gb.onclick = () => { if (goal.tool) this.showMeTool(goal.tool); else if (goal.id === 'million') this.showFinances(); else this.showRating(); };
  }

  showMeTool(tool) {
    this.sfx.play('click');
    if (tool === 'hatch') { this.showSpeciesPicker(); return; }
    const grp = TOOL_GROUPS.find((gr) => gr.tools.includes(tool));
    this.setTool(tool);
    if (grp && window.innerWidth >= 760) this.openFlyout(grp);
    const tips = { paddock: 'Drag a rectangle inside the blue power coverage, then put a Feeder inside.', power: 'Place it near where your paddocks will go. Fences only work in its blue radius.', visitor: 'Build it next to a path, and connect paths from the gate past your paddocks.' };
    this.toast(tips[tool] || `Selected: ${(BUILDINGS[tool] || TOOL_INFO[tool]).name}. Click on the map to place it.`, 'info');
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
    $('#sDay').textContent = 'Day ' + g.day + ' · ' + g.season.name;
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
    const over = p.demand > p.supply && p.demand > 0;
    $('#sPowK').textContent = p.outage > 0 ? 'GRID DOWN' : over ? 'OVERLOADED' : 'Power';
    $('#sPowK').style.color = p.outage > 0 ? '#e04838' : over ? '#f0a030' : '';
    $('#alarmBtn').classList.toggle('on', g.alarm);
    this.renderAlerts();
    if (g.scenario) {
      const sc = SCENARIOS[g.scenario], p = $('#scProg');
      if (p) p.textContent = sc.progress(g);
      if (sc.hints) { const h = GOALS.slice(0, 12).find((x) => !x.check(g)); if ((h ? h.id : null) !== this.hintId) this.renderGoal(); }
    } else if (this.speed === 0) g.checkGoals(); // building while paused still ticks goals off
    if (!g.scenario && GOALS[g.goalIdx] && GOALS[g.goalIdx].id === 'hatch1') {
      const k = g.eggs.length ? Math.max(1, Math.ceil(Math.min(...g.eggs.map((e) => e.t)) / g.secPerHour)) : null;
      if (k !== this.eggGoalKey) this.renderGoal();
    }
    this.infoT -= 0.2;
    if (this.infoT <= 0) { this.infoT = 0.4; this.renderInfo(false); }
    this.renderDeals(false);
    this.renderSquad(false);
    // an InGen call waits until no other dialog is open
    // the board can wait while the park is in crisis
    const crisis = g.looseCount > 0 || g.alarm || g.world.power.outage > 0 || (g.events.storm && g.events.storm.phase === 'active');
    if (g.pendingDecision && !this.decisionShown && !crisis && $('#modal').style.display !== 'flex') this.showDecision(g.pendingDecision);
  }

  renderAlerts() {
    const g = this.game, w = g.world;
    const out = [];
    const loose = g.creatures().filter((d) => d.loose && !d.carried);
    if (loose.length) {
      const sed = loose.filter((d) => d.sedatedT > 0).length;
      const canRespond = g.hasBuilding('ranger') || g.helis.length;
      out.push(`<div class="alert red" data-act="loose">⚠ ${loose.length} DINOSAUR${loose.length > 1 ? 'S' : ''} LOOSE${sed ? ` (${sed} SEDATED)` : ''}${!g.hasBuilding('ranger') ? ' · NO RANGERS!' : ''}${canRespond && loose.length > sed ? ' <button data-act="respond" title="Alarm on, every ranger and helicopter on the escapees (B)">RESPOND</button>' : ''}</div>`);
    }
    if (w.power.outage > 0) {
      const backup = g.hasBuilding('backup') && w.power.backupFuel > 0 && w.power.backupDelay <= 0;
      out.push(`<div class="alert red" data-act="none">⚡ GRID DOWN ${Math.ceil(w.power.outage)}s${backup ? ' · BACKUP ON' : ''} <button data-act="reboot">Manual restart $15k</button></div>`);
    }
    if (w.power.outage <= 0 && w.power.demand > w.power.supply && !g.buildingsOfType('power').some((b) => b.offline > 0)) {
      const dark = [...w.buildings.values()].filter((b) => BUILDINGS[b.type].power && !b.powered && (w.covered[w.idx(b.x, b.y)] || w.covered[w.idx(b.x + b.w - 1, b.y + b.h - 1)])).length;
      if (dark) out.push(`<div class="alert amber" data-act="none">⚡ POWER SHORTAGE · ${dark} BUILDING${dark > 1 ? 'S' : ''} DARK · needs ${Math.round(w.power.demand)} MW, has ${Math.round(w.power.supply)} <button data-act="buildpower">Build Power Plant</button></div>`);
    }
    const s = g.events.storm;
    if (s) out.push(`<div class="alert ${s.phase === 'active' ? 'blue' : 'amber'}">⛈ ${s.phase === 'warn' ? 'STORM INCOMING ' + Math.ceil(s.t) + 's' : 'TROPICAL STORM ' + Math.ceil(s.t) + 's'}</div>`);
    const v = g.events.volcano;
    if (v) out.push(`<div class="alert ${v.phase === 'erupt' ? 'red' : 'amber'}" data-act="volcano">🌋 ${v.phase === 'warn' ? 'ERUPTION IN ' + Math.ceil(v.t) + 's' : 'ERUPTION!'}</div>`);
    if (g.events.quake > 0) out.push(`<div class="alert red">〰 EARTHQUAKE</div>`);
    const broken = g.countBrokenFences();
    if (broken >= 3 || (broken && !g.hasBuilding('maint'))) out.push(`<div class="alert amber" data-act="none">🔧 ${broken} BREACHED FENCE${broken > 1 ? 'S' : ''} <button data-act="fixall" title="Instantly repair every damaged fence">Repair all ${fmtMoney(g.repairAllCost().cost)}</button></div>`);
    const starving = g.hungryWithoutFood().filter((x) => x.d.hunger > 75);
    if (starving.length) out.push(`<div class="alert red" data-act="starve">🍖 ${starving.length} DINOSAUR${starving.length > 1 ? 'S' : ''} STARVING (NO FEEDER) <button data-act="feedall">Add feeders</button></div>`);
    if (g.alarm) out.push(`<div class="alert amber" data-act="alarm">🔊 EVACUATION IN PROGRESS · ${IS_TOUCH ? 'tap' : 'click'} to sound all-clear</div>`);
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
          else if (a === 'respond') this.respond();
          else if (a === 'fixall') { const r = g.repairFences(); this.toast(r.n ? `Repaired ${r.n} fence tiles (${fmtMoney(r.cost)}).${r.blocked ? ` ${r.blocked} gaps have a dinosaur in them.` : ''}` : r.blocked ? 'Dinosaurs are standing in the gaps. Sedate them first.' : 'Not enough money.', r.n ? 'good' : 'warn'); if (r.n) this.sfx.play('build'); }
          else if (a === 'reboot') {
            if (g.world.power.outage > 8 && g.canAfford(15000)) { g.spend(15000, 'ops'); g.world.power.outage = 8; g.log('Engineers are rebooting the grid manually...', 'info', null, true); }
          } else if (a === 'volcano') this.centerOn(w.volcano.x, w.volcano.y);
          else if (a === 'alarm') this.toggleAlarm();
          else if (a === 'buildpower') this.showMeTool('power');
          else if (a === 'feedall') { const n = g.replaceLostFeeders(true); this.toast(n ? `Added ${n} feeder${n > 1 ? 's' : ''}.` : 'No room for a feeder (or not enough money).', n ? 'good' : 'warn'); if (n) this.sfx.play('build'); }
          else if (a === 'starve') { const x = g.hungryWithoutFood()[0]; if (x) this.centerOn(x.d.tx, x.d.ty); }
        };
      }
    }
  }

  // ---------------- per-frame ----------------
  frame(dt) {
    if (this.slowmo > 0) this.slowmo -= dt;
    // building while paused: keep power and paddocks up to date so hatching, feeders and fence lights work
    const w = this.game.world;
    if (this.speed === 0 || $('#modal').style.display === 'flex') {
      if (w.regionsDirty) { w.computeRegions(); this.game.updateLoose(); }
      if (w.powerDirty) w.computePower(this.game);
    }
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

// Phones and tablets: tap, not click, and no keyboard hints
const IS_TOUCH = typeof window !== 'undefined' && window.matchMedia && window.matchMedia('(pointer: coarse)').matches;
function forTouch(msg) {
  if (!IS_TOUCH || typeof msg !== 'string') return msg;
  return msg.replace(/\bClick\b/g, 'Tap').replace(/\bclick\b/g, 'tap').replace(/,? ?\(?Esc to (stop|cancel)\)?/g, '');
}
