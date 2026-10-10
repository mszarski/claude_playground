// ---------- Rendering ----------
'use strict';

class Renderer {
  constructor(canvas, game) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.game = game;
    this.cam = { x: 0, y: 0, zoom: 3 };
    const W = game.world.W * TILE, H = game.world.H * TILE;
    this.cache = [makeCanvas(W, H), makeCanvas(W, H)];
    [this.lightC, this.lightX] = makeCanvas(16, 16);
    this.time = 0;
    this.rain = [];
    for (let i = 0; i < 260; i++) this.rain.push({ x: Math.random(), y: Math.random(), s: 0.6 + Math.random() * 0.6 });
    this.overlay = null; // 'power' | 'paddock'
  }

  resize(w, h) {
    this.canvas.width = w; this.canvas.height = h;
    this.lightC.width = w; this.lightC.height = h;
    this.ctx.imageSmoothingEnabled = false;
  }

  setGame(game) {
    this.game = game;
    game.world.dirtyTerrain = true;
  }

  // ---------- terrain cache ----------
  redrawTerrain() {
    const w = this.game.world;
    for (let f = 0; f < 2; f++) this.drawTerrainFrame(this.cache[f][1], f, 0, 0, w.W - 1, w.H - 1);
    w.dirtyTerrain = false;
    w.dirtyRects = [];
  }

  redrawDirty() {
    const w = this.game.world;
    const rects = w.dirtyRects;
    w.dirtyRects = [];
    if (rects.length > 120) { this.redrawTerrain(); return; }
    for (const [x, y, rw, rh] of rects) {
      // include neighbours: autotiled paths/fences/foam depend on them; trees overlap the row above
      const x0 = Math.max(0, x - 1), y0 = Math.max(0, y - 1), x1 = Math.min(w.W - 1, x + rw), y1 = Math.min(w.H - 1, y + rh);
      for (let f = 0; f < 2; f++) {
        const ctx = this.cache[f][1];
        ctx.save();
        ctx.beginPath(); ctx.rect(x0 * TILE, y0 * TILE, (x1 - x0 + 1) * TILE, (y1 - y0 + 1) * TILE); ctx.clip();
        this.drawTerrainFrame(ctx, f, x0, y0, x1, y1);
        ctx.restore();
      }
    }
  }

  drawTerrainFrame(ctx, frame, bx0, by0, bx1, by1) {
    const w = this.game.world;
    const W = w.W, H = w.H;
    ctx.imageSmoothingEnabled = false;
    for (let y = by0; y <= by1; y++) for (let x = bx0; x <= bx1; x++) {
      const i = w.idx(x, y), t = w.terrain[i], v = w.variant[i];
      const px = x * TILE, py = y * TILE;
      let img;
      switch (t) {
        case T_DEEP: img = TILES.deep[frame][v % 3]; break;
        case T_WATER: img = TILES.water[frame][v % 3]; break;
        case T_SAND: img = TILES.sand[v % 4]; break;
        case T_ROCK: img = TILES.rock[v % 4]; break;
        case T_LAVA: img = TILES.lava[frame][v % 3]; break;
        case T_BASALT: img = TILES.basalt[v % 3]; break;
        case T_VOLCANO: img = TILES.volcano[v % 3]; break;
        default: img = TILES.grass[v % 6];
      }
      ctx.drawImage(img, px, py);
      // shoreline foam
      if (t === T_WATER || t === T_DEEP) {
        ctx.fillStyle = frame ? '#cfeaf8' : '#a8d8f0';
        for (const [dx, dy] of DIRS4) {
          const nt = w.terrainAt(x + dx, y + dy);
          if (nt === T_WATER || nt === T_DEEP) continue;
          if (dx === 1) ctx.fillRect(px + 15 - frame, py, 1, 16);
          if (dx === -1) ctx.fillRect(px + frame, py, 1, 16);
          if (dy === 1) ctx.fillRect(px, py + 15 - frame, 16, 1);
          if (dy === -1) ctx.fillRect(px, py + frame, 16, 1);
        }
        if (t === T_DEEP) {
          ctx.fillStyle = 'rgba(58,136,200,0.5)';
          for (const [dx, dy] of DIRS4) {
            if (w.terrainAt(x + dx, y + dy) !== T_WATER) continue;
            if (dx === 1) ctx.fillRect(px + 12, py, 4, 16);
            if (dx === -1) ctx.fillRect(px, py, 4, 16);
            if (dy === 1) ctx.fillRect(px, py + 12, 16, 4);
            if (dy === -1) ctx.fillRect(px, py, 16, 4);
          }
        }
      } else if (t === T_SAND || t === T_GRASS || t === T_FOREST) {
        // sandy edges where grass meets water
        if (t !== T_SAND) {
          ctx.fillStyle = '#d8c080';
          for (const [dx, dy] of DIRS4) {
            const nt = w.terrainAt(x + dx, y + dy);
            if (nt !== T_WATER && nt !== T_DEEP) continue;
            if (dx === 1) ctx.fillRect(px + 14, py, 2, 16);
            if (dx === -1) ctx.fillRect(px, py, 2, 16);
            if (dy === 1) ctx.fillRect(px, py + 14, 16, 2);
            if (dy === -1) ctx.fillRect(px, py, 16, 2);
          }
        }
      }
      if (t === T_ROCK) {
        const below = w.terrainAt(x, y + 1);
        if (below !== T_ROCK && below !== T_VOLCANO) {
          ctx.fillStyle = '#4a453c'; ctx.fillRect(px, py + 11, 16, 5);
          ctx.fillStyle = '#5c564a'; for (let k = 0; k < 16; k += 3) ctx.fillRect(px + k, py + 11, 1, 5);
        }
        const above = w.terrainAt(x, y - 1);
        if (above !== T_ROCK && above !== T_VOLCANO) { ctx.fillStyle = '#a8a294'; ctx.fillRect(px, py, 16, 1); }
      }
      if (t === T_LAVA) {
        ctx.fillStyle = frame ? 'rgba(248,200,64,0.6)' : 'rgba(248,140,40,0.4)';
        ctx.fillRect(px + ((v + frame * 5) % 12), py + ((v >> 2) % 12), 3, 2);
      }
      // elevation: lighter uplands, darker lowlands, a step edge where the land drops a band
      if (t === T_GRASS || t === T_FOREST || t === T_SAND || t === T_BASALT) {
        const hh = w.height[i];
        if (hh >= 4) { ctx.fillStyle = `rgba(255,250,200,${(hh - 3) * 0.035})`; ctx.fillRect(px, py, 16, 16); }
        else if (hh <= 1) { ctx.fillStyle = `rgba(0,30,20,${(2 - hh) * 0.05})`; ctx.fillRect(px, py, 16, 16); }
        if (y + 1 < H) {
          const hb = w.height[i + W], tb = w.terrain[i + W];
          if (hb < hh && (tb === T_GRASS || tb === T_FOREST || tb === T_SAND)) {
            ctx.fillStyle = 'rgba(30,50,20,0.35)'; ctx.fillRect(px, py + 14, 16, 2);
            ctx.fillStyle = 'rgba(255,255,220,0.12)'; ctx.fillRect(px, py + 13, 16, 1);
          }
        }
        if (x + 1 < W && w.height[i + 1] < hh && (w.terrain[i + 1] === T_GRASS || w.terrain[i + 1] === T_FOREST)) { ctx.fillStyle = 'rgba(30,50,20,0.2)'; ctx.fillRect(px + 15, py, 1, 16); }
      }
      // paths
      if (w.path[i]) this.drawPath(ctx, x, y, px, py, v);
      if (w.track[i]) this.drawTrack(ctx, x, y, px, py);
    }
    // fences & trees row by row so taller objects overlap correctly
    const fx0 = Math.max(0, bx0 - 1), fx1 = Math.min(W - 1, bx1 + 1);
    for (let y = by0; y <= Math.min(H - 1, by1 + 1); y++) {
      for (let x = fx0; x <= fx1; x++) {
        const i = w.idx(x, y);
        if (w.fence[i]) this.drawFence(ctx, x, y, frame);
      }
      for (let x = fx0; x <= fx1; x++) {
        const i = w.idx(x, y);
        if (w.terrain[i] === T_FOREST && !w.bld[i]) {
          const v = w.variant[i];
          // Mix of full trees and undergrowth so the jungle reads well and doesn't hide everything
          if ((v % 7) < 4) {
            const tree = TILES.trees[v % TILES.trees.length];
            const ox = ((v >> 3) % 5) - 2;
            ctx.drawImage(tree, x * TILE + 8 - 10 + ox, y * TILE + 14 - 26);
          } else {
            ctx.drawImage(TILES.bush[v % 3], x * TILE + 2 + ((v >> 1) % 3), y * TILE + 5);
            ctx.drawImage(TILES.bush[(v + 1) % 3], x * TILE + ((v >> 2) % 5), y * TILE + 1);
          }
        }
      }
    }
  }

  drawPath(ctx, x, y, px, py, v) {
    const w = this.game.world;
    ctx.fillStyle = '#c8aa7a'; ctx.fillRect(px, py, 16, 16);
    ctx.fillStyle = '#b8986a';
    for (let k = 0; k < 6; k++) ctx.fillRect(px + ((v * (k + 3)) % 15), py + ((v * (k + 7)) % 15), 1, 1);
    ctx.fillStyle = '#d8bc8e';
    for (let k = 0; k < 4; k++) ctx.fillRect(px + ((v * (k + 11)) % 15), py + ((v * (k + 5)) % 15), 1, 1);
    const isP = (dx, dy) => w.inb(x + dx, y + dy) && (w.path[w.idx(x + dx, y + dy)] || w.bld[w.idx(x + dx, y + dy)]);
    ctx.fillStyle = '#8a7048';
    if (!isP(0, -1)) ctx.fillRect(px, py, 16, 1);
    if (!isP(0, 1)) ctx.fillRect(px, py + 15, 16, 1);
    if (!isP(-1, 0)) ctx.fillRect(px, py, 1, 16);
    if (!isP(1, 0)) ctx.fillRect(px + 15, py, 1, 16);
  }

  drawTrack(ctx, x, y, px, py) {
    const w = this.game.world;
    const isT = (dx, dy) => w.inb(x + dx, y + dy) && (w.track[w.idx(x + dx, y + dy)] || (w.bld[w.idx(x + dx, y + dy)] && w.buildings.get(w.bld[w.idx(x + dx, y + dy)]).type === 'tour'));
    const L = isT(-1, 0), R = isT(1, 0), U = isT(0, -1), D = isT(0, 1);
    ctx.fillStyle = '#4f8a32'; ctx.fillRect(px, py, 16, 16);
    ctx.fillStyle = '#5a5a58';
    ctx.fillRect(px + 3, py + 3, 10, 10);
    if (L) ctx.fillRect(px, py + 3, 3, 10);
    if (R) ctx.fillRect(px + 13, py + 3, 3, 10);
    if (U) ctx.fillRect(px + 3, py, 10, 3);
    if (D) ctx.fillRect(px + 3, py + 13, 10, 3);
    ctx.fillStyle = '#6a6a66';
    ctx.fillRect(px + 4, py + 4, 8, 1);
    // guide rail slot
    ctx.fillStyle = '#2a2a28';
    if (L) ctx.fillRect(px, py + 8, 8, 1);
    if (R) ctx.fillRect(px + 8, py + 8, 8, 1);
    if (U) ctx.fillRect(px + 8, py, 1, 8);
    if (D) ctx.fillRect(px + 8, py + 8, 1, 8);
    ctx.fillStyle = '#f8d040';
    if (L || R) { ctx.fillRect(px + 2, py + 5, 2, 1); ctx.fillRect(px + 11, py + 5, 2, 1); }
    if (U || D) { ctx.fillRect(px + 5, py + 2, 1, 2); ctx.fillRect(px + 5, py + 11, 1, 2); }
  }

  drawFence(ctx, x, y, frame) {
    const w = this.game.world;
    const i = w.idx(x, y);
    const f = w.fence[i];
    const px = x * TILE, py = y * TILE;
    const conn = (dx, dy) => {
      if (!w.inb(x + dx, y + dy)) return false;
      const j = w.idx(x + dx, y + dy);
      return isSolidFence(w.fence[j]) || (w.bld[j] && f !== F_BROKEN);
    };
    const L = conn(-1, 0), R = conn(1, 0), U = conn(0, -1), D = conn(0, 1);
    if (f === F_ELECTRIC) {
      const hp = w.fenceHp[i] / FENCE_DEF[F_ELECTRIC].hp;
      const pw = w.fencePowered[i];
      const wire = pw ? (frame ? '#f8f080' : '#f8d838') : '#6a6a62';
      const lean = hp < 0.4 ? 1 : 0;
      // wires
      ctx.fillStyle = wire;
      if (L) { ctx.fillRect(px, py + 5, 8, 1); ctx.fillRect(px, py + 9, 8, 1); }
      if (R) { ctx.fillRect(px + 8, py + 5 + lean, 8, 1); ctx.fillRect(px + 8, py + 9 + lean, 8, 1); }
      if (U) { ctx.fillRect(px + 6, py, 1, 8); ctx.fillRect(px + 9, py, 1, 8); }
      if (D) { ctx.fillRect(px + 6, py + 6, 1, 10); ctx.fillRect(px + 9, py + 6, 1, 10); }
      // post
      ctx.fillStyle = '#26262a'; ctx.fillRect(px + 7 + lean, py + 1, 2, 12);
      ctx.fillStyle = '#5a5a60'; ctx.fillRect(px + 7 + lean, py + 1, 1, 12);
      ctx.fillStyle = pw ? '#f8d040' : '#8a8a80'; ctx.fillRect(px + 7 + lean, py + 1, 2, 2);
      ctx.fillStyle = 'rgba(0,0,0,0.25)'; ctx.fillRect(px + 7, py + 13, 4, 2);
      if (hp < 0.6) { ctx.fillStyle = '#c84020'; ctx.fillRect(px + 7 + lean, py + 6, 1, 2); }
    } else if (f === F_WALL) {
      const hp = w.fenceHp[i] / FENCE_DEF[F_WALL].hp;
      const top = '#b4b4ac', face = '#7a7a74', dark = '#4a4a46';
      const x0 = L ? 0 : 2, x1 = R ? 16 : 14, y0 = U ? 0 : 1, y1 = D ? 16 : 12;
      ctx.fillStyle = dark; ctx.fillRect(px + x0, py + y0, x1 - x0, y1 - y0 + 3);
      ctx.fillStyle = top; ctx.fillRect(px + x0, py + y0, x1 - x0, y1 - y0 - (D ? 0 : 3));
      if (!D) { ctx.fillStyle = face; ctx.fillRect(px + x0, py + y1 - 3, x1 - x0, 5); }
      ctx.fillStyle = '#9a9a92';
      for (let k = 2; k < 16; k += 5) ctx.fillRect(px + k, py + y0 + 3, 1, 1);
      // hazard stripe
      if (!D) { for (let k = x0; k < x1; k += 4) { ctx.fillStyle = '#e8c020'; ctx.fillRect(px + k, py + y1, 2, 1); } }
      if (hp < 0.5) { ctx.fillStyle = '#2a2a28'; ctx.fillRect(px + 5, py + 3, 1, 3); ctx.fillRect(px + 6, py + 5, 2, 1); ctx.fillRect(px + 10, py + 6, 1, 3); }
    } else if (f === F_GATE) {
      // heavy wooden gate frame straddling the track
      const vertical = L || R ? false : true; // fence runs left-right -> gate posts left/right
      const hp = w.fenceHp[i] / FENCE_DEF[F_GATE].hp;
      ctx.fillStyle = '#2a1a0c';
      if (!vertical) { ctx.fillRect(px, py - 6, 3, 18); ctx.fillRect(px + 13, py - 6, 3, 18); ctx.fillRect(px, py - 7, 16, 3); }
      else { ctx.fillRect(px, py, 16, 3); ctx.fillRect(px, py + 13, 16, 3); ctx.fillRect(px - 1, py - 2, 3, 18); }
      ctx.fillStyle = '#6a4a28';
      if (!vertical) { ctx.fillRect(px + 1, py - 5, 1, 16); ctx.fillRect(px + 14, py - 5, 1, 16); ctx.fillRect(px + 1, py - 6, 14, 1); }
      else { ctx.fillRect(px + 1, py + 1, 14, 1); ctx.fillRect(px + 1, py + 14, 14, 1); }
      ctx.fillStyle = '#f8d040';
      for (let k = 2; k < 14; k += 4) { if (!vertical) ctx.fillRect(px + k, py - 6, 2, 1); }
      if (hp < 0.5) { ctx.fillStyle = '#c84020'; ctx.fillRect(px + 6, py - 6, 3, 1); }
    } else if (f === F_BROKEN) {
      ctx.drawImage(RUBBLE_SPR, px, py + 8);
      ctx.fillStyle = '#26262a'; ctx.fillRect(px + 3, py + 6, 2, 6); ctx.fillRect(px + 5, py + 5, 1, 2);
      ctx.fillStyle = '#c84020'; ctx.fillRect(px + 9, py + 10, 4, 1);
    }
  }

  // ---------- main draw ----------
  draw(dt, ui) {
    const g = this.game, w = g.world;
    this.time += dt;
    if (w.dirtyTerrain) this.redrawTerrain();
    else if (w.dirtyRects && w.dirtyRects.length) this.redrawDirty();
    const ctx = this.ctx;
    const cw = this.canvas.width, ch = this.canvas.height;
    const z = this.cam.zoom;
    ctx.imageSmoothingEnabled = false;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = '#1a4a7a'; ctx.fillRect(0, 0, cw, ch);
    let sx = 0, sy = 0;
    if (g.shake > 0) { sx = Math.round(randf(-1, 1) * g.shake); sy = Math.round(randf(-1, 1) * g.shake); }
    const camX = Math.round(this.cam.x * z) / z, camY = Math.round(this.cam.y * z) / z;
    ctx.setTransform(z, 0, 0, z, Math.round(-camX * z) + sx, Math.round(-camY * z) + sy);
    const frame = Math.floor(this.time * 1.4) % 2;
    ctx.drawImage(this.cache[frame][0], 0, 0);

    // view bounds in tiles
    const vx0 = Math.floor(camX / TILE) - 3, vy0 = Math.floor(camY / TILE) - 3;
    const vx1 = Math.ceil((camX + cw / z) / TILE) + 3, vy1 = Math.ceil((camY + ch / z) / TILE) + 4;
    const vis = (x, y) => x >= vx0 && x <= vx1 && y >= vy0 && y <= vy1;

    // overlays beneath entities
    this.drawOverlays(ctx, ui, vx0, vy0, vx1, vy1);

    // dead electric fences blink a red warning light on each post
    if (Math.floor(this.time * 2) % 2 === 0) {
      ctx.fillStyle = '#ff3020';
      for (let y = Math.max(0, vy0); y <= Math.min(w.H - 1, vy1); y++) for (let x = Math.max(0, vx0); x <= Math.min(w.W - 1, vx1); x++) {
        const i = w.idx(x, y);
        if (w.fence[i] === F_ELECTRIC && !w.fencePowered[i]) ctx.fillRect(x * TILE + 7, y * TILE + 1, 2, 2);
      }
    }
    // fence sparkle
    if (w.power.ratio > 0) {
      ctx.fillStyle = '#ffffc0';
      for (let k = 0; k < 6; k++) {
        const x = randi(Math.max(0, vx0), Math.min(w.W - 1, vx1)), y = randi(Math.max(0, vy0), Math.min(w.H - 1, vy1));
        if (w.fencePowered[w.idx(x, y)] && w.fence[w.idx(x, y)] === F_ELECTRIC) ctx.fillRect(x * TILE + randi(0, 15), y * TILE + (chance(0.5) ? 5 : 9), 1, 1);
      }
    }

    // collect drawables
    const items = [];
    for (const b of w.buildings.values()) if (vis(b.x, b.y) || vis(b.x + b.w, b.y + b.h)) items.push({ y: b.y + b.h, t: 0, o: b });
    for (const d of g.dinos) if (!d.carried && vis(d.tx, d.ty)) items.push({ y: d.y, t: 1, o: d });
    for (const gu of g.guests) if (!gu.hidden && vis(gu.tx, gu.ty)) items.push({ y: gu.y, t: 2, o: gu });
    for (const s of g.staff) if (vis(s.tx, s.ty)) items.push({ y: s.y, t: 2, o: s });
    for (const j of g.jeeps) if (vis(j.tx, j.ty)) items.push({ y: j.y + 0.2, t: 5, o: j });
    for (const e of g.eggs) items.push({ y: e.y + 0.6, t: 3, o: e });
    if (w.volcano && vis(w.volcano.x, w.volcano.y)) items.push({ y: w.volcano.y + 3, t: 4, o: w.volcano });
    items.sort((a, b) => a.y - b.y);

    // shadows first
    ctx.fillStyle = 'rgba(0,0,0,0.25)';
    for (const it of items) {
      if (it.t === 1) { const d = it.o; const S0 = DINO_SPRITES[d.species]; const sw = (d.moving && d.vdir ? S0.up[0].width * 0.8 : S0.w * 0.7) * (d.growth < 1 ? d.growth : 1); ctx.fillRect(Math.round(d.x * TILE - sw / 2), Math.round(d.y * TILE + 2), Math.round(sw), 3); }
      else if (it.t === 2) ctx.fillRect(Math.round(it.o.x * TILE - 3), Math.round(it.o.y * TILE), 6, 2);
    }
    for (const it of items) {
      if (it.t === 0) this.drawBuilding(ctx, it.o, ui);
      else if (it.t === 1) this.drawDino(ctx, it.o, ui);
      else if (it.t === 2) this.drawPerson(ctx, it.o, ui);
      else if (it.t === 4) this.drawVolcano(ctx, it.o);
      else if (it.t === 5) this.drawJeep(ctx, it.o);
      else this.drawEgg(ctx, it.o);
    }

    // darts
    ctx.fillStyle = '#f0f0f0';
    for (const d of g.darts) { ctx.fillRect(Math.round(d.x * TILE), Math.round(d.y * TILE), 2, 1); }

    // particles (non-glow)
    for (const p of g.particles) {
      if (p.glow) continue;
      ctx.globalAlpha = clamp(p.life / (p.max || 1) * 1.5, 0, 1);
      ctx.fillStyle = p.color; ctx.fillRect(Math.round(p.x * TILE), Math.round(p.y * TILE), p.size, p.size);
    }
    ctx.globalAlpha = 1;
    // smoke from power plants
    for (const b of g.buildingsOfType('power')) {
      if (b.offline > 0 || w.power.outage > 0) continue;
      for (let k = 0; k < 5; k++) {
        const ph = (this.time * 0.6 + k / 5) % 1;
        ctx.globalAlpha = 0.45 * (1 - ph);
        ctx.fillStyle = '#e8e8e8';
        const r = 2 + ph * 5;
        ctx.fillRect(Math.round(b.x * TILE + 34 + Math.sin(ph * 6 + k) * 3 + ph * 8 - r / 2), Math.round((b.y + b.h) * TILE - BSPR.power.canvas.height - ph * 22 - r / 2), Math.round(r), Math.round(r));
      }
      ctx.globalAlpha = 1;
    }

    // pteranodons fly above everything on the ground
    for (const p of g.pteros) if (!p.carried && vis(p.tx, p.ty)) this.drawPtera(ctx, p, ui);
    // helicopters
    for (const h of g.helis) this.drawHeli(ctx, h);
    this.drawClouds(ctx, dt);
    this.drawFlyers(ctx, dt);

    // glow particles
    ctx.globalCompositeOperation = 'lighter';
    for (const p of g.particles) {
      if (!p.glow) continue;
      ctx.globalAlpha = clamp(p.life / (p.max || 1) * 2, 0, 1);
      ctx.fillStyle = p.color; ctx.fillRect(Math.round(p.x * TILE), Math.round(p.y * TILE), 1, 1);
    }
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = 'source-over';

    // lightning bolts
    for (const b of g.events.bolts) this.drawBolt(ctx, b, camY);

    // tool previews
    this.drawToolPreview(ctx, ui);

    // floating texts / thoughts
    ctx.font = '8px "Press Start 2P", monospace';
    ctx.textAlign = 'center';
    for (const gu of g.guests) {
      if (gu.thought && !gu.hidden && vis(gu.tx, gu.ty)) {
        const tx = Math.round(gu.x * TILE), ty = Math.round(gu.y * TILE - 16);
        this.bubble(ctx, gu.thought, tx, ty, gu.thought === 'AAAH!' ? '#c82020' : '#1a1a1a');
      }
    }
    for (const f of g.floats) {
      ctx.globalAlpha = clamp(f.life, 0, 1);
      ctx.fillStyle = '#000'; ctx.fillText(f.text, Math.round(f.x * TILE) + 1, Math.round(f.y * TILE) + 1);
      ctx.fillStyle = f.color; ctx.fillText(f.text, Math.round(f.x * TILE), Math.round(f.y * TILE));
    }
    ctx.globalAlpha = 1;

    // screen-space effects
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    this.drawLighting(ctx, cw, ch, camX, camY, z);
    this.drawWeather(ctx, cw, ch, dt);
  }

  bubble(ctx, text, x, y, col) {
    ctx.font = '5px monospace';
    const w = Math.max(10, text.length * 4 + 4);
    ctx.fillStyle = '#fff'; ctx.fillRect(x - w / 2, y - 6, w, 8);
    ctx.fillStyle = '#1a1a1a'; ctx.fillRect(x - w / 2, y + 2, w, 1); ctx.fillRect(x - 1, y + 2, 2, 2);
    drawText3(ctx, text.replace(/[^A-Z0-9!$+\- ]/gi, ''), Math.round(x - w / 2 + 2), y - 4, col);
  }

  drawOverlays(ctx, ui, vx0, vy0, vx1, vy1) {
    const g = this.game, w = g.world;
    const tool = ui.tool;
    const showPower = this.overlay === 'power' || ['power', 'pylon', 'backup', 'fence', 'paddock'].includes(tool);
    const showPaddock = this.overlay === 'paddock' || tool === 'hatch' || tool === 'feeder_h' || tool === 'feeder_c' || tool === 'movedest';
    if (showPower) {
      for (let y = Math.max(0, vy0); y <= Math.min(w.H - 1, vy1); y++) for (let x = Math.max(0, vx0); x <= Math.min(w.W - 1, vx1); x++) {
        const i = w.idx(x, y);
        if (w.covered[i]) {
          ctx.fillStyle = 'rgba(80,160,255,0.16)'; ctx.fillRect(x * TILE, y * TILE, TILE, TILE);
          ctx.fillStyle = 'rgba(140,200,255,0.75)';
          if (x > 0 && !w.covered[i - 1]) ctx.fillRect(x * TILE, y * TILE, 1, TILE);
          if (x < w.W - 1 && !w.covered[i + 1]) ctx.fillRect(x * TILE + TILE - 1, y * TILE, 1, TILE);
          if (y > 0 && !w.covered[i - w.W]) ctx.fillRect(x * TILE, y * TILE, TILE, 1);
          if (y < w.H - 1 && !w.covered[i + w.W]) ctx.fillRect(x * TILE, y * TILE + TILE - 1, TILE, 1);
        }
        if (w.fence[i] === F_ELECTRIC && !w.fencePowered[i]) { ctx.fillStyle = 'rgba(255,40,40,0.45)'; ctx.fillRect(x * TILE, y * TILE, TILE, TILE); }
      }
    }
    if (showPaddock) {
      for (let y = Math.max(0, vy0); y <= Math.min(w.H - 1, vy1); y++) for (let x = Math.max(0, vx0); x <= Math.min(w.W - 1, vx1); x++) {
        const r = w.regionAt(x, y);
        if (!r) continue;
        if (!r.public && r.size < 1200) { ctx.fillStyle = 'rgba(120,255,120,0.18)'; ctx.fillRect(x * TILE, y * TILE, TILE, TILE); }
      }
    }
    // selected dino's region outline
    const sel = ui.selected;
    if (sel && sel.kind === 'dino' && !sel.dead) {
      const rid = w.region[w.idx(sel.tx, sel.ty)];
      ctx.fillStyle = sel.loose ? 'rgba(255,60,60,0.14)' : 'rgba(255,255,255,0.10)';
      for (let y = Math.max(0, vy0); y <= Math.min(w.H - 1, vy1); y++) for (let x = Math.max(0, vx0); x <= Math.min(w.W - 1, vx1); x++) {
        if (w.region[w.idx(x, y)] === rid) ctx.fillRect(x * TILE, y * TILE, TILE, TILE);
      }
    }
  }

  drawBuilding(ctx, b, ui) {
    const g = this.game;
    const spr = BSPR[b.type];
    const def = BUILDINGS[b.type];
    const x = b.x * TILE, y = (b.y + b.h) * TILE - spr.canvas.height;
    if (b.type !== 'helipad') {
      ctx.fillStyle = 'rgba(0,0,0,0.22)';
      ctx.fillRect(x + spr.canvas.width, y + Math.min(spr.extra + 6, spr.canvas.height - 4), 4, spr.canvas.height - Math.min(spr.extra + 6, spr.canvas.height - 4));
      ctx.fillRect(x + 3, y + spr.canvas.height, spr.canvas.width, 2);
    }
    const lit = spr.night && g.darkness > 0.3 && (b.powered || !def.power);
    ctx.drawImage(lit ? spr.night : spr.canvas, x, y);
    this.animateBuilding(ctx, b, x, y, spr);
    // torches flicker on gate
    if (b.type === 'gate') {
      const fl = Math.sin(this.time * 20) > 0;
      ctx.fillStyle = fl ? '#f8e060' : '#f88020';
      ctx.fillRect(x + 6, y - 2, 2, 3); ctx.fillRect(x + spr.canvas.width - 8, y - 2, 2, 3);
    }
    if (b.type === 'lagoon' && !b.escaped) this.drawMosasaur(ctx, b, x, y + spr.extra);
    if (b.type === 'hatchery' && g.eggs.length) {
      ctx.fillStyle = Math.sin(this.time * 6) > 0 ? '#f8f080' : '#e0c040';
      ctx.fillRect(x + 22, y + 1, 4, 1);
    }
    if (b.type === 'siren' && (g.alarm || g.sirenAlarm) && b.powered) {
      ctx.fillStyle = Math.sin(this.time * 12) > 0 ? '#ff4030' : '#601010';
      ctx.fillRect(x + 4, y + 1, 8, 5);
    }
    // damage
    if (b.hp < b.maxHp * 0.5) {
      ctx.fillStyle = '#1a1a1a';
      for (let k = 0; k < 6; k++) ctx.fillRect(x + ((b.id * 7 + k * 11) % (spr.canvas.width - 4)), y + 8 + ((b.id * 3 + k * 13) % (spr.canvas.height - 12)), 2, 1);
      if (Math.floor(this.time * 3 + b.id) % 4 === 0) { ctx.fillStyle = 'rgba(80,80,80,0.6)'; ctx.fillRect(x + spr.canvas.width / 2, y - 4 - (this.time * 10 % 8), 3, 3); }
    }
    const flash = Math.floor(this.time * 2) % 2 === 0;
    // status icons
    let icon = null;
    if (def.power && !b.powered) icon = 'power';
    else if (def.cat === 'guest' && !g.world.accessTiles(b).length) icon = 'path';
    if (icon && flash) {
      const ix = x + spr.canvas.width / 2 - 4, iy = y - 9;
      ctx.fillStyle = '#1a1a1a'; ctx.fillRect(ix - 1, iy - 1, 10, 10);
      ctx.fillStyle = icon === 'power' ? '#f8d040' : '#e04838'; ctx.fillRect(ix, iy, 8, 8);
      if (icon === 'power') drawText3(ctx, '!', ix + 3, iy + 1, '#1a1a1a');
      else drawText3(ctx, '!', ix + 3, iy + 1, '#fff');
    }
    if (ui.selected === b) {
      ctx.strokeStyle = '#f8f080'; ctx.lineWidth = 1;
      ctx.strokeRect(b.x * TILE + 0.5, b.y * TILE + 0.5, b.w * TILE - 1, b.h * TILE - 1);
    }
    if (b.inside > 0 && (def.shelter) && (g.alarm || g.looseCount)) {
      drawText3(ctx, String(b.inside), x + 2, y + 2, '#fff');
    }
  }

  animateBuilding(ctx, b, x, y, spr) {
    const t = this.time + b.id * 0.7;
    const on = b.powered || !BUILDINGS[b.type].power;
    switch (b.type) {
      case 'restaurant':
        if (!on) break;
        for (let k = 0; k < 3; k++) {
          const ph = (t * 0.5 + k / 3) % 1;
          ctx.globalAlpha = 0.5 * (1 - ph); ctx.fillStyle = '#e8e8e0';
          ctx.fillRect(Math.round(x + 25 + Math.sin(ph * 5 + k) * 2 + ph * 4), Math.round(y + 2 - ph * 14), 2 + Math.round(ph * 2), 2 + Math.round(ph * 2));
        }
        ctx.globalAlpha = 1;
        break;
      case 'ranger': {
        const f = Math.floor(t * 4) % 2;
        ctx.fillStyle = '#f8d040'; ctx.fillRect(x + 29, y + f, 3 + f, 3);
        ctx.fillStyle = '#c8381e'; ctx.fillRect(x + 29, y + 1 + f, 3 + f, 1);
        break;
      }
      case 'visitor': {
        const f = Math.floor(t * 2) % 2;
        ctx.fillStyle = '#e04838'; ctx.fillRect(x + 8 + f, y + spr.extra + 30, 3, 8); ctx.fillRect(x + spr.canvas.width - 11 - f, y + spr.extra + 30, 3, 8);
        break;
      }
      case 'helipad':
        if (Math.floor(t * 1.5) % 2) { ctx.fillStyle = '#ff5040'; for (const [px, py] of [[3, 3], [44, 3], [3, 44], [44, 44]]) ctx.fillRect(x + px, y + py, 2, 2); }
        break;
      case 'feeder_c':
        if (Math.floor(t * 1.3) % 3 === 0) { ctx.fillStyle = '#e8e4dc'; ctx.fillRect(x + 8, y + spr.extra + 3, 3, 2); }
        break;
      case 'shop':
        if (on && Math.floor(t * 3) % 2) { ctx.fillStyle = '#f8f080'; ctx.fillRect(x + 4, y + 4, 1, 1); ctx.fillRect(x + 27, y + 4, 1, 1); }
        break;
    }
  }

  drawDino(ctx, d, ui) {
    const S = DINO_SPRITES[d.species];
    const sleeping = d.sedatedT > 0 || (d.sleeping && d.state === 'idle');
    let img;
    const animRate = d.speed * 2.2;
    const fr = d.moving ? Math.floor(d.anim * animRate) % 2 : 0;
    if (sleeping) img = d.facing > 0 ? S.sleepR : S.sleepL;
    else if (d.flash > 0) img = (d.facing > 0 ? S.flashR : S.flashL)[fr];
    else if (d.sick > 0) img = (d.facing > 0 ? S.sickR : S.sickL)[fr];
    else if (d.moving && d.vdir && S.up) img = (d.vdir > 0 ? S.down : S.up)[fr];
    else img = (d.facing > 0 ? S.right : S.left)[fr];
    let lx = 0;
    if (d.lunge > 0) { d.lunge -= 1 / 60; lx = d.facing * 2; }
    const bob = d.moving && fr === 1 ? -1 : 0;
    const gs = d.growth === undefined || d.growth >= 1 ? 1 : Math.round(d.growth * 10) / 10;
    const dw = Math.round(img.width * gs), dh = Math.round(img.height * gs);
    const x = Math.round(d.x * TILE - dw / 2 + lx), y = Math.round(d.y * TILE - dh + 5 + bob + (sleeping ? 2 : 0));
    if (d.camouflaged) ctx.globalAlpha = 0.15 + Math.max(0, Math.sin(this.time * 1.3 + d.id)) * 0.2;
    if (gs === 1) ctx.drawImage(img, x, y); else ctx.drawImage(img, x, y, dw, dh);
    ctx.globalAlpha = 1;
    if (sleeping) {
      const t = this.time * 1.5 + d.id;
      const ph = t % 1;
      drawText3(ctx, 'Z', x + dw - 4 + Math.round(ph * 3), y - 2 - Math.round(ph * 6), '#ffffff');
    }
    if (d.loose && !sleeping && !d.camouflaged) {
      if (Math.floor(this.time * 4) % 2 === 0) {
        ctx.fillStyle = '#1a1a1a'; ctx.fillRect(x + dw / 2 - 2, y - 10, 5, 8);
        ctx.fillStyle = '#ff3020'; ctx.fillRect(x + dw / 2 - 1, y - 9, 3, 6);
        ctx.fillStyle = '#fff'; ctx.fillRect(x + dw / 2, y - 8, 1, 3); ctx.fillRect(x + dw / 2, y - 4, 1, 1);
      }
    }
    if (d.sick > 0 && Math.floor(this.time * 2 + d.id) % 3 === 0) {
      ctx.fillStyle = '#7ad04a'; ctx.fillRect(x + dw / 2 + 3, y - 3 - Math.round((this.time * 4) % 4), 2, 2);
    }
    if (ui.selected === d) {
      ctx.strokeStyle = '#f8f080'; ctx.lineWidth = 1;
      ctx.strokeRect(x - 1.5, y - 1.5, dw + 3, dh + 3);
    }
    // stress meter when very stressed
    if (!sleeping && d.stress > 70 && !d.loose) {
      ctx.fillStyle = Math.floor(this.time * 3) % 2 ? '#f8a020' : '#e04020';
      ctx.fillRect(x + dw / 2, y - 5, 1, 3); ctx.fillRect(x + dw / 2, y - 1, 1, 1);
    }
  }

  drawPerson(ctx, p, ui) {
    const fr = p.moving ? Math.floor(p.anim * 7) % 2 : 0;
    const img = p.sprite[fr];
    const x = Math.round(p.x * TILE - 3), y = Math.round(p.y * TILE - 9);
    ctx.drawImage(img, x, y);
    if (p.kind === 'ranger' && p.target) {
      // rifle
      ctx.fillStyle = '#2a2a2a'; ctx.fillRect(p.facing > 0 ? x + 4 : x - 3, y + 4, 5, 1);
    }
    if (p.kind === 'worker' && p.state === 'repair' && Math.floor(this.time * 8) % 2) {
      ctx.fillStyle = '#c8c8c8'; ctx.fillRect(p.facing > 0 ? x + 6 : x - 2, y + 3, 2, 2);
    }
    if (p.kind === 'guest' && p.state === 'flee') {
      ctx.fillStyle = '#fff'; ctx.fillRect(x + 2, y - 4, 1, 2); ctx.fillRect(x + 2, y - 1, 1, 1);
    }
    if (ui.selected === p) { ctx.strokeStyle = '#f8f080'; ctx.strokeRect(x - 1.5, y - 1.5, 9, 13); }
  }

  drawClouds(ctx, dt) {
    const w = this.game.world;
    if (!this.clouds) {
      this.clouds = [];
      const r = mulberry32(5);
      for (let i = 0; i < 7; i++) {
        const blobs = [];
        for (let k = 0; k < 6; k++) blobs.push([r() * 90 - 45, r() * 30 - 15, 18 + r() * 26]);
        this.clouds.push({ x: r() * w.W * TILE, y: r() * w.H * TILE, blobs, s: 4 + r() * 4 });
      }
    }
    const storm = this.game.events.storm;
    ctx.fillStyle = storm ? 'rgba(10,20,40,0.12)' : 'rgba(10,20,40,0.07)';
    for (const c of this.clouds) {
      c.x += c.s * dt * (storm ? 3 : 1);
      if (c.x > w.W * TILE + 80) c.x = -80;
      for (const [bx, by, br] of c.blobs) {
        // chunky pixel blobs
        const x = Math.round((c.x + bx) / 4) * 4, y = Math.round((c.y + by) / 4) * 4, rr = Math.round(br / 4) * 4;
        ctx.fillRect(x - rr, y - rr / 2, rr * 2, rr);
        ctx.fillRect(x - rr / 2, y - rr, rr, rr * 2);
      }
    }
  }

  // Purely decorative pterosaurs gliding over the island
  drawFlyers(ctx, dt) {
    const w = this.game.world;
    this.flyers = this.flyers || [];
    this.flyT = (this.flyT || 8) - dt;
    if (this.flyT <= 0 && this.game.darkness < 0.6) {
      this.flyT = randf(20, 45);
      const dir = chance(0.5) ? 1 : -1;
      const y0 = randf(4, w.H - 8) * TILE;
      const n = randi(1, 4);
      for (let k = 0; k < n; k++) this.flyers.push({ x: dir > 0 ? -40 - k * 22 : w.W * TILE + 40 + k * 22, y: y0 + randf(-20, 20), dir, vy: randf(-6, 6), ph: rand() * 6, alt: randf(40, 70) });
    }
    for (const f of this.flyers) {
      f.x += f.dir * 34 * dt; f.y += f.vy * dt; f.ph += dt * 5;
      const fr = Math.sin(f.ph) > 0 ? 0 : 1;
      const img = (f.dir > 0 ? PTERO.R : PTERO.L)[fr];
      ctx.globalAlpha = 0.18; ctx.fillStyle = '#000';
      ctx.fillRect(Math.round(f.x - 6), Math.round(f.y + f.alt), 12, 2);
      ctx.globalAlpha = 1;
      ctx.drawImage(img, Math.round(f.x - img.width / 2), Math.round(f.y));
    }
    this.flyers = this.flyers.filter((f) => f.x > -100 && f.x < w.W * TILE + 100);
  }

  drawMosasaur(ctx, b, x, y) {
    const t = this.time + b.id * 3;
    const W = b.w * TILE;
    if (b.leap > 0) {
      // breach: arc out of the water
      const ph = 1 - b.leap / 2.2;
      const cx = x + W * (0.25 + ph * 0.5), cy = y + 46 - Math.sin(ph * Math.PI) * 34;
      ctx.save(); ctx.translate(Math.round(cx), Math.round(cy)); ctx.rotate((ph - 0.5) * 1.6);
      ctx.fillStyle = '#1a2a2a'; ctx.fillRect(-21, -5, 42, 10);
      ctx.fillStyle = '#4a6a68'; ctx.fillRect(-20, -4, 40, 8);
      ctx.fillStyle = '#c8d0b8'; ctx.fillRect(-14, 2, 30, 2);
      ctx.fillStyle = '#4a6a68'; ctx.fillRect(14, -2, 10, 6); ctx.fillRect(-26, -2, 7, 3); ctx.fillRect(-30, -5, 5, 4); ctx.fillRect(-30, 1, 5, 4);
      ctx.fillStyle = '#f4f0e0'; for (let k = 0; k < 4; k++) ctx.fillRect(16 + k * 2, 3, 1, 2);
      ctx.fillStyle = '#f8d040'; ctx.fillRect(17, -1, 2, 1);
      ctx.fillStyle = '#4a6a68'; ctx.fillRect(-6, 4, 6, 4); ctx.fillRect(6, 4, 6, 4);
      ctx.restore();
    } else {
      // dark shape cruising under the surface
      const cx = x + W / 2 + Math.cos(t * 0.5) * (W / 2 - 22), cy = y + 46 + Math.sin(t * 0.5) * 12;
      ctx.globalAlpha = 0.55; ctx.fillStyle = '#0a2440';
      ctx.fillRect(Math.round(cx - 16), Math.round(cy - 3), 32, 6); ctx.fillRect(Math.round(cx - 22), Math.round(cy - 1), 6, 2);
      ctx.globalAlpha = 1;
      if (Math.sin(t * 2) > 0.7) { ctx.fillStyle = '#a8d8f0'; ctx.fillRect(Math.round(cx + 10), Math.round(cy - 4), 3, 1); }
    }
  }

  drawPtera(ctx, p, ui) {
    const x = Math.round(p.x * TILE), y = Math.round(p.y * TILE);
    if (p.sedatedT > 0 && p.alt <= 0.05) {
      const S = DINO_SPRITES.ptera;
      ctx.drawImage(p.facing > 0 ? S.sleepR : S.sleepL, x - S.w / 2, y - S.h + 5);
      const ph = (this.time * 1.5 + p.id) % 1;
      drawText3(ctx, 'Z', x + 4 + Math.round(ph * 3), y - S.h - Math.round(ph * 6), '#ffffff');
    } else {
      const ay = Math.round(p.alt * TILE);
      ctx.fillStyle = 'rgba(0,0,0,0.22)'; ctx.fillRect(x - 6, y + 2, 12, 2);
      const fr = Math.sin(p.anim * (p.loose ? 9 : 6)) > 0 ? 0 : 1;
      const img = (p.facing > 0 ? PTERO.R : PTERO.L)[fr];
      ctx.drawImage(p.flash > 0 ? tintCanvas(img, '#fff', 0.8) : img, x - Math.floor(img.width / 2), y - ay - img.height);
      if (p.loose && Math.floor(this.time * 4) % 2 === 0) {
        ctx.fillStyle = '#ff3020'; ctx.fillRect(x - 1, y - ay - img.height - 7, 3, 5);
      }
    }
    if (ui.selected === p) { ctx.strokeStyle = '#f8f080'; ctx.strokeRect(x - 10.5, y - Math.round(p.alt * TILE) - 12.5, 21, 16); }
  }

  drawJeep(ctx, j) {
    const x = Math.round(j.x * TILE - 8), y = Math.round(j.y * TILE - 8);
    if (j.wrecked) { ctx.drawImage(JEEP_WRECK, x, y - 2); if (Math.floor(this.time * 3) % 2) { ctx.fillStyle = 'rgba(60,60,60,0.6)'; ctx.fillRect(x + 6, y - 6 - (this.time * 8 % 6), 3, 3); } return; }
    const bob = j.moving && Math.floor(j.anim * 8) % 2 ? 1 : 0;
    ctx.fillStyle = 'rgba(0,0,0,0.25)'; ctx.fillRect(x + 1, y + 9, 14, 2);
    ctx.drawImage(j.facing > 0 ? JEEP_R : JEEP_L, x, y + bob);
    // passengers' heads
    for (let k = 0; k < j.riders.length; k++) {
      ctx.fillStyle = ['#f0c8a0', '#b07850', '#d8a078', '#8a5a3a'][k % 4];
      ctx.fillRect(x + 4 + k * 2 + (j.facing > 0 ? 0 : 1), y + 1 + bob, 1, 2);
    }
    const g = this.game;
    if (j.state !== 'parked' && (!j.station.powered || g.world.power.outage > 0) && Math.floor(this.time * 2) % 2) {
      drawText3(ctx, '!', x + 6, y - 7, '#ff4030');
    }
  }

  drawVolcano(ctx, v) {
    const g = this.game;
    const x = Math.round((v.x + 0.5) * TILE - VOLCANO_SPR.width / 2), y = Math.round((v.y + 3) * TILE - VOLCANO_SPR.height);
    ctx.drawImage(VOLCANO_SPR, x, y);
    const ev = g.events.volcano;
    const hot = ev ? (ev.phase === 'erupt' ? 1 : 0.6) : 0.25;
    // crater flicker
    const fl = Math.sin(this.time * (ev ? 14 : 3)) * 0.5 + 0.5;
    ctx.fillStyle = `rgba(248,${Math.round(150 + fl * 80)},60,${0.4 + hot * 0.6})`;
    ctx.fillRect(x + VOLCANO_SPR.width / 2 - 6, y + 1, 12, 2);
    if (ev && ev.phase === 'erupt') {
      // lava fountain
      for (let k = 0; k < 14; k++) {
        const ph = (this.time * 1.8 + k / 14) % 1;
        const ang = (k * 2.39) % 1 - 0.5;
        const px = x + VOLCANO_SPR.width / 2 + ang * 40 * ph;
        const py = y - 30 * Math.sin(ph * Math.PI) + ph * 10;
        ctx.fillStyle = ph < 0.5 ? '#f8e070' : '#f86a20';
        ctx.fillRect(Math.round(px), Math.round(py), 2, 2);
      }
    }
    // smoke
    const n = ev ? 9 : 4;
    for (let k = 0; k < n; k++) {
      const ph = (this.time * (ev ? 0.35 : 0.12) + k / n) % 1;
      const r = 3 + ph * (ev ? 18 : 10);
      ctx.globalAlpha = (ev ? 0.55 : 0.35) * (1 - ph);
      ctx.fillStyle = ev && ev.phase === 'erupt' ? '#3a3030' : '#d8d0c8';
      ctx.fillRect(Math.round(x + VOLCANO_SPR.width / 2 + Math.sin(ph * 4 + k) * 5 + ph * 20 - r / 2), Math.round(y - ph * (ev ? 70 : 40) - r / 2), Math.round(r), Math.round(r));
    }
    ctx.globalAlpha = 1;
  }

  drawEgg(ctx, e) {
    const wob = e.t < 2.5 ? Math.round(Math.sin(this.time * 30)) : 0;
    ctx.drawImage(EGG_SPR, e.x * TILE + 4 + wob, e.y * TILE + 4);
    // progress
    ctx.fillStyle = '#1a1a1a'; ctx.fillRect(e.x * TILE + 2, e.y * TILE + 14, 12, 2);
    ctx.fillStyle = '#f8e060'; ctx.fillRect(e.x * TILE + 2, e.y * TILE + 14, Math.round(12 * (1 - e.t / e.max)), 2);
  }

  drawHeli(ctx, h) {
    const x = Math.round(h.x * TILE), y = Math.round(h.y * TILE);
    const ay = Math.round(h.alt * TILE);
    // shadow
    ctx.globalAlpha = 0.3;
    ctx.drawImage(HELI_SHADOW, x - 14, y - 4, 28, 8);
    ctx.globalAlpha = 1;
    const img = HELI_SPR;
    ctx.save();
    if (h.facing < 0) { ctx.translate(x, 0); ctx.scale(-1, 1); ctx.translate(-x, 0); }
    // cargo hangs below
    if (h.cargo) {
      const S = DINO_SPRITES[h.cargo.species];
      ctx.fillStyle = '#2a2a2a'; ctx.fillRect(x, y - ay + 4, 1, ay - S.h + 2);
      ctx.drawImage(S.sleepR, x - S.w / 2, y - S.h + 6);
    }
    ctx.drawImage(img, x - 14, y - ay - 10);
    // rotor
    const rl = Math.abs(Math.sin(h.rotor)) * 16 + 6;
    ctx.fillStyle = 'rgba(30,30,30,0.8)';
    ctx.fillRect(Math.round(x - rl), y - ay - 10, Math.round(rl * 2), 1);
    ctx.restore();
  }

  drawBolt(ctx, b, camY) {
    const r = mulberry32(b.seed);
    ctx.strokeStyle = 'rgba(255,255,220,' + clamp(b.life * 4, 0, 1) + ')';
    ctx.lineWidth = 2;
    ctx.beginPath();
    let x = b.x * TILE, y = camY - 10;
    ctx.moveTo(x, y);
    const ty = b.y * TILE;
    const steps = 10;
    for (let i = 1; i <= steps; i++) {
      const ny = y + (ty - y) * (i / steps);
      x = b.x * TILE + (i === steps ? 0 : (r() - 0.5) * 24);
      ctx.lineTo(x, ny);
    }
    ctx.stroke();
    ctx.lineWidth = 1;
  }

  drawToolPreview(ctx, ui) {
    const g = this.game, w = g.world;
    const h = ui.hover;
    if (!h) return;
    const tool = ui.tool;
    if (BUILDINGS[tool]) {
      const def = BUILDINGS[tool];
      const bx = h.x - Math.floor((def.w - 1) / 2), by = h.y - Math.floor((def.h - 1) / 2);
      const ok = w.canPlaceBuilding(tool, bx, by) && g.canAfford(def.cost);
      const spr = BSPR[tool];
      ctx.globalAlpha = 0.65;
      ctx.drawImage(spr.canvas, bx * TILE, (by + def.h) * TILE - spr.canvas.height);
      ctx.globalAlpha = 1;
      ctx.fillStyle = ok ? 'rgba(80,255,80,0.25)' : 'rgba(255,60,60,0.35)';
      ctx.fillRect(bx * TILE, by * TILE, def.w * TILE, def.h * TILE);
      if (def.range) {
        ctx.strokeStyle = 'rgba(120,200,255,0.8)';
        ctx.beginPath(); ctx.arc((bx + def.w / 2) * TILE, (by + def.h / 2) * TILE, def.range * TILE, 0, Math.PI * 2); ctx.stroke();
      }
      if (tool === 'viewing') {
        ctx.strokeStyle = 'rgba(255,240,120,0.6)';
        ctx.beginPath(); ctx.arc((bx + 1) * TILE, (by + 1) * TILE, def.view * TILE, 0, Math.PI * 2); ctx.stroke();
      }
      return;
    }
    if (ui.dragTiles && ui.dragTiles.length) {
      for (const [x, y] of ui.dragTiles) {
        ctx.fillStyle = tool === 'demolish' ? 'rgba(255,60,60,0.4)' : tool === 'path' || tool === 'route' ? 'rgba(232,200,140,0.6)' : tool === 'track' ? 'rgba(90,90,88,0.7)' : tool === 'trees' ? 'rgba(60,160,60,0.5)' : tool === 'clear' ? 'rgba(200,160,80,0.4)' : 'rgba(248,224,64,0.5)';
        ctx.fillRect(x * TILE, y * TILE, TILE, TILE);
      }
    }
    ctx.strokeStyle = tool === 'demolish' ? '#ff6050' : '#ffffff';
    ctx.lineWidth = 1;
    ctx.strokeRect(h.x * TILE + 0.5, h.y * TILE + 0.5, TILE - 1, TILE - 1);
    if (tool === 'hatch' && ui.species) {
      const S = DINO_SPRITES[ui.species];
      ctx.globalAlpha = 0.6;
      ctx.drawImage(S.right[0], h.x * TILE + 8 - S.w / 2, h.y * TILE + 13 - S.h);
      ctx.globalAlpha = 1;
    }
  }

  drawLighting(ctx, cw, ch, camX, camY, z) {
    const g = this.game, w = g.world;
    const dark = g.darkness;
    const storm = g.events.storm && g.events.storm.phase === 'active' ? 0.35 : g.events.storm ? 0.15 : 0;
    const v = g.events.volcano && g.events.volcano.phase === 'erupt' ? 0.2 : 0;
    const amount = Math.max(dark * 0.62, storm, v);
    if (amount > 0.01) {
      const lc = this.lightX;
      lc.globalCompositeOperation = 'source-over';
      lc.fillStyle = v > 0 && dark < 0.3 ? `rgba(60,20,10,${amount})` : `rgba(8,12,40,${amount})`;
      lc.clearRect(0, 0, cw, ch);
      lc.fillRect(0, 0, cw, ch);
      if (dark > 0.05) {
        lc.globalCompositeOperation = 'destination-out';
        const light = (wx, wy, r, a) => {
          const sx = (wx - camX) * z, sy = (wy - camY) * z, rr = r * z;
          if (sx < -rr || sy < -rr || sx > cw + rr || sy > ch + rr) return;
          const gr = lc.createRadialGradient(sx, sy, 0, sx, sy, rr);
          gr.addColorStop(0, `rgba(0,0,0,${a})`); gr.addColorStop(1, 'rgba(0,0,0,0)');
          lc.fillStyle = gr; lc.fillRect(sx - rr, sy - rr, rr * 2, rr * 2);
        };
        for (const b of w.buildings.values()) {
          if (!b.powered && BUILDINGS[b.type].power) continue;
          if (b.type === 'lamp') light((b.x + 0.5) * TILE, b.y * TILE, 70, 0.95);
          else if (BUILDINGS[b.type].power) light((b.x + b.w / 2) * TILE, (b.y + b.h / 2) * TILE, 26 + b.w * 12, 0.7);
          else if (b.type === 'gate') light((b.x + b.w / 2) * TILE, b.y * TILE, 40, 0.8);
        }
        for (let i = 0; i < w.W * w.H; i++) if (w.terrain[i] === T_LAVA) light((i % w.W + 0.5) * TILE, ((i / w.W | 0) + 0.5) * TILE, 22, 0.8);
        if (w.volcano) light((w.volcano.x + 0.5) * TILE, (w.volcano.y + 0.5) * TILE, 40, 0.8);
        for (const h of g.helis) if (h.state !== 'parked') light(h.x * TILE + h.facing * 30, h.y * TILE, 30, 0.9);
      }
      ctx.drawImage(this.lightC, 0, 0);
      // warm tint for lamps
      if (dark > 0.3) {
        ctx.globalCompositeOperation = 'lighter';
        for (const b of g.buildingsOfType('lamp')) {
          const sx = ((b.x + 0.5) * TILE - camX) * z, sy = (b.y * TILE + 4 - camY) * z;
          const gr = ctx.createRadialGradient(sx, sy, 0, sx, sy, 40 * z);
          gr.addColorStop(0, `rgba(255,200,100,${0.18 * dark})`); gr.addColorStop(1, 'rgba(0,0,0,0)');
          ctx.fillStyle = gr; ctx.fillRect(sx - 40 * z, sy - 40 * z, 80 * z, 80 * z);
        }
        ctx.globalCompositeOperation = 'source-over';
      }
    }
    // power outage tint: flashing red edge
    if (w.power.outage > 0 || g.looseCount > 0 || g.alarm) {
      const a = (Math.sin(this.time * 5) * 0.5 + 0.5) * 0.25;
      const gr = ctx.createRadialGradient(cw / 2, ch / 2, Math.min(cw, ch) * 0.35, cw / 2, ch / 2, Math.max(cw, ch) * 0.7);
      gr.addColorStop(0, 'rgba(255,0,0,0)'); gr.addColorStop(1, `rgba(255,0,0,${a})`);
      ctx.fillStyle = gr; ctx.fillRect(0, 0, cw, ch);
    }
  }

  drawWeather(ctx, cw, ch, dt) {
    const g = this.game;
    const s = g.events.storm;
    if (s && s.phase === 'active') {
      ctx.strokeStyle = 'rgba(180,200,255,0.5)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (const r of this.rain) {
        r.y += dt * 1.6 * r.s; r.x -= dt * 0.25 * r.s;
        if (r.y > 1) { r.y -= 1; r.x = Math.random() * 1.2; }
        if (r.x < 0) r.x += 1.2;
        const x = r.x * cw, y = r.y * ch;
        ctx.moveTo(x, y); ctx.lineTo(x - 4, y + 12);
      }
      ctx.stroke();
    } else if (s && s.phase === 'warn') {
      ctx.strokeStyle = 'rgba(180,200,255,0.25)';
      ctx.beginPath();
      for (let i = 0; i < 40; i++) { const r = this.rain[i]; r.y += dt * 1.2 * r.s; if (r.y > 1) r.y -= 1; ctx.moveTo(r.x * cw, r.y * ch); ctx.lineTo(r.x * cw - 3, r.y * ch + 8); }
      ctx.stroke();
    }
    if (g.events.lightning > 0) {
      ctx.fillStyle = `rgba(255,255,255,${g.events.lightning * 2})`;
      ctx.fillRect(0, 0, cw, ch);
    }
  }

  // ---------- breach cam (picture-in-picture) ----------
  drawPip(pctx, pw, ph, target, ui) {
    const g = this.game, w = g.world;
    const z = 3;
    const camX = Math.round(target.x * TILE - pw / z / 2), camY = Math.round(target.y * TILE - ph / z / 2);
    pctx.imageSmoothingEnabled = false;
    pctx.setTransform(1, 0, 0, 1, 0, 0);
    pctx.fillStyle = '#1a4a7a'; pctx.fillRect(0, 0, pw, ph);
    pctx.setTransform(z, 0, 0, z, -camX * z, -camY * z);
    pctx.drawImage(this.cache[Math.floor(this.time * 1.4) % 2][0], 0, 0);
    const x0 = camX / TILE - 3, y0 = camY / TILE - 3, x1 = (camX + pw / z) / TILE + 3, y1 = (camY + ph / z) / TILE + 4;
    const vis = (x, y) => x >= x0 && x <= x1 && y >= y0 && y <= y1;
    const items = [];
    for (const b of w.buildings.values()) if (vis(b.x, b.y) || vis(b.x + b.w, b.y + b.h)) items.push({ y: b.y + b.h, t: 0, o: b });
    for (const d of g.dinos) if (!d.carried && vis(d.x, d.y)) items.push({ y: d.y, t: 1, o: d });
    for (const gu of g.guests) if (!gu.hidden && vis(gu.x, gu.y)) items.push({ y: gu.y, t: 2, o: gu });
    for (const s of g.staff) if (vis(s.x, s.y)) items.push({ y: s.y, t: 2, o: s });
    for (const j of g.jeeps) if (vis(j.x, j.y)) items.push({ y: j.y, t: 5, o: j });
    items.sort((a, b) => a.y - b.y);
    const noSel = { selected: null };
    for (const it of items) {
      if (it.t === 0) this.drawBuilding(pctx, it.o, noSel);
      else if (it.t === 1) this.drawDino(pctx, it.o, noSel);
      else if (it.t === 2) this.drawPerson(pctx, it.o, noSel);
      else this.drawJeep(pctx, it.o);
    }
    for (const p of g.pteros) if (vis(p.x, p.y)) this.drawPtera(pctx, p, noSel);
    for (const h of g.helis) if (vis(h.x, h.y)) this.drawHeli(pctx, h);
    pctx.setTransform(1, 0, 0, 1, 0, 0);
    // scanlines + REC dot
    pctx.fillStyle = 'rgba(0,0,0,0.12)';
    for (let y = 0; y < ph; y += 3) pctx.fillRect(0, y, pw, 1);
    if (Math.floor(this.time * 2) % 2) { pctx.fillStyle = '#ff2020'; pctx.beginPath(); pctx.arc(10, 10, 4, 0, Math.PI * 2); pctx.fill(); }
  }

  // ---------- minimap ----------
  drawMinimap(mctx, mw, mh) {
    const g = this.game, w = g.world;
    const sx = mw / w.W, sy = mh / w.H;
    if (!this.miniImg || this.miniV !== w.pathVersion || this.miniFrame++ % 30 === 0) {
      this.miniV = w.pathVersion;
      const img = mctx.createImageData(w.W, w.H);
      const col = (t) => ({ [T_DEEP]: [30, 80, 140], [T_WATER]: [58, 136, 200], [T_SAND]: [220, 198, 138], [T_GRASS]: [80, 138, 50], [T_FOREST]: [40, 96, 36], [T_ROCK]: [120, 116, 104], [T_LAVA]: [232, 90, 26], [T_BASALT]: [58, 52, 54], [T_VOLCANO]: [80, 50, 40] })[t];
      for (let i = 0; i < w.W * w.H; i++) {
        let c = col(w.terrain[i]);
        if (w.path[i]) c = [210, 180, 130];
        if (w.track[i]) c = [90, 90, 88];
        if (w.fence[i] === F_ELECTRIC) c = w.fencePowered[i] ? [248, 224, 64] : [200, 60, 40];
        if (w.fence[i] === F_WALL) c = [200, 200, 196];
        if (w.fence[i] === F_GATE) c = [140, 100, 50];
        if (w.fence[i] === F_BROKEN) c = [255, 0, 0];
        if (w.bld[i]) c = [240, 236, 224];
        img.data[i * 4] = c[0]; img.data[i * 4 + 1] = c[1]; img.data[i * 4 + 2] = c[2]; img.data[i * 4 + 3] = 255;
      }
      if (!this.miniC) [this.miniC, this.miniX] = makeCanvas(w.W, w.H);
      this.miniX.putImageData(img, 0, 0);
      this.miniImg = true;
    }
    mctx.imageSmoothingEnabled = false;
    mctx.drawImage(this.miniC, 0, 0, mw, mh);
    for (const d of g.creatures()) {
      if (d.carried) continue;
      mctx.fillStyle = d.loose ? (Math.floor(this.time * 4) % 2 ? '#ff2020' : '#ffffff') : (d.sp.diet === 'carn' ? '#e07030' : '#70e070');
      mctx.fillRect(Math.floor(d.x * sx) - 1, Math.floor(d.y * sy) - 1, 3, 3);
    }
    mctx.fillStyle = '#f0f0f0';
    for (const gu of g.guests) if (!gu.hidden) mctx.fillRect(Math.floor(gu.x * sx), Math.floor(gu.y * sy), 1, 1);
    const z = this.cam.zoom;
    mctx.strokeStyle = '#ffffff';
    mctx.strokeRect(this.cam.x / TILE * sx + 0.5, this.cam.y / TILE * sy + 0.5, this.canvas.width / z / TILE * sx, this.canvas.height / z / TILE * sy);
  }
}
