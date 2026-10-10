// ---------- Bootstrap, main loop, title screen, save/load ----------
'use strict';

let GAME = null, RENDER = null, UIX = null, SFX = null;
let lastT = 0, titleAnim = null;

function startGame(opts = {}) {
  const seed = opts.seed || (Math.floor(Math.random() * 1e9) + 1);
  const g = new Game(seed, opts);
  if (opts.sandbox) { g.sandbox = true; g.money = 5000000; }
  setActiveGame(g);
  if (window.innerWidth < 700) RENDER.cam.zoom = 2;
  if (!opts.loaded) {
    UIX.centerOn(g.start.x, g.start.y - 6);
    g.log('Welcome to Isla Nublar! Follow the goals in the bottom-left to build your park.', 'goal', null, true);
  }
  hideTitle();
}

function setActiveGame(g) {
  GAME = g;
  const cv = document.getElementById('view');
  if (!RENDER) { RENDER = new Renderer(cv, g); RENDER.resize(window.innerWidth, window.innerHeight); }
  else RENDER.setGame(g);
  if (!UIX) UIX = new UI(g, RENDER, SFX); else UIX.attachGame(g);
  UIX.setTool('inspect'); UIX.closeFlyout(); UIX.setSpeed(1);
  window.GAME = g;
}

// ---------------- save / load ----------------
function b64(u8) { let s = ''; for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000)); return btoa(s); }
function unb64(str, Type) { const s = atob(str); const u8 = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u8[i] = s.charCodeAt(i); return new Type(u8.buffer); }

function serialize(g) {
  const w = g.world;
  return JSON.stringify({
    v: 1, seed: g.seed, diff: g.difficulty, money: g.money, time: g.time, rep: g.reputation, ticket: g.ticket, goalIdx: g.goalIdx, sandbox: !!g.sandbox,
    stats: g.stats, history: g.history, deathLog: g.deathLog, auto: g.events.auto, nextDay: g.events.nextDay, start: g.start,
    terrain: b64(w.terrain), fence: b64(w.fence), fenceOrig: b64(w.fenceOrig), path: b64(w.path), track: b64(w.track), fenceHp: b64(new Uint8Array(w.fenceHp.buffer)),
    buildings: Array.from(w.buildings.values()).map((b) => ({ type: b.type, x: b.x, y: b.y, hp: b.hp, visitors: b.visitors, revenue: b.revenue || 0 })),
    dinos: g.creatures().filter((d) => !d.dead).map((d) => ({ s: d.species, x: d.x, y: d.y, hx: d.home.x, hy: d.home.y, hp: d.hp, hu: d.hunger, st: d.stress, n: d.name, sick: d.sick, k: d.kills })),
  });
}

function saveGame(g) {
  try { localStorage.setItem('jt_save', serialize(g)); return true; } catch (e) { return false; }
}

function loadGame(json) {
  const s = JSON.parse(json);
  const g = new Game(s.seed, { skipSetup: true, difficulty: s.diff });
  const w = g.world;
  w.terrain.set(unb64(s.terrain, Uint8Array));
  w.fence.set(unb64(s.fence, Uint8Array));
  w.fenceOrig.set(unb64(s.fenceOrig, Uint8Array));
  w.path.set(unb64(s.path, Uint8Array));
  if (s.track) w.track.set(unb64(s.track, Uint8Array));
  w.fenceHp.set(unb64(s.fenceHp, Float32Array));
  for (const b of Array.from(w.buildings.values())) w.removeBuilding(b);
  for (const bd of s.buildings) {
    const b = w.addBuilding(bd.type, bd.x, bd.y);
    b.hp = bd.hp; b.visitors = bd.visitors; b.revenue = bd.revenue;
    const def = BUILDINGS[bd.type];
    if (def.staff) for (let k = 0; k < def.staffN; k++) g.staff.push(new Staff(g, def.staff, b));
    if (bd.type === 'helipad') g.helis.push(new Helicopter(g, b));
    if (bd.type === 'tour') { b.queue = []; b.jeepT = 0; }
  }
  for (const dd of s.dinos) {
    const d = SPECIES[dd.s].flying ? new Ptera(g, Math.floor(dd.x), Math.floor(dd.y)) : new Dino(g, dd.s, Math.floor(dd.x), Math.floor(dd.y));
    d.x = dd.x; d.y = dd.y; d.home = { x: dd.hx, y: dd.hy }; d.hp = dd.hp; d.hunger = dd.hu; d.stress = dd.st; d.name = dd.n; d.sick = dd.sick; d.kills = dd.k || 0;
    if (d.isPtera) g.pteros.push(d); else g.dinos.push(d);
  }
  g.money = s.money; g.time = s.time; g.reputation = s.rep; g.ticket = s.ticket; g.goalIdx = s.goalIdx; g.sandbox = s.sandbox;
  g.stats = Object.assign(g.stats, s.stats); g.history = s.history || []; g.deathLog = s.deathLog || [];
  g.events.auto = s.auto; g.events.nextDay = s.nextDay; g.start = s.start;
  g.lastDay = g.day;
  w.invalidate(); w.computeRegions(); w.computePower(g); g.updateLoose();
  for (const d of g.dinos) d.wasLooseFlag = d.loose;
  return g;
}

function loadGameFromStorage() {
  try {
    const s = localStorage.getItem('jt_save');
    if (!s) return false;
    const g = loadGame(s);
    setActiveGame(g);
    UIX.centerOn(g.start.x, g.start.y - 6);
    hideTitle();
    UIX.toast('Park loaded. Welcome back!', 'good');
    return true;
  } catch (e) { console.error(e); return false; }
}

// ---------------- title ----------------
function showTitle() {
  const t = document.getElementById('title');
  t.style.display = 'flex';
  let has = false;
  try { has = !!localStorage.getItem('jt_save'); } catch (e) { /* storage unavailable */ }
  document.getElementById('btnContinue').style.display = has ? '' : 'none';
  if (GAME) UIX.setSpeed(0);
  startTitleAnim();
}
function hideTitle() { document.getElementById('title').style.display = 'none'; titleAnim = null; }

function startTitleAnim() {
  const c = document.getElementById('tbg');
  const ctx = c.getContext('2d');
  const S = 4;
  c.width = Math.ceil(window.innerWidth / S); c.height = Math.ceil(window.innerHeight / S);
  ctx.imageSmoothingEnabled = false;
  const W = c.width, H = c.height;
  const r = mulberry32(42);
  const parade = [];
  const order = ['galli', 'galli', 'para', 'trike', 'stego', 'brachio', 'anky', 'raptor', 'raptor', 'trex'];
  let x = -40;
  for (const sp of order) { parade.push({ sp, x, spd: SPECIES[sp].speed * 6 }); x -= DINO_SPRITES[sp].w + 16 + r() * 30; }
  const trees = []; for (let i = 0; i < W / 10; i++) trees.push({ x: r() * W, v: Math.floor(r() * 8), s: r() });
  let t = 0;
  titleAnim = true;
  const step = (now) => {
    if (!titleAnim) return;
    t += 1 / 60;
    // sky
    const g = ctx.createLinearGradient(0, 0, 0, H);
    g.addColorStop(0, '#f08a3a'); g.addColorStop(0.45, '#c84a2a'); g.addColorStop(1, '#2a1a20');
    ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
    // sun
    ctx.fillStyle = '#f8d878'; ctx.beginPath(); ctx.arc(W * 0.7, H * 0.42, Math.min(W, H) * 0.12, 0, Math.PI * 2); ctx.fill();
    // volcano silhouette
    ctx.fillStyle = '#3a1a1e';
    ctx.beginPath(); ctx.moveTo(W * 0.05, H * 0.75); ctx.lineTo(W * 0.25, H * 0.38); ctx.lineTo(W * 0.31, H * 0.38); ctx.lineTo(W * 0.55, H * 0.75); ctx.fill();
    ctx.fillStyle = 'rgba(80,60,60,0.5)';
    for (let i = 0; i < 6; i++) { const ph = (t * 0.1 + i / 6) % 1; ctx.fillRect(W * 0.28 - 3 + Math.sin(ph * 5 + i) * 6 + ph * 18, H * 0.38 - ph * H * 0.25, 4 + ph * 8, 4 + ph * 8); }
    ctx.fillStyle = '#2a141a'; ctx.fillRect(0, H * 0.75, W, H * 0.25);
    // trees
    for (const tr of trees) ctx.drawImage(treeSil(tr.v), Math.round(tr.x), Math.round(H * 0.75 - 24 - tr.s * 6));
    ctx.globalAlpha = 1;
    // dinos
    for (const p of parade) {
      p.x += p.spd / 60;
      const D = DINO_SPRITES[p.sp];
      if (p.x > W + 20) p.x = -D.w - 60 - r() * 100;
      const fr = Math.floor(t * p.spd / 3) % 2;
      const img = tintCanvasCached(p.sp, fr);
      ctx.drawImage(img, Math.round(p.x), Math.round(H * 0.78 - D.h + 4));
    }
    requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}
const _silCache = {};
function treeSil(v) {
  const k = 'tree' + v;
  if (!_silCache[k]) _silCache[k] = tintCanvas(TILES.trees[v], '#24121a', 0.9);
  return _silCache[k];
}
function tintCanvasCached(sp, fr) {
  const k = sp + fr;
  if (!_silCache[k]) _silCache[k] = tintCanvas(DINO_SPRITES[sp].right[fr], '#1a0c10', 0.92);
  return _silCache[k];
}

// ---------------- loop ----------------
function loop(now) {
  const dtReal = Math.min(0.1, (now - lastT) / 1000 || 0.016);
  lastT = now;
  if (GAME && document.getElementById('title').style.display === 'none') {
    const speed = UIX.speed;
    const modalOpen = document.getElementById('modal').style.display === 'flex';
    if (speed > 0 && !modalOpen) {
      let remaining = dtReal * speed;
      while (remaining > 1e-6) {
        const step = Math.min(1 / 30, remaining);
        GAME.update(step);
        remaining -= step;
      }
    }
    UIX.frame(dtReal);
    // ambient jungle
    if (UIX.speed > 0 && Math.random() < dtReal * 0.35) SFX.play(GAME.isNight ? 'cricket' : (Math.random() < 0.15 ? 'caw' : 'chirp'), 0.8);
    RENDER.draw(dtReal, UIX);
    RENDER.miniT = (RENDER.miniT || 0) + 1;
    if (RENDER.miniT % 3 === 0) {
      const mm = document.getElementById('minimap');
      RENDER.drawMinimap(mm.getContext('2d'), mm.width, mm.height);
    }
    // autosave each new day
    if (GAME.day !== RENDER.lastSaveDay) { RENDER.lastSaveDay = GAME.day; if (GAME.day > 1 && !GAME.gameOver) saveGame(GAME); }
  }
  requestAnimationFrame(loop);
}

function boot() {
  buildAllSprites();
  SFX = new Sfx();
  window.addEventListener('resize', () => {
    if (RENDER) { RENDER.resize(window.innerWidth, window.innerHeight); UIX.clampCam(); }
    if (titleAnim) startTitleAnim();
  });
  document.getElementById('btnNew').onclick = () => {
    SFX.ensure(); SFX.play('click');
    document.querySelector('#title .menu').hidden = true;
    document.getElementById('diffMenu').hidden = false;
  };
  for (const b of document.querySelectorAll('#diffMenu [data-diff]')) b.onclick = () => {
    SFX.play('click');
    document.querySelector('#title .menu').hidden = false;
    document.getElementById('diffMenu').hidden = true;
    startGame({ difficulty: b.dataset.diff });
  };
  document.getElementById('diffBack').onclick = () => { document.querySelector('#title .menu').hidden = false; document.getElementById('diffMenu').hidden = true; };
  document.getElementById('btnSandbox').onclick = () => { SFX.ensure(); SFX.play('click'); startGame({ sandbox: true }); };
  document.getElementById('btnContinue').onclick = () => { SFX.ensure(); if (!loadGameFromStorage()) { startGame(); } };
  document.getElementById('btnHelp').onclick = () => {
    if (!UIX) { setActiveGame(new Game(1)); }
    UIX.showHelp();
    document.getElementById('modal').style.zIndex = 60;
  };
  showTitle();
  requestAnimationFrame(loop);
}

window.addEventListener('load', boot);
