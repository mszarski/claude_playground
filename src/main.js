// ---------- Bootstrap, main loop, title screen, save/load ----------
'use strict';

let GAME = null, RENDER = null, UIX = null, SFX = null;
let lastT = 0, titleAnim = null;

function startGame(opts = {}) {
  const seed = opts.seed || window.TEST_SEED || (Math.floor(Math.random() * 1e9) + 1);
  const g = new Game(seed, opts);
  if (opts.sandbox) { g.sandbox = true; g.money = 5000000; }
  setActiveGame(g);
  if (window.innerWidth < 700) RENDER.cam.zoom = 2;
  if (!opts.loaded) {
    UIX.centerOn(g.start.x, g.start.y - 6);
    if (!opts.scenario && !opts.tutorial) g.log('Welcome to Isla Nublar! Follow the goals in the bottom-left to build your park.', 'goal', null, true);
  }
  if (opts.tutorial) {
    g.tutorial = true; g.events.auto = false; g.advisorOn = false;
    g.secPerHour = 7.5; // slow days: time to read and build without the park running away
    document.getElementById('goal').hidden = true;
    UIX.tutorial = new Tutorial(UIX);
  } else if (UIX.tutorial) { UIX.tutorial.finish(); }
  hideTitle();
}

function setActiveGame(g) {
  GAME = g;
  const cv = document.getElementById('view');
  if (!RENDER) { RENDER = new Renderer(cv, g); RENDER.resize(window.innerWidth, window.innerHeight); }
  else RENDER.setGame(g);
  if (!UIX) UIX = new UI(g, RENDER, SFX); else UIX.attachGame(g);
  UIX.setTool('inspect'); UIX.closeFlyout(); UIX.setSpeed(1);
  if ($('#modal').style.display === 'flex') UIX.closeModal(); // nothing from the last game carries over
  window.GAME = g;
}

// ---------------- save / load ----------------
function b64(u8) { let s = ''; for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000)); return btoa(s); }
function unb64(str, Type) { const s = atob(str); const u8 = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u8[i] = s.charCodeAt(i); return new Type(u8.buffer); }

function serialize(g) {
  const w = g.world;
  return JSON.stringify({
    v: 1, size: g.mapSize, sph: g.secPerHour, lastInc: g.lastIncidentDay, closed: g.parkClosed ? 1 : 0,
    mods: g.mods || [], contracts: g.contracts || [], freeEgg: g.freeEgg || null, seenDec: g.seenDecisions || {}, nextDec: g.nextDecisionDay || 0, decOn: g.decisionsOn !== false, research: { done: [...g.research.done], active: g.research.active }, breedOff: !!g.breedingOff, seed: g.seed, diff: g.difficulty, scen: g.scenario || null, scenState: g.scenarioState || null, money: g.money, time: g.time, rep: g.reputation, ticket: g.ticket, goalIdx: g.goalIdx, sandbox: !!g.sandbox,
    stats: g.stats, history: g.history, deathLog: g.deathLog, auto: g.events.auto, nextDay: g.events.nextDay, start: g.start,
    // anything in flight when the player saved: eggs, crises, queued fences
    eggs: g.eggs.map((e) => ({ s: e.species, x: e.x, y: e.y, t: e.t, max: e.max, un: e.unstable ? 1 : 0 })), pend: g.pendingFences || [],
    ev: { storm: g.events.storm, quake: g.events.quake, volcano: g.events.volcano, outage: w.power.outage, bDelay: w.power.backupDelay, bFuel: w.power.backupFuel },
    alarm: g.alarm ? 1 : 0, gateLocked: g.gateLocked ? 1 : 0, lavaT: b64(new Uint8Array(w.lavaT.buffer)),
    terrain: b64(w.terrain), fence: b64(w.fence), fenceOrig: b64(w.fenceOrig), path: b64(w.path), track: b64(w.track), fenceHp: b64(new Uint8Array(w.fenceHp.buffer)),
    buildings: Array.from(w.buildings.values()).map((b) => ({ type: b.type, x: b.x, y: b.y, hp: b.hp, visitors: b.visitors, revenue: b.revenue || 0, esc: b.escaped ? 1 : 0, door: b.doorsBroken ? 1 : 0, off: b.offline > 0 ? b.offline : 0 })),
    dinos: g.creatures().filter((d) => !d.dead).map((d) => ({ s: d.species, x: d.x, y: d.y, hx: d.home.x, hy: d.home.y, hp: d.hp, hu: d.hunger, st: d.stress, n: d.name, sick: d.sick, k: d.kills, gr: d.growth, ag: d.ageDays, ls: d.lifespan, un: d.unstable ? 1 : 0, lo: d.loose ? 1 : 0 })),
  });
}

function saveGame(g) {
  try { localStorage.setItem('jt_save', serialize(g)); return true; } catch (e) { return false; }
}

function loadGame(json) {
  const s = JSON.parse(json);
  // saves from before island sizes existed are all 'small'
  const g = new Game(s.seed, { skipSetup: true, difficulty: s.diff, size: s.size || 'small' });
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
    if (bd.esc) b.escaped = true; if (bd.door) b.doorsBroken = true; if (bd.off) b.offline = bd.off;
    const def = BUILDINGS[bd.type];
    if (def.staff) for (let k = 0; k < def.staffN; k++) g.staff.push(new Staff(g, def.staff, b));
    if (bd.type === 'helipad') g.helis.push(new Helicopter(g, b));
    if (bd.type === 'tour' || bd.type === 'gyro') { b.queue = []; b.jeepT = 0; }
  }
  for (const dd of s.dinos) {
    const d = SPECIES[dd.s].flying ? new Ptera(g, Math.floor(dd.x), Math.floor(dd.y)) : new Dino(g, dd.s, Math.floor(dd.x), Math.floor(dd.y));
    d.x = dd.x; d.y = dd.y; d.home = { x: dd.hx, y: dd.hy }; d.hp = dd.hp; d.hunger = dd.hu; d.stress = dd.st; d.name = dd.n; d.sick = dd.sick; d.kills = dd.k || 0; if (dd.un) { d.unstable = true; } if (dd.lo) d.loose = d.wasLooseFlag = true; if (dd.gr !== undefined) d.growth = dd.gr; if (dd.ag !== undefined) { d.ageDays = dd.ag; d.lifespan = dd.ls; } // dd.lo: already loose when saved, not a new escape
    g.breedingOff = s.breedOff;
    if (d.isPtera) g.pteros.push(d); else g.dinos.push(d);
  }
  g.money = s.money; g.time = s.time; g.reputation = s.rep; g.ticket = s.ticket; g.goalIdx = s.goalIdx; g.sandbox = s.sandbox;
  g.stats = Object.assign(g.stats, s.stats); g.history = s.history || []; g.deathLog = s.deathLog || [];
  g.events.auto = s.auto; g.events.nextDay = s.nextDay; g.start = s.start;
  g.lastDay = g.day;
  if (s.sph) g.secPerHour = s.sph;
  if (s.lastInc) g.lastIncidentDay = s.lastInc;
  g.parkClosed = !!s.closed;
  g.mods = s.mods || []; g.contracts = s.contracts || []; g.freeEgg = s.freeEgg || null;
  g.seenDecisions = s.seenDec || {}; g.nextDecisionDay = s.nextDec || 0; g.decisionsOn = s.decOn !== false;
  if (s.research) { g.research.done = new Set(s.research.done); g.research.active = s.research.active; g.research.apply(); }
  if (s.scen) { g.scenario = s.scen; g.scenarioState = s.scenState || {}; }
  if (s.lavaT) w.lavaT.set(unb64(s.lavaT, Float32Array));
  g.eggs = (s.eggs || []).map((e) => ({ species: e.s, x: e.x, y: e.y, t: e.t, max: e.max, unstable: !!e.un }));
  g.pendingFences = s.pend || [];
  if (s.ev) {
    g.events.storm = s.ev.storm || null; g.events.quake = s.ev.quake || 0; g.events.volcano = s.ev.volcano || null;
    w.power.outage = s.ev.outage || 0; if (s.ev.bDelay !== undefined) w.power.backupDelay = s.ev.bDelay; if (s.ev.bFuel !== undefined) w.power.backupFuel = s.ev.bFuel;
  }
  g.alarm = !!s.alarm; g.gateLocked = !!s.gateLocked;
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

// ---------------- campaign menu ----------------
function showCampaignMenu() {
  const m = document.getElementById('scenMenu');
  const done = loadCampaign();
  m.innerHTML = CAMPAIGN_ORDER.map((k) => {
    const c = SCENARIOS[k], open = campaignUnlocked(k), won = done[k];
    const perk = CAMPAIGN_PERKS[k] ? `<br><small style="font-size:8px;color:#f8d040">REWARD: ${CAMPAIGN_PERKS[k].desc.toUpperCase()}</small>` : '';
    return `<button data-camp="${k}" ${open ? '' : 'disabled'} title="${c.blurb}">${won ? MEDAL_ICON[won.medal] + ' ' : open ? '' : '🔒 '}CHAPTER ${c.chapter}: ${c.name.toUpperCase()}<br><small style="font-size:8px;color:#9ab08a">${c.objective.toUpperCase()}</small>${perk}</button>`;
  }).join('') + '<button id="scenBack">BACK</button>';
  document.querySelector('#title .menu').hidden = true; m.hidden = false;
  for (const b of m.querySelectorAll('[data-camp]')) b.onclick = () => {
    document.querySelector('#title .menu').hidden = false; m.hidden = true;
    startChapter(b.dataset.camp);
  };
  document.getElementById('scenBack').onclick = () => { document.querySelector('#title .menu').hidden = false; m.hidden = true; };
}
function startChapter(id) {
  const c = SCENARIOS[id];
  startGame({ scenario: id, difficulty: c.difficulty || 'normal' });
  UIX.showChapterIntro(id);
}

// ---------------- title ----------------
function showTitle() {
  const t = document.getElementById('title');
  t.style.display = 'flex';
  let has = false;
  try { has = !!localStorage.getItem('jt_save'); } catch (e) { /* storage unavailable */ }
  document.getElementById('btnContinue').style.display = has ? '' : 'none';
  let tutDone = false;
  try { tutDone = !!localStorage.getItem('jt_tutorial_done'); } catch (e) { /* storage unavailable */ }
  document.getElementById('btnTut').textContent = tutDone ? 'TUTORIAL' : '★ TUTORIAL (START HERE) ★';
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
      let remaining = dtReal * speed * (UIX.slowmo > 0 ? 0.25 : 1);
      while (remaining > 1e-6) {
        const step = Math.min(1 / 30, remaining);
        GAME.update(step);
        remaining -= step;
      }
    }
    UIX.frame(dtReal);
    if (UIX.tutorial) UIX.tutorial.update(dtReal);
    if (GAME.alarm && UIX.speed > 0) SFX.play('alarm', 0.35);
    SFX.tense = GAME.looseCount > 0 || GAME.world.power.outage > 0 || GAME.alarm;
    // ambient jungle
    if (UIX.speed > 0 && Math.random() < dtReal * 0.35) SFX.play(GAME.isNight ? 'cricket' : (Math.random() < 0.15 ? 'caw' : 'chirp'), 0.8);
    RENDER.draw(dtReal, UIX);
    // breach cam follows the most dangerous loose creature
    const pip = document.getElementById('pip');
    const looseC = GAME.creatures().filter((d) => d.loose && !d.carried && !d.camouflaged);
    if (looseC.length && !UIX.pipHidden) {
      looseC.sort((a, b) => b.sp.danger - a.sp.danger);
      const tgt = looseC[0];
      if (pip.hidden) { pip.hidden = false; pip.onclick = (e) => { if (e.target.id === 'pipClose') { UIX.pipHidden = true; pip.hidden = true; return; } UIX.centerOn(UIX.pipTarget.x, UIX.pipTarget.y); UIX.select(UIX.pipTarget); }; }
      UIX.pipTarget = tgt;
      document.getElementById('pipLabel').textContent = `BREACH CAM · ${tgt.name.toUpperCase()}`;
      const pc = document.getElementById('pipCanvas');
      RENDER.drawPip(pc.getContext('2d'), pc.width, pc.height, tgt, UIX);
    } else {
      if (!pip.hidden) pip.hidden = true;
      if (!looseC.length) UIX.pipHidden = false;
    }
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
  let pref = 'all';
  try { pref = localStorage.getItem('jt_audio') || 'all'; } catch (e) { /* storage unavailable */ }
  SFX.wantMusic = pref === 'all'; SFX.muted = pref === 'mute';
  // browsers only allow audio after a user gesture
  const kick = () => { if (SFX.wantMusic && !SFX.muted) SFX.startMusic(); document.getElementById('sndBtn').textContent = SFX.muted ? '✕' : SFX.musicOn ? '♫' : '♪'; window.removeEventListener('pointerdown', kick); };
  window.addEventListener('pointerdown', kick);
  // installed as a home-screen app: cache everything so it runs offline
  if ('serviceWorker' in navigator && (location.protocol === 'https:' || location.hostname === 'localhost') && document.querySelector('link[rel="manifest"]')) {
    navigator.serviceWorker.register('sw.js').catch(() => { /* offline support is optional */ });
    // an update was installed while the app was open: reload straight away on the title screen,
    // otherwise offer it, so a park in progress is never yanked away
    const hadController = !!navigator.serviceWorker.controller;
    navigator.serviceWorker.addEventListener('controllerchange', () => {
      if (!hadController) return;
      const title = document.getElementById('title');
      if (!GAME || (title && !title.hidden && getComputedStyle(title).display !== 'none')) { location.reload(); return; }
      if (UIX) UIX.toast('A new version of the game is ready. Save, then reload the page to play it.', 'info');
    });
  }
  // iOS ignores user-scalable=no: stop page pinch-zoom so pinches only zoom the map
  document.addEventListener('gesturestart', (e) => e.preventDefault());
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
    startGame({ difficulty: b.dataset.diff, dayLength: window.currentDayLength(), size: window.currentMapSize(), seed: parseSeed(document.getElementById('seedIn').value) || undefined });
  };
  document.getElementById('btnCamp').onclick = () => { SFX.ensure(); SFX.play('click'); showCampaignMenu(); };
  document.getElementById('btnScen').onclick = () => {
    SFX.ensure(); SFX.play('click');
    const m = document.getElementById('scenMenu');
    const won = loadMedals();
    m.innerHTML = Object.entries(SCENARIOS).filter(([, sc]) => !sc.campaign).map(([k, sc]) => `<button data-scen="${k}" title="${sc.blurb}">${won[k] ? MEDAL_ICON[won[k].medal] + ' ' : ''}${sc.name.toUpperCase()}<br><small style="font-size:8px;color:#9ab08a">${sc.objective.toUpperCase()}</small></button>`).join('') + '<button id="scenBack">BACK</button>';
    document.querySelector('#title .menu').hidden = true; m.hidden = false;
    for (const b of m.querySelectorAll('[data-scen]')) b.onclick = () => {
      document.querySelector('#title .menu').hidden = false; m.hidden = true;
      startGame({ scenario: b.dataset.scen, difficulty: 'normal' });
      UIX.toast(SCENARIOS[b.dataset.scen].blurb, 'warn');
    };
    document.getElementById('scenBack').onclick = () => { document.querySelector('#title .menu').hidden = false; m.hidden = true; };
  };
  const lens = ['normal', 'long', 'short'];
  let dayLen = 'normal';
  try { dayLen = localStorage.getItem('jt_daylen') || 'normal'; } catch (e) { /* storage unavailable */ }
  const dlb = document.getElementById('dayLenBtn');
  const showLen = () => { dlb.innerHTML = `DAY LENGTH: ${dayLen.toUpperCase()}<br><small style="font-size:8px;color:#9ab08a">${{ short: '40 SECONDS PER DAY', normal: '60 SECONDS PER DAY', long: '90 SECONDS PER DAY' }[dayLen]}</small>`; };
  showLen();
  dlb.onclick = () => { dayLen = lens[(lens.indexOf(dayLen) + 1) % 3]; try { localStorage.setItem('jt_daylen', dayLen); } catch (e) { /* storage unavailable */ } showLen(); SFX.play('click'); };
  window.currentDayLength = () => dayLen;
  const sizes = Object.keys(MAP_SIZES);
  let mapSize = 'medium';
  try { mapSize = localStorage.getItem('jt_mapsize') || 'medium'; } catch (e) { /* storage unavailable */ }
  if (!MAP_SIZES[mapSize]) mapSize = 'medium';
  const msb = document.getElementById('mapSizeBtn');
  const showSize = () => { const m = MAP_SIZES[mapSize]; msb.innerHTML = `ISLAND SIZE: ${m.label}<br><small style="font-size:8px;color:#9ab08a">${m.desc} · ${m.w}×${m.h}</small>`; };
  showSize();
  msb.onclick = () => { mapSize = sizes[(sizes.indexOf(mapSize) + 1) % sizes.length]; try { localStorage.setItem('jt_mapsize', mapSize); } catch (e) { /* storage unavailable */ } showSize(); SFX.play('click'); };
  window.currentMapSize = () => mapSize;
  document.getElementById('diffBack').onclick = () => { document.querySelector('#title .menu').hidden = false; document.getElementById('diffMenu').hidden = true; };
  document.getElementById('btnTut').onclick = () => { SFX.ensure(); SFX.play('click'); startGame({ tutorial: true, difficulty: 'easy', seed: 51 }); }; // an island with open grass in front of the gate
  document.getElementById('btnSandbox').onclick = () => { SFX.ensure(); SFX.play('click'); startGame({ sandbox: true, size: window.currentMapSize() }); };
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
