// ---------- Pixel art: dinosaur templates, people, tiles, buildings ----------
'use strict';

function makeCanvas(w, h) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d');
  ctx.imageSmoothingEnabled = false;
  return [c, ctx];
}

// Render character-grid rows into a canvas using a palette map.
function gridToCanvas(rows, pal) {
  const h = rows.length;
  let w = 0;
  for (const r of rows) w = Math.max(w, r.length);
  const [c, ctx] = makeCanvas(w, h);
  for (let y = 0; y < h; y++) {
    const r = rows[y];
    for (let x = 0; x < r.length; x++) {
      const ch = r[x];
      if (ch === '.' || ch === ' ') continue;
      const col = pal[ch];
      if (!col) continue;
      ctx.fillStyle = col;
      ctx.fillRect(x, y, 1, 1);
    }
  }
  return c;
}

function flipCanvas(src) {
  const [c, ctx] = makeCanvas(src.width, src.height);
  ctx.translate(src.width, 0); ctx.scale(-1, 1);
  ctx.drawImage(src, 0, 0);
  return c;
}

// Tint every opaque pixel of a canvas (used for hit flashes / silhouettes)
function tintCanvas(src, color, alpha = 1) {
  const [c, ctx] = makeCanvas(src.width, src.height);
  ctx.drawImage(src, 0, 0);
  ctx.globalCompositeOperation = 'source-atop';
  ctx.globalAlpha = alpha;
  ctx.fillStyle = color;
  ctx.fillRect(0, 0, c.width, c.height);
  return c;
}

// ---------------- Dinosaur templates (facing right) ----------------
// O outline, B body, D dark shade, L light belly, S accent/stripe, E eye, T teeth, c horn/bone
const DINO_TEMPLATES = {
  raptor: {
    body: [
      '             OOOO',
      '            OBBBBO',
      '            OBBEBBO',
      '           OBBBBBBBO',
      'OO         OBBLTTTTO',
      'OBOO      OBBBOOOOO',
      ' OSBOO   OBBBBO',
      '  OBSBOOOBBBBLO',
      '   OOBSBSBSBBLOO',
      '     OOBBBBBLLOLO',
      '       OBBBBLO OO',
      '        ODDDDO',
    ],
    legsA: [
      '       ODDO ODO',
      '      ODO    ODO',
      '      OOO    OOOO',
    ],
    legsB: [
      '         ODDOO',
      '        ODOODO',
      '       OOOOOOO',
    ],
  },
  dilo: {
    body: [
      '            OO OO',
      '           OSSOSSO',
      '           OSBSSBO',
      '           OBBEBBBO',
      'OO         OBBBLTTO',
      'OBO       OBBBOOOO',
      ' OBOO    OBBBO',
      '  OBBOOOOBBBLO',
      '   OOBSBSBBBLOO',
      '     OOBBBBLLOLO',
      '       OBBBLO OO',
      '        ODDDO',
    ],
    legsA: [
      '       ODO ODO',
      '      ODO   ODO',
      '      OOO   OOOO',
    ],
    legsB: [
      '        ODDOO',
      '        ODOODO',
      '       OOOOOOO',
    ],
  },
  galli: {
    body: [
      '             OOO',
      '            OBBEO',
      '            OBBBLLO',
      '            OBOOOO',
      '            OBO',
      '           OBLO',
      '           OBLO',
      'OOO       OBBLO',
      ' OBBOOOOOOBBBLO',
      '  OOBBSBSBBBLLO',
      '    OOBBBBBLLO',
      '      OBBDDOO',
    ],
    legsA: [
      '      ODO ODO',
      '      ODO  ODO',
      '     ODO    ODO',
      '     OOO    OOO',
    ],
    legsB: [
      '       ODODO',
      '       ODODO',
      '       ODOODO',
      '      OOO OOO',
    ],
  },
  trex: {
    body: [
      '                 OOOOOOO',
      '                OBBBBBBBO',
      '               OBBBBBBEBBO',
      '               OBBBBBBBBBBO',
      '               OBBBBLLLLLLO',
      '               OBBBOTTTTTO',
      '               OBBBOOOOOO',
      'OO            OBBBBO',
      'OBOO        OOBBBBBLO',
      ' OBBOO     OBBBBBBBLOO',
      '  OBSBOO OOBBSBBBBBLOLO',
      '   OBBSBOBBSBBBBBBBLLOO',
      '    OOBBSBBSBBBBBBBLLO',
      '      OOBBBBBBBBBBLLO',
      '        OOBBBBBBBLLO',
      '          OBBBBDDDO',
      '          ODDBBDDO',
    ],
    legsA: [
      '         ODDDO ODDO',
      '        ODDDO   ODDO',
      '        ODDO     ODO',
      '       OOOOO    OOOOO',
    ],
    legsB: [
      '           ODDDODDO',
      '           ODDOODDO',
      '           ODO ODDO',
      '          OOOOOOOOO',
    ],
  },
  spino: {
    body: [
      '            O O O',
      '           OSOSOSO   OOOO',
      '          OSSSSSSSO OBBBBO',
      '         OSSSSSSSSSOOBBEBBOOO',
      '        OSSSSSSSSSSOBBBBBBBBBO',
      'OO     OSSSSSSSSSSSOBBBBLLTTTO',
      'OBOO  OBSSSSSSSSSSOBBBOOOOOOO',
      ' OBBOOBBBBBBBBBBBBBBBO',
      '  OBBBBBBBBBBBBBBBBBLO',
      '   OOBBSBBSBBBBBBBBLLOO',
      '     OOBBBBBBBBBBBBLLOLO',
      '       OOBBBBBBBBBLLO OO',
      '         OBBBBBDDDO',
      '         ODDBBBDDO',
    ],
    legsA: [
      '        ODDDO ODDO',
      '       ODDDO   ODDO',
      '       ODDO     ODO',
      '      OOOOO    OOOOO',
    ],
    legsB: [
      '          ODDDODDO',
      '          ODDOODDO',
      '          ODO ODDO',
      '         OOOOOOOOO',
    ],
  },
  trike: {
    body: [
      '                 OOO',
      '                OSSSO',
      '               OSSSSSO  OO',
      '     OOOOOOOO  OSSSSSOOOcO',
      '   OOBBBBBBBBOOOSSSBBBBcOO',
      '  OBBBDBBBBDBBBOSSBBEBBBOOO',
      ' OBBBBBBBBBBBBBBOSBBBBBBccO',
      'OBBBBBBBBBBBBBBBBOBBBBLLOO',
      'OOBBBBBBBBBBBBBBBBBBLLLO',
      '  OLLLLLLLLLLLLLLLLLLOO',
      '   OBDDOOOOOOOOOODDBO',
    ],
    legsA: [
      '   ODDO  ODDO  ODDO ODDO',
      '  ODDO   ODDO ODDO   ODDO',
      '  OOOO   OOOO OOOO   OOOO',
    ],
    legsB: [
      '    ODDOODDO   ODDOODDO',
      '    ODDOODDO   ODDOODDO',
      '    OOOOOOOO   OOOOOOOO',
    ],
  },
  stego: {
    body: [
      '          O   O   O',
      '         OSO OSO OSO',
      '     O  OSSSOSSSOSSSO',
      '    OSO OSSSSSSSSSSSSO',
      '    OSSOBBBBBBBBBBBBBBOO',
      ' O  OSBBBBBBBBBBBBBBBBBBO',
      'OcOOBBBBBDBBBBDBBBBDBBBBBO',
      'OcBBBBBBBBBBBBBBBBBBBBBBBBOO',
      ' OOOOBBBBBBBBBBBBBBBBBBBBBBEOO',
      '     OOBBBBBBBBBBBBBBBBBOLLLLO',
      '       OLLLLLLLLLLLLLLLO OOOO',
      '        OBDDOOOOOOOODDBO',
    ],
    legsA: [
      '        ODDO ODDO ODDO ODDO',
      '       ODDO  ODDO  ODDO ODDO',
      '       OOOO  OOOO  OOOO OOOO',
    ],
    legsB: [
      '         ODDOODDO  ODDOODDO',
      '         ODDOODDO  ODDOODDO',
      '         OOOOOOOO  OOOOOOOO',
    ],
  },
  anky: {
    body: [
      '        OOOOOOOOOOOO',
      '      OOSBSBSBSBSBSBOO',
      '     OSBSBSBSBSBSBSBSBO',
      'OOO OSBSBSBSBSBSBSBSBSBO',
      'OSSOBBBBBBBBBBBBBBBBBBBBOOO',
      'OSSBBBBBBBBBBBBBBBBBBBBBBBEOO',
      'OOOOOOBBBBBBBBBBBBBBBBBBBBBBBO',
      '      OLLLLLLLLLLLLLLLLLLOLLLO',
      '       OBDDOOOOOOOOOOODDBOOOO',
    ],
    legsA: [
      '       ODDO  ODDO ODDO  ODDO',
      '       OOOO  OOOO OOOO  OOOO',
    ],
    legsB: [
      '        ODDOODDO   ODDOODDO',
      '        OOOOOOOO   OOOOOOOO',
    ],
  },
  para: {
    body: [
      '          OOOO',
      '        OOSSSSO',
      '      OOSSSSOOBO',
      '     OSSSOO OBEBO',
      '      OO    OBBBLLO',
      '            OBBBOOO',
      '           OBBBO',
      '          OBBBLO',
      'OO       OBBBBLO',
      'OBOOOOOOOBBBBBLO',
      ' OOBBSBSBBBBBLLO',
      '   OOBBBBBBBBLLO',
      '     OOBBBBBLLO',
      '       OBDDDBO',
    ],
    legsA: [
      '      ODDO ODDO',
      '      ODO   ODO',
      '     OOOO   OOOO',
    ],
    legsB: [
      '        ODDODDO',
      '        ODOODO',
      '       OOOOOOOO',
    ],
  },
  ptera: {
    body: [
      '     OOO',
      '   OOSSSOOO',
      ' OOSSBBEBBOOOOO',
      '     OBBBOOOOOOO',
      '     OBBO',
      '    OBLBO',
      '  OOBLLBOO',
      ' OBDBLLBDBO',
      'OBDDOOBBOODDBO',
      'OOO  OBBO  OOO',
    ],
    legsA: [
      '     ODOODO',
      '    OOO OOO',
    ],
    legsB: [
      '     ODOODO',
      '     OOOOOO',
    ],
  },
  brachio: {
    body: [
      '                        OOOO',
      '                       OBBBBOO',
      '                       OBBBEBBO',
      '                       OBBBBBBLO',
      '                       OBBBOOOO',
      '                      OBBBO',
      '                      OBBLO',
      '                     OBBBO',
      '                     OBBLO',
      '                    OBBSO',
      '                    OBBLO',
      '                   OBBSO',
      '                   OBBLO',
      '                  OBBBLO',
      '           OOOOOOOBBBBLO',
      '        OOOBBBBBBBBBBBBLO',
      '      OOBBBBSBBBBBSBBBBBO',
      '    OOBBBBBBBBBBBBBBBBBBO',
      '  OOBBBSBBBBBSBBBBBBBBBLO',
      ' OBBOBBBBBBBBBBBBBBBBBLLO',
      'OBO  OBBBBBBBBBBBBBBBBLLO',
      'OO    OOLLLLLLLLLLLLLLLO',
      '       OBDDOOOOOOOODDBO',
    ],
    legsA: [
      '       ODDO ODDO  ODDO ODDO',
      '       ODDO ODDO  ODDO ODDO',
      '       ODDO  ODDO ODDO  ODDO',
      '      OOOOO  OOOOOOOOO  OOOO',
    ],
    legsB: [
      '        ODDOODDO   ODDOODDO',
      '        ODDOODDO   ODDOODDO',
      '        ODDOODDO   ODDOODDO',
      '       OOOOOOOOO  OOOOOOOOO',
    ],
  },
};

DINO_TEMPLATES.indom = {
  body: DINO_TEMPLATES.trex.body.map((r, i) => i === 1 ? r.replace('OBBBBBBBO', 'OSBSBSBSO') : i === 9 ? r.replace('OBBBBBBBLOO', 'OBSBSBBBLOO') : r),
  legsA: DINO_TEMPLATES.trex.legsA, legsB: DINO_TEMPLATES.trex.legsB,
};
const DINO_SPRITES = {}; // species -> {right:[a,b], left:[a,b], sleep:{right,left}, w, h}

function buildDinoSprites() {
  for (const key of Object.keys(SPECIES)) {
    const sp = SPECIES[key];
    const tpl = DINO_TEMPLATES[key];
    const pal = Object.assign({ T: '#f4f0e0', c: '#efe6c8' }, sp.colors);
    const fa = gridToCanvas(tpl.body.concat(tpl.legsA), pal);
    const fb = gridToCanvas(tpl.body.concat(tpl.legsB), pal);
    // Sleep: eyes closed (E -> D), legs tucked (use legsB)
    const sleepRows = tpl.body.map((r) => r.replace(/E/g, 'D')).concat(tpl.legsB);
    const sl = gridToCanvas(sleepRows, pal);
    const w = Math.max(fa.width, fb.width), h = Math.max(fa.height, fb.height);
    // Normalize sizes to same canvas
    const norm = (src) => { const [c, ctx] = makeCanvas(w, h); ctx.drawImage(src, 0, h - src.height); return c; };
    const a = norm(fa), b = norm(fb), s = norm(sl);
    DINO_SPRITES[key] = {
      w, h,
      right: [a, b], left: [flipCanvas(a), flipCanvas(b)],
      sleepR: s, sleepL: flipCanvas(s),
      flashR: [tintCanvas(a, '#ffffff', 0.85), tintCanvas(b, '#ffffff', 0.85)],
      flashL: [tintCanvas(flipCanvas(a), '#ffffff', 0.85), tintCanvas(flipCanvas(b), '#ffffff', 0.85)],
      sickR: [tintCanvas(a, '#7ad04a', 0.35), tintCanvas(b, '#7ad04a', 0.35)],
      sickL: [tintCanvas(flipCanvas(a), '#7ad04a', 0.35), tintCanvas(flipCanvas(b), '#7ad04a', 0.35)],
    };
  }
}

// ---------------- People ----------------
const SKIN = ['#f0c8a0', '#d8a078', '#b07850', '#8a5a3a', '#603a24'];
const HAIR = ['#2a1a10', '#5a3a1a', '#c89a4a', '#1a1a1a', '#8a4a2a', '#d8d0c0'];
const SHIRT = ['#e04838', '#3878c8', '#f0a030', '#68a088', '#9858b8', '#f8e060', '#e8e8e8', '#e86aa0', '#40a0a0'];
const PANTS = ['#2a3a5a', '#5a4a3a', '#3a3a3a', '#7a6a4a', '#284878'];

function makePersonFrames(opts) {
  // 5x9 tiny person, two walking frames
  const { skin, hair, shirt, pants, hat, vest } = opts;
  const frames = [];
  for (let f = 0; f < 2; f++) {
    const [c, ctx] = makeCanvas(6, 10);
    const p = (x, y, col) => { ctx.fillStyle = col; ctx.fillRect(x, y, 1, 1); };
    // head
    if (hat) { for (let x = 0; x < 6; x++) p(x, 1, hat); p(1, 0, hat); p(2, 0, hat); p(3, 0, hat); p(4, 0, hat); }
    else { p(1, 0, hair); p(2, 0, hair); p(3, 0, hair); p(4, 0, hair); p(1, 1, hair); p(4, 1, hair); }
    p(2, 1, hat ? hat : skin); p(3, 1, hat ? hat : skin);
    p(1, 2, skin); p(2, 2, skin); p(3, 2, skin); p(4, 2, skin);
    // body
    for (let y = 3; y < 6; y++) for (let x = 1; x < 5; x++) p(x, y, shirt);
    if (vest) { p(1, 3, vest); p(4, 3, vest); p(1, 4, vest); p(4, 4, vest); p(2, 5, vest); p(3, 5, vest); }
    // arms
    p(0, 3 + f, skin); p(5, 4 - f, skin);
    // legs
    p(1, 6, pants); p(2, 6, pants); p(3, 6, pants); p(4, 6, pants);
    if (f === 0) { p(1, 7, pants); p(4, 7, pants); p(1, 8, '#201810'); p(4, 8, '#201810'); }
    else { p(2, 7, pants); p(3, 7, pants); p(2, 8, '#201810'); p(3, 8, '#201810'); }
    frames.push(c);
  }
  return frames;
}

const PEOPLE = { guests: [], ranger: null, worker: null, vet: null };
function buildPeople() {
  const r = mulberry32(1234);
  const pk = (a) => a[Math.floor(r() * a.length)];
  for (let i = 0; i < 40; i++) {
    PEOPLE.guests.push(makePersonFrames({ skin: pk(SKIN), hair: pk(HAIR), shirt: pk(SHIRT), pants: pk(PANTS), hat: r() < 0.25 ? pk(['#e8e0c0', '#e04838', '#3878c8', '#f8e060']) : null }));
  }
  PEOPLE.ranger = makePersonFrames({ skin: '#d8a078', hair: '#2a1a10', shirt: '#8a7a4a', pants: '#5a4a2a', hat: '#c8b078' });
  PEOPLE.worker = makePersonFrames({ skin: '#b07850', hair: '#1a1a1a', shirt: '#3a5a8a', pants: '#2a3a5a', hat: '#f8e060', vest: '#f08020' });
}

// ---------------- Terrain tiles ----------------
const TILES = {}; // name -> canvas (or arrays of variants)

function noiseFill(ctx, w, h, base, specks, r) {
  ctx.fillStyle = base; ctx.fillRect(0, 0, w, h);
  for (const [col, n] of specks) {
    ctx.fillStyle = col;
    for (let i = 0; i < n; i++) ctx.fillRect(Math.floor(r() * w), Math.floor(r() * h), 1, 1);
  }
}

function buildTiles() {
  const r = mulberry32(777);
  const mk = (fn) => { const [c, ctx] = makeCanvas(TILE, TILE); fn(ctx); return c; };
  TILES.grass = [];
  for (let v = 0; v < 6; v++) {
    TILES.grass.push(mk((ctx) => {
      noiseFill(ctx, 16, 16, '#4f8a32', [['#5c9a3a', 26], ['#447a2a', 18], ['#6aa840', 6]], r);
      // grass tufts
      for (let i = 0; i < 3; i++) {
        const x = 1 + Math.floor(r() * 13), y = 2 + Math.floor(r() * 12);
        ctx.fillStyle = '#6fb048'; ctx.fillRect(x, y, 1, 1); ctx.fillRect(x + 2, y, 1, 1);
        ctx.fillStyle = '#3c6a24'; ctx.fillRect(x + 1, y + 1, 1, 1);
      }
      if (r() < 0.3) { // flower
        const x = 2 + Math.floor(r() * 12), y = 2 + Math.floor(r() * 12);
        ctx.fillStyle = pick2(r, ['#f8e060', '#f0f0f0', '#e86aa0']); ctx.fillRect(x, y, 1, 1);
      }
    }));
  }
  TILES.sand = [];
  for (let v = 0; v < 4; v++) TILES.sand.push(mk((ctx) => noiseFill(ctx, 16, 16, '#dcc68a', [['#e8d6a0', 20], ['#c8b074', 16], ['#b89c64', 4]], r)));
  TILES.basalt = [];
  for (let v = 0; v < 3; v++) TILES.basalt.push(mk((ctx) => noiseFill(ctx, 16, 16, '#3a3436', [['#4a4244', 30], ['#2a2426', 20], ['#5a3a30', 4]], r)));
  TILES.water = [[], []];
  TILES.deep = [[], []];
  for (let f = 0; f < 2; f++) {
    for (let v = 0; v < 3; v++) {
      TILES.water[f].push(mk((ctx) => {
        noiseFill(ctx, 16, 16, '#3a88c8', [['#4898d4', 18], ['#3078b8', 14]], r);
        ctx.fillStyle = '#7cc0ec';
        for (let i = 0; i < 2; i++) { const x = Math.floor(r() * 12), y = Math.floor(r() * 15); ctx.fillRect(x + f, y, 3, 1); }
      }));
      TILES.deep[f].push(mk((ctx) => {
        noiseFill(ctx, 16, 16, '#24609e', [['#2a6aa8', 16], ['#1e5490', 16]], r);
        ctx.fillStyle = '#4a8ac8';
        for (let i = 0; i < 2; i++) { const x = Math.floor(r() * 12), y = Math.floor(r() * 15); ctx.fillRect(x + f * 2, y, 2, 1); }
      }));
    }
  }
  TILES.lava = [[], []];
  for (let f = 0; f < 2; f++) for (let v = 0; v < 3; v++) {
    TILES.lava[f].push(mk((ctx) => {
      noiseFill(ctx, 16, 16, '#d84a1a', [['#f08a2a', 30], ['#a02a10', 26], ['#f8d050', 6 + f * 6]], r);
    }));
  }
  TILES.rock = [];
  for (let v = 0; v < 4; v++) TILES.rock.push(mk((ctx) => {
    noiseFill(ctx, 16, 16, '#7a7468', [['#8a8478', 26], ['#6a6458', 22], ['#9a9488', 8]], r);
    // boulders
    for (let i = 0; i < 2; i++) {
      const x = 1 + Math.floor(r() * 10), y = 1 + Math.floor(r() * 10);
      ctx.fillStyle = '#5a554c'; ctx.fillRect(x, y + 3, 5, 1);
      ctx.fillStyle = '#9c968a'; ctx.fillRect(x, y, 4, 3);
      ctx.fillStyle = '#b4aea2'; ctx.fillRect(x + 1, y, 2, 1);
    }
  }));
  TILES.volcano = [];
  for (let v = 0; v < 3; v++) TILES.volcano.push(mk((ctx) => noiseFill(ctx, 16, 16, '#4a3e3a', [['#5a4c46', 30], ['#3a2e2a', 26], ['#6a3a2a', 6]], r)));

  // Trees (drawn bigger than a tile, anchored bottom-center)
  TILES.trees = [];
  for (let v = 0; v < 8; v++) TILES.trees.push(makeTree(r, v));
  TILES.bush = [];
  for (let v = 0; v < 3; v++) TILES.bush.push(makeBush(r));
}

function pick2(r, a) { return a[Math.floor(r() * a.length)]; }

function makeTree(r, v) {
  const [c, ctx] = makeCanvas(20, 26);
  const p = (x, y, w, h, col) => { ctx.fillStyle = col; ctx.fillRect(x, y, w, h); };
  if (v % 3 === 0) {
    // Palm
    p(9, 10, 2, 14, '#7a5a32'); p(10, 10, 1, 14, '#5a4022');
    for (let y = 12; y < 24; y += 3) p(9, y, 2, 1, '#9a7a4a');
    const leaf = ['#3f8a2a', '#57a838', '#2e6a20'];
    const fronds = [[-8, 2], [-6, -3], [0, -6], [6, -3], [8, 2], [-3, 5], [3, 5]];
    for (const [dx, dy] of fronds) {
      const steps = 8;
      for (let i = 0; i <= steps; i++) {
        const t = i / steps;
        const x = Math.round(10 + dx * t), y = Math.round(9 + dy * t + (Math.abs(dx) > 5 ? t * t * 4 : 0));
        p(x - 1, y, 3, 1, leaf[i % 3]);
      }
    }
    p(9, 8, 3, 3, '#6a4a22'); p(8, 10, 2, 2, '#4a3a1a');
  } else if (v % 3 === 1) {
    // Broadleaf jungle tree
    p(9, 16, 3, 9, '#5a3e22'); p(9, 16, 1, 9, '#7a5a32');
    const cols = ['#2a5a1e', '#3a7a2a', '#4f9a36', '#6ab848'];
    const blobs = [[10, 9, 8], [5, 12, 5], [15, 12, 5], [10, 5, 5], [7, 7, 4], [13, 7, 4]];
    for (let pass = 0; pass < 4; pass++) {
      for (const [bx, by, br] of blobs) {
        const rr = br - pass * 1.2;
        if (rr <= 0) continue;
        for (let y = -br; y <= br; y++) for (let x = -br; x <= br; x++) {
          if (x * x + y * y <= rr * rr) {
            const px = bx + x - pass, py = by + y - pass;
            if (px >= 0 && px < 20 && py >= 0 && py < 26) p(px, py, 1, 1, cols[pass]);
          }
        }
      }
    }
    for (let i = 0; i < 10; i++) p(2 + Math.floor(r() * 16), 2 + Math.floor(r() * 14), 1, 1, '#1e4a16');
  } else {
    // Conifer / cycad-ish tall tree
    p(9, 19, 2, 6, '#5a3e22');
    const cols = ['#1e4a24', '#2a6232', '#3a7a3e'];
    for (let y = 0; y < 20; y++) {
      const wdt = Math.min(9, 1 + Math.floor(y * 0.55) + (y % 4 === 3 ? -1 : 0));
      for (let x = -wdt; x <= wdt; x++) {
        const col = x < -wdt / 3 ? cols[0] : x > wdt / 2 ? cols[0] : (y % 4 === 0 ? cols[2] : cols[1]);
        p(10 + x, y + 1, 1, 1, col);
      }
    }
  }
  return c;
}

function makeBush(r) {
  const [c, ctx] = makeCanvas(12, 9);
  const cols = ['#2e6a20', '#3f8a2a', '#57a838'];
  for (let pass = 0; pass < 3; pass++) {
    for (let i = 0; i < 18; i++) {
      const x = 2 + Math.floor(r() * 8) - pass + 1, y = 2 + Math.floor(r() * 6) - pass + 1;
      ctx.fillStyle = cols[pass]; ctx.fillRect(x, y, 2, 2);
    }
  }
  return c;
}

// ---------------- Building sprites (procedural pixel art) ----------------
const BSPR = {}; // key -> {canvas, ox, oy} where oy = extra height above footprint

function P(ctx) {
  return {
    r(x, y, w, h, c) { ctx.fillStyle = c; ctx.fillRect(x, y, w, h); },
    box(x, y, w, h, fill, line) { ctx.fillStyle = line; ctx.fillRect(x, y, w, h); ctx.fillStyle = fill; ctx.fillRect(x + 1, y + 1, w - 2, h - 2); },
    px(x, y, c) { ctx.fillStyle = c; ctx.fillRect(x, y, 1, 1); },
  };
}

// Tiny 3x5 pixel font for signs
const FONT3 = {
  A: '010101111101101', B: '110101110101110', C: '011100100100011', D: '110101101101110', E: '111100110100111',
  F: '111100110100100', G: '011100101101011', H: '101101111101101', I: '111010010010111', J: '001001001101010',
  K: '101101110101101', L: '100100100100111', M: '101111111101101', N: '110101101101101', O: '010101101101010',
  P: '110101110100100', Q: '010101101110011', R: '110101110101101', S: '011100010001110', T: '111010010010010',
  U: '101101101101111', V: '101101101101010', W: '101101111111101', X: '101101010101101', Y: '101101010010010',
  Z: '111001010100111', ' ': '000000000000000', '!': '010010010000010', '0': '111101101101111', '1': '010110010010111',
  '+': '000010111010000', '$': '011110010011110', '-': '000000111000000',
};
function drawText3(ctx, text, x, y, col) {
  ctx.fillStyle = col;
  for (const ch of text.toUpperCase()) {
    const g = FONT3[ch] || FONT3[' '];
    for (let i = 0; i < 15; i++) if (g[i] === '1') ctx.fillRect(x + (i % 3), y + Math.floor(i / 3), 1, 1);
    x += 4;
  }
}

function roofThatch(p, x, y, w, h) {
  // Visitor-center style thatched roof
  for (let yy = 0; yy < h; yy++) {
    const inset = Math.max(0, Math.floor((h - yy) * 0.25));
    for (let xx = inset; xx < w - inset; xx++) {
      const stripe = (xx + yy * 2) % 5 === 0;
      const shade = yy < 2 ? '#c8a050' : yy > h - 3 ? '#7a5a2a' : stripe ? '#9a7432' : '#b08a42';
      p.px(x + xx, y + yy, shade);
    }
  }
  p.r(x, y + h - 1, w, 1, '#4a3418');
}

function roofFlat(p, x, y, w, h, base, light, dark) {
  p.r(x, y, w, h, dark);
  p.r(x + 1, y, w - 2, h - 1, base);
  p.r(x + 1, y, w - 2, 1, light);
}

function roofGable(p, x, y, w, h, base, light, dark) {
  // Front-facing gable roof viewed top-down 3/4: lighter top band, darker lower band
  p.r(x, y, w, h, dark);
  p.r(x + 1, y + 1, w - 2, Math.floor(h / 2), light);
  p.r(x + 1, y + 1 + Math.floor(h / 2), w - 2, h - 2 - Math.floor(h / 2), base);
  for (let xx = x + 2; xx < x + w - 2; xx += 3) p.r(xx, y + 1 + Math.floor(h / 2), 1, h - 2 - Math.floor(h / 2), dark);
}

function windowRow(p, x, y, w, n, lit) {
  const gap = Math.floor(w / n);
  for (let i = 0; i < n; i++) {
    const wx = x + i * gap + Math.floor(gap / 2) - 1;
    p.r(wx, y, 3, 3, '#1a2430');
    p.r(wx, y, 3, 1, lit ? '#f8e080' : '#6ab0d8');
    p.px(wx, y + 1, lit ? '#f0c060' : '#3a7aa8');
  }
}

function buildBuildingSprites() {
  const make = (key, extraH, fn) => {
    const b = BUILDINGS[key];
    const W = b.w * TILE, H = b.h * TILE + extraH;
    const [c, ctx] = makeCanvas(W, H);
    fn(P(ctx), W, H, ctx);
    BSPR[key] = { canvas: c, extra: extraH };
  };

  make('gate', 22, (p, W, H, ctx) => {
    // ground pad
    p.r(0, H - 10, W, 10, '#8a7a5a');
    // pillars
    for (const px of [2, W - 12]) {
      p.r(px, 6, 10, H - 8, '#3a2814');
      p.r(px + 1, 6, 8, H - 9, '#6a4a28');
      p.r(px + 2, 6, 2, H - 9, '#8a6a3a');
      for (let y = 10; y < H - 4; y += 6) p.r(px + 1, y, 8, 1, '#4a3418');
      // torch
      p.r(px + 3, 0, 4, 6, '#3a2814'); p.r(px + 4, 1, 2, 3, '#f8c040');
    }
    // crossbeam + sign
    p.r(0, 8, W, 12, '#2a1a0c');
    p.r(1, 9, W - 2, 10, '#5a3a1c');
    p.r(6, 10, W - 12, 8, '#c84a1e');
    p.r(7, 11, W - 14, 6, '#e8a028');
    drawText3(ctx, 'PARK', Math.floor(W / 2) - 8, 11, '#2a1408');
    // gate doors (open)
    p.r(12, 20, 4, H - 22, '#4a3418'); p.r(W - 16, 20, 4, H - 22, '#4a3418');
  });

  make('visitor', 18, (p, W, H, ctx) => {
    // walls
    p.box(3, 26, W - 6, H - 28, '#e8dcc0', '#5a4a34');
    windowRow(p, 6, 32, W - 12, 7, false);
    windowRow(p, 6, 42, W - 12, 7, false);
    p.r(W / 2 - 4, H - 12, 8, 10, '#3a2a18'); p.r(W / 2 - 3, H - 11, 6, 9, '#6ab0d8');
    // banners
    p.r(8, 46, 3, 8, '#c8381e'); p.r(W - 11, 46, 3, 8, '#c8381e');
    // big thatched roofs (two cones like the JP visitor center)
    roofThatch(p, 0, 4, 26, 24);
    roofThatch(p, 22, 0, 26, 28);
    p.r(34, 0, 2, 3, '#4a3418');
    p.r(11, 2, 2, 4, '#4a3418');
  });

  make('restaurant', 12, (p, W, H, ctx) => {
    p.box(2, 16, W - 4, H - 22, '#f0e4c8', '#5a4a34');
    windowRow(p, 4, 22, W - 8, 4, false);
    p.r(13, H - 16, 6, 10, '#5a3a1c');
    roofGable(p, 0, 2, W, 16, '#3a8a4a', '#58a860', '#24603a');
    // sign
    p.r(10, 0, 12, 5, '#3a2a18'); p.r(11, 1, 10, 3, '#f8e060');
    p.px(13, 2, '#c8381e'); p.px(16, 2, '#c8381e'); p.px(19, 2, '#c8381e');
    // tables w/ umbrellas
    p.r(1, H - 6, 6, 1, '#e04838'); p.r(3, H - 5, 1, 4, '#6a4a28');
    p.r(W - 7, H - 6, 6, 1, '#f8e060'); p.r(W - 5, H - 5, 1, 4, '#6a4a28');
  });

  make('shop', 10, (p, W, H, ctx) => {
    p.box(2, 14, W - 4, H - 18, '#e8d8b8', '#5a4a34');
    p.r(5, 24, 10, 7, '#6ab0d8'); p.r(5, 24, 10, 1, '#a8d8f0');
    p.r(19, 24, 7, 12, '#5a3a1c');
    // awning stripes
    for (let x = 0; x < W; x++) p.r(x, 14, 1, 6, (Math.floor(x / 3) % 2) ? '#f8e060' : '#e04838');
    p.r(0, 20, W, 1, '#8a2a18');
    roofFlat(p, 1, 2, W - 2, 12, '#c8381e', '#e85a3a', '#7a2010');
    drawText3(ctx, 'GIFTS', 6, 5, '#f8e8c0');
    // plush dino in window
    p.r(8, 27, 4, 3, '#58a860'); p.px(12, 27, '#58a860');
  });

  make('restroom', 6, (p, W, H) => {
    p.box(1, 6, W - 2, H - 7, '#c8d8e8', '#3a4a5a');
    roofFlat(p, 0, 1, W, 6, '#3878c8', '#58a0e0', '#204878');
    p.r(6, 12, 4, 8, '#3a4a5a');
    p.px(3, 9, '#204878'); p.px(12, 9, '#c8381e');
  });

  make('viewing', 8, (p, W, H) => {
    // wooden deck
    p.r(0, 10, W, H - 12, '#6a4a28');
    for (let y = 12; y < H - 2; y += 3) p.r(1, y, W - 2, 1, '#8a6a3a');
    // legs
    p.r(2, H - 3, 2, 3, '#3a2814'); p.r(W - 4, H - 3, 2, 3, '#3a2814');
    // railing
    p.r(0, 8, W, 2, '#3a2814'); p.r(0, 8, 2, H - 10, '#3a2814'); p.r(W - 2, 8, 2, H - 10, '#3a2814');
    for (let x = 3; x < W - 2; x += 4) p.r(x, 6, 1, 4, '#3a2814');
    p.r(0, 5, W, 1, '#5a3a1c');
    // telescopes
    p.r(8, 1, 2, 6, '#3a3a3a'); p.r(6, 0, 6, 2, '#5a5a6a');
    p.r(22, 1, 2, 6, '#3a3a3a'); p.r(20, 0, 6, 2, '#5a5a6a');
  });

  make('hotel', 26, (p, W, H) => {
    p.box(2, 22, W - 4, H - 24, '#8a6a42', '#3a2814');
    for (let fl = 0; fl < 3; fl++) {
      windowRow(p, 4, 26 + fl * 9, W - 8, 8, true);
      p.r(3, 31 + fl * 9, W - 6, 1, '#6a4a28');
    }
    p.r(W / 2 - 4, H - 10, 8, 8, '#2a1a0c'); p.r(W / 2 - 3, H - 9, 6, 7, '#f8d080');
    roofGable(p, 0, 4, W, 20, '#2a5a3a', '#3a7a4a', '#163a24');
    p.r(8, 0, 4, 8, '#5a5a5a'); p.r(W - 14, 0, 4, 8, '#5a5a5a');
  });

  make('shelter', 4, (p, W, H) => {
    // half-buried concrete bunker
    p.r(0, 8, W, H - 8, '#5a6a4a');
    p.box(2, 4, W - 4, H - 6, '#8a8a84', '#3a3a38');
    p.r(3, 5, W - 6, 2, '#a8a8a0');
    for (let x = 4; x < W - 4; x += 4) { p.r(x, 9, 2, 3, '#f8d040'); p.r(x + 2, 9, 2, 3, '#1a1a1a'); }
    p.r(10, 16, 12, H - 18, '#3a3a38'); p.r(11, 17, 10, H - 20, '#5a5a58');
    p.r(15, 17, 2, H - 20, '#3a3a38');
  });

  make('power', 18, (p, W, H, ctx) => {
    p.box(2, 20, W - 4, H - 22, '#9a9a90', '#3a3a38');
    p.r(3, 21, W - 6, 2, '#c8c8c0');
    // cooling tower
    p.r(26, 0, 16, 34, '#3a3a38'); p.r(27, 1, 14, 32, '#c8c8c0'); p.r(28, 1, 4, 32, '#e0e0d8'); p.r(37, 1, 3, 32, '#a8a8a0');
    p.r(26, 0, 16, 2, '#5a5a58');
    // building details
    p.r(6, 28, 14, 10, '#5a5a58'); p.r(7, 29, 12, 8, '#f8d040');
    // lightning bolt
    const bolt = [[13, 30], [12, 31], [11, 32], [12, 33], [13, 33], [12, 34], [11, 35]];
    for (const [x, y] of bolt) p.px(x, y, '#1a1a1a');
    // pipes
    p.r(4, 44, W - 8, 2, '#5a5a58');
    p.r(8, 14, 3, 8, '#5a5a58'); p.r(16, 10, 3, 12, '#5a5a58');
    p.r(8, 14, 3, 1, '#c8381e'); p.r(16, 10, 3, 1, '#c8381e');
    for (let x = 4; x < W - 4; x += 6) { p.r(x, H - 4, 3, 2, '#f8d040'); }
  });

  make('pylon', 16, (p, W, H) => {
    p.r(7, 2, 2, H - 2, '#5a5a58');
    p.r(4, H - 2, 8, 2, '#3a3a38');
    for (let y = 4; y < H - 2; y += 4) { p.px(6, y, '#5a5a58'); p.px(9, y + 2, '#5a5a58'); p.px(6, y + 2, '#5a5a58'); p.px(9, y, '#5a5a58'); }
    p.r(2, 4, 12, 1, '#3a3a38'); p.r(3, 9, 10, 1, '#3a3a38');
    p.px(2, 5, '#a8d8f0'); p.px(13, 5, '#a8d8f0'); p.px(3, 10, '#a8d8f0'); p.px(12, 10, '#a8d8f0');
  });

  make('backup', 8, (p, W, H) => {
    p.box(2, 8, W - 4, H - 10, '#6a7a5a', '#2a3424');
    p.r(3, 9, W - 6, 2, '#8a9a7a');
    for (let x = 5; x < W - 5; x += 3) p.r(x, 14, 1, 10, '#3a4434');
    p.r(22, 0, 4, 10, '#3a3a38'); p.r(21, 0, 6, 2, '#5a5a58');
    p.r(6, H - 10, 8, 6, '#f8d040'); p.px(9, H - 9, '#1a1a1a'); p.px(10, H - 8, '#1a1a1a'); p.px(9, H - 7, '#1a1a1a');
  });

  make('ranger', 12, (p, W, H, ctx) => {
    p.box(2, 16, W - 4, H - 18, '#7a5a32', '#3a2814');
    for (let y = 18; y < H - 2; y += 3) p.r(3, y, W - 6, 1, '#6a4a28');
    windowRow(p, 4, 22, 12, 2, false);
    p.r(20, H - 14, 7, 12, '#3a2814'); p.r(21, H - 13, 5, 11, '#5a3a1c');
    roofGable(p, 0, 4, W, 14, '#4a6a32', '#6a8a42', '#2a4a1e');
    // flag
    p.r(28, 0, 1, 10, '#3a3a3a'); p.r(29, 0, 3, 3, '#f8d040'); p.r(29, 1, 3, 1, '#c8381e');
    drawText3(ctx, 'R', 13, 9, '#f8f0c0');
  });

  make('maint', 8, (p, W, H, ctx) => {
    p.box(1, 10, W - 2, H - 12, '#8a8a8a', '#3a3a3a');
    p.r(4, 18, 18, H - 20, '#3a3a3a');
    for (let y = 19; y < H - 2; y += 2) p.r(5, y, 16, 1, '#f08020');
    roofFlat(p, 0, 2, W, 9, '#5a5a5a', '#7a7a7a', '#2a2a2a');
    for (let x = 1; x < W; x += 4) { p.r(x, 2, 2, 2, '#f8d040'); }
    // wrench sign
    p.r(24, 20, 5, 5, '#f8d040'); p.px(25, 21, '#1a1a1a'); p.px(26, 22, '#1a1a1a'); p.px(27, 23, '#1a1a1a');
  });

  make('helipad', 0, (p, W, H, ctx) => {
    p.r(1, 1, W - 2, H - 2, '#4a4a48');
    p.r(2, 2, W - 4, H - 4, '#6a6a68');
    // circle
    const cx = W / 2, cy = H / 2;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const d = Math.sqrt((x - cx + 0.5) ** 2 + (y - cy + 0.5) ** 2);
      if (d > 17 && d < 19.5) p.px(x, y, '#f8d040');
    }
    // H
    p.r(cx - 7, cy - 8, 3, 16, '#e8e8e0'); p.r(cx + 4, cy - 8, 3, 16, '#e8e8e0'); p.r(cx - 4, cy - 1, 8, 3, '#e8e8e0');
    for (const [x, y] of [[3, 3], [W - 5, 3], [3, H - 5], [W - 5, H - 5]]) p.r(x, y, 2, 2, '#e04838');
  });

  make('vet', 10, (p, W, H, ctx) => {
    p.box(2, 14, W - 4, H - 16, '#f0f0e8', '#4a5a5a');
    windowRow(p, 4, 20, W - 8, 3, false);
    p.r(12, H - 12, 8, 10, '#3a8a7a'); p.r(13, H - 11, 6, 9, '#a8e0d8');
    roofFlat(p, 0, 2, W, 13, '#3aa090', '#5ac8b0', '#1e6a5a');
    // green cross
    p.r(14, 4, 4, 9, '#f0f0e8'); p.r(11, 7, 10, 3, '#f0f0e8');
    p.r(15, 5, 2, 7, '#3ac85a'); p.r(12, 8, 8, 1, '#3ac85a');
  });

  make('hatchery', 14, (p, W, H, ctx) => {
    p.box(2, 16, W - 4, H - 18, '#e8ecf0', '#3a4a5a');
    windowRow(p, 4, 22, W - 8, 6, true);
    p.r(W / 2 - 4, H - 12, 8, 10, '#3a4a5a'); p.r(W / 2 - 3, H - 11, 6, 9, '#a8d8f0');
    roofFlat(p, 0, 4, W, 13, '#a8b0b8', '#c8d0d8', '#5a6470');
    // glass domes
    for (const dx of [10, 36]) {
      for (let y = 0; y < 9; y++) for (let x = -7; x <= 7; x++) {
        if (x * x / 49 + (y - 9) * (y - 9) / 81 <= 1) p.px(dx + x, y + 1, (x < -2 && y < 6) ? '#d8f0ff' : '#7ac0e8');
      }
      p.r(dx - 7, 9, 15, 1, '#3a4a5a');
    }
    // egg
    p.r(22, 3, 4, 6, '#f0e8c8'); p.r(23, 2, 2, 1, '#f0e8c8'); p.px(23, 5, '#8ab0a0'); p.px(24, 7, '#8ab0a0');
  });

  make('feeder_h', 4, (p, W, H) => {
    p.r(1, 8, 14, 10, '#5a3a1c');
    p.r(2, 9, 12, 8, '#8a6a3a');
    // hay
    for (let i = 0; i < 26; i++) p.px(2 + (i * 7) % 12, 4 + (i * 5) % 7, i % 3 ? '#e8c860' : '#c8a040');
    p.r(1, 18, 2, 2, '#3a2814'); p.r(13, 18, 2, 2, '#3a2814');
  });

  make('feeder_c', 8, (p, W, H) => {
    // goat on a tether post
    p.r(7, 0, 2, H - 2, '#5a3a1c');
    p.r(4, 0, 8, 2, '#3a2814');
    // goat
    p.r(2, 14, 7, 4, '#e8e4dc'); p.r(8, 12, 3, 3, '#e8e4dc'); p.px(10, 11, '#8a8478'); p.px(10, 13, '#1a1a1a');
    p.r(2, 18, 1, 3, '#5a5a5a'); p.r(7, 18, 1, 3, '#5a5a5a');
    p.px(1, 14, '#e8e4dc');
    // chain
    for (let y = 2; y < 12; y += 2) p.px(9, y, '#9a9a9a');
    p.r(0, H - 2, W, 2, '#6a5a3a');
  });

  make('lamp', 10, (p, W, H) => {
    p.r(7, 4, 2, H - 4, '#2a2a2a');
    p.r(5, H - 2, 6, 2, '#3a3a3a');
    p.r(5, 0, 6, 5, '#2a2a2a'); p.r(6, 1, 4, 3, '#f8e080');
  });

  make('tour', 14, (p, W, H, ctx) => {
    // open-sided station with a big canopy, like a ride platform
    p.r(1, 14, W - 2, H - 16, '#8a7a5a');
    p.r(2, 15, W - 4, H - 18, '#b8a878');
    for (let x = 4; x < W - 3; x += 8) p.r(x, 14, 2, H - 16, '#5a3a1c');
    roofGable(p, 0, 4, W, 12, '#c8381e', '#e85a3a', '#7a2010');
    p.r(4, 0, W - 8, 6, '#2a1a0c'); p.r(5, 1, W - 10, 4, '#f8d040');
    drawText3(ctx, 'TOUR', 8, 1, '#2a1408');
    // turnstile
    p.r(12, H - 8, 8, 2, '#5a5a5a');
  });

  make('aviary', 12, (p, W, H, ctx) => {
    // concrete base ring
    p.r(2, H - 14, W - 4, 12, '#6a6a64'); p.r(3, H - 13, W - 6, 2, '#9a9a92');
    // inner habitat: rocks and trees
    p.r(6, H - 26, W - 12, 13, '#4f8a32');
    p.r(10, H - 24, 12, 6, '#7a7468'); p.r(12, H - 26, 6, 3, '#9c968a');
    p.r(W - 26, H - 22, 8, 7, '#2e6a20'); p.r(W - 24, H - 26, 4, 5, '#3f8a2a');
    // dome mesh
    const cx = W / 2, cy = H - 12, rx = W / 2 - 2, ry = 58;
    for (let y = 0; y < ry; y++) {
      const t = 1 - y / ry;
      const half = Math.round(rx * Math.sqrt(1 - t * t));
      const yy = cy - ry + y;
      p.px(cx - half, yy, '#3a3a3a'); p.px(cx + half - 1, yy, '#3a3a3a');
      if (y % 7 === 0) for (let x = cx - half; x < cx + half; x++) if ((x + y) % 2 === 0) p.px(x, yy, 'rgba(220,230,235,0.55)');
      if (y < 3) for (let x = cx - half; x < cx + half; x++) p.px(x, yy, '#3a3a3a');
    }
    for (let k = -3; k <= 3; k++) {
      // meridians
      for (let y = 0; y < ry; y++) {
        const t = 1 - y / ry;
        const half = rx * Math.sqrt(1 - t * t);
        const x = Math.round(cx + half * k / 3.5);
        if (y % 2 === 0) p.px(x, cy - ry + y, 'rgba(200,210,215,0.7)');
      }
    }
    p.r(cx - 3, cy - ry - 2, 6, 3, '#3a3a3a');
    // sign
    p.r(W / 2 - 14, H - 10, 28, 7, '#2a1a0c'); drawText3(ctx, 'AVIARY', W / 2 - 12, H - 9, '#f8d040');
  });

  make('lagoon', 14, (p, W, H, ctx) => {
    // bleachers along the top, deep pool below
    p.r(0, 0, W, 22, '#5a5a58');
    for (let r = 0; r < 4; r++) { p.r(2, 2 + r * 5, W - 4, 3, r % 2 ? '#8a8a84' : '#a8a8a0'); }
    for (let x = 4; x < W - 4; x += 3) p.px(x, 3 + ((x * 7) % 4) * 5, ['#e04838', '#3878c8', '#f8e060', '#68a088'][x % 4]);
    p.r(0, 22, W, H - 22, '#3a3a38');
    p.r(3, 25, W - 6, H - 28, '#1e5a8a');
    p.r(5, 27, W - 10, H - 32, '#18487a');
    for (let k = 0; k < 30; k++) p.px(6 + (k * 37) % (W - 12), 28 + (k * 23) % (H - 34), '#3a88c8');
    p.r(0, 22, W, 2, '#c8c8c0');
    drawText3(ctx, 'LAGOON', W / 2 - 12, H - 7, '#f8d040');
  });

  make('siren', 14, (p, W, H) => {
    p.r(7, 6, 2, H - 6, '#5a5a58');
    p.r(4, H - 2, 8, 2, '#3a3a38');
    p.r(3, 0, 10, 7, '#3a3a38'); p.r(4, 1, 8, 5, '#c8381e'); p.r(5, 2, 2, 2, '#f8a080');
    p.r(1, 2, 2, 3, '#8a8a88'); p.r(13, 2, 2, 3, '#8a8a88');
  });
}

let PTERO = null;
let JEEP_R = null, JEEP_L = null, JEEP_WRECK = null;
// Egg sprite
let EGG_SPR = null, HELI_SPR = null, HELI_SHADOW = null, RUBBLE_SPR = null, JEEP_SPR = null;
function buildMisc() {
  EGG_SPR = gridToCanvas([
    '  OOO  ',
    ' OWWWO ',
    'OWWSWWO',
    'OWWWWSO',
    'OWSWWWO',
    'OWWWWWO',
    ' OWWWO ',
    '  OOO  ',
  ], { O: '#3a3428', W: '#f0e8c8', S: '#8ab07a' });

  // Helicopter (top-down-ish side view), facing right, 28x14
  HELI_SPR = gridToCanvas([
    'OOOOOOOOOOOOOOOOOOOOOOOOOOOO',
    '             OO',
    '        OOOOOOOOOOO',
    'OO     OYYYYYYYYYGGGO',
    'OYOOOOOYYYYYYYYYYGGGGO',
    ' OYYYYYYYYYYYYYYYGGGGGO',
    '  OOOOOYYYYYYYYYYYYYYYO',
    '       OYYKKYYYYYYYYYO',
    '        OOOOOOOOOOOOO',
    '          O      O',
    '       OOOOOOOOOOOOOO',
  ], { O: '#1a1a1a', Y: '#e8b020', G: '#88c8e8', K: '#2a2a2a' });
  HELI_SHADOW = tintCanvas(HELI_SPR, '#000000', 1);

  // Tour jeep: green body, yellow stripe, red band (classic park explorer look)
  JEEP_R = gridToCanvas([
    '     OOOOOOO    ',
    '    OjjOjjjjO   ',
    '   OjjjOjjjjjO  ',
    'OOOOOOOOOOOOOOOO',
    'OGGGGGGGGGGGGGLO',
    'ORRRRRRRRRRRRRRO',
    'OYYYYYYYYYYYYYYO',
    'OGGOOOGGGGGOOOGO',
    ' OOkkkOOOOOkkkO ',
    '   OkO     OkO  ',
  ], { O: '#1a1a1a', G: '#3a8a3a', R: '#c8381e', Y: '#f8d040', j: '#88c8e8', k: '#3a3a3a', L: '#f8f0a0' });
  JEEP_L = flipCanvas(JEEP_R);
  JEEP_WRECK = (() => { const [c, ctx] = makeCanvas(16, 16); ctx.translate(8, 8); ctx.rotate(Math.PI / 2); ctx.drawImage(JEEP_R, -8, -5); return c; })();

  const ptPal = { O: '#2a1a14', B: '#9a6a4a', D: '#6a4a32', S: '#c84a2a' };
  const up = gridToCanvas([
    'OO             OO',
    'OBOO         OOBO',
    ' OBBOO     OOBBO ',
    '  ODBBOOOOOBBDO  ',
    '   OODBBBBBDOOSSO',
    '     OOOBBOOOOO  ',
    '       OOO       ',
  ], ptPal);
  const down = gridToCanvas([
    '                 ',
    '                 ',
    '      OOOOO   SSO',
    '   OOOBBBBBOOOO  ',
    ' OOBBDBBBBBDBBOO ',
    'OBBOO OOBBO  OOBO',
    'OO      O      OO',
  ], ptPal);
  PTERO = { R: [up, down], L: [flipCanvas(up), flipCanvas(down)] };

  RUBBLE_SPR = gridToCanvas([
    '    O    O      ',
    '  OaO  OaaO     ',
    ' OaaaOOaabaO  O ',
    'OabaaaaabaaaOOaO',
    ' OOOOOOOOOOOOOOO',
  ], { O: '#3a3a38', a: '#8a8a84', b: '#b0b0a8' });
}

// Volcano: a cone ~7 tiles wide, crater at the top
let VOLCANO_SPR = null;
function buildVolcano() {
  const W = 112, H = 84;
  const [c, ctx] = makeCanvas(W, H);
  const r = mulberry32(99);
  const cx = W / 2;
  for (let y = 0; y < H; y++) {
    const t = y / (H - 1);
    const half = 9 + Math.pow(t, 0.85) * (W / 2 - 9);
    for (let x = Math.floor(cx - half); x < Math.ceil(cx + half); x++) {
      const rel = (x - (cx - half)) / (half * 2); // 0 left .. 1 right
      let col;
      if (rel < 0.18) col = '#7a6656';
      else if (rel < 0.42) col = '#5e4c40';
      else if (rel < 0.75) col = '#4a3c34';
      else col = '#382c26';
      // ridges
      const ridge = Math.sin(x * 0.45 + y * 0.12) > 0.82;
      if (ridge) col = rel < 0.5 ? '#86705e' : '#2e241f';
      if (r() < 0.05) col = '#2a201c';
      if (y > H - 10 && r() < 0.25 + (y - (H - 10)) * 0.07) col = r() < 0.5 ? '#4f8a32' : '#3e7228';
      ctx.fillStyle = col; ctx.fillRect(x, y, 1, 1);
    }
  }
  // outline edges
  ctx.fillStyle = '#1e1714';
  for (let y = 0; y < H - 8; y++) {
    const t = y / (H - 1);
    const half = 9 + Math.pow(t, 0.85) * (W / 2 - 9);
    ctx.fillRect(Math.floor(cx - half), y, 1, 1); ctx.fillRect(Math.ceil(cx + half) - 1, y, 1, 1);
  }
  // crater rim + glow
  ctx.fillStyle = '#1e1714'; ctx.fillRect(cx - 10, 0, 20, 5);
  ctx.fillStyle = '#c83a10'; ctx.fillRect(cx - 8, 1, 16, 3);
  ctx.fillStyle = '#f8a030'; ctx.fillRect(cx - 5, 1, 10, 2);
  ctx.fillStyle = '#f8e070'; ctx.fillRect(cx - 2, 1, 4, 1);
  // old lava channels
  for (const [sx, len] of [[-4, 30], [3, 44], [7, 22]]) {
    let x = cx + sx;
    for (let y = 4; y < 4 + len; y++) {
      x += (r() - 0.5) * 1.6 + sx * 0.03;
      ctx.fillStyle = '#6a2a16'; ctx.fillRect(Math.round(x), y, 2, 1);
    }
  }
  VOLCANO_SPR = c;
}

// Toolbar icons (24x24) generated from sprites
const ICONS = {};
function iconFrom(src, bg) {
  const S = 24;
  const [c, ctx] = makeCanvas(S, S);
  if (bg) { ctx.fillStyle = bg; ctx.fillRect(0, 0, S, S); }
  const fit = Math.min((S - 2) / src.width, (S - 2) / src.height);
  const s2 = fit >= 2 ? 2 : fit >= 1 ? 1 : fit >= 0.5 ? 0.5 : 0.25;
  const w = Math.round(src.width * s2), h = Math.round(src.height * s2);
  ctx.drawImage(src, Math.floor((S - w) / 2), Math.floor((S - h) / 2), w, h);
  return c;
}

function buildIcons() {
  for (const k of Object.keys(BSPR)) ICONS[k] = iconFrom(BSPR[k].canvas);
  for (const k of Object.keys(DINO_SPRITES)) ICONS['dino_' + k] = iconFrom(DINO_SPRITES[k].right[0]);
  const mk = (rows, pal) => iconFrom(gridToCanvas(rows, pal));
  ICONS.inspect = mk([
    '   OOOO    ',
    '  OjjjjO   ',
    ' OjccjjjO  ',
    ' OjcjjjjO  ',
    ' OjjjjjjO  ',
    ' OjjjjjjO  ',
    '  OjjjjO   ',
    '   OOOOOO  ',
    '       OBO ',
    '        OBO',
    '         OO',
  ], { O: '#1a1a1a', j: '#70b8f0', c: '#f0f8ff', B: '#6a4a28' });
  ICONS.path = mk([
    'GGGGddddGGGG',
    'GGGdDddDdGGG',
    'GGGddddddGGG',
    'GGdDddddDdGG',
    'GGddddddddGG',
    'GdddDddddDdG',
    'GddddddddddG',
    'dDdddddddDdd',
  ], { G: '#4f8a32', d: '#c8a878', D: '#a88858' });
  ICONS.fence = mk([
    'O    O    O ',
    'OyyyyOyyyyO ',
    'O    O    O ',
    'OyyyyOyyyyO ',
    'O    O    O ',
    'O    O    O ',
    'O    O    O ',
  ], { O: '#3a3a3a', y: '#f8e040' });
  ICONS.paddock = mk([
    'OyyyyOyyyyO',
    'y         y',
    'y  ggggg  y',
    'O  ggggg  O',
    'y  ggggg  y',
    'y         y',
    'OyyyyOyyyyO',
  ], { O: '#3a3a3a', y: '#f8e040', g: '#58a838' });
  ICONS.wall = mk([
    'OOOOOOOOOOOO',
    'OaaaaOaaaaaO',
    'OOOOOOOOOOOO',
    'OaaOaaaaaOaO',
    'OOOOOOOOOOOO',
    'OaaaaaOaaaaO',
    'OOOOOOOOOOOO',
  ], { O: '#4a4a48', a: '#b0b0a8' });
  ICONS.demolish = mk([
    '      OOOO  ',
    '     OaaaaO ',
    '    OaaaaaaO',
    '    OaaaaaaO',
    '   OOOaaaaO ',
    '  OBO OOOO  ',
    ' OBO        ',
    'OBO         ',
    'OO          ',
  ], { O: '#1a1a1a', a: '#8a8a8a', B: '#8a5a2a' });
  ICONS.trees = iconFrom(TILES.trees[1]);
  ICONS.track = mk([
    'GGaaaaaaaGGG',
    'GGaaayaaaGGG',
    'GGaaaaaaaGGG',
    'GGaaayaaaGGG',
    'GGaaaaaaaGGG',
    'GGaaayaaaGGG',
  ], { G: '#4f8a32', a: '#5a5a58', y: '#f8d040' });
  ICONS.clear = mk([
    '   OOOOOOO  ',
    '  OyyyyyyyO ',
    ' OyyOOOOyyO ',
    'OOOOOOOOOOOO',
    'OaOOaOOaOOaO',
    ' OOOOOOOOOO ',
  ], { O: '#1a1a1a', y: '#f8c020', a: '#5a5a5a' });
  ICONS.hatch = iconFrom(EGG_SPR);
  ICONS.siren_btn = ICONS.siren;
}

function buildAllSprites() {
  buildDinoSprites();
  buildPeople();
  buildTiles();
  buildBuildingSprites();
  buildMisc();
  buildVolcano();
  buildIcons();
}
