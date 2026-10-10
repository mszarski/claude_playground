// ---------- Hand-drawn front/back views (walking toward / away from the camera) ----------
// Same palette keys as the side templates. legs: [x, y, length] (drawn under the body, alternating per frame)
'use strict';

const VIEW_ART = {
  galli: {
    back: [
      '   OOO   ', '  OBBBO  ', '  OBBBO  ', '   OBO   ', '   OBO   ', '   OBO   ', '  OBBBO  ', ' OBBDBBO ',
      ' OBBDBBO ', ' OBBDBBO ', '  OBDBO  ', '   OBO   ', '   OBO   ', '    O    ',
    ],
    front: [
      '    O    ', '   ODO   ', '  OBBBO  ', ' OBBLBBO ', ' OBLLLBO ', ' OBBLBBO ', '  OBBBO  ', '   OBO   ',
      '   OBO   ', '   OBO   ', '  OBBBO  ', '  OEBEO  ', '  OBBBO  ', '   OLO   ', '    O    ',
    ],
    legsB: [[2, 9, 4], [5, 9, 4]], legsF: [[2, 5, 5], [5, 5, 5]],
  },
  raptor: {
    back: [
      '   OOO   ', '  OBBBO  ', '  OBSBO  ', '   OBO   ', '  OBBBO  ', ' OBSBSBO ', ' OBBSBBO ', ' OBSBSBO ',
      '  OBBBO  ', '   OBO   ', '   OBO   ', '   OSO   ', '   OBO   ', '    O    ',
    ],
    front: [
      '    O    ', '   OSO   ', '  OBBBO  ', ' OBSLSBO ', ' OBBLBBO ', ' OBBLBBO ', '  OBBBO  ', '  OBBBO  ',
      ' OBEBEBO ', ' OBBBBBO ', '  OTTTO  ', '   OOO   ',
    ],
    legsB: [[2, 7, 4], [5, 7, 4]], legsF: [[2, 5, 5], [5, 5, 5]],
  },
  dilo: {
    back: [
      '  OO OO  ', '  OSOSO  ', '  OBBBO  ', '   OBO   ', '  OBBBO  ', ' OBSBSBO ', ' OBBBBBO ', ' OBSBSBO ',
      '  OBBBO  ', '   OBO   ', '   OBO   ', '   OBO   ', '    O    ',
    ],
    front: [
      '    O    ', '   OBO   ', '  OBBBO  ', ' OBBLBBO ', ' OBBLBBO ', '  OBLBO  ', 'OSSOBOSSO', 'OSBEBEBSO',
      'OSSBBBSSO', ' OO OTO O', '    O    ',
    ],
    legsB: [[2, 7, 4], [5, 7, 4]], legsF: [[2, 4, 5], [5, 4, 5]],
  },
  para: {
    back: [
      '    OOO    ', '   OBBBO   ', '   OBSBO   ', '    OSO    ', '   OBSBO   ', '  OBBSBBO  ', ' OBBBBBBBO ', ' OBBBDBBBO ',
      ' OBBBDBBBO ', '  OBBDBBO  ', '   OBDBO   ', '    OBO    ', '    OBO    ', '     O     ',
    ],
    front: [
      '     O     ', '    OBO    ', '   OBBBO   ', '  OBBLBBO  ', ' OBBLLLBBO ', ' OBBLLLBBO ', '  OBBLBBO  ', '    OSO    ',
      '   OSSSO   ', '   OBBBO   ', '   OEBEO   ', '   OBBBO   ', '    OLO    ', '     O     ',
    ],
    legsB: [[2, 8, 4], [7, 8, 4]], legsF: [[2, 5, 5], [7, 5, 5]],
  },
  trex: {
    back: [
      '    OOOOO    ', '   OBBBBBO   ', '   OBBBBBO   ', '    OBBBO    ', '   OBBBBBO   ', '  OBBBDBBBO  ', ' OBBBBDBBBBO ', ' OBBBBDBBBBO ',
      ' OBBBBDBBBBO ', '  OBBBDBBBO  ', '   OBBDBBO   ', '    OBDBO    ', '    OBBBO    ', '     OBO     ', '     OBO     ', '      O      ',
    ],
    front: [
      '      O      ', '     OBO     ', '    OBBBO    ', '   OBBBBBO   ', '  OBBLLLBBO  ', ' OBBLLLLLBBO ', ' OBBLLLLLBBO ', '  OLBLLLBLO  ',
      '   OBBLBBO   ', '  OBBBBBBBO  ', ' OBBEBBBEBBO ', ' OBBBBBBBBBO ', '  OTTTTTTTO  ', '   OBBBBBO   ', '    OOOOO    ',
    ],
    legsB: [[3, 9, 5], [8, 9, 5]], legsF: [[3, 6, 6], [8, 6, 6]],
  },
  spino: {
    back: [
      '    OOOOO    ', '   OBBBBBO   ', '    OBBBO    ', '    OBSBO    ', '   OBSSSBO   ', '  OBBSSSBBO  ', ' OBBBSSSBBBO ', ' OBBBBSBBBBO ',
      ' OBBBBSBBBBO ', '  OBBBSBBBO  ', '   OBBSBBO   ', '    OBSBO    ', '    OBBBO    ', '     OBO     ', '     OBO     ', '      O      ',
    ],
    front: [
      '     OSO     ', '    OSSSO    ', '    OBSBO    ', '   OBBSBBO   ', '  OBBLLLBBO  ', ' OBBLLLLLBBO ', ' OBBLLLLLBBO ', '  OLBLLLBLO  ',
      '   OBBLBBO   ', '   OBBBBBO   ', '  OBEBBBEBO  ', '   OBBBBBO   ', '   OTTTTTO   ', '    OBBBO    ', '     OOO     ',
    ],
    legsB: [[3, 9, 5], [8, 9, 5]], legsF: [[3, 6, 6], [8, 6, 6]],
  },
  trike: {
    back: [
      '  OOOOOOOOOOO  ', ' OSSSSSSSSSSSO ', ' OSSSSSSSSSSSO ', '  OSSSSSSSSSO  ', '   OBBBBBBBO   ', '  OBBBBBBBBBO  ', ' OBBBBDBDBBBBO ', ' OBBBBBBBBBBBO ',
      ' OBBBBDBDBBBBO ', ' OBBBBBBBBBBBO ', '  OBBBBBBBBBO  ', '   OBBBBBBBO   ', '     OBBBO     ', '      OBO      ', '       O       ',
    ],
    front: [
      '       O       ', '      OBO      ', '   OOBBBBBOO   ', '  OBBBBBBBBBO  ', ' OBBBBBBBBBBBO ', ' OBBBBBBBBBBBO ', ' OSSSSSSSSSSSO ', 'OSSScSSSSScSSSO',
      'OSSScBBBBBcSSSO', 'OSSSOBEBEBOSSSO', ' OSSOBBBBBOSSO ', '  OOOBBcBBOOO  ', '     OBBBO     ', '      OOO      ',
    ],
    legsB: [[2, 6, 4], [11, 6, 4], [2, 10, 4], [11, 10, 4]], legsF: [[2, 4, 4], [11, 4, 4], [4, 10, 4], [9, 10, 4]],
  },
  stego: {
    back: [
      '      OOO      ', '     OBBBO     ', '      OBO      ', '    OBBBBBO    ', '   OBSBBBSBO   ', '  OBBBSBSBBBO  ', ' OBBSBBBBBSBBO ', ' OBBBSBBBSBBBO ',
      ' OBBSBBBBBSBBO ', '  OBBBSBSBBBO  ', '   OBSBBBSBO   ', '    OBBBBBO    ', '     OBSBO     ', '     OBBBO     ', '    OcO OcO    ', '     O   O     ',
    ],
    front: [
      '     OS SO     ', '    OSSOSSO    ', '   OSSSOSSSO   ', '  OBBBBBBBBBO  ', ' OBBBBBBBBBBBO ', ' OBBBLLLLLBBBO ', ' OBBLLLLLLLBBO ', '  OBBLLLLLBBO  ',
      '    OBBBBBO    ', '     OBBBO     ', '     OEBEO     ', '     OBBBO     ', '      OOO      ',
    ],
    legsB: [[2, 6, 4], [11, 6, 4], [2, 9, 4], [11, 9, 4]], legsF: [[2, 5, 4], [11, 5, 4], [4, 7, 4], [9, 7, 4]],
  },
  anky: {
    back: [
      '       OOO       ', '      OBBBO      ', '   OOOBBBBBOOO   ', '  OSBSBSBSBSBSO  ', ' OSBSBSBSBSBSBSO ', 'OSBSBSBSBSBSBSBSO', 'OSBSBSBSBSBSBSBSO', ' OSBSBSBSBSBSBSO ',
      '  OSBSBSBSBSBSO  ', '   OOOBBBBBOOO   ', '      OBBBO      ', '      OBBBO      ', '     OSSSSSO     ', '     OSSSSSO     ', '      OOOOO      ',
    ],
    front: [
      '       OOO       ', '      OSSSO      ', '      OBBBO      ', '   OOOBBBBBOOO   ', '  OSBSBSBSBSBSO  ', ' OSBSBSBSBSBSBSO ', 'OSBSBSBSBSBSBSBSO', 'OSBSBSBSBSBSBSBSO',
      ' OOOOBBBBBBBOOOO ', '    OBBBBBBBO    ', '    OBEBBBEBO    ', '     OBBBBBO     ', '      OOOOO      ',
    ],
    legsB: [[3, 4, 5], [12, 4, 5], [3, 7, 4], [12, 7, 4]], legsF: [[3, 6, 4], [12, 6, 4], [5, 8, 4], [10, 8, 4]],
  },
  brachio: {
    back: [
      '     OOO     ', '    OBBBO    ', '    OBBBO    ', '     OBO     ', '     OBO     ', '     OBO     ', '     OBO     ', '     OBO     ',
      '     OBO     ', '    OBBBO    ', '   OBBBBBO   ', '  OBBBDBBBO  ', ' OBBBBDBBBBO ', ' OBBBBDBBBBO ', ' OBBBBDBBBBO ', '  OBBBDBBBO  ',
      '   OBBDBBO   ', '    OBBBO    ', '     OBO     ', '     OBO     ', '      O      ',
    ],
    front: [
      '      O      ', '     OBO     ', '   OBBBBBO   ', '  OBBBBBBBO  ', ' OBBBBBBBBBO ', ' OBBBLLLBBBO ', ' OBBLLLLLBBO ', '  OBBLLLBBO  ',
      '   OBBBBBO   ', '     OBO     ', '     OBO     ', '     OBO     ', '     OBO     ', '    OBBBO    ', '    OEBEO    ', '    OBBBO    ', '     OOO     ',
    ],
    legsB: [[2, 12, 5], [9, 12, 5], [3, 15, 5], [8, 15, 5]], legsF: [[2, 5, 5], [9, 5, 5], [3, 7, 5], [8, 7, 5]],
  },
  ptera: {
    back: ['    OOO    ', '   OSSSO   ', '   OBBBO   ', '    OBO    ', '  OOBBBOO  ', ' OBDBBBDBO ', ' OBDBBBDBO ', '  OOBBBOO  ', '    OBO    ', '     O     '],
    front: ['    OSO    ', '   OSSSO   ', '  OOBBBOO  ', ' OBDBLBDBO ', ' OBDBLBDBO ', '  OOBBBOO  ', '   OEBEO   ', '    OBO    ', '    OLO    ', '     O     '],
    legsB: [[3, 7, 3], [6, 7, 3]], legsF: [[3, 5, 3], [6, 5, 3]],
  },
};
VIEW_ART.indom = VIEW_ART.trex;

function makeViewArt(key, front, frame) {
  const art = VIEW_ART[key];
  const sp = SPECIES[key];
  const pal = Object.assign({ T: '#f4f0e0', c: '#efe6c8' }, sp.colors);
  const rows = front ? art.front : art.back;
  const legs = front ? art.legsF : art.legsB;
  const w = Math.max(...rows.map((r) => r.length));
  let h = rows.length;
  for (const [, y, len] of legs) h = Math.max(h, y + len + 1);
  const [c, ctx] = makeCanvas(w, h);
  legs.forEach(([x, y, len], k) => {
    const L = (k % 2 === frame) ? len : len - 1;
    ctx.fillStyle = pal.D; ctx.fillRect(x, y, 2, L);
    ctx.fillStyle = pal.O; ctx.fillRect(x, y + L, 2, 1); ctx.fillRect(x - (front ? 0 : 0), y, 0, 0);
  });
  ctx.drawImage(gridToCanvas(rows, pal), 0, 0);
  return c;
}
