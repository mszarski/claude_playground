// ---------- Shared entity helpers ----------
'use strict';

const DINO_NAMES = ['Rexy', 'Blue', 'Delta', 'Echo', 'Charlie', 'Clever Girl', 'Big One', 'Spike', 'Cera', 'Littlefoot', 'Bumpy', 'Ducky', 'Petrie', 'Sarah', 'Roberta', 'Tiny', 'Nibbles', 'Chomper', 'Mabel', 'Gertie', 'Bob', 'Kevin', 'Pebbles', 'Doris', 'Hammond', 'Muldoon', 'Grant', 'Ellie', 'Ian', 'Dennis', 'Lex', 'Tim', 'Henry', 'Nedry Jr', 'Sprinkles', 'Mr. Teeth', 'Steve', 'Bubbles', 'Ziggy', 'Moxie'];

function stepAlong(e, dt, speed, passFn, world) {
  if (!e.path || e.pi >= e.path.length) return true;
  const [wx, wy] = e.path[e.pi];
  if (passFn && !passFn.call(world, wx, wy) && !e.allowGoalBlock) { e.path = null; return true; }
  const tx = wx + 0.5, ty = wy + 0.5;
  const dx = tx - e.x, dy = ty - e.y;
  const d = Math.sqrt(dx * dx + dy * dy);
  const s = speed * dt;
  if (Math.abs(dx) > 0.02) e.facing = dx > 0 ? 1 : -1;
  e.vdir = Math.abs(dy) > Math.abs(dx) * 1.5 ? (dy > 0 ? 1 : -1) : 0;
  if (d <= s) { e.x = tx; e.y = ty; e.pi++; return e.pi >= e.path.length; }
  e.x += (dx / d) * s; e.y += (dy / d) * s;
  e.moving = true;
  return false;
}

let _eid = 1;
