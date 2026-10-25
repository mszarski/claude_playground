// ---------- Shared entity helpers ----------
'use strict';

// the famous ones get famous names (and don't hand them to a Gallimimus)
const SPECIES_NAMES = {
  raptor: ['Blue', 'Charlie', 'Delta', 'Echo', 'Clever Girl', 'Big One', 'Muldoon', 'Ripper', 'Talon'],
  trex: ['Rexy', 'Roberta', 'Big Red', 'Queenie', 'Tiny', 'Sue'],
  indom: ['Indy', 'Snowflake', 'Ghost', 'Dr. Wu\'s Mistake'],
  spino: ['Sail', 'Spike', 'Riverqueen'],
};
const RESERVED_NAMES = new Set(Object.values(SPECIES_NAMES).flat());
const nameFor = (species) => {
  const own = SPECIES_NAMES[species];
  if (own && chance(0.75)) return pick(own);
  return pick(DINO_NAMES.filter((n) => !RESERVED_NAMES.has(n)));
};
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
