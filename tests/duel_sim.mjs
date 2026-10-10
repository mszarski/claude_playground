// Simulated rater for the adaptive listening duels (deploy/listen_duel_page.html): adaptive vs random pairs.
// node tests/duel_sim.mjs deploy/listen_duel_page.html [votes] [trials]
import fs from 'fs';
const page = fs.readFileSync(process.argv[2], 'utf8');
const code = page.split('// <duel>')[1].split('\n').slice(1).join('\n').split('// </duel>')[0];
const { fitBT, nextDuel } = new Function(code + '; return { fitBT, nextDuel };')();
function mulberry(seed) { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const S = Array.from({ length: 24 }, (_, i) => 's' + i), clips = ['c0', 'c1', 'c2', 'c3'];
const N = +process.argv[3] || 40, T = +process.argv[4] || 300;
function trial(seed, adaptive) {
  const rnd = mulberry(seed);
  const g = () => { let u = 0, v = 0; while (!u) u = rnd(); while (!v) v = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  const truth = Object.fromEntries(S.map((s) => [s, 0.8 * g()]));
  const votes = [];
  for (let i = 0; i < N; i++) {
    let d;
    if (adaptive) d = nextDuel(S, votes, clips, rnd);
    else { const a = S[Math.floor(rnd() * 24)]; let b; do b = S[Math.floor(rnd() * 24)]; while (b === a); d = { l: a, r: b, clip: clips[i % 4] }; }
    const p = 1 / (1 + Math.exp(truth[d.r] - truth[d.l])), u = rnd();
    const choice = u < 0.1 ? 'same' : (rnd() < p ? 'left' : 'right');
    votes.push({ ...d, choice });
  }
  const { th } = fitBT(S, votes.filter((v) => !v.repeat));
  const pick = S.reduce((x, y) => th[y] > th[x] ? y : x);
  const ranked = [...S].sort((a, b) => truth[b] - truth[a]);
  const best = ranked[0];
  return { top1: pick === best, top3: ranked.slice(0, 3).includes(pick), regret: truth[best] - truth[pick] };
}
for (const adaptive of [false, true]) {
  let t1 = 0, t3 = 0, r = 0;
  for (let k = 0; k < T; k++) { const o = trial(1000 + k, adaptive); t1 += o.top1; t3 += o.top3; r += o.regret; }
  console.log(`${adaptive ? 'adaptive' : 'random  '} ${N} votes: picks the true best ${(100 * t1 / T).toFixed(0)}%, a true top-3 ${(100 * t3 / T).toFixed(0)}%, mean regret ${(r / T).toFixed(2)}`);
}
