import { SceneManager } from './SceneManager.js';
import { RobotManager } from './RobotManager.js';
import { Player } from './Player.js';

// Gallery of motions built by `python -m rmr.viewer` (examples/examples.json):
// [{prompt, recipe, source, moves: [move, ...]}], grouped in the panel by source.
// Adapted from visualizer/src/app.js in pham-tuan-binh/reachy-motion-generator (Apache-2.0).

const $ = (id) => document.getElementById(id);
let player, current = null, sample = 0;

function status(msg, kind = '') { const s = $('status'); s.textContent = msg; s.className = kind; }

function show(entry) {
    current = entry; sample = 0;
    $('prompt').textContent = entry.prompt || '';
    $('recipe').textContent = entry.recipe || '—';
    $('meta').textContent = entry.source || '';
    const tabs = $('samples'); tabs.innerHTML = '';
    if (entry.moves.length > 1) entry.moves.forEach((_, i) => {
        const b = document.createElement('button'); b.textContent = `variant ${i + 1}`; b.className = i === 0 ? 'on' : '';
        b.onclick = () => { sample = i; [...tabs.children].forEach((c, j) => c.className = j === i ? 'on' : ''); play(); };
        tabs.appendChild(b);
    });
    $('result').hidden = false;
    document.querySelectorAll('#gallery button').forEach((b) => b.classList.toggle('on', b.entry === entry));
    play();
}

function play() {
    const d = player.load(current.moves[sample]);
    $('dur').textContent = `${d.toFixed(1)} s`;
    $('playbtn').textContent = '❚❚';
}

/** Accept a gallery entry, a single move, or a list of moves. */
function fromJSON(obj, name) {
    if (obj.moves) return obj;
    if (obj.set_target_data) return { prompt: obj.description || name, moves: [obj], source: name };
    if (Array.isArray(obj) && obj[0]?.set_target_data) return { prompt: name, moves: obj, source: name };
    throw new Error('not a Reachy Mini move');
}

async function loadGallery() {
    let ex = [];
    try { ex = await (await fetch('examples/examples.json')).json(); }
    catch (e) { status('No gallery yet: build it with `python -m rmr.viewer`, or drop a move JSON here.'); return; }
    const box = $('gallery'), groups = new Map();
    ex.forEach((e) => { const g = e.source || 'motions'; if (!groups.has(g)) groups.set(g, []); groups.get(g).push(e); });
    for (const [g, entries] of groups) {
        const h = document.createElement('label'); h.textContent = g; box.appendChild(h);
        const wrap = document.createElement('div'); wrap.className = 'chips';
        entries.forEach((e) => {
            const b = document.createElement('button'); b.textContent = e.prompt.split('.')[0]; b.title = e.prompt; b.entry = e;
            b.onclick = () => { show(e); status(''); };
            wrap.appendChild(b);
        });
        box.appendChild(wrap);
    }
    const first = new URLSearchParams(location.search).get('prompt');
    const start = ex.find((e) => first && e.prompt.toLowerCase().startsWith(first.toLowerCase())) || ex[0];
    if (start) show(start);
}

async function main() {
    status('Loading the robot…');
    const scene = new SceneManager($('container'));
    const robot = new RobotManager((m) => status(m), scene.envMap);
    const robotObj = await robot.loadRobot(); scene.add(robotObj);
    scene.animate?.();
    player = window.__player = new Player(robot, (e, d) => { $('scrub').value = d ? (e / d) * 1000 : 0; $('clock').textContent = `${e.toFixed(1)}`; });
    status('');

    $('playbtn').onclick = () => { player.toggle(); $('playbtn').textContent = player.playing() ? '❚❚' : '▶'; };
    $('restart').onclick = () => player.restart();
    $('speed').onchange = (e) => player.setSpeed(+e.target.value);
    $('loop').onchange = (e) => { player.loop = e.target.checked; };
    $('scrub').oninput = (e) => player.seek(e.target.value / 1000);
    $('download').onclick = () => {
        if (!current) return;
        const a = document.createElement('a');
        a.href = URL.createObjectURL(new Blob([JSON.stringify(current.moves[sample])], { type: 'application/json' }));
        a.download = `${(current.prompt || 'move').split('.')[0].replace(/\W+/g, '_')}_${sample + 1}.json`; a.click();
    };
    const openFile = async (f) => {
        try { show(fromJSON(JSON.parse(await f.text()), f.name)); status(f.name, 'ok'); }
        catch (e) { status(`${f.name}: ${e.message}`, 'err'); }
    };
    $('file').onchange = (e) => e.target.files[0] && openFile(e.target.files[0]);
    window.addEventListener('dragover', (e) => e.preventDefault());
    window.addEventListener('drop', (e) => { e.preventDefault(); e.dataTransfer.files[0] && openFile(e.dataTransfer.files[0]); });

    await loadGallery();
    window.__ready = true;
}

main().catch((e) => { status(`Failed to start: ${e.message}`, 'err'); console.error(e); });
