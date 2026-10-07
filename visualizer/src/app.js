import { SceneManager } from './SceneManager.js';
import { RobotManager } from './RobotManager.js';
import { Player } from './Player.js';
import { Listener, FPS, dbfs } from './Listen.js';

// Gallery of motions built by `python -m rmr.viewer` (examples/examples.json):
// [{prompt, recipe, source, moves: [move, ...]}], grouped in the panel by source.
// Adapted from visualizer/src/app.js in pham-tuan-binh/reachy-motion-generator (Apache-2.0).

const $ = (id) => document.getElementById(id);
let player, current = null, sample = 0;

function status(msg, kind = '') { const s = $('status'); s.textContent = msg; s.className = kind; }

function show(entry) {
    current = entry; sample = 0; player.live = null;
    $('prompt').textContent = entry.prompt || '';
    $('recipe').textContent = entry.recipe || '—';
    $('idea').textContent = entry.idea || ''; $('idea-row').hidden = !entry.idea;
    const h = entry.heard;
    $('heard').textContent = h ? `“${h.text || '…'}” · sounds ${h.emotion} (${Math.round(100 * h.confidence)}%)` +
        (h.arousal !== undefined ? ` · arousal ${h.arousal.toFixed(2)} · valence ${h.valence.toFixed(2)}` : '') : '';
    $('heard-row').hidden = !h;
    $('reading').textContent = entry.reading || ''; $('reading-row').hidden = !entry.reading;
    const t = entry.timing_ms;
    $('meta').textContent = t ? `generated live · ${t.listen !== undefined ? `listening ${t.listen} ms · ` : ''}planner ${t.planner} ms · generator ${t.generator} ms` : (entry.source || '');
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

// Voice: record while the button is held, encode 16 kHz mono WAV here, POST it to /api/respond.
function encodeWav(chunks, rate) {
    const n = chunks.reduce((s, c) => s + c.length, 0), x = new Float32Array(n);
    let o = 0; for (const c of chunks) { x.set(c, o); o += c.length; }
    const step = rate / 16000, m = Math.floor(n / step), pcm = new Int16Array(m);
    for (let i = 0; i < m; i++) { const v = Math.max(-1, Math.min(1, x[Math.floor(i * step)])); pcm[i] = v * 32767; }
    const buf = new ArrayBuffer(44 + pcm.length * 2), d = new DataView(buf);
    const w = (p, s) => [...s].forEach((ch, i) => d.setUint8(p + i, ch.charCodeAt(0)));
    w(0, 'RIFF'); d.setUint32(4, 36 + pcm.length * 2, true); w(8, 'WAVE'); w(12, 'fmt '); d.setUint32(16, 16, true);
    d.setUint16(20, 1, true); d.setUint16(22, 1, true); d.setUint32(24, 16000, true); d.setUint32(28, 32000, true);
    d.setUint16(32, 2, true); d.setUint16(34, 16, true); w(36, 'data'); d.setUint32(40, pcm.length * 2, true);
    new Int16Array(buf, 44).set(pcm);
    return new Blob([buf], { type: 'audio/wav' });
}

function enableTalk() {
    let rec = null, idle = null;
    // While you speak, the robot listens: loudness every 1/25 s drives rmr/listen.py's controller (Listen.js).
    // After you let go it keeps listening to the silence (the closing nod) until the response move arrives.
    const stopIdle = () => { clearInterval(idle); idle = null; };
    const start = async (e) => {
        e.preventDefault(); if (rec) return;
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true } });
            const ctx = new AudioContext(), src = ctx.createMediaStreamSource(stream), node = ctx.createScriptProcessor(1024, 1, 1);
            const hop = Math.round(ctx.sampleRate / FPS), lis = new Listener();
            let acc = 0, n = 0;
            const chunks = []; node.onaudioprocess = (ev) => {
                const x = ev.inputBuffer.getChannelData(0); chunks.push(new Float32Array(x));
                for (let i = 0; i < x.length; i++) {
                    acc += x[i] * x[i];
                    if (++n === hop) { player.live = lis.step(dbfs(acc / hop)); acc = 0; n = 0; }
                }
            };
            src.connect(node); node.connect(ctx.destination);
            stopIdle();
            rec = { stream, ctx, chunks, lis, t0: performance.now() };
            $('talk').classList.add('on'); status('Listening… release to send.');
        } catch (err) { status(`Microphone unavailable: ${err.message}`, 'err'); }
    };
    const stop = async (e) => {
        e.preventDefault(); if (!rec) return;
        const { stream, ctx, chunks, lis, t0 } = rec; rec = null; $('talk').classList.remove('on');
        stream.getTracks().forEach((t) => t.stop()); const rate = ctx.sampleRate; await ctx.close();
        if (performance.now() - t0 < 500) { player.live = null; return status('Hold the button while you speak.', 'err'); }
        idle = setInterval(() => { player.live = lis.step(-100); }, 1000 / FPS);
        status('Listening to what you said and how you sound…');
        try {
            const r = await fetch(`api/respond?n=${+$('n').value}&seed=${Math.floor(Math.random() * 1e6)}`,
                { method: 'POST', headers: { 'content-type': 'audio/wav' }, body: encodeWav(chunks, rate) });
            const body = await r.json().catch(() => ({}));
            if (!r.ok) throw new Error(body.detail || `server error ${r.status}`);
            stopIdle(); show(body); status('');
        } catch (err) { stopIdle(); player.live = null; status(`Response failed: ${err.message}`, 'err'); }
    };
    const b = $('talk');
    b.addEventListener('pointerdown', start); b.addEventListener('pointerup', stop); b.addEventListener('pointerleave', stop);
    b.addEventListener('keydown', (e) => { if (e.key === ' ' && !e.repeat) start(e); });
    b.addEventListener('keyup', (e) => { if (e.key === ' ') stop(e); });
}

// Live mode: when a server answers /api/health (python -m rmr.server), show the prompt box.
async function enableLive() {
    try { if (!(await fetch('api/health')).ok) return; } catch { return; }
    $('live').hidden = false;
    enableTalk();
    $('live').onsubmit = async (e) => {
        e.preventDefault();
        const prompt = $('ask').value.trim();
        if (!prompt) return status('Type what the robot should express first.', 'err');
        $('go').disabled = true; status('Planning and generating… (about 5–15 s)');
        try {
            const r = await fetch('api/generate', { method: 'POST', headers: { 'content-type': 'application/json' },
                body: JSON.stringify({ prompt, n: +$('n').value, seed: Math.floor(Math.random() * 1e6) }) });
            const body = await r.json().catch(() => ({}));
            if (!r.ok) throw new Error(body.detail || `server error ${r.status}`);
            show(body); status('');
        } catch (err) { status(`Generation failed: ${err.message}`, 'err'); }
        finally { $('go').disabled = false; }
    };
    $('ask').addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); $('live').requestSubmit(); } });
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

    await enableLive();
    await loadGallery();
    window.__ready = true;
}

main().catch((e) => { status(`Failed to start: ${e.message}`, 'err'); console.error(e); });
