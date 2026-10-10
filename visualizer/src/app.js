import { SceneManager } from './SceneManager.js';
import { RobotManager } from './RobotManager.js';
import { Player } from './Player.js';
import { Listener, FPS, dbfs } from './Listen.js';
import { LearnedHead } from './ListenModel.js';

// Gallery of motions built by `python -m rmr.viewer` (examples/examples.json):
// [{prompt, recipe, source, moves: [move, ...]}], grouped in the panel by source.
// Adapted from visualizer/src/app.js in pham-tuan-binh/reachy-motion-generator (Apache-2.0).

const $ = (id) => document.getElementById(id);
let player, current = null, sample = 0;
let listenerWeights = null;
const SESSION = Math.random().toString(36).slice(2, 12);   // the server remembers this conversation's last lines     // learned listener (api/listener), if the server has one
const newListener = () => new Listener(1.0, listenerWeights ? new LearnedHead(listenerWeights) : null);

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

/** Send one turn of speech; the server answers in stages (heard, motion, done) as newline-delimited JSON, and the
 * robot starts moving at "motion", before the reading is written. Resolves with the motion's duration (s). */
async function respondTo(wav, onMotion) {
    const r = await fetch(`api/respond?stream=1&session=${SESSION}&n=${+$('n').value}&seed=${Math.floor(Math.random() * 1e6)}`,
        { method: 'POST', headers: { 'content-type': 'audio/wav' }, body: wav });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `server error ${r.status}`);
    const reader = r.body.getReader(), dec = new TextDecoder();
    let buf = '', entry = {}, duration = 0;
    for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let i;
        while ((i = buf.indexOf('\n')) >= 0) {
            const part = JSON.parse(buf.slice(0, i)); buf = buf.slice(i + 1);
            if (part.stage === 'error') throw new Error(part.detail);
            entry = { ...entry, ...part, timing_ms: { ...entry.timing_ms, ...part.timing_ms } };
            if (part.stage === 'heard') status(`Heard “${part.heard.text || '…'}”. Thinking…`);
            if (part.stage === 'motion') { onMotion?.(); show(entry); status('Reading you…'); duration = player.duration(); }
            if (part.stage === 'done') {
                $('reading').textContent = entry.reading || ''; $('reading-row').hidden = !entry.reading; status('');
                const t = entry.timing_ms;
                $('meta').textContent = `generated live · listening ${t.listen} ms · first motion ${t.first_motion} ms · total ${t.total} ms`;
            }
        }
    }
    return duration;
}

/** Microphone -> onSamples(Float32Array) per ~21 ms block. Returns {rate, close()}. */
async function openMic(onSamples) {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true } });
    const ctx = new AudioContext(), src = ctx.createMediaStreamSource(stream), node = ctx.createScriptProcessor(1024, 1, 1);
    node.onaudioprocess = (ev) => onSamples(new Float32Array(ev.inputBuffer.getChannelData(0)));
    src.connect(node); node.connect(ctx.destination);
    return { rate: ctx.sampleRate, close: async () => { stream.getTracks().forEach((t) => t.stop()); await ctx.close(); } };
}

/** Feeds samples to a Listener at 25 Hz and drives the robot with its pose while `active()`. */
function listenerFeed(lis, rate, active) {
    const hop = Math.round(rate / FPS); let acc = 0, n = 0;
    return (x) => {
        for (let i = 0; i < x.length; i++) {
            acc += x[i] * x[i];
            if (++n === hop) { const s = lis.step(dbfs(acc / hop)); if (active()) player.live = s; acc = 0; n = 0; }
        }
    };
}

function enableTalk() {
    let rec = null, idle = null, free = null;
    // Hold to talk: while you speak, the robot listens (nods, perks; rmr/listen.py's controller in Listen.js).
    // After you let go it keeps listening to the silence (the closing nod) until the response move arrives.
    const stopIdle = () => { clearInterval(idle); idle = null; };
    const start = async (e) => {
        e.preventDefault(); if (rec || free) return;
        try {
            const lis = newListener(), chunks = [];
            let feed = null;
            const mic = await openMic((x) => { chunks.push(x); feed?.(x); });
            feed = listenerFeed(lis, mic.rate, () => true);
            stopIdle();
            rec = { mic, chunks, lis, t0: performance.now() };
            $('talk').classList.add('on'); status('Listening… release to send.');
        } catch (err) { status(`Microphone unavailable: ${err.message}`, 'err'); }
    };
    const stop = async (e) => {
        e.preventDefault(); if (!rec) return;
        const { mic, chunks, lis, t0 } = rec; rec = null; $('talk').classList.remove('on');
        const rate = mic.rate; await mic.close();
        if (performance.now() - t0 < 500) { player.live = null; return status('Hold the button while you speak.', 'err'); }
        idle = setInterval(() => { player.live = lis.step(-100); }, 1000 / FPS);
        status('Listening to what you said and how you sound…');
        try { await respondTo(encodeWav(chunks, rate), stopIdle); }
        catch (err) { player.live = null; status(`Response failed: ${err.message}`, 'err'); }
        finally { stopIdle(); }
    };
        const b = $('talk');
    b.addEventListener('pointerdown', start); b.addEventListener('pointerup', stop); b.addEventListener('pointerleave', stop);
    b.addEventListener('keydown', (e) => { if (e.key === ' ' && !e.repeat) start(e); });
    b.addEventListener('keyup', (e) => { if (e.key === ' ') stop(e); });

    // Hands-free: the mic stays open; the end of each turn (0.9 s of silence after 0.6 s of speech, Listen.js) sends
    // that turn. While the robot answers (waiting, then playing its move) it doesn't take a new turn.
    const hf = $('handsfree');
    hf.onclick = async () => {
        if (free) {
            clearInterval(free.idleTimer); await free.mic.close(); free = null; player.live = null; player.loop = true;
            hf.classList.remove('on'); hf.textContent = '👂 Hands-free'; return status('');
        }
        try {
            const lis = newListener(), ring = [];       // ring: recent sample blocks, ~30 s
            let total = 0, busyUntil = 0, feed = null, mic = null, idling = false;
            const busy = () => performance.now() < busyUntil;
            const onSamples = (x) => {
                ring.push([total, x]); total += x.length;
                while (ring.length && total - ring[0][0] > 30 * mic.rate) ring.shift();
                feed(x);
                if (idling && lis.sinceVoice === 0) { idling = false; busyUntil = 0; player.loop = true; }   // you spoke: listen
                const te = lis.turnEnd;
                if (!te || busy()) return;
                const hop = Math.round(mic.rate / FPS);
                const s0 = Math.max(0, Math.round((te[0] - 0.3) * FPS) * hop), s1 = Math.round(te[1] * FPS) * hop;
                const parts = ring.filter(([s, b]) => s + b.length > s0 && s < s1)
                    .map(([s, b]) => b.subarray(Math.max(0, s0 - s), Math.min(b.length, s1 - s)));
                idling = false; busyUntil = performance.now() + 120000;    // until the answer has been played
                status('Listening to what you said and how you sound…');
                respondTo(encodeWav(parts, mic.rate))
                    .then((d) => { busyUntil = performance.now() + 1000 * (d + 0.5); setTimeout(() => { if (free) status('Hands-free: just talk.'); }, 1000 * d); })
                    .catch((err) => { busyUntil = 0; status(`Response failed: ${err.message}`, 'err'); });
            };
            mic = await openMic((x) => onSamples(x));
            feed = listenerFeed(lis, mic.rate, () => !busy());
            // idle behaviour (rmr/idle.py): when the room has been quiet for a while, now and then do something small
            let nextIdle = performance.now() + 10000;
            const idleTimer = setInterval(async () => {
                const quiet = lis.sinceVoice;                     // s since anyone spoke
                if (busy() || quiet < 10 || performance.now() < nextIdle) return;
                nextIdle = performance.now() + 12000 + Math.random() * 13000;
                try {
                    const r = await fetch(`api/idle?silence=${Math.round(Math.min(quiet, 1e4))}&seed=${Math.floor(Math.random() * 1e6)}`);
                    if (!r.ok || busy() || lis.sinceVoice < 10) return;
                    const m = await r.json();
                    player.live = null; player.load(m.moves[0]); player.loop = false; idling = true;
                    busyUntil = performance.now() + 1000 * (player.duration() + 0.3);
                    setTimeout(() => { idling = false; player.loop = true; }, 1000 * (player.duration() + 0.3));
                } catch { /* idle is optional */ }
            }, 1000);
            free = { mic, idleTimer };
            hf.classList.add('on'); hf.textContent = '👂 Listening (tap to stop)'; status('Hands-free: just talk.');
        } catch (err) { status(`Microphone unavailable: ${err.message}`, 'err'); }
    };
}

// Live mode: when a server answers /api/health (python -m rmr.server), show the prompt box.
async function enableLive() {
    try { if (!(await fetch('api/health')).ok) return; } catch { return; }
    $('live').hidden = false;
    try { const r = await fetch('api/listener'); if (r.ok) listenerWeights = await r.json(); } catch { /* rules only */ }
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
