import { SceneManager } from './SceneManager.js';
import { RobotManager } from './RobotManager.js';
import { Player } from './Player.js';
import { FPS } from './Listen.js';
import { Retarget, fromMediaPipe, toFrame } from './Puppet.js';

const MP = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21';
const MAX_S = 15;
const $ = (id) => document.getElementById(id);
const status = (msg, kind = '') => { const s = $('status'); s.textContent = msg; s.className = kind; };

let player, tracker = null, retarget = new Retarget(), latest = null, frame = null;
let recording = null, clip = null;

async function startCamera() {
    status('Starting the camera…');
    const video = $('cam');
    video.srcObject = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: 640, height: 480 }, audio: false });
    video.hidden = false;
    await video.play();
    status('Loading face tracking…');
    const { FaceLandmarker, FilesetResolver } = await import(`${MP}/vision_bundle.mjs`);
    const files = await FilesetResolver.forVisionTasks(`${MP}/wasm`);
    tracker = await FaceLandmarker.createFromOptions(files, {
        baseOptions: { modelAssetPath: 'assets/face_landmarker.task', delegate: 'GPU' }, runningMode: 'VIDEO',
        numFaces: 1, outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true });
    let lastT = -1;
    const track = () => {
        if (video.readyState >= 2 && video.currentTime !== lastT) {
            lastT = video.currentTime;
            const r = tracker.detectForVideo(video, performance.now());
            if (r.facialTransformationMatrixes?.length) {
                const shapes = Object.fromEntries(r.faceBlendshapes[0].categories.map((c) => [c.categoryName, c.score]));
                latest = { m: fromMediaPipe(r.facialTransformationMatrixes[0].data), shapes };
            } else latest = null;
        }
        requestAnimationFrame(track);
    };
    track();
    // the robot follows at the move format's 25 Hz, whatever the camera's frame rate
    setInterval(() => {
        if (!latest || clip?.playing) return;
        frame = toFrame(retarget.frame(latest.m, latest.shapes), frame);
        player.live = frame.view;
        if (recording) {
            recording.push(frame.move);
            if (recording.length >= MAX_S * FPS) stopRecording();
        }
    }, 1000 / FPS);
    ['calib', 'rec'].forEach((id) => { $(id).disabled = false; });
    $('start').disabled = true;
    status('Look straight at the phone with a relaxed face, then tap “Reset neutral”.', 'ok');
}

function countdown(n, done) {
    $('count').textContent = n || '';
    if (n === 0) return done();
    setTimeout(() => countdown(n - 1, done), 700);
}

function startRecording() {
    $('rec').disabled = true;
    countdown(3, () => {
        recording = [];
        $('rec').disabled = false; $('rec').classList.add('on'); $('rec').textContent = '⏹ Stop';
        $('count').textContent = '●';
    });
}

function stopRecording() {
    const frames = recording; recording = null;
    $('rec').classList.remove('on'); $('rec').textContent = '⏺ Record'; $('count').textContent = '';
    if (!frames || frames.length < FPS) return status('Too short: record at least a second.', 'err');
    clip = { move: { description: '', time: frames.map((_, i) => i / FPS), set_target_data: frames } };
    ['play', 'save', 'download'].forEach((id) => { $(id).disabled = false; });
    status(`Recorded ${(frames.length / FPS).toFixed(1)} s. Play it back, name it, and save it.`, 'ok');
}

function playBack() {
    if (!clip) return;
    clip.playing = true; player.live = null; player.loop = false;
    const d = player.load(clip.move, player.shown);
    setTimeout(() => { clip.playing = false; player.loop = true; }, (d + 0.6) * 1000);
}

function labelled() {
    const label = $('label').value.trim().toLowerCase();
    if (!/^[a-z][a-z -]{1,38}$/.test(label)) { status('Give the clip a feeling word first (letters only).', 'err'); return null; }
    return { ...clip.move, description: `${label}. ${$('desc').value.trim()}`.trim() };
}

async function save() {
    const move = labelled(); if (!move) return;
    $('save').disabled = true;
    try {
        const r = await fetch('api/clips', { method: 'POST', headers: { 'content-type': 'application/json' },
            body: JSON.stringify({ label: $('label').value.trim().toLowerCase(), description: $('desc').value.trim(),
                                   mirror: retarget.mirror, move }) });
        if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `server error ${r.status}`);
        status(`Saved “${$('label').value.trim()}”. Thank you! Record another?`, 'ok');
    } catch (e) { status(`Could not save: ${e.message}`, 'err'); $('save').disabled = false; }
}

function download() {
    const move = labelled(); if (!move) return;
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([JSON.stringify(move)], { type: 'application/json' }));
    a.download = `${$('label').value.trim().replace(/\W+/g, '_') || 'puppet'}.json`; a.click();
}

async function main() {
    status('Loading the robot…');
    const scene = new SceneManager($('container'));
    const robot = new RobotManager((m) => status(m), scene.envMap);
    scene.add(await robot.loadRobot());
    scene.animate?.();
    player = new Player(robot);
    status('Tap “Start camera” to begin.');
    $('start').onclick = () => startCamera().catch((e) => status(`Camera failed: ${e.message}`, 'err'));
    $('calib').onclick = () => { if (latest) { retarget.calibrate(latest.m); status('Neutral set.', 'ok'); } };
    $('mirror').onchange = (e) => { retarget = new Retarget({ mirror: e.target.checked }); };
    $('rec').onclick = () => (recording ? stopRecording() : startRecording());
    $('play').onclick = playBack;
    $('save').onclick = save;
    $('download').onclick = download;
}

main().catch((e) => status(`Failed to start: ${e.message}`, 'err'));
