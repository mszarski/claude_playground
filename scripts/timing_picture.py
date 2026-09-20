"""Stage timings of examples/21's picture, forward and backward.

    cd examples && python ../scripts/timing_picture.py

Times every stage of the 300 m picture in float32, alone on the machine's
cores: the trace, the reverberation patches, the boat's echo, the beamformer
in each of its forms, the display; then a step of a fit with a graph on the
boat's position, through the coherent picture and through the incoherent one
(the model side of examples/22).  Each number is the better of two runs.  The
README's "Where the time goes now" table is this script's output; re-run it
after changing anything on the picture's path, and quote the new numbers.

Runs from the examples directory, because it imports 21_long_range_300m.py for
the scene; 21's HYDROPT_* switches apply (FAR, NEAR, BOAT are set here).
"""
import sys, os, importlib.util, math, warnings, time, torch
os.environ.update(HYDROPT_FAR="300", HYDROPT_NEAR="40", HYDROPT_BOAT="250")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
warnings.simplefilter("ignore")
torch.set_default_dtype(torch.float32); torch.set_num_threads(min(4, os.cpu_count() or 1))
spec = importlib.util.spec_from_file_location("ex21", "21_long_range_300m.py"); ex = importlib.util.module_from_spec(spec); spec.loader.exec_module(ex)
ex15 = ex._ex15()
from hydropt import (LambertScattering, add_receiver_noise, azimuth_steering, beam_noise_power, beam_power_scale,
                     beamform, calibrate, line_array_directivity_db, make_time_grid, reverberation_arrivals,
                     shading_window, target_arrivals, trace)
from hydropt.beamform import ArrivalSet
from hydropt.mesh import boat_hull_mesh, mesh_target
def t(label, fn, n=2):
    best = 1e9
    for _ in range(n):
        t0 = time.perf_counter(); out = fn(); best = min(best, time.perf_counter() - t0)
    print(f"  {label:58s} {best:6.2f} s", flush=True); return out
C = ex.C; rx = ex.horizontal_array(); scene, *_ = ex.build_scene(rx, learnable=False)
b = math.radians(ex.BOAT_BEARING_DEG); girth = ex.HULL_BEAM + 2.0 * ex.HULL_DRAUGHT
v, f = boat_hull_mesh(ex.HULL_LENGTH, ex.HULL_BEAM, ex.HULL_DRAUGHT, n_long=int(round(110 * ex.HULL_LENGTH / 12.0)), n_around=int(round(34 * girth / (3.2 + 2.0 * 4.0))))
def build(learnable):
    return mesh_target(v, f, position=(ex.BOAT_RANGE * math.cos(b) + 2, ex.BOAT_RANGE * math.sin(b) - 3, 0.0), yaw=ex.BOAT_HEADING_DEG, n_patches=6, sound_speed=C, diffuse_db=ex.DIFFUSE_DB, learnable=learnable, learnable_shape=False, facet_chunk=4096, checkpoint=False)
dirs, _ = ex.transmit_fan(seed=ex.SEED); w_tx = ex.transmit_pattern(dirs); tilt = ex.passes_deg()[0]
rx_beam = lambda d: ex.receive_beam(d, tilt)
steer, bearings = azimuth_steering(181, ex.SECTOR_DEG); grid = make_time_grid(2 * ex.NEAR / C, 2 * ex.FAR / C, ex.N_BINS); rng = grid * C / 2
shading = shading_window(ex.N_RX, "hamming"); scale = beam_power_scale(shading, ex.PULSE_S)
noise = float(beam_noise_power(scene.freqs_khz, bandwidth_hz=1 / ex.PULSE_S, directivity_db=line_array_directivity_db(ex.N_RX), wind_speed=ex.WIND))
solid = math.radians(2 * ex.SECTOR_DEG) * math.radians(ex.ELEV_DEG[1] - ex.ELEV_DEG[0]) / dirs.shape[0]
print(f"{dirs.shape[0]} transmit rays, {ex.N_RX} elements, 181 beams x {ex.N_BINS} bins, hull {f.shape[0]} facets")
print("forward, no grad:")
with torch.no_grad():
    result = t("trace (RK4 fan, bounces)", lambda: trace(scene, dirs))
    rev = t("reverberation patches from the trace", lambda: reverberation_arrivals(result, dirs, scene.freqs_khz, scattering=LambertScattering(-27.0, learnable=False), solid_angle_per_ray=solid, ray_weights=w_tx * rx_beam(-dirs), boundary="both", surface=scene.surface, bottom=scene.bottom, max_arrivals=ex.PATCHES, generator=torch.Generator().manual_seed(ex.SEED + 1)))
    boat = build(False)
    echo = lambda bt: target_arrivals(scene, bt, dirs, return_leg="eigenray", n_rx_rays=2000, rx_half_angle_deg=45.0, tx_weights=w_tx, tx_pattern=ex.transmit_pattern, rx_pattern=rx_beam, max_arrivals_per_leg=24, generator=torch.Generator().manual_seed(ex.SEED))
    arr = t(f"boat echo (images + physical optics, 6 patches)", lambda: echo(boat))
    both = ArrivalSet(*(torch.cat([a, c], 0) if a is not None else None for a, c in zip(rev, arr)))
    print(f"    {rev.n_arrivals} reverberation + {arr.n_arrivals} boat arrivals")
    img = t("beamform, fft kernel, all arrivals", lambda: beamform(both, rx, scene.freqs_khz, grid, steer, sigma_t=ex.PULSE_S, shading=shading, steer_chunk=8))
    b_rev = t("beamform, reverberation only, complex (once per fit)", lambda: beamform(rev, rx, scene.freqs_khz, grid, steer, sigma_t=ex.PULSE_S, shading=shading, steer_chunk=8, complex_output=True))
    t("beamform, boat only, coherent fft", lambda: beamform(arr, rx, scene.freqs_khz, grid, steer, sigma_t=ex.PULSE_S, shading=shading, steer_chunk=8, complex_output=True))
    t("beamform, boat only, incoherent (direct kernel)", lambda: beamform(arr, rx, scene.freqs_khz, grid, steer, sigma_t=ex.PULSE_S, shading=shading, steer_chunk=8, coherent=False, checkpoint=False))
    span_y = ex.FAR * math.sin(math.radians(ex.SECTOR_DEG)) * 1.02; x_range = (-0.03 * ex.FAR, 1.02 * ex.FAR); pixel_m = (x_range[1] - x_range[0]) / 299
    def display(img):
        noisy = add_receiver_noise(calibrate(img, ex.SOURCE_LEVEL_DB, beam_scale=scale), noise, generator=torch.Generator().manual_seed(ex.SEED + 2))
        shown, _ = ex.display(noisy, rng, pixel_m=pixel_m)
        return ex15.to_cartesian(shown, bearings, grid, n_x=300, n_y=300, x_range=x_range, y_range=(-span_y, span_y))[0]
    t("calibrate + noise + median gain + cartesian", lambda: display(img))
print("with a graph on the boat's position:")
boat = build(True)
def coherent_step():
    boat.position.grad = None
    a = echo(boat)
    bb = b_rev + beamform(a, rx, scene.freqs_khz, grid, steer, sigma_t=ex.PULSE_S, shading=shading, steer_chunk=8, complex_output=True)
    cart = display(bb.real ** 2 + bb.imag ** 2)
    t0 = time.perf_counter(); torch.log10(cart + 1e-6 * float(cart.max())).mean().backward(); return time.perf_counter() - t0
def incoherent_step():
    boat.position.grad = None
    a = echo(boat)
    inc = beamform(a, rx, scene.freqs_khz, grid, steer, sigma_t=ex.PULSE_S, shading=shading, steer_chunk=8, coherent=False, checkpoint=False)
    cart = display(b_rev.real ** 2 + b_rev.imag ** 2 + inc)
    t0 = time.perf_counter(); torch.log10(cart + 1e-6 * float(cart.max())).mean().backward(); return time.perf_counter() - t0
for label, fn in (("coherent picture: forward + backward", coherent_step), ("incoherent picture (the fit's): forward + backward", incoherent_step)):
    best, bwd = 1e9, 0
    for _ in range(2):
        t0 = time.perf_counter(); tb = fn(); tot = time.perf_counter() - t0
        if tot < best: best, bwd = tot, tb
    print(f"  {label:58s} {best:6.2f} s  (backward {bwd:.2f} s)", flush=True)
