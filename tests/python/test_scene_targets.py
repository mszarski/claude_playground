"""Highlight targets, aspect dependence and specular glints (sim/targets.py).

The closed forms are those of Lavia, Gonzalez and Blanc, arXiv:1805.06945 (Eqs. 1, 5, 6, 7);
the module's own extensions are checked against them or against Eq. (1) integrated numerically.
"""

import warnings

import numpy as np
import pytest
from scipy import special

from gpusonar import recon
from gpusonar.sim import FreeFieldProvider, Scene, build_scene, make_pose, simulate_ping, targets
from gpusonar.sonar import SonarModel

C = 1500.0
F = 100e3
K = 2.0 * np.pi * F / C
LAM = C / F


def far_field(points, amplitudes, tx, rx, k=K, ref=np.zeros(3)):
    """What the simulator's point scatterers add up to far away, as the amplitude of one
    scatterer at `ref`: sum of a exp(-i k (path - path_ref)) with the spreading referred to ref."""
    leg_t, leg_r = np.linalg.norm(tx - points, axis=1), np.linalg.norm(rx - points, axis=1)
    ref_t, ref_r = np.linalg.norm(tx - ref), np.linalg.norm(rx - ref)
    return np.sum(amplitudes * (ref_t * ref_r) / (leg_t * leg_r) * np.exp(-1j * k * (leg_t + leg_r - ref_t - ref_r)))


def kirchhoff(points, normals, areas, toward, k=K):
    """Eq. (1) of arXiv:1805.06945 by direct summation, conjugated to the simulator's
    exp(+i w t): f = conj[(i k / 2 pi) sum exp(2 i k khat.x) khat.n dS] over the lit side,
    khat = -toward the direction the incident wave travels in."""
    khat = -np.asarray(toward, dtype=float)
    kn = normals @ khat
    lit = kn < 0.0
    return np.conj(1j * k / (2.0 * np.pi) * np.sum(np.exp(2j * k * (points[lit] @ khat)) * kn[lit] * areas[lit]))


def direction(theta_n_deg, phi_deg=90.0):
    """Unit vector at `theta_n` from broadside of a cylinder along x, in the plane at `phi` about it."""
    t, p = np.radians(theta_n_deg), np.radians(phi_deg)
    return np.array([np.sin(t), np.cos(t) * np.sin(p), np.cos(t) * np.cos(p)])


# --- sphere ---------------------------------------------------------------------------------

def test_sphere_strength_is_a_squared_over_four_at_every_aspect():
    a = 0.4
    sphere = targets.SphereGlint([0.2, -0.1, 0.3], a)
    rng = np.random.default_rng(0)
    for d in rng.standard_normal((20, 3)):
        sonar = sphere.centre + 300.0 * d / np.linalg.norm(d)
        p, amp = sphere.evaluate(sonar, sonar, K)
        assert 20 * np.log10(abs(amp[0])) == pytest.approx(10 * np.log10(a**2 / 4), abs=1e-12)
        # The glint is the point of the sphere nearest the sonar.
        assert np.allclose(p[0], sphere.centre + a * d / np.linalg.norm(d), atol=1e-9)
    # The Kirchhoff sphere (Eq. 5) tends to the same thing: |f| -> a / 2.
    kr = K * a
    eq5 = 1j / (4 * K) * np.exp(-2j * kr) * (np.exp(2j * kr) - 2j * kr - 1)
    assert abs(eq5) == pytest.approx(a / 2, rel=1.0 / kr)
    assert not sphere.checks(sonar, sonar, K) and sphere.checks(sonar, sonar, 5.0 / a)


def test_rigid_sphere_validity():
    """How good a^2/4 is: the exact rigid sphere by its partial-wave series,
    f = (2 / ka) sum (-1)^n (2n + 1) j_n'(ka) / h_n'(ka) relative to a / 2 (the Neumann
    boundary condition on each partial wave; a derivation, not a number taken from a paper)."""
    def exact_db(ka):
        n = np.arange(int(ka + 4 * ka ** (1 / 3) + 10) + 1)
        jp = special.spherical_jn(n, ka, derivative=True)
        hp = jp + 1j * special.spherical_yn(n, ka, derivative=True)
        return 20 * np.log10(abs(2 / ka * np.sum((-1.0) ** n * (2 * n + 1) * jp / hp)))
    ka = np.linspace(1.0, 60.0, 2951)
    dev = np.abs([exact_db(x) for x in ka])
    assert dev[ka >= targets.KA_MIN].max() < 0.65          # the bound KA_MIN's note states
    assert dev[ka >= 20.0].max() < 0.25
    assert dev[ka >= 5.0].max() < 1.4 and dev[ka >= 2.0].max() > 2.0


def test_bistatic_glint_on_a_sphere_is_the_specular_point():
    a = 0.5
    sphere = targets.SphereGlint([0.0, 0.0, 0.0], a)
    tx, rx = np.array([4.0, 1.0, -0.5]), np.array([-1.0, 5.0, 2.0])       # close enough for the finite range to matter
    p, amp = sphere.evaluate(tx, rx, K)
    n = p[0] / a
    assert np.linalg.norm(p[0]) == pytest.approx(a)
    s, r = (tx - p[0]) / np.linalg.norm(tx - p[0]), (rx - p[0]) / np.linalg.norm(rx - p[0])
    # The normal bisects the two directions: equal angles, and all three in one plane.
    assert s @ n == pytest.approx(r @ n, abs=1e-9) and s @ n > 0
    assert abs(np.cross(s, r) @ n) < 1e-9
    # Fermat: of all points of the sphere it has the shortest path source - point - receiver.
    v = np.random.default_rng(1).standard_normal((200000, 3))
    v = a * v / np.linalg.norm(v, axis=1, keepdims=True)
    path = np.linalg.norm(tx - v, axis=1) + np.linalg.norm(rx - v, axis=1)
    best = np.linalg.norm(tx - p[0]) + np.linalg.norm(rx - p[0])
    assert best <= path.min() + 1e-12
    assert np.linalg.norm(v[np.argmin(path)] - p[0]) < 0.01
    assert abs(amp[0]) == pytest.approx(a / 2)


# --- cylinder --------------------------------------------------------------------------------

def eq6(a, b, theta, k=K):
    """Eq. (6): cylinder of radius a, length 2b, no end caps, theta from the axis."""
    x = 2 * k * a * np.sin(theta)
    return -(a / 2) * np.tan(theta) * np.sin(2 * k * b * np.cos(theta)) * (2 / np.pi - special.struve(1, x) - 1j * special.jv(1, x))


@pytest.mark.parametrize("resolved", [False, True])
def test_cylinder_pattern_matches_the_closed_form(resolved):
    a, length = 0.1, 1.0                                         # k a = 42, k L = 419
    cyl = targets.CylinderGlint([-length / 2, 0, 0], [length / 2, 0, 0], a, resolved=resolved,
                                spacing=LAM / 2 if resolved else None)
    assert cyl.n == (134 if resolved else 1)
    peak = np.sqrt(a * length**2 / (2 * LAM))

    def ours(theta_n_deg, phi_deg=90.0):
        sonar = 20000.0 * direction(theta_n_deg, phi_deg)        # far beyond L^2 / lambda = 67 m
        return far_field(*cyl.evaluate(sonar, sonar, K), sonar, sonar)

    # Broadside: a L^2 / (2 lambda), the large-argument value of Eq. (6); the same all round.
    assert abs(ours(0.0)) == pytest.approx(peak, rel=1e-3)
    assert abs(ours(0.0)) == pytest.approx(abs(eq6(a, length / 2, np.pi / 2 - 1e-9)), rel=1e-3)
    assert abs(ours(0.0, 17.0)) == pytest.approx(peak, rel=1e-3)
    # Nulls where k L sin(theta_n) = m pi, as in Eq. (6) (sin(2 k b cos(theta)) = 0).
    for m in (1, 2, 5):
        null = np.degrees(np.arcsin(m * np.pi / (K * length)))
        assert abs(ours(null)) < 2e-3 * peak
        assert abs(eq6(a, length / 2, np.pi / 2 - np.radians(null))) < 1e-6 * peak
    # The whole pattern against Eq. (6), out to 60 degrees from broadside (beyond that the
    # glint line's k a cos(theta_n) is no longer large), and against the sinc form.
    for theta_n in (0.2, 0.37, 0.9, 1.7, 4.3, 11.0, 23.5, 41.0, 60.0):
        t = np.radians(theta_n)
        want = abs(eq6(a, length / 2, np.pi / 2 - t))
        sinc_form = peak * np.sqrt(np.cos(t)) * abs(np.sinc(K * length * np.sin(t) / np.pi))
        assert abs(ours(theta_n)) == pytest.approx(want, rel=0.01, abs=2e-4 * peak)
        assert abs(ours(theta_n)) == pytest.approx(sinc_form, rel=0.002, abs=2e-4 * peak)


def test_cylinder_glint_line_against_the_kirchhoff_integral():
    """Amplitude and phase of the glint line against Eq. (1) summed over the cylinder's surface."""
    a, length = 0.06, 0.5
    cyl = targets.CylinderGlint([-length / 2, 0, 0], [length / 2, 0, 0], a)
    nz, nphi = 1200, 2400
    z = (np.arange(nz) + 0.5) / nz * length - length / 2
    phi = (np.arange(nphi) + 0.5) / nphi * 2 * np.pi
    zz, pp = np.meshgrid(z, phi, indexing="ij")
    normals = np.column_stack([np.zeros(zz.size), np.sin(pp).ravel(), np.cos(pp).ravel()])
    pts = a * normals + np.column_stack([zz.ravel(), np.zeros(zz.size), np.zeros(zz.size)])
    areas = np.full(zz.size, length / nz * a * 2 * np.pi / nphi)
    for theta_n, phi_deg in ((0.0, 90.0), (0.4, 90.0), (3.0, 20.0), (25.0, 140.0)):
        d = direction(theta_n, phi_deg)
        want = kirchhoff(pts, normals, areas, d)
        sonar = 1e6 * d
        got = far_field(*cyl.evaluate(sonar, sonar, K), sonar, sonar)
        # Stationary phase leaves a relative error of order 1 / (k a) (here 1/25) from the shadow edges.
        assert abs(got - want) < 0.06 * np.sqrt(a * length**2 / (2 * LAM)) * max(abs(np.sinc(K * length * np.sin(np.radians(theta_n)) / np.pi)), 0.05), (theta_n, got, want)
    # At broadside the phase is the glint's path (2 a short of the axis) and an eighth of a period early.
    d = direction(0.0)
    assert np.angle(kirchhoff(pts, normals, areas, d) * np.exp(-2j * K * a)) == pytest.approx(np.pi / 4, abs=0.05)


def test_glint_moves_along_and_around_a_cylinder():
    a, length = 0.2, 3.0
    cyl = targets.CylinderGlint([0, 0, 0], [length, 0, 0], a, resolved=True, spacing=LAM / 2)
    sonar = np.array([1.0, -8.0, -3.0])
    p, amp = cyl.evaluate(sonar, sonar, K)
    radial = p - np.column_stack([p[:, 0], np.zeros(len(p)), np.zeros(len(p))])
    assert np.allclose(np.linalg.norm(radial, axis=1), a)
    # Every highlight sits on the side facing the sonar, in the plane of the axis and the sonar.
    assert np.allclose(radial / a, np.tile([0.0, -8.0, -3.0] / np.hypot(8.0, 3.0), (len(p), 1)), atol=1e-9)
    # Near field (range 8.5 m against L^2 / lambda = 600 m): the echo comes from the Fresnel
    # zone about the foot of the perpendicular from the sonar, x = 1, not from the middle.
    r = np.linalg.norm(sonar - p, axis=1)
    phasor = amp / r**2 * np.exp(-2j * K * r)
    total = phasor.sum()

    def part(lo, hi):
        return abs(phasor[(p[:, 0] > lo) & (p[:, 0] < hi)].sum())
    assert part(0.6, 1.4) > 0.9 * abs(total) and part(1.9, 2.7) < 0.2 * abs(total)
    # The level is that of a line mirror: per unit length sqrt(a / 2 lambda), over a Fresnel
    # zone sqrt(lambda d / 2) long, d the distance to the glint: |p| = sqrt(a / (4 d^3)).
    d = np.hypot(8.0, 3.0) - a
    assert abs(total) == pytest.approx(np.sqrt(a / (4 * d**3)), rel=0.08)
    # A fixed scatterer cannot do this: move the sonar and the glint follows it.
    sonar2 = np.array([2.0, -8.0, 3.0])
    p2, amp2 = cyl.evaluate(sonar2, sonar2, K)
    assert np.allclose((p2 - p)[:, 0], 0.0) and np.allclose(p2[:, 2], -p[:, 2])


# --- flat facets ------------------------------------------------------------------------------

def test_plate_and_disc_patterns():
    lu, lv = 0.4, 0.25
    plate = targets.PlateGlint([-lu / 2, -lv / 2, 0], [lu, 0, 0], [0, lv, 0])            # normal +z
    grid = targets.PlateGlint([-lu / 2, -lv / 2, 0], [lu, 0, 0], [0, lv, 0], resolved=True, spacing=LAM / 2)
    area = lu * lv
    for d in ([0, 0, 1.0], [0.02, 0, 1.0], [0.05, 0.03, 1.0], [0.3, -0.2, 1.0], [0.0, 0.7, 0.7]):
        d = np.asarray(d) / np.linalg.norm(d)
        sonar = 1e5 * d
        # Eq. (1) over a rectangle, in closed form.
        want = area / LAM * d[2] * np.sinc(K * d[0] * lu / np.pi) * np.sinc(K * d[1] * lv / np.pi)
        for shape in (plate, grid):
            got = far_field(*shape.evaluate(sonar, sonar, K), sonar, sonar)
            assert got == pytest.approx(1j * want, abs=2e-3 * area / LAM)
    # Normal incidence: TS = 20 log10(A / lambda). Nothing from behind unless two-sided.
    assert abs(plate.evaluate(np.array([0, 0, 50.0]), np.array([0, 0, 50.0]), K)[1][0]) == pytest.approx(area / LAM)
    back = np.array([0.0, 0.0, -50.0])
    assert plate.evaluate(back, back, K)[1][0] == 0
    both = targets.PlateGlint([-lu / 2, -lv / 2, 0], [lu, 0, 0], [0, lv, 0], two_sided=True)
    assert abs(both.evaluate(back, back, K)[1][0]) == pytest.approx(area / LAM)
    # Bistatic glint: brightest where the normal bisects the two directions, and reciprocal.
    tx = 1e4 * np.array([np.sin(0.5), 0.0, np.cos(0.5)])
    levels = {ang: abs(plate.evaluate(tx, 1e4 * np.array([np.sin(ang), 0.0, np.cos(ang)]), K)[1][0])
              for ang in (-0.5, -0.45, -0.3, 0.0, 0.5)}
    assert levels[-0.5] == pytest.approx(area / LAM * np.cos(0.5)) and levels[-0.5] == max(levels.values())
    assert levels[0.5] < 0.02 * levels[-0.5]                          # backscatter at 29 degrees off the normal
    rx = 1e4 * np.array([0.3, 0.2, 0.8])
    assert plate.evaluate(tx, rx, K)[1][0] == pytest.approx(plate.evaluate(rx, tx, K)[1][0])
    # Disc: Eq. (7).
    a = 0.15
    for resolved in (False, True):
        disc = targets.DiscGlint([0, 0, 0], [0, 0, 1.0], a, resolved=resolved, spacing=LAM / 3 if resolved else None)
        for theta in (0.0, 0.03, 0.11, 0.4):
            sonar = 1e5 * np.array([np.sin(theta), 0.0, np.cos(theta)])
            x = max(2 * K * a * np.sin(theta), 1e-9)
            eq7 = K * a**2 * np.cos(theta) * special.jv(1, x) / x
            got = far_field(*disc.evaluate(sonar, sonar, K), sonar, sonar)
            assert abs(got) == pytest.approx(abs(eq7), abs=(0.02 if resolved else 1e-6) * K * a**2 / 2)


def test_large_plate_is_a_mirror():
    """Close to a plate many Fresnel zones across, the resolved facet adds up to the image of
    the source: amplitude 1 / (2 d) and the phase of the path 2 d. This fixes the element's
    amplitude 1 / lambda and its quarter-period phase independently of the far-field formulas."""
    d, half = 1.0, 1.5
    plate = targets.PlateGlint([-half, -half, 0], [2 * half, 0, 0], [0, 2 * half, 0], resolved=True, spacing=LAM / 2)
    sonar = np.array([0.0, 0.0, d])
    p, amp = plate.evaluate(sonar, sonar, K)
    r = np.linalg.norm(sonar - p, axis=1)
    # Taper the outer part so that the edges (which diffract) do not add their ripple.
    rho = np.hypot(p[:, 0], p[:, 1])
    taper = np.where(rho < 0.7, 1.0, np.where(rho < 1.4, 0.5 * (1 + np.cos(np.pi * (rho - 0.7) / 0.7)), 0.0))
    total = np.sum(taper * amp / r**2 * np.exp(-2j * K * r))
    image = np.exp(-2j * K * d) / (2 * d)
    assert total == pytest.approx(image, abs=0.02 * abs(image))


# --- aspect tables ----------------------------------------------------------------------------

def test_aspect_table_interpolates_in_the_target_frame():
    table = targets.AspectTable([0, 90, 180], [10.0, 25.0, 15.0])                    # the same on both sides
    def ts(aspect_deg, elev_deg=0.0, t=table):
        a, e = np.radians(aspect_deg), np.radians(elev_deg)
        return float(t.ts_db(np.array([np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), -np.sin(e)]))[0])
    assert ts(0) == pytest.approx(10.0) and ts(90) == pytest.approx(25.0) and ts(-90) == pytest.approx(25.0)
    assert ts(45) == pytest.approx(17.5) and ts(135) == pytest.approx(20.0) and ts(-135) == pytest.approx(20.0)
    full = targets.AspectTable([0, 90, 180, 270], [10.0, 25.0, 15.0, 5.0])           # all round, periodic
    assert ts(-90, t=full) == pytest.approx(5.0) and ts(315, t=full) == pytest.approx(7.5)
    two = targets.AspectTable([0, 180], [[0.0, 10.0], [20.0, 30.0]], elevation_deg=[0.0, 40.0])
    assert ts(90, 20, t=two) == pytest.approx(15.0) and ts(0, 60, t=two) == pytest.approx(20.0)
    with pytest.raises(ValueError):
        targets.AspectTable([0, 90], [1.0, 2.0, 3.0])

    # On a target: the table is the strength of the whole highlight set, whatever the heading.
    target = targets.HighlightTarget(origin=[100.0, 0.0, 0.0], heading_deg=30.0, aspect_table=table,
                                     highlights=[{"position": [x, 0, 0], "target_strength_db": w}
                                                 for x, w in ((-20, 0.0), (0, 6.0), (15, -3.0))])
    scene = Scene().add_target(target, "sub")
    for bearing in (0.0, 30.0, 75.0, 120.0, 210.0):
        b = np.radians(bearing)
        sonar = target.origin + 5000.0 * np.array([np.cos(b), np.sin(b), 0.0])
        _, amp, _ = scene.evaluate(0.0, sonar, sonar)
        want = ts(bearing - 30.0)
        assert 10 * np.log10(np.sum(np.abs(amp) ** 2)) == pytest.approx(want, abs=1e-9)
        assert np.abs(amp[1] / amp[0]) == pytest.approx(10 ** (6 / 20))            # the shares stay
    # A table on one highlight is added to that highlight's own strength.
    one = targets.HighlightTarget(highlights=[{"position": [0, 0, 0], "target_strength_db": -5.0, "aspect_table": table},
                                              {"position": [1, 0, 0], "target_strength_db": -5.0}])
    sonar = np.array([0.0, 4000.0, 0.0])
    _, amp, _ = one.evaluate(targets.PingContext(0.0, sonar, sonar, None, C, None, None, None, None))
    assert 20 * np.log10(np.abs(amp)) == pytest.approx([20.0, -5.0], abs=0.01)


# --- highlight targets ------------------------------------------------------------------------

def context(t, tx, rx=None, f=F):
    return targets.PingContext(t, np.asarray(tx, float), np.asarray(tx if rx is None else rx, float), f, C, None, None, None, None)


def test_highlights_ride_on_the_trajectory():
    target = targets.HighlightTarget(origin=[50.0, 10.0, 5.0], velocity=[2.0, 0.0, 0.0], heading_deg=90.0, turn_rate_deg=10.0,
                                     highlights=[{"position": [10.0, 0.0, 0.0], "target_strength_db": 0.0},
                                                 {"position": [0.0, 2.0, 1.0], "amplitude": 0.5j}])
    p, amp, v = target.evaluate(context(0.0, [0, 0, 0]))
    assert np.allclose(p, [[50.0, 20.0, 5.0], [48.0, 10.0, 6.0]])                   # bow towards +y, starboard towards -x
    assert np.allclose(amp, [1.0, 0.5j])
    omega = np.radians(10.0)
    assert np.allclose(v, [[2.0 - omega * 10.0, 0.0, 0.0], [2.0, -omega * 2.0, 0.0]])
    p9, _, _ = target.evaluate(context(9.0, [0, 0, 0]))                             # 9 s later: turned to 180 deg
    assert np.allclose(p9, [[50.0 + 18.0 - 10.0, 10.0, 5.0], [68.0, 8.0, 6.0]], atol=1e-9)
    # In a scene: positions, amplitudes and velocities all come from the ping.
    scene = Scene().add_points([[1.0, 2.0, 3.0]], 2.0).add_target(target, "boat")
    pts, a, vel = scene.evaluate(9.0, np.zeros(3), np.zeros(3), frequency=F, sound_speed=C)
    assert np.allclose(pts[0], [1, 2, 3]) and a[0] == 2.0 and np.allclose(pts[1:], p9)
    assert np.allclose(vel[1], [2.0, -omega * 10.0, 0.0], atol=1e-9) and np.allclose(vel[0], 0.0)
    assert scene.labels == ["points", "boat"] and len(scene) == 3
    with pytest.raises(ValueError, match="projector and receiver"):
        scene.evaluate(0.0)
    # A trajectory function does the same job.
    path = targets.HighlightTarget(trajectory=lambda t: (np.array([50.0 + 2.0 * t, 10.0, 5.0]), np.array([0.0, 0.0, np.pi / 2 + omega * t])),
                                   highlights=target.highlights)
    pp, _, vv = path.evaluate(context(9.0, [0, 0, 0]))
    assert np.allclose(pp, p9, atol=1e-6) and np.allclose(vv, target.evaluate(context(9.0, [0, 0, 0]))[2], atol=1e-4)


def test_highlight_delay_response_common_centre_and_jitter():
    tx, rx = np.array([0.0, 0.0, 0.0]), np.array([3.0, 0.0, 0.0])
    where = np.array([40.0, 30.0, 10.0])

    def path(p):
        return np.linalg.norm(tx - p, axis=-1) + np.linalg.norm(rx - p, axis=-1)

    def spreading(p):
        return np.linalg.norm(tx - p, axis=-1) * np.linalg.norm(rx - p, axis=-1)
    # A delay between reception and re-emission: the path is longer by c tau, the direction
    # and the level at the receiver are those of the highlight itself.
    tau = 2e-3
    plain = targets.HighlightTarget(highlights=[{"position": where, "target_strength_db": -10.0}])
    late = targets.HighlightTarget(highlights=[{"position": where, "target_strength_db": -10.0, "delay": tau}])
    p0, a0, _ = plain.evaluate(context(0.0, tx, rx))
    p1, a1, _ = late.evaluate(context(0.0, tx, rx))
    assert path(p1[0]) - path(p0[0]) == pytest.approx(C * tau, abs=1e-9)
    assert abs(a1[0]) / spreading(p1[0]) == pytest.approx(abs(a0[0]) / spreading(p0[0]), rel=1e-12)
    assert np.degrees(np.arccos(p1[0] @ p0[0] / np.linalg.norm(p1[0]) / np.linalg.norm(p0[0]))) < 0.1
    # A frequency response: magnitude and phase at the ping's frequency, the phase slope as a delay.
    tau_g = 0.4e-3
    freqs = np.linspace(80e3, 120e3, 41)
    table = {"frequency": freqs, "magnitude_db": np.linspace(-6.0, 2.0, 41),
             "phase_deg": np.degrees(0.7 - 2 * np.pi * freqs * tau_g)}
    shaped = targets.HighlightTarget(highlights=[{"position": where, "target_strength_db": -10.0, "response": table}])
    p2, a2, _ = shaped.evaluate(context(0.0, tx, rx))
    assert path(p2[0]) - path(p0[0]) == pytest.approx(C * tau_g, rel=1e-6)
    # Echo spectrum of the moved highlight at f: a exp(-2 pi i f path / c) / spreading. Against H(f) x the plain one:
    for f in (95e3, 100e3, 105e3):
        h = 10 ** (np.interp(f, freqs, table["magnitude_db"]) / 20) * np.exp(1j * (0.7 - 2 * np.pi * f * tau_g))
        got = a2[0] / spreading(p2[0]) * np.exp(-2j * np.pi * f * path(p2[0]) / C)
        ref = a0[0] / spreading(p0[0]) * np.exp(-2j * np.pi * f * path(p0[0]) / C)
        assert np.angle(got / (h * ref)) == pytest.approx(0.0, abs=1e-6)
        if f == 100e3:
            assert abs(got / ref) == pytest.approx(abs(h), rel=1e-9)
    fn = targets.HighlightTarget(highlights=[targets.Highlight(where, -10.0, response=lambda f: 0.5 * np.exp(-2j * np.pi * f * tau_g))])
    p3, a3, _ = fn.evaluate(context(0.0, tx, rx))
    assert np.allclose(p3, p2, atol=1e-6) and abs(a3[0] * spreading(p0[0]) / spreading(p3[0])) == pytest.approx(0.5 * 10 ** -0.5)
    with pytest.raises(ValueError, match="frequency"):
        shaped.evaluate(context(0.0, tx, rx, f=None))
    # commonCenter: every highlight on the line of the target's origin, at its own arrival time.
    offsets = np.array([[-30.0, 0.0, 0.0], [0.0, 0.0, 0.0], [25.0, 4.0, -2.0]])
    kw = dict(origin=[400.0, 300.0, 20.0], heading_deg=20.0,
              highlights=[{"position": o, "target_strength_db": 0.0} for o in offsets])
    apart, together = targets.HighlightTarget(**kw), targets.HighlightTarget(common_center=True, **kw)
    pa, aa, _ = apart.evaluate(context(0.0, tx, rx))
    pc, ac, _ = together.evaluate(context(0.0, tx, rx))
    assert np.allclose(path(pc), path(pa), atol=1e-9) and np.allclose(ac, aa)
    assert np.allclose(pc[1], pa[1])
    line = (pc - pc[1]) / np.maximum(np.linalg.norm(pc - pc[1], axis=1, keepdims=True), 1e-12)
    assert abs(line[0] @ line[2]) == pytest.approx(1.0, abs=1e-12)                  # collinear with the origin
    assert np.linalg.norm(pc[0] - pa[0]) > 5.0                                      # and not where it really is
    # randomPosition: a Gaussian jitter of the highlight positions, drawn once.
    kw = dict(highlights=[{"position": [float(i), 0, 0]} for i in range(4000)], random_position=0.05)
    j1, j2 = (targets.HighlightTarget(rng=np.random.default_rng(3), **kw) for _ in range(2))
    assert np.array_equal(j1.local, j2.local)
    assert (j1.local - np.array([h.position for h in j1.highlights])).std() == pytest.approx(0.05, rel=0.05)
    assert np.array_equal(j1.evaluate(context(0.0, tx))[0], j1.evaluate(context(1.0, tx))[0])


# --- through the simulator ---------------------------------------------------------------------

def fls():
    return SonarModel.load("fls_300k", {
        "receive": {"kind": "baseband", "sample_rate": 150e3, "window": {"range": [3.0, 10.5]}, "tvg": None}})


def cylinder_item(foot_bearing_deg, shift):
    """A 1.2 m cylinder, radius 0.1 m, level with the sonar, placed so that the perpendicular
    from the sonar (at the origin) meets its axis at range 6 m and the given bearing, `shift`
    metres along the axis from its middle."""
    b = np.radians(foot_bearing_deg)
    foot = 6.0 * np.array([np.cos(b), np.sin(b), 0.0])
    along = np.array([-np.sin(b), np.cos(b), 0.0])
    mid = foot - shift * along
    return {"type": "target", "label": "cylinder", "origin": [0.0, 0.0, 0.0],
            "shapes": [{"shape": "cylinder", "start": list(mid - 0.6 * along), "end": list(mid + 0.6 * along),
                        "radius": 0.1, "resolved": True, "spacing": 0.0025}]}, foot, along, mid


@pytest.mark.parametrize("bearing,shift", [(-3.0, 0.3), (4.0, -0.35)])
def test_forward_looking_image_shows_the_cylinder_glint_where_geometry_says(bearing, shift):
    """One ping sees one glint: on the near side of the cylinder at the foot of the
    perpendicular from the sonar, wherever along the cylinder that is; not its middle, not its ends."""
    item, foot, along, mid = cylinder_item(bearing, shift)
    scene = build_scene([item], np.random.default_rng(0))
    assert len(scene) == 480 and not scene.occluders
    sonar = fls()
    rec = simulate_ping(sonar, make_pose(), scene, environment={"sound_speed": C}, absorption_model="none", adc=False)
    ranges = np.arange(5.6, 6.4, 0.004)
    bearings = np.radians(np.arange(-16.0, 16.01, 0.1))
    power = np.abs(recon.fan_image([rec], ranges, bearings, interp="linear").image) ** 2
    i, j = np.unravel_index(np.argmax(power), power.shape)
    assert ranges[i] == pytest.approx(6.0 - 0.1, abs=0.008)                    # the near side: one radius short of the axis
    assert np.degrees(bearings[j]) == pytest.approx(bearing, abs=0.3)
    # Level: a line mirror, sqrt(a / (4 d^3)) at d = 5.9 m, against a 0 dB point at the same place.
    ref = simulate_ping(sonar, make_pose(), Scene().add_points([foot * (5.9 / 6.0)], 1.0), environment={"sound_speed": C},
                        absorption_model="none", adc=False)
    ref_power = np.abs(recon.fan_image([ref], ranges, bearings, interp="linear").image) ** 2
    measured = 10 * np.log10(power.max() / ref_power.max())
    expected = 10 * np.log10(0.1 / (4 * 5.9**3) * 5.9**4)                      # equivalent point strength a d / 4
    # (Measured 0.7 dB under: the echo of a line mirror has the wavefront of an image 2 d away along
    # the cylinder, so an array focused at d is slightly out of focus on it.)
    assert measured == pytest.approx(expected, abs=1.0), (measured, expected)
    # The cylinder's middle and ends are at least 15 dB down.
    def level_at(point):
        r, b = np.linalg.norm(point), np.arctan2(point[1], point[0])
        box = power[np.abs(ranges - r) < 0.02][:, np.abs(bearings - b) < np.radians(0.5)]
        return 10 * np.log10(box.max() / power.max())
    radial = foot / 6.0
    assert level_at(mid - 0.1 * radial) < -15.0
    for s in (-0.6, 0.6):
        if abs(s - shift) > 0.5:                                               # the end away from the glint
            assert level_at(mid + s * along - 0.1 * radial) < -15.0


def test_synthetic_aperture_image_shows_the_glint_line():
    """A sonar stepping along a rail past a rod parallel to it (rail_cylinder_glint): the glint
    steps along the rod with the sonar, so the image shows the whole line on the rod's near
    side. A rod turned 20 degrees never presents its side squarely to the rail, and all but
    vanishes: what a row of fixed point scatterers would not do."""
    img, xs, ys, glint_y = run_cylinder_sas(0.0)
    line = img[:, np.argmin(np.abs(ys - glint_y))]
    on = (xs > 0.11) & (xs < 0.19)                                # the rod is x = 0.10 .. 0.20
    off = (xs < 0.09) | (xs > 0.21)
    assert line[on].min() > 0.5 * line[on].max()                  # a line, evenly bright
    assert line[off].max() < 0.1 * line[on].max()                 # that ends where the rod ends
    assert ys[np.argmax(img[on].mean(axis=0))] == pytest.approx(glint_y, abs=0.0008)
    turned, _, _, _ = run_cylinder_sas(20.0)
    assert turned.max() < 0.1 * img.max()


def run_cylinder_sas(turn_deg):
    from gpusonar.config import load_scenario_config
    from gpusonar.sim import run_scenario
    cfg = load_scenario_config("rail_cylinder_glint")
    cfg["scene"][0]["heading_deg"] = turn_deg
    cfg["sonar_overrides"]["receive"]["window"]["range"] = [1.96, 2.03]      # the rod only: a third of the samples
    result = run_scenario(cfg)
    recon.prepare(result.records)
    # The glint line: one radius from the axis (y = 1.88, z = -0.01) towards the rail (y = 0, z = -0.684).
    toward = np.array([-1.88, -0.674]) / np.hypot(1.88, 0.674)
    glint_y, glint_z = 1.88 + 0.01 * toward[0], -0.01 + 0.01 * toward[1]
    xs, ys = np.arange(0.05, 0.2501, 0.002), np.arange(1.86, 1.8851, 0.0004)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    grid = np.stack([X, Y, np.full(X.shape, glint_z)], axis=-1)
    return np.abs(recon.sas_image(result.records, grid, interp="linear")), xs, ys, glint_y


def test_both_engines_get_the_per_ping_amplitudes():
    """Targets and seabed models are worked out before the path pairing, so the C++ and NumPy
    engines are handed the same scatterers and return the same record."""
    scene = build_scene([
        {"type": "seafloor", "x_range": [4.0, 8.0], "y_range": [-1.0, 1.0], "density": 150.0, "depth": 1.0,
         "model": "fluid_sediment", "rho": 1.9, "nu": 1.1, "delta": 0.0147, "w2": 2.0e-5, "gamma2": 3.25, "sigma2": 0.002},
        {"type": "target", "origin": [6.0, 0.0, 0.3], "heading_deg": 70.0, "velocity": [0.5, 0.0, 0.0],
         "aspect_table": {"aspect_deg": [0, 90, 180], "ts_db": [-30.0, -15.0, -25.0]},
         "highlights": [{"position": [-0.5, 0, 0]}, {"position": [0.5, 0, 0], "delay": 1e-4}],
         "shapes": [{"shape": "sphere", "centre": [0, 0, -0.5], "radius": 0.2},
                    {"shape": "cylinder", "start": [-0.5, 0, 0], "end": [0.5, 0, 0], "radius": 0.04, "caps": True,
                     "resolved": True, "spacing": 0.0025}]}], np.random.default_rng(5))
    sonar = fls()
    from gpusonar.sim import ImageSourceProvider
    for provider in (FreeFieldProvider(C), ImageSourceProvider(C, surface_z=-2.0, bottom_z=None)):
        recs = {}
        for engine in ("cpp", "numpy"):
            provider.engine = engine
            recs[engine] = simulate_ping(sonar, make_pose(), scene, environment={"sound_speed": C}, absorption_model="none",
                                         adc=False, provider=provider, scenario_time=0.4, engine=engine)
        a, b = recs["cpp"].samples.array, recs["numpy"].samples.array
        assert np.abs(a).max() > 0 and np.allclose(a, b, rtol=1e-5, atol=1e-6 * np.abs(a).max())
    # And the pairing is given what Scene.evaluate returns.
    tx = np.array(recs["cpp"].tx_position())
    rx = recs["cpp"].rx_positions().mean(axis=0)
    pts, amp, vel = scene.evaluate(0.4, tx, rx, frequency=sonar.center_frequency, sound_speed=C)
    moving = scene.label_masks()["target"]
    assert np.allclose(vel[moving], [0.5, 0.0, 0.0]) and np.allclose(vel[~moving], 0.0)
    assert not np.allclose(pts[moving], scene.positions[moving])


def test_scenes_without_modifiers_are_untouched():
    rng = np.random.default_rng(4)
    scene = build_scene([
        {"type": "seafloor", "x_range": [3.5, 6.0], "y_range": [-1.0, 1.0], "density": 100.0, "lambert_mu_db": -27.0},
        {"type": "sphere", "centre": [5.0, 0.0, -0.3], "radius": 0.3, "density": 500.0, "mu_db": -15.0},
        {"type": "points", "positions": [[4.0, 0.5, -0.5]], "velocity": [0.1, 0.0, 0.0]}], rng)
    assert not scene.modifiers
    pts, amp, vel = scene.evaluate(1.5)
    assert np.array_equal(pts, scene.at_time(1.5)) and amp is scene.amplitudes and vel is scene.velocities
    with pytest.raises(ValueError, match="unknown seafloor model"):
        build_scene([{"type": "seafloor", "x_range": [0, 1], "y_range": [0, 1], "density": 10.0, "model": "mud"}], rng)
    with pytest.raises(ValueError, match="needs"):
        build_scene([{"type": "seafloor", "x_range": [0, 1], "y_range": [0, 1], "density": 10.0, "model": "fluid_sediment",
                      "rho": 1.9}], rng)


def test_validity_warnings_and_occluders():
    target = targets.HighlightTarget(origin=[5.0, 0.0, 0.0], shapes=[
        targets.SphereGlint([0, 0, 0], 0.3), targets.CylinderGlint([0, -1, 1], [0, 1, 1], 0.2)])
    scene = Scene().add_target(target, occlude=True)
    assert [o[0] for o in scene.occluders] == ["sphere", "capsule"]
    assert np.allclose(scene.occluders[0][1], [5.0, 0.0, 0.0]) and np.allclose(scene.occluders[1][1][0], [5.0, -1.0, 1.0])
    sonar = np.zeros(3)
    with pytest.warns(UserWarning, match="inside L\\^2/lambda"):
        pts, amp, _ = scene.evaluate(0.0, sonar, sonar, frequency=F)
    # The glints are on the near side, so their own occluders do not hide them; a point behind is hidden.
    assert scene.visible(sonar, pts).all() and not scene.visible(sonar, np.array([[9.0, 0.0, 0.0]]))[0]
    with pytest.warns(UserWarning, match="k a"):
        scene.evaluate(0.0, sonar, sonar, frequency=3e3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                           # each condition is reported once
        scene.evaluate(0.0, sonar, sonar, frequency=3e3)
    with pytest.raises(ValueError, match="do not move"):
        Scene().add_target(targets.HighlightTarget(velocity=[1, 0, 0], shapes=[targets.SphereGlint([0, 0, 0], 0.3)]), occlude=True)
    with pytest.raises(ValueError, match="spacing"):
        targets.CylinderGlint([0, 0, 0], [1, 0, 0], 0.1, resolved=True)
    with pytest.warns(UserWarning, match="half a wavelength"):
        Scene().add_target(targets.HighlightTarget(shapes=[targets.PlateGlint([5, 0, 0], [0, 1, 0], [0, 0, 1], resolved=True, spacing=0.05)])
                           ).evaluate(0.0, sonar, sonar, frequency=F)


def test_scenarios_with_targets_run():
    from gpusonar.sim import run_scenario
    # A sphere and a cylinder on a seabed that follows the bottom model: the sphere's echo in
    # the image is that of a point of strength a^2 / 4 at its glint.
    result = run_scenario("fls_glints_on_sediment")
    scene, rec = result.scene, result.records[0]
    assert {"seafloor", "sphere", "cylinder"} == set(scene.labels) and len(scene.occluders) == 2
    assert len(scene.modifiers) == 3 and scene.lambert[scene.label_masks()["seafloor"]].all()
    sonar_at, centre, a = np.array([0.0, 0.0, -1.0]), np.array([5.0, -1.0, -0.3]), 0.3
    glint = centre + a * (sonar_at - centre) / np.linalg.norm(sonar_at - centre)
    r0, b0 = np.linalg.norm(glint - sonar_at), np.arctan2(glint[1], glint[0])
    ranges, bearings = r0 + np.arange(-0.1, 0.1001, 0.004), b0 + np.radians(np.arange(-2.0, 2.01, 0.1))
    power = np.abs(recon.fan_image([rec], ranges, bearings, interp="linear").image) ** 2
    i, j = np.unravel_index(np.argmax(power), power.shape)
    assert ranges[i] == pytest.approx(r0, abs=0.008) and bearings[j] == pytest.approx(b0, abs=np.radians(0.3))
    ref = simulate_ping(fls(), make_pose(position=sonar_at), Scene().add_points([glint], 1.0), environment={"sound_speed": C},
                        absorption_model="none", adc=False)
    ref_power = np.abs(recon.fan_image([ref], ranges, bearings, interp="linear").image) ** 2
    assert 10 * np.log10(power.max() / ref_power.max()) == pytest.approx(10 * np.log10(a**2 / 4), abs=0.5)
    # A highlight target that turns as it crosses: at each ping the highlights' summed strength
    # is the table's value at that ping's aspect, and the late highlight arrives late.
    result = run_scenario("fls_highlight_target_aspect")
    scene, target = result.scene, result.scene.modifiers[0][2]
    table = target.aspect_table
    seen = []
    for rec, t in zip(result.records, result.times):
        tx = np.array(rec.tx_position())
        pts, amp, vel = scene.evaluate(t, tx, tx, frequency=120e3)
        origin, rot, _, _ = target.pose(t)
        want = float(table.ts_db(((tx - origin) / np.linalg.norm(tx - origin)) @ rot)[0])
        # The third highlight answers 4 ms late: 3 m further in one-way range than where it is,
        # with its amplitude raised so that it arrives at the level of its true place.
        true_place = origin + rot @ target.local[2]
        r_true, r_late = np.linalg.norm(true_place - tx), np.linalg.norm(pts[2] - tx)
        assert r_late - r_true == pytest.approx(0.5 * C * 0.004, abs=1e-6)
        level = np.abs(amp) * np.where(np.arange(len(amp)) == 2, (r_true / r_late) ** 2, 1.0)
        assert 10 * np.log10(np.sum(level**2)) == pytest.approx(want, abs=1e-6)
        seen.append(want)
        assert np.abs(rec.samples.array).max() > 0
    assert max(seen) - min(seen) > 5.0                               # the aspect really changed
