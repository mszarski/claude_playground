"""Image measurement helpers: peak location, sub-pixel -3 dB width, DRC."""

import numpy as np


def drc(z):
    """Log-magnitude display (dB, normalized to peak) -- as in sas_min.py."""
    m = np.abs(z)
    m = m / m.max()
    return 20 * np.log10(m + 1e-6)


def peak_position(slc, img_x, img_y, x0, y0, search=0.25):
    """Location of the |slc| maximum within a (2*search)^2 box around
    (x0, y0). slc is indexed [y, x] (as produced by meshgrid + imshow)."""
    mag = np.abs(slc)
    mx = (img_x >= x0 - search) & (img_x <= x0 + search)
    my = (img_y >= y0 - search) & (img_y <= y0 + search)
    sub = mag[np.ix_(my, mx)]
    iy, ix = np.unravel_index(np.argmax(sub), sub.shape)
    return img_x[mx][ix], img_y[my][iy]


def width_3db(axis, profile):
    """-3 dB (half-power) width of the main lobe of `profile`, with linear
    interpolation of the crossing points for sub-pixel accuracy."""
    profile = np.abs(profile)
    ipk = int(np.argmax(profile))
    half = profile[ipk] / np.sqrt(2)

    # walk left from the peak to the half-power crossing
    il = ipk
    while il > 0 and profile[il] >= half:
        il -= 1
    if profile[il] >= half:
        return np.nan
    fl = (half - profile[il]) / (profile[il + 1] - profile[il])
    xl = axis[il] + fl * (axis[il + 1] - axis[il])

    # walk right
    ir = ipk
    while ir < len(profile) - 1 and profile[ir] >= half:
        ir += 1
    if profile[ir] >= half:
        return np.nan
    fr = (half - profile[ir]) / (profile[ir - 1] - profile[ir])
    xr = axis[ir] - fr * (axis[ir] - axis[ir - 1])

    return xr - xl
