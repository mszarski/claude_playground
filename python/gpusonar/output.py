"""Output products: GeoTIFF (single and multi-layer), written with rasterio/GDAL.

A stripmap image lives on a rotated grid in the local north-east-down frame. With a
geodetic origin for that frame the GeoTIFF is georeferenced in UTM; without one it is
written in a local engineering frame (metres north and east of the frame origin) and
says so in its tags.
"""

from __future__ import annotations

import json

import numpy as np


def _utm_epsg(lat: float, lon: float) -> int:
    zone = int(np.floor((lon + 180.0) / 6.0)) % 60 + 1
    return (32600 if lat >= 0 else 32700) + zone


def write_geotiff(path: str, layers, points: np.ndarray, frame_origin_lla=None,
                  layer_names=None, metadata: dict | None = None) -> dict:
    """Write an image, or a stack of layers, sampled on a regular planar grid.

    layers: [rows, cols] or [n_layers, rows, cols], real (magnitude or display values).
    points: [rows, cols, 3] NED positions of the pixel centres (x north, y east).
    frame_origin_lla: (latitude, longitude, altitude) of the NED origin, or None.
    Returns the georeferencing that was written.
    """
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import Affine

    data = np.asarray(layers, dtype=np.float32)
    if data.ndim == 2:
        data = data[None]
    rows, cols = data.shape[1:]
    if points.shape[:2] != (rows, cols):
        raise ValueError("points must have the image's shape")
    p00 = points[0, 0, :2]
    d_row = (points[-1, 0, :2] - p00) / max(rows - 1, 1)      # NED metres per row step: (north, east)
    d_col = (points[0, -1, :2] - p00) / max(cols - 1, 1)

    east0, north0, crs, frame = 0.0, 0.0, None, "local NED (metres east and north of the frame origin)"
    georeferenced = frame_origin_lla is not None and np.all(np.isfinite(np.asarray(frame_origin_lla[:2], dtype=float)))
    if georeferenced:
        from pyproj import Transformer
        lat, lon = float(frame_origin_lla[0]), float(frame_origin_lla[1])
        epsg = _utm_epsg(lat, lon)
        east0, north0 = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
        crs = CRS.from_epsg(epsg)
        frame = f"EPSG:{epsg}"
        # Grid north of the UTM zone differs from true north by the convergence angle,
        # under a degree away from the zone edge; ignored for these image sizes.
    # Map (col, row) pixel indices to (easting, northing); pixel centres at index + 0.5.
    a, b = d_col[1], d_row[1]          # easting per column, per row
    d, e = d_col[0], d_row[0]          # northing per column, per row
    c = east0 + p00[1] - 0.5 * (a + b)
    f = north0 + p00[0] - 0.5 * (d + e)
    transform = Affine(a, b, c, d, e, f)
    profile = dict(driver="GTiff", height=rows, width=cols, count=data.shape[0], dtype="float32",
                   transform=transform, compress="deflate")
    if crs is not None:
        profile["crs"] = crs
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        tags = {"gpusonar_frame": frame}
        if metadata:
            tags["gpusonar_metadata"] = json.dumps(metadata)
        dst.update_tags(**tags)
        if layer_names:
            for i, name in enumerate(layer_names):
                dst.set_band_description(i + 1, str(name))
    return {"frame": frame, "transform": tuple(transform)[:6], "georeferenced": bool(georeferenced)}
