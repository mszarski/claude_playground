"""
Coordinate conversion utilities for trajectory analysis.

Converts spherical coordinates (lat/lon/alt) to local Cartesian (East/North/Up)
for geometric calculations.
"""

import numpy as np
from typing import Tuple, Optional
import pandas as pd


# WGS84 ellipsoid parameters
WGS84_A = 6378137.0  # Semi-major axis (meters)
WGS84_B = 6356752.314245  # Semi-minor axis (meters)
WGS84_E2 = 1 - (WGS84_B**2 / WGS84_A**2)  # First eccentricity squared


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great-circle distance between two points using Haversine formula.

    Args:
        lat1, lon1: First point (degrees)
        lat2, lon2: Second point (degrees)

    Returns:
        Distance in meters
    """
    R = 6371000  # Earth radius in meters

    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def geodetic_to_ecef(lat: np.ndarray, lon: np.ndarray, alt: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert geodetic coordinates (lat/lon/alt) to ECEF (Earth-Centered Earth-Fixed).

    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        alt: Altitude in meters (above WGS84 ellipsoid)

    Returns:
        Tuple of (x, y, z) in meters (ECEF coordinates)
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    # Radius of curvature in the prime vertical
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat_rad)**2)

    x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - WGS84_E2) + alt) * np.sin(lat_rad)

    return x, y, z


def ecef_to_enu(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    ref_lat: float, ref_lon: float, ref_alt: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert ECEF coordinates to local ENU (East-North-Up) frame.

    Args:
        x, y, z: ECEF coordinates in meters
        ref_lat, ref_lon, ref_alt: Reference point (origin of ENU frame)

    Returns:
        Tuple of (east, north, up) in meters
    """
    # Get ECEF of reference point
    x_ref, y_ref, z_ref = geodetic_to_ecef(
        np.array([ref_lat]), np.array([ref_lon]), np.array([ref_alt])
    )
    x_ref, y_ref, z_ref = x_ref[0], y_ref[0], z_ref[0]

    # Compute differences
    dx = x - x_ref
    dy = y - y_ref
    dz = z - z_ref

    # Rotation matrix from ECEF to ENU
    lat_rad = np.radians(ref_lat)
    lon_rad = np.radians(ref_lon)

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    # ENU = R * (ECEF - ECEF_ref)
    east = -sin_lon * dx + cos_lon * dy
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

    return east, north, up


def to_local_cartesian(
    df: pd.DataFrame,
    lat_col: str = 'latitude',
    lon_col: str = 'longitude',
    alt_col: str = 'altitude',
    ref_point: Optional[Tuple[float, float, float]] = None
) -> pd.DataFrame:
    """
    Convert a trajectory DataFrame from geodetic to local Cartesian coordinates.

    Uses ENU (East-North-Up) local tangent plane projection centered at the
    first point (or a specified reference point).

    Args:
        df: DataFrame with lat/lon/alt columns
        lat_col: Name of latitude column (degrees)
        lon_col: Name of longitude column (degrees)
        alt_col: Name of altitude column (meters)
        ref_point: Optional (lat, lon, alt) reference point; uses first point if None

    Returns:
        DataFrame with added 'x', 'y', 'z' columns (East, North, Up in meters)
    """
    result = df.copy()

    lat = df[lat_col].values
    lon = df[lon_col].values
    alt = df[alt_col].values

    # Use first point as reference if not specified
    if ref_point is None:
        ref_lat, ref_lon, ref_alt = lat[0], lon[0], alt[0]
    else:
        ref_lat, ref_lon, ref_alt = ref_point

    # Convert to ECEF then to ENU
    x_ecef, y_ecef, z_ecef = geodetic_to_ecef(lat, lon, alt)
    east, north, up = ecef_to_enu(x_ecef, y_ecef, z_ecef, ref_lat, ref_lon, ref_alt)

    result['x'] = east
    result['y'] = north
    result['z'] = up

    return result


def compute_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute initial bearing from point 1 to point 2.

    Args:
        lat1, lon1: First point (degrees)
        lat2, lon2: Second point (degrees)

    Returns:
        Bearing in degrees (0-360, where 0 is North)
    """
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlon_rad = np.radians(lon2 - lon1)

    x = np.sin(dlon_rad) * np.cos(lat2_rad)
    y = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon_rad)

    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360
