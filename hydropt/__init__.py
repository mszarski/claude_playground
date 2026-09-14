"""hydropt -- a differentiable 3-D underwater acoustic ray tracer in PyTorch.

Coordinates: ``x``, ``y`` horizontal (m); ``z`` depth (m, positive downward,
zero at the sea surface).  Times in s, sound speed in m/s, losses in dB,
frequencies in kHz.
"""

from .absorption import octave_bands, thorp_db_per_km
from .active import (
    EchoResult,
    PointTarget,
    compose_arrivals,
    render_echo,
    render_extended_echo,
    return_fan,
    target_arrivals,
)
from .beamform import (
    ArrivalSet,
    azimuth_steering,
    beamform,
    element_field,
    extract_arrivals,
    shading_window,
)
from .beams import GaussianBeams, gaussian_beams, suggest_beam_width
from .boundaries import (
    BilinearHeightField,
    BoundaryLoss,
    ConstantLoss,
    FlatHeight,
    HeightField,
    RayleighBottomLoss,
    find_crossing,
    grazing_angle,
    reflect,
)
from .environment import (
    fractal_bathymetry,
    gaussian_seamount,
    internal_wave_perturbation,
    pierson_moskowitz_surface,
    significant_wave_height_pm,
    spectral_field,
    wave_number_peak_pm,
)
from .fields import (
    DepthProfile,
    GriddedField,
    IsoProfile,
    LinearGradientProfile,
    MunkProfile,
    PiecewiseLinearProfile,
    SoundSpeedField,
)
from .launch import (
    directions_from_angles,
    fan_2d,
    fibonacci_cone,
    fibonacci_sphere,
    receiver_cone_importance,
    spherical_fan,
    structured_fan,
)
from .receiver import (
    horizontal_line_array,
    make_time_grid,
    splat_etc,
    vertical_line_array,
)
from .reverb import (
    LambertScattering,
    cone_solid_angle,
    render_reverberation,
    reverberation_arrivals,
)
from .scene import Scene
from .targets import (
    CylinderScattering,
    ExtendedTarget,
    IsotropicScattering,
    PlateScattering,
    ScatteringPattern,
    rotation_matrix,
)
from .spreading import RayTube, ray_tube, ray_tube_jvp, spherical_spreading
from .tracer import BounceEvents, RayState, TraceResult, bounce_events, trace

__version__ = "0.1.0"

__all__ = [
    "Scene", "trace", "TraceResult", "RayState", "BounceEvents", "bounce_events",
    "SoundSpeedField", "DepthProfile", "IsoProfile", "LinearGradientProfile",
    "MunkProfile", "PiecewiseLinearProfile", "GriddedField",
    "HeightField", "FlatHeight", "BilinearHeightField",
    "BoundaryLoss", "ConstantLoss", "RayleighBottomLoss",
    "reflect", "grazing_angle", "find_crossing",
    "thorp_db_per_km", "octave_bands",
    "spherical_fan", "structured_fan", "fan_2d", "fibonacci_sphere", "fibonacci_cone",
    "receiver_cone_importance", "directions_from_angles",
    "vertical_line_array", "horizontal_line_array", "make_time_grid", "splat_etc",
    # active sonar
    "PointTarget", "EchoResult", "render_echo", "return_fan", "compose_arrivals",
    # environment generators
    "gaussian_seamount", "spectral_field", "pierson_moskowitz_surface",
    "fractal_bathymetry", "internal_wave_perturbation",
    "significant_wave_height_pm", "wave_number_peak_pm",
    # extended targets
    "ExtendedTarget", "ScatteringPattern", "IsotropicScattering",
    "PlateScattering", "CylinderScattering", "rotation_matrix",
    "target_arrivals", "render_extended_echo",
    # coherent arrivals and beamforming
    "ArrivalSet", "extract_arrivals", "element_field", "beamform",
    "shading_window", "azimuth_steering",
    # ray-tube spreading
    "RayTube", "ray_tube", "ray_tube_jvp", "spherical_spreading",
    # Gaussian beams
    "GaussianBeams", "gaussian_beams", "suggest_beam_width",
    # reverberation
    "LambertScattering", "reverberation_arrivals", "render_reverberation",
    "cone_solid_angle",
    "__version__",
]
