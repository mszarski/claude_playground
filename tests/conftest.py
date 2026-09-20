import os

import pytest
import torch

_DTYPES = {"float32": torch.float32, "float64": torch.float64}


@pytest.fixture(autouse=True)
def _default_precision():
    """Every test runs in float64: these are physics assertions, not perf tests.

    ``HYDROPT_TEST_DTYPE=float32 pytest`` runs the same assertions in single
    precision instead.  Many of them are written to float64 tolerances and are
    expected to fail there; the point of the switch is to separate those from
    the ones that fail because something in the library is pinned to a float64
    constant.
    """
    name = os.environ.get("HYDROPT_TEST_DTYPE", "float64")
    if name not in _DTYPES:
        raise ValueError(f"HYDROPT_TEST_DTYPE must be one of {sorted(_DTYPES)}, "
                         f"got {name!r}")
    prev = torch.get_default_dtype()
    torch.set_default_dtype(_DTYPES[name])
    torch.manual_seed(0)
    yield
    torch.set_default_dtype(prev)
