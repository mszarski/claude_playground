import pytest
import torch


@pytest.fixture(autouse=True)
def _double_precision():
    """Every test runs in float64: these are physics assertions, not perf tests."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    yield
    torch.set_default_dtype(prev)
