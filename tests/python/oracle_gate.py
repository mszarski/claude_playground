"""Oracle tests: skip when the external program is missing, or fail when it is expected.

The outside oracles (Bellhop, KRAKEN; GPL-3 programs run as separate processes) are not part of
the suite, so a test that needs one skips without it, and a plain `pytest` passes with no oracle
run at all. Every such skip names the variable that points at the program (GPUSONAR_BELLHOP,
GPUSONAR_KRAKEN_DIR) in its reason; this plugin recognises those skips by that name.

- With GPUSONAR_REQUIRE_ORACLES=1 an oracle skip is turned into a failure (a run that is meant
  to have the oracles, such as `GPUSONAR_ORACLES=1 tools/gpu_job.sh`, then cannot pass without them).
- Without it, the terminal summary says how many tests were skipped for want of an oracle.

Loaded by tests/python/conftest.py (`pytest_plugins`).
"""

from __future__ import annotations

import os

import pytest

ORACLE_VARIABLES = ("GPUSONAR_BELLHOP", "GPUSONAR_KRAKEN_DIR")
REQUIRE_VARIABLE = "GPUSONAR_REQUIRE_ORACLES"

_skipped: dict[str, int] = {}


def oracles_required() -> bool:
    return os.environ.get(REQUIRE_VARIABLE, "").strip() not in ("", "0")


def oracle_variable(reason: str) -> str | None:
    """The oracle variable a skip reason names, or None for a skip that is not about an oracle."""
    for name in ORACLE_VARIABLES:
        if name in reason:
            return name
    return None


def _skip_reason(report) -> str:
    longrepr = report.longrepr
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        return str(longrepr[2])
    return str(longrepr)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if not report.skipped or hasattr(report, "wasxfail"):
        return
    reason = _skip_reason(report)
    name = oracle_variable(reason)
    if name is None:
        return
    if oracles_required():
        report.outcome = "failed"
        report.longrepr = (f"{REQUIRE_VARIABLE} is set, so the oracle this test needs must be present, "
                           f"but the test skipped: {reason}")
    else:
        _skipped[name] = _skipped.get(name, 0) + 1


def pytest_terminal_summary(terminalreporter):
    if not _skipped:
        return
    parts = ", ".join(f"{n} for want of {name}" for name, n in sorted(_skipped.items()))
    terminalreporter.write_line(f"oracle tests skipped: {parts} "
                                f"(set {REQUIRE_VARIABLE}=1 to make these skips fail)")
