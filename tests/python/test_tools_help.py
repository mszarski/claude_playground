"""Every Python tool under tools/ answers `--help` with its help and does nothing else (audit B6, A14).

Each runs in a subprocess from the repository root: it must exit 0, print a usage line, and leave the working tree as
it was (no new, changed or deleted file in `git status`, no file it lists changed in content, no `--help` directory). A
tool that needs something this machine may not have is skipped only for the reason stated in SKIP.
The shell scripts in tools/ (build_*.sh, gpu_job.sh, test.sh) are not covered: they take positional arguments and have
no help option."""
import hashlib
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOLS = sorted(p.name for p in (ROOT / "tools").glob("*.py"))

# tool -> (condition under which it cannot answer --help here, reason)
SKIP = {}


def _git_state():
    out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    state = {}
    for line in out.splitlines():
        path = ROOT / line[3:].split(" -> ")[-1]
        state[line] = hashlib.sha1(path.read_bytes()).hexdigest() if path.is_file() else None
    return state


@pytest.fixture(scope="module")
def git_available():
    try:
        subprocess.run(["git", "status"], cwd=ROOT, capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout: the tree cannot be compared")


def test_every_tool_is_listed():
    assert len(TOOLS) >= 50 and "fetch_picmus.py" in TOOLS


@pytest.mark.parametrize("name", TOOLS)
def test_tool_help_only_prints_help(name, git_available):
    if name in SKIP and SKIP[name][0]():
        pytest.skip(SKIP[name][1])
    before = _git_state()
    env = dict(os.environ, OMP_NUM_THREADS="1", MPLBACKEND="Agg")
    out = subprocess.run([sys.executable, str(ROOT / "tools" / name), "--help"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, f"{name}: exit {out.returncode}\n{out.stderr[-2000:]}"
    assert "usage:" in out.stdout.lower(), f"{name} printed no usage line:\n{out.stdout[:500]}"
    assert not (ROOT / "--help").exists(), f"{name} made a directory or file named --help"
    assert _git_state() == before, f"{name} --help changed the working tree"
