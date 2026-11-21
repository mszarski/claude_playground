"""tools/fetch_picmus.py without the network: range reads that a server does not honour are refused, and a fetch that
fails part-way leaves no file a later run would take as done."""

import importlib.util
import io
import pathlib
import zipfile

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("fetch_picmus", ROOT / "tools" / "fetch_picmus.py")
fp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fp)


class FakeResponse(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def serve(blob, honour_range=True, length=True):
    """A urlopen stand-in for one remote file."""
    def urlopen(req, timeout=None):
        assert timeout is not None and timeout > 0                  # every request has a timeout
        if getattr(req, "get_method", lambda: "GET")() == "HEAD":
            return FakeResponse(b"", headers={"Content-Length": str(len(blob))} if length else {})
        rng = req.headers.get("Range") if hasattr(req, "headers") else None
        if rng and honour_range:
            a, b = (int(v) for v in rng.split("=")[1].split("-"))
            return FakeResponse(blob[a:b + 1], 206, {"Content-Range": f"bytes {a}-{b}/{len(blob)}"})
        return FakeResponse(blob, 200, {"Content-Length": str(len(blob))})
    return urlopen


def archive_bytes():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a/one.hdf5", b"x" * 5000)
    return buf.getvalue()


def test_range_reads_give_the_archive(monkeypatch, tmp_path):
    monkeypatch.setattr(fp.urllib.request, "urlopen", serve(archive_bytes()))
    with zipfile.ZipFile(io.BufferedReader(fp.RangeReader("https://example.invalid/a.zip"), buffer_size=1024)) as z:
        fp.fetch_member(z, "a/one.hdf5", tmp_path / "one.hdf5")
    assert (tmp_path / "one.hdf5").read_bytes() == b"x" * 5000
    assert not (tmp_path / "one.hdf5.part").exists()


def test_a_server_that_ignores_range_is_refused(monkeypatch):
    monkeypatch.setattr(fp.urllib.request, "urlopen", serve(archive_bytes(), honour_range=False))
    reader = fp.RangeReader("https://example.invalid/a.zip")
    with pytest.raises(RuntimeError, match="does not honour range requests"):
        reader.readinto(bytearray(100))


def test_no_content_length_is_refused(monkeypatch):
    monkeypatch.setattr(fp.urllib.request, "urlopen", serve(archive_bytes(), length=False))
    with pytest.raises(RuntimeError, match="Content-Length"):
        fp.RangeReader("https://example.invalid/a.zip")


def test_a_fetch_that_fails_part_way_leaves_no_file(tmp_path):
    class Broken:
        def open(self, member):
            class Src(io.BytesIO):
                def read(self, n=-1):
                    if self.tell() > 100:
                        raise zipfile.BadZipFile("Bad CRC-32")
                    return super().read(min(n, 64) if n and n > 0 else 64)
            return Src(b"y" * 1000)

    dst = tmp_path / "ref.hdf5"
    with pytest.raises(zipfile.BadZipFile):
        fp.fetch_member(Broken(), "m", dst)
    assert not dst.exists() and not (tmp_path / "ref.hdf5.part").exists()


def test_a_url_fetch_that_fails_part_way_leaves_no_file(monkeypatch, tmp_path):
    class Dropped(FakeResponse):
        def read(self, n=-1):
            if self.tell() > 10:
                raise ConnectionResetError("dropped")
            return super().read(8)

    monkeypatch.setattr(fp.urllib.request, "urlopen", lambda url, timeout=None: Dropped(b"z" * 100))
    dst = tmp_path / "data.hdf5"
    with pytest.raises(ConnectionResetError):
        fp.fetch_url("https://example.invalid/data.hdf5", dst)
    assert not dst.exists() and not (tmp_path / "data.hdf5.part").exists()
