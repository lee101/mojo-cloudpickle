"""Correctness-gated benchmark for mojo-cloudpickle.

Decoding a pickle opcode stream is inherently sequential, so there is no
vectorised NumPy formulation to race against. The two baselines are therefore
the real ``pickletools.genops`` and a hand-written table-driven Python scanner
using the same geometry tables, which is the fastest a Python implementation
of this job can be. The histogram case does have a fair vectorised baseline:
``np.bincount``.
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import cloudpickle  # noqa: E402
import mojo_cloudpickle as mcp  # noqa: E402
import pickletools  # noqa: E402

from mojo_cloudpickle._lib import _tables  # noqa: E402


def _time(fn, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def python_scan(data: bytes):
    """Table-driven Python scanner over the same opcode geometry."""
    mode, fix, lenw = _tables()
    mode = mode.tolist()
    fix = fix.tolist()
    lenw = lenw.tolist()
    view = memoryview(data)
    out = []
    i = 0
    n = len(data)
    while i < n:
        code = view[i]
        m = mode[code]
        if m == 0:
            size = fix[code]
        elif m == 1:
            size = view.index(b"\n", i + 1) - i
        elif m == 2:
            first = view.index(b"\n", i + 1) - i
            size = first + view.index(b"\n", i + 1 + first) - i
        else:
            size = fix[code] + int.from_bytes(view[i + 1 : i + 1 + lenw[code]], "little")
        out.append((code, i, size))
        if code == 0x2E:
            break
        i += 1 + size
    return out


def bench_scan(n=400_000):
    data = cloudpickle.dumps(list(range(n)), protocol=5)
    ref = python_scan(data)
    got = mcp.scan(data)
    assert got.codes.tolist() == [c for c, _, _ in ref], "opcode mismatch"
    assert got.offsets.tolist() == [p for _, p, _ in ref], "offset mismatch"
    assert got.arg_sizes.tolist() == [s for _, _, s in ref], "size mismatch"

    return (
        f"scan {len(data) >> 10}KiB/{len(ref)} ops",
        _time(lambda: python_scan(data), 3),
        _time(lambda: mcp.scan(data), 5),
    )


def bench_frames(n=400_000):
    data = cloudpickle.dumps(list(range(n)), protocol=5)
    got = mcp.frames(data)
    ref = [(p + 9, None) for op, _a, p in pickletools.genops(data) if op.name == "FRAME"]
    assert [int(x) for x in got.offsets] == [p for p, _ in ref], "frame offsets"

    def python_frames():
        out = []
        for op, _a, p in pickletools.genops(data):
            if op.name == "FRAME":
                out.append((p + 9, int.from_bytes(data[p + 1 : p + 9], "little")))
        return out

    expect = python_frames()
    assert [int(x) for x in got.lengths] == [l for _, l in expect]
    return (
        f"frames {len(got)}",
        _time(python_frames, 3),
        _time(lambda: mcp.frames(data), 5),
    )


def bench_histogram(n=400_000):
    data = cloudpickle.dumps(list(range(n)), protocol=5)
    scan = mcp.scan(data)
    got = mcp.histogram(scan)
    assert got.tolist() == np.bincount(scan.codes, minlength=256).tolist()
    return (
        f"histogram {len(scan)}",
        _time(lambda: np.bincount(scan.codes, minlength=256), 5),
        _time(lambda: mcp.histogram(scan), 5),
    )


def main():
    print(f"{'case':<26}{'python/pickletools':>21}{'mojo-cloudpickle':>20}{'ratio':>9}")
    print("-" * 76)
    for fn in (bench_scan, bench_frames, bench_histogram):
        label, ref, got = fn()
        print(f"{label:<26}{ref*1e3:>19.2f}ms{got*1e3:>18.2f}ms{ref/got:>8.2f}x")

    data = cloudpickle.dumps(list(range(400_000)), protocol=5)
    genops = _time(lambda: [op.name for op, _a, _p in pickletools.genops(data)], 3)
    mojo = _time(lambda: mcp.scan(data), 5)
    print(f"{'scan vs pickletools':<26}{genops*1e3:>19.2f}ms{mojo*1e3:>18.2f}ms{genops/mojo:>8.2f}x")


if __name__ == "__main__":
    main()
