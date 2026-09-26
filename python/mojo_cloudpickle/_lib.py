"""ctypes bridge to the compiled Mojo kernels.

The shared library owns no memory. Every buffer crosses the C ABI as a 64-bit
address, so the argtypes below must stay `c_int64` for addresses; `c_int`
truncates them and segfaults.
"""

from __future__ import annotations

import ctypes
import pathlib

import numpy as np
import pickletools

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-cloudpickle.so"

MODE_FIXED, MODE_LINE, MODE_PAIR, MODE_PREFIX, MODE_BAD = 0, 1, 2, 3, 4

_GEOMETRY = {
    None: (MODE_FIXED, 0, 0),
    "uint1": (MODE_FIXED, 1, 0),
    "uint2": (MODE_FIXED, 2, 0),
    "uint4": (MODE_FIXED, 4, 0),
    "int4": (MODE_FIXED, 4, 0),
    "float8": (MODE_FIXED, 8, 0),
    "decimalnl_short": (MODE_LINE, 0, 0),
    "decimalnl_long": (MODE_LINE, 0, 0),
    "uint8": (MODE_FIXED, 8, 0),
    "stringnl": (MODE_LINE, 0, 0),
    "stringnl_noescape": (MODE_LINE, 0, 0),
    "unicodestringnl": (MODE_LINE, 0, 0),
    "floatnl": (MODE_LINE, 0, 0),
    "stringnl_noescape_pair": (MODE_PAIR, 0, 0),
    "long1": (MODE_PREFIX, 1, 1),
    "long4": (MODE_PREFIX, 4, 4),
    "string1": (MODE_PREFIX, 1, 1),
    "string4": (MODE_PREFIX, 4, 4),
    "string8": (MODE_PREFIX, 8, 8),
    "bytes1": (MODE_PREFIX, 1, 1),
    "bytes4": (MODE_PREFIX, 4, 4),
    "bytes8": (MODE_PREFIX, 8, 8),
    "unicodestring1": (MODE_PREFIX, 1, 1),
    "unicodestring4": (MODE_PREFIX, 4, 4),
    "unicodestring8": (MODE_PREFIX, 8, 8),
    "bytearray8": (MODE_PREFIX, 8, 8),
}

NAMES = ["<unknown>"] * 256

_TABLES = None


def _tables():
    """The opcode geometry table, derived from the real pickletools tables."""
    global _TABLES
    if _TABLES is None:
        mode = np.full(256, MODE_BAD, dtype=np.uint8)
        fix = np.zeros(256, dtype=np.int32)
        lenw = np.zeros(256, dtype=np.uint8)
        for op in pickletools.opcodes:
            code = ord(op.code)
            arg = None if op.arg is None else op.arg.name
            if arg not in _GEOMETRY:
                raise RuntimeError(f"unmapped pickle argument descriptor {arg!r}")
            m, f, w = _GEOMETRY[arg]
            mode[code] = m
            fix[code] = f
            lenw[code] = w
            NAMES[code] = op.name
        _TABLES = (mode, fix, lenw)
    return _TABLES


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(f"{_LIB_PATH} not found; run `bash build/build.sh` first")
    lib = ctypes.CDLL(str(_LIB_PATH))
    i64, i64p = ctypes.c_int64, ctypes.c_int64
    lib.cp_scan.restype = i64
    lib.cp_scan.argtypes = [i64] * 10
    lib.cp_frames.restype = i64
    lib.cp_frames.argtypes = [i64] * 10
    lib.cp_histogram.restype = None
    lib.cp_histogram.argtypes = [i64, ctypes.c_int64, i64]
    return lib


lib = _load()

ERRORS = {
    0: "ok",
    1: "truncated operand",
    2: "unknown opcode",
    3: "unterminated text line",
    4: "declared length overflows int64",
    5: "output buffer full",
    6: "frame length overruns the stream",
}


def opcode_name(code: int) -> str:
    return NAMES[code]


def scan(data: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode a pickle stream.

    Returns ``(codes, offsets, arg_sizes, status)`` where ``codes`` is uint8,
    the other two int32, and ``status`` is ``[error, detail, position]``.
    """
    buf = np.frombuffer(data, dtype=np.uint8)
    n = int(buf.size)
    mode, fix, lenw = _tables()
    codes = np.empty(max(n, 1), dtype=np.uint8)
    offsets = np.empty(max(n, 1), dtype=np.int32)
    args = np.empty(max(n, 1), dtype=np.int32)
    status = np.zeros(4, dtype=np.int64)
    count = int(
        lib.cp_scan(
            buf.ctypes.data,
            n,
            mode.ctypes.data,
            fix.ctypes.data,
            lenw.ctypes.data,
            codes.ctypes.data,
            offsets.ctypes.data,
            args.ctypes.data,
            max(n, 1),
            status.ctypes.data,
        )
    )
    return codes[:count], offsets[:count], args[:count], status


def frames(data: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(payload_offsets, payload_lengths, totals, status)``."""
    buf = np.frombuffer(data, dtype=np.uint8)
    n = int(buf.size)
    mode, fix, lenw = _tables()
    cap = 64
    while True:
        offs = np.empty(max(cap, 1), dtype=np.int32)
        lens = np.empty(max(cap, 1), dtype=np.int64)
        totals = np.zeros(4, dtype=np.int64)
        status = np.zeros(4, dtype=np.int64)
        count = int(
            lib.cp_frames(
                buf.ctypes.data,
                n,
                mode.ctypes.data,
                fix.ctypes.data,
                lenw.ctypes.data,
                offs.ctypes.data,
                lens.ctypes.data,
                max(cap, 1),
                totals.ctypes.data,
                status.ctypes.data,
            )
        )
        if status[0] != 5 or count < cap:
            return offs[:count], lens[:count], totals, status
        cap *= 4


def histogram(codes: np.ndarray) -> np.ndarray:
    """Tally decoded opcode codes into 256 bins."""
    codes = np.ascontiguousarray(codes, dtype=np.uint8)
    bins = np.zeros(256, dtype=np.int64)
    lib.cp_histogram(codes.ctypes.data, int(codes.size), bins.ctypes.data)
    return bins
