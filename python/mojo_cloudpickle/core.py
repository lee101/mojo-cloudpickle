"""Pickle byte-stream inspection built on the Mojo kernels.

Object serialisation itself is forwarded to the real ``cloudpickle``; what
lives here is the decoder for the byte stream it produces: the opcode walk,
the framing arithmetic, and the opcode histogram.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._lib import ERRORS, histogram as _histogram
from ._lib import frames as _frames
from ._lib import opcode_name, scan as _scan

STOP = 0x2E
FRAME = 0x95


class PickleStreamError(ValueError):
    """Raised when a byte stream is not a well-formed pickle."""


@dataclass(frozen=True)
class Scan:
    codes: np.ndarray
    offsets: np.ndarray
    arg_sizes: np.ndarray

    def __len__(self) -> int:
        return int(self.codes.size)

    def names(self) -> list[str]:
        return [opcode_name(int(c)) for c in self.codes]

    def consume(self) -> int:
        return int(self.offsets[-1] + 1 + self.arg_sizes[-1])


@dataclass(frozen=True)
class FrameInfo:
    offsets: np.ndarray
    lengths: np.ndarray
    total_bytes: int
    largest: int
    smallest: int

    def __len__(self) -> int:
        return int(self.offsets.size)


def _check(status: np.ndarray) -> None:
    code = int(status[0])
    if code:
        raise PickleStreamError(
            f"{ERRORS.get(code, code)} at offset {int(status[2])}"
            + (f" (opcode {opcode_name(int(status[1]))})" if code == 2 else "")
        )


def scan(data: bytes) -> Scan:
    """Decode the pickle opcode stream in ``data``."""
    codes, offsets, args, status = _scan(bytes(data))
    if int(status[0]):
        _check(status)
    return Scan(codes, offsets, args)


def frames(data: bytes) -> FrameInfo:
    """Collect the frame headers a protocol 4+ stream carries."""
    offsets, lengths, totals, status = _frames(bytes(data))
    _check(status)
    return FrameInfo(
        offsets, lengths, int(totals[1]), int(totals[2]), int(totals[3])
    )


def histogram(data_or_scan) -> np.ndarray:
    """Opcode histogram of a byte stream or an already decoded scan."""
    if isinstance(data_or_scan, Scan):
        codes = data_or_scan.codes
    else:
        codes = scan(data_or_scan).codes
    return _histogram(codes)


def profile(data: bytes) -> dict[str, int]:
    """Opcode histogram keyed by opcode name, as produced by cloudpickle."""
    bins = histogram(data)
    out = {}
    for code in np.nonzero(bins)[0]:
        out[opcode_name(int(code))] = int(bins[code])
    return out


def verify(data: bytes) -> bool:
    """True when the stream decodes cleanly, ends on STOP and consumes
    exactly the whole buffer with no frame that overruns it. Never raises."""
    try:
        s = scan(data)
        f = frames(data)
    except PickleStreamError:
        return False
    if len(s) == 0 or int(s.codes[-1]) != STOP or s.consume() != len(data):
        return False
    if f.total_bytes > len(data):
        return False
    return not len(f) or int(f.offsets[-1]) + int(f.lengths[-1]) <= len(data)
