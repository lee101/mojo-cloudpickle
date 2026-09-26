"""mojo-cloudpickle: Mojo kernels for the pickle byte stream cloudpickle emits.

Installable alongside the real ``cloudpickle`` package, whose output these
kernels are tested against opcode by opcode.
"""

from ._lib import ERRORS, opcode_name
from .core import (
    FRAME,
    STOP,
    FrameInfo,
    PickleStreamError,
    Scan,
    frames,
    histogram,
    profile,
    scan,
    verify,
)

__all__ = [
    "ERRORS",
    "FRAME",
    "STOP",
    "FrameInfo",
    "PickleStreamError",
    "Scan",
    "frames",
    "histogram",
    "opcode_name",
    "profile",
    "scan",
    "verify",
]
__version__ = "0.1.0"
