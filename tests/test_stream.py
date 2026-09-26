"""Parity tests: the Mojo opcode decoder against the real cloudpickle output.

Every expectation is derived from `pickletools`, the reference description of
the pickle byte stream, so a wrong geometry table entry, a dropped opcode, an
off-by-one in a length prefix or a missed stream tail all show up as a
mismatch.
"""

import struct

import numpy as np
import pytest

import cloudpickle
import mojo_cloudpickle as mcp
import pickletools

REQUIRED_OPCODES = {
    "PROTO",
    "FRAME",
    "MEMOIZE",
    "STACK_GLOBAL",
    "GLOBAL",
    "NEWOBJ",
    "REDUCE",
    "TUPLE",
    "TUPLE1",
    "TUPLE2",
    "MARK",
    "NONE",
    "POP",
    "BINPUT",
    "BINGET",
    "GET",
    "PUT",
    "SHORT_BINUNICODE",
    "BINUNICODE",
    "SHORT_BINBYTES",
    "BINBYTES",
    "BYTEARRAY8",
    "LONG1",
    "LONG",
    "INT",
    "UNICODE",
    "BINFLOAT",
    "FLOAT",
    "BININT1",
    "BININT2",
    "SETITEM",
    "SETITEMS",
    "APPENDS",
    "EMPTY_SET",
    "ADDITEMS",
    "FROZENSET",
    "STOP",
}


def _closure(z):
    def inner(a, b=1):
        return a + b + z

    return inner


class Holder:
    field = 3

    def method(self, v):
        return v * self.field


def _battery():
    return {
        "closure": _closure(99),
        "lambda": lambda a: a * 2,
        "class": Holder,
        "bound_method": Holder().method,
        "small_list": [1, 2, 3],
        "big_list": list(range(300)),
        "bigint": 2**400,
        "small_bytes": b"abc",
        "big_bytes": bytes(300),
        "bytearray": bytearray(300),
        "float": 1.5,
        "set": {1, 2, 3},
        "frozenset": frozenset(range(40)),
        "complex": 1 + 2j,
        "nested": {"a": [1, {"b": (2, 3)}]},
        "code": _closure(1).__code__,
        "tuple": (),
        "none": None,
        "string": "a string with spaces",
    }


def reference(data):
    """Opcode codes, offsets and operand sizes straight from pickletools."""
    ops = list(pickletools.genops(data))
    codes = [ord(op.code) for op, _arg, _pos in ops]
    offsets = [pos for _op, _arg, pos in ops]
    sizes = []
    for i, pos in enumerate(offsets):
        nxt = offsets[i + 1] if i + 1 < len(offsets) else len(data)
        sizes.append(nxt - pos - 1)
    return codes, offsets, sizes


def test_table_covers_every_pickletools_opcode():
    from mojo_cloudpickle._lib import _tables

    mode, fix, lenw = _tables()
    for op in pickletools.opcodes:
        assert mode[ord(op.code)] != 4, f"{op.name} has no operand geometry"
    assert mode[0x2E] == 0 and fix[0x2E] == 0
    assert mode[0x95] == 0 and fix[0x95] == 8
    assert mode[0x8C] == 3 and fix[0x8C] == 1 and lenw[0x8C] == 1
    assert mode[0x58] == 3 and fix[0x58] == 4 and lenw[0x58] == 4
    assert mode[0x8D] == 3 and fix[0x8D] == 8 and lenw[0x8D] == 8
    assert mode[0x8A] == 3 and fix[0x8A] == 1 and lenw[0x8A] == 1
    assert mode[0x63] == 2
    assert mode[0x49] == 1


@pytest.mark.parametrize("proto", [0, 1, 2, 3, 4, 5])
def test_scan_matches_pickletools(proto):
    for name, obj in _battery().items():
        data = cloudpickle.dumps(obj, protocol=proto)
        codes, offsets, sizes = reference(data)
        got = mcp.scan(data)
        assert got.codes.tolist() == codes, f"{name} proto {proto} codes"
        assert got.offsets.tolist() == offsets, f"{name} proto {proto} offsets"
        assert got.arg_sizes.tolist() == sizes, f"{name} proto {proto} sizes"


def test_battery_covers_the_opcode_kinds_we_claim():
    seen = set()
    for obj in _battery().values():
        for proto in (0, 1, 2, 3, 4, 5):
            seen.update(mcp.scan(cloudpickle.dumps(obj, protocol=proto)).names())
    assert not REQUIRED_OPCODES - seen, sorted(REQUIRED_OPCODES - seen)
    assert seen <= {op.name for op in pickletools.opcodes}


HANDMADE = b"".join(
    [
        b"(",
        b"0",
        b"2",
        b"1",
        b"F3.14\n",
        b"G" + struct.pack("<d", 2.5),
        b"I42\n",
        b"J" + struct.pack("<i", -7),
        b"K" + bytes([200]),
        b"M" + struct.pack("<H", 40000),
        b"L123456789012345678901234567890L\n",
        b"Ppersid\n",
        b"Q",
        b"S'bytes'\n",
        b"T" + struct.pack("<i", 3) + b"abc",
        b"U\x05short",
        b"Vunicode\n",
        b"X" + struct.pack("<i", 3) + b"abc",
        b"\x8d" + struct.pack("<Q", 2) + b"hi",
        b"d",
        b"g0\n",
        b"h\x05",
        b"iinst\ncls\n",
        b"j" + struct.pack("<I", 70000),
        b"l",
        b"o",
        b"p1\n",
        b"q\x07",
        b"r" + struct.pack("<I", 70000),
        b"s",
        b"t",
        b"u",
        b"}\x8a\x02\x01\x02",
        b"\x82\x05",
        b"\x83\x05\x06",
        b"\x84" + struct.pack("<i", 9),
        b"\x85\x86\x88\x89\x8f\x90\x91\x81\x92\x93\x94",
        b"B" + struct.pack("<i", 2) + b"hi",
        b"C\x04tiny",
        b"\x8e" + struct.pack("<Q", 1) + b"z",
        b"\x96" + struct.pack("<Q", 2) + b"zz",
        b"aeb}" + b")" + b")" + b"(" + b"2" + b"t.",
    ]
)


def test_handmade_stream_covers_the_rest_of_the_table():
    codes, offsets, sizes = reference(HANDMADE)
    names = [mcp.opcode_name(c) for c in codes]
    assert not {"BUILD", "STRING", "INST", "OBJ", "PERSID", "EXT1"} - set(
        names
    )
    got = mcp.scan(HANDMADE)
    assert got.codes.tolist() == codes
    assert got.offsets.tolist() == offsets
    assert got.arg_sizes.tolist() == sizes
    assert got.consume() == len(HANDMADE)
    assert mcp.verify(HANDMADE)


@pytest.mark.parametrize("proto", [0, 2, 4, 5])
def test_scan_consumes_exactly_the_stream(proto):
    for obj in _battery().values():
        data = cloudpickle.dumps(obj, protocol=proto)
        got = mcp.scan(data)
        assert got.consume() == len(data)
        assert int(got.codes[-1]) == mcp.STOP
        assert mcp.verify(data)


def test_large_stream_scan():
    data = cloudpickle.dumps(list(range(200_000)), protocol=5)
    codes, offsets, sizes = reference(data)
    got = mcp.scan(data)
    assert len(got) == len(codes)
    assert got.codes.tolist() == codes
    assert got.offsets.tolist() == offsets
    assert got.arg_sizes.tolist() == sizes


def test_frame_byte_accounting():
    data = cloudpickle.dumps(list(range(200_000)), protocol=5)
    got = mcp.frames(data)
    assert len(got) >= 2, "a stream this large must use more than one frame"
    codes, offsets, sizes = reference(data)
    frame_positions = [p for c, p in zip(codes, offsets) if c == mcp.FRAME]
    assert [int(p) for p in got.offsets] == [p + 9 for p in frame_positions]

    covered = np.zeros(len(data), dtype=bool)
    for off, length in zip(got.offsets, got.lengths):
        span = slice(int(off), int(off) + int(length))
        assert not covered[span].any(), "frame payloads must not overlap"
        covered[span] = True

    starts = np.array(offsets, dtype=np.int64)
    ends = np.array(
        [p + (9 if c == mcp.FRAME else s + 1) for c, p, s in zip(codes, offsets, sizes)],
        dtype=np.int64,
    )
    running = np.concatenate([[0], np.cumsum(covered)])
    inside = running[ends] - running[starts]
    assert np.all((inside == 0) | (inside == ends - starts)), (
        "no opcode may straddle a frame boundary"
    )
    headers = np.zeros(len(data), dtype=bool)
    for pos in frame_positions:
        headers[pos : pos + 9] = True
    assert not (headers & covered).any(), "a frame header cannot sit in a payload"
    assert int(covered.sum()) + 9 * len(got) == got.total_bytes
    assert got.total_bytes == int((9 + got.lengths).sum())
    assert got.largest == int(got.lengths.max())
    assert got.smallest == int(got.lengths.min())


def test_handmade_frame_length_is_exact():
    def stream(length):
        return b"\x80\x04\x95" + struct.pack("<Q", length) + b"N."

    got = mcp.frames(stream(1))
    assert len(got) == 1
    assert got.offsets.tolist() == [11]
    assert got.lengths.tolist() == [1]
    assert got.total_bytes == 10
    assert mcp.verify(stream(1))
    assert mcp.frames(stream(2)).lengths.tolist() == [2]
    assert mcp.frames(stream(0)).lengths.tolist() == [0]
    assert mcp.scan(stream(1)).names() == ["PROTO", "FRAME", "NONE", "STOP"]


def test_frames_absent_without_framing():
    got = mcp.frames(cloudpickle.dumps({"a": 1}, protocol=3))
    assert len(got) == 0
    assert got.total_bytes == 0
    assert got.largest == 0


def test_histogram_matches_bincount():
    for obj in _battery().values():
        data = cloudpickle.dumps(obj, protocol=5)
        scan = mcp.scan(data)
        got = mcp.histogram(data)
        expect = np.bincount(scan.codes, minlength=256).astype(np.int64)
        assert got.tolist() == expect.tolist()
        assert int(got.sum()) == len(scan)


def test_histogram_of_scanned_codes_agrees():
    data = cloudpickle.dumps({"k": list(range(500))}, protocol=4)
    assert mcp.histogram(mcp.scan(data)).tolist() == mcp.histogram(data).tolist()


def test_profile_is_keyed_by_pickletools_names():
    prof = mcp.profile(cloudpickle.dumps(list(range(1000)), protocol=5))
    assert prof["BININT1"] == 256
    assert prof["BININT2"] == 744
    assert prof["PROTO"] == 1
    assert prof["STOP"] == 1
    assert set(prof) <= {op.name for op in pickletools.opcodes}


@pytest.mark.parametrize("cut", [1, 2, 5, 40])
def test_truncated_stream_is_rejected(cut):
    data = cloudpickle.dumps(list(range(2000)), protocol=5)
    assert not mcp.verify(data[:-cut])
    with pytest.raises(mcp.PickleStreamError):
        mcp.frames(data[:-cut])


def test_unknown_opcode_is_reported_with_its_offset():
    data = bytearray(cloudpickle.dumps([1, 2, 3], protocol=5))
    victim = int(mcp.scan(bytes(data)).offsets[3])
    data[victim] = 0x06
    with pytest.raises(mcp.PickleStreamError, match="unknown opcode"):
        mcp.scan(bytes(data))


def test_length_prefix_overrun_is_rejected():
    data = bytearray(cloudpickle.dumps(b"x" * 16, protocol=4))
    scan = mcp.scan(bytes(data))
    header = next(
        int(p) for c, p in zip(scan.codes, scan.offsets) if int(c) == 0x43
    )
    data[header + 1] = 0xFF
    with pytest.raises(mcp.PickleStreamError, match="truncated"):
        mcp.scan(bytes(data))


def test_frame_overrun_is_rejected():
    data = bytearray(cloudpickle.dumps(list(range(2000)), protocol=5))
    header = int(mcp.frames(bytes(data)).offsets[0]) - 9
    data[header + 1 : header + 9] = b"\xff" * 8
    with pytest.raises(mcp.PickleStreamError, match="frame length overruns"):
        mcp.frames(bytes(data))


def test_output_buffer_cap_is_reported():
    from mojo_cloudpickle._lib import _tables, lib

    data = cloudpickle.dumps(list(range(1000)), protocol=5)
    buf = np.frombuffer(data, dtype=np.uint8)
    mode, fix, lenw = _tables()
    codes = np.zeros(4, dtype=np.uint8)
    offsets = np.zeros(4, dtype=np.int32)
    args = np.zeros(4, dtype=np.int32)
    status = np.zeros(4, dtype=np.int64)
    count = lib.cp_scan(
        buf.ctypes.data, buf.size, mode.ctypes.data, fix.ctypes.data,
        lenw.ctypes.data, codes.ctypes.data, offsets.ctypes.data,
        args.ctypes.data, 4, status.ctypes.data,
    )
    assert count == 4
    assert int(status[0]) == 5


def test_real_pickle_still_loads_after_scanning():
    for obj in _battery().values():
        data = cloudpickle.dumps(obj, protocol=5)
        mcp.scan(data)
        cloudpickle.loads(data)
