"""Byte-level pickle stream kernels for the cloudpickle port.

cloudpickle is object plumbing: it decides *what* to serialise and forwards to
CPython's C pickler. The one part of the resulting artefact that is a genuine
loop over a byte buffer is the pickle opcode stream itself, so that is what is
ported here: a table-driven decoder for pickle opcodes (protocol 0-5) plus the
frame bookkeeping that protocol 4+ framing adds.

The opcode geometry table is not hard-coded here. The Python shim derives it
from `pickletools.opcodes` and passes it in as three flat byte arrays:

    mode[256]   0 fixed size, 1 one newline-terminated line,
                2 two such lines, 3 fixed prefix plus a length field
    fix[256]    the fixed size, or the prefix length when mode == 3
    lenw[256]   the width in bytes of the big-endian length field, mode == 3

Every exported symbol takes buffer addresses as plain `Int` values and rebuilds
the pointer inside the body, because `@export` rejects parametric functions and
an inferred pointer origin would make the symbol parametric.
"""

comptime U8Ptr = Pointer[UInt8, AnyOrigin[mut=True]]
comptime I32Ptr = Pointer[Int32, AnyOrigin[mut=True]]
comptime I64Ptr = Pointer[Int64, AnyOrigin[mut=True]]

comptime MODE_FIXED: Int = 0
comptime MODE_LINE: Int = 1
comptime MODE_PAIR: Int = 2
comptime MODE_PREFIX: Int = 3
comptime MODE_BAD: Int = 4

comptime ERR_NONE: Int64 = 0
comptime ERR_TRUNCATED: Int64 = 1
comptime ERR_BAD_OPCODE: Int64 = 2
comptime ERR_SHORT_LINE: Int64 = 3
comptime ERR_LENGTH_OVERFLOW: Int64 = 4
comptime ERR_FULL: Int64 = 5
comptime ERR_BAD_FRAME: Int64 = 6


def _line_len(p: U8Ptr, start: Int, n: Int) -> Int:
    var i = start
    while i < n and p[unsafe_offset=i] != 0x0a:
        i += 1
    if i >= n:
        return -1
    return i - start + 1


def _le_uint(p: U8Ptr, at: Int, width: Int) -> Int64:
    """Every binary argument, FRAME's 8-byte length included, is
    little-endian."""
    var acc: Int64 = 0
    var k = width - 1
    while k >= 0:
        acc = (acc << 8) | Int64(p[unsafe_offset=at + k])
        if acc < 0:
            return -1
        k -= 1
    return acc



@export("cp_scan")
def cp_scan(
    data_addr: Int,
    n: Int,
    mode_addr: Int,
    fix_addr: Int,
    lenw_addr: Int,
    codes_addr: Int,
    pos_addr: Int,
    arg_addr: Int,
    max_ops: Int,
    status_addr: Int,
) abi("C") -> Int64:
    """Decode one pickle stream into parallel opcode, offset and operand-size
    arrays. Returns the number of opcodes decoded; status[0] is an error code
    and status[2] the offset it was raised at, both zero on success."""
    var p = U8Ptr(unsafe_from_address=data_addr)
    var mode = U8Ptr(unsafe_from_address=mode_addr)
    var fix = I32Ptr(unsafe_from_address=fix_addr)
    var lenw = U8Ptr(unsafe_from_address=lenw_addr)
    var codes = U8Ptr(unsafe_from_address=codes_addr)
    var pos_out = I32Ptr(unsafe_from_address=pos_addr)
    var arg_out = I32Ptr(unsafe_from_address=arg_addr)
    var status = I64Ptr(unsafe_from_address=status_addr)

    status[unsafe_offset=0] = ERR_NONE
    status[unsafe_offset=1] = 0
    status[unsafe_offset=2] = 0

    var i = 0
    var count = 0
    while i < n:
        if count >= max_ops:
            status[unsafe_offset=0] = ERR_FULL
            status[unsafe_offset=2] = Int64(i)
            break
        var code = Int(p[unsafe_offset=i])
        var m = Int(mode[unsafe_offset=code])
        var size: Int = 0
        if m == MODE_FIXED:
            size = Int(fix[unsafe_offset=code])
        elif m == MODE_LINE:
            size = _line_len(p, i + 1, n)
            if size < 0:
                status[unsafe_offset=0] = ERR_SHORT_LINE
                status[unsafe_offset=2] = Int64(i)
                break
        elif m == MODE_PAIR:
            var first = _line_len(p, i + 1, n)
            if first < 0:
                status[unsafe_offset=0] = ERR_SHORT_LINE
                status[unsafe_offset=2] = Int64(i)
                break
            var second = _line_len(p, i + 1 + first, n)
            if second < 0:
                status[unsafe_offset=0] = ERR_SHORT_LINE
                status[unsafe_offset=2] = Int64(i)
                break
            size = first + second
        elif m == MODE_PREFIX:
            var width = Int(lenw[unsafe_offset=code])
            var base = Int(fix[unsafe_offset=code])
            if i + 1 + width > n:
                status[unsafe_offset=0] = ERR_TRUNCATED
                status[unsafe_offset=2] = Int64(i)
                break
            var length = _le_uint(p, i + 1, width)
            if length < 0:
                status[unsafe_offset=0] = ERR_LENGTH_OVERFLOW
                status[unsafe_offset=2] = Int64(i)
                break
            size = base + Int(length)
        else:
            status[unsafe_offset=0] = ERR_BAD_OPCODE
            status[unsafe_offset=1] = Int64(code)
            status[unsafe_offset=2] = Int64(i)
            break

        if i + 1 + size > n:
            status[unsafe_offset=0] = ERR_TRUNCATED
            status[unsafe_offset=2] = Int64(i)
            break

        codes[unsafe_offset=count] = UInt8(code)
        pos_out[unsafe_offset=count] = Int32(i)
        arg_out[unsafe_offset=count] = Int32(size)
        count += 1

        if code == 0x2e:
            break
        i += 1 + size

    return Int64(count)


@export("cp_frames")
def cp_frames(
    data_addr: Int,
    n: Int,
    mode_addr: Int,
    fix_addr: Int,
    lenw_addr: Int,
    off_addr: Int,
    len_addr: Int,
    max_frames: Int,
    totals_addr: Int,
    status_addr: Int,
) abi("C") -> Int64:
    """Walk the same stream, collecting every FRAME header (protocol >= 4).

    totals = [n_frames, bytes held by frame headers plus payloads,
    largest frame payload, smallest frame payload]. Returns the number of
    frames found."""
    var p = U8Ptr(unsafe_from_address=data_addr)
    var mode = U8Ptr(unsafe_from_address=mode_addr)
    var fix = I32Ptr(unsafe_from_address=fix_addr)
    var lenw = U8Ptr(unsafe_from_address=lenw_addr)
    var off_out = I32Ptr(unsafe_from_address=off_addr)
    var len_out = I64Ptr(unsafe_from_address=len_addr)
    var totals = I64Ptr(unsafe_from_address=totals_addr)
    var status = I64Ptr(unsafe_from_address=status_addr)

    status[unsafe_offset=0] = ERR_NONE
    status[unsafe_offset=1] = 0
    status[unsafe_offset=2] = 0
    totals[unsafe_offset=0] = 0
    totals[unsafe_offset=1] = 0
    totals[unsafe_offset=2] = 0
    totals[unsafe_offset=3] = 0

    var i = 0
    var count = 0
    var span = 0
    var biggest: Int64 = 0
    var smallest: Int64 = -1
    while i < n:
        var code = Int(p[unsafe_offset=i])
        var m = Int(mode[unsafe_offset=code])
        var size: Int = 0
        if m == MODE_FIXED:
            size = Int(fix[unsafe_offset=code])
        elif m == MODE_LINE:
            size = _line_len(p, i + 1, n)
            if size < 0:
                status[unsafe_offset=0] = ERR_SHORT_LINE
                status[unsafe_offset=2] = Int64(i)
                break
        elif m == MODE_PAIR:
            var first = _line_len(p, i + 1, n)
            if first < 0:
                status[unsafe_offset=0] = ERR_SHORT_LINE
                status[unsafe_offset=2] = Int64(i)
                break
            var second = _line_len(p, i + 1 + first, n)
            if second < 0:
                status[unsafe_offset=0] = ERR_SHORT_LINE
                status[unsafe_offset=2] = Int64(i)
                break
            size = first + second
        elif m == MODE_PREFIX:
            var width = Int(lenw[unsafe_offset=code])
            var base = Int(fix[unsafe_offset=code])
            if i + 1 + width > n:
                status[unsafe_offset=0] = ERR_TRUNCATED
                status[unsafe_offset=2] = Int64(i)
                break
            var length = _le_uint(p, i + 1, width)
            if length < 0:
                status[unsafe_offset=0] = ERR_LENGTH_OVERFLOW
                status[unsafe_offset=2] = Int64(i)
                break
            size = base + Int(length)
        else:
            status[unsafe_offset=0] = ERR_BAD_OPCODE
            status[unsafe_offset=1] = Int64(code)
            status[unsafe_offset=2] = Int64(i)
            break

        if i + 1 + size > n:
            status[unsafe_offset=0] = ERR_TRUNCATED
            status[unsafe_offset=2] = Int64(i)
            break

        if code == 0x95:
            if count >= max_frames:
                status[unsafe_offset=0] = ERR_FULL
                status[unsafe_offset=2] = Int64(i)
                break
            var payload = _le_uint(p, i + 1, 8)
            if payload < 0 or i + 9 + Int(payload) > n:
                status[unsafe_offset=0] = ERR_BAD_FRAME
                status[unsafe_offset=1] = payload
                status[unsafe_offset=2] = Int64(i)
                break
            off_out[unsafe_offset=count] = Int32(i + 9)
            len_out[unsafe_offset=count] = payload
            count += 1
            span += 9 + Int(payload)
            if payload > biggest:
                biggest = payload
            if smallest < 0 or payload < smallest:
                smallest = payload

        if code == 0x2e:
            break
        i += 1 + size

    totals[unsafe_offset=0] = Int64(count)
    totals[unsafe_offset=1] = Int64(span)
    totals[unsafe_offset=2] = biggest
    if smallest < 0:
        smallest = 0
    totals[unsafe_offset=3] = smallest
    return Int64(count)


@export("cp_histogram")
def cp_histogram(codes_addr: Int, count: Int, bins_addr: Int) abi("C"):
    """Tally the decoded opcode codes into 256 bins."""
    var codes = U8Ptr(unsafe_from_address=codes_addr)
    var bins = I64Ptr(unsafe_from_address=bins_addr)
    for i in range(256):
        bins[unsafe_offset=i] = 0
    var i = 0
    while i < count:
        bins[unsafe_offset=Int(codes[unsafe_offset=i])] += 1
        i += 1
