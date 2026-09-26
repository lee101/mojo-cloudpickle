# mojo-cloudpickle

`cloudpickle` has no numeric core. It decides *what* to serialise, builds
reducers for functions, classes and closures, and then hands everything to
CPython's C pickler. There is no array, no statistical kernel, nothing that a
compiled inner loop would speed up.

What there *is*, in the artefact cloudpickle produces, is a real loop over a
real byte buffer: the pickle opcode stream. `mojo-cloudpickle` ports that
layer. The opcode geometry is not hard-coded in the kernel; it is derived at
import time from the real `pickletools` opcode table, and the tests compare
the decoded streams against `pickletools.genops` for every object cloudpickle
can serialise, at every protocol from 0 to 5.

The Python package is `mojo_cloudpickle`, so it installs alongside the real
`cloudpickle`.

```python
import cloudpickle, mojo_cloudpickle as mcp

data = cloudpickle.dumps(lambda x: x * 2, protocol=5)
mcp.scan(data).names()      # ['PROTO', 'FRAME', 'SHORT_BINUNICODE', ...]
mcp.profile(data)           # {'PROTO': 1, 'BININT1': 1, 'STOP': 1, ...}
mcp.frames(data).total_bytes
mcp.verify(data)            # True
```

## Covered subset

| area | implemented API |
| --- | --- |
| Opstream decode | `scan` -> parallel opcode / offset / operand-size arrays, `Scan.names`, `Scan.consume` |
| Opcode table | geometry for every opcode in `pickletools.opcodes`, protocols 0-5, derived at import |
| Framing | `frames` -> payload offsets, payload lengths, total / largest / smallest frame |
| Profile | `histogram` (256 bins), `profile` keyed by opcode name |
| Validation | `verify` predicate, `PickleStreamError` with the failing offset and opcode |

Not implemented, and left to the real `cloudpickle`: serialisation itself
(`dumps`, `Pickler`, the reducer dispatch table), `loads`, dynamic class
reconstruction, the `_extract_code_globals` closure analysis, and protocol
selection. Those are object plumbing, not numeric work.

Two deliberate limits. The decoder stops at `STOP`, as `pickletools` does, so
bytes after the last opcode are not decoded. And `LONG4` is not decoded:
pickle spells it `L` followed by `0x8b`, which a flat per-opcode geometry
table cannot express, and no CPython pickler emits it.

## Install

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` produces `dist/libmojo-cloudpickle.so`. Set
`PYTHONPATH=python` when using the package outside a Pixi task. The parity
tests need the real `cloudpickle` and `pickletools` importable, which the
shared test environment provides.

## Performance

Best-of-N wall clock in one process, on a `cloudpickle.dumps(list(range(400000)))`
stream: 1.8 MiB carrying 400833 opcodes. Every case verifies exact agreement
with its reference before timing.

| case | reference | mojo-cloudpickle | result |
| --- | ---: | ---: | ---: |
| scan 1.8 MiB / 400833 opcodes | 248.35 ms | 6.37 ms | 38.97x faster |
| scan, same stream, vs `pickletools.genops` | 421.54 ms | 6.51 ms | 64.76x faster |
| frame walk, 29 frames | 412.39 ms | 7.06 ms | 58.43x faster |
| opcode histogram, 400833 codes | 1.97 ms | 1.04 ms | 1.90x faster |

The scan baselines are a hand-written table-driven Python scanner over the same
geometry tables, which is the fastest a Python implementation of a sequential
opcode walk can be, and the real `pickletools.genops`. There is no vectorised
NumPy formulation of this job to compare against, because the walk is
inherently serial: every opcode's size decides where the next one starts. The
histogram row does have a fair vectorised baseline, `np.bincount`, and Mojo
still wins it, though a histogram is the one case where NumPy is close.

Reproduce with `pixi run bench`.

## How it works

All kernels live in `src/kernels.mojo`, one compilation unit, compiled by
`build/build.sh` with `mojo build --emit shared-lib`.

The opcode geometry is passed in as three flat byte arrays indexed by opcode
code: `mode` (fixed size, one line, two lines, or fixed prefix plus a length
field), `fix` and `lenw`. `python/mojo_cloudpickle/_lib.py` builds them from
`pickletools.opcodes`, mapping each opcode's `ArgumentDescriptor` name to a
geometry, and raises if a pickle release ever introduces a descriptor it does
not know. `cp_scan` is then a straight loop: read a code, look up the mode,
resolve the operand length, bounds-check, record, advance. `cp_frames` is the
same walk with the frame accounting collected on the way past, and
`cp_histogram` tallies the decoded codes.

Every exported symbol takes buffer addresses as plain `Int` and rebuilds the
pointer inside the body, because `@export` rejects parametric functions.

Decode results are exact integer, offset and bitwise work, so the parity tests
use `rtol=0, atol=0` equality throughout; no tolerance is needed or justified
anywhere in this port.

## License

MIT
