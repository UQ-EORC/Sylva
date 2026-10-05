# Reading the core without writing Rust

Most of Sylva's computation lives in `crates/sylva-rs`, in Rust. You do not
need to write Rust to read it: the algorithms are the same ones the papers
describe, and almost all of the language that appears in them is covered
below. This page is for the reader who knows Python, R or C and wants to
check what a function actually does.

## The shape of a file

Every file opens with the licence header, then a module comment, then the
imports, then the code:

```rust
//! Pulse traversal: the accumulation half of rayvoxel.   ← what this file is

use rayon::prelude::*;                                    ← imports

/// The voxels a segment crosses, in order.               ← what this item is
pub(crate) fn walk_grid(...) { ... }
```

`//!` describes **the file it is in**; `///` describes **the item just below
it**; `//` is an ordinary comment. Doc comments become the API documentation
(`cargo doc --open`), so they are written for a reader, not a compiler.

`pub` means other modules can use it, `pub(crate)` means only the rest of this
crate can, and no marker at all means only this file. Nothing is exported by
accident.

## Where things are

| Directory | What |
|---|---|
| `io/`, `riscan.rs` | file formats: LAS/LAZ, PLY, text, RIEGL `.rxp`, RiSCAN projects |
| `filters.rs`, `cluster.rs`, `geo/` | subsampling, neighbourhoods, crops, masks, coordinates |
| `ground.rs`, `raster.rs` | ground classification, DTM/CHM rasters |
| `trees/`, `qsm/`, `leaves/` | stems, segmentation, cylinder models, foliage |
| `canopy/`, `voxel/`, `shots/` | gap fraction, ray-traced voxels, pulse data |
| `als/`, `waveform/`, `fusion/`, `change/` | airborne lidar, full waveforms, TLS+ALS, change between surveys |
| `coreg/`, `registration.rs`, `quality/` | scan alignment and scan quality |
| `synthetic/` | simulated trees, plots and scans used by the tests |
| `util/` | shared machinery: progress, memory limits, ordered parallelism, NumPy-exact helpers |

Two types carry most of the data. `PointCloud` is an `(N, 3)` array of
coordinates plus named attributes; `Shots` is pulse-centric — per-pulse origin
and direction with a compressed list of echoes. A `Point` is simply
`[f64; 3]`, an array of three doubles, and `Attr` is "an attribute column of
one of nine numeric types" (`pointcloud.rs:18`).

## The constructs that carry the code

**`&` is a borrow, not a pointer you have to manage.** `&[Point]` is a *view*
of a run of points — a NumPy slice, in effect: no copy is made, and the
compiler guarantees the data outlives the view. `&mut` is a view you may write
through, and only one may exist at a time, which is how the language rules out
two threads writing the same array.

**`Option<T>` is "a `T` or nothing"** — Python's `None`, but you cannot forget
to check it. `Result<T>` is "a `T` or an error". The `?` after a call means
"if this failed, return that error to my caller", so a chain of fallible steps
reads top to bottom instead of nesting:

```rust
let file = File::open(path)?;        // on failure, return the error
```

`let Some(idx) = self.index(cell) else { return ... };` takes the value when
there is one and runs the block when there is not
(`voxel/traverse.rs`).

**Iterators are the loop idiom.** `points.iter().map(...).filter(...).collect()`
builds a new vector; nothing is evaluated until `collect`, and the chain
compiles to the same machine code as the hand-written loop.

**Parallel loops are the same chain with `par_iter`.** Rayon spreads the work
over the machine's cores:

```rust
let hits: Vec<_> = (0..shots.n_shots())
    .into_par_iter()         // parallel loop over pulse numbers
    .with_min_len(256)       // in chunks, so the bookkeeping is not the cost
    .fold(|| Local::default(), |mut local, s| { ...; local })   // per-thread state
    .reduce(Vec::new, |mut a, mut b| { a.append(&mut b); a });  // join them
```

`fold` gives every thread its own scratch and output; `reduce` merges those.
This is how `Engine::add` traces pulses (`voxel/traverse.rs`).

**Shared counters are atomics.** Where threads must write the same grid, each
accumulator is an atomic the hardware updates indivisibly, so no locking is
needed. Floating-point sums have no atomic type, so they are kept as the
`f64`'s bits inside a 64-bit integer and updated with a compare-and-swap loop
— read, add, put back if nobody else changed it meanwhile (`fadd`,
`voxel/traverse.rs`).

**Where order matters, work is handed out in order.** `util::relay::ordered_map`
runs a function over a list on N threads but passes results on strictly in
list order, so a run's output and its log do not depend on how many cores it
had. The coregistration pipeline depends on this.

**`'a` is a lifetime.** In `struct Parser<'a> { src: &'a str, ... }` it says
the parser borrows the expression text and so cannot outlive it. Lifetimes
have no run-time cost and change no behaviour; they are notes to the compiler,
and can be read past.

**`impl Trait` in an argument is a callback or a constraint.** `visit: impl
FnMut(...) -> bool` takes any closure; `f: impl Fn(usize) -> T + Sync` takes
one that is safe to call from several threads. In a return type,
`impl Iterator<Item = (usize, usize)>` means "some lazy sequence", which is
how `BlockedGrid::slabs` walks a grid taller than memory
(`voxel/blocked.rs`).

**`impl Thing for Type` adds behaviour to a type** — the nearest thing to a
class method. `Default` provides the default settings structs, `Display`
provides the text form used in error messages.

**`unsafe` appears in exactly one place**: loading RiVLib, the proprietary
RIEGL library, at run time (`io/riegl.rs`). Calling into C cannot be checked
by the compiler, so each block carries a `SAFETY:` note saying why it is
sound. Nothing else in the crate uses it.

## Conventions

*Errors.* Functions that can fail return `Result<T>`; the error type is in
`error.rs` and carries the file path where there is one. Messages are written
for the person running the command, not for the developer.

*Progress.* Long work opens a task with `util::progress::start(label, total)`
and counts up; nothing is printed unless a caller is watching. See
`sylva.progress` in the Python docs.

*Memory.* Allocations that scale with settings rather than with the data —
voxel grids, neighbour graphs — are sized and checked first
(`util::limits`), so an impossible request is refused with a message instead
of killing the process.

*Tests.* Unit tests live at the bottom of the file they test, inside
`#[cfg(test)] mod tests`, which is compiled only by `cargo test`. They are the
quickest description of what a function is expected to do, and they read as
plain assertions.

## From Python to the core

A Python call reaches Rust in three short steps. Taking `crop_box_mask`:

1. `python/sylva/filters.py` — the public function: docstring, argument
   checking, NumPy in and out.
2. `crates/sylva-py/src/filters_py.rs` — the binding. It converts and nothing
   else:

    ```rust
    #[pyfunction]
    fn crop_box_mask<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, min_xyz: [f64; 3], max_xyz: [f64; 3])
        -> PyResult<Bound<'py, PyArray1<bool>>> {
        let p = xyz_from_py(xyz)?;                                   // NumPy array -> &[Point]
        Ok(filters::crop_box_mask(&p, min_xyz, max_xyz).into_pyarray(py))  // Vec<bool> -> NumPy array
    }
    ```

3. `crates/sylva-rs/src/filters.rs` — the computation itself.

So to find what a Python function does, open the file of the same name under
`crates/sylva-rs/src/`; the binding in between is only plumbing. The division
of labour, and why it is drawn there, is in
[The Rust core](rust-core.md).
