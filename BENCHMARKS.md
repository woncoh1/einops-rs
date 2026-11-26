# Performance benchmarks

## Results — 2026-11-26

The current runtime API adds several microseconds to small operations. The largest
relative gap is splitting attention heads: direct Burn can retain a strided view,
while the current einops-rs executor flattens after swapping axes and materializes
that data. Medium repetition, mean reduction, and permutation-plus-merge were
about **1.15–1.26x** the direct Burn latency in this run.

**Mean wall time per call, in microseconds (µs).** Ratio = einops-rs / direct Burn,
computed from unrounded means; larger values mean higher latency.

| Operation | Input shape | Direct Burn (µs) | einops-rs (µs) | Ratio |
| --- | --- | ---: | ---: | ---: |
| Flatten (small) | `[2, 8, 32]` | 0.206 | 3.351 | 16.27x |
| Flatten (medium) | `[8, 128, 256]` | 0.184 | 4.306 | 23.38x |
| Split heads (small) | `[2, 8, 32], h=4` | 0.258 | 7.303 | 28.35x |
| Split heads (medium) | `[8, 128, 256], h=8` | 0.529 | 206.805 | 391.22x |
| Permute + merge (small) | `[2, 4, 4, 8]` | 0.865 | 6.603 | 7.63x |
| Permute + merge (medium) | `[8, 32, 32, 32]` | 1009.467 | 1226.522 | 1.22x |
| Repeat x4 (small) | `[4, 32]` | 0.571 | 4.667 | 8.17x |
| Repeat x4 (medium) | `[128, 256]` | 29.768 | 34.364 | 1.15x |
| Mean over time (small) | `[2, 8, 32]` | 0.548 | 5.179 | 9.45x |
| Mean over time (medium) | `[8, 128, 256]` | 33.478 | 42.121 | 1.26x |

`parse_shape(&[8, 128, 256], "batch time width")` averaged
**0.676 µs**. It has no tensor-computation baseline.

These are approximate results from one shared Linux runner, not GPU measurements
or end-to-end model throughput. Noise is visible: the medium direct-Burn head-split
mean has a 95% confidence interval of **0.366–0.756 µs**, so its large ratio is
especially approximate. All mean/median estimates and their 95% confidence
intervals are retained in [the measurement data](benchmarks/flex-cpu-2026-11-26.json).

## Reproduce

```bash
cargo bench --locked --bench tensor_ops
```

Open `target/criterion/report/index.html` for Criterion's HTML reports. The source
is [benches/tensor_ops.rs](benches/tensor_ops.rs); the dependency versions are locked
in `Cargo.lock`. CI checks the cases with `-- --test`, without enforcing timings.

## Method

- Intel Xeon Platinum 8573C; 9 visible logical CPUs; x86-64 Linux container.
- Rust 1.100.0, LLVM 23.1.5; default optimized Cargo bench profile, no `RUSTFLAGS`.
- Burn 0.22.1, Flex CPU, `f32`, autodiff disabled; Criterion 0.8.2.
- 50 samples per case, 1-second warmup, 2-second target measurement time.
- Each pair checks shapes, dtype, and values using deterministic nonconstant
  inputs before timing. Both paths include input-handle cloning, synchronization,
  and output destruction; input construction and tensor readback are excluded.
- Direct Burn uses the known shape parameters. einops-rs includes runtime parsing,
  validation, planning, and execution on every call. This is a complete-call
  comparison; the difference does not isolate parsing alone.
- Flex executes these operations eagerly; its explicit synchronization is a no-op.
  Direct Burn may retain views. The permutation-plus-merge case requires
  materialization in both paths.

Library source is unchanged from
[`76b68e5`](https://github.com/woncoh1/einops-rs/commit/76b68e585235a21cf69c2ab08f6c36cd2334064a).
This baseline supports prioritizing fewer reshapes/materializations and cheaper
repeated planning. It does not measure an optimized or cached implementation.
