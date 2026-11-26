# Changelog

## Unreleased

- Standardize rustfmt and strict Clippy checks in the editor and CI.
- Add Criterion benchmarks against equivalent Burn operations, with output
  checks and a documented CPU performance baseline.

## 0.1.0 — 2026-11-24

Initial source release.

- Add `rearrange`, `reduce`, `repeat`, and `parse_shape`.
- Support Burn 0.22.1 tensors with runtime pattern and shape validation.
- Support named axes, grouped dimensions, singletons, ellipses, and numeric
  reduction or repetition factors.
- Include reference-value tests, examples, and GitHub Actions checks.
- Release under MIT and acknowledge Alex Rogozhnikov's original Einops.
