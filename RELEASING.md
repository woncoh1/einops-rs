# Release 0.1.0

## Development tools

`rust-toolchain.toml` selects stable Rust with rustfmt, Clippy, and rust-analyzer.
Use the default rustfmt style:

```bash
cargo fmt
```

In VS Code or VSCodium, install the recommended rust-analyzer extension. The
workspace formats Rust files on save and runs Clippy with warnings treated as
errors. CI checks the same tools and the minimum supported Rust version, 1.95.

## Verify locally

Install Rust 1.95 or newer, then run from this directory:

```bash
cargo fmt --check
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo clippy --locked --all-targets --no-default-features -- -D warnings
cargo test --locked --all-features
cargo test --locked --no-default-features
cargo doc --locked --all-features --no-deps
cargo run --locked --example basics
cargo bench --locked --bench tensor_ops -- --test
cargo publish --locked --dry-run
```

`--locked` uses the committed dependency versions. The benchmark smoke check
validates the benchmark cases without collecting performance measurements. The
publish dry run packages and checks the crate without uploading it.

## Measure performance

Run the Criterion suite on an otherwise idle machine:

```bash
cargo bench --locked --bench tensor_ops
```

Record the Rust version, CPU, tensor backend, input sizes, and benchmark settings
with results. Compare matching cases on the same machine. CI runs the correctness
smoke check only; it does not enforce timing thresholds on shared runners.

## Push to GitHub

Create an **empty** repository named `einops-rs` in your GitHub account.
The source already includes a README, license, and `.gitignore`.
For a fresh checkout with no Git history:

```bash
git init -b main
git add .
git commit -m "Initial release v0.1.0"
git remote add origin https://github.com/woncoh1/einops-rs.git
git push -u origin main
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0
```

If you use another account or repository name, update the remote and the
`repository` field in `Cargo.toml`. If the remote already has commits, clone it
first and copy these files into that checkout before committing.

After CI passes, create a GitHub release from `v0.1.0` and use the entry in
[CHANGELOG.md](CHANGELOG.md) as its release notes.

## Publish to crates.io later

Confirm the crate name is available, verify the metadata in `Cargo.toml`, and
authenticate with `cargo login`. Run `cargo publish --locked --dry-run`, then
`cargo publish --locked` when ready. GitHub and crates.io publication are separate
steps.

Reference: [Cargo's publishing guide](https://doc.rust-lang.org/cargo/reference/publishing.html).
