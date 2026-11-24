# Release 0.1.0

## Verify locally

Install Rust 1.95 or newer, then run from this directory:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
cargo test --no-default-features
cargo run --example basics
cargo publish --dry-run
```

The dry run packages and checks the crate without uploading it.

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
authenticate with `cargo login`. Run `cargo publish --dry-run`, then
`cargo publish` when ready. GitHub and crates.io publication are separate steps.

Reference: [Cargo's publishing guide](https://doc.rust-lang.org/cargo/reference/publishing.html).
