# PR #90 — Status Tracker

Fork-local notes tracking the status of [TheTom/turboquant_plus#90][pr90]
(my bundled fixes & tests). Not part of the upstream project docs.

[pr90]: https://github.com/TheTom/turboquant_plus/pull/90

## Summary

PR #90 proposed six commits. The maintainer reviewed it, curated five
items into a separate cherry-pick PR ([#91][pr91]) with himself as author
on the cherry-picks (commit attribution preserved), and deferred three
items for stated reasons. As of writing, both #90 and #91 are open; the
upstream `main` branch has not yet incorporated any of the curated items.

[pr91]: https://github.com/TheTom/turboquant_plus/pull/91

## State by item

### Curated into PR #91 (awaiting merge to `main`)

| Item | PR #91 cherry-pick |
|---|---|
| V-norm fix in `KVCacheCompressor.memory_stats` + `TurboQuantMSE.compressed_size_bits` | `b75813b0` |
| `SeedSequence(seed).spawn(2)` PRNG cleanup (replaces `seed + 1000` offset) | `b75813b0` (same commit) |
| QJL regression-guard assertion in `test_turboquant_improves_over_polarquant` | `10746253` |
| `fast_rotate` / `fast_unrotate` correctness + round-trip tests | `f23570e0` |
| Ruff config in `pyproject.toml` (workflow dropped per follow-up) | `3e37572a` + `0ca5bcc3` |

#91 also carries a parallel fix authored by the maintainer:

- `8afc4bf3` — K-side norm accounting (`compressed_size_bits` for
  TurboQuant K was undercounting; K stores two norms — `vector_norm` and
  `residual_norm` — and only one was being counted). Surfaced by the
  V-side fix above.

### Deferred from PR #90

**Streaming API (`compress_token`, `get_compressed_cache`) and binary /
npz serialization (`to_bytes`, `from_bytes`, `save`, `load`).** The
maintainer wants to design these against a real consumer rather than in
the abstract. Could revisit once a downstream caller emerges and the
API shape can be validated against that caller's needs.

**`OutlierTurboQuant.calibrate()`.** The maintainer notes that the
`OutlierTurboQuant` rotation-free path is experimentally retired per
[`docs/turboquant-plus-experiments.md`][exp]:

> Even removing 32 of 128 channels (25%), kurtosis stays at 8–50. WHT
> rotation gets it to 2.9. Heavy tails are a structural property of
> attention, not concentrated in outlier channels. WUSH paper confirmed.

Adding usability features to a class flagged as a dead path is therefore
not a priority for the upstream project. The `calibrate()` method itself
follows the standard LLM.int8() / SmoothQuant calibration pattern and
remains available on this fork's `main` for anyone who wants to
experiment with `OutlierTurboQuant` standalone.

[exp]: https://github.com/TheTom/turboquant_plus/blob/main/docs/turboquant-plus-experiments.md

**HIP/AMD NaN warning in `docs/turboquant-recommendations.md`.** The
maintainer notes that the warning's framing — attributing the NaN to
large K norms — is not consistent with the upstream finding in
[`docs/papers/asymmetric-kv-compression.md`][asym] that extreme K norms
produce tightly Gaussian post-normalization distributions and compress
*better* than middle-layer K. The empirical observation (q8_0/turbo3 on
HIP for Qwen2.5-7B → NaN) is reproducible; the proposed mechanism is
where the disagreement lies. A revised version that attributes the NaN
to a HIP-kernel-specific issue (matching the maintainer's framing) and
keeps the safe-alternative recommendation (`q8_0/turbo4`, already
validated on AMD HIP in §3.7 of the asymmetric-KV paper) would be a
cleaner re-submission once kernel triage produces a concrete root cause.

[asym]: https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/asymmetric-kv-compression.md

## What's in upstream `main` right now

Upstream `main` HEAD: `1224fef` (`docs(papers): add longctx-1m-and-triattention + TriAttention V3 §10 addendum`, 2026-05-08).

None of the items listed under "Curated into PR #91" are in `main` yet —
they sit on the open #91 branch.

## What's in this fork's `main`

This fork's `main` carries all six original commits from #90 on top of
upstream `main`. The fork is 0 commits behind upstream `main` and 6
commits ahead.

## Possible follow-ups

- Revise the HIP/AMD warning in `docs/turboquant-recommendations.md` to
  attribute the NaN to a HIP-kernel-specific cause rather than to K-norm
  magnitude, and re-propose. Empirical content stays; mechanism framing
  changes.
- Hold the streaming API and serialization commits on this fork's
  `main` until a consumer materializes, then re-propose with that
  caller's interface as motivation.
- `OutlierTurboQuant.calibrate()` is retained on this fork as-is for
  any standalone experimentation; no plan to re-propose unless the
  upstream verdict on `OutlierTurboQuant` changes.
