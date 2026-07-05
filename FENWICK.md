# Fenwick-tree sampling (`-AMS 1`)

Optional **O(log N)** weighted BH sampling for Rapster, added on the `fenwick` branch.
`-AMS 0` (exact, default) is byte-for-byte the original; `-AMS 1` swaps the per-event
`np.random.choice(mBH, p=weights)` draws for a Fenwick tree.

## What changed

Rapster's per-event BH selection drew from `np.random.choice(mBH, p=weights)`, which
**rebuilds the full O(N) cumulative distribution on every draw**. In a dense cluster's
first timestep, k ≈ N_BH/2 draws each cost O(N) → **O(N²) per step** — the wall that made
massive/dense clusters unrunnable.

A **Fenwick tree (binary indexed tree)** does a weighted draw in **O(log N)** and a removal
in **O(log N)** → **O(k log N)** per step. It is exposed behind a new flag:

**`-AMS` / `--approx_mBH_sampling`**: `0` = exact (`np.random.choice`, the original, default);
`1` = Fenwick sampling.

All new code is gated behind `-AMS 1`; the `-AMS 0` path is inert and byte-identical to HEAD.

Files touched:

- **`rapster/auxiliary.py`** (new, © Tousif Islam) — `FenwickTree` class + `fenwick_sampler()`
  drop-in for `np.random.choice`. The build is a **vectorized log₂(N)-pass numpy** construction,
  bit-identical to the naive loop (**10.3 s → 1.09 s at N=33M, ~9×** — the pure-Python build
  loop had been cancelling the draw win).
- **`two_body_capture.py`** — `m^(2/7)` proposal tree built once per step; distinct pairs via
  `sample_indices(2, replace=False)` + rejection on `(m1+m2)^(10/7)`; removals O(log N); the
  BH-array `np.delete`s deferred to one end-of-step compaction (`consumed`).
- **`three_body_binary.py`** — same pattern with a **uniform** tree (the 3bb draw is uniform with
  rejection on the interaction term); only the binary pair is consumed, the catalyst stays.
- **`tidal_disruptions.py`** — TDE partner draw (`p = (m_star+mBH)·mBH^(1/3)`); tree built once per
  call, chosen BH `set_weight` in O(log N) after it accretes, `mBH.sum()` maintained incrementally
  so the per-event `np.mean(mBH)` stops being O(N). **Gated on `k_tde ≥ 8`** so diffuse clusters
  firing 1–2 micro-TDEs/step don't pay a per-step O(N) build.
- **`binary_evolution.py`** (the #1 sampling site) — BBH-BH single draw via **rejection off a
  reusable per-step proposal tree** on the `mBH^(3/2)` factor; identity-based lazy rebuild;
  **warmup = 2** consecutive same-`mBH` draws before switching to the tree (keeps
  singleton-interaction binaries fully exact); cached fractional `mBH` powers shared by the exact
  and tree paths.
- **`run_cluster.py`** — `-AMS` plumbing + help text.

Deliberately left **exact** (documented in-code): `exchanges.py` ex1/ex2 (k always ≤ 6–7, < 2% of
cost) and `binary_evolution:272` / `cluster_evolution:828` (drawn over the small binary population).

## Correctness — distribution-faithful, not bit-identical

A different (valid) draw sequence → not byte-reproducible per seed, but the *distribution* is
unbiased. From `fenwick_sampling_test.ipynb`:

- **Static (with replacement):** KS p = 0.53, JSD = 3.4e-5 bits — same order as choice-vs-theory
  (identical up to sampling noise).
- **Without replacement (the real use case):** KS p = 0.22, JSD = 2.4e-4 bits.
- **JSD stays 9e-5 – 1.3e-4 bits across N = 1e5 → 1e8** — distributions stay identical at every scale.
- **Multi-seed ensemble (E1):** per-seed counts overlap the exact seed-to-seed band (mergers exact
  2703 ± 54 vs Fenwick 2710 ± 41); pooled merger-mass KS p = 0.84.
- `-AMS 0` verified **byte-identical** to committed HEAD (all output files) → default path inert.
- Dense clusters are chaotic (both samplers occasionally land on a collapsed branch) →
  **validate dense clusters by ensembles, never single seeds.**

## Timing

**Per-step sampler scaling** (notebook, fitted): `np.random.choice ~ N^2.01`, `Fenwick ~ N^1.09`.
Per-draw at N = 1e7: choice **114 ms** vs Fenwick **5.7 µs**; build 271 ms once per step.

**End-to-end wall time** (env `popsynth2`):

| Cluster | exact `-AMS 0` | Fenwick `-AMS 1` | note |
|---|---|---|---|
| **N=1e7, rh=0.1** (dense, seed 90018000) | **never completes** (hangs in first step) | **~880 s** | Fenwick makes it *runnable at all* |
| N=1e7, rh=1 (diffuse) | 184 s | 104 s | ~1.8× |
| Mcl=2e7, N=33M, rh=1 | 1553 s | 972 s | 1.60× (−37%) |
| Dense N=3M ensemble (5 seeds) | 40.4 ± 8.2 s | 28.9 ± 0.4 s | **1.40×**, far more consistent |
| E1 (N=5M, TDE-bound) | ~51 s | 37 s | not sampling-bound; warmup keeps it safe |

**Commit-walk on the dense N=1e7 / rh=0.1 cluster** (the headline result):

| step | change | commit | wall | cumulative |
|---|---|---|---|---|
| 0 | old Rapster (cosmology bug) | upstream | hung / unrunnable | — |
| 1 | + cosmology fix (exact) | `redshift_interp` | ~1 hr (est.) | 1.0× ref |
| 2 | **+ Fenwick sampling** | `9a0d0f4` | **1327 s** | ~2.7× |
| 3 | + list-accumulate output | `74a7dc0` | 893 s | ~4.0× |
| 4 | + np.min/max → builtin | `cb08294` | 882 s | ~4.1× |
| 5 | + np.sqrt → math.sqrt | `f95955c` | 885 s | ~4.1× |

**Bottom line:** on the diffuse large clusters that dominate catalog wall time, Fenwick is ~1.6×;
on the dense corner it is the difference between **unrunnable and ~15 min**. Projected per-step
speedup at N = 1e8 is ~10⁵× (choice O(N²) vs Fenwick O(N log N)).

## Usage

```bash
python -m rapster.run_cluster ... -AMS 1     # Fenwick sampling
python -m rapster.run_cluster ... -AMS 0     # exact (default)
```

Recommended for large / dense clusters (Mcl ≳ 1e7, or low rh) and for ensemble catalog
statistics. Because `-AMS 1` is distribution-faithful but not per-seed bit-reproducible,
validate by merger/TDE **distributions over many seeds**, not single-seed byte comparison.
