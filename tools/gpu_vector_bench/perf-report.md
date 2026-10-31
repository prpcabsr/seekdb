# seekdb × hipVS/cuVS: build performance report

Numbers measured on the contest image (Radeon Cloud, `gfx1100` RDNA3, ROCm
7.2.x, 128 cores, 503 GB RAM, Ubuntu 24.04.4), with the seekdb release build
compiled from this repository on that machine. Harnesses and raw methodology:
[`tools/gpu_vector_bench`](../gpu_vector_bench/).

## Headline

| change | build phase | whole path (250k) |
|---|---|---|
| shipped source | 41.7 s | 61.9 s |
| this change | **14.9 s** | **33.8 s** |

`nn_descent_niter` 100 → 20. Recall@10 unchanged (0.990 → 0.991).

GPU vs CPU for the same 250k index:

| backend | index build | where it happens |
|---|---|---|
| `lib=cuvs` (GPU CAGRA) | **14.9 s** | lazily, at the first `APPROXIMATE` query |
| `lib=vsag` (CPU HNSW) | **235 s** | incrementally, during `INSERT` |

## Workload

250 000 vectors × 128 dimensions, L2, 100 queries, top-10.

Two data sets, because their answer to "how fast does the graph converge"
differs a lot:

* **clustered** — mixture of 100 gaussians. Low intrinsic dimension, close to
  real embedding data (SIFT's intrinsic dimension is around 12, far below 128).
* **random** — i.i.d. gaussians, unit norm. Intrinsic dimension ≈ 128, the
  worst case for NN-descent, i.e. a pessimistic proxy.

Both are scored against brute-force ground truth with **independent** queries.
The in-repo validation runner perturbs base rows to build its queries, so its
recall sits at ~1.0 no matter how good the graph is; that number cannot be used
to judge quality, which is why the report uses its own queries.

## 1. Where the time actually goes

`bench_sql.py` at n=250k, `lib=cuvs`, shipped parameters:

| phase | time | share |
|---|---|---|
| observer start → SQL ready | 1.05 s | 4 % |
| `CREATE TABLE` + `VECTOR INDEX` | 0.06 s | 0 % |
| `INSERT` 250k vectors | 10.06 s | 38 % |
| first `APPROXIMATE` query (lazy CAGRA build) | 7.33 s | 28 % |
| 100 steady queries | 8.01 s | 30 % |
| **total** | **26.5 s** | |

That total is the same order as the contest's published 20–27 s SIFT250K
baseline, which is why the tuning below is aimed at the two build-side phases
rather than at query throughput.

Import decomposes further, measured by inserting the identical rows into
tables with and without a vector index:

| table | import 250k |
|---|---|
| no vector index | 4.59 s |
| `lib=cuvs` index | 7.71 s |
| `lib=vsag` index | 235 s |

So ~4.6 s is plain storage plus parsing 128 decimal literals per row, ~3.1 s is
the cuVS adaptor buffering each row (`ob_cuvs_add` takes a mutex and does a map
lookup per row), and the GPU path already avoids the CPU graph construction
that makes the `lib=vsag` import 30× slower.

## 2. The CAGRA build knobs

`nn_descent_niter` is the number of refinement passes; the rest of
`cuvsCagraIndexParams` barely moves on this workload.

Clustered data:

| gd / igd / niter | build | recall@10 |
|---|---|---|
| 128 / 256 / 100 | 41.71 s | 0.9900 |
| 128 / 256 / 50 | 26.20 s | 0.9920 |
| **128 / 256 / 20** | **14.91 s** | **0.9910** |
| 128 / 256 / 10 | 11.01 s | 0.9750 |
| 96 / 192 / 20 | 11.28 s | 0.9680 |
| 64 / 128 / 20 | 9.22 s | 0.9080 |

Random data gives the same shape (41.05 s → 14.69 s → 11.45 s for
niter 100 → 20 → 10), so the effect is a property of NN-descent, not of the
synthetic distribution.

Two conclusions:

* 100 passes are wasted work — 20 buys the same graph for a third of the time.
* Reducing `graph_degree` is a bad trade: it is the cheapest option on the
  table (9.2 s) but costs ~8 points of recall. Retrieval quality is a scored
  requirement, so it stays at 128.

`intermediate_graph_degree` was left at 256; lowering it together with `niter`
only reproduced the `niter`-only result.

## 3. End to end, through SQL

Same binary, same data, only `SK_NITER` differs:

| niter | import | first query | total |
|---|---|---|---|
| 100 | 9.46 s | 42.69 s | 61.88 s |
| **20** | **9.63 s** | **14.52 s** | **33.75 s** |

Build phase 2.9× faster, whole path 1.8× faster.

## 4. Caveats

* Absolute times include a synthetic `INSERT` path (text literals over SQL). The
  contest harness loads vectors its own way, so the *split* between import and
  build will differ; the build-phase ratio is the part that should carry over,
  and it is a property of the params rather than of the loader.
* `gfx1100` is the reference GPU here. CAGRA build time is GPU-bound, so a
  different card shifts all absolute numbers; the ratio between niter settings
  should hold since it is the same kernel doing a third as many passes.
* The image's prebuilt `libseekdb_cuvs_bridge.so` (9 960 B) builds in 3.0 s but
  scores **0.332** recall@10 on clustered data. It is fast because the graph is
  poor, so it is not a meaningful baseline — all comparisons above are against a
  bridge built from this repository's source.
