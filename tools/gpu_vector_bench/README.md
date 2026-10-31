# GPU vector index benchmark

Reproducible measurements for the hipVS/cuVS CAGRA backend: what the index
build costs, what it buys in recall, and where the time goes in the whole
"load vectors, make a queryable index" path.

Two harnesses, because they answer different questions.

## `bench_cuvs.py` -- the build itself

Calls `seekdb_cuvs_build` / `seekdb_cuvs_search` through the bridge's C ABI with
ctypes. No server, no SQL, one index build per run, so a parameter sweep is
seconds per point instead of minutes.

Recall is scored against brute force with **independent random queries**. The
repository's `docs/gpu-vector-index-hipvs-cuvs/validation/run_gpu_smoke.sh`
derives its queries from base rows (perturbed copies), which any reasonable
graph answers trivially -- recall there sits at ~1.0 regardless of graph
quality and cannot be used for tuning.

`--data random` uses i.i.d. gaussians (intrinsic dimension ~= dim, the worst
case for NN-descent). `--data clustered` uses a gaussian mixture, which is much
closer to real embedding data such as SIFT. Both are reported because a tuning
result that only holds on one of them is not trustworthy; the conclusion below
holds on both.

```bash
LD_LIBRARY_PATH=/path/to/bridge:/opt/hipvs/lib:/opt/rocm/lib \
  python3 bench_cuvs.py --lib /path/to/libseekdb_cuvs_bridge.so \
    --n 250000 --dim 128 --data clustered
```

## `bench_sql.py` -- the whole path

Starts an observer, then times the phases the contest baseline of 20-27 s for
SIFT250K is made of:

| phase | what it measures |
|---|---|
| `t0_observer_ready` | process start to SQL-ready |
| `t1_ddl` | `CREATE TABLE` with `VECTOR INDEX ... lib=cuvs` |
| `t2_import` | `INSERT` of n vectors, text literals, in 250-row statements |
| `t3_first_approx` | the first `APPROXIMATE` query -- this triggers the lazy CAGRA build |
| `t4_100_queries` | 100 more queries, for QPS |

It also reads back the `OB_VSAG_TRACE` markers (`cuvs_build`, `cuvs_serve`,
`add_dense`, ...), so a run proves whether the GPU path actually served rather
than silently falling back.

```bash
LD_LIBRARY_PATH=/path/to/bridge:/opt/hipvs/lib:/opt/rocm/lib \
  OB_VSAG_TRACE=1 OB_VSAG_TRACE_FILE=/tmp/trace.log \
  python3 bench_sql.py --n 250000 --port 2996 \
    --observer /workspace/seekdb/build_release/src/observer/seekdb \
    --bridge /path/to/libseekdb_cuvs_bridge.so --out /tmp/bench.json
```

Notes from running it a lot:

* Each observer preallocates `datafile_size`. Two runs in a row on a 98 GB
  volume will die with `OB_SERVER_OUTOF_DISK_SPACE` in `create_ls`; the harness
  uses 6G/10G/3G for datafile/maxsize/log, which is ample for 250k x 128.
* Every run wipes its `--base-dir`, so always point it at a scratch path.
* A stale observer on the same port silently invalidates the run (it shows up
  as `t0` of ~0.01 s). The harness waits for the port to stop answering and
  aborts rather than measuring someone else's server.

## What the measurements say

Build parameters are in `tools/hipvs_bridge/seekdb_cuvs_bridge.c`. Only
`nn_descent_niter` is worth moving.

250k x 128, gfx1100, clustered data, independent queries:

| gd / igd / niter | build | recall@10 |
|---|---|---|
| 128 / 256 / 100 (original) | 41.71 s | 0.9900 |
| 128 / 256 / 50 | 26.20 s | 0.9920 |
| **128 / 256 / 20** | **14.91 s** | **0.9910** |
| 128 / 256 / 10 | 11.01 s | 0.9750 |

The same sweep on i.i.d. random data lands within a few percent, so the effect
is not an artifact of the synthetic distribution.

Lowering `graph_degree` is much less attractive: 64 / 128 / 20 builds in 9.22 s
but recall drops to 0.908. Since the contest gates on retrieval quality, that
trade is not worth it.

End-to-end through SQL at 250k, changing only `nn_descent_niter`:

| niter | import | first query (lazy build) | total |
|---|---|---|---|
| 100 | 9.5 s | 42.7 s | 61.9 s |
| **20** | **9.6 s** | **14.5 s** | **33.8 s** |

### Where the rest of the time goes

* **Import** (~9.5 s for 250k) splits into ~4.6 s of plain storage and vector
  literal parsing, plus ~3.1 s inside the cuVS adaptor's per-row buffering
  (`ob_cuvs_add` takes a mutex and does a map lookup per row; see
  `src/oblib/lib/vector/ob_vsag_adaptor.cpp`).
* A `lib=vsag` index on the same data takes **235 s** to import, because VSAG
  builds its HNSW graph row by row. The `lib=cuvs` path skips that entirely --
  it only buffers the vectors and builds on the GPU at first query.
* The bridge that ships in the Radeon Cloud image builds in 3.0 s but scores
  0.332 recall@10 on clustered data, i.e. it is fast because the graph is bad.
  Measuring against it would be meaningless; compare against the source build.
