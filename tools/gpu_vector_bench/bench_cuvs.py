#!/usr/bin/env python3
"""Measure CAGRA build time and recall through the bridge's C ABI directly.

This bypasses the SQL layer so the build itself can be characterised in
isolation, and it is fast enough for parameter sweeps: one index build either
side, no server start.

Recall is computed against brute force with INDEPENDENT random queries. That
matters: the repository's validation runner derives its queries from base rows
(perturbed copies), which any decent graph answers trivially -- recall there is
pinned at ~1.0 and tells you nothing about graph quality.

Data sets
  --data random     i.i.d. gaussian, unit norm. Intrinsic dimension ~= dim, the
                    hardest case for NN-descent, so it is a pessimistic proxy.
  --data clustered  mixture of gaussians, low intrinsic dimension -- closer to
                    real embedding data such as SIFT.

Build knobs come from the bridge's env overrides (see the comment in
tools/hipvs_bridge/seekdb_cuvs_bridge.c):
  SK_GD, SK_IGD, SK_NITER

Usage:
  LD_LIBRARY_PATH=/path/to/bridge:/opt/hipvs/lib:/opt/rocm/lib \
    python3 bench_cuvs.py --lib /path/to/libseekdb_cuvs_bridge.so \
      --n 250000 --dim 128 --nq 100 --k 10 --data clustered
"""
import argparse
import ctypes
import os
import sys
import time

import numpy as np

F32 = ctypes.POINTER(ctypes.c_float)
U32 = ctypes.POINTER(ctypes.c_uint)


def load(path):
    lib = ctypes.CDLL(path)
    lib.seekdb_cuvs_build.restype = ctypes.c_void_p
    lib.seekdb_cuvs_build.argtypes = [F32, ctypes.c_long, ctypes.c_long]
    lib.seekdb_cuvs_search.restype = ctypes.c_int
    lib.seekdb_cuvs_search.argtypes = [ctypes.c_void_p, F32, ctypes.c_long,
                                       ctypes.c_long, U32, F32]
    lib.seekdb_cuvs_free.restype = None
    lib.seekdb_cuvs_free.argtypes = [ctypes.c_void_p]
    return lib


def gen(rng, n, dim, kind, clusters, spread):
    if kind == 'random':
        x = rng.normal(size=(n, dim)).astype(np.float32)
    else:
        centers = rng.normal(size=(clusters, dim)).astype(np.float32)
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        pick = rng.integers(0, clusters, size=n)
        x = centers[pick] + spread * rng.normal(size=(n, dim)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lib', default='/workspace/s13/bridge/libseekdb_cuvs_bridge.so')
    ap.add_argument('--n', type=int, default=250000)
    ap.add_argument('--dim', type=int, default=128)
    ap.add_argument('--nq', type=int, default=100)
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--data', choices=['random', 'clustered'], default='random')
    ap.add_argument('--clusters', type=int, default=100)
    ap.add_argument('--spread', type=float, default=0.25)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    lib = load(args.lib)
    rng = np.random.default_rng(args.seed)
    base = gen(rng, args.n, args.dim, args.data, args.clusters, args.spread)
    queries = gen(rng, args.nq, args.dim, args.data, args.clusters, args.spread)
    ptr = lambda a: a.ctypes.data_as(F32)

    t0 = time.monotonic()
    handle = lib.seekdb_cuvs_build(ptr(base), args.n, args.dim)
    build = time.monotonic() - t0
    if not handle:
        print('BUILD FAILED')
        return 1

    ids = np.zeros((args.nq, args.k), dtype=np.uint32)
    dist = np.zeros((args.nq, args.k), dtype=np.float32)
    t0 = time.monotonic()
    rc = lib.seekdb_cuvs_search(handle, ptr(queries), args.nq, args.k,
                                ids.ctypes.data_as(U32), ptr(dist))
    search = time.monotonic() - t0
    lib.seekdb_cuvs_free(handle)
    if rc != 0:
        print('SEARCH FAILED rc=%d' % rc)
        return 2

    hits = 0
    for i in range(args.nq):
        d = np.sum((base - queries[i]) ** 2, axis=1)
        truth = set(np.argsort(d)[:args.k].tolist())
        hits += len(set(ids[i].tolist()) & truth)
    recall = hits / float(args.nq * args.k)

    print('%-9s n=%-7d dim=%-4d gd=%-4s igd=%-4s niter=%-4s '
          'build=%7.3fs search=%5.3fs recall@%d=%.4f'
          % (args.data, args.n, args.dim,
             os.environ.get('SK_GD', '128(default)'),
             os.environ.get('SK_IGD', '256(default)'),
             os.environ.get('SK_NITER', '20(default)'),
             build, search, args.k, recall))
    return 0


if __name__ == '__main__':
    sys.exit(main())
