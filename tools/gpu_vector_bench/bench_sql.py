#!/usr/bin/env python3
"""Phase-by-phase timing of a seekdb lib=cuvs vector index through SQL.

Measures the four places the contest's 20-27 s SIFT250K baseline can hide:
    T0  observer start        -> SQL ready
    T1  DDL                   -> CREATE TABLE with vector index lib=cuvs
    T2  import                -> INSERT n rows of 128-d vectors
    T3  first APPROXIMATE q   -> includes the lazy CAGRA build
    T4  steady queries        -> 100 queries, QPS
plus recall@10 against brute force, and the OB_VSAG trace marker counts so we
can tell whether the GPU path actually served the query.

usage: bench.py --n 10000 --port 2991 --base-dir /workspace/run/b1 \
                --observer <path> --bridge <path> [--reuse]
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import pathlib

import numpy as np


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)


def kill_observers(bd):
    """Kill seekdb observers rooted at bd, without ever matching this process.

    `pkill -f <base-dir>` is unusable here: bench.py's own command line carries
    --base-dir, so the pattern matches the benchmark itself and SIGTERMs it
    (exit 143). Walk /proc and additionally require the observer binary.
    """
    me = {os.getpid(), os.getppid()}
    needle = str(bd)
    killed = []
    for d in pathlib.Path('/proc').iterdir():
        if not d.name.isdigit():
            continue
        pid = int(d.name)
        if pid in me:
            continue
        try:
            cmd = (d / 'cmdline').read_bytes().replace(b'\0', b' ').decode('utf-8', 'replace')
        except OSError:
            continue
        if 'observer/seekdb' in cmd and needle in cmd:
            try:
                os.kill(pid, signal.SIGTERM)
                killed.append(pid)
            except OSError:
                pass
    return killed


def mysql(port, sql=None, infile=None, timeout=1800):
    base = ["mysql", "--protocol=TCP", "-h127.0.0.1", f"-P{port}", "-uroot",
            "-N", "-B"]
    if infile:
        return sh(" ".join(base) + f" < {infile}", timeout=timeout)
    return sh(" ".join(base) + f' -e "{sql}"', timeout=timeout)


def wait_ready(port, pid_path, timeout=240):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        r = mysql(port, "select 1", timeout=10)
        if r.returncode == 0:
            return time.monotonic() - t0
        time.sleep(0.5)
    return None


def vec_literal(v):
    return "[" + ",".join(f"{x:.7f}" for x in v) + "]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--nq", type=int, default=100)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--port", type=int, default=2991)
    ap.add_argument("--base-dir", default="/workspace/run/bench")
    ap.add_argument("--observer", required=True)
    ap.add_argument("--bridge", required=True)
    ap.add_argument("--lib", default="cuvs", choices=["cuvs", "vsag"])
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--efc", type=int, default=200)
    ap.add_argument("--efs", type=int, default=200)
    ap.add_argument("--reuse-observer", action="store_true",
                    help="do not start one, assume --port already serves")
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--out", default="/workspace/run/bench.json")
    args = ap.parse_args()

    bd = pathlib.Path(args.base_dir)
    ev = pathlib.Path("/workspace/run/bench-data")
    ev.mkdir(parents=True, exist_ok=True)
    res = {"n": args.n, "dim": args.dim, "lib": args.lib, "phases": {}}

    # ---- data ------------------------------------------------------------
    rng = np.random.default_rng(1314)
    base = rng.normal(size=(args.n, args.dim)).astype(np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    probe_rows = np.linspace(0, args.n - 1, args.nq, dtype=np.int64)
    queries = base[probe_rows] + rng.normal(scale=0.002,
                                            size=(args.nq, args.dim)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    ins = ev / f"insert-{args.n}.sql"
    if not ins.exists():
        with ins.open("w") as f:
            f.write("use bench;\n")
            for s in range(0, args.n, args.chunk):
                rows = ["(%d,'%s')" % (i, vec_literal(base[i]))
                        for i in range(s, min(s + args.chunk, args.n))]
                f.write("insert into t values " + ",".join(rows) + ";\n")

    q100 = ev / "q100.sql"
    if not q100.exists():
        with q100.open("w") as f:
            f.write("use bench;\n")
            for i in range(100):
                f.write("select group_concat(id order by id) from "
                        f"(select id from t order by l2_distance(v,'{vec_literal(queries[i % args.nq])}') "
                        f"APPROXIMATE limit {args.k}) x;\n")

    # ---- T0 observer -----------------------------------------------------
    proc = None
    if not args.reuse_observer:
        kill_observers(bd)
        # A stale observer on the same port silently invalidates the whole run:
        # wait until the port really stops answering before starting ours.
        for _ in range(40):
            if mysql(args.port, "select 1", timeout=5).returncode != 0:
                break
            sh(f"fuser -k {args.port}/tcp 2>/dev/null")
            time.sleep(1)
        r = mysql(args.port, "select 1", timeout=5)
        if r.returncode == 0:
            print(f"[bench] port {args.port} still serving another observer; aborting",
                  file=sys.stderr)
            return 3
        time.sleep(1)
        shutil.rmtree(bd, ignore_errors=True)
        bd.mkdir(parents=True, exist_ok=True)
        trace = "/workspace/run/bench-trace.log"
        pathlib.Path(trace).write_text("")
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = (str(pathlib.Path(args.bridge).parent)
                                  + ":/opt/hipvs/lib:/opt/rocm/lib")
        env["OB_VSAG_TRACE"] = "1"
        env["OB_VSAG_TRACE_FILE"] = trace
        cmd = [args.observer, "--nodaemon", "--port", str(args.port),
               "--data-dir", str(bd / "store"), "--base-dir", str(bd),
               # exactly the parameters the image's own start_observer.sh uses;
               # anything else risks OB_LOG_OUTOF_DISK_SPACE at bootstrap
               # Smaller datafiles than the image default (20G/40G/10G): each
               # observer preallocates them and /workspace is only ~98G, so two
               # runs in a row would hit OB_SERVER_OUTOF_DISK_SPACE. 128-d x
               # 250k vectors need well under 1G, so this changes nothing about
               # the phases we time (bootstrap is measured separately as T0).
               "--parameter", "memory_limit=8G",
               "--parameter", "datafile_size=6G",
               "--parameter", "datafile_maxsize=10G",
               "--parameter", "log_disk_size=3G",
               "--parameter", "local_ip=127.0.0.1"]
        log = open("/workspace/run/bench-observer.log", "w")
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        ready = wait_ready(args.port, None)
        if ready is None or proc.poll() is not None:
            print("[bench] observer failed to start (alive=%s)" % (proc.poll() is None),
                  file=sys.stderr)
            print(sh("tail -25 /workspace/run/bench-observer.log").stdout)
            return 1
        res["phases"]["t0_observer_ready_s"] = round(ready, 2)

    mysql(args.port, "alter system set ob_vector_memory_limit_percentage=30")
    mysql(args.port, "create database if not exists bench")
    mysql(args.port, "drop table if exists bench.t")

    # ---- T1 DDL ----------------------------------------------------------
    t = time.monotonic()
    r = mysql(args.port,
              f"use bench; create table t(id int primary key, v vector({args.dim}), "
              f"vector index iv(v) with (distance=l2,type=hnsw,lib={args.lib},"
              f"m={args.m},ef_construction={args.efc},ef_search={args.efs}))")
    if r.returncode != 0:
        print("[bench] DDL failed:\n" + r.stderr, file=sys.stderr)
        return 2
    res["phases"]["t1_ddl_s"] = round(time.monotonic() - t, 3)

    # ---- T2 import -------------------------------------------------------
    t = time.monotonic()
    r = mysql(args.port, infile=ins)
    res["phases"]["t2_import_s"] = round(time.monotonic() - t, 3)
    r = mysql(args.port, "use bench; select count(*) from t")
    res["rows_loaded"] = r.stdout.strip()

    # ---- T3 first approximate query --------------------------------------
    q0 = vec_literal(queries[0])
    t = time.monotonic()
    r = mysql(args.port,
              f"use bench; select group_concat(id order by id) from "
              f"(select id from t order by l2_distance(v,'{q0}') APPROXIMATE limit {args.k}) x")
    res["phases"]["t3_first_approx_s"] = round(time.monotonic() - t, 3)
    got = [int(x) for x in (r.stdout.strip() or "0").split(",") if x.strip().isdigit()] \
        if r.stdout.strip() else []

    # recall vs brute force for the same query
    d = np.sum((base - queries[0]) ** 2, axis=1)
    truth = set(np.argsort(d)[:args.k].tolist())
    res["recall_at_k_first_query"] = round(len(set(got) & truth) / max(len(truth), 1), 4)

    # ---- T4 steady -------------------------------------------------------
    t = time.monotonic()
    mysql(args.port, infile=q100)
    el = time.monotonic() - t
    res["phases"]["t4_100_queries_s"] = round(el, 3)
    res["phases"]["t4_qps"] = round(100.0 / el, 1) if el > 0 else None

    # ---- trace -----------------------------------------------------------
    trace = pathlib.Path("/workspace/run/bench-trace.log")
    counts = {}
    if trace.exists():
        for line in trace.read_text().splitlines():
            k = line.split()[0] if line.split() else ""
            if k:
                counts[k] = counts.get(k, 0) + 1
    res["trace"] = counts

    res["total_s"] = round(sum(v for k, v in res["phases"].items()
                               if k.endswith("_s") and v), 2)
    pathlib.Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))

    if proc:
        kill_observers(bd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
