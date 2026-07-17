#!/usr/bin/env python
"""Short apples-to-apples timing test: UMA vs tblite (GFN2-xTB) optimization.

Loads one rotaxane .xyz, runs the SAME short LBFGS relaxation with each
calculator (same fmax, same step cap, same starting geometry), and prints
per-call and total wall times so you can compare throughput on this machine.

Usage:
  .venv/bin/python code/bench_uma_tblite.py [--input ...] [--steps N] [--fmax F]

Defaults: input=output_files/rot2_relaxed.xyz, steps=25, fmax=0.05 eV/A.
Both runs use the neutral closed-shell rotaxane (charge=0, multiplicity=1).
"""

import argparse
import os
import sys
import time

import numpy as np
from ase.io import read
from ase.optimize import LBFGS

VACUUM = 20.0  # A, same as optimize_uma.py


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="output_files/rot2_relaxed.xyz")
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--fmax", type=float, default=0.05)
    p.add_argument("--method", default="GFN2-xTB", help="tblite method")
    p.add_argument("--engine", default="both", choices=["both", "uma", "tblite"],
                   help="run only one engine per process to avoid the "
                        "torch/tblite libomp segfault (exit 139). 'both' runs "
                        "each in its own subprocess via --driver.")
    p.add_argument("--driver", action="store_true",
                   help="internal: run both engines in separate subprocesses "
                        "and print a combined summary.")
    return p.parse_args()


def max_force(atoms):
    return float(np.linalg.norm(atoms.get_forces(), axis=1).max())


def time_single_point(atoms, label):
    """Time a single energy+force eval (first call)."""
    t0 = time.perf_counter()
    e = atoms.get_potential_energy()
    f = max_force(atoms)
    dt = time.perf_counter() - t0
    print(f"  [{label}] single point: E={e:.4f} eV  max|F|={f:.4f} eV/A  "
          f"wall={dt:.3f} s")
    return dt


def run_opt(atoms, label, steps, fmax):
    """Run LBFGS, return (n_steps, final E, final max|F|, total wall, per-step)."""
    t0 = time.perf_counter()
    opt = LBFGS(atoms, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    dt = time.perf_counter() - t0
    n = opt.get_number_of_steps()
    e = atoms.get_potential_energy()
    f = max_force(atoms)
    per = dt / max(n, 1)
    print(f"  [{label}] optimize:   E={e:.4f} eV  max|F|={f:.4f} eV/A  "
          f"steps={n}  wall={dt:.2f} s  ({per:.3f} s/step)")
    return n, e, f, dt, per


def make_uma_calc():
    import torch
    from fairchem.core import pretrained_mlip, FAIRChemCalculator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"UMA device: {device}")
    predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device=device)
    return FAIRChemCalculator(predictor, task_name="omol"), device


def make_tblite_calc(method):
    from tblite.ase import TBLite
    return TBLite(method=method, charge=0, multiplicity=1, verbosity=0)


def fresh_atoms(path):
    a = read(path)
    a.set_pbc(False)
    a.center(vacuum=VACUUM)
    a.info["charge"] = 0
    a.info["spin"] = 1
    return a


def run_engine(engine, args):
    """Run a single engine in this process. Returns a result dict via stdout."""
    if engine == "uma":
        if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
            sys.exit("No HF_TOKEN set (UMA needs it; tblite doesn't).")
        print("=== UMA (uma-s-1p1, omol) ===")
        calc, _ = make_uma_calc()
        a = fresh_atoms(args.input)
        a.calc = calc
        time_single_point(a, "UMA")
        run_opt(a, "UMA", args.steps, args.fmax)
    else:
        print(f"=== tblite ({args.method}) ===")
        a = fresh_atoms(args.input)
        a.calc = make_tblite_calc(args.method)
        time_single_point(a, "tblite")
        run_opt(a, "tblite", args.steps, args.fmax)


def main():
    args = parse_args()
    print(f"input: {args.input}  steps={args.steps}  fmax={args.fmax} eV/A  engine={args.engine}\n")

    if args.driver or args.engine == "both":
        # Run each engine in its OWN subprocess: torch (UMA) and tblite each
        # bundle a copy of libomp, and using both in one process segfaults
        # (exit 139) even with KMP_DUPLICATE_LIB_OK=TRUE. Isolate them.
        import subprocess, re, json
        results = {}
        for engine in ("tblite", "uma"):
            cmd = [sys.executable, __file__, "--engine", engine,
                   "--input", args.input, "--steps", str(args.steps),
                   "--fmax", str(args.fmax), "--method", args.method]
            env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE",
                       PYTHONUNBUFFERED="1")
            log = subprocess.run(cmd, capture_output=True, text=True, env=env)
            print(f"\n--- {engine} (subprocess exit={log.returncode}) ---")
            print(log.stdout)
            if log.stderr.strip():
                # surface only non-trivial stderr lines
                for ln in log.stderr.splitlines():
                    if any(s in ln for s in ("Error", "error", "Traceback",
                                             "segfault", "Killed")):
                        print("STDERR:", ln)
            # parse the optimize line: "[UMA] optimize: E=.. max|F|=.. steps=N wall=.. (.. s/step)"
            m = re.search(r"\[(?:UMA|tblite)\] optimize:\s+E=([\-\d.]+) eV\s+"
                          r"max\|F\|=([\d.]+) eV/A\s+steps=(\d+)\s+"
                          r"wall=([\d.]+) s\s+\(([\d.]+) s/step\)", log.stdout)
            if m:
                results[engine] = dict(e=float(m.group(1)), f=float(m.group(2)),
                                       n=int(m.group(3)),
                                       wall=float(m.group(4)),
                                       per=float(m.group(5)))
        if "uma" in results and "tblite" in results:
            u, t = results["uma"], results["tblite"]
            print("=== combined summary ===")
            print(f"{'calc':<8} {'steps':>6} {'wall(s)':>9} {'s/step':>9} "
                  f"{'finalE(eV)':>14} {'maxF':>8}")
            print(f"{'UMA':<8} {u['n']:>6} {u['wall']:>9.2f} {u['per']:>9.3f} "
                  f"{u['e']:>14.4f} {u['f']:>8.4f}")
            print(f"{'tblite':<8} {t['n']:>6} {t['wall']:>9.2f} {t['per']:>9.3f} "
                  f"{t['e']:>14.4f} {t['f']:>8.4f}")
            if t["wall"] > 0 and u["wall"] > 0:
                print(f"\nspeedup: tblite {u['wall']/t['wall']:.2f}x faster overall "
                      f"(per-step {u['per']/t['per']:.2f}x)")
        return

    run_engine(args.engine, args)


if __name__ == "__main__":
    main()