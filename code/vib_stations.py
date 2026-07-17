#!/usr/bin/env python
"""Constrained partial-Hessian vibrational free energies at scan stations.

The displace_wheel chain scan gives *potential-energy* shuttle barriers (well vs
saddle) plus Eyring TST rate estimates. This tool adds the vibrational
free-energy correction to turn those into *free-energy* (Gibbs/Helmholtz)
barriers -- but does it correctly for a *constrained* station, not a free gas
molecule.

Why constrained, and why a *partial* Hessian:
  A scan station lives on the wheel-rigid / rod-endpoint-anchored constrained
  surface. A *free* relax of a station escapes that surface (~25 kcal lower,
  d-independent strain release -- the wheel stays put, the rod relaxes), so it
  is the WRONG gauge for a station free energy. Instead we hold the reaction
  coordinate d fixed BY CONSTRUCTION by reusing the scan's EXACT constraint set:
  FixAtoms ALL wheel atoms (rigid wheel) + the two rod-tip anchors. This freezes
  the wheel centroid and the rod rigid body, so d cannot drift. (Pinning only a
  few wheel atoms does NOT work for the flexible 56-atom crown-ether wheel -- the
  free wheel atoms deform and shift the centroid, giving a spurious imaginary
  mode and a ~0.5 A d drift; --no-rigid-wheel keeps that mode for testing.)
  Tight-relax the free atoms (the rod's internal flex) on that constrained
  surface, and take a PARTIAL Hessian of the free atoms only. ASE
  ``Vibrations`` auto-detects ``FixAtoms`` and builds the Hessian over
  non-fixed atoms, so the reaction coordinate is removed:
  - at a WELL, the partial Hessian has only real modes (a constrained minimum);
  - at a SADDLE, the one unstable direction IS the frozen reaction coordinate,
    so it too is removed -> the partial Hessian is positive-definite there as
    well. No spurious imaginary mode, no ln(negative) -> NaN.
  The constraint strain is ~constant across d, so it CANCELS in
  DeltaG = G_saddle - G_well.

Thermo: ``ase.thermochemistry.HarmonicThermo`` (vib-only -- ZPE + thermal +
entropy over the real modes), NOT ``IdealGasThermo`` (which adds gas-phase
trans/rot partition functions for a free molecule; wrong for a constrained
station). F_vib(T) = Sum[1/2 h nu + kT ln(1 - e^{-h nu/kT})]; station free
energy G = E_relaxed + F_vib.

Stations are auto-identified from ``<stem>_scan[_engine].csv`` (global min,
local-min wells separated from it by > BARRIER_MIN, and the saddle =
highest-energy station on the path between each well and the global min), or
given explicitly via ``--stations d1,d2,...``. Geometries are read from
``output_files/<stem>_stations[_engine]/d{d:+.2f}.xyz`` produced by
``displace_wheel.py --engine <eng> --dump-stations`` (a re-run with that flag
is required -- pos_by_d is in-memory only). ``--engine`` MUST match the scan
engine -- a GFN2 vib run (``--engine tblite``) pairs with a tblite scan and
reads the ``_tblite``-tagged CSV + stations dir; UMA (default) pairs with an
untagged UMA scan.

Run from the project root:
    .venv/bin/python code/vib_stations.py --input output_files/rot2_displaced.xyz
    .venv/bin/python code/vib_stations.py --engine tblite --input output_files/rot2_displaced_tblite.xyz

Timing: ~13-15 min per station single-proc on 114 atoms with UMA (central-
difference Hessian, ~2*(3N-3*fix) force evals at ~1.2 s each); tblite (GFN2-xTB)
is ~5x faster per force eval, so a full Hessian becomes tractable. ~7 stations
-> ~2 h (UMA) / ~25 min (tblite).
``--n-procs K`` runs K ASE Vibrations workers sharing the scratch cache dir
(file-based parallelism) for an ~Kx speedup (memory permitting).
"""

import argparse
import os
import sys
import time

import numpy as np
from ase.io import read
from ase.constraints import FixAtoms
from ase.optimize import LBFGS
from ase.vibrations import Vibrations
from ase.thermochemistry import HarmonicThermo
# NOTE: fairchem/torch is NOT imported at module top. The force source is
# chosen at runtime with --engine; the engine's libs load only inside the
# branch that uses them (see optimize_uma.make_calculator). A top-level
# fairchem import would pull torch into every run including tblite (GFN2)
# runs, and torch + tblite co-load segfaults (exit 139; each bundles its own
# libomp). Keeping it lazy means a GFN2 vib run never loads torch at all.

from optimize_uma import (
    MODEL, TASK, VACUUM, get_hf_token, make_calculator, pick_device,
    read_charge_spin, write_plain_xyz,
)
from rotaxane_paths import (
    resolve_stem, default_smiles, out_path, OUTPUT_DIR,
    ENGINES, DEFAULT_ENGINE, engine_tag,
)
from displace_wheel import (
    fragment_counts, rod_axis, EV_TO_KCAL_MOL, BARRIER_MIN, SMOOTH_PTS,
)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=None,
                   help="any <stem>_*.xyz -- only a STEM source (resolve_stem "
                        "strips the role suffix); its geometry is NOT read. The "
                        "station geometries come from output_files/<stem>_stations/. "
                        "Defaults to the rot_smiles stem if neither this nor "
                        "--smiles is given.")
    p.add_argument("--smiles", default=None,
                   help="override the SMILES file (default <stem>.txt).")
    p.add_argument("--temperature", type=float, default=300.0, help="K (default 300)")
    p.add_argument("--relax-fmax", type=float, default=0.005,
                   help="eV/A: tight relax of the free atoms on the wheel-position-"
                        "constrained surface before the Hessian. 0 skips (only safe "
                        "on an already-tight constrained minimum).")
    p.add_argument("--rigid-wheel", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="FixAtoms ALL wheel atoms + the 2 rod-tip anchors -- the "
                        "scan's exact constrained surface (DEFAULT). Freezes d by "
                        "construction (0 imaginary modes, E_relaxed ~ scan energy, "
                        "DeltaE matches the scan barriers); F_vib is the rod-internal "
                        "vib free energy. Use --no-rigid-wheel to instead pin only "
                        "--fix-wheel spread wheel atoms (frees the rest of the wheel) "
                        "-- WARNING: for a flexible ring this does NOT hold d (the "
                        "free wheel atoms deform and shift the centroid -> d drift + "
                        "imaginary modes); only safe for a genuinely rigid wheel.")
    p.add_argument("--fix-wheel", type=int, default=3,
                   help="(only with --no-rigid-wheel) number of maximally-spread "
                        "non-collinear wheel atoms to pin. 3 fixes all 6 rigid-body "
                        "DOF in principle, but a flexible ring still drifts; 6 is a "
                        "robustness check. Ignored when --rigid-wheel (the default).")
    p.add_argument("--no-rod-anchors", action="store_true",
                   help="do NOT anchor the two rod-tip atoms. By default the rod "
                        "endpoints are fixed (as in the scan) so the rod cannot "
                        "slide/rotate and the reaction coordinate d is frozen BY "
                        "CONSTRUCTION -- otherwise the free rod drifts and d is "
                        "only ~held. Leave this off unless you are deliberately "
                        "testing the wheel-only constraint.")
    p.add_argument("--delta", type=float, default=0.01,
                   help="finite-difference displacement in A (ASE default 0.01)")
    p.add_argument("--no-hessian", action="store_true",
                   help="skip the partial Hessian + thermo -- just constrained-"
                        "relax each station (deterministic from the same scan seed "
                        "-> reproduces the relaxed geometry the free energies were "
                        "computed on) and save it. ~700 s/station (relax only) vs "
                        "~1200 s (relax + Hessian). Use when you already have the "
                        "free-energy CSV and only want the relaxed structures for "
                        "viewing. E_relaxed and the d-held check are still reported; "
                        "F_vib/G/barriers are skipped.")
    p.add_argument("--stations", default="auto",
                   help="'auto' (global min + wells + their saddles from the scan "
                        "CSV) or an explicit d list like '0.5,-0.5,0.0,3.5,-3.5'")
    p.add_argument("--all-stations", action="store_true",
                   help="with --stations auto, ALSO run every Nth dumped scan "
                        "station so vib covers the whole shuttle coordinate (a "
                        "dense G(d) curve in addition to the auto "
                        "wells/saddles/global min, which are kept so the barrier "
                        "block still pairs them). --all-stations-step sets the "
                        "spacing (A, default 0.20); each target d is snapped to "
                        "the nearest dumped station geometry. Should be a multiple "
                        "of the 0.10 A scan grid so targets land on real stations. "
                        "Cost GFN2: ~N x ~65 s (~53 stations at 0.20 A over "
                        "-4.7..4.7 -> ~55 min serial).")
    p.add_argument("--all-stations-step", type=float, default=0.20,
                   help="A: spacing of the --all-stations dense curve (default "
                        "0.20 -- a multiple of the 0.10 A scan grid so targets "
                        "land on dumped stations). Snapped to the nearest dumped "
                        "scan station.")
    p.add_argument("--n-procs", type=int, default=1,
                   help="ASE Vibrations parallel workers sharing the scratch cache "
                        "(file-based). 1 = serial. >1 spawns K procs for an ~Kx "
                        "Hessian speedup (memory permitting).")
    p.add_argument("--resume", action="store_true",
                   help="Continue a previous (killed or interrupted) run: read the "
                        "existing <stem>_freeenergy[_engine].csv, skip stations "
                        "already present (matched by d to 0.01 A), and append the "
                        "rest. The CSV is written INCREMENTALLY (after every "
                        "station), so a killed run leaves a complete partial CSV "
                        "-- pass --resume on the next launch to finish it without "
                        "recomputing finished stations. Aborts if the existing CSV "
                        "contains stations absent from the current scan (a stale "
                        "CSV from a different scan): back it up and remove it, or "
                        "rerun without --resume. The per-station d*.xyz on disk "
                        "are reused; the multi-state view PDB on a partial resume "
                        "only includes newly-computed frames.")
    p.add_argument("--barrier-min", type=float, default=BARRIER_MIN,
                   help="kcal/mol: a local min counts as a well only if separated "
                        f"from the global min by > this (default {BARRIER_MIN}).")
    p.add_argument("--smooth-pts", type=int, default=SMOOTH_PTS,
                   help="point-count smoothing window for well detection (matches "
                        "displace_wheel --scan-smooth-pts).")
    p.add_argument("--save-geometries", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="save each station's vib-RELAXED geometry (the structure the "
                        "free energy was computed on) to output_files/<stem>_vibstations/"
                        " and a multi-state PDB output_files/<stem>_vibstations_view.pdb "
                        "(PyMOL mplay). Each frame is translated so its rod centroid "
                        "coincides with the global-min frame's (no rotation -> the C2 "
                        "rod's real 180-deg flip between sides is preserved), and the "
                        "MODELs are ordered by PHYSICAL wheel position along a fixed "
                        "reference axis (the global-min frame's rod PCA axis), NOT the "
                        "grid d -- the tight relax shifts each station's rod reference "
                        "so the grid order is not monotonic in the realized wheel "
                        "position. DEFAULT on; --no-save-geometries skips.")
    p.add_argument("--engine", default=DEFAULT_ENGINE, choices=ENGINES,
                   help="force source for the constrained relax + partial Hessian: "
                        "'uma' (Meta UMA MLIP, default; needs HF_TOKEN) or 'tblite' "
                        "(GFN2-xTB; no HF_TOKEN, ~5x faster/step -- a full central-"
                        "difference Hessian becomes tractable). The engine MUST "
                        "match the scan engine: it reads <stem>_scan[_engine].csv "
                        "and <stem>_stations[_engine]/ for the station set + "
                        "geometries, so a GFN2 vib run pairs with a --engine tblite "
                        "displace_wheel scan. A non-default engine tags the "
                        "freeenergy CSV + view PDB/xyz (<stem>_freeenergy_tblite.csv, "
                        "<stem>_vibstations_tblite/).")
    p.add_argument("--method", default="GFN2-xTB",
                   help="tblite method (default GFN2-xTB; also GFN1-xTB, GFN0-xTB, "
                        "CEH). Ignored for --engine uma.")
    return p.parse_args()


def read_scan_csv(path):
    """Return (ds, energies_eV) arrays from a displace_wheel scan CSV.

    Columns: displacement_A, energy_eV, energy_rel_kcal_mol, min_contact_A.
    The CSV has no convergence flag, so all stations are treated as converged
    (run the scan with --production so threading events actually converge)."""
    import csv
    ds, es = [], []
    with open(path, newline="") as fh:
        r = csv.reader(fh)
        header = next(r)
        for row in r:
            if not row or not row[0].strip():
                continue
            ds.append(float(row[0]))
            es.append(float(row[1]))
    order = np.argsort(ds)
    return np.asarray(ds)[order], np.asarray(es)[order]


# Columns of the free-energy CSV (shared by the writer + the --resume parser).
FREEENERGY_COLS = ["role", "d_A", "d_after_A", "d_shift_A", "E_relaxed_eV",
                   "Fvib_eV", "Fvib_kcal_mol", "G_eV", "G_rel_kcal_mol",
                   "n_imag", "barrier_kcal_mol"]


def _barrier_map(results, wells):
    """Pair wells with their saddles (role+side, matched on d within 0.05 A)
    and return {(role, d): dG_kcal_mol} for the CSV's barrier column. Recomputed
    from G_eV every write, so barriers fill in as soon as a well/saddle pair is
    both present in `results` (e.g. incrementally, as the second of the pair
    finishes)."""
    by_role = {}
    for r in results:
        by_role.setdefault(r["role"], []).append(r)
    barrier_for = {}
    for w in wells:
        side = w["side"]
        wr = next((r for r in by_role.get("well", [])
                   if abs(r["d"] - w["d"]) < 0.05), None)
        sr = next((r for r in by_role.get(f"saddle({side})", [])
                   if abs(r["d"] - w["saddle_d"]) < 0.05), None)
        if wr and sr and wr["G_eV"] is not None and sr["G_eV"] is not None:
            dG = (sr["G_eV"] - wr["G_eV"]) * EV_TO_KCAL_MOL
            barrier_for[("well", wr["d"])] = dG
            barrier_for[(f"saddle({side})", sr["d"])] = dG
    return barrier_for


def write_freeenergy_csv(csv_path, results, wells):
    """Atomically (re)write the free-energy CSV from `results`.

    Called after EVERY station completes (incremental checkpoint) so a kill
    never loses a finished station's row -- the file is consistent at every
    moment, with well/saddle barriers filled as their pair completes. Atomic:
    write to <path>.tmp then os.replace, so a kill mid-write cannot leave a
    torn/half-written file (fsync'd first)."""
    import csv
    barrier_for = _barrier_map(results, wells)

    def cell(v, fmt):
        return "" if v is None else fmt % v

    tmp = csv_path + ".tmp"
    with open(tmp, "w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(FREEENERGY_COLS)
        for r in results:
            b = barrier_for.get((r["role"], r["d"]), "")
            wtr.writerow([r["role"], f"{r['d']:.2f}", f"{r['d_after']:.2f}",
                          f"{r['d_shift']:+.2f}", f"{r['E_relaxed_eV']:.6f}",
                          cell(r["Fvib_eV"], "%.6f"), cell(r["Fvib_kcal"], "%.2f"),
                          cell(r["G_eV"], "%.6f"), cell(r["G_rel_kcal"], "%.2f"),
                          "" if r["n_imag"] is None else r["n_imag"],
                          f"{b:.2f}" if b != "" else ""])
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, csv_path)


def parse_freeenergy_csv(csv_path):
    """Read a free-energy CSV back into result dicts + the set of completed d
    values (rounded to 0.01 A). Used by --resume to seed `results` (so the
    end-of-run barrier pass and every incremental rewrite see the already-
    completed stations) and to know which stations to skip. The barrier column
    is NOT parsed -- it is recomputed from G_eV on write."""
    import csv
    results, done = [], set()
    with open(csv_path, newline="") as fh:
        r = csv.reader(fh)
        header = next(r, None)
        if header is None:
            return [], set()
        idx = {name: i for i, name in enumerate(header)}

        def fget(row, name):
            i = idx.get(name)
            if i is None or i >= len(row) or row[i].strip() == "":
                return None
            return float(row[i])

        for row in r:
            if not row or not row[0].strip():
                continue
            d = fget(row, "d_A")
            if d is None:
                continue
            ni = fget(row, "n_imag")
            results.append({
                "role": row[idx["role"]],
                "d": d,
                "d_after": fget(row, "d_after_A"),
                "d_shift": fget(row, "d_shift_A"),
                "E_relaxed_eV": fget(row, "E_relaxed_eV"),
                "Fvib_eV": fget(row, "Fvib_eV"),
                "Fvib_kcal": fget(row, "Fvib_kcal_mol"),
                "G_eV": fget(row, "G_eV"),
                "G_rel_kcal": fget(row, "G_rel_kcal_mol"),
                "n_imag": (None if ni is None else int(round(ni))),
            })
            done.add(round(d, 2))
    return results, done


def identify_stations(ds, energies, smooth_pts=SMOOTH_PTS, barrier_min=BARRIER_MIN):
    """Global min + wells + per-well saddle from the scan curve.

    Replicates displace_wheel.detect_wells' smoothing + local-minimum + merge +
    snap-to-raw-floor + barrier logic, but ALSO returns the saddle *station*
    (the argmax-energy station on the path between each well and the global min),
    which detect_wells discards (it only keeps the barrier height). All stations
    are assumed converged (the CSV carries no convergence flag).

    Returns (g_d, g_e, wells) where each well is a dict:
      {d, e, rel_kcal, side, barrier_kcal, saddle_d, saddle_e, saddle_rel_kcal}
    """
    ds = np.asarray(ds, dtype=float)
    energies = np.asarray(energies, dtype=float)
    n = len(energies)
    g_idx = int(np.argmin(energies))
    e_min = float(energies[g_idx])
    d_min = float(ds[g_idx])

    w = max(int(smooth_pts), 3)
    if w % 2 == 0:
        w += 1
    half = w // 2
    smooth = np.empty_like(energies)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        smooth[i] = energies[lo:hi].mean()

    candidates = []
    for i in range(2, n - 2):
        if (smooth[i] < smooth[i - 1] and smooth[i] < smooth[i + 1]
                and smooth[i] <= smooth[i - 2] and smooth[i] <= smooth[i + 2]):
            candidates.append(i)
    merged = []
    for i in candidates:
        if merged and abs(ds[i] - ds[merged[-1]]) <= 1.0:
            if energies[i] < energies[merged[-1]]:
                merged[-1] = i
        else:
            merged.append(i)

    wells = []
    for i in merged:
        if i == g_idx:
            continue
        lo_s, hi_s = max(0, i - half), min(n, i + half + 1)
        j = int(lo_s + np.argmin(energies[lo_s:hi_s]))
        if j == g_idx:
            continue
        lo, hi = (j, g_idx) if j < g_idx else (g_idx, j)
        s_idx = int(lo + np.argmax(energies[lo:hi + 1]))
        barrier_kcal = (energies[s_idx] - energies[j]) * EV_TO_KCAL_MOL
        if barrier_kcal < barrier_min:
            continue
        wells.append({
            "d": float(ds[j]),
            "e": float(energies[j]),
            "rel_kcal": float((energies[j] - e_min) * EV_TO_KCAL_MOL),
            "side": "left" if ds[j] < d_min else "right",
            "barrier_kcal": barrier_kcal,
            "saddle_d": float(ds[s_idx]),
            "saddle_e": float(energies[s_idx]),
            "saddle_rel_kcal": float((energies[s_idx] - e_min) * EV_TO_KCAL_MOL),
        })
    wells.sort(key=lambda w: (abs(w["d"] - d_min), w["rel_kcal"]))
    return d_min, e_min, wells


def pick_spread_wheel_atoms(wheel_pos, k=3):
    """Indices (into the wheel_pos block) of k maximally-spread, non-collinear
    wheel atoms. 3 non-collinear points pin a rigid body's position + all 3
    orientation DOF, freezing the reaction coordinate. Greedy: farthest from the
    centroid, then farthest from the first, then max triangle area, then (k>3)
    maximize min-distance to the chosen set."""
    c = wheel_pos.mean(axis=0)
    chosen = [int(np.argmax(((wheel_pos - c) ** 2).sum(axis=1)))]
    chosen.append(int(np.argmax(((wheel_pos - wheel_pos[chosen[0]]) ** 2).sum(axis=1))))
    while len(chosen) < k:
        best_i, best_score = None, -1.0
        p0, p1 = wheel_pos[chosen[0]], wheel_pos[chosen[1]]
        for i in range(len(wheel_pos)):
            if i in chosen:
                continue
            if len(chosen) < 3:
                # triangle area = 0.5 |(p1-p0) x (pi-p0)| (non-collinearity)
                score = np.linalg.norm(np.cross(p1 - p0, wheel_pos[i] - p0))
            else:
                # min distance to any already-chosen atom (spread)
                score = min(np.linalg.norm(wheel_pos[i] - wheel_pos[j])
                            for j in chosen)
            if score > best_score:
                best_score, best_i = score, i
        chosen.append(best_i)
    return chosen


def wheel_displacement(pos, rod_n, u):
    """d = (wheel_centroid - rod_centroid) . rod_axis (A), matching the scan."""
    rod, wheel = pos[:rod_n], pos[rod_n:]
    return float((wheel.mean(axis=0) - rod.mean(axis=0)) @ u)


def station_geometry_path(stem, d, engine=DEFAULT_ENGINE):
    """Path to a dumped station geometry. The dir is engine-tagged
    (``<stem>_stations/`` for UMA, ``<stem>_stations_tblite/`` for tblite) so the
    two engines' station geometries don't collide; vib_stations reads the one
    matching its --engine, which must match the displace_wheel scan engine."""
    return os.path.join(OUTPUT_DIR, f"{stem}_stations{engine_tag(engine)}",
                        f"d{d:+.2f}.xyz")


def list_station_ds(stem, engine=DEFAULT_ENGINE):
    """Sorted d values of every dumped station geometry in
    ``<stem>_stations[_engine]/`` (written by displace_wheel --dump-stations).
    Filenames are ``d{d:+.2f}.xyz`` -> parse the signed float after the leading
    'd' (e.g. ``d-1.40.xyz`` -> -1.40, ``d+0.00.xyz`` -> 0.0). Used by
    --all-stations to walk the whole shuttle coordinate."""
    d = os.path.join(OUTPUT_DIR, f"{stem}_stations{engine_tag(engine)}")
    out = []
    if not os.path.isdir(d):
        return out
    for name in os.listdir(d):
        if not (name.startswith("d") and name.endswith(".xyz")):
            continue
        try:
            out.append(float(name[1:-4]))
        except ValueError:
            continue
    out.sort()
    return out


def run_station(atoms, fix_idx, relax_fmax, delta, temperature, scratch, label,
                compute_vib=True):
    """Tight-relax the free atoms on the wheel-position-constrained surface,
    take the partial Hessian, and return (E_relaxed_eV, F_vib_eV, n_imag, freqs).

    Vibrations auto-skips the FixAtoms indices -> partial Hessian over the free
    atoms only (the reaction coordinate is removed). HarmonicThermo gives the
    vib-only Helmholtz free energy F_vib(T) (no trans/rot: a constrained station
    is not a free gas molecule).

    With compute_vib=False (--no-hessian) the Hessian + thermo are skipped after
    the relax; F_vib/n_imag/freqs are returned as None. The relaxed geometry (the
    structure the free energies were computed on, reproduced deterministically
    from the same scan seed) is then available for saving/viewing."""
    atoms.set_constraint(FixAtoms(indices=fix_idx))

    if relax_fmax and relax_fmax > 0:
        print(f"  [{label}] constrained relax fmax={relax_fmax} ...", flush=True)
        t0 = time.time()
        opt = LBFGS(atoms, logfile=None)
        opt.run(fmax=relax_fmax, steps=500)
        print(f"  [{label}] relax done in {time.time()-t0:.0f} s "
              f"(E={atoms.get_potential_energy():.6f} eV)", flush=True)

    E_rel = atoms.get_potential_energy()

    if not compute_vib:
        return E_rel, None, None, None

    vib = Vibrations(atoms, name=os.path.join(scratch, label), delta=delta)
    t0 = time.time()
    vib.run()
    n_force_evals = 2 * (3 * len(atoms) - 3 * len(fix_idx))
    print(f"  [{label}] partial Hessian done in {time.time()-t0:.0f} s "
          f"(~{n_force_evals} force evals)", flush=True)

    raw = vib.get_energies()
    freqs = np.where(np.abs(raw.imag) > 1e-9, -np.abs(raw.imag), raw.real)
    n_imag = int(np.sum(freqs < -1e-6))
    real_freqs = [float(f) for f in freqs if f > -1e-6]

    thermo = HarmonicThermo(vib_energies=real_freqs, potentialenergy=0.0,
                            ignore_imag_modes=True)
    F_vib = thermo.get_helmholtz_energy(temperature, verbose=False)
    return E_rel, float(F_vib), n_imag, freqs


def main():
    args = parse_args()
    # Stem comes from --input (its role suffix is stripped by resolve_stem) or,
    # if only --smiles is given, from the SMILES file's basename. With neither,
    # fall back to the project default stem. NOTE: --input's geometry is NOT
    # read -- it is only a stem source; station geometries come from
    # output_files/<stem>_stations/ (written by displace_wheel --dump-stations).
    if args.input is not None:
        stem = resolve_stem(args.input)
    elif args.smiles is not None:
        stem = os.path.splitext(os.path.basename(args.smiles))[0]
    else:
        stem = "rot_smiles"
    smiles_path = args.smiles or default_smiles(stem)

    if args.engine == "uma":
        get_hf_token()
    charge, spin = read_charge_spin(smiles_path)
    rod_n, wheel_n = fragment_counts(smiles_path)
    n_atoms = rod_n + wheel_n

    # Per-station calculator. UMA's predictor is loaded ONCE and reused (loading
    # the checkpoint per station would be N x the cost); a fresh FAIRChemCalculator
    # wraps it per station. tblite has no heavy predictor, so a fresh TBLite per
    # station via make_calculator is cheap. The engine's libs import only inside
    # the branch used -> the two never co-load (libomp segfault; see
    # optimize_uma.make_calculator). --engine must match the scan engine: the
    # station set + geometries come from the engine-tagged scan CSV + stations dir.
    if args.engine == "uma":
        from fairchem.core import pretrained_mlip, FAIRChemCalculator
        device = pick_device()
        predictor = pretrained_mlip.get_predict_unit(MODEL, device=device)

        def attach_calc(atoms):
            atoms.info["charge"] = charge
            atoms.info["spin"] = spin
            atoms.calc = FAIRChemCalculator(predictor, task_name=TASK)

        eng_banner = (f"engine=uma model={MODEL} task={TASK} device={device} "
                      f"charge={charge} spin={spin}")
    else:
        device = None

        def attach_calc(atoms):
            calc, _ = make_calculator("tblite", atoms, charge, spin,
                                      method=args.method)
            atoms.calc = calc

        eng_banner = (f"engine=tblite method={args.method} charge={charge} "
                      f"spin={spin} (no torch device)")

    scan_csv = out_path(stem, role="scan", ext="csv", engine=args.engine)
    if not os.path.exists(scan_csv):
        sys.exit(f"scan CSV not found: {scan_csv} -- run displace_wheel.py "
                 f"--engine {args.engine} first (the vib engine must match the "
                 f"scan engine)")
    ds, energies = read_scan_csv(scan_csv)
    d_min, e_min, wells = identify_stations(
        ds, energies, smooth_pts=args.smooth_pts, barrier_min=args.barrier_min)

    print(f"vib_stations: stem={stem}  {n_atoms} atoms (rod {rod_n} + wheel "
          f"{wheel_n})  {eng_banner}  T={args.temperature} K")
    print(f"vib_stations: scan CSV {scan_csv}: {len(ds)} stations, "
          f"global min d={d_min:+.2f} A  E={e_min:.6f} eV")

    # Resolve the station set (role, d).
    if args.stations != "auto":
        stations = [("explicit", float(x)) for x in args.stations.split(",")]
    else:
        # auto = global min + wells + their saddles. The barrier block below
        # pairs wells with saddles by role, so these are always included.
        stations = [("global_min", d_min)]
        for w in wells:
            stations.append(("well", w["d"]))
            stations.append((f"saddle({w['side']})", w["saddle_d"]))
        # --all-stations: ALSO run every Nth dumped scan station so vib covers the
        # whole shuttle coordinate (dense G(d) curve). Targets are n*step, each
        # snapped to the nearest dumped station geometry; step should be a
        # multiple of the 0.10 A scan grid so targets land on real stations. The
        # auto wells/saddles above are kept (de-dup below drops the 'scan' label
        # where a well/saddle already sits, so roles survive for the barrier
        # block).
        if args.all_stations:
            avail = list_station_ds(stem, engine=args.engine)
            if not avail:
                print(f"vib_stations: --all-stations but no station geometries in "
                      f"{stem}_stations{engine_tag(args.engine)}/ -- run "
                      f"displace_wheel.py --engine {args.engine} --dump-stations.",
                      flush=True)
            else:
                step = args.all_stations_step
                a_min, a_max = avail[0], avail[-1]
                n_lo = int(np.floor(a_min / step))
                n_hi = int(np.ceil(a_max / step))
                av = np.asarray(avail)
                for n in range(n_lo, n_hi + 1):
                    tgt = n * step
                    j = int(np.argmin(np.abs(av - tgt)))
                    stations.append(("scan", float(avail[j])))
    # de-dup by rounded d (well + saddle + scan can coincide at coarse grids;
    # auto roles are added first so they win over a coincident 'scan' label)
    seen = {}
    for role, d in stations:
        key = round(d, 2)
        if key not in seen:
            seen[key] = (role, d)
    stations = list(seen.values())
    # Run left -> right by d so progress is monotonic and the CSV is ordered.
    stations.sort(key=lambda rd: rd[1])
    print(f"vib_stations: {len(stations)} stations to analyse:")
    for role, d in stations:
        e = energies[np.argmin(np.abs(ds - d))]
        print(f"    {role:16s} d={d:+.2f} A  E_scan={e:.6f} eV "
              f"(rel {(e-e_min)*EV_TO_KCAL_MOL:+.2f} kcal/mol)")

    scratch = f"/tmp/vibst_{stem}_{os.getpid()}"
    if args.n_procs > 1:
        print(f"vib_stations: --n-procs {args.n_procs} requested but multi-process "
              f"Hessian parallelism is not yet implemented -- running serial. "
              f"(~13-15 min/station on 114 atoms with UMA; tblite is ~5x faster. "
              f"~{len(stations)} stations.)",
              flush=True)
    os.makedirs(scratch, exist_ok=True)

    csv_path = out_path(stem, role="freeenergy", ext="csv", engine=args.engine)
    meta_path = csv_path + ".meta"
    run_meta = {
        "stem": stem,
        "engine": args.engine,
        "all_stations_step": args.all_stations_step,
        "n_stations": len(stations),
        "scan_csv": os.path.basename(scan_csv),
    }
    results = []
    done_d = set()
    if args.resume and os.path.exists(csv_path) and not args.no_hessian:
        # Verify the on-disk CSV is from THIS run config before resuming. The
        # d-sets of different scan grids can be nested (every 0.20 step is also
        # a 0.10 step), so d-matching alone can't tell a stale 0.20 CSV from a
        # partial 0.10 run -- the .meta sidecar does.
        if not os.path.exists(meta_path):
            sys.exit(f"vib_stations: --resume: {csv_path} exists but has no "
                     f"{meta_path} sidecar -- it pre-dates the resume-safety check "
                     f"or is from a different run. Back it up and remove it (and "
                     f"the .meta), or rerun without --resume.")
        import json
        with open(meta_path) as fh:
            old = json.load(fh)
        mism = [k for k in ("stem", "engine", "all_stations_step", "n_stations",
                            "scan_csv") if old.get(k) != run_meta.get(k)]
        if mism:
            sys.exit(f"vib_stations: --resume: {meta_path} differs from the "
                     f"current run on {mism} -- {csv_path} is from a different "
                     f"run. Back it up and remove it (and the .meta), or rerun "
                     f"without --resume.")
        prev, done_d = parse_freeenergy_csv(csv_path)
        planned_d = {round(d, 2) for _, d in stations}
        stale = done_d - planned_d
        if stale:
            sys.exit(f"vib_stations: --resume: {csv_path} contains stations "
                     f"(d={sorted(stale)}) absent from the current scan's station "
                     f"set -- it looks like a CSV from a different scan. Back it "
                     f"up and remove it, or rerun without --resume.")
        if prev:
            results = prev
            write_freeenergy_csv(csv_path, results, wells)  # rewrite clean state
            print(f"vib_stations: --resume: {len(done_d)} station(s) already in "
                  f"{csv_path} -- skipping them, appending the rest. (The view "
                  f"PDB will only include newly-computed frames; the per-station "
                  f"d*.xyz on disk are reused.)", flush=True)
        else:
            print(f"vib_stations: --resume: {csv_path} parsed no rows; starting "
                  f"fresh.", flush=True)
    if not args.no_hessian:
        import json
        with open(meta_path, "w") as fh:
            json.dump(run_meta, fh, indent=2)
    saved_frames = []
    t_start = time.time()
    for role, d in stations:
        if round(d, 2) in done_d:
            print(f"  [{role} d={d:+.2f}] already in {csv_path} (--resume) -- skip",
                  flush=True)
            continue
        st_path = station_geometry_path(stem, d, engine=args.engine)
        if not os.path.exists(st_path):
            print(f"  [{role} d={d:+.2f}] MISSING station geometry {st_path} -- "
                  f"re-run displace_wheel.py --engine {args.engine} --dump-stations. "
                  f"skipping.", flush=True)
            continue
        atoms = read(st_path)
        assert len(atoms) == n_atoms, \
            f"{st_path}: {len(atoms)} atoms != rod+wheel {n_atoms}"
        atoms.set_pbc(False)
        atoms.center(vacuum=VACUUM)
        atoms.info["charge"] = charge
        atoms.info["spin"] = spin
        attach_calc(atoms)  # UMA: fresh FAIRChemCalculator(predictor); tblite: TBLite

        # Rod axis from the station's own rod (PCA), matching the scan gauge.
        u = rod_axis(atoms.get_positions()[:rod_n])
        d0 = wheel_displacement(atoms.get_positions(), rod_n, u)

        # Constraint set.
        # Rigid wheel (DEFAULT): FixAtoms ALL wheel atoms + the two rod-tip
        # anchors -- exactly the scan's constrained surface (wheel rigid, rod
        # endpoints anchored). This freezes d BY CONSTRUCTION (wheel centroid
        # fixed; rod rigid body fixed), so there are 0 imaginary modes at both
        # well and saddle (the reaction coordinate is removed), and
        # E_relaxed ~ scan energy (same surface, just tighter fmax) -> DeltaE
        # matches the scan's potential barriers. F_vib is the ROD-internal
        # vibrational free energy (the d-dependent part that shifts the
        # barrier); wheel-internal vib is excluded -- it is ~d-independent and
        # cancels, and the scan already treats the wheel as rigid. This is the
        # vibrational correction to the scan's potential barriers.
        # --no-rigid-wheel: instead pin `--fix-wheel` spread wheel atoms (frees
        # the rest of the wheel so it can relax internally). NOTE: for the
        # flexible 56-atom crown-ether wheel this does NOT hold d -- the free
        # wheel atoms deform and shift the centroid (d drifts ~0.5 A, a
        # spurious imaginary mode appears). Use only for rigid-wheel systems.
        pos_now = atoms.get_positions()
        if args.rigid_wheel:
            fix_idx = list(range(rod_n, rod_n + wheel_n))
            wheel_desc = f"all {wheel_n} wheel"
        else:
            wheel_pos = pos_now[rod_n:]
            local_idx = pick_spread_wheel_atoms(wheel_pos, k=args.fix_wheel)
            fix_idx = [rod_n + i for i in local_idx]
            wheel_desc = f"{args.fix_wheel} wheel"
        if not args.no_rod_anchors:
            proj = pos_now[:rod_n] @ u
            fix_idx += [int(np.argmin(proj)), int(np.argmax(proj))]
            wheel_desc += " + 2 rod anchors"

        print(f"\n=== station {role}  d={d:+.2f} A  (d_measured={d0:+.2f})  "
              f"fixing {len(fix_idx)} atoms ({wheel_desc}) ===", flush=True)
        label = f"st_d{d:+.2f}".replace(".", "p").replace("+", "m").replace("-", "n")
        E_rel, F_vib, n_imag, freqs = run_station(
            atoms, fix_idx, args.relax_fmax, args.delta, args.temperature,
            scratch, label, compute_vib=not args.no_hessian)

        # Confirm the wheel was held (d after the constrained relax).
        d1 = wheel_displacement(atoms.get_positions(), rod_n, u)
        dd = d1 - d0
        G = (E_rel + F_vib) if F_vib is not None else None
        results.append({
            "role": role, "d": d, "d_after": d1, "d_shift": dd,
            "E_relaxed_eV": E_rel, "Fvib_eV": F_vib,
            "Fvib_kcal": (F_vib * EV_TO_KCAL_MOL) if F_vib is not None else None,
            "G_eV": G, "G_kcal": (G * EV_TO_KCAL_MOL) if G is not None else None,
            "G_rel_kcal": ((G - e_min) * EV_TO_KCAL_MOL) if G is not None else None,
            "n_imag": n_imag,
        })
        if not args.no_hessian:
            write_freeenergy_csv(csv_path, results, wells)  # incremental checkpoint
        if F_vib is not None:
            print(f"  [{role} d={d:+.2f}] E_relaxed={E_rel:.6f} eV  "
                  f"F_vib={F_vib:.4f} eV ({F_vib*EV_TO_KCAL_MOL:.1f} kcal)  "
                  f"G={G:.6f} eV", flush=True)
        else:
            print(f"  [{role} d={d:+.2f}] E_relaxed={E_rel:.6f} eV  "
                  f"(Hessian skipped: F_vib/G not computed)", flush=True)
        print(f"  [{role} d={d:+.2f}] d held: {d0:+.2f} -> {d1:+.2f} A "
              f"(shift {dd:+.2f})  n_imag={n_imag}", flush=True)
        if n_imag and n_imag > 0:
            print(f"  [{role} d={d:+.2f}] WARNING {n_imag} imaginary modes -- "
                  f"reaction coordinate may not be fully frozen (try --fix-wheel 6) "
                  f"or the geometry is not a constrained stationary point.",
                  flush=True)

        # Capture the vib-RELAXED geometry for the view PDB. atoms has just been
        # tight-relaxed on the wheel-position-constrained surface; this is the
        # structure the free energy was computed on (the scan-station input was
        # only fmax 0.05). The frame is later translated so its rod centroid
        # coincides with the global-min frame's (no rotation -- the rod is
        # ~end-symmetric, so a best-fit rotation would have a 180 deg ambiguity
        # and flip some frames, scrambling the wheel's station order).
        if args.save_geometries:
            anchor_idx = (fix_idx[-2:] if not args.no_rod_anchors else None)
            saved_frames.append({
                "role": role, "d": d, "d_measured": d1,  # post-relax (matches the
                # saved geometry; the wheel is rigid so it doesn't move, but the
                # rod centroid shifts during the rod-internal relax, so the
                # pre-relax d0 would not match the frame)
                "E_relaxed_eV": E_rel, "Fvib_eV": F_vib, "G_eV": G,
                "n_imag": n_imag,
                "positions": atoms.get_positions().copy(),
                "symbols": list(atoms.get_chemical_symbols()),
                "anchor_idx": anchor_idx,
            })

        atoms.calc = None  # detach before reuse

    print(f"\nvib_stations: all stations done in {time.time()-t_start:.0f} s")

    # Barriers: G_saddle - G_well, paired by side.
    by_role = {}
    for r in results:
        by_role.setdefault(r["role"], []).append(r)
    print("\n=== free-energy barriers (G_saddle - G_well) ===")
    if args.no_hessian:
        print("  (--no-hessian: G not computed -> barriers skipped; only the")
        print("   potential-energy DeltaE is available from the scan CSV)")
    for w in wells:
        side = w["side"]
        wr = next((r for r in by_role.get("well", []) if abs(r["d"]-w["d"]) < 0.05), None)
        sr = next((r for r in by_role.get(f"saddle({side})", [])
                   if abs(r["d"]-w["saddle_d"]) < 0.05), None)
        if not wr or not sr:
            print(f"  well d={w['d']:+.2f} ({side}): missing well/saddle result, skip")
            continue
        if wr["G_eV"] is None or sr["G_eV"] is None:
            print(f"  well d={w['d']:+.2f} ({side}): G not computed (--no-hessian), "
                  f"skip DeltaG")
            continue
        dG = (sr["G_eV"] - wr["G_eV"]) * EV_TO_KCAL_MOL
        dE = (sr["E_relaxed_eV"] - wr["E_relaxed_eV"]) * EV_TO_KCAL_MOL
        print(f"  well d={w['d']:+.2f} ({side}):  "
              f"DeltaE_potential={dE:+.2f}  DeltaG={dG:+.2f} kcal/mol  "
              f"(scan barrier was {w['barrier_kcal']:.2f})  "
              f"[F_vib correction {dG-dE:+.2f}]")

    # Final CSV pass. write_freeenergy_csv was already called after EVERY
    # station (incremental checkpoint), so the file is current and a kill
    # never loses a completed row; this final call just ensures every
    # well/saddle barrier pair is filled. Skipped under --no-hessian: only
    # E_relaxed (no F_vib/G) is available, so a partial CSV would mislead and
    # would clobber a full freeenergy.csv from a prior Hessian run.
    if args.no_hessian:
        print(f"vib_stations: --no-hessian -> not writing {csv_path} (only "
              f"E_relaxed available; keep the full CSV from the Hessian run)")
    else:
        write_freeenergy_csv(csv_path, results, wells)
        print(f"\nvib_stations: wrote {csv_path}")
    print(f"vib_stations: scratch (freq files) left at {scratch}")

    # Save the vib-RELAXED geometries: per-station .xyz + a multi-state PDB for
    # PyMOL (mplay). Each frame is TRANSLATED so its rod centroid coincides with
    # the global-min frame's rod centroid (translation only, no rotation) so the
    # wheel's shuttle motion is what you see stepping through the states, not the
    # rod jumping from the per-frame vacuum centering. No rotation is used on
    # purpose: the rod is approximately end-symmetric, so a best-fit ROTATION
    # (Kabsch) has a 180 deg ambiguity and flips some frames onto the rod's
    # mirror, putting the wheel on the wrong side and scrambling the station
    # order. The rod long axes are nearly parallel across stations (only the
    # bowing differs), so a pure centroid translation leaves them overlapping and
    # the wheel at its actual relaxed d_measured; the residual station-to-station
    # rod bowing (~1-3 A) is real and stays visible.
    if args.save_geometries and saved_frames:
        from ase import Atoms
        from ase.io import write as ase_write

        ref_fr = next((f for f in saved_frames if f["role"] == "global_min"),
                      saved_frames[0])
        ref_rc = ref_fr["positions"][:rod_n].mean(axis=0)
        # Consistent reference axis = global-min frame's rod PCA long axis. The
        # per-station rod PCA axis rotates slightly as the rod bows, so measuring
        # each station's wheel offset against a *fixed* axis gives a physical
        # position that is comparable across stations (the saved d_measured uses
        # each station's own axis and is NOT monotonic in the grid d -- see the
        # rod-centroid-gauge / over-relaxation notes in the README).
        ref_u = rod_axis(ref_fr["positions"][:rod_n])
        ref_u = ref_u / np.linalg.norm(ref_u)
        realigned = []
        for fr in saved_frames:
            pos = fr["positions"].copy()
            pos += (ref_rc - pos[:rod_n].mean(axis=0))
            # Physical wheel position along the fixed axis (what a viewer sees
            # after the centroid alignment). The wheel is rigid (FixAtoms) so its
            # absolute position is the scan's; this measures where it sits along
            # a common rod axis. Sort by this so stepping through the states shows
            # the wheel moving smoothly left -> right.
            phys = float((pos[rod_n:].mean(axis=0) - pos[:rod_n].mean(axis=0)) @ ref_u)
            realigned.append((fr, pos, phys))

        # Order frames left -> right by PHYSICAL wheel position for sequential
        # playback. (The grid d is NOT monotonic in the physical position because
        # the tight relax shifts the rod reference -- sorting by grid d makes the
        # wheel jump back and forth when you step through the PDB.)
        realigned.sort(key=lambda rpp: rpp[2])

        st_dir = os.path.join(OUTPUT_DIR, f"{stem}_vibstations{engine_tag(args.engine)}")
        os.makedirs(st_dir, exist_ok=True)
        frames_pdb = []

        def _fmt(v, fmt):
            return "n/a" if v is None else fmt % v

        for fr, pos, phys in realigned:
            d = fr["d"]
            tag = (f"{fr['role']} d={d:+.2f}A phys={phys:+.2f}A "
                   f"E_rel={fr['E_relaxed_eV']:.4f}eV "
                   f"Fvib={_fmt(fr['Fvib_eV'], '%.4f')}eV "
                   f"G={_fmt(fr['G_eV'], '%.4f')}eV n_imag={fr['n_imag']}")
            write_plain_xyz(os.path.join(st_dir, f"d{d:+.2f}.xyz"),
                            Atoms(fr["symbols"], positions=pos), comment=tag)
            a = Atoms(fr["symbols"], positions=pos)
            a.info["name"] = tag
            a.set_pbc(False)
            frames_pdb.append(a)

        pdb_path = os.path.join(OUTPUT_DIR,
                                f"{stem}_vibstations_view{engine_tag(args.engine)}.pdb")
        ase_write(pdb_path, frames_pdb)
        # Insert a REMARK after each MODEL line so the per-frame role/d/energy is
        # visible in PyMOL (ASE's PDB writer does not title the MODELs).
        with open(pdb_path) as fh:
            lines = fh.read().splitlines()
        out = []
        for ln in lines:
            out.append(ln)
            if ln.startswith("MODEL "):
                idx = int(ln.split()[1]) - 1
                fr, _pos, phys = realigned[idx]
                out.append(
                    f"REMARK   1 STATE {idx+1}  role={fr['role']}  "
                    f"phys_pos={phys:+.2f} A (wheel pos along fixed axis, "
                    f"sorted)  d={fr['d']:+.2f} A (grid)  "
                    f"d_relaxed={fr['d_measured']:+.2f} A (post-relax, own axis)  "
                    f"E_relaxed={fr['E_relaxed_eV']:.4f} eV  "
                    f"Fvib={_fmt(fr['Fvib_eV'], '%.4f')} eV  "
                    f"G={_fmt(fr['G_eV'], '%.4f')} eV  "
                    f"n_imag={fr['n_imag']}"
                )
        with open(pdb_path, "w") as fh:
            fh.write("\n".join(out) + "\n")
        # Report the residual rod-atom RMSD after centroid alignment (= the real
        # station-to-station rod bowing, which no translation can remove).
        ref_pos = next(p for fr, p, _ in realigned if fr is ref_fr)
        rmsds = [float(np.sqrt(((p[:rod_n] - ref_pos[:rod_n])**2)
                                .sum(axis=1).mean())) for fr, p, _ in realigned]
        print(f"vib_stations: wrote {len(realigned)} relaxed station geometries "
              f"-> {st_dir}/d*.xyz")
        print(f"vib_stations: wrote multi-state view PDB -> {pdb_path}")
        print("  (open in PyMOL; mplay / <- -> buttons step through the states "
              "left -> right by PHYSICAL wheel position, not grid d)")
        print(f"  rod residual RMSD vs global-min frame (rod bowing): "
              f"max {max(rmsds):.2f} A")
        print("  (open in PyMOL; mplay / <- -> buttons step through the stations, "
              "left -> right by d)")
    elif args.save_geometries:
        print("vib_stations: no station geometries captured (no completed stations)")


if __name__ == "__main__":
    main()