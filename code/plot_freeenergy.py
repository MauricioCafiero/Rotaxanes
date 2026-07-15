#!/usr/bin/env python
"""Plot potential energy and Gibbs free energy along the rot2 relaxed scan.

Overlays, all referenced to the scan global-min station (the role=global_min
row of the freeenergy CSV):
  - scan DeltaE potential curve (fmax 0.05) -- from <stem>_scan.csv
  - vib DeltaE_relaxed -- from <stem>_freeenergy.csv. With --relax-fmax 0
    (fork 1, rigid self-consistent) the Hessian is taken on the scan's OWN
    converged geometry, so E_relaxed is on the SAME surface as the scan and
    these points sit ON the scan DeltaE curve (a self-consistency check); only
    DeltaG = E_relaxed + F_vib separates from it. With the old tight re-relax
    (fmax 0.005) they instead diverge as the over-relaxed rod collapses.
  - vib DeltaG = E_relaxed + F_vib at T=300 K

With vib_stations --all-stations the freeenergy CSV carries a dense 'scan'
curve (every Nth station) PLUS the auto global_min/well/saddle stations; the
'scan' rows are drawn as a line + small dots and the well/saddle/global_min
rows as larger annotated markers (the discrete well/barrier overlay).

Run from the project root:
    .venv/bin/python code/plot_freeenergy.py [--stem rot2]
"""
import argparse, csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rotaxane_paths import (
    resolve_stem, out_path, IMAGES_DIR, default_smiles,
    ENGINES, DEFAULT_ENGINE, engine_tag,
)

EV_TO_KCAL = 23.0605


def read_scan(path):
    ds, es = [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            ds.append(float(r["displacement_A"]))
            es.append(float(r["energy_rel_kcal_mol"]))  # rel to scan global min (+0.80)
    return np.array(ds), np.array(es)


def read_freeenergy(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "role": r["role"], "d": float(r["d_A"]),
                "d_relaxed": float(r["d_after_A"]),
                "E_relaxed_eV": float(r["E_relaxed_eV"]),
                "Fvib_eV": float(r["Fvib_eV"]),
                "G_eV": float(r["G_eV"]), "n_imag": int(r["n_imag"]) if r["n_imag"] else None,
            })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stem", default="rot2")
    ap.add_argument("--engine", default=DEFAULT_ENGINE, choices=ENGINES,
                    help="engine tag of the scan + freeenergy CSVs to read "
                         "(default uma = untagged; tblite = <stem>_scan_tblite.csv "
                         "+ <stem>_freeenergy_tblite.csv). Must match the engine "
                         "the vib_stations run used.")
    ap.add_argument("--out", default=None, help="PNG path (default images/<stem>_freeenergy[_engine].png)")
    ap.add_argument("--emax", type=float, default=None, help="kcal/mol y-axis clip")
    args = ap.parse_args()
    stem = args.stem

    scan_csv = out_path(stem, "scan", "csv", engine=args.engine)
    fe_csv = out_path(stem, "freeenergy", "csv", engine=args.engine)
    if not os.path.exists(scan_csv):
        sys.exit(f"missing {scan_csv}")
    if not os.path.exists(fe_csv):
        sys.exit(f"missing {fe_csv} -- run vib_stations.py --engine {args.engine} first")
    ds, scan_de = read_scan(scan_csv)
    st = read_freeenergy(fe_csv)

    # reference = the d=+0.80 station (scan global min). vib E_relaxed/G there:
    ref = next(r for r in st if r["role"] == "global_min")
    E0, G0 = ref["E_relaxed_eV"], ref["G_eV"]
    for r in st:
        r["dE_relaxed"] = (r["E_relaxed_eV"] - E0) * EV_TO_KCAL
        r["dG"] = (r["G_eV"] - G0) * EV_TO_KCAL

    # ---- table ----
    print(f"\n=== {stem}: energy + Gibbs along the relaxed scan "
          f"(ref: d={ref['d']:+.2f}, role=global_min) ===")
    print(f"{'role':>12} {'d_grid':>7} {'d_relax':>8} | {'scan dE':>8} {'vib dE':>8} {'vib dG':>8} | {'Fvib':>6} {'nimag':>5}")
    # scan dE at each station's grid d (nearest scan point)
    def scan_de_at(d):
        i = int(np.argmin(np.abs(ds - d)))
        return float(scan_de[i])
    for r in st:
        print(f"{r['role']:>12} {r['d']:+7.2f} {r['d_relaxed']:+8.2f} | "
              f"{scan_de_at(r['d']):+8.2f} {r['dE_relaxed']:+8.2f} {r['dG']:+8.2f} | "
              f"{r['Fvib_eV']*EV_TO_KCAL:6.1f} {str(r['n_imag']):>5}")

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    order = np.argsort(ds)
    ax.plot(ds[order], scan_de[order], "-", color="#888", lw=1.5, zorder=1,
            label=r"scan $\Delta E$ (potential, fmax 0.05)")
    ax.axhline(0, color="#ccc", lw=0.8, zorder=0)

    sd = sorted(st, key=lambda r: r["d"])
    scan_rows = [r for r in sd if r["role"] == "scan"]
    key_rows = [r for r in sd if r["role"] != "scan"]

    if scan_rows:
        # Dense --all-stations curve: line + small dots, no per-point annotation.
        ax.plot([r["d"] for r in scan_rows],
                [r["dE_relaxed"] for r in scan_rows],
                "s-", color="#1f77b4", ms=3, lw=1.0, alpha=0.7, zorder=2,
                label=r"vib $\Delta E_{relaxed}$ (rigid, self-consistent w/ scan)")
        ax.plot([r["d"] for r in scan_rows], [r["dG"] for r in scan_rows],
                "o-", color="#d62728", ms=3, lw=1.4, zorder=3,
                label=r"vib $\Delta G = E_{relaxed}+F_{vib}$ (T=300 K)")
        # Auto wells/saddles/global_min: larger markers + annotations on top.
        ax.plot([r["d"] for r in key_rows],
                [r["dE_relaxed"] for r in key_rows],
                "s", color="#1f77b4", ms=8, zorder=4)
        ax.plot([r["d"] for r in key_rows], [r["dG"] for r in key_rows],
                "o", color="#d62728", ms=8, zorder=5,
                label="well / saddle / global-min stations")
        for r in key_rows:
            ax.annotate(r["role"].split("(")[0], (r["d"], r["dG"]),
                        textcoords="offset points", xytext=(6, 6), fontsize=7,
                        color="#d62728")
    else:
        # Legacy 5-station CSV (no 'scan' rows): plot all as annotated markers.
        ax.plot([r["d"] for r in sd], [r["dE_relaxed"] for r in sd],
                "s--", color="#1f77b4", ms=7, zorder=3,
                label=r"vib $\Delta E_{relaxed}$ (potential)")
        ax.plot([r["d"] for r in sd], [r["dG"] for r in sd],
                "o-", color="#d62728", ms=8, zorder=4,
                label=r"vib $\Delta G = E_{relaxed}+F_{vib}$ (T=300 K)")
        for r in sd:
            ax.annotate(r["role"].split("(")[0], (r["d"], r["dG"]),
                        textcoords="offset points", xytext=(6, 6), fontsize=7,
                        color="#d62728")
    gmin = next((r for r in st if r["role"] == "global_min"), None)
    if gmin is not None:
        ax.plot(gmin["d"], gmin["dG"], "*", color="#d62728", ms=16, zorder=6)

    ax.set_xlabel("wheel displacement $d$ along rod (Å, grid)")
    ax.set_ylabel(f"energy / free energy (kcal/mol, rel. to $d={ref['d']:+.2f}$)")
    ax.set_title(f"{stem} ({args.engine}): potential vs Gibbs free energy along the shuttle scan")
    if args.emax:
        ax.set_ylim(top=args.emax)
    ax.legend(loc="upper center", fontsize=8, framealpha=0.9)
    ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()

    out = args.out or out_path(stem, "freeenergy", "png", engine=args.engine)
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()