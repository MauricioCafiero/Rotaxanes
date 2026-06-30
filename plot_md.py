#!/usr/bin/env python
"""Plot MD observables from a run_md.py run and save them as PNGs.

Reads the stdout log (step / E_pot / E_kin / E_tot / T / wheel_x / d) and the
multi-state PDB trajectory, then writes:
  <prefix>temperature.png   T vs time
  <prefix>energy.png        E_pot, E_kin, E_tot vs time
  <prefix>wheel.png         wheel displacement along the rod vs time
  <prefix>rmsd.png          whole-structure RMSD vs frame 0 (Kabsch-aligned)
  <prefix>overview.png      2x2 panel of all four

The log is emitted every `--log-interval` MD steps; time = step * dt (fs).
"""
import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.io import read

from rotaxane_paths import resolve_stem, out_path

HERE = Path(__file__).resolve().parent
LOG_RE = re.compile(
    r"step\s+(\d+).*E_pot=([-\d.eE]+)\s+E_kin=([-\d.eE]+)\s+E_tot=([-\d.eE]+)"
    r"\s+eV\s+T=([\d.]+)\s+K(?:\s+wheel_x=([-+\d.]+)\s+A\s+d=([-+\d.]+))?"
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log", default=None,
                   help="MD stdout log (default: <stem>_md.log, stem from --pdb)")
    p.add_argument("--pdb", default=str(out_path("rot_smiles", "md", "pdb")),
                   help="multi-state PDB trajectory (default: <stem>_md.pdb). "
                        "The stem also sets --log and --prefix defaults.")
    p.add_argument("--dt", type=float, default=1.0, help="MD timestep in fs")
    p.add_argument("--log-interval", type=int, default=10,
                   help="log lines per MD step (default 10)")
    p.add_argument("--prefix", default=None,
                   help="output filename prefix (default: <stem>_md_)")
    p.add_argument("--no-rmsd", action="store_true",
                   help="skip RMSD (don't read the PDB)")
    return p.parse_args()


def kabsch_rmsd(p, p0):
    """RMSD of positions p onto p0 after optimal rotation (Kabsch)."""
    pc = p - p.mean(axis=0)
    p0c = p0 - p0.mean(axis=0)
    h = pc.T @ p0c
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    corr = np.eye(p.shape[1])
    corr[-1, -1] = d
    r = vt.T @ corr.T @ u.T
    return float(np.sqrt(((pc @ r - p0c) ** 2).sum(axis=1).mean()))


def read_log(path):
    rows = {}  # step -> dict (dedup by keeping last occurrence)
    with open(path) as fh:
        for line in fh:
            m = LOG_RE.search(line)
            if not m:
                continue
            s = int(m.group(1))
            rows[s] = dict(ep=float(m.group(2)), ek=float(m.group(3)),
                           et=float(m.group(4)), t=float(m.group(5)),
                           wx=(float(m.group(6)) if m.group(6) else None),
                           d=(float(m.group(7)) if m.group(7) else None))
    steps = sorted(rows)
    return steps, rows


def main():
    args = parse_args()
    stem = resolve_stem(args.pdb)
    log = args.log or out_path(stem, "md", "log")
    prefix = args.prefix or f"{stem}_md_"
    steps, rows = read_log(log)
    if not steps:
        raise SystemExit(f"no log lines parsed from {log}")
    t = np.array(steps, dtype=float) * args.dt  # fs
    T = np.array([rows[s]["t"] for s in steps])
    ep = np.array([rows[s]["ep"] for s in steps])
    ek = np.array([rows[s]["ek"] for s in steps])
    et = np.array([rows[s]["et"] for s in steps])
    wx = np.array([rows[s]["wx"] if rows[s]["wx"] is not None else np.nan
                  for s in steps])
    d = np.array([rows[s]["d"] if rows[s]["d"] is not None else np.nan
                 for s in steps])
    print(f"parsed {len(steps)} log points, t = {t[0]:.1f}..{t[-1]:.1f} fs")

    # --- RMSD from the trajectory PDB (Kabsch-aligned to frame 0) ---
    rmsd_t = rmsd = None
    if not args.no_rmsd and Path(args.pdb).exists():
        frames = read(args.pdb, index=":")
        p0 = frames[0].get_positions()
        rmsd = np.array([kabsch_rmsd(f.get_positions(), p0) for f in frames])
        # frames were stored every --stride steps (stride=5 here); t per frame.
        # stride = (last MD step) / (number of frames); the ratio must be
        # steps/frames (not the inverse, which rounds to 0 for stride>1).
        stride = max(1, round(steps[-1] / len(frames)))
        rmsd_t = np.arange(len(frames)) * stride * args.dt
        print(f"read {len(frames)} PDB frames (stride={stride}); RMSD range "
              f"{rmsd.min():.2f}..{rmsd.max():.2f} A")

    plt.rcParams.update({"figure.dpi": 130, "font.size": 11})

    # 1. temperature
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, T, color="tab:red", lw=1.0)
    ax.axhline(T.mean(), color="k", ls="--", lw=0.8,
               label=f"mean = {T.mean():.0f} K")
    ax.set_xlabel("time (fs)")
    ax.set_ylabel("temperature (K)")
    ax.set_title(f"MD temperature  (T0 = {T[0]:.0f} K)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{prefix}temperature.png")
    plt.close(fig)

    # 2. energy (normalized to the first value = 0, so the drift is readable)
    ep0, ek0, et0 = ep[0], ek[0], et[0]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, ep - ep0, label="E_pot", lw=1.0)
    ax.plot(t, ek - ek0, label="E_kin", lw=1.0)
    ax.plot(t, et - et0, label="E_tot", lw=1.2)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("time (fs)")
    ax.set_ylabel("energy - E(0)  (eV)")
    ax.set_title("MD energies  (normalized: first value = 0)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{prefix}energy.png")
    plt.close(fig)

    # 3. wheel displacement along the rod
    if not np.all(np.isnan(d)):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(t, d, color="tab:purple", lw=1.0)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("time (fs)")
        ax.set_ylabel("wheel displacement along rod (A)")
        ax.set_title("Wheel shuttle along the rod  "
                     "(+ = toward rod center)")
        fig.tight_layout()
        fig.savefig(f"{prefix}wheel.png")
        plt.close(fig)

    # 4. RMSD
    if rmsd is not None:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(rmsd_t, rmsd, color="tab:green", lw=1.0)
        ax.set_xlabel("time (fs)")
        ax.set_ylabel("RMSD vs frame 0 (A)")
        ax.set_title("Whole-structure RMSD (Kabsch-aligned)")
        fig.tight_layout()
        fig.savefig(f"{prefix}rmsd.png")
        plt.close(fig)

    # combined overview
    fig, axs = plt.subplots(2, 2, figsize=(11, 7))
    axs[0, 0].plot(t, T, color="tab:red", lw=1.0)
    axs[0, 0].axhline(T.mean(), color="k", ls="--", lw=0.8)
    axs[0, 0].set_ylabel("T (K)"); axs[0, 0].set_title("Temperature")

    axs[0, 1].plot(t, ep - ep[0], lw=1.0, label="E_pot")
    axs[0, 1].plot(t, ek - ek[0], lw=1.0, label="E_kin")
    axs[0, 1].plot(t, et - et[0], lw=1.2, label="E_tot")
    axs[0, 1].axhline(0, color="k", lw=0.6)
    axs[0, 1].set_ylabel("E - E(0) (eV)"); axs[0, 1].set_title("Energies (norm.)")
    axs[0, 1].legend(fontsize=8)

    if not np.all(np.isnan(d)):
        axs[1, 0].plot(t, d, color="tab:purple", lw=1.0)
        axs[1, 0].axhline(0, color="k", lw=0.6)
        axs[1, 0].set_ylabel("d (A)")
        axs[1, 0].set_title("Wheel displacement along rod")
    if rmsd is not None:
        axs[1, 1].plot(rmsd_t, rmsd, color="tab:green", lw=1.0)
        axs[1, 1].set_ylabel("RMSD (A)")
        axs[1, 1].set_title("Structure RMSD vs frame 0")
    for ax in axs[1]:
        ax.set_xlabel("time (fs)")
    fig.suptitle("Rotaxane shuttle MD (5 ps, 500 K Langevin)", y=1.0)
    fig.tight_layout()
    fig.savefig(f"{prefix}overview.png")
    plt.close(fig)
    print("wrote:", *[f"{prefix}{n}.png" for n in
          ("temperature", "energy", "wheel", "rmsd", "overview")])


if __name__ == "__main__":
    main()