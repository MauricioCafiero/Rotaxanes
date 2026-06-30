#!/usr/bin/env python
"""Slide the wheel along the rod to a steric extreme, for a more interesting
MD starting point.

Reads the UMA-relaxed rotaxane (the MD input, rotaxane_uma_clean.xyz), finds
the rod's long axis by PCA, and translates the wheel along +/- that axis as
far as possible without running into a rod stopper. Writes a plain standard
XYZ of the displaced rotaxane (centered), ready to feed to run_md.py.

The rod is bumpy (phenyl/CF3 groups), so as the wheel slides its closest
contact dips and recovers as features pass through the ring. A real stopper
is therefore a *sustained* overlap: the minimum rod-wheel distance stays
below a floor over a width of several angstroms. We scan outward and stop at
the first such sustained wall; the wheel is placed just before it (minus a
safety margin).
"""

import argparse
import os

import numpy as np
from ase.io import read
from rdkit import Chem

from build_rotaxane import SMILES_FILE, read_smiles

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IN = os.path.join(HERE, "rotaxane_uma_clean.xyz")
OUT_FILE = os.path.join(HERE, "rotaxane_displaced.xyz")

FLOOR = 1.0        # A; closer than this between a rod-wheel pair = overlap
WALL_WIDTH = 3.0   # A; a stopper is an overlap sustained over this width
SCAN_STEP = 0.05   # A scan increment
MARGIN = 0.3       # A safety back-off from the wall
MAX_SLIDE = 25.0   # A search limit (rod is ~28 A long)


def fragment_counts():
    """rod / wheel atom counts (with H) from rot_smiles.txt, matching build."""
    rod_smi, wheel_smi = read_smiles(SMILES_FILE)
    rod_n = Chem.AddHs(Chem.MolFromSmiles(rod_smi)).GetNumAtoms()
    wheel_n = Chem.AddHs(Chem.MolFromSmiles(wheel_smi)).GetNumAtoms()
    return rod_n, wheel_n


def rod_axis(rod_pos):
    """Unit vector along the rod's long axis (largest PCA eigenvalue)."""
    centered = rod_pos - rod_pos.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    return eigvecs[:, np.argmax(eigvals)]


def min_distance(rod_pos, wheel_pos):
    """Closest rod-wheel atom-atom distance (A)."""
    diff = rod_pos[:, None, :] - wheel_pos[None, :, :]
    return float(np.sqrt((diff ** 2).sum(axis=-1)).min())


def sustained_wall(rod_pos, wheel_pos, direction, d0):
    """True if the overlap below FLOOR is sustained over WALL_WIDTH starting d0."""
    for dd in np.arange(d0, d0 + WALL_WIDTH + SCAN_STEP, SCAN_STEP):
        w = wheel_pos + direction * dd
        if min_distance(rod_pos, w) >= FLOOR:
            return False
    return True


def max_slide(rod_pos, wheel_pos, direction):
    """Largest shift along `direction` before the first sustained stopper wall."""
    d = 0.0
    while d <= MAX_SLIDE:
        if sustained_wall(rod_pos, wheel_pos, direction, d):
            return max(0.0, d - SCAN_STEP)
        d += SCAN_STEP
    return MAX_SLIDE


def write_plain_xyz(path, symbols, pos, comment=""):
    with open(path, "w") as fh:
        fh.write(f"{len(symbols)}\n{comment}\n")
        for s, (x, y, z) in zip(symbols, pos):
            fh.write(f"{s:<2} {x: .6f} {y: .6f} {z: .6f}\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=DEFAULT_IN)
    p.add_argument("--side", choices=["left", "right", "farther"],
                   default="farther",
                   help="left/right = -/+ rod axis; farther (default) = the "
                        "side with the larger clash-free travel")
    p.add_argument("--margin", type=float, default=MARGIN,
                   help="A back-off from the stopper wall (default 0.3)")
    p.add_argument("--out", default=OUT_FILE)
    args = p.parse_args()

    atoms = read(args.input)
    symbols = atoms.get_chemical_symbols()
    pos = atoms.get_positions()
    rod_n, wheel_n = fragment_counts()
    assert len(symbols) == rod_n + wheel_n, \
        f"atom count {len(symbols)} != rod {rod_n} + wheel {wheel_n}"
    rod_pos = pos[:rod_n]
    wheel_pos = pos[rod_n:]

    u = rod_axis(rod_pos)
    proj = (rod_pos - rod_pos.mean(axis=0)) @ u
    rod_len = float(proj.max() - proj.min())

    right = max_slide(rod_pos, wheel_pos, u)
    left = max_slide(rod_pos, wheel_pos, -u)
    print(f"rod atoms: {rod_n}  wheel atoms: {wheel_n}  "
          f"rod length along axis: {rod_len:.2f} A")
    print(f"clash-free slide before stopper:  +right = {right:.2f} A,  "
          f"-left = {left:.2f} A")

    if args.side == "right":
        direction, travel, side = u, right, "right"
    elif args.side == "left":
        direction, travel, side = -u, left, "left"
    else:  # farther
        if right >= left:
            direction, travel, side = u, right, "right"
        else:
            direction, travel, side = -u, left, "left"

    d_place = max(0.0, travel - args.margin)
    wheel_new = wheel_pos + direction * d_place
    placed_min = min_distance(rod_pos, wheel_new)
    print(f"selected side: {side}  ->  wheel placed at {d_place:.2f} A "
          f"(travel {travel:.2f} minus margin {args.margin}); "
          f"closest rod-wheel contact there = {placed_min:.3f} A")

    new_pos = np.vstack([rod_pos, wheel_new])
    new_pos = new_pos - new_pos.mean(axis=0)
    write_plain_xyz(
        args.out, symbols, new_pos,
        comment=f"rotaxane wheel displaced {side} {d_place:.2f} A along rod "
                f"axis; stopper at {travel:.2f} A; closest contact "
                f"{placed_min:.3f} A",
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()