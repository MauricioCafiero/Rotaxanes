#!/usr/bin/env python
"""Build a rotaxane XYZ structure from rod + wheel SMILES.

Reads two SMILES (rod, then wheel) from a .txt file (default rot_smiles.txt),
generates 3D structures with RDKit, aligns the rod's long axis to x and the
wheel's 2D plane to yz, co-centroids the two, checks for steric overlap, and
writes the combined molecule in XYZ format. Output is <stem>_center.xyz where
<stem> is derived from the input filename (e.g. rot1.txt -> rot1_center.xyz).
"""

import argparse
import os
import sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from rotaxane_paths import resolve_stem, out_path

HERE = os.path.dirname(os.path.abspath(__file__))
SMILES_FILE = os.path.join(HERE, "rot_smiles.txt")

# Distance (A) below which a rod/wheel atom pair is flagged as overlapping.
OVERLAP_THRESHOLD = 1.0

# vdW radii (A) for the elements appearing in the rod/wheel, used to build a
# soft steric repulsion that drives the wheel-position optimization.
VDW_RADII = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47}


def read_smiles(path):
    """Parse rod: / wheel: lines into (rod_smiles, wheel_smiles)."""
    rod = wheel = None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("rod:"):
                rod = line[len("rod:"):].strip()
            elif line.startswith("wheel:"):
                wheel = line[len("wheel:"):].strip()
    if not rod or not wheel:
        raise ValueError(f"Could not find rod/wheel SMILES in {path}")
    return rod, wheel


def mol_to_xyz_block(mol):
    """Embed + optimize a molecule, return (elements list, Nx3 coords)."""
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC0FFEE
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise RuntimeError("RDKit embedding failed")
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        # Fall back to UFF if MMFF parameterization is unavailable.
        AllChem.UFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    pos = np.array(conf.GetPositions(), dtype=float)
    elems = [atom.GetSymbol() for atom in mol.GetAtoms()]
    return elems, pos


def pca_rotation(coords, order="desc"):
    """Rotation matrix R (3x3) that maps centered coords onto principal axes.

    order='desc': rows are eigenvectors sorted by descending eigenvalue
        -> largest-variance direction becomes the new x (rod long axis).
    order='asc':  rows are eigenvectors sorted by ascending eigenvalue
        -> smallest-variance direction becomes new x (wheel normal),
           so the wheel's high-variance plane lands in yz.
    """
    centered = coords - coords.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending by default
    if order == "desc":
        idx = np.argsort(eigvals)[::-1]
    else:
        idx = np.argsort(eigvals)
    R = eigvecs[:, idx].T  # rows are the chosen eigenvectors
    # Guarantee a right-handed frame (det = +1) so it's a pure rotation.
    if np.linalg.det(R) < 0:
        R[-1] *= -1
    return R


def contact_distances(rod_elems, wheel_elems):
    """Pairwise sum-of-vdW-radii contact distances (rod_atoms x wheel_atoms)."""
    rod_r = np.array([VDW_RADII[e] for e in rod_elems])[:, None]
    wheel_r = np.array([VDW_RADII[e] for e in wheel_elems])[None, :]
    return rod_r + wheel_r


def steric_energy(rod_pos, wheel_pos, d0):
    """Soft repulsive energy between rod and wheel atoms.

    Sum of (d0/r)^12 over pairs closer than 1.3*d0, where d0 is the sum of
    vdW radii for the pair. Vanishes once nothing is in contact, so it is
    smooth to minimize and rewards pushing close contacts apart.
    """
    diff = rod_pos[:, None, :] - wheel_pos[None, :, :]
    r = np.sqrt((diff ** 2).sum(axis=-1))
    mask = r < 1.3 * d0
    if not mask.any():
        return 0.0
    return float(((d0[mask] / r[mask]) ** 12).sum())


def optimize_wheel_offset(rod_pos, wheel_centered, d0, base,
                          step=0.1, min_step=1e-3):
    """Greedy coordinate-descent search for the wheel translation that
    minimizes steric energy.

    The wheel is placed at `wheel_centered + base + offset`. Offsets are
    explored along y and z first (nudge within the wheel's plane), then
    along x (slide along the rod), matching the requested fix order. The
    step halves whenever no single-axis move improves the energy, until it
    drops below `min_step`.
    """
    def energy(off):
        return steric_energy(rod_pos, wheel_centered + base + off, d0)

    off = np.zeros(3)
    best = energy(off)
    axes = [1, 2, 0]  # y, z (in-plane nudge), then x (slide along rod)
    while step > min_step:
        improved = False
        for ax in axes:
            for sign in (+1, -1):
                trial = off.copy()
                trial[ax] += sign * step
                e = energy(trial)
                if e < best - 1e-12:
                    off, best = trial, e
                    improved = True
        if not improved:
            step /= 2.0
    return off, best


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smiles", default=SMILES_FILE,
                   help="rod:/wheel: SMILES file (default: rot_smiles.txt). "
                        "Output names are derived from this file's stem "
                        "(e.g. rot1.txt -> rot1_center.xyz).")
    p.add_argument("--out", default=None,
                   help="output XYZ path (default: <stem>_center.xyz, derived "
                        "from the --smiles filename)")
    return p.parse_args()


def main():
    args = parse_args()
    smiles_file = args.smiles
    stem = resolve_stem(smiles_file)
    out_file = args.out or out_path(stem, "center", "xyz")
    rod_smi, wheel_smi = read_smiles(smiles_file)
    print(f"rod SMILES  : {rod_smi}")
    print(f"wheel SMILES: {wheel_smi}")

    rod_mol = Chem.MolFromSmiles(rod_smi)
    wheel_mol = Chem.MolFromSmiles(wheel_smi)
    if rod_mol is None:
        raise ValueError("Failed to parse rod SMILES")
    if wheel_mol is None:
        raise ValueError("Failed to parse wheel SMILES")

    rod_elems, rod_pos = mol_to_xyz_block(rod_mol)
    wheel_elems, wheel_pos = mol_to_xyz_block(wheel_mol)
    print(f"rod atoms  : {len(rod_elems)}  wheel atoms: {len(wheel_elems)}")

    # Align rod long axis -> x, wheel plane -> yz.
    R_rod = pca_rotation(rod_pos, order="desc")
    R_wheel = pca_rotation(wheel_pos, order="asc")
    rod_aligned = (rod_pos - rod_pos.mean(axis=0)) @ R_rod.T
    wheel_aligned = (wheel_pos - wheel_pos.mean(axis=0)) @ R_wheel.T

    # Place the wheel with its centroid at the rod's centroid, then optimize
    # the translation: nudge in the yz plane, then slide along x, to minimize
    # steric clashes with the rod.
    rod_c = rod_aligned.mean(axis=0)
    d0 = contact_distances(rod_elems, wheel_elems)

    e0 = steric_energy(rod_aligned, wheel_aligned + rod_c, d0)
    offset, e_opt = optimize_wheel_offset(rod_aligned, wheel_aligned, d0, rod_c)
    wheel_placed = wheel_aligned + rod_c + offset
    print(f"steric energy: {e0:.3g} -> {e_opt:.3g} after optimization")
    print(f"wheel offset (x,y,z): {np.round(offset, 3)} A")

    # Report extents to confirm the alignment did what we wanted.
    def extents(p):
        return p.max(axis=0) - p.min(axis=0)
    print(f"rod extents   (x,y,z): {np.round(extents(rod_aligned), 2)}")
    print(f"wheel extents (x,y,z): {np.round(extents(wheel_placed), 2)}")
    print(f"rod centroid: {np.round(rod_c, 3)}")

    # Overlap check: rod vs wheel atom pairs closer than threshold.
    diff = rod_aligned[:, None, :] - wheel_placed[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    pairs = np.argwhere(dist < OVERLAP_THRESHOLD)
    print(f"overlap pairs (< {OVERLAP_THRESHOLD} A): {len(pairs)}")
    for i, j in pairs:
        print(f"  rod {rod_elems[i]}-{i}  wheel {wheel_elems[j]}-{j}  "
              f"d={dist[i, j]:.3f} A")

    # Write combined XYZ.
    all_elems = rod_elems + wheel_elems
    all_pos = np.vstack([rod_aligned, wheel_placed])
    with open(out_file, "w") as fh:
        fh.write(f"{len(all_elems)}\n")
        fh.write(f"rotaxane  rod={len(rod_elems)}  wheel={len(wheel_elems)}  "
                 f"overlaps={len(pairs)}\n")
        for elem, (x, y, z) in zip(all_elems, all_pos):
            fh.write(f"{elem:<2} {x: .6f} {y: .6f} {z: .6f}\n")
    print(f"wrote {out_file}  ({len(all_elems)} atoms)")


if __name__ == "__main__":
    sys.exit(main())