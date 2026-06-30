#!/usr/bin/env python
"""Optimize the rotaxane geometry with Meta's UMA MLIP via fairchem-core/ASE.

Reads the assembled rotaxane XYZ produced by build_rotaxane.py, attaches a
UMA calculator (uma-s-1p1, molecular 'omol' task), relaxes the structure with
an ASE optimizer, and writes PyMOL-friendly outputs only:
  - rotaxane_uma_clean.xyz : plain standard XYZ of the final relaxed frame
                             (element + x y z, re-centered to the origin).
  - rotaxane_uma.pdb       : multi-state PDB of every frame (initial + each
                             optimization step), each re-centered, so PyMOL
                             loads it as an animated trajectory (`mplay`).

ASE's default XYZ writer emits extended XYZ (forces/lattice/long comment)
which PyMOL misreads, so we write a plain XYZ by hand and a PDB instead.

Requires:
  - fairchem-core installed (brings ASE + torch)
  - a HuggingFace token with access to the UMA weights, available as the
    HF_TOKEN environment variable (e.g. exported from ~/.zshrc).
"""

import argparse
import os
import sys

import numpy as np
import torch
from ase.io import read, write
from ase.optimize import LBFGS
from fairchem.core import pretrained_mlip, FAIRChemCalculator

HERE = os.path.dirname(os.path.abspath(__file__))
IN_FILE = os.path.join(HERE, "rotaxane.xyz")
CLEAN_XYZ = os.path.join(HERE, "rotaxane_uma_clean.xyz")
PDB_FILE = os.path.join(HERE, "rotaxane_uma.pdb")
SMILES_FILE = os.path.join(HERE, "rot_smiles.txt")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=IN_FILE,
                   help="starting geometry (default: rotaxane.xyz)")
    p.add_argument("--out-xyz", default=CLEAN_XYZ,
                   help="plain-XYZ output of the final frame")
    p.add_argument("--out-pdb", default=PDB_FILE,
                   help="multi-state PDB output of the whole relaxation")
    p.add_argument("--fmax", type=float, default=FMAX,
                   help="force convergence tolerance in eV/A (default 0.05)")
    p.add_argument("--steps", type=int, default=STEPS,
                   help="max optimization steps (default 200)")
    p.add_argument("--smiles", default=SMILES_FILE,
                   help="rod:/wheel: file to read optional charge/spin from "
                        "(default: rot_smiles.txt)")
    return p.parse_args()

MODEL = "uma-s-1p1"          # UMA small checkpoint (auto-downloaded from HF)
TASK = "omol"                # molecular (non-periodic) task
FMAX = 0.05                  # eV/A force convergence tolerance
STEPS = 200                  # max optimization steps
VACUUM = 20.0                # A padding around the molecule (non-periodic)


def pick_device():
    """fairchem's MLIP predict unit only accepts 'cpu' or 'cuda' (it
    hard-asserts this; Apple Silicon MPS is not supported by the library).
    Prefer CUDA when available (e.g. on a Linux/GPU host), else CPU."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def read_charge_spin(path):
    """Read optional 'charge:' and 'spin:' lines from rot_smiles.txt.

    Defaults are charge=0, spin=1 (a neutral, closed-shell rotaxane). If the
    file carries explicit values, they override the defaults.
    """
    charge, spin = 0, 1
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("charge:"):
                    charge = int(line.split(":", 1)[1].strip())
                elif line.startswith("spin:"):
                    spin = int(line.split(":", 1)[1].strip())
    return charge, spin


def get_hf_token():
    """Return the HuggingFace token from the environment, or error out."""
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not tok:
        sys.exit(
            "No HuggingFace token found. Set HF_TOKEN in your environment "
            "(e.g. export HF_TOKEN=... in ~/.zshrc) and re-run in a shell "
            "that has sourced it."
        )
    return tok


def centered_copy(atoms):
    """Copy of an Atoms object translated so its centroid is at the origin."""
    a = atoms.copy()
    a.set_positions(a.get_positions() - a.get_positions().mean(axis=0))
    a.set_pbc(False)
    return a


def write_plain_xyz(path, atoms, comment=""):
    """Write a plain standard XYZ (element + x y z), not ASE's extended XYZ."""
    sym = atoms.get_chemical_symbols()
    pos = atoms.get_positions()
    with open(path, "w") as fh:
        fh.write(f"{len(sym)}\n{comment}\n")
        for s, (x, y, z) in zip(sym, pos):
            fh.write(f"{s:<2} {x: .6f} {y: .6f} {z: .6f}\n")


def np_max_force(atoms):
    f = atoms.get_forces()
    return float(np.linalg.norm(f, axis=1).max())


def main():
    args = parse_args()
    # huggingface_hub reads HF_TOKEN automatically; we only check it is present
    # so we can give a clear error before downloading the checkpoint.
    get_hf_token()

    atoms = read(args.input)
    atoms.set_pbc(False)
    atoms.center(vacuum=VACUUM)  # large non-periodic box; no self-images
    # Charge / spin for the combined rotaxane: read from the SMILES file if
    # present, else defaults (neutral, closed-shell).
    charge, spin = read_charge_spin(args.smiles)
    atoms.info["charge"] = charge
    atoms.info["spin"] = spin

    device = pick_device()
    print(f"loaded {len(atoms)} atoms from {args.input}")
    print(f"model={MODEL} task={TASK} device={device} charge={charge} spin={spin}")

    predictor = pretrained_mlip.get_predict_unit(MODEL, device=device)
    calc = FAIRChemCalculator(predictor, task_name=TASK)
    atoms.calc = calc

    e0 = atoms.get_potential_energy()
    f0 = np_max_force(atoms)
    print(f"initial: energy={e0:.4f} eV  max|F|={f0:.4f} eV/A")

    # Capture every frame in memory (initial + one after each optimizer step)
    # for the multi-state PDB trajectory.
    frames = []

    def capture():
        frames.append(atoms.copy())

    capture()  # initial geometry, before any step
    opt = LBFGS(atoms)
    opt.attach(capture)
    opt.run(fmax=args.fmax, steps=args.steps)

    e1 = atoms.get_potential_energy()
    f1 = np_max_force(atoms)
    print(f"final:   energy={e1:.4f} eV  max|F|={f1:.4f} eV/A  "
          f"(steps={opt.get_number_of_steps()})")

    # Plain XYZ of the final relaxed frame (centered).
    write_plain_xyz(args.out_xyz, centered_copy(atoms),
                    comment=f"rotaxane UMA-relaxed  E={e1:.6f} eV  "
                            f"max|F|={f1:.4f} eV/A  "
                            f"input={os.path.basename(args.input)}")
    # Multi-state PDB of the whole relaxation (each frame centered).
    write(args.out_pdb, [centered_copy(a) for a in frames])
    print(f"wrote {args.out_xyz} (final frame)")
    print(f"wrote {args.out_pdb} ({len(frames)} states)")


if __name__ == "__main__":
    main()