#!/usr/bin/env python
"""Vibrational analysis + thermochemistry on one geometry with the UMA MLIP.

Gauges the cost of a finite-difference Hessian (ASE ``Vibrations``, central
difference -> 6N force evals for N atoms) and reports ZPE / enthalpy / Gibbs at
a temperature, so we can decide whether Gibbs-on-the-full-scan is feasible or
whether to restrict it to peaks and troughs (wells + barrier tops).

Uses ASE's built-in vibrational + thermochemistry machinery -- no hand-rolled
finite differences. The calculator setup mirrors optimize_uma.py (UMA
uma-s-1p1, omol task), and charge/spin come from the <stem>.txt file.

CAVEAT: ``IdealGasThermo`` is ideal-GAS thermo (trans/rot/vib partition
functions), meant for gas-phase molecules, not a rotaxane in solution. Absolute
Gibbs is therefore not physically trustworthy for a condensed-phase rotaxane.
For RELATIVE barrier heights (well vs saddle, same atom count/mass) the trans
and rot terms nearly cancel, so the difference is dominated by the vibrational
ZPE + thermal correction -- which is the useful part. We print the vibrational
free-energy correction separately so it can be used alone if the gas-phase terms
look dubious.

Run from the project root:
    .venv/bin/python code/vib_uma.py --input output_files/rot2_displaced.xyz
"""

import argparse
import os
import sys
import time

import numpy as np
from ase.io import read
from fairchem.core import pretrained_mlip, FAIRChemCalculator

from optimize_uma import (
    MODEL, TASK, VACUUM, get_hf_token, pick_device, read_charge_spin,
)
from rotaxane_paths import resolve_stem, default_smiles, out_path

EV_TO_KCAL_MOL = 23.0605  # 1 eV = 23.0605 kcal/mol


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True,
                   help="geometry .xyz to analyse (its stem picks the SMILES "
                        "file for charge/spin).")
    p.add_argument("--smiles", default=None,
                   help="override the SMILES file (default: <stem>.txt).")
    p.add_argument("--temperature", type=float, default=300.0, help="K (default 300)")
    p.add_argument("--pressure", type=float, default=101325.0,
                   help="Pa (default 1 atm)")
    p.add_argument("--delta", type=float, default=0.01,
                   help="finite-difference displacement in A (ASE default 0.01)")
    p.add_argument("--no-thermo", action="store_true",
                   help="skip IdealGasThermo, report frequencies + ZPE only")
    p.add_argument("--relax-fmax", type=float, default=0.005,
                   help="eV/A: re-relax the input geometry to this tight fmax "
                        "BEFORE the Hessian (a scan-relaxed geometry at fmax 0.05 "
                        "is too loose -- its 6 trans/rot modes come out imaginary, "
                        "poisoning IdealGasThermo). Default 0.005; pass 0 to skip "
                        "(only safe on an already-tight minimum).")
    return p.parse_args()


def main():
    args = parse_args()
    stem = resolve_stem(args.input)
    smiles_path = args.smiles or default_smiles(stem)

    get_hf_token()
    charge, spin = read_charge_spin(smiles_path)
    device = pick_device()

    atoms = read(args.input)
    atoms.set_pbc(False)
    atoms.center(vacuum=VACUUM)
    atoms.info["charge"] = charge
    atoms.info["spin"] = spin

    predictor = pretrained_mlip.get_predict_unit(MODEL, device=device)
    atoms.calc = FAIRChemCalculator(predictor, task_name=TASK)

    natoms = len(atoms)
    n_dof = 3 * natoms
    n_force_evals = 2 * n_dof  # central difference
    print(f"vib: {natoms} atoms  {n_dof} DOF  -> ~{n_force_evals} UMA force evals "
          f"(central diff, delta={args.delta} A)")
    print(f"vib: model={MODEL} task={TASK} device={device} charge={charge} spin={spin} "
          f"T={args.temperature} K  p={args.pressure} Pa")

    # Re-relax to a tight fmax before the Hessian. A scan-relaxed geometry
    # (fmax 0.05) is too loose: its 6 trans/rot modes come out imaginary and a
    # couple of low modes go negative, so IdealGasThermo gets spurious imaginary
    # "vibrations". A tight free-molecule relax (no constraints) cleans the
    # trans/rot to ~0 and removes the spurious imaginary modes.
    if args.relax_fmax and args.relax_fmax > 0:
        from ase.optimize import LBFGS
        print(f"vib: pre-relaxing to fmax={args.relax_fmax} eV/A before Hessian",
              flush=True)
        t0 = time.time()
        opt = LBFGS(atoms, logfile=None)
        opt.run(fmax=args.relax_fmax, steps=500)
        print(f"vib: pre-relax done in {time.time() - t0:.1f} s "
              f"(E={atoms.get_potential_energy():.6f} eV)", flush=True)

    from ase.vibrations import Vibrations
    # ASE Vibrations writes per-displacement pickle files (<name>.xxxxx.pckl);
    # run them in a /tmp scratch so the repo isn't cluttered.
    scratch = f"/tmp/vib_{stem}_{os.getpid()}"
    os.makedirs(scratch, exist_ok=True)
    label = os.path.join(scratch, "vib")

    Epot = atoms.get_potential_energy()
    print(f"vib: potential energy = {Epot:.6f} eV "
          f"({Epot * EV_TO_KCAL_MOL:.2f} kcal/mol)")

    t0 = time.time()
    vib = Vibrations(atoms, name=label, delta=args.delta)
    vib.run()
    elapsed = time.time() - t0
    print(f"vib: Hessian done in {elapsed:.1f} s "
          f"({elapsed / n_force_evals:.2f} s/force-eval)")

    vib.summary()  # prints the frequency table
    # ASE get_energies() returns complex: real modes have e.real>0, e.imag~0;
    # imaginary modes have e.real~0, e.imag != 0. Convert to a real-signed array
    # (imaginary -> negative magnitude) WITHOUT a silent float() cast.
    raw = vib.get_energies()
    freqs = np.where(np.abs(raw.imag) > 1e-9, -np.abs(raw.imag), raw.real)
    n_imag = int(np.sum(freqs < -1e-6))
    ZPE = vib.get_zero_point_energy()
    print(f"vib: imaginary modes = {n_imag}  ZPE = {ZPE:.4f} eV "
          f"({ZPE * EV_TO_KCAL_MOL:.2f} kcal/mol)")
    # A clean minimum should have ~6 (trans/rot) imaginary-or-near-zero modes.
    # More than ~6 means the geometry is not at a tight minimum (spurious negative
    # curvature) -> tighten --relax-fmax, or the thermo is unreliable.
    if n_imag > 6:
        print(f"vib: WARNING {n_imag} imaginary modes (>6 trans/rot): geometry not "
              f"at a tight minimum; IdealGasThermo may be unreliable. Re-relax "
              f"tighter (--relax-fmax).")

    if args.no_thermo:
        return

    # IdealGasThermo(vib_selection='highest' default) keeps the HIGHEST 3N-6
    # energies (dropping the lowest 6 = trans/rot). So pass ALL energies and let
    # it select -- do NOT pre-strip (that leaves too few and raises ValueError).
    from ase.thermochemistry import IdealGasThermo
    # read_charge_spin returns the spin MULTIPLICITY (default 1); IdealGasThermo
    # wants the spin quantum number S = (multiplicity - 1) / 2.
    s_quantum = (spin - 1) / 2
    thermo = IdealGasThermo(
        vib_energies=list(freqs), potentialenergy=Epot, atoms=atoms,
        geometry="nonlinear", symmetrynumber=1, spin=s_quantum,
        ignore_imag_modes=True,  # tiny-imag trans/rot modes otherwise survive the
                                 # highest-3N-6 selection (f^2 ignores sign) and
                                 # give ln(negative) -> G = NaN
    )
    H = thermo.get_enthalpy(args.temperature, args.pressure)
    G = thermo.get_gibbs_energy(args.temperature, args.pressure)
    # Vibrational free-energy correction alone (no trans/rot): F_vib = ZPE - T*S_vib
    # over the real modes. ASE DOES expose this -- HarmonicThermo.get_helmholtz_energy
    # -- and that is what code/vib_stations.py uses for the constrained stations
    # (where IdealGasThermo's gas-phase trans/rot terms are wrong). This gauge script
    # keeps the full ideal-gas H/G for the free-molecule reference case, and reports
    # ZPE as the leading correction.
    print(f"vib: ideal-gas H({args.temperature}K) = {H:.6f} eV "
          f"({H * EV_TO_KCAL_MOL:.2f} kcal/mol)")
    print(f"vib: ideal-gas G({args.temperature}K) = {G:.6f} eV "
          f"({G * EV_TO_KCAL_MOL:.2f} kcal/mol)")
    print(f"vib: (H - Epot) = {(H - Epot) * EV_TO_KCAL_MOL:.2f} kcal/mol  "
          f"(G - Epot) = {(G - Epot) * EV_TO_KCAL_MOL:.2f} kcal/mol "
          f"[thermal corrections; G-Epot includes -T*S]")
    print(f"vib: scratch (freq files) left at {scratch}")


if __name__ == "__main__":
    main()