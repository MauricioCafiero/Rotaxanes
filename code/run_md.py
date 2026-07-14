#!/usr/bin/env python
"""Run molecular dynamics on the UMA-relaxed rotaxane, using UMA for forces.

Defaults to a short test run (Langevin NVT at 300 K, 0.5 fs step, 100 fs).
Step size and total length are CLI options. Outputs PyMOL-friendly files only,
named from the input file's stem (e.g. rot1_relaxed.xyz -> rot1_md.xyz +
rot1_md.pdb):
  - <stem>_md.xyz : plain standard XYZ of the final frame.
  - <stem>_md.pdb : multi-state PDB trajectory (one state every `--stride`
                    steps); `mplay` in PyMOL to animate.

Requires HF_TOKEN in the environment (see optimize_uma.py / CLAUDE.md).
"""

import argparse
import os

import numpy as np
from ase import units
from ase.io import read, write
from ase.md import Langevin, VelocityVerlet
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
# NOTE: fairchem/torch is NOT imported at module top. run_md imports helpers
# from optimize_uma, whose own module top is engine-agnostic; the engine's
# libs load only inside make_calculator (see the tblite-libomp segfault note in
# optimize_uma.py). A top-level fairchem import here would pull torch into every
# process including tblite runs, defeating that isolation.

from optimize_uma import (
    MODEL,
    TASK,
    VACUUM,
    centered_copy,
    get_hf_token,
    make_calculator,
    np_max_force,
    read_charge_spin,
    write_plain_xyz,
)
from displace_wheel import fragment_counts, rod_axis
from rotaxane_paths import (
    resolve_stem,
    out_path,
    default_smiles,
    ENGINES,
    DEFAULT_ENGINE,
)

DEFAULT_IN = out_path("rot_smiles", "relaxed", "xyz")


def parse_args():
    p = argparse.ArgumentParser(description="UMA-driven MD on the rotaxane.")
    p.add_argument("--input", default=DEFAULT_IN,
                   help="starting geometry (default: <stem>_relaxed.xyz). "
                        "Outputs are named from this file's stem.")
    p.add_argument("--dt", type=float, default=0.5,
                   help="timestep in fs (default 0.5)")
    p.add_argument("--time", type=float, default=100.0,
                   help="total simulation length in fs (default 100)")
    p.add_argument("--thermostat", choices=["langevin", "nve"],
                   default="langevin",
                   help="langevin (NVT, default) or nve (Velocity-Verlet)")
    p.add_argument("--temperature", type=float, default=300.0,
                   help="temperature in K for velocity init / Langevin (default 300)")
    p.add_argument("--friction", type=float, default=0.01,
                   help="Langevin friction in 1/fs (default 0.01)")
    p.add_argument("--stride", type=int, default=1,
                   help="store a trajectory frame every N steps (default 1)")
    p.add_argument("--flush", type=int, default=100,
                   help="rewrite the PDB+XYZ outputs every N steps so a "
                        "killed/aborted run keeps its trajectory (default 100)")
    p.add_argument("--seed", type=int, default=0xC0FFEE,
                   help="RNG seed for initial velocities (default 0xC0FFEE)")
    p.add_argument("--out-xyz", default=None,
                   help="plain-XYZ output of the final frame "
                        "(default: <stem>_md.xyz)")
    p.add_argument("--out-pdb", default=None,
                   help="multi-state PDB output of the trajectory "
                        "(default: <stem>_md.pdb)")
    p.add_argument("--smiles", default=None,
                   help="rod:/wheel: file for atom counts + charge/spin "
                        "(default: <stem>.txt matching the input)")
    p.add_argument("--engine", default=DEFAULT_ENGINE, choices=ENGINES,
                   help="force source: 'uma' (Meta UMA MLIP, default; needs "
                        "HF_TOKEN) or 'tblite' (GFN-xTB; no HF_TOKEN). A "
                        "non-default engine tags outputs, e.g. "
                        "<stem>_md_tblite.xyz, so the engines coexist.")
    p.add_argument("--method", default="GFN2-xTB",
                   help="tblite method (default GFN2-xTB). Ignored for --engine uma.")
    return p.parse_args()


def kinetic_temperature(atoms):
    """Temperature (K) from kinetic energy, using 3N-6 DOF (non-linear mol)."""
    dof = max(1, 3 * len(atoms) - 6)
    return 2.0 * atoms.get_kinetic_energy() / (dof * units.kB)


def wheel_offset_along_rod(atoms, rod_n, ref_axis):
    """Wheel-centroid displacement along the rod's long axis (A).

    The rod's PCA axis sign is arbitrary, so it is re-aligned to `ref_axis`
    (the axis from the initial frame) every call; this keeps the reported
    position monotonically signed as the wheel shuttles, instead of flipping
    sign frame-to-frame.
    """
    pos = atoms.get_positions()
    rod_pos, wheel_pos = pos[:rod_n], pos[rod_n:]
    u = rod_axis(rod_pos)
    if np.dot(u, ref_axis) < 0:
        u = -u
    rc = rod_pos.mean(axis=0)
    return float((wheel_pos.mean(axis=0) - rc) @ u)


def main():
    args = parse_args()
    # tblite needs no HF_TOKEN; only gate UMA (which downloads the checkpoint).
    if args.engine == "uma":
        get_hf_token()

    stem = resolve_stem(args.input)
    out_xyz = args.out_xyz or out_path(stem, "md", "xyz", engine=args.engine)
    out_pdb = args.out_pdb or out_path(stem, "md", "pdb", engine=args.engine)
    smiles_path = args.smiles or default_smiles(stem)

    atoms = read(args.input)
    atoms.set_pbc(False)
    atoms.center(vacuum=VACUUM)
    charge, spin = read_charge_spin(smiles_path)
    atoms.info["charge"] = charge
    atoms.info["spin"] = spin

    calc, device = make_calculator(args.engine, atoms, charge, spin,
                                   method=args.method)
    atoms.calc = calc

    print(f"loaded {len(atoms)} atoms from {args.input}")
    if args.engine == "uma":
        print(f"engine=uma model={MODEL} task={TASK} device={device} "
              f"charge={charge} spin={spin}")
    else:
        print(f"engine=tblite method={args.method} charge={charge} spin={spin} "
              f"(no torch device)")

    # Initial velocities at T, then remove center-of-mass drift and rotation.
    rng = np.random.default_rng(args.seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature, rng=rng)
    Stationary(atoms)
    ZeroRotation(atoms)

    steps = max(1, int(round(args.time / args.dt)))
    # Rod/wheel split for the wheel-position-along-rod shuttle log.
    rod_n, _wheel_n = fragment_counts(smiles_path)
    assert len(atoms) == rod_n + _wheel_n, (
        f"atom count {len(atoms)} != rod {rod_n} + wheel {_wheel_n}; "
        "rot_smiles.txt must match the input geometry")
    ref_axis = rod_axis(atoms.get_positions()[:rod_n])
    off0 = wheel_offset_along_rod(atoms, rod_n, ref_axis)
    print(f"MD: thermostat={args.thermostat} dt={args.dt} fs "
          f"time={args.time} fs steps={steps} T0={args.temperature} K "
          f"friction={args.friction}/fs stride={args.stride} flush={args.flush}")
    print(f"initial: E_pot={atoms.get_potential_energy():.4f} eV  "
          f"T={kinetic_temperature(atoms):.1f} K  "
          f"max|F|={np_max_force(atoms):.4f} eV/A  "
          f"wheel_x={off0:+.2f} A (start)")

    if args.thermostat == "langevin":
        dyn = Langevin(atoms,
                       timestep=args.dt * units.fs,
                       temperature_K=args.temperature,
                       friction=args.friction / units.fs)
    else:
        dyn = VelocityVerlet(atoms, timestep=args.dt * units.fs)

    # Capture trajectory frames in memory.
    frames = []
    counter = {"n": 0}

    def capture(force=False):
        counter["n"] += 1
        if force or counter["n"] % args.stride == 0:
            frames.append(atoms.copy())

    capture(force=True)  # initial frame, regardless of stride
    dyn.attach(capture)

    # Periodic energy/temperature + wheel-shuttle log to stdout.
    def log_status():
        ep = atoms.get_potential_energy()
        ek = atoms.get_kinetic_energy()
        off = wheel_offset_along_rod(atoms, rod_n, ref_axis)
        print(f"step {dyn.get_number_of_steps():>4}  "
              f"E_pot={ep:.4f}  E_kin={ek:.4f}  "
              f"E_tot={ep + ek:.4f} eV  T={kinetic_temperature(atoms):.1f} K  "
              f"wheel_x={off:+.2f} A  d={off - off0:+.2f}",
              flush=True)

    log_status()
    dyn.attach(log_status, interval=10)

    # Incrementally rewrite the outputs so a killed/aborted run still has the
    # trajectory up to the last flush (the PDB is rewritten from all captured
    # frames; the clean XYZ is the current frame).
    def flush_outputs():
        try:
            if frames:
                write(out_pdb, [centered_copy(a) for a in frames])
            write_plain_xyz(out_xyz, centered_copy(atoms),
                            comment=f"rotaxane MD (partial) "
                                    f"thermostat={args.thermostat} "
                                    f"dt={args.dt}fs step="
                                    f"{dyn.get_number_of_steps()}/"
                                    f"{steps}  T={args.temperature}K  "
                                    f"input={os.path.basename(args.input)}")
        except Exception as exc:  # never let a flush abort the dynamics
            print(f"(flush warning at step "
                  f"{dyn.get_number_of_steps()}: {exc})", flush=True)

    flush_outputs()
    dyn.attach(flush_outputs, interval=max(1, args.flush))

    dyn.run(steps)

    # Final flush ensures the very last frame(s) are on disk even if the run
    # ended cleanly between flushes.
    flush_outputs()

    ep = atoms.get_potential_energy()
    ek = atoms.get_kinetic_energy()
    print(f"final:   E_pot={ep:.4f}  E_kin={ek:.4f}  E_tot={ep + ek:.4f} eV  "
          f"T={kinetic_temperature(atoms):.1f} K  "
          f"max|F|={np_max_force(atoms):.4f} eV/A")

    write_plain_xyz(out_xyz, centered_copy(atoms),
                    comment=f"rotaxane MD final  thermostat={args.thermostat} "
                            f"dt={args.dt}fs time={args.time}fs "
                            f"T={args.temperature}K  E_tot={ep + ek:.4f} eV  "
                            f"input={os.path.basename(args.input)}")
    write(out_pdb, [centered_copy(a) for a in frames])
    print(f"wrote {out_xyz} (final frame)")
    print(f"wrote {out_pdb} ({len(frames)} states)")


if __name__ == "__main__":
    main()