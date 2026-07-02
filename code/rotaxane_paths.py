"""Stem-driven filename helpers for the rotaxane pipeline.

Every output file in the pipeline is named from a *stem* derived from the input
SMILES file (e.g. `rot1.txt` -> stem `rot1`). Downstream stages read a `.xyz`
whose name carries a role suffix (`rot1_center.xyz`, `rot1_relaxed.xyz`, ...);
they recover the stem by stripping that suffix, then name their own outputs
`<stem>_<role>.<ext>`.

This keeps a whole run's files linked by name without any hardcoded literals,
so the pipeline runs unchanged on `rot1.txt`, `rot_smiles.txt`, etc.

Repository layout (this module is the single source of truth for it): the
scripts live in `code/` (this file's directory); the input `.txt` files live at
the project root alongside the README; generated run data (structures /
trajectories / logs / CSVs) lives in `output_files/`; plots (`.png`) live in
`images/`. `out_path` auto-routes by extension so callers just ask for
`<stem>_<role>.<ext>` and the file lands in the right folder.
"""

import os

# This file lives in code/; the project root is its parent.
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output_files")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "images")

# Role suffixes appended to the stem, longest first so resolve_stem strips the
# most specific match (e.g. `_displaced_relaxed` before `_relaxed`).
ROLES = [
    "_displaced_relaxed",
    "_center",
    "_relaxed",
    "_displaced",
    "_md",
    "_scan",
]

# Extension -> role-free input is just the stem (the .txt itself has no role).


def resolve_stem(path):
    """Recover the stem from any pipeline file path.

    `rot1.txt`               -> `rot1`
    `rot1_center.xyz`        -> `rot1`
    `rot1_displaced_relaxed` -> `rot1`
    """
    base = os.path.basename(path)
    root, ext = os.path.splitext(base)
    for role in ROLES:
        if root.endswith(role):
            root = root[: -len(role)]
            break
    return root


def out_path(stem, role, ext, directory=None):
    """`<directory>/<stem>_<role>.<ext>` (role="" -> no trailing underscore).

    `directory` defaults to `images/` for `.png` and `output_files/` for
    everything else, so plots and data files are filed automatically.
    """
    if directory is None:
        directory = IMAGES_DIR if ext == "png" else OUTPUT_DIR
    suffix = f"_{role}" if role else ""
    return os.path.join(directory, f"{stem}{suffix}.{ext}")


def default_smiles(stem, directory=PROJECT_ROOT):
    """Path to the SMILES/charge/spin .txt for a given stem (project root)."""
    return os.path.join(directory, f"{stem}.txt")