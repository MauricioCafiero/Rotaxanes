"""Stem-driven filename helpers for the rotaxane pipeline.

Every output file in the pipeline is named from a *stem* derived from the input
SMILES file (e.g. `rot1.txt` -> stem `rot1`). Downstream stages read a `.xyz`
whose name carries a role suffix (`rot1_center.xyz`, `rot1_relaxed.xyz`, ...);
they recover the stem by stripping that suffix, then name their own outputs
`<stem>_<role>.<ext>`.

This keeps a whole run's files linked by name without any hardcoded literals,
so the pipeline runs unchanged on `rot1.txt`, `rot_smiles.txt`, etc.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))

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


def out_path(stem, role, ext, directory=HERE):
    """`<directory>/<stem>_<role>.<ext>` (role="" -> no trailing underscore)."""
    suffix = f"_{role}" if role else ""
    return os.path.join(directory, f"{stem}{suffix}.{ext}")


def default_smiles(stem, directory=HERE):
    """Path to the SMILES/charge/spin .txt for a given stem."""
    return os.path.join(directory, f"{stem}.txt")