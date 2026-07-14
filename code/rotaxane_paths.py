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

# Force-source "engines" usable as the calculator behind the pipeline
# (UMA MLIP via fairchem, or tblite GFN-xTB). A non-default engine tags every
# output file so the two engines' results coexist on disk without clobbering
# each other (UMA keeps the legacy untagged names for back-compat). The tag is
# stripped by resolve_stem so the pipeline still chains on either engine:
# rot2_relaxed_tblite.xyz -> stem rot2 -> rot2_displaced_tblite.xyz, etc.
ENGINES = ("uma", "tblite")
DEFAULT_ENGINE = "uma"


def engine_tag(engine):
    """Filename suffix for an engine: ``""`` for the default (uma) / None,
    ``f"_{engine}"`` otherwise. Used to tag outputs and to strip in resolve_stem.
    """
    if not engine or engine == DEFAULT_ENGINE:
        return ""
    return f"_{engine}"


# Extension -> role-free input is just the stem (the .txt itself has no role).


def resolve_stem(path):
    """Recover the stem from any pipeline file path.

    Strips a trailing engine tag FIRST, then a trailing role suffix. The engine
    tag sits at the very end of the name (`<stem>_<role>_<engine>`), so the role
    suffix is only trailing once the tag is gone -- stripping the role first
    would miss it (e.g. `rot2_relaxed_tblite` does not end in `_relaxed` until
    `_tblite` is removed). UMA (the default engine) is untagged, so its paths are
    unchanged.

    `rot1.txt`                   -> `rot1`
    `rot1_center.xyz`            -> `rot1`
    `rot1_displaced_relaxed`     -> `rot1`
    `rot2_relaxed_tblite.xyz`     -> `rot2`
    `rot2_scan_tblite.csv`        -> `rot2`
    `rot2_displaced_relaxed_tblite.xyz` -> `rot2`
    """
    base = os.path.basename(path)
    root, ext = os.path.splitext(base)
    # strip a trailing engine tag (e.g. `_tblite`) first -- it is at the very end
    for eng in ENGINES:
        tag = f"_{eng}"
        if eng != DEFAULT_ENGINE and root.endswith(tag):
            root = root[: -len(tag)]
            break
    # then strip a trailing role suffix (now that the tag is gone it is trailing)
    for role in ROLES:
        if root.endswith(role):
            root = root[: -len(role)]
            break
    return root


def out_path(stem, role, ext, directory=None, engine=None):
    """`<directory>/<stem>_<role>[_engine].<ext>` (role="" -> no trailing underscore).

    `directory` defaults to `images/` for `.png` and `output_files/` for
    everything else, so plots and data files are filed automatically. `engine`
    (a name in ENGINES, or None) appends a `_<engine>` tag after the role for any
    non-default engine, so e.g. tblite outputs land as `rot2_scan_tblite.csv`
    alongside (not over) the UMA `rot2_scan.csv`. Default/uma -> no tag.
    """
    if directory is None:
        directory = IMAGES_DIR if ext == "png" else OUTPUT_DIR
    suffix = f"_{role}" if role else ""
    suffix += engine_tag(engine)
    return os.path.join(directory, f"{stem}{suffix}.{ext}")


def default_smiles(stem, directory=PROJECT_ROOT):
    """Path to the SMILES/charge/spin .txt for a given stem (project root)."""
    return os.path.join(directory, f"{stem}.txt")