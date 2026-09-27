"""
package_submission.py — Create the final submission ZIP archive.

Produces: ``<team_name>_submission.zip``

Required archive structure (per challenge rules):

    <team_name>_submission.zip
    ├── output/
    │   ├── matching_results.tsv
    │   └── candidate_pairs.tsv
    ├── code/
    │   └── business_entity_resolution/
    │       ├── src/                 # all source files
    │       ├── README.md
    │       └── requirements.txt
    └── Documentation_template.md

The script does NOT generate or fabricate output files.
If either required output file (matching_results.tsv or
candidate_pairs.tsv) is missing, the script fails with a clear message
and exits with code 1.

Excluded from the ZIP
---------------------
  - .git/ directories
  - __pycache__/ directories
  - *.pyc / *.pyo files
  - *.joblib model files (large binary; models should not be distributed)
  - virtual-environment directories (venv/, env/, .venv/, .env/)
  - .DS_Store and Thumbs.db
  - Any ``output/`` directory content other than the two required TSVs
  - Datasets (train_source*.tsv, test_source*.tsv, *.tsv in dataset/)

CLI usage
---------
    python src/package_submission.py --team-name <team_name> [--repo-root <path>]

    --team-name   Required. Used as the ZIP filename prefix and root
                  directory inside the archive.
    --repo-root   Optional. Path to the repository root.  Defaults to the
                  parent of the ``src/`` directory where this script lives.

The ZIP is written to the repository root directory.

Exit codes
----------
    0  ZIP created successfully.
    1  Required files missing or other error.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_OUTPUT_FILES = (
    "matching_results.tsv",
    "candidate_pairs.tsv",
)

CODE_SUBDIR = Path("code") / "business_entity_resolution"

DOCUMENTATION_FILE = "Documentation_template.md"

# Patterns excluded from the archive.
EXCLUDED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        "__pycache__",
        "venv",
        "env",
        ".venv",
        ".env",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        "node_modules",
        ".ipynb_checkpoints",
    }
)

EXCLUDED_SUFFIXES: frozenset[str] = frozenset(
    {
        ".pyc",
        ".pyo",
        ".pyd",
        ".joblib",
        ".pkl",
        ".pickle",
        ".DS_Store",
        ".log",
    }
)

EXCLUDED_FILENAMES: frozenset[str] = frozenset(
    {
        "Thumbs.db",
        ".gitattributes",
        ".gitignore",
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _should_exclude(path: Path, repo_root: Path) -> bool:
    """
    Return True if ``path`` should be excluded from the ZIP.

    Checks directory names along the relative path, file suffix, and
    filename against the exclusion lists.
    """
    try:
        rel = path.relative_to(repo_root)
    except ValueError:
        return False

    # Check each path component.
    for part in rel.parts[:-1]:  # parent directories
        if part in EXCLUDED_DIRS:
            return True

    # Check the filename itself.
    if path.name in EXCLUDED_FILENAMES:
        return True
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return True

    return False


def _collect_code_files(code_dir: Path, repo_root: Path) -> list[Path]:
    """
    Collect all source files under ``code_dir``, applying exclusions.

    Parameters
    ----------
    code_dir  : Path  Absolute path to code/business_entity_resolution/.
    repo_root : Path  Repository root (for relative-path computation).

    Returns
    -------
    list[Path]
        Absolute paths of files to include.
    """
    files: list[Path] = []
    for path in sorted(code_dir.rglob("*")):
        if not path.is_file():
            continue
        # Exclude any path component that is a known excluded dir.
        skip = False
        for part in path.relative_to(repo_root).parts[:-1]:
            if part in EXCLUDED_DIRS:
                skip = True
                break
        if skip:
            continue
        if path.name in EXCLUDED_FILENAMES:
            continue
        if path.suffix.lower() in EXCLUDED_SUFFIXES:
            continue
        files.append(path)
    return files


def _arcname(path: Path, repo_root: Path, team_name: str) -> str:
    """
    Compute the archive path for a file.

    All paths inside the ZIP are prefixed with ``<team_name>_submission/``.
    """
    rel = path.relative_to(repo_root)
    return f"{team_name}_submission/{rel.as_posix()}"


# ---------------------------------------------------------------------------
# Packaging logic
# ---------------------------------------------------------------------------


def build_zip(
    team_name: str,
    repo_root: Path,
) -> Path:
    """
    Build the submission ZIP.

    Parameters
    ----------
    team_name : str
        Used as ZIP filename prefix and root directory inside the archive.
    repo_root : Path
        Repository root directory.

    Returns
    -------
    Path
        Absolute path to the created ZIP file.

    Raises
    ------
    SystemExit
        If any required file is missing.
    """
    output_dir = repo_root / "output"
    code_dir = repo_root / CODE_SUBDIR
    doc_file = repo_root / DOCUMENTATION_FILE

    print(f"Repository root : {repo_root}")
    print(f"Team name       : {team_name}")
    print()

    # ------------------------------------------------------------------
    # 1. Verify required output files.
    # ------------------------------------------------------------------
    print("Checking required output files...")
    missing: list[str] = []
    for fname in REQUIRED_OUTPUT_FILES:
        fpath = output_dir / fname
        if not fpath.exists():
            missing.append(str(fpath))
        else:
            size_kb = fpath.stat().st_size / 1024
            print(f"  ✓  {fpath.relative_to(repo_root)}  ({size_kb:.1f} KB)")

    if missing:
        print(
            "\nERROR: The following required output files are missing:\n"
            + "\n".join(f"  - {p}" for p in missing),
            file=sys.stderr,
        )
        print(
            "\nRun the matching pipeline first to generate these files:\n"
            "  python src/pipeline.py \\\n"
            "    --source1 dataset/test/test_source1.tsv \\\n"
            "    --source2 dataset/test/test_source2.tsv \\\n"
            "    --source3 dataset/test/test_source3.tsv \\\n"
            "    --output  output \\\n"
            "    --threshold 0.85",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Verify code directory.
    # ------------------------------------------------------------------
    if not code_dir.exists():
        print(
            f"ERROR: Code directory not found: {code_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Warn if Documentation_template.md is missing (not fatal).
    # ------------------------------------------------------------------
    if not doc_file.exists():
        print(
            f"WARNING: {DOCUMENTATION_FILE} not found at {doc_file}. "
            "It will be omitted from the ZIP.",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------
    # 4. Collect files.
    # ------------------------------------------------------------------
    files_to_add: list[tuple[Path, str]] = []

    # output/ — only the two required TSVs.
    for fname in REQUIRED_OUTPUT_FILES:
        fpath = output_dir / fname
        files_to_add.append((fpath, _arcname(fpath, repo_root, team_name)))

    # code/business_entity_resolution/ — everything not excluded.
    code_files = _collect_code_files(code_dir, repo_root)
    for fpath in code_files:
        files_to_add.append((fpath, _arcname(fpath, repo_root, team_name)))

    # Documentation_template.md — if present.
    if doc_file.exists():
        files_to_add.append(
            (doc_file, _arcname(doc_file, repo_root, team_name))
        )

    # ------------------------------------------------------------------
    # 5. Write ZIP.
    # ------------------------------------------------------------------
    zip_name = f"{team_name}_submission.zip"
    zip_path = repo_root / zip_name

    print(f"\nWriting {zip_name}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath, arcname in files_to_add:
            zf.write(fpath, arcname)
            print(f"  + {arcname}")

    # ------------------------------------------------------------------
    # 6. Summary.
    # ------------------------------------------------------------------
    total_kb = zip_path.stat().st_size / 1024
    print(
        f"\n✓  Created {zip_path.name}  "
        f"({len(files_to_add)} files, {total_kb:.1f} KB)"
    )
    print(f"   Path: {zip_path}")

    # Print archive contents for verification.
    print("\nArchive contents:")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in sorted(zf.infolist(), key=lambda x: x.filename):
            kb = info.file_size / 1024
            print(f"  {info.filename}  ({kb:.1f} KB)")

    return zip_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for package_submission.py."""
    parser = argparse.ArgumentParser(
        description=(
            "Package the submission ZIP for the Amazon ML Challenge.\n\n"
            "Both output/matching_results.tsv and output/candidate_pairs.tsv\n"
            "must exist (generated by the matching pipeline) before packaging."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--team-name",
        required=True,
        help="Team name used as ZIP filename prefix (e.g. team_alpha).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help=(
            "Path to the repository root. "
            "Defaults to three levels above this script "
            "(src/ → business_entity_resolution/ → code/ → root)."
        ),
    )
    args = parser.parse_args()

    # Resolve repo root.
    if args.repo_root is not None:
        repo_root = args.repo_root.resolve()
    else:
        # This script lives at: <repo_root>/code/business_entity_resolution/src/
        repo_root = Path(__file__).resolve().parent.parent.parent.parent

    if not repo_root.is_dir():
        print(f"ERROR: Repository root not found: {repo_root}", file=sys.stderr)
        sys.exit(1)

    build_zip(
        team_name=args.team_name,
        repo_root=repo_root,
    )


if __name__ == "__main__":
    main()
