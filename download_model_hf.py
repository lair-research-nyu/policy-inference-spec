#!/usr/bin/env python3
"""Download model checkpoints from the Hugging Face Hub into the local models/ folder.

Each checkpoint is stored as a flat directory under models/ named after the repo,
e.g. the repo `lair-nyu/cubeinbox-uniform-pi0-20-010000` is downloaded to
`models/cubeinbox-uniform-pi0-20-010000`.

Usage:
    python download_model_hf.py lair-nyu/cubeinbox-uniform-pi0-20-010000
    python download_model_hf.py lair-nyu/repo-a lair-nyu/repo-b
    python download_model_hf.py lair-nyu/repo --name custom-folder-name
    python download_model_hf.py lair-nyu/repo --force   # re-download even if it exists
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

MODELS_DIR = Path(__file__).resolve().parent / "models"


def download(repo_id: str, name: str | None = None, force: bool = False) -> Path:
    """Download a single HF repo into models/<name>."""
    folder_name = name or repo_id.split("/")[-1]
    dest = MODELS_DIR / folder_name

    if dest.exists() and any(dest.iterdir()) and not force:
        print(f"[skip] {dest} already exists and is non-empty (use --force to re-download)")
        return dest

    print(f"[download] {repo_id} -> {dest}")
    # local_dir copies real files into models/ (no cache symlinks), so the folder
    # is self-contained and matches the other model dirs.
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest),
        force_download=force,
    )
    print(f"[done] {repo_id} -> {dest}")
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "repo_ids",
        nargs="+",
        help="One or more HF repo ids, e.g. lair-nyu/cubeinbox-uniform-pi0-20-010000",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Local folder name to use under models/ (only valid with a single repo id).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the destination folder already exists.",
    )
    args = parser.parse_args()

    if args.name and len(args.repo_ids) > 1:
        parser.error("--name can only be used with a single repo id")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for repo_id in args.repo_ids:
        download(repo_id, name=args.name, force=args.force)

    return 0


if __name__ == "__main__":
    sys.exit(main())
