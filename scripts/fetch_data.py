"""Download the competition data from Kaggle.

The raw CSVs are not committed. They are licensed CC BY 4.0, so redistribution
would be allowed, but a repository is easier to trust when the data it analyses
comes straight from the source: anyone can run this and get byte-identical
inputs, and the repo stays small.

Requires a Kaggle API token at ~/.kaggle/kaggle.json and the competition rules
accepted on the competition page — Kaggle refuses the download otherwise.
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPETITION = "playground-series-s6e9"
EXPECTED = ("train.csv", "test.csv", "sample_submission.csv")


def credentials_present() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition", default=COMPETITION)
    parser.add_argument("--out", type=Path, default=ROOT / "data")
    args = parser.parse_args()

    if not credentials_present():
        sys.exit(
            "No Kaggle credentials found.\n"
            "  1. https://www.kaggle.com/settings -> API -> Create New Token\n"
            "  2. mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/\n"
            "  3. chmod 600 ~/.kaggle/kaggle.json"
        )

    args.out.mkdir(parents=True, exist_ok=True)

    # Imported here so the credential message above is reached first: the Kaggle
    # client authenticates at import time and would raise a less useful error.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    print(f"downloading {args.competition} -> {args.out}")
    api.competition_download_files(args.competition, path=str(args.out), quiet=False)

    archive = args.out / f"{args.competition}.zip"
    if archive.exists():
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(args.out)
        archive.unlink()

    missing = [name for name in EXPECTED if not (args.out / name).exists()]
    if missing:
        sys.exit(f"expected files not found after download: {', '.join(missing)}\n"
                 f"Have you accepted the rules at "
                 f"https://www.kaggle.com/competitions/{args.competition}/rules ?")

    for name in EXPECTED:
        path = args.out / name
        print(f"  {name:24} {path.stat().st_size / 1e6:6.1f} MB")


if __name__ == "__main__":
    main()
