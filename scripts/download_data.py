"""Download the official Approved Document B PDFs from gov.uk.

The PDFs are Crown copyright, published under the Open Government Licence v3.0.
They are not committed to the repository; run this script to fetch them.

Usage:
    python scripts/download_data.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import requests

SOURCE_PAGE = "https://www.gov.uk/government/publications/fire-safety-approved-document-b"

# Official gov.uk asset URLs, as linked from SOURCE_PAGE (last updated 16 September 2026).
DOCUMENTS: dict[str, str] = {
    "AD_B_vol1_dwellings.pdf": (
        "https://assets.publishing.service.gov.uk/media/6aa3bb335f6e942efe37f150/AD_B1_2026.pdf"
    ),
    "AD_B_vol2_non_dwellings.pdf": (
        "https://assets.publishing.service.gov.uk/media/6aa3bb7492e72b8ac437f145/AD_B2_2026.pdf"
    ),
}

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def download(url: str, dest: Path) -> str:
    """Stream ``url`` to ``dest`` and return its SHA-256 hex digest."""
    sha = hashlib.sha256()
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with dest.open("wb") as fh:
            for block in resp.iter_content(chunk_size=1 << 16):
                fh.write(block)
                sha.update(block)
    return sha.hexdigest()


def main() -> int:
    """Download every document that is not already present."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Source: {SOURCE_PAGE}\nLicence: Open Government Licence v3.0\n")
    for name, url in DOCUMENTS.items():
        dest = RAW_DIR / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[skip] {name} already present")
            continue
        print(f"[get ] {name} <- {url}")
        digest = download(url, dest)
        print(f"       {dest.stat().st_size / 1e6:.2f} MB  sha256={digest[:16]}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())
