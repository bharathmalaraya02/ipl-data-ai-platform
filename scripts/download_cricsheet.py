"""Download and validate the official Cricsheet IPL JSON archive."""

from __future__ import annotations

import hashlib
import json
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import requests

SOURCE_URL = "https://cricsheet.org/downloads/ipl_json.zip"
OUTPUT_DIR = Path("data/raw/cricsheet")
ARCHIVE_PATH = OUTPUT_DIR / "ipl_json.zip"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"


def sha256_file(path: Path) -> str:
    """Calculate a file's SHA-256 checksum without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            dir=OUTPUT_DIR, suffix=".part", delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)
            downloaded_bytes = 0

            with requests.get(
                SOURCE_URL,
                stream=True,
                timeout=(20, 120),
                headers={"User-Agent": "ipl-data-ai-platform/0.1"},
            ) as response:
                response.raise_for_status()
                expected_size = response.headers.get("Content-Length")

                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        temp_file.write(chunk)
                        downloaded_bytes += len(chunk)

            if expected_size and downloaded_bytes != int(expected_size):
                raise ValueError(
                    f"Incomplete download: expected {expected_size} bytes, "
                    f"received {downloaded_bytes}"
                )

        checksum = sha256_file(temp_path)

        with zipfile.ZipFile(temp_path) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError(f"Corrupt ZIP member: {bad_member}")

            members = [
                item for item in archive.infolist()
                if not item.is_dir()
            ]
            json_files = [
                item.filename for item in members
                if item.filename.lower().endswith(".json")
            ]

            if not json_files:
                raise ValueError("Archive contains no JSON files")

        if ARCHIVE_PATH.exists():
            existing_checksum = sha256_file(ARCHIVE_PATH)
            if existing_checksum != checksum:
                raise FileExistsError(
                    f"{ARCHIVE_PATH} already exists with a different checksum. "
                    "The existing archive was preserved; move it to a backup "
                    "location before intentionally replacing it."
                )
            temp_path.unlink()
            temp_path = None
            print("Existing archive matches the downloaded checksum.")
        else:
            temp_path.replace(ARCHIVE_PATH)
            temp_path = None

        manifest = {
            "source": "Cricsheet",
            "source_url": SOURCE_URL,
            "retrieved_at_utc": datetime.now(UTC).isoformat(),
            "archive_path": str(ARCHIVE_PATH),
            "archive_size_bytes": ARCHIVE_PATH.stat().st_size,
            "sha256": checksum,
            "zip_members": len(members),
            "json_file_count": len(json_files),
            "validation": "passed",
        }

        MANIFEST_PATH.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

        print("Download and validation successful.")
        print(f"Archive: {ARCHIVE_PATH}")
        print(f"Size: {manifest['archive_size_bytes']:,} bytes")
        print(f"JSON files: {manifest['json_file_count']:,}")
        print(f"ZIP members: {manifest['zip_members']:,}")
        print(f"SHA-256: {checksum}")
        print(f"Manifest: {MANIFEST_PATH}")

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    download_archive()
