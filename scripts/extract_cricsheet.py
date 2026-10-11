"""Safely extract and validate Cricsheet IPL JSON into the Bronze layer."""

import hashlib
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

ARCHIVE_PATH = Path("data/raw/cricsheet/ipl_json.zip")
BRONZE_ROOT = Path("data/bronze/cricsheet")
MATCHES_DIR = BRONZE_ROOT / "matches"
DOCS_DIR = BRONZE_ROOT / "source_docs"
MANIFEST_PATH = BRONZE_ROOT / "extraction_manifest.json"


def sha256_bytes(content: bytes) -> str:
    """Return the SHA-256 checksum of bytes."""
    return hashlib.sha256(content).hexdigest()


def write_unchanged_or_new(path: Path, content: bytes) -> None:
    """Write a file, refusing to overwrite different existing content."""
    if path.exists():
        existing_hash = sha256_bytes(path.read_bytes())
        incoming_hash = sha256_bytes(content)

        if existing_hash != incoming_hash:
            raise RuntimeError(
                f"Existing file differs from archive: {path}. "
                "No overwrite was performed."
            )
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def main() -> None:
    if not ARCHIVE_PATH.is_file():
        raise FileNotFoundError(
            f"Source archive not found: {ARCHIVE_PATH}"
        )

    archive_bytes = ARCHIVE_PATH.read_bytes()
    archive_sha256 = sha256_bytes(archive_bytes)

    extracted_files = []
    match_ids = set()
    total_bytes = 0

    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        corrupt_member = archive.testzip()
        if corrupt_member is not None:
            raise RuntimeError(
                f"Corrupt ZIP member detected: {corrupt_member}"
            )

        members = [
            member
            for member in archive.infolist()
            if not member.is_dir()
        ]

        
        json_members = sorted(
            (
                member
                for member in members
                if member.filename.lower().endswith(".json")
            ),
            key=lambda member: member.filename,
        )

        if not json_members:
            raise RuntimeError("No JSON match files found in archive.")

        for member in json_members:
            member_path = PurePosixPath(member.filename)

            if (
                member_path.is_absolute()
                or len(member_path.parts) != 1
                or ".." in member_path.parts
            ):
                raise RuntimeError(
                    f"Unexpected JSON archive path: {member.filename}"
                )

            filename = member_path.name
            match_id = member_path.stem

            if match_id in match_ids:
                raise RuntimeError(f"Duplicate match ID: {match_id}")
            match_ids.add(match_id)

            content = archive.read(member)
            data = json.loads(content)

            if not isinstance(data, dict):
                raise TypeError(f"{filename}: expected a JSON object")

            if not all(key in data for key in ("meta", "info", "innings")):
                raise TypeError(
                    f"{filename}: missing meta, info, or innings"
                )

            if (
                not isinstance(data["meta"], dict)
                or not isinstance(data["info"], dict)
                or not isinstance(data["innings"], list)
            ):
                raise TypeError(
                    f"{filename}: invalid top-level field types"
                )

            output_path = MATCHES_DIR / filename
            write_unchanged_or_new(output_path, content)

            extracted_files.append(
                {
                    "filename": filename,
                    "match_id": match_id,
                    "size_bytes": len(content),
                    "sha256": sha256_bytes(content),
                }
            )
            total_bytes += len(content)

        # Preserve the archive's README and license as source documentation.
        for member in members:
            member_path = PurePosixPath(member.filename)

            if member_path.name.upper() not in {"README.TXT", "LICENSE.TXT"}:
                continue

            if len(member_path.parts) != 1:
                raise RuntimeError(
                    f"Unexpected documentation path: {member.filename}"
                )

            content = archive.read(member)
            write_unchanged_or_new(
                DOCS_DIR / member_path.name.lower(),
                content,
            )

    manifest = {
        "source": "Cricsheet",
        "source_archive": str(ARCHIVE_PATH),
        "source_archive_sha256": archive_sha256,
        "extracted_at_utc": datetime.now(UTC).isoformat(),
        "bronze_matches_directory": str(MATCHES_DIR),
        "json_file_count": len(extracted_files),
        "unique_match_id_count": len(match_ids),
        "total_json_bytes": total_bytes,
        "validation": "passed",
        "files": extracted_files,
    }

    BRONZE_ROOT.mkdir(parents=True, exist_ok=True)
    temporary_manifest = MANIFEST_PATH.with_suffix(".tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(MANIFEST_PATH)

    print("=== CRICSHEET BRONZE EXTRACTION ===")
    print("Source archive:", ARCHIVE_PATH)
    print("Source SHA-256:", archive_sha256)
    print("JSON files extracted/verified:", len(extracted_files))
    print("Unique match IDs:", len(match_ids))
    print("Total JSON bytes:", total_bytes)
    print("Validation: passed")
    print("Manifest:", MANIFEST_PATH)


if __name__ == "__main__":
    main()
