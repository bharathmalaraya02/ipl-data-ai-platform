"""Audit the Cricsheet IPL JSON archive without extracting it."""

import json
import zipfile
from collections import Counter
from datetime import date
from pathlib import PurePosixPath

ARCHIVE_PATH = "data/raw/cricsheet/ipl_json.zip"


def main() -> None:
    seasons = Counter()
    teams = Counter()
    match_types = Counter()
    match_dates = []
    innings_counts = Counter()
    outcomes = Counter()
    issues = []
    match_ids = []
    total_overs = 0
    total_deliveries = 0

    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        json_files = sorted(
            name
            for name in archive.namelist()
            if name.lower().endswith(".json")
            and not name.endswith("/")
        )

        for name in json_files:
            match_id = PurePosixPath(name).stem
            match_ids.append(match_id)

            try:
                data = json.loads(archive.read(name))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                issues.append(f"{name}: invalid JSON ({exc})")
                continue

            if not all(
                key in data for key in ("meta", "info", "innings")
            ):
                issues.append(f"{name}: missing expected top-level key")
                continue

            info = data["info"]
            innings = data["innings"]

            if not isinstance(info, dict) or not isinstance(innings, list):
                issues.append(f"{name}: unexpected info/innings type")
                continue

            seasons[str(info.get("season", "UNKNOWN"))] += 1
            match_types[str(info.get("match_type", "UNKNOWN"))] += 1

            for team in info.get("teams", []):
                teams[team] += 1

            for value in info.get("dates", []):
                try:
                    match_dates.append(date.fromisoformat(str(value)))
                except ValueError:
                    issues.append(f"{name}: invalid date {value!r}")

            outcome = info.get("outcome", {})
            if isinstance(outcome, dict):
                if "winner" in outcome:
                    outcomes["winner recorded"] += 1
                elif "result" in outcome:
                    outcomes[str(outcome["result"])] += 1
                else:
                    outcomes["no winner/result recorded"] += 1
            else:
                outcomes["outcome missing or unexpected"] += 1

            innings_counts[len(innings)] += 1

            for inning in innings:
                if not isinstance(inning, dict):
                    issues.append(f"{name}: invalid innings entry")
                    continue

                for over in inning.get("overs", []):
                    total_overs += 1
                    deliveries = over.get("deliveries", [])
                    total_deliveries += len(deliveries)

                    for delivery in deliveries:
                        if not isinstance(delivery, dict):
                            issues.append(
                                f"{name}: invalid delivery entry"
                            )
                        elif "runs" not in delivery:
                            issues.append(
                                f"{name}: delivery missing runs"
                            )

    duplicate_ids = [
        match_id
        for match_id, count in Counter(match_ids).items()
        if count > 1
    ]

    print("=== CRICSHEET IPL ARCHIVE AUDIT ===")
    print("JSON files:", len(json_files))
    print("Unique filename IDs:", len(set(match_ids)))
    print("Duplicate filename IDs:", len(duplicate_ids))
    print("Seasons:", dict(sorted(seasons.items())))
    print("Match types:", dict(match_types))
    print("Teams:", dict(sorted(teams.items())))
    print(
        "Earliest match date:",
        min(match_dates) if match_dates else "N/A",
    )
    print(
        "Latest match date:",
        max(match_dates) if match_dates else "N/A",
    )
    print("Innings per match:", dict(sorted(innings_counts.items())))
    print("Outcome categories:", dict(outcomes))
    print("Total overs:", total_overs)
    print("Total deliveries:", total_deliveries)
    print("Validation issues:", len(issues))

    for issue in issues[:20]:
        print(" -", issue)

    if len(issues) > 20:
        print(f" ... and {len(issues) - 20} more")


if __name__ == "__main__":
    main()
