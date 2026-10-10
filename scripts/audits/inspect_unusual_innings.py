"""Inspect IPL matches with unusual numbers of innings."""

import json
import zipfile
from pathlib import PurePosixPath

ARCHIVE_PATH = "data/raw/cricsheet/ipl_json.zip"
UNUSUAL_INNINGS_COUNTS = {1, 4, 6}


def main() -> None:
    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        json_files = sorted(
            name
            for name in archive.namelist()
            if name.lower().endswith(".json")
            and not name.endswith("/")
        )

        for name in json_files:
            data = json.loads(archive.read(name))
            info = data["info"]
            innings = data["innings"]

            if len(innings) not in UNUSUAL_INNINGS_COUNTS:
                continue

            print("=" * 72)
            print("Match ID:", PurePosixPath(name).stem)
            print("Season:", info.get("season"))
            print("Date:", info.get("dates"))
            print("Teams:", info.get("teams"))
            print("Venue:", info.get("venue"))
            print("Outcome:", info.get("outcome"))
            print("Overs scheduled:", info.get("overs"))

            for index, inning in enumerate(innings, start=1):
                overs = inning.get("overs", [])
                delivery_count = sum(
                    len(over.get("deliveries", []))
                    for over in overs
                )

                print(
                    f"  Innings {index}: "
                    f"team={inning.get('team')!r}, "
                    f"overs={len(overs)}, "
                    f"deliveries={delivery_count}, "
                    f"super_over={inning.get('super_over', False)}"
                )


if __name__ == "__main__":
    main()
