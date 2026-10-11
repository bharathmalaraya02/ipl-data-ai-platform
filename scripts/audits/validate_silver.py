import json
import sys
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BRONZE_DIR = PROJECT_ROOT / "data/bronze/cricsheet/matches"
SILVER_DIR = PROJECT_ROOT / "data/silver/cricsheet"

MATCHES_PATH = str(SILVER_DIR / "matches/**/*.parquet")
INNINGS_PATH = str(SILVER_DIR / "innings/**/*.parquet")
DELIVERIES_PATH = str(SILVER_DIR / "deliveries/**/*.parquet")


def source_totals():
    """Calculate expected totals directly from Bronze JSON."""
    totals = {
        "matches": 0,
        "innings": 0,
        "deliveries": 0,
        "batter_runs": 0,
        "extras_runs": 0,
        "total_runs": 0,
        "byes": 0,
        "legbyes": 0,
        "penalty_runs": 0,
        "wide_deliveries": 0,
        "no_ball_deliveries": 0,
        "wicket_deliveries": 0,
        "super_over_deliveries": 0,
    }
    match_ids = set()
    innings_keys = set()
    delivery_counts = {}

    for path in sorted(BRONZE_DIR.glob("*.json")):
        with path.open(encoding="utf-8") as file:
            match = json.load(file)

        match_id = str(match["info"].get("event", {}).get("match_number", ""))
        # Cricsheet's match ID is the JSON filename, not match_number.
        match_id = path.stem

        if match_id in match_ids:
            raise ValueError(f"Duplicate Bronze match ID: {match_id}")
        match_ids.add(match_id)
        totals["matches"] += 1

        for innings_number, innings in enumerate(
            match.get("innings", []), start=1
        ):
            key = (match_id, innings_number)
            innings_keys.add(key)
            totals["innings"] += 1
            delivery_counts[key] = 0
            is_super_over = bool(innings.get("super_over", False))

            for over in innings.get("overs", []):
                for delivery in over.get("deliveries", []):
                    totals["deliveries"] += 1
                    delivery_counts[key] += 1

                    runs = delivery.get("runs", {})
                    extras = delivery.get("extras", {})
                    batter_runs = runs.get("batter", 0)
                    extras_runs = runs.get("extras", 0)

                    totals["batter_runs"] += batter_runs
                    totals["extras_runs"] += extras_runs
                    totals["total_runs"] += runs.get("total", 0)
                    totals["byes"] += extras.get("byes", 0)
                    totals["legbyes"] += extras.get("legbyes", 0)
                    totals["penalty_runs"] += extras.get("penalty", 0)
                    totals["wide_deliveries"] += int(
                        extras.get("wides", 0) > 0
                    )
                    totals["no_ball_deliveries"] += int(
                        extras.get("noballs", 0) > 0
                    )
                    totals["wicket_deliveries"] += int(
                        bool(delivery.get("wickets"))
                    )
                    totals["super_over_deliveries"] += int(is_super_over)

    return totals, match_ids, innings_keys, delivery_counts


def run_audit():
    failures = []
    checks = []

    def check(name, passed, details=""):
        checks.append((name, bool(passed), details))
        if not passed:
            failures.append(name)

    if not BRONZE_DIR.is_dir():
        print(f"ERROR: Bronze directory not found: {BRONZE_DIR}")
        return 1

    for path in (MATCHES_PATH, INNINGS_PATH, DELIVERIES_PATH):
        if not list(Path(path.split("/**")[0]).glob("*.parquet")):
            print(f"ERROR: No Parquet files found under {path}")
            return 1

    expected, source_match_ids, source_innings_keys, source_delivery_counts = (
        source_totals()
    )

    con = duckdb.connect()
    try:
        matches = con.execute(
            "SELECT * FROM read_parquet(?)", [MATCHES_PATH]
        ).fetchdf()
        innings = con.execute(
            "SELECT * FROM read_parquet(?)", [INNINGS_PATH]
        ).fetchdf()
        deliveries = con.execute(
            "SELECT * FROM read_parquet(?)", [DELIVERIES_PATH]
        ).fetchdf()
    finally:
        con.close()

    # Volume and key checks.
    check("Match count", len(matches) == expected["matches"],
          f"Bronze={expected['matches']}, Silver={len(matches)}")
    check("Innings count", len(innings) == expected["innings"],
          f"Bronze={expected['innings']}, Silver={len(innings)}")
    check("Delivery count", len(deliveries) == expected["deliveries"],
          f"Bronze={expected['deliveries']}, Silver={len(deliveries)}")

    silver_match_ids = set(matches["match_id"].astype(str))
    silver_innings_keys = set(
        zip(innings["match_id"].astype(str), innings["innings_number"])
    )

    check("Match IDs complete", silver_match_ids == source_match_ids)
    check("Innings keys complete", silver_innings_keys == source_innings_keys)

    check("Duplicate match IDs", matches["match_id"].is_unique)
    check(
        "Duplicate innings keys",
        not innings.duplicated(["match_id", "innings_number"]).any(),
    )
    check(
        "Duplicate delivery keys",
        not deliveries.duplicated(
            ["match_id", "innings_number", "delivery_sequence"]
        ).any(),
    )

    innings_delivery_counts = (
        deliveries.groupby(["match_id", "innings_number"])
        .size()
        .to_dict()
    )
    silver_delivery_keys = set(innings_delivery_counts)
    check(
        "Delivery groups complete",
        silver_delivery_keys == source_innings_keys,
        f"Missing={len(source_innings_keys - silver_delivery_keys)}, "
        f"unexpected={len(silver_delivery_keys - source_innings_keys)}",
    )

    mismatched_counts = []
    for key, expected_count in source_delivery_counts.items():
        actual_count = innings_delivery_counts.get(key, 0)
        if actual_count != expected_count:
            mismatched_counts.append((key, expected_count, actual_count))

    check(
        "Per-innings delivery counts",
        not mismatched_counts,
        f"Mismatches={len(mismatched_counts)}",
    )

    # Run arithmetic and field quality.
    check(
        "Run arithmetic",
        (
            deliveries["batter_runs"] + deliveries["extras_runs"]
            == deliveries["total_runs"]
        ).all(),
    )
    run_columns = ["batter_runs", "extras_runs", "total_runs"]
    check(
        "No negative runs",
        not (deliveries[run_columns] < 0).any().any(),
    )
    required_columns = [
        "match_id", "batting_team", "batter", "bowler", "non_striker"
    ]
    check(
        "Required fields populated",
        not deliveries[required_columns].isna().any().any()
        and not (deliveries[required_columns].astype(str) == "").any().any(),
    )
    check(
        "No orphan innings",
        set(zip(
            deliveries["match_id"].astype(str),
            deliveries["innings_number"],
        )).issubset(silver_innings_keys),
    )

    # Reconcile source aggregates with Silver.
    aggregate_columns = {
        "batter_runs": "batter_runs",
        "extras_runs": "extras_runs",
        "total_runs": "total_runs",
        "byes": "byes",
        "legbyes": "legbyes",
        "penalty_runs": "penalty_runs",
    }

    for source_name, column in aggregate_columns.items():
        actual = int(deliveries[column].sum())
        check(
            f"Aggregate: {source_name}",
            actual == expected[source_name],
            f"Bronze={expected[source_name]}, Silver={actual}",
        )

    check(
        "Wide delivery count",
        int(deliveries["is_wide"].sum()) == expected["wide_deliveries"],
    )
    check(
        "No-ball delivery count",
        int(deliveries["is_no_ball"].sum()) == expected["no_ball_deliveries"],
    )
    check(
        "Wicket delivery count",
        int(deliveries["is_wicket"].sum()) == expected["wicket_deliveries"],
    )
    check(
        "Super Over delivery count",
        int(deliveries["is_super_over"].sum())
        == expected["super_over_deliveries"],
    )

    print("\n=== SILVER QUALITY AUDIT ===")
    for name, passed, details in checks:
        status = "PASS" if passed else "FAIL"
        suffix = f" — {details}" if details else ""
        print(f"[{status}] {name}{suffix}")

    print(f"\nChecks passed: {len(checks) - len(failures)}/{len(checks)}")
    print(f"Checks failed: {len(failures)}")

    if failures:
        print("Failed checks:", ", ".join(failures))
        return 1

    print("Overall result: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(run_audit())
