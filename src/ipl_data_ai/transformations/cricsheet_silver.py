
"""Transform Cricsheet Bronze JSON into foundational Silver tables."""

import json
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

BRONZE_MATCHES = Path("data/bronze/cricsheet/matches")
SILVER_ROOT = Path("data/silver/cricsheet")

MATCH_SCHEMA = StructType(
    [
        StructField("match_id", StringType(), False),
        StructField("match_date", StringType(), True),
        StructField("season", StringType(), True),
        StructField("city", StringType(), True),
        StructField("venue", StringType(), True),
        StructField("team_1", StringType(), True),
        StructField("team_2", StringType(), True),
        StructField("toss_winner", StringType(), True),
        StructField("toss_decision", StringType(), True),
        StructField("winner", StringType(), True),
        StructField("outcome_type", StringType(), True),
        StructField("margin_runs", IntegerType(), True),
        StructField("margin_wickets", IntegerType(), True),
        StructField("player_of_match", StringType(), True),
    ]
)

INNINGS_SCHEMA = StructType(
    [
        StructField("match_id", StringType(), False),
        StructField("innings_number", IntegerType(), False),
        StructField("batting_team", StringType(), True),
        StructField("is_super_over", BooleanType(), True),
        StructField("target_runs", IntegerType(), True),
        StructField("target_overs", IntegerType(), True),
        StructField("declared_overs", IntegerType(), True),
        StructField("source_over_count", IntegerType(), False),
        StructField("source_delivery_count", IntegerType(), False),
    ]
)

DELIVERY_SCHEMA = StructType(
    [
        StructField("match_id", StringType(), False),
        StructField("innings_number", IntegerType(), False),
        StructField("over_number", IntegerType(), False),
        StructField("delivery_number", StringType(), False),
        StructField("delivery_sequence", IntegerType(), False),
        StructField("batting_team", StringType(), True),
        StructField("batter", StringType(), True),
        StructField("bowler", StringType(), True),
        StructField("non_striker", StringType(), True),
        StructField("batter_runs", IntegerType(), False),
        StructField("extras_runs", IntegerType(), False),
        StructField("total_runs", IntegerType(), False),
        StructField("is_wide", BooleanType(), False),
        StructField("is_no_ball", BooleanType(), False),
        StructField("byes", IntegerType(), False),
        StructField("legbyes", IntegerType(), False),
        StructField("penalty_runs", IntegerType(), False),
        StructField("is_wicket", BooleanType(), False),
        StructField("wicket_types", StringType(), True),
        StructField("players_out", StringType(), True),
        StructField("is_super_over", BooleanType(), False),
    ]
)


def _read_source_matches(
    bronze_dir: Path,
) -> list[tuple[str, dict]]:
    """Read JSON files in stable filename order."""
    if not bronze_dir.is_dir():
        raise FileNotFoundError(
            f"Bronze match directory not found: {bronze_dir}"
        )

    paths = sorted(bronze_dir.glob("*.json"))
    if not paths:
        raise RuntimeError(f"No match JSON files found in {bronze_dir}")

    return [
        (path.stem, json.loads(path.read_text(encoding="utf-8")))
        for path in paths
    ]


def _outcome_fields(outcome: dict) -> tuple:
    """Extract the match result, winner, and winning margin."""
    winner = outcome.get("winner")
    by = outcome.get("by") or {}

    if winner:
        if "runs" in by:
            return winner, "winner", int(by["runs"]), None
        if "wickets" in by:
            return winner, "winner", None, int(by["wickets"])
        if outcome.get("eliminator"):
            return winner, "eliminator", None, None
        return winner, "winner", None, None

    if outcome.get("result") == "tie":
        return None, "tie", None, None

    if outcome.get("result") == "no result":
        return None, "no result", None, None

    return None, outcome.get("result"), None, None


def build_matches(
    spark: SparkSession,
    bronze_dir: Path = BRONZE_MATCHES,
) -> DataFrame:
    """Build one row per source match."""
    rows = []

    for match_id, data in _read_source_matches(bronze_dir):
        info = data["info"]
        toss = info.get("toss") or {}
        outcome = info.get("outcome") or {}

        winner, outcome_type, margin_runs, margin_wickets = (
            _outcome_fields(outcome)
        )

        teams = info.get("teams") or []
        dates = info.get("dates") or []

        rows.append(
            (
                match_id,
                str(dates[0]) if dates else None,
                str(info.get("season"))
                if info.get("season") is not None
                else None,
                info.get("city"),
                info.get("venue"),
                teams[0] if len(teams) > 0 else None,
                teams[1] if len(teams) > 1 else None,
                toss.get("winner"),
                toss.get("decision"),
                winner,
                outcome_type,
                margin_runs,
                margin_wickets,
                (info.get("player_of_match") or [None])[0],
            )
        )

    return spark.createDataFrame(rows, schema=MATCH_SCHEMA).withColumn(
        "match_date", F.to_date("match_date")
    )


def build_innings(
    spark: SparkSession,
    bronze_dir: Path = BRONZE_MATCHES,
) -> DataFrame:
    """Build one row per innings, retaining source innings order."""
    rows = []

    for match_id, data in _read_source_matches(bronze_dir):
        for innings_number, innings in enumerate(data["innings"], start=1):
            target = innings.get("target") or {}
            overs = innings.get("overs") or []

            rows.append(
                (
                    match_id,
                    innings_number,
                    innings.get("team"),
                    innings.get("super_over", False),
                    int(target["runs"]) if "runs" in target else None,
                    int(target["overs"]) if "overs" in target else None,
                    int(data["info"].get("overs") or 0)
                    if data["info"].get("overs") is not None
                    else None,
                    len(overs),
                    sum(len(over.get("deliveries") or []) for over in overs),
                )
            )

    return spark.createDataFrame(rows, schema=INNINGS_SCHEMA)


def build_deliveries(
    spark: SparkSession,
    bronze_dir: Path = BRONZE_MATCHES,
) -> DataFrame:
    """Build one row per source delivery in innings order."""
    rows = []

    for match_id, data in _read_source_matches(bronze_dir):
        for innings_number, innings in enumerate(data["innings"], start=1):
            batting_team = innings.get("team")
            is_super_over = bool(innings.get("super_over", False))
            delivery_sequence = 0

            for over in innings.get("overs") or []:
                over_number = int(over["over"])

                for delivery in over.get("deliveries") or []:
                    delivery_sequence += 1

                    extras = delivery.get("extras") or {}
                    runs = delivery.get("runs") or {}
                    wickets = delivery.get("wickets") or []

                    wicket_types = [
                        str(wicket["kind"])
                        for wicket in wickets
                        if wicket.get("kind") is not None
                    ]
                    players_out = [
                        str(wicket["player_out"])
                        for wicket in wickets
                        if wicket.get("player_out") is not None
                    ]

                    actual_delivery = delivery.get("actual_delivery")

                    rows.append(
                        (
                            match_id,
                            innings_number,
                            over_number,
                            str(actual_delivery)
                            if actual_delivery is not None
                            else str(delivery_sequence),
                            delivery_sequence,
                            batting_team,
                            delivery.get("batter"),
                            delivery.get("bowler"),
                            delivery.get("non_striker"),
                            int(runs.get("batter", 0)),
                            int(runs.get("extras", 0)),
                            int(runs.get("total", 0)),
                            int(extras.get("wides", 0)) > 0,
                            int(extras.get("noballs", 0)) > 0,
                            int(extras.get("byes", 0)),
                            int(extras.get("legbyes", 0)),
                            int(extras.get("penalty", 0)),
                            bool(wickets),
                            "|".join(wicket_types) or None,
                            "|".join(players_out) or None,
                            is_super_over,
                        )
                    )

    return spark.createDataFrame(rows, schema=DELIVERY_SCHEMA)


def validate_deliveries(
    deliveries: DataFrame,
    innings: DataFrame,
    expected_delivery_count: int = 295732,
) -> None:
    """Validate delivery counts, keys, values, and innings references."""
    delivery_count = deliveries.count()

    if delivery_count != expected_delivery_count:
        raise ValueError(
            f"Expected {expected_delivery_count} deliveries; "
            f"found {delivery_count}"
        )

    duplicate_keys = (
        deliveries.groupBy(
            "match_id",
            "innings_number",
            "delivery_sequence",
        )
        .count()
        .filter(F.col("count") > 1)
        .count()
    )
    if duplicate_keys:
        raise ValueError(f"Found {duplicate_keys} duplicate delivery keys")

    unmatched_deliveries = deliveries.join(
        innings.select("match_id", "innings_number"),
        on=["match_id", "innings_number"],
        how="left_anti",
    ).count()
    if unmatched_deliveries:
        raise ValueError(
            f"Found {unmatched_deliveries} deliveries without a matching "
            "innings"
        )

    invalid_runs = deliveries.filter(
        F.col("batter_runs").isNull()
        | (F.col("batter_runs") < 0)
        | F.col("extras_runs").isNull()
        | (F.col("extras_runs") < 0)
        | F.col("total_runs").isNull()
        | (F.col("total_runs") < 0)
        | (
            F.col("total_runs")
            != F.col("batter_runs") + F.col("extras_runs")
        )
    ).count()

    if invalid_runs:
        raise ValueError(f"Found {invalid_runs} deliveries with invalid runs")

    invalid_keys = deliveries.filter(
        F.col("innings_number").isNull()
        | (F.col("innings_number") < 1)
        | F.col("over_number").isNull()
        | (F.col("over_number") < 0)
        | F.col("delivery_sequence").isNull()
        | (F.col("delivery_sequence") < 1)
    ).count()

    if invalid_keys:
        raise ValueError(f"Found {invalid_keys} deliveries with invalid keys")


def validate_foundation(
    matches: DataFrame,
    innings: DataFrame,
    expected_match_count: int = 1243,
) -> None:
    """Fail fast when foundational Silver data violates core checks."""
    match_count = matches.count()
    if match_count != expected_match_count:
        raise ValueError(
            f"Expected {expected_match_count} matches; found {match_count}"
        )

    duplicate_matches = (
        matches.groupBy("match_id")
        .count()
        .filter(F.col("count") > 1)
        .count()
    )
    if duplicate_matches:
        raise ValueError(f"Found {duplicate_matches} duplicate match IDs")

    unmatched_innings = innings.join(
        matches.select("match_id"),
        on="match_id",
        how="left_anti",
    ).count()
    if unmatched_innings:
        raise ValueError(
            f"Found {unmatched_innings} innings without a matching match"
        )

    duplicate_innings = (
        innings.groupBy("match_id", "innings_number")
        .count()
        .filter(F.col("count") > 1)
        .count()
    )
    if duplicate_innings:
        raise ValueError(
            f"Found {duplicate_innings} duplicate innings keys"
        )

    invalid_innings = innings.filter(
        F.col("innings_number").isNull()
        | (F.col("innings_number") < 1)
        | F.col("source_over_count").isNull()
        | (F.col("source_over_count") < 0)
        | F.col("source_delivery_count").isNull()
        | (F.col("source_delivery_count") < 0)
    ).count()
    if invalid_innings:
        raise ValueError(f"Found {invalid_innings} invalid innings rows")


def main() -> None:
    """Build and validate foundational Silver tables."""
    spark = (
        SparkSession.builder
        .appName("IPL-Cricsheet-Silver-Foundation")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        matches = build_matches(spark)
        innings = build_innings(spark)
        deliveries = build_deliveries(spark)

        validate_foundation(matches, innings)
        validate_deliveries(deliveries, innings)

        matches_path = SILVER_ROOT / "matches"
        innings_path = SILVER_ROOT / "innings"
        deliveries_path = SILVER_ROOT / "deliveries"

        matches.write.mode("overwrite").parquet(str(matches_path))
        innings.write.mode("overwrite").parquet(str(innings_path))
        deliveries.write.mode("overwrite").parquet(str(deliveries_path))

        print("=== SILVER FOUNDATION ===")
        print("Matches:", matches.count())
        print("Innings:", innings.count())
        print("Super Over innings:", innings.filter("is_super_over").count())
        print("Deliveries:", deliveries.count())
        print("Deliveries with wickets:", deliveries.filter("is_wicket").count())
        print("Deliveries output:", deliveries_path)
        print("Validation: passed")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()