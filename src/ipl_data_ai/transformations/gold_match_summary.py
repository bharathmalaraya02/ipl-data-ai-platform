
"""Build a match-level Gold summary from Silver Parquet datasets."""

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

SILVER_ROOT = Path("data/silver/cricsheet")
GOLD_ROOT = Path("data/gold/cricsheet")


def build_match_summary(
    matches: DataFrame,
    innings: DataFrame,
    deliveries: DataFrame,
) -> DataFrame:
    """Return one analytics-ready row per match."""

    # Exclude Super Overs from the regular match summary.
    main_innings = (
        innings.filter(~F.col("is_super_over"))
        .select(
            "match_id",
            "innings_number",
            F.col("batting_team").alias("innings_batting_team"),
        )
    )

    # Aggregate deliveries separately to avoid ambiguous column names.
    delivery_stats = (
        deliveries.groupBy("match_id", "innings_number")
        .agg(
            F.coalesce(F.sum("total_runs"), F.lit(0)).alias("innings_runs"),
            F.coalesce(
                F.sum(
                    F.when(
                        F.col("players_out").isNotNull()
                        & (F.trim(F.col("players_out")) != ""),
                        1,
                    ).otherwise(0)
                ),
                F.lit(0),
            ).alias("dismissal_count"),
            F.coalesce(
                F.sum(
                    F.when(
                        ~F.col("is_wide") & ~F.col("is_no_ball"),
                        1,
                    ).otherwise(0)
                ),
                F.lit(0),
            ).alias("legal_balls"),
        )
    )

    # Join innings metadata to the pre-aggregated delivery statistics.
    innings_scores = (
        main_innings.join(
            delivery_stats,
            on=["match_id", "innings_number"],
            how="left",
        )
        .fillna(
            {
                "innings_runs": 0,
                "dismissal_count": 0,
                "legal_balls": 0,
            }
        )
    )

    # Identify the first two regular innings.
    first_innings = (
        innings_scores.filter(F.col("innings_number") == 1)
        .select(
            "match_id",
            F.col("innings_batting_team").alias("team_1_batting_team"),
            F.col("innings_runs").alias("innings_1_runs"),
            F.col("dismissal_count").alias("innings_1_wickets"),
            F.col("legal_balls").alias("innings_1_legal_balls"),
        )
    )

    second_innings = (
        innings_scores.filter(F.col("innings_number") == 2)
        .select(
            "match_id",
            F.col("innings_batting_team").alias("team_2_batting_team"),
            F.col("innings_runs").alias("innings_2_runs"),
            F.col("dismissal_count").alias("innings_2_wickets"),
            F.col("legal_balls").alias("innings_2_legal_balls"),
        )
    )

    # Map scores to the actual team names; team_1 does not always bat first.
    result = (
        matches.join(first_innings, on="match_id", how="left")
        .join(second_innings, on="match_id", how="left")
        .withColumn(
            "team_1_runs",
            F.when(
                F.col("team_1") == F.col("team_1_batting_team"),
                F.col("innings_1_runs"),
            ).when(
                F.col("team_1") == F.col("team_2_batting_team"),
                F.col("innings_2_runs"),
            ),
        )
        .withColumn(
            "team_1_wickets",
            F.when(
                F.col("team_1") == F.col("team_1_batting_team"),
                F.col("innings_1_wickets"),
            ).when(
                F.col("team_1") == F.col("team_2_batting_team"),
                F.col("innings_2_wickets"),
            ),
        )
        .withColumn(
            "team_1_legal_balls",
            F.when(
                F.col("team_1") == F.col("team_1_batting_team"),
                F.col("innings_1_legal_balls"),
            ).when(
                F.col("team_1") == F.col("team_2_batting_team"),
                F.col("innings_2_legal_balls"),
            ),
        )
        .withColumn(
            "team_2_runs",
            F.when(
                F.col("team_2") == F.col("team_1_batting_team"),
                F.col("innings_1_runs"),
            ).when(
                F.col("team_2") == F.col("team_2_batting_team"),
                F.col("innings_2_runs"),
            ),
        )
        .withColumn(
            "team_2_wickets",
            F.when(
                F.col("team_2") == F.col("team_1_batting_team"),
                F.col("innings_1_wickets"),
            ).when(
                F.col("team_2") == F.col("team_2_batting_team"),
                F.col("innings_2_wickets"),
            ),
        )
        .withColumn(
            "team_2_legal_balls",
            F.when(
                F.col("team_2") == F.col("team_1_batting_team"),
                F.col("innings_1_legal_balls"),
            ).when(
                F.col("team_2") == F.col("team_2_batting_team"),
                F.col("innings_2_legal_balls"),
            ),
        )
        .withColumn(
            "total_match_runs",
            F.coalesce(F.col("team_1_runs"), F.lit(0))
            + F.coalesce(F.col("team_2_runs"), F.lit(0)),
        )
        .drop(
            "team_1_batting_team",
            "team_2_batting_team",
            "innings_1_runs",
            "innings_1_wickets",
            "innings_1_legal_balls",
            "innings_2_runs",
            "innings_2_wickets",
            "innings_2_legal_balls",
        )
    )

    return result


def main() -> None:
    """Read Silver data, validate the result, and write Gold Parquet."""

    spark = (
        SparkSession.builder.appName("IPL-Gold-Match-Summary").getOrCreate()
    )

    try:
        matches = spark.read.parquet(str(SILVER_ROOT / "matches"))
        innings = spark.read.parquet(str(SILVER_ROOT / "innings"))
        deliveries = spark.read.parquet(str(SILVER_ROOT / "deliveries"))

        summary = build_match_summary(matches, innings, deliveries)

        source_count = matches.count()
        output_count = summary.count()
        duplicate_ids = (
            summary.groupBy("match_id")
            .count()
            .filter(F.col("count") > 1)
            .count()
        )

        missing_scores = summary.filter(
        F.col("team_1_runs").isNull() | F.col("team_2_runs").isNull()
        )

        invalid_missing_scores = missing_scores.filter(
            F.col("outcome_type") != "no result"
        ).count()

        if (
            source_count != output_count
            or duplicate_ids > 0
            or invalid_missing_scores > 0
        ):
            raise ValueError(
                "Gold validation failed: "
                f"source={source_count}, output={output_count}, "
                f"duplicate_ids={duplicate_ids}, "
                f"missing_scores_in_non_no_result_matches="
                f"{invalid_missing_scores}"
            )

        output_path = GOLD_ROOT / "match_summary"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary.write.mode("overwrite").parquet(str(output_path))

        print(f"Matches: {output_count}")
        print(f"Duplicate match IDs: {duplicate_ids}")
        print(f"Matches with missing team scores: {missing_scores.count()}")
        print(
            "Missing scores in matches not marked no result: "
            f"{invalid_missing_scores}"
        )
        print(f"Gold output: {output_path}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
