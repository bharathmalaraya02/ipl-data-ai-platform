
"""Build an overall team performance summary from Gold match data."""

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

GOLD_ROOT = Path("data/gold/cricsheet")


def build_team_performance(matches: DataFrame) -> DataFrame:
    """Return one analytics-ready row per team."""

    team_1 = matches.select(
        F.col("team_1").alias("team"),
        "match_id",
        "winner",
        "outcome_type",
        F.col("team_1_runs").alias("runs_scored"),
        F.col("team_2_runs").alias("runs_conceded"),
    )

    team_2 = matches.select(
        F.col("team_2").alias("team"),
        "match_id",
        "winner",
        "outcome_type",
        F.col("team_2_runs").alias("runs_scored"),
        F.col("team_1_runs").alias("runs_conceded"),
    )

    team_matches = team_1.unionByName(team_2)

    summary = (
        team_matches.groupBy("team")
        .agg(
            F.countDistinct("match_id").alias("matches_played"),

            F.sum(
                F.when(F.col("winner") == F.col("team"), 1).otherwise(0)
            ).cast("long").alias("wins"),
            F.sum(
                F.when(
                    (F.col("outcome_type") == "winner")
                    & (F.col("winner") != F.col("team")),
                    1,
                ).otherwise(0)
            ).cast("long").alias("losses"),
            F.sum(
                F.when(F.col("outcome_type") == "tie", 1).otherwise(0)
            ).cast("long").alias("ties"),
            F.sum(
                F.when(F.col("outcome_type") == "no result", 1).otherwise(0)
            ).cast("long").alias("no_results"),
            F.sum("runs_scored").alias("total_runs_scored"),
            F.round(F.avg("runs_scored"), 2).alias("average_runs_scored"),
            F.sum("runs_conceded").alias("total_runs_conceded"),
            F.round(F.avg("runs_conceded"), 2).alias("average_runs_conceded"),
        )
        .withColumn(
            "win_percentage",
            F.when(
                (F.col("wins") + F.col("losses") + F.col("ties")) > 0,
                F.round(
                    F.col("wins")
                    * F.lit(100.0)
                    / (F.col("wins") + F.col("losses") + F.col("ties")),
                    2,
                ),
            ).otherwise(F.lit(0.0)),
        )
        .select(
            "team",
            "matches_played",
            "wins",
            "losses",
            "ties",
            "no_results",
            "win_percentage",
            "total_runs_scored",
            "average_runs_scored",
            "total_runs_conceded",
            "average_runs_conceded",
        )
        .orderBy(F.desc("wins"), F.desc("win_percentage"), "team")
    )

    return summary


def main() -> None:
    """Read the Gold match summary, validate, and write team performance."""

    spark = (
        SparkSession.builder
        .appName("IPL-Gold-Team-Performance")
        .getOrCreate()
    )

    try:
        matches = spark.read.parquet(str(GOLD_ROOT / "match_summary"))
        summary = build_team_performance(matches)

        team_count = summary.count()
        duplicate_teams = (
            summary.groupBy("team")
            .count()
            .filter(F.col("count") > 1)
            .count()
        )

        if team_count == 0 or duplicate_teams > 0:
            raise ValueError(
                f"Team performance validation failed: "
                f"teams={team_count}, duplicate_teams={duplicate_teams}"
            )

        output_path = GOLD_ROOT / "team_performance"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary.write.mode("overwrite").parquet(str(output_path))

        print(f"Teams: {team_count}")
        print(f"Duplicate team names: {duplicate_teams}")
        print(f"Gold output: {output_path}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()