
"""Build player-level batting and bowling performance from Silver deliveries."""

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

SILVER_ROOT = Path("data/silver/cricsheet")
GOLD_ROOT = Path("data/gold/cricsheet")
EXPECTED_DELIVERIES = 295732

BOWLER_CREDITED_WICKETS = (
    "bowled",
    "caught",
    "caught and bowled",
    "lbw",
    "stumped",
    "hit wicket",
)


def build_player_performance(deliveries: DataFrame) -> DataFrame:
    """Return one analytics-ready row per batter or bowler."""

    # Count dismissals by the player actually dismissed.
    dismissals = (
        deliveries.filter(F.col("players_out").isNotNull())
        .select(
            "match_id",
            "innings_number",
            F.explode(
                F.split("players_out", r"\|")
            ).alias("player"),
        )
        .groupBy("player")
        .agg(F.count("*").alias("outs"))
    )

    # Batting statistics.
    batting = (
        deliveries.groupBy(F.col("batter").alias("player"))
        .agg(
            F.countDistinct("match_id").alias("batting_matches"),
            F.countDistinct(
                F.struct("match_id", "innings_number")
            ).alias("batting_innings"),
            F.sum("batter_runs").alias("runs_scored"),
            F.sum(
                F.when(
                    ~F.coalesce(F.col("is_wide"), F.lit(False)),
                    1,
                ).otherwise(0)
            ).alias("balls_faced"),
            F.sum(
                F.when(F.col("batter_runs") == 4, 1).otherwise(0)
            ).alias("fours"),
            F.sum(
                F.when(F.col("batter_runs") == 6, 1).otherwise(0)
            ).alias("sixes"),
            F.sum(
                F.when(
                    ~F.coalesce(F.col("is_wide"), F.lit(False))
                    & (F.col("total_runs") == 0),
                    1,
                ).otherwise(0)
            ).alias("dot_balls_faced"),
        )
        .withColumn(
            "batting_strike_rate",
            F.when(
                F.col("balls_faced") > 0,
                F.round(
                    F.col("runs_scored") * F.lit(100.0)
                    / F.col("balls_faced"),
                    2,
                ),
            ),
        )
        .join(dismissals, on="player", how="full_outer")
    )

    # Bowling delivery-level calculations.
    bowling_deliveries = (
        deliveries.withColumn(
            "bowler_wickets",
            F.when(
                F.coalesce(F.col("is_wicket"), F.lit(False))
                & F.col("wicket_types").isNotNull()
                & (
                    F.size(
                        F.array_intersect(
                            F.split("wicket_types", r"\|"),
                            F.array(
                                *[
                                    F.lit(kind)
                                    for kind in BOWLER_CREDITED_WICKETS
                                ]
                            ),
                        )
                    )
                    > 0
                ),
                1,
            ).otherwise(0),
        )
        .withColumn(
            "bowling_runs_conceded",
            F.col("total_runs")
            - F.coalesce(F.col("byes"), F.lit(0))
            - F.coalesce(F.col("legbyes"), F.lit(0))
            - F.coalesce(F.col("penalty_runs"), F.lit(0)),
        )
    )

    # Aggregate bowling statistics.
    bowling = (
        bowling_deliveries.groupBy(F.col("bowler").alias("player"))
        .agg(
            F.countDistinct("match_id").alias("bowling_matches"),
            F.sum("bowler_wickets").alias("wickets_taken"),
            F.sum("bowling_runs_conceded").alias("runs_conceded"),
            F.sum(
                F.when(
                    ~F.coalesce(F.col("is_wide"), F.lit(False))
                    & ~F.coalesce(F.col("is_no_ball"), F.lit(False)),
                    1,
                ).otherwise(0)
            ).alias("legal_balls_bowled"),
        )
        .withColumn(
            "bowling_economy",
            F.when(
                F.col("legal_balls_bowled") > 0,
                F.round(
                    F.col("runs_conceded") * F.lit(6.0)
                    / F.col("legal_balls_bowled"),
                    2,
                ),
            ),
        )
    )

    # Count distinct matches across batting, bowling, and dismissal records.
    dismissed_player_matches = (
        deliveries.filter(F.col("players_out").isNotNull())
        .select(
            "match_id",
            F.explode(
                F.split("players_out", r"\|")
            ).alias("player"),
        )
    )

    player_matches = (
        deliveries.select(
            F.col("batter").alias("player"),
            "match_id",
        )
        .unionByName(
            deliveries.select(
                F.col("bowler").alias("player"),
                "match_id",
            )
        )
        .unionByName(
            dismissed_player_matches.select("player", "match_id")
        )
        .distinct()
        .groupBy("player")
        .agg(F.countDistinct("match_id").alias("matches_played"))
    )

    # Combine role-specific statistics without multiplying records.
    combined = (
        batting.join(bowling, on="player", how="full_outer")
        .join(player_matches, on="player", how="full_outer")
    )

    count_columns = [
        "matches_played",
        "batting_matches",
        "batting_innings",
        "runs_scored",
        "balls_faced",
        "fours",
        "sixes",
        "dot_balls_faced",
        "outs",
        "wickets_taken",
        "runs_conceded",
        "legal_balls_bowled",
    ]

    for column in count_columns:
        combined = combined.withColumn(
            column,
            F.coalesce(F.col(column), F.lit(0)).cast("long"),
        )

    # Additional batting and bowling metrics.
    combined = (
        combined
        .withColumn(
            "boundary_runs",
            F.col("fours") * 4 + F.col("sixes") * 6,
        )
        .withColumn(
            "batting_average",
            F.when(
                F.col("outs") > 0,
                F.round(
                    F.col("runs_scored") / F.col("outs"),
                    2,
                ),
            ),
        )
        .withColumn(
            "boundary_percentage",
            F.when(
                F.col("runs_scored") > 0,
                F.round(
                    F.col("boundary_runs") * F.lit(100.0)
                    / F.col("runs_scored"),
                    2,
                ),
            ),
        )
        .withColumn(
            "dot_ball_percentage",
            F.when(
                F.col("balls_faced") > 0,
                F.round(
                    F.col("dot_balls_faced") * F.lit(100.0)
                    / F.col("balls_faced"),
                    2,
                ),
            ),
        )
        .withColumn(
            "bowling_average",
            F.when(
                F.col("wickets_taken") > 0,
                F.round(
                    F.col("runs_conceded") / F.col("wickets_taken"),
                    2,
                ),
            ),
        )
        .withColumn(
            "bowling_strike_rate",
            F.when(
                F.col("wickets_taken") > 0,
                F.round(
                    F.col("legal_balls_bowled")
                    / F.col("wickets_taken"),
                    2,
                ),
            ),
        )
    )

    return combined.select(
        "player",
        "matches_played",
        "batting_matches",
        "batting_innings",
        "runs_scored",
        "outs",
        "batting_average",
        "balls_faced",
        "fours",
        "sixes",
        "boundary_runs",
        "boundary_percentage",
        "dot_balls_faced",
        "dot_ball_percentage",
        "batting_strike_rate",
        "wickets_taken",
        "runs_conceded",
        "legal_balls_bowled",
        "bowling_average",
        "bowling_strike_rate",
        "bowling_economy",
    ).orderBy(F.desc("runs_scored"), "player")


def main() -> None:
    """Build, validate, and write the Gold player performance dataset."""

    spark = (
        SparkSession.builder
        .appName("IPL-Gold-Player-Performance")
        .getOrCreate()
    )

    try:
        # Read and validate the complete Silver delivery dataset.
        all_deliveries = spark.read.parquet(
            str(SILVER_ROOT / "deliveries")
        )

        source_count = all_deliveries.count()
        if source_count != EXPECTED_DELIVERIES:
            raise ValueError(
                f"Unexpected Silver delivery count: {source_count}"
            )

        # Standard IPL career statistics exclude Super Overs.
        deliveries = all_deliveries.filter(
            ~F.coalesce(
                F.col("is_super_over"),
                F.lit(False),
            )
        )

        performance = build_player_performance(deliveries)

        player_count = performance.count()
        duplicate_players = (
            performance.groupBy("player")
            .count()
            .filter(F.col("count") > 1)
            .count()
        )

        null_player_names = performance.filter(
            F.col("player").isNull()
        ).count()

        if player_count == 0 or duplicate_players > 0 or null_player_names > 0:
            raise ValueError(
                "Player performance validation failed: "
                f"players={player_count}, "
                f"duplicates={duplicate_players}, "
                f"null_player_names={null_player_names}"
            )

        output_path = GOLD_ROOT / "player_performance"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        performance.write.mode("overwrite").parquet(str(output_path))

        print(f"Silver deliveries: {source_count}")
        print(f"Players: {player_count}")
        print(f"Duplicate player names: {duplicate_players}")
        print(f"Null player names: {null_player_names}")
        print(f"Gold output: {output_path}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()