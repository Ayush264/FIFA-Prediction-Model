"""
FIFA World Cup Prediction - PySpark ETL: Ranking Pipeline
Merges FIFA rankings, ELO scores, form data, and player aggregates
into a unified team profile used downstream by the simulation.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ranking_pipeline")

FEATURE_STORE = "data/feature_store"
PROCESSED_PATH = "data/processed"


def create_spark_session():
    import os
    import platform

    # Configure for Windows compatibility
    if platform.system() == "Windows":
        os.environ["JAVA_TOOL_OPTIONS"] = "-Dfile.encoding=UTF-8"

    spark = (
        SparkSession.builder
        .appName("FIFA_WC_RankingPipeline")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.warehouse.dir", "./tmp/warehouse")
        .config("spark.local.dir", "./tmp")
        .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def build_team_profiles(spark: SparkSession):
    log.info("Loading feature store lookups ...")

    elo = spark.read.csv(f"{FEATURE_STORE}/elo_ratings.csv",
                         header=True, inferSchema=True)
    form = spark.read.csv(f"{FEATURE_STORE}/team_form.csv",
                          header=True, inferSchema=True)
    ranks = spark.read.csv(
        f"{FEATURE_STORE}/fifa_rankings_latest.csv", header=True, inferSchema=True)
    players = spark.read.csv(
        f"{PROCESSED_PATH}/player_aggregates.csv", header=True, inferSchema=True)

    # Join all sources on team name
    profile = (
        elo
        .join(form, on="team", how="left")
        .join(ranks, on="team", how="left")
        .join(players.select(
            F.col("country").alias("team"),
            "avg_overall", "avg_attack_overall", "avg_defense_overall",
            "avg_pace", "avg_passing", "avg_dribbling"
        ), on="team", how="left")
    )

    # Fill missing numerics with global medians / defaults
    numeric_defaults = {
        "elo_rating": 1500.0,
        "form_score": 7.5,
        "goals_scored_avg": 1.3,
        "goals_conceded_avg": 1.3,
        "win_pct": 0.33,
        "fifa_rank": 100,
        "fifa_points": 1000.0,
        "avg_overall": 65.0,
        "avg_attack_overall": 65.0,
        "avg_defense_overall": 65.0,
        "avg_pace": 65.0,
        "avg_passing": 65.0,
        "avg_dribbling": 65.0,
    }
    profile = profile.fillna(numeric_defaults)

    # Composite strength score (normalized weighted sum)
    profile = profile.withColumn(
        "team_strength_score",
        (
            F.col("elo_rating") * 0.30
            + (200.0 - F.col("fifa_rank")) * 3.0 * 0.20
            + F.col("form_score") / 15.0 * 100.0 * 0.15
            + F.col("win_pct") * 100.0 * 0.15
            + F.col("avg_overall") * 0.20
        ).cast(DoubleType())
    )

    log.info(f"  -> Team profile built for {profile.count()} teams")
    # Export as CSV for downstream simulation and Power BI.
    profile.toPandas().to_csv(
        f"{FEATURE_STORE}/team_profiles.csv", index=False)
    log.info("  [OK] CSV export written for Power BI")

    return profile


def run():
    spark = create_spark_session()
    log.info("=== Starting Ranking Pipeline ===")
    build_team_profiles(spark)
    log.info("=== Ranking Pipeline Complete ===")
    spark.stop()


if __name__ == "__main__":
    run()
