"""
FIFA World Cup Prediction - PySpark ETL: Data Cleaning
Loads all raw CSVs, standardizes team names, handles missing values,
formats dates, removes duplicates, and writes cleaned Parquet files.
"""

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, IntegerType, DoubleType, DateType
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("data_cleaning")

RAW_PATH = "data/raw"
PROCESSED_PATH = "data/processed"


def write_csv(df, output_name: str) -> None:
    """Write a Spark DataFrame as a single local CSV without Hadoop file commits."""
    import os

    os.makedirs(PROCESSED_PATH, exist_ok=True)
    df.toPandas().to_csv(f"{PROCESSED_PATH}/{output_name}.csv", index=False)


def create_spark_session():
    import os
    import platform

    # Configure for Windows compatibility
    if platform.system() == "Windows":
        os.environ["JAVA_TOOL_OPTIONS"] = "-Dfile.encoding=UTF-8"

    spark = (
        SparkSession.builder
        .appName("FIFA_WC_DataCleaning")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.warehouse.dir", "./tmp/warehouse")
        .config("spark.local.dir", "./tmp")
        .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# ---------------------------------------------
# Team Name Standardization Map
# ---------------------------------------------
TEAM_NAME_MAP = {
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "China PR": "China",
    "USA": "United States",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Republic of Ireland": "Ireland",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "Cape Verde": "Cape Verde Islands",
    "Kyrgyzstan": "Kyrgyz Republic",
    "North Macedonia": "Macedonia",
    "St. Kitts and Nevis": "Saint Kitts and Nevis",
    "St. Lucia": "Saint Lucia",
    "St. Vincent and the Grenadines": "Saint Vincent and the Grenadines",
}


def build_name_normalizer(mapping: dict):
    """Return a PySpark UDF that maps legacy/variant names to canonical names."""
    def normalize(name):
        if name is None:
            return None
        return mapping.get(name.strip(), name.strip())
    return F.udf(normalize, StringType())


normalize_name = build_name_normalizer(TEAM_NAME_MAP)


# ---------------------------------------------
# Dataset Loaders & Cleaners
# ---------------------------------------------

def clean_all_matches(spark: SparkSession) -> None:
    log.info("Cleaning all_matches.csv ...")
    df = spark.read.csv(f"{RAW_PATH}/all_matches.csv",
                        header=True, inferSchema=True)

    df = (
        df
        .withColumn("date", F.to_date("date", "yyyy-MM-dd"))
        .withColumn("home_team", normalize_name("home_team"))
        .withColumn("away_team", normalize_name("away_team"))
        .withColumn("home_score", F.col("home_score").cast(IntegerType()))
        .withColumn("away_score", F.col("away_score").cast(IntegerType()))
        .dropDuplicates(["date", "home_team", "away_team"])
        .dropna(subset=["home_team", "away_team", "home_score", "away_score"])
        .withColumn("year", F.year("date"))
        .withColumn(
            "result",
            F.when(F.col("home_score") > F.col("away_score"), "home_win")
             .when(F.col("home_score") < F.col("away_score"), "away_win")
             .otherwise("draw")
        )
    )

    log.info(f"  -> {df.count()} rows after cleaning")
    write_csv(df, "all_matches")
    log.info("  [OK] Written to data/processed/all_matches.csv")


def clean_fifa_rankings(spark: SparkSession) -> None:
    log.info("Cleaning fifa_mens_rank.csv ...")
    df = spark.read.csv(f"{RAW_PATH}/fifa_mens_rank.csv",
                        header=True, inferSchema=True)

    df = (
        df
        .withColumnRenamed("total.points", "total_points")
        .withColumnRenamed("previous.points", "previous_points")
        .withColumnRenamed("diff.points", "diff_points")
        .withColumn("team", normalize_name("team"))
        .dropDuplicates(["date", "team"])
        .dropna(subset=["date", "rank", "team", "total_points"])
        .withColumn("rank", F.col("rank").cast(IntegerType()))
        .withColumn("total_points", F.col("total_points").cast(DoubleType()))
    )

    log.info(f"  -> {df.count()} rows after cleaning")
    write_csv(df, "fifa_rankings")
    log.info("  [OK] Written to data/processed/fifa_rankings.csv")


def clean_world_cup_matches(spark: SparkSession) -> None:
    log.info("Cleaning WorldCupMatches.csv ...")
    df = spark.read.csv(f"{RAW_PATH}/WorldCupMatches.csv",
                        header=True, inferSchema=True)

    df = (
        df
        .withColumnRenamed("Home Team Name", "home_team")
        .withColumnRenamed("Away Team Name", "away_team")
        .withColumnRenamed("Home Team Goals", "home_goals")
        .withColumnRenamed("Away Team Goals", "away_goals")
        .withColumnRenamed("Win conditions", "win_conditions")
        .withColumnRenamed("Half-time Home Goals", "ht_home_goals")
        .withColumnRenamed("Half-time Away Goals", "ht_away_goals")
        .withColumn("home_team", normalize_name(F.trim("home_team")))
        .withColumn("away_team", normalize_name(F.trim("away_team")))
        .withColumn("home_goals", F.col("home_goals").cast(IntegerType()))
        .withColumn("away_goals", F.col("away_goals").cast(IntegerType()))
        .withColumn("Attendance", F.col("Attendance").cast(IntegerType()))
        .withColumn("Year", F.col("Year").cast(IntegerType()))
        .dropDuplicates(["MatchID"])
        .dropna(subset=["home_team", "away_team", "home_goals", "away_goals"])
        .withColumn(
            "result",
            F.when(F.col("home_goals") > F.col("away_goals"), "home_win")
             .when(F.col("home_goals") < F.col("away_goals"), "away_win")
             .otherwise("draw")
        )
    )

    log.info(f"  -> {df.count()} rows after cleaning")
    write_csv(df, "world_cup_matches")
    log.info("  [OK] Written to data/processed/world_cup_matches.csv")


def clean_player_aggregates(spark: SparkSession) -> None:
    log.info("Cleaning player_aggregates.csv ...")
    df = spark.read.csv(f"{RAW_PATH}/player_aggregates.csv",
                        header=True, inferSchema=True)

    numeric_cols = [
        "avg_overall", "max_overall", "avg_pace", "avg_shooting",
        "avg_passing", "avg_dribbling", "avg_defending",
        "avg_physic", "avg_attack_overall", "avg_defense_overall"
    ]
    for col in numeric_cols:
        df = df.withColumn(col, F.col(col).cast(DoubleType()))

    df = (
        df
        .withColumn("country", normalize_name("country"))
        .dropDuplicates(["country", "fifa_version"])
        .dropna(subset=["country", "avg_overall"])
        # Keep only most recent FIFA version per country
        .withColumn(
            "row_num",
            F.row_number().over(Window.partitionBy("country").orderBy(F.col("fifa_version").desc()))
        )
        .filter(F.col("row_num") == 1)
    )
    log.info(f"  -> {df.count()} rows after cleaning")
    write_csv(df, "player_aggregates")
    log.info("  [OK] Written to data/processed/player_aggregates.csv")


def clean_former_names(spark: SparkSession) -> None:
    log.info("Cleaning former_names.csv ...")
    df = spark.read.csv(f"{RAW_PATH}/former_names.csv",
                        header=True, inferSchema=True)

    df = (
        df
        .dropDuplicates(["current", "former"])
        .dropna(subset=["current", "former"])
    )
    log.info(f"  -> {df.count()} rows after cleaning")
    write_csv(df, "former_names")
    log.info("  [OK] Written to data/processed/former_names.csv")


# ---------------------------------------------
# Entry Point
# ---------------------------------------------

def run():
    spark=create_spark_session()
    log.info("=== Starting Data Cleaning ETL ===")

    clean_all_matches(spark)
    clean_fifa_rankings(spark)
    clean_world_cup_matches(spark)
    clean_player_aggregates(spark)
    clean_former_names(spark)

    log.info("=== Data Cleaning Complete ===")
    spark.stop()


if __name__ == "__main__":
    run()
