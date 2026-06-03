"""
FIFA World Cup Prediction - PySpark ETL: Feature Engineering

Builds a leakage-aware training set from all historical international matches.
For each match, features are computed only from information available before
that match date, then rolling team state is updated with the match result.
"""

from collections import defaultdict, deque
import bisect
import logging
import os
import platform
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("feature_engineering")

PROCESSED_PATH = "data/processed"
FEATURE_STORE = "data/feature_store"

DEFAULT_ELO = 1500.0
DEFAULT_FIFA_RANK = 200
DEFAULT_PLAYER_STRENGTH = 65.0

CONTINENTAL_TOURNAMENTS = (
    "uefa euro",
    "copa america",
    "copa américa",
    "afc asian cup",
    "africa cup of nations",
    "concacaf gold cup",
    "ofc nations cup",
    "cafu nations cup",
)


def create_spark_session():
    if platform.system() == "Windows":
        os.environ["JAVA_TOOL_OPTIONS"] = "-Dfile.encoding=UTF-8"

    spark = (
        SparkSession.builder
        .appName("FIFA_WC_FeatureEngineering")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.warehouse.dir", "./tmp/warehouse")
        .config("spark.local.dir", "./tmp")
        .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def get_k_factor(tournament: str) -> int:
    name = str(tournament or "").lower()
    if "fifa world cup" in name and "qualification" not in name and "qualifier" not in name:
        return 60
    if any(token in name for token in CONTINENTAL_TOURNAMENTS):
        return 40
    if "qualification" in name or "qualifier" in name or "qualifying" in name:
        return 30
    if "friendly" in name:
        return 20
    return 30


def expected_score(team_elo: float, opponent_elo: float) -> float:
    return 1.0 / (1.0 + 10 ** ((opponent_elo - team_elo) / 400.0))


def result_scores(home_score: int, away_score: int):
    if home_score > away_score:
        return 1.0, 0.0, 2
    if home_score < away_score:
        return 0.0, 1.0, 0
    return 0.5, 0.5, 1


def ranking_effective_date(row) -> pd.Timestamp:
    year = int(row["date"])
    month = 1 if int(row.get("semester", 1) or 1) == 1 else 7
    return pd.Timestamp(year=year, month=month, day=1)


def build_ranking_lookup(rankings_pd: pd.DataFrame):
    rankings = rankings_pd.dropna(subset=["team", "rank"]).copy()
    rankings["ranking_date"] = rankings.apply(ranking_effective_date, axis=1)
    rankings = rankings.sort_values(["team", "ranking_date"])

    lookup = {}
    for team, grp in rankings.groupby("team"):
        dates = list(grp["ranking_date"])
        ranks = [int(v) for v in grp["rank"]]
        lookup[team] = (dates, ranks)
    return lookup


def latest_rank_before(team: str, match_date: pd.Timestamp, ranking_lookup) -> int:
    dates, ranks = ranking_lookup.get(team, ([], []))
    idx = bisect.bisect_left(dates, match_date) - 1
    if idx < 0:
        return DEFAULT_FIFA_RANK
    return ranks[idx]


def build_player_strength_lookup(players_pd: pd.DataFrame):
    if players_pd.empty:
        return {}

    players = players_pd.dropna(subset=["country"]).copy()
    if "fifa_version" in players.columns:
        players = players.sort_values("fifa_version").groupby("country").tail(1)

    strength_cols = [c for c in ["avg_overall", "avg_attack_overall", "avg_defense_overall"] if c in players.columns]
    players["player_strength"] = players[strength_cols].mean(axis=1)
    return {
        row["country"]: float(row["player_strength"])
        for _, row in players.iterrows()
        if pd.notna(row["player_strength"])
    }


def empty_team_state():
    return {
        "matches": 0,
        "wins": 0,
        "goals_scored": 0,
        "goals_conceded": 0,
        "recent_points": deque(maxlen=5),
    }


def team_snapshot(team: str, states) -> dict:
    state = states[team]
    matches = state["matches"]
    return {
        "form": float(sum(state["recent_points"])),
        "avg_goals_scored": float(state["goals_scored"] / matches) if matches else 0.0,
        "avg_goals_conceded": float(state["goals_conceded"] / matches) if matches else 0.0,
        "win_percentage": float(state["wins"] / matches) if matches else 0.0,
    }


def update_team_state(team: str, goals_for: int, goals_against: int, points: int, states) -> None:
    state = states[team]
    state["matches"] += 1
    state["wins"] += 1 if points == 3 else 0
    state["goals_scored"] += goals_for
    state["goals_conceded"] += goals_against
    state["recent_points"].append(points)


def build_match_features(matches_pd: pd.DataFrame, ranking_lookup, player_strength_lookup):
    matches = matches_pd.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"]).copy()
    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    matches = matches.dropna(subset=["date"]).sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)

    elo = defaultdict(lambda: DEFAULT_ELO)
    states = defaultdict(empty_team_state)
    rows = []

    for _, match in matches.iterrows():
        home = match["home_team"]
        away = match["away_team"]
        match_date = match["date"]
        home_score = int(match["home_score"])
        away_score = int(match["away_score"])
        tournament = match.get("tournament", "")

        home_elo = float(elo[home])
        away_elo = float(elo[away])
        home_state = team_snapshot(home, states)
        away_state = team_snapshot(away, states)

        home_rank = latest_rank_before(home, match_date, ranking_lookup)
        away_rank = latest_rank_before(away, match_date, ranking_lookup)
        home_strength = player_strength_lookup.get(home, DEFAULT_PLAYER_STRENGTH)
        away_strength = player_strength_lookup.get(away, DEFAULT_PLAYER_STRENGTH)
        _, _, label = result_scores(home_score, away_score)

        rows.append({
            "home_team": home,
            "away_team": away,
            "date": match_date.date().isoformat(),
            "tournament": tournament,
            "home_score": home_score,
            "away_score": away_score,
            "home_elo": round(home_elo, 3),
            "away_elo": round(away_elo, 3),
            "elo_diff": round(home_elo - away_elo, 3),
            "home_form": home_state["form"],
            "away_form": away_state["form"],
            "form_diff": round(home_state["form"] - away_state["form"], 3),
            "home_avg_goals_scored": round(home_state["avg_goals_scored"], 3),
            "away_avg_goals_scored": round(away_state["avg_goals_scored"], 3),
            "goals_scored_diff": round(home_state["avg_goals_scored"] - away_state["avg_goals_scored"], 3),
            "home_avg_goals_conceded": round(home_state["avg_goals_conceded"], 3),
            "away_avg_goals_conceded": round(away_state["avg_goals_conceded"], 3),
            "goals_conceded_diff": round(home_state["avg_goals_conceded"] - away_state["avg_goals_conceded"], 3),
            "home_win_percentage": round(home_state["win_percentage"], 4),
            "away_win_percentage": round(away_state["win_percentage"], 4),
            "win_percentage_diff": round(home_state["win_percentage"] - away_state["win_percentage"], 4),
            "home_fifa_rank": home_rank,
            "away_fifa_rank": away_rank,
            "ranking_diff": away_rank - home_rank,
            "home_player_strength": round(home_strength, 3),
            "away_player_strength": round(away_strength, 3),
            "player_strength_diff": round(home_strength - away_strength, 3),
            "neutral_match_flag": 1 if str(match.get("neutral", "")).lower() == "true" else 0,
            "label": label,
        })

        home_result, away_result, _ = result_scores(home_score, away_score)
        k = get_k_factor(tournament)
        home_expected = expected_score(home_elo, away_elo)
        away_expected = expected_score(away_elo, home_elo)
        elo[home] = home_elo + k * (home_result - home_expected)
        elo[away] = away_elo + k * (away_result - away_expected)

        if label == 2:
            home_points, away_points = 3, 0
        elif label == 0:
            home_points, away_points = 0, 3
        else:
            home_points, away_points = 1, 1

        update_team_state(home, home_score, away_score, home_points, states)
        update_team_state(away, away_score, home_score, away_points, states)

    feature_df = pd.DataFrame(rows)
    elo_df = pd.DataFrame(
        [{"team": team, "elo_rating": round(rating, 3)} for team, rating in sorted(elo.items())]
    )
    form_df = pd.DataFrame([
        {
            "team": team,
            "form_score": float(sum(state["recent_points"])),
            "goals_scored_avg": round(state["goals_scored"] / state["matches"], 3) if state["matches"] else 0.0,
            "goals_conceded_avg": round(state["goals_conceded"] / state["matches"], 3) if state["matches"] else 0.0,
            "win_pct": round(state["wins"] / state["matches"], 4) if state["matches"] else 0.0,
            "total_matches": state["matches"],
        }
        for team, state in sorted(states.items())
    ])
    return feature_df, elo_df, form_df


def build_feature_store(spark: SparkSession):
    log.info("Loading cleaned datasets with Spark ...")
    matches_spark = spark.read.csv(f"{PROCESSED_PATH}/all_matches.csv", header=True, inferSchema=True)
    rankings_spark = spark.read.csv(f"{PROCESSED_PATH}/fifa_rankings.csv", header=True, inferSchema=True)
    players_spark = spark.read.csv(f"{PROCESSED_PATH}/player_aggregates.csv", header=True, inferSchema=True)

    matches_spark = matches_spark.select(
        "date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral"
    ).where(
        F.col("date").isNotNull()
        & F.col("home_team").isNotNull()
        & F.col("away_team").isNotNull()
        & F.col("home_score").isNotNull()
        & F.col("away_score").isNotNull()
    )

    log.info("Collecting ordered inputs for rolling feature generation ...")
    matches_pd = matches_spark.toPandas()
    rankings_pd = rankings_spark.toPandas()
    players_pd = players_spark.toPandas()

    log.info("Building ranking and player lookup tables ...")
    ranking_lookup = build_ranking_lookup(rankings_pd)
    player_strength_lookup = build_player_strength_lookup(players_pd)

    log.info("Generating leakage-safe match features from all_matches.csv ...")
    features_df, elo_df, form_df = build_match_features(
        matches_pd, ranking_lookup, player_strength_lookup
    )

    latest_rank = (
        rankings_pd.assign(ranking_date=rankings_pd.apply(ranking_effective_date, axis=1))
        .sort_values("ranking_date")
        .groupby("team")
        .tail(1)[["team", "rank", "total_points"]]
        .rename(columns={"rank": "fifa_rank", "total_points": "fifa_points"})
        .sort_values("team")
    )

    Path(FEATURE_STORE).mkdir(parents=True, exist_ok=True)
    features_df.to_csv(f"{FEATURE_STORE}/match_features.csv", index=False)
    elo_df.to_csv(f"{FEATURE_STORE}/elo_ratings.csv", index=False)
    form_df.to_csv(f"{FEATURE_STORE}/team_form.csv", index=False)
    latest_rank.to_csv(f"{FEATURE_STORE}/fifa_rankings_latest.csv", index=False)

    log.info(f"  -> {len(features_df):,} leakage-safe feature rows written")
    log.info(f"  -> {len(elo_df):,} final ELO rows written")
    return features_df


def run():
    spark = create_spark_session()
    log.info("=== Starting Feature Engineering ===")
    build_feature_store(spark)
    log.info("=== Feature Engineering Complete ===")
    spark.stop()


if __name__ == "__main__":
    run()
