import math
from pathlib import Path
import pandas as pd
import numpy as np

OUTPUT_DIR = Path("powerbi_exports")
OUTPUT_DIR.mkdir(exist_ok=True)

TEAM_RANKINGS_PATH = Path("dashboard/team_rankings.csv")
CHAMP_PROBS_PATH = Path("dashboard/championship_probabilities.csv")
FEATURE_IMPORTANCE_PATH = Path("dashboard/feature_importance.csv")
WORLD_CUP_MATCHES_PATH = Path("data/processed/world_cup_matches.csv")


def normalize(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    low = series.min()
    high = series.max()
    if high == low:
        return pd.Series(1.0, index=series.index)
    return (series - low) / (high - low)


def compute_power_index(rankings: pd.DataFrame, probs: pd.DataFrame) -> pd.DataFrame:
    df = rankings.copy()
    if df.empty:
        return df

    df["championship_probability"] = df["team"].map(
        probs.set_index("team")["championship_probability"]
    ).fillna(0.0)
    df["elo_score"] = normalize(df["elo"])
    df["form_score_norm"] = normalize(df["form"])
    df["player_score_norm"] = normalize(df["player_strength"])
    df["win_score_norm"] = normalize(df["win_pct"])
    df["rank_score_norm"] = 1.0 - normalize(df["fifa_rank"].fillna(df["fifa_rank"].max()))

    weights = {
        "elo_score": 0.3,
        "form_score_norm": 0.2,
        "player_score_norm": 0.2,
        "win_score_norm": 0.15,
        "rank_score_norm": 0.15,
    }
    df["power_index"] = (
        df["elo_score"] * weights["elo_score"]
        + df["form_score_norm"] * weights["form_score_norm"]
        + df["player_score_norm"] * weights["player_score_norm"]
        + df["win_score_norm"] * weights["win_score_norm"]
        + df["rank_score_norm"] * weights["rank_score_norm"]
    ) * 100
    df["elo_rank"] = df["elo"].rank(method="min", ascending=False).astype(int)
    df["form_rank"] = df["form"].rank(method="min", ascending=False).astype(int)
    df["player_strength_rank"] = df["player_strength"].rank(method="min", ascending=False).astype(int)
    df["win_pct_rank"] = df["win_pct"].rank(method="min", ascending=False).astype(int)
    df["power_rank"] = df["power_index"].rank(method="min", ascending=False).astype(int)
    df["fifa_rank_inv"] = df["fifa_rank"].max() - df["fifa_rank"] + 1
    df["fifa_rank_score_norm"] = normalize(df["fifa_rank_inv"])
    return df


def stage_category(stage: str) -> str:
    if not isinstance(stage, str):
        return "Unknown"
    stage = stage.strip()
    if "Group" in stage or stage in ["Preliminary round", "First round"]:
        return "Group Stage"
    if "Final" in stage and stage != "Match for third place":
        return "Final"
    if "Third place" in stage or "Play-off" in stage:
        return "Third Place"
    if any(k in stage for k in ["Round of", "Quarter", "Semi", "Knockout"]):
        return "Knockout Stage"
    return "Other"


def summarize_executive(df: pd.DataFrame, team_df: pd.DataFrame, prob_df: pd.DataFrame) -> pd.DataFrame:
    metrics = []
    total_matches = len(df)
    total_goals = int(df["total_goals"].sum())
    avg_goals = round(df["total_goals"].mean(), 2)
    unique_teams = pd.unique(df[["home_team", "away_team"]].values.ravel()).tolist()
    total_teams = len(unique_teams)
    home_wins = int(df["home_win_flag"].sum())
    away_wins = int(df["away_win_flag"].sum())
    draws = int(df["draw_flag"].sum())
    total_years = int(df["Year"].nunique())
    attendance = df["Attendance"].replace({np.nan: 0}).astype(float)
    avg_attendance = round(attendance[attendance > 0].mean(), 0)
    champion_counts = (
        df[df["Stage"] == "Final"]
        .assign(winner=lambda x: np.where(x["result"] == "home_win", x["home_team"], np.where(x["result"] == "away_win", x["away_team"], "Draw")))
        .groupby("winner")
        .size()
        .sort_values(ascending=False)
    )
    most_successful_team = champion_counts.index[0] if not champion_counts.empty else ""
    most_successful_titles = int(champion_counts.iloc[0]) if not champion_counts.empty else 0
    top_prob = prob_df.sort_values("championship_probability", ascending=False).head(1)
    top_prob_team = top_prob.iloc[0]["team"] if not top_prob.empty else ""
    top_prob_value = float(top_prob.iloc[0]["championship_probability"]) if not top_prob.empty else 0.0
    top_power = team_df.sort_values("power_index", ascending=False).head(1)
    top_power_team = top_power.iloc[0]["team"] if not top_power.empty else ""
    top_power_value = round(float(top_power.iloc[0]["power_index"]), 2) if not top_power.empty else 0.0
    top_5_power_avg = round(team_df.sort_values("power_index", ascending=False).head(5)["power_index"].mean(), 2)
    top_10_prob_share = round(prob_df.head(10)["championship_probability"].sum(), 2)
    simulation_runs = None
    if not prob_df.empty and prob_df["championship_probability"].gt(0).any():
        candidate = prob_df.loc[prob_df["championship_probability"] > 0].iloc[0]
        if candidate["championship_probability"] > 0:
            simulation_runs = int(round(candidate["champion_count"] / (candidate["championship_probability"] / 100)))
    metrics.extend([
        {"metric_name": "total_world_cup_tournaments", "value": total_years, "category": "Tournament", "description": "Unique World Cup editions represented in historical match data."},
        {"metric_name": "total_world_cup_matches", "value": total_matches, "category": "Tournament", "description": "Total World Cup matches played across all editions."},
        {"metric_name": "total_world_cup_goals", "value": total_goals, "category": "Match Analytics", "description": "Total goals scored in World Cup history."},
        {"metric_name": "average_goals_per_match", "value": avg_goals, "category": "Match Analytics", "description": "Average number of goals scored per World Cup match."},
        {"metric_name": "total_unique_teams", "value": total_teams, "category": "Teams", "description": "Unique national teams that have played in World Cup matches."},
        {"metric_name": "average_attendance_per_match", "value": avg_attendance, "category": "Fan Experience", "description": "Average match attendance for World Cup matches."},
        {"metric_name": "home_win_percentage", "value": round(home_wins / total_matches * 100, 2) if total_matches else 0.0, "category": "Match Analytics", "description": "Share of World Cup matches won by the home team."},
        {"metric_name": "away_win_percentage", "value": round(away_wins / total_matches * 100, 2) if total_matches else 0.0, "category": "Match Analytics", "description": "Share of World Cup matches won by the away team."},
        {"metric_name": "draw_percentage", "value": round(draws / total_matches * 100, 2) if total_matches else 0.0, "category": "Match Analytics", "description": "Share of World Cup matches ending in a draw."},
        {"metric_name": "most_successful_team", "value": most_successful_team, "category": "Teams", "description": "Team with the most World Cup titles in the historical match record."},
        {"metric_name": "most_successful_team_titles", "value": most_successful_titles, "category": "Teams", "description": "Number of World Cup titles won by the most successful team."},
        {"metric_name": "top_champion_probability_team", "value": top_prob_team, "category": "Simulation", "description": "Team with the highest championship probability in the latest simulation output."},
        {"metric_name": "top_champion_probability_percent", "value": top_prob_value, "category": "Simulation", "description": "Highest simulated championship probability percentage."},
        {"metric_name": "top_power_index_team", "value": top_power_team, "category": "Power Index", "description": "Team with the highest computed World Cup Power Index."},
        {"metric_name": "top_power_index_score", "value": top_power_value, "category": "Power Index", "description": "World Cup Power Index score of the top-ranked team."},
        {"metric_name": "top_5_power_index_average", "value": top_5_power_avg, "category": "Power Index", "description": "Average Power Index score for the top 5 teams."},
        {"metric_name": "top_10_championship_probability_share", "value": top_10_prob_share, "category": "Simulation", "description": "Combined championship probability percentage for the top 10 candidate teams."},
    ])
    if simulation_runs is not None:
        metrics.append({"metric_name": "simulation_runs", "value": simulation_runs, "category": "Simulation", "description": "Number of Monte Carlo tournament simulations implied by the probability output."})
    return pd.DataFrame(metrics)


def enrich_championship_probs(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["championship_probability"] = pd.to_numeric(result["championship_probability"], errors="coerce")
    result["champion_count"] = pd.to_numeric(result["champion_count"], errors="coerce").fillna(0).astype(int)
    if not result.empty and result["championship_probability"].gt(0).any():
        sample = result.loc[result["championship_probability"] > 0].iloc[0]
        runs = int(round(sample["champion_count"] / (sample["championship_probability"] / 100)))
        result["simulation_runs"] = runs
    else:
        result["simulation_runs"] = np.nan
    result["probability_rank"] = result["championship_probability"].rank(method="min", ascending=False).astype(int)
    result["top_10_flag"] = result["probability_rank"] <= 10
    result["championship_probability_share"] = result["championship_probability"] / 100
    return result


def enrich_feature_importance(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["importance"] = pd.to_numeric(result["importance"], errors="coerce").fillna(0.0)
    result["importance_rank"] = result["importance"].rank(method="min", ascending=False).astype(int)
    total = result["importance"].sum()
    result["importance_pct"] = round(result["importance"] / total * 100, 2) if total else 0.0
    result["importance_norm"] = normalize(result["importance"])
    result["feature_group"] = result["feature"].apply(lambda x: categorize_feature(x))
    result["description"] = result["feature"].apply(lambda x: feature_description(x))
    return result


def categorize_feature(feature: str) -> str:
    key = feature.lower()
    if "elo" in key:
        return "ELO"
    if "fifa" in key:
        return "FIFA"
    if "goal" in key:
        return "Goals"
    if "win" in key and "diff" in key:
        return "Win Gap"
    if "win" in key:
        return "Win Rate"
    if "form" in key:
        return "Form"
    if "player" in key:
        return "Player Strength"
    if "neutral" in key:
        return "Match Context"
    return "Other"


def feature_description(feature: str) -> str:
    mapping = {
        "neutral_match_flag": "Indicates matches played at neutral venues.",
        "elo_diff": "Difference between home and away ELO ratings.",
        "win_percentage_diff": "Difference in win percentage between teams.",
        "goals_conceded_diff": "Difference in goals conceded averages.",
        "ranking_diff": "Difference in FIFA ranking positions.",
        "away_elo": "Away team ELO rating.",
        "player_strength_diff": "Difference in average player strength.",
        "goals_scored_diff": "Difference in goals scored averages.",
        "home_avg_goals_conceded": "Home team average goals conceded.",
        "home_elo": "Home team ELO rating.",
        "away_avg_goals_conceded": "Away team average goals conceded.",
        "away_avg_goals_scored": "Away team average goals scored.",
        "away_win_percentage": "Away team win percentage.",
        "home_win_percentage": "Home team win percentage.",
        "home_avg_goals_scored": "Home team average goals scored.",
        "away_player_strength": "Away team player strength.",
        "home_player_strength": "Home team player strength.",
        "away_fifa_rank": "Away team FIFA ranking.",
        "home_fifa_rank": "Home team FIFA ranking.",
        "form_diff": "Difference in recent form between home and away teams.",
        "home_form": "Home team recent form score.",
        "away_form": "Away team recent form score.",
    }
    return mapping.get(feature, "Feature importance contribution used by the match outcome model.")


def build_historical_match_summary(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["home_goals"] = pd.to_numeric(result["home_goals"], errors="coerce").fillna(0).astype(int)
    result["away_goals"] = pd.to_numeric(result["away_goals"], errors="coerce").fillna(0).astype(int)
    result["total_goals"] = result["home_goals"] + result["away_goals"]
    result["goal_difference"] = (result["home_goals"] - result["away_goals"]).abs()
    result["match_date"] = pd.to_datetime(
        result["Datetime"].astype(str).str.extract(r"^(\d{1,2} [A-Za-z]+ \d{4})")[0],
        format="%d %b %Y",
        errors="coerce",
    )
    result["match_time"] = result["Datetime"].astype(str).str.extract(r"-\s*(.+)$")[0].fillna("")
    result["scoreline"] = result["home_goals"].astype(str) + "-" + result["away_goals"].astype(str)
    result["winner_team"] = np.where(
        result["result"] == "home_win",
        result["home_team"],
        np.where(result["result"] == "away_win", result["away_team"], "Draw"),
    )
    result["margin_of_victory"] = np.where(result["result"] == "draw", 0, result["goal_difference"])
    result["home_win_flag"] = (result["result"] == "home_win").astype(int)
    result["away_win_flag"] = (result["result"] == "away_win").astype(int)
    result["draw_flag"] = (result["result"] == "draw").astype(int)
    result["stage_category"] = result["Stage"].apply(stage_category)
    columns = [
        "Year",
        "match_date",
        "match_time",
        "Stage",
        "stage_category",
        "Stadium",
        "City",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "scoreline",
        "total_goals",
        "goal_difference",
        "winner_team",
        "margin_of_victory",
        "home_win_flag",
        "away_win_flag",
        "draw_flag",
        "win_conditions",
        "Attendance",
        "ht_home_goals",
        "ht_away_goals",
        "Referee",
        "Assistant 1",
        "Assistant 2",
        "RoundID",
        "MatchID",
        "Home Team Initials",
        "Away Team Initials",
        "result",
    ]
    return result[columns]


def build_tournament_statistics(df: pd.DataFrame) -> pd.DataFrame:
    match_df = build_historical_match_summary(df)
    agg = (
        match_df.groupby(["Year", "stage_category", "Stage"], dropna=False)
        .agg(
            matches_played=("scoreline", "count"),
            total_goals=("total_goals", "sum"),
            avg_goals_per_match=("total_goals", "mean"),
            home_wins=("home_win_flag", "sum"),
            away_wins=("away_win_flag", "sum"),
            draws=("draw_flag", "sum"),
            avg_attendance=("Attendance", lambda x: round(pd.to_numeric(x, errors="coerce").dropna().mean() or 0.0, 0)),
            total_attendance=("Attendance", lambda x: pd.to_numeric(x, errors="coerce").dropna().sum()),
            average_margin=("margin_of_victory", "mean"),
        )
        .reset_index()
    )
    agg["avg_goals_per_match"] = agg["avg_goals_per_match"].round(2)
    agg["average_margin"] = agg["average_margin"].round(2)
    overall = (
        match_df.groupby(["Year"], dropna=False)
        .agg(
            matches_played=("scoreline", "count"),
            total_goals=("total_goals", "sum"),
            avg_goals_per_match=("total_goals", "mean"),
            home_wins=("home_win_flag", "sum"),
            away_wins=("away_win_flag", "sum"),
            draws=("draw_flag", "sum"),
            avg_attendance=("Attendance", lambda x: round(pd.to_numeric(x, errors="coerce").dropna().mean() or 0.0, 0)),
            total_attendance=("Attendance", lambda x: pd.to_numeric(x, errors="coerce").dropna().sum()),
            average_margin=("margin_of_victory", "mean"),
        )
        .reset_index()
    )
    overall["stage_category"] = "Overall"
    overall["Stage"] = "Tournament"
    overall["avg_goals_per_match"] = overall["avg_goals_per_match"].round(2)
    overall["average_margin"] = overall["average_margin"].round(2)
    return pd.concat([agg, overall], ignore_index=True, sort=False)


def build_data_dictionary(datasets: dict) -> pd.DataFrame:
    records = []
    for dataset_name, columns in datasets.items():
        for column, description, dtype in columns:
            records.append({
                "dataset": dataset_name,
                "column_name": column,
                "data_type": dtype,
                "description": description,
            })
    return pd.DataFrame(records)


def main():
    team_rankings = pd.read_csv(TEAM_RANKINGS_PATH)
    championship_probs = pd.read_csv(CHAMP_PROBS_PATH)
    feature_importance = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    world_matches = pd.read_csv(WORLD_CUP_MATCHES_PATH)

    team_rankings_export = compute_power_index(team_rankings, championship_probs)
    team_rankings_export = team_rankings_export.sort_values("power_index", ascending=False)
    team_rankings_export.to_csv(OUTPUT_DIR / "team_rankings.csv", index=False)

    championship_probs_export = enrich_championship_probs(championship_probs)
    championship_probs_export = championship_probs_export.sort_values("championship_probability", ascending=False)
    championship_probs_export.to_csv(OUTPUT_DIR / "championship_probabilities.csv", index=False)

    feature_importance_export = enrich_feature_importance(feature_importance)
    feature_importance_export = feature_importance_export.sort_values("importance", ascending=False)
    feature_importance_export.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    match_summary_export = build_historical_match_summary(world_matches)
    match_summary_export.to_csv(OUTPUT_DIR / "historical_match_summary.csv", index=False)

    tournament_stats_export = build_tournament_statistics(world_matches)
    tournament_stats_export.to_csv(OUTPUT_DIR / "tournament_statistics.csv", index=False)

    executive_summary_export = summarize_executive(match_summary_export, team_rankings_export, championship_probs_export)
    executive_summary_export.to_csv(OUTPUT_DIR / "executive_summary.csv", index=False)

    data_dictionary = build_data_dictionary({
        "executive_summary": [
            ("metric_name", "string", "Unique KPI identifier."),
            ("value", "numeric_or_string", "KPI value for dashboard display."),
            ("category", "string", "Business domain category for the KPI."),
            ("description", "string", "Detailed explanation of the KPI."),
        ],
        "team_rankings": [
            ("team", "string", "National team name."),
            ("elo", "numeric", "ELO rating for the team."),
            ("form", "numeric", "Recent form score."),
            ("fifa_rank", "numeric", "FIFA ranking position."),
            ("fifa_points", "numeric", "FIFA ranking points."),
            ("win_pct", "numeric", "Win percentage."),
            ("avg_goals_scored", "numeric", "Average goals scored."),
            ("avg_goals_conceded", "numeric", "Average goals conceded."),
            ("player_strength", "numeric", "Average player strength."),
            ("team_strength_score", "numeric", "Composite team strength score."),
            ("championship_probability", "numeric", "Simulated championship probability percentage."),
            ("power_index", "numeric", "Computed World Cup Power Index score."),
            ("power_rank", "numeric", "Rank by power index."),
        ],
        "championship_probabilities": [
            ("team", "string", "National team name."),
            ("champion_count", "numeric", "Monte Carlo champion selection count."),
            ("championship_probability", "numeric", "Champion probability percentage."),
            ("simulation_runs", "numeric", "Inferred number of simulations."),
            ("probability_rank", "numeric", "Rank by championship probability."),
            ("top_10_flag", "boolean", "Indicates top 10 forecasted teams."),
            ("championship_probability_share", "numeric", "Champion probability as a fraction."),
        ],
        "feature_importance": [
            ("feature", "string", "Model feature name."),
            ("importance", "numeric", "Raw feature importance score."),
            ("importance_rank", "numeric", "Rank by importance."),
            ("importance_pct", "numeric", "Share of total importance."),
            ("importance_norm", "numeric", "Normalized importance between 0 and 1."),
            ("feature_group", "string", "Feature category grouping."),
            ("description", "string", "Human-readable feature meaning."),
        ],
        "historical_match_summary": [
            ("Year", "numeric", "Tournament year."),
            ("match_date", "date", "Match date."),
            ("match_time", "string", "Match kickoff time."),
            ("Stage", "string", "Competition stage."),
            ("stage_category", "string", "Stage group for analytics."),
            ("Stadium", "string", "Match stadium."),
            ("City", "string", "Match city."),
            ("home_team", "string", "Home team name."),
            ("away_team", "string", "Away team name."),
            ("home_goals", "numeric", "Home team goals."),
            ("away_goals", "numeric", "Away team goals."),
            ("scoreline", "string", "Formatted scoreline."),
            ("total_goals", "numeric", "Total goals in match."),
            ("goal_difference", "numeric", "Absolute goal difference."),
            ("winner_team", "string", "Match winner or Draw."),
            ("margin_of_victory", "numeric", "Winning margin."),
            ("home_win_flag", "boolean", "Home win indicator."),
            ("away_win_flag", "boolean", "Away win indicator."),
            ("draw_flag", "boolean", "Draw indicator."),
        ],
        "tournament_statistics": [
            ("Year", "numeric", "Tournament year."),
            ("stage_category", "string", "Group, Knockout, Final, or Overall."),
            ("Stage", "string", "Stage name."),
            ("matches_played", "numeric", "Number of matches in aggregation."),
            ("total_goals", "numeric", "Total goals scored."),
            ("avg_goals_per_match", "numeric", "Average goals per match."),
            ("home_wins", "numeric", "Number of home wins."),
            ("away_wins", "numeric", "Number of away wins."),
            ("draws", "numeric", "Number of draws."),
            ("avg_attendance", "numeric", "Average attendance."),
            ("total_attendance", "numeric", "Total attendance."),
            ("average_margin", "numeric", "Average victory margin."),
        ],
    })
    data_dictionary.to_csv(OUTPUT_DIR / "data_dictionary.csv", index=False)
    print("Power BI export files created in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
