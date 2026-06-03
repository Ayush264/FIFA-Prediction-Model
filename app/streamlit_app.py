from pathlib import Path
import pickle

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = BASE_DIR / "dashboard"
FEATURE_STORE = BASE_DIR / "data" / "feature_store"
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"

DEFAULT_ELO = 1500.0
DEFAULT_FORM = 0.0
DEFAULT_RANK = 200
DEFAULT_GOALS = 0.0
DEFAULT_WIN_PCT = 0.0
DEFAULT_PLAYER_STRENGTH = 65.0


st.set_page_config(
    page_title="FIFA World Cup Analytics Platform",
    layout="wide",
)


st.markdown(
    """
    <style>
    :root {
        --fifa-blue: #0b3d91;
        --fifa-navy: #081f3f;
        --fifa-gold: #d6a82e;
        --panel: #f7f9fc;
        --border: #d9e2ef;
    }
    .main .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f7f9fc 100%);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 18px 18px 14px;
        min-height: 112px;
        box-shadow: 0 2px 10px rgba(8, 31, 63, 0.06);
    }
    .metric-card .label {
        color: #526173;
        font-size: 0.88rem;
        font-weight: 650;
        margin-bottom: 8px;
    }
    .metric-card .value {
        color: var(--fifa-navy);
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .hero {
        border-bottom: 3px solid var(--fifa-gold);
        padding-bottom: 0.75rem;
        margin-bottom: 1rem;
    }
    .hero h1 {
        color: var(--fifa-navy);
        margin-bottom: 0.2rem;
    }
    .hero p {
        color: #526173;
        margin-top: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


FLAG_MAP = {
    "Argentina": "🇦🇷",
    "Australia": "🇦🇺",
    "Belgium": "🇧🇪",
    "Brazil": "🇧🇷",
    "Canada": "🇨🇦",
    "Colombia": "🇨🇴",
    "Croatia": "🇭🇷",
    "Denmark": "🇩🇰",
    "England": "🏴",
    "France": "🇫🇷",
    "Germany": "🇩🇪",
    "Italy": "🇮🇹",
    "Japan": "🇯🇵",
    "Mexico": "🇲🇽",
    "Morocco": "🇲🇦",
    "Netherlands": "🇳🇱",
    "Portugal": "🇵🇹",
    "Spain": "🇪🇸",
    "United States": "🇺🇸",
}


def flag(team: str) -> str:
    return FLAG_MAP.get(team, "🏳️")


def card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def warn_missing(path: Path) -> None:
    st.warning(f"Missing file: `{path.relative_to(BASE_DIR)}`")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    try:
        bundle = joblib.load(MODEL_PATH)
    except Exception:
        with open(MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)

    if isinstance(bundle, dict):
        model = bundle.get("model")
        features = bundle.get("features")
        name = bundle.get("name", "trained_model")
    else:
        model = bundle
        features = None
        name = "trained_model"

    if model is None:
        raise ValueError("Model bundle does not contain a `model` object.")
    if not features:
        raise ValueError(
            "Model bundle does not contain expected feature order.")

    return model, list(features), name


@st.cache_data
def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_lookup_tables():
    return {
        "elo": read_csv(FEATURE_STORE / "elo_ratings.csv"),
        "form": read_csv(FEATURE_STORE / "team_form.csv"),
        "rank": read_csv(FEATURE_STORE / "fifa_rankings_latest.csv"),
        "profiles": read_csv(FEATURE_STORE / "team_profiles.csv"),
        "team_rankings": read_csv(DASHBOARD_DIR / "team_rankings.csv"),
    }


def normalize_team_rankings(tables: dict) -> pd.DataFrame:
    team_rankings = tables["team_rankings"].copy()
    if not team_rankings.empty:
        return coerce_ranking_columns(team_rankings)

    profiles = tables["profiles"].copy()
    if not profiles.empty:
        df = profiles.rename(
            columns={
                "elo_rating": "elo",
                "form_score": "form",
                "fifa_rank": "fifa_rank",
                "win_pct": "win_pct",
                "goals_scored_avg": "avg_goals_scored",
                "goals_conceded_avg": "avg_goals_conceded",
                "avg_overall": "player_strength",
            }
        )
        if "player_strength" not in df.columns:
            strength_cols = [c for c in [
                "avg_overall", "avg_attack_overall", "avg_defense_overall"] if c in df.columns]
            df["player_strength"] = df[strength_cols].mean(
                axis=1) if strength_cols else DEFAULT_PLAYER_STRENGTH
        return coerce_ranking_columns(df)

    elo = tables["elo"].copy()
    form = tables["form"].copy()
    rank = tables["rank"].copy()
    if elo.empty and form.empty:
        return pd.DataFrame()

    df = elo.rename(columns={"elo_rating": "elo"})
    if not form.empty:
        df = df.merge(
            form.rename(columns={"form_score": "form", "win_pct": "win_pct"}),
            on="team",
            how="outer",
        )
    if not rank.empty:
        df = df.merge(rank[["team", "fifa_rank"]], on="team", how="outer")
    return coerce_ranking_columns(df)


def coerce_ranking_columns(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "elo",
        "elo_rating",
        "form",
        "form_score",
        "fifa_rank",
        "fifa_points",
        "win_pct",
        "avg_goals_scored",
        "avg_goals_conceded",
        "player_strength",
        "team_strength_score",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def available_teams(rankings: pd.DataFrame, tables: dict) -> list[str]:
    teams = set()
    for df in [rankings, tables["elo"], tables["form"], tables["rank"], tables["profiles"]]:
        if not df.empty and "team" in df.columns:
            teams.update(df["team"].dropna().astype(str).tolist())
    return sorted(teams)


def row_for(df: pd.DataFrame, team: str) -> dict:
    if df.empty or "team" not in df.columns:
        return {}
    rows = df[df["team"].astype(str) == team]
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


def value_from(row: dict, keys: list[str], default: float) -> float:
    for key in keys:
        if key in row and pd.notna(row[key]):
            return float(row[key])
    return float(default)


def team_stats(team: str, tables: dict, rankings: pd.DataFrame) -> dict:
    ranking_row = row_for(rankings, team)
    elo_row = row_for(tables["elo"], team)
    form_row = row_for(tables["form"], team)
    rank_row = row_for(tables["rank"], team)
    profile_row = row_for(tables["profiles"], team)

    merged = {}
    for source in [profile_row, form_row, rank_row, elo_row, ranking_row]:
        merged.update({k: v for k, v in source.items() if pd.notna(v)})

    strength_cols = ["player_strength", "avg_overall", "team_strength_score"]
    if not any(col in merged for col in strength_cols):
        attack = value_from(
            merged, ["avg_attack_overall"], DEFAULT_PLAYER_STRENGTH)
        defense = value_from(
            merged, ["avg_defense_overall"], DEFAULT_PLAYER_STRENGTH)
        merged["player_strength"] = (attack + defense) / 2.0

    return {
        "elo": value_from(merged, ["elo", "elo_rating"], DEFAULT_ELO),
        "form": value_from(merged, ["form", "form_score"], DEFAULT_FORM),
        "fifa_rank": value_from(merged, ["fifa_rank", "rank"], DEFAULT_RANK),
        "win_pct": value_from(merged, ["win_pct", "home_win_percentage"], DEFAULT_WIN_PCT),
        "avg_goals_scored": value_from(
            merged,
            ["avg_goals_scored", "goals_scored_avg", "home_avg_goals_scored"],
            DEFAULT_GOALS,
        ),
        "avg_goals_conceded": value_from(
            merged,
            ["avg_goals_conceded", "goals_conceded_avg", "home_avg_goals_conceded"],
            DEFAULT_GOALS,
        ),
        "player_strength": value_from(merged, strength_cols, DEFAULT_PLAYER_STRENGTH),
    }


def team_label(team: str) -> str:
    return f"{flag(team)} {team}"


CONFEDERATION_MAP = {
    "Argentina": "CONMEBOL",
    "Brazil": "CONMEBOL",
    "Colombia": "CONMEBOL",
    "Uruguay": "CONMEBOL",
    "England": "UEFA",
    "France": "UEFA",
    "Germany": "UEFA",
    "Spain": "UEFA",
    "Portugal": "UEFA",
    "Belgium": "UEFA",
    "Netherlands": "UEFA",
    "Croatia": "UEFA",
    "Italy": "UEFA",
    "Mexico": "CONCACAF",
    "United States": "CONCACAF",
    "Canada": "CONCACAF",
    "Japan": "AFC",
    "South Korea": "AFC",
    "Australia": "AFC",
    "Saudi Arabia": "AFC",
    "Iran": "AFC",
    "Morocco": "CAF",
    "Nigeria": "CAF",
    "Cameroon": "CAF",
    "Senegal": "CAF",
    "Algeria": "CAF",
    "Egypt": "CAF",
}


def team_dropdown_options(teams: list[str]) -> list[str]:
    return [team_label(team) for team in teams]


def option_to_team(option: str) -> str:
    return option.split(" ", 1)[1] if " " in option else option


def confederation_of(team: str) -> str:
    return CONFEDERATION_MAP.get(team, "Other")


def build_team_profile(team: str, tables: dict, rankings: pd.DataFrame, probs: pd.DataFrame) -> dict:
    stats = team_stats(team, tables, rankings)
    profile = row_for(tables["profiles"], team)
    probability = row_for(probs, team).get("championship_probability", 0.0)
    attack = value_from(
        profile, ["avg_attack_overall"], stats["player_strength"])
    defense = value_from(
        profile, ["avg_defense_overall"], stats["player_strength"])

    if attack <= 0:
        attack = min(max(stats["avg_goals_scored"] * 35.0, 35.0), 90.0)
    if defense <= 0:
        defense = min(
            max((3.5 - stats["avg_goals_conceded"]) * 22.0, 30.0), 90.0)

    return {
        "team": team,
        "label": team_label(team),
        "confederation": confederation_of(team),
        "elo": stats["elo"],
        "form": stats["form"],
        "fifa_rank": stats["fifa_rank"],
        "fifa_points": value_from(profile, ["fifa_points"], 0.0),
        "win_pct": stats["win_pct"],
        "avg_goals_scored": stats["avg_goals_scored"],
        "avg_goals_conceded": stats["avg_goals_conceded"],
        "player_strength": stats["player_strength"],
        "team_strength_score": value_from(profile, ["team_strength_score"], stats["player_strength"]),
        "attack": attack,
        "defense": defense,
        "championship_probability": float(probability),
    }


def compute_power_index(rankings: pd.DataFrame, probs: pd.DataFrame) -> pd.DataFrame:
    df = rankings.copy()
    if df.empty:
        return df

    df = df.dropna(subset=["elo", "form", "player_strength"]).copy()
    df["championship_probability"] = df["team"].map(
        probs.set_index("team")["championship_probability"]).fillna(0.0)
    df["fifa_points"] = df.get("fifa_points", 0.0).fillna(0.0)
    df["win_pct"] = df["win_pct"].fillna(0.0)
    df["fifa_rank"] = df["fifa_rank"].fillna(
        df["fifa_rank"].max() if "fifa_rank" in df.columns else DEFAULT_RANK)

    def normalize(series: pd.Series) -> pd.Series:
        minimum = series.min()
        maximum = series.max()
        if minimum == maximum:
            return pd.Series(1.0, index=series.index)
        return (series - minimum) / (maximum - minimum)

    df["elo_score"] = normalize(df["elo"])
    df["form_score_norm"] = normalize(df["form"])
    df["player_score_norm"] = normalize(df["player_strength"])
    df["win_score_norm"] = normalize(df["win_pct"])
    df["rank_score_norm"] = 1.0 - normalize(df["fifa_rank"])

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

    df["elo_rank"] = df["elo"].rank(method="min", ascending=False)
    df["form_rank"] = df["form"].rank(method="min", ascending=False)
    df["power_rank"] = df["power_index"].rank(method="min", ascending=False)
    return df


def describe_advantage(base: dict, challenger: dict) -> list[str]:
    bullets = []
    if base["elo"] > challenger["elo"]:
        bullets.append("Higher ELO rating")
    if base["fifa_rank"] < challenger["fifa_rank"]:
        bullets.append("Better FIFA ranking")
    if base["form"] > challenger["form"]:
        bullets.append("Stronger recent form")
    if base["win_pct"] > challenger["win_pct"]:
        bullets.append("Higher win percentage")
    if base["player_strength"] > challenger["player_strength"]:
        bullets.append("Stronger squad rating")
    if base["team_strength_score"] > challenger["team_strength_score"]:
        bullets.append("Higher composite team strength")
    return bullets


def render_summary_cards(team_a: dict, team_b: dict) -> None:
    left, right = st.columns(2)
    with left:
        card("Team", team_a["label"])
        card("Championship Chance",
             f"{team_a['championship_probability']:.2f}%")
        card("World Cup Power Index", f"{team_a['power_index']:.1f}")
    with right:
        card("Team", team_b["label"])
        card("Championship Chance",
             f"{team_b['championship_probability']:.2f}%")
        card("World Cup Power Index", f"{team_b['power_index']:.1f}")


def render_team_comparison(team_a: dict, team_b: dict) -> None:
    metrics = [
        ("ELO Rating", "elo"),
        ("FIFA Rank", "fifa_rank"),
        ("FIFA Points", "fifa_points"),
        ("Form Score", "form"),
        ("Win %", "win_pct"),
        ("Avg Goals Scored", "avg_goals_scored"),
        ("Avg Goals Conceded", "avg_goals_conceded"),
        ("Player Strength", "player_strength"),
        ("Strength Score", "team_strength_score"),
    ]
    for label, key in metrics:
        col1, col2 = st.columns(2)
        with col1:
            card(label, f"{team_a.get(key, 0):,.2f}" if isinstance(
                team_a.get(key), float) else f"{team_a.get(key)}")
        with col2:
            card(label, f"{team_b.get(key, 0):,.2f}" if isinstance(
                team_b.get(key), float) else f"{team_b.get(key)}")


def render_radar_chart(team_a: dict, team_b: dict) -> None:
    metrics = [
        ("Attack", "attack", 30, 90),
        ("Defense", "defense", 30, 90),
        ("Form", "form", 0, 20),
        ("ELO", "elo", 1200, 2100),
        ("Player Strength", "player_strength", 50, 90),
        ("FIFA Ranking", "fifa_rank", 200, 1),
    ]
    rows = []
    for name, key, minimum, maximum in metrics:
        for team in (team_a, team_b):
            value = float(team.get(key, 0.0))
            if minimum < maximum:
                score = 100.0 * \
                    min(max((value - minimum) / (maximum - minimum), 0.0), 1.0)
            else:
                score = 100.0 * \
                    min(max((minimum - value) / (minimum - maximum), 0.0), 1.0)
            rows.append({"metric": name, "score": score, "team": team["team"]})
    radar_df = pd.DataFrame(rows)
    fig = px.line_polar(
        radar_df,
        r="score",
        theta="metric",
        color="team",
        line_close=True,
        markers=True,
        title="Team Comparison Radar",
    )
    fig.update_layout(height=520, legend_title_text="Team")
    st.plotly_chart(fig, use_container_width=True)


def render_probability_comparison(team_a: dict, team_b: dict) -> None:
    diff = team_a["championship_probability"] - \
        team_b["championship_probability"]
    direction = "+" if diff >= 0 else ""
    cols = st.columns(3)
    with cols[0]:
        card(team_a["label"], f"{team_a['championship_probability']:.2f}%")
    with cols[1]:
        card("Difference", f"{direction}{diff:.2f}%")
    with cols[2]:
        card(team_b["label"], f"{team_b['championship_probability']:.2f}%")


def render_why_favored(team_a: dict, team_b: dict) -> None:
    advantage = describe_advantage(team_a, team_b)
    disadvantage = describe_advantage(team_b, team_a)
    st.subheader("Why This Team is Favored")
    if advantage:
        st.markdown(f"**{team_a['team']} is favored because:**")
        for item in advantage:
            st.markdown(f"- {item}")
    else:
        st.markdown(
            f"**{team_a['team']} and {team_b['team']} are closely matched.**")
    if disadvantage:
        st.markdown(f"**{team_b['team']} strengths:**")
        for item in disadvantage:
            st.markdown(f"- {item}")


def render_title_journey(team: dict, team_label: str) -> None:
    cols = st.columns(5)
    with cols[0]:
        card("Global Rank", f"{int(team.get('power_rank', 0))}")
    with cols[1]:
        card("Championship%",
             f"{team.get('championship_probability', 0.0):.2f}%")
    with cols[2]:
        card("ELO Rank", f"{int(team.get('elo_rank', 0))}")
    with cols[3]:
        card("FIFA Rank", f"{int(team.get('fifa_rank', 0))}")
    with cols[4]:
        card("Form Rank", f"{int(team.get('form_rank', 0))}")


def render_contender_table(power_df: pd.DataFrame) -> None:
    st.subheader("Top 10 Contenders")
    table = power_df.sort_values(
        "championship_probability", ascending=False).head(10).copy()
    table["Team"] = table["team"].map(flag).fillna("") + " " + table["team"]
    table = table[["Team", "championship_probability", "elo",
                   "fifa_rank", "form", "team_strength_score", "confederation"]]
    table = table.rename(
        columns={
            "championship_probability": "Championship Probability",
            "elo": "ELO",
            "fifa_rank": "FIFA Rank",
            "form": "Form",
            "team_strength_score": "Strength Score",
        }
    )
    query = st.text_input("Filter contenders",
                          placeholder="Search team or confederation...")
    if query:
        table = table[table.apply(
            lambda row: query.lower() in str(row).lower(), axis=1)]
    st.dataframe(table, use_container_width=True, height=420)


def render_power_index(power_df: pd.DataFrame) -> None:
    st.subheader("World Cup Power Index")
    top20 = power_df.sort_values("power_index", ascending=False).head(20)
    fig = px.bar(
        top20,
        x="team",
        y="power_index",
        color="power_index",
        color_continuous_scale="Blues",
        title="Top 20 World Cup Favorites by Power Index",
    )
    fig.update_layout(yaxis_title="Power Index")
    st.plotly_chart(fig, use_container_width=True)


def render_interactive_plots(power_df: pd.DataFrame) -> None:
    st.subheader("Interactive World Cup Visuals")
    bubble = px.scatter(
        power_df,
        x="elo",
        y="championship_probability",
        size="player_strength",
        color="confederation",
        hover_name="team",
        title="ELO vs Championship Probability",
        labels={"elo": "ELO Rating",
                "championship_probability": "Championship Probability (%)"},
    )
    bubble.update_traces(opacity=0.8, marker=dict(
        line=dict(width=1, color="#0e2746")))
    scatter = px.scatter(
        power_df,
        x="elo",
        y="form",
        size="player_strength",
        color="confederation",
        hover_name="team",
        title="ELO vs Form Score",
        labels={"elo": "ELO Rating", "form": "Form Score"},
    )
    scatter.update_traces(opacity=0.8, marker=dict(
        line=dict(width=1, color="#0e2746")))

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.plotly_chart(bubble, use_container_width=True)
    with chart_cols[1]:
        st.plotly_chart(scatter, use_container_width=True)


def construct_feature_row(home_team: str, away_team: str, neutral_flag: int, expected_features: list[str], tables: dict, rankings: pd.DataFrame) -> pd.DataFrame:
    home = team_stats(home_team, tables, rankings)
    away = team_stats(away_team, tables, rankings)

    aliases = {
        "home_elo": home["elo"],
        "away_elo": away["elo"],
        "elo_diff": home["elo"] - away["elo"],
        "home_form": home["form"],
        "away_form": away["form"],
        "form_diff": home["form"] - away["form"],
        "home_fifa_rank": home["fifa_rank"],
        "away_fifa_rank": away["fifa_rank"],
        "ranking_diff": away["fifa_rank"] - home["fifa_rank"],
        "home_rank": home["fifa_rank"],
        "away_rank": away["fifa_rank"],
        "home_win_percentage": home["win_pct"],
        "away_win_percentage": away["win_pct"],
        "win_percentage_diff": home["win_pct"] - away["win_pct"],
        "home_win_pct": home["win_pct"],
        "away_win_pct": away["win_pct"],
        "win_pct_diff": home["win_pct"] - away["win_pct"],
        "home_avg_goals_scored": home["avg_goals_scored"],
        "away_avg_goals_scored": away["avg_goals_scored"],
        "goals_scored_diff": home["avg_goals_scored"] - away["avg_goals_scored"],
        "home_avg_goals": home["avg_goals_scored"],
        "away_avg_goals": away["avg_goals_scored"],
        "goal_diff": home["avg_goals_scored"] - away["avg_goals_scored"],
        "home_avg_goals_conceded": home["avg_goals_conceded"],
        "away_avg_goals_conceded": away["avg_goals_conceded"],
        "goals_conceded_diff": home["avg_goals_conceded"] - away["avg_goals_conceded"],
        "home_player_strength": home["player_strength"],
        "away_player_strength": away["player_strength"],
        "player_strength_diff": home["player_strength"] - away["player_strength"],
        "neutral_match_flag": neutral_flag,
    }

    missing = [feature for feature in expected_features if feature not in aliases]
    if missing:
        raise ValueError(f"Cannot construct model features: {missing}")

    return pd.DataFrame([{feature: aliases[feature] for feature in expected_features}])


def probability_frame(home_team: str, away_team: str, probabilities) -> pd.DataFrame:
    class_probs = {0: 0.0, 1: 0.0, 2: 0.0}
    for class_id, prob in enumerate(probabilities):
        class_probs[class_id] = float(prob)
    return pd.DataFrame(
        {
            "Outcome": [f"{home_team} Win", "Draw", f"{away_team} Win"],
            "Probability": [
                class_probs.get(2, 0.0) * 100,
                class_probs.get(1, 0.0) * 100,
                class_probs.get(0, 0.0) * 100,
            ],
        }
    )


def render_feature_importance():
    st.subheader("Feature Importance")
    fi = read_csv(DASHBOARD_DIR / "feature_importance.csv")
    if fi.empty:
        warn_missing(DASHBOARD_DIR / "feature_importance.csv")
        return

    top = fi.sort_values("importance", ascending=False).head(
        20).sort_values("importance")
    fig = px.bar(
        top,
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale=["#7aa6d9", "#0b3d91"],
        title="Top 20 Model Drivers",
    )
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        "Higher importance means the feature contributed more to Random Forest and XGBoost split decisions. "
        "These values describe model behavior, not guaranteed causal effects."
    )


def render_match_predictor(tables: dict, rankings: pd.DataFrame):
    st.subheader("Match Predictor")

    try:
        model, expected_features, model_name = load_model()
    except Exception as exc:
        st.error(f"Could not load model: {exc}")
        return

    teams = available_teams(rankings, tables)
    if len(teams) < 2:
        st.warning("Not enough teams available to build match predictions.")
        return

    left, right, neutral_col = st.columns([1, 1, 0.7])
    with left:
        default_home = teams.index("Argentina") if "Argentina" in teams else 0
        home_team = st.selectbox("Home Team", teams, index=default_home)
    with right:
        default_away = teams.index(
            "France") if "France" in teams else min(1, len(teams) - 1)
        away_team = st.selectbox("Away Team", teams, index=default_away)
    with neutral_col:
        neutral_match = st.toggle("Neutral venue", value=True)

    if home_team == away_team:
        st.warning("Select two different teams to predict a match.")
        return

    st.caption(
        f"Model: `{model_name}` | Features loaded from `models/best_model.pkl`")

    if st.button("Predict Match", type="primary", use_container_width=True):
        try:
            X = construct_feature_row(
                home_team,
                away_team,
                int(neutral_match),
                expected_features,
                tables,
                rankings,
            )
            if not hasattr(model, "predict_proba"):
                st.error("Loaded model does not support probability predictions.")
                return

            probs = model.predict_proba(X)[0]
            pred = int(model.predict(X)[0])
            prob_df = probability_frame(home_team, away_team, probs)

            home_prob = prob_df.loc[prob_df["Outcome"] ==
                                    f"{home_team} Win", "Probability"].iloc[0]
            draw_prob = prob_df.loc[prob_df["Outcome"]
                                    == "Draw", "Probability"].iloc[0]
            away_prob = prob_df.loc[prob_df["Outcome"] ==
                                    f"{away_team} Win", "Probability"].iloc[0]
            pred_label = {0: f"{away_team} Win", 1: "Draw",
                          2: f"{home_team} Win"}.get(pred, "Unknown")

            st.markdown(f"### Predicted Outcome: **{pred_label}**")
            c1, c2, c3 = st.columns(3)
            with c1:
                card(f"{flag(home_team)} {home_team} Win", f"{home_prob:.1f}%")
            with c2:
                card("Draw", f"{draw_prob:.1f}%")
            with c3:
                card(f"{flag(away_team)} {away_team} Win", f"{away_prob:.1f}%")

            fig = px.bar(
                prob_df,
                x="Outcome",
                y="Probability",
                text=prob_df["Probability"].map(lambda v: f"{v:.1f}%"),
                color="Outcome",
                color_discrete_sequence=["#0b3d91", "#d6a82e", "#2f6f4e"],
                title="Prediction Probabilities",
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                yaxis_range=[0, max(100, prob_df["Probability"].max() + 10)], showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Model input feature vector"):
                st.dataframe(X, use_container_width=True)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    render_feature_importance()


def render_team_rankings(rankings: pd.DataFrame):
    st.subheader("Team Rankings")
    if rankings.empty:
        st.warning("No team ranking data available.")
        return

    query = st.text_input("Search team", placeholder="Type a team name...")
    view = rankings.copy()
    if query:
        view = view[view["team"].astype(str).str.contains(
            query, case=False, na=False)]

    metric_cols = st.columns(3)
    if "elo" in rankings.columns:
        with metric_cols[0]:
            top_elo = rankings.sort_values("elo", ascending=False).iloc[0]
            card("Top ELO", f"{flag(top_elo['team'])} {top_elo['team']}")
    if "fifa_rank" in rankings.columns:
        ranked = rankings[rankings["fifa_rank"].notna()
                          ].sort_values("fifa_rank")
        if not ranked.empty:
            with metric_cols[1]:
                top_rank = ranked.iloc[0]
                card("Best FIFA Rank",
                     f"{flag(top_rank['team'])} {top_rank['team']}")
    if "form" in rankings.columns:
        with metric_cols[2]:
            top_form = rankings.sort_values("form", ascending=False).iloc[0]
            card("Best Form", f"{flag(top_form['team'])} {top_form['team']}")

    st.dataframe(view, use_container_width=True, height=360)

    chart_cols = st.columns(3)
    with chart_cols[0]:
        if "elo" in rankings.columns:
            top = rankings.sort_values("elo", ascending=False).head(15)
            fig = px.bar(top, x="team", y="elo", title="Top Teams by ELO",
                         color="elo", color_continuous_scale="Blues")
            st.plotly_chart(fig, use_container_width=True)
    with chart_cols[1]:
        if "fifa_rank" in rankings.columns:
            top = rankings[rankings["fifa_rank"].notna()].sort_values(
                "fifa_rank").head(15)
            fig = px.bar(top, x="team", y="fifa_rank", title="Top Teams by FIFA Ranking",
                         color="fifa_rank", color_continuous_scale="Cividis")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
    with chart_cols[2]:
        if "form" in rankings.columns:
            top = rankings.sort_values("form", ascending=False).head(15)
            fig = px.bar(top, x="team", y="form", title="Top Teams by Form",
                         color="form", color_continuous_scale="Greens")
            st.plotly_chart(fig, use_container_width=True)


def render_world_cup_simulation(tables: dict, rankings: pd.DataFrame):
    st.subheader("World Cup Intelligence Center")
    probs = read_csv(DASHBOARD_DIR / "championship_probabilities.csv")
    if probs.empty:
        warn_missing(DASHBOARD_DIR / "championship_probabilities.csv")
        return

    probs = probs.sort_values("championship_probability", ascending=False)
    power_df = compute_power_index(rankings, probs)
    power_df["confederation"] = power_df["team"].map(
        confederation_of).fillna("Other")

    teams = available_teams(rankings, tables)
    if len(teams) < 2:
        st.warning("Not enough teams available to compare.")
        return

    st.markdown("### Compare World Cup Teams")
    team_options = team_dropdown_options(teams)
    default_a = "Argentina" if "Argentina" in teams else teams[0]
    default_b = "France" if "France" in teams else (
        teams[1] if len(teams) > 1 else teams[0])
    left, right = st.columns(2)
    with left:
        selected_a = st.selectbox(
            "Dropdown A: Select Team",
            team_options,
            index=team_options.index(team_label(default_a)),
        )
    with right:
        selected_b = st.selectbox(
            "Dropdown B: Select Team",
            team_options,
            index=team_options.index(team_label(default_b)),
        )

    team_a = option_to_team(selected_a)
    team_b = option_to_team(selected_b)

    if team_a == team_b:
        st.warning(
            "Choose two different teams to compare their strengths and title chances.")
        return

    profile_a = build_team_profile(team_a, tables, rankings, probs)
    profile_b = build_team_profile(team_b, tables, rankings, probs)

    for profile in (profile_a, profile_b):
        if profile["team"] in power_df["team"].values:
            row = power_df.set_index("team").loc[profile["team"]]
            profile["power_index"] = float(row["power_index"])
            profile["power_rank"] = int(row["power_rank"])
            profile["elo_rank"] = int(row["elo_rank"])
            profile["form_rank"] = int(row["form_rank"])
        else:
            profile["power_index"] = 0.0
            profile["power_rank"] = 0
            profile["elo_rank"] = 0
            profile["form_rank"] = 0

    st.markdown("---")
    render_summary_cards(profile_a, profile_b)
    st.markdown("---")
    st.subheader("Head-to-Head Comparison")
    render_team_comparison(profile_a, profile_b)
    st.markdown("---")
    render_radar_chart(profile_a, profile_b)
    st.markdown("---")
    render_probability_comparison(profile_a, profile_b)
    st.markdown("---")
    render_why_favored(profile_a, profile_b)
    st.markdown("---")
    st.subheader("Team Journey to the Title")
    render_title_journey(profile_a, profile_a["label"])
    render_title_journey(profile_b, profile_b["label"])
    st.markdown("---")
    render_power_index(power_df)
    render_contender_table(power_df)
    render_interactive_plots(power_df)

    st.markdown("---")
    st.subheader("Top Championship Probabilities")
    favourites = probs.head(5)
    cols = st.columns(min(5, len(favourites)))
    for col, (_, row) in zip(cols, favourites.iterrows()):
        with col:
            card(f"{flag(row['team'])} {row['team']}",
                 f"{row['championship_probability']:.2f}%")

    top20 = probs.head(20)
    fig = px.bar(
        top20,
        x="team",
        y="championship_probability",
        color="championship_probability",
        color_continuous_scale=["#7aa6d9", "#0b3d91"],
        title="Top 20 Championship Probabilities",
    )
    fig.update_layout(yaxis_title="Win Probability (%)")
    st.plotly_chart(fig, use_container_width=True)

    chart_col, table_col = st.columns([1, 1])
    with chart_col:
        pie = px.pie(
            top20,
            names="team",
            values="championship_probability",
            title="Probability Share Among Top 20",
            hole=0.45,
        )
        st.plotly_chart(pie, use_container_width=True)
    with table_col:
        st.dataframe(probs, use_container_width=True, height=500)


def main():
    st.markdown(
        """
        <div class="hero">
            <h1>FIFA World Cup Intelligence Center</h1>
            <p>Compare teams, explore tournament favorites, and discover why countries are favored for World Cup victory.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tables = load_lookup_tables()
    rankings = normalize_team_rankings(tables)

    missing_core = [
        path for path in [
            FEATURE_STORE / "elo_ratings.csv",
            FEATURE_STORE / "team_form.csv",
            FEATURE_STORE / "fifa_rankings_latest.csv",
            MODEL_PATH,
        ]
        if not path.exists()
    ]
    if missing_core:
        with st.expander("Missing project files", expanded=True):
            for path in missing_core:
                warn_missing(path)

    tab1, tab2, tab3 = st.tabs(
        ["World Cup Simulation", "Match Predictor", "Team Rankings"])
    with tab1:
        render_world_cup_simulation(tables, rankings)
    with tab2:
        render_match_predictor(tables, rankings)
    with tab3:
        render_team_rankings(rankings)


if __name__ == "__main__":
    main()
