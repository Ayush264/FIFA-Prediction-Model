"""
⚽ FIFA World Cup 2026 Intelligence Platform
Premium Analytics Dashboard — Production Grade
"""

from pathlib import Path
from typing import Any, cast
import pickle
import json
import time

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = BASE_DIR / "dashboard"
POWERBI_DIR = BASE_DIR / "powerbi_exports"
FEATURE_STORE = BASE_DIR / "data" / "feature_store"
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
EVAL_REPORT = BASE_DIR / "models" / "evaluation_report.json"

DEFAULT_ELO = 1500.0
DEFAULT_FORM = 0.0
DEFAULT_RANK = 200
DEFAULT_GOALS = 0.0
DEFAULT_WIN_PCT = 0.0
DEFAULT_PLAYER_STRENGTH = 65.0

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FIFA WC 2026 Intelligence Platform",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# PREMIUM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── GLOBAL RESET ─────────────────────────── */
:root {
    --navy:    #081229;
    --navy2:   #0d1b35;
    --navy3:   #112244;
    --gold:    #FFD700;
    --gold2:   #e6b800;
    --blue:    #00A8FF;
    --blue2:   #0077cc;
    --green:   #00E676;
    --red:     #FF4444;
    --white:   #FFFFFF;
    --grey:    #8899aa;
    --glass:   rgba(255,255,255,0.04);
    --glass2:  rgba(255,255,255,0.08);
    --border:  rgba(255,215,0,0.15);
    --border2: rgba(0,168,255,0.2);
    --shadow:  0 8px 40px rgba(0,0,0,0.45);
    --shadow2: 0 4px 20px rgba(0,168,255,0.12);
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--navy) !important;
    color: var(--white) !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 10% 0%, #0d2060 0%, #081229 50%, #020a1a 100%) !important;
}

/* ── HIDE STREAMLIT CHROME ───────────────── */
#MainMenu, footer,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
.stDeployButton { display: none !important; }

/* The app no longer depends on Streamlit's native sidebar toggle for navigation.
   This avoids the desktop/mobile bug where Streamlit's collapsed-sidebar button
   disappears in some browser/version combinations. */
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 0 !important;
    min-height: 0 !important;
    visibility: hidden !important;
}

/* If the native sidebar is opened by Streamlit/browser state, keep it harmless. */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628 0%, #060e1e 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--white) !important; }

/* ── MAIN CONTENT ────────────────────────── */
.main .block-container {
    padding: 1.5rem 2rem 3rem !important;
    max-width: 1400px !important;
}

/* ── HERO BANNER ─────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #0d1f4a 0%, #081229 60%, #03080f 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow);
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(255,215,0,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -60px; left: 20%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(0,168,255,0.06) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.4rem;
    letter-spacing: 3px;
    background: linear-gradient(135deg, #FFD700 0%, #ffffff 50%, #00A8FF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem 0;
    line-height: 1;
}
.hero-subtitle {
    color: var(--grey);
    font-size: 1.05rem;
    font-weight: 400;
    letter-spacing: 1px;
    margin: 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,215,0,0.1);
    border: 1px solid rgba(255,215,0,0.3);
    color: var(--gold);
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 1rem;
}

/* ── METRIC CARDS ────────────────────────── */
.metric-card {
    background: var(--glass);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    backdrop-filter: blur(12px);
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    position: relative;
    overflow: hidden;
    min-height: 110px;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow2);
    border-color: rgba(0,168,255,0.35);
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--gold), var(--blue));
    opacity: 0.6;
}
.metric-card .m-label {
    color: var(--grey);
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.metric-card .m-value {
    color: var(--white);
    font-size: 1.9rem;
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.5px;
}
.metric-card .m-sub {
    color: var(--grey);
    font-size: 0.78rem;
    margin-top: 0.3rem;
}

/* ── GOLD CARD (highlight) ───────────────── */
.metric-card.gold {
    background: linear-gradient(135deg, rgba(255,215,0,0.12), rgba(255,215,0,0.04));
    border-color: rgba(255,215,0,0.4);
}
.metric-card.gold .m-value { color: var(--gold); }

/* ── SECTION HEADERS ─────────────────────── */
.section-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.8rem;
    letter-spacing: 3px;
    color: var(--white);
    margin: 2rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid rgba(255,215,0,0.2);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-title span { color: var(--gold); }

/* ── INSIGHT PANEL ───────────────────────── */
.insight-card {
    background: linear-gradient(135deg, rgba(0,168,255,0.08), rgba(0,168,255,0.03));
    border: 1px solid var(--border2);
    border-left: 3px solid var(--blue);
    border-radius: 12px;
    padding: 1rem 1.4rem;
    margin-bottom: 0.7rem;
    font-size: 0.92rem;
    color: #c0d8f0;
}
.insight-card .insight-icon { margin-right: 0.5rem; }

/* ── TEAM VS CARD ────────────────────────── */
.team-vs-card {
    background: var(--glass2);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    backdrop-filter: blur(8px);
    transition: border-color 0.2s;
}
.team-vs-card:hover { border-color: rgba(255,215,0,0.4); }
.team-flag { font-size: 4rem; display: block; margin-bottom: 0.3rem; }
.team-name {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.8rem;
    letter-spacing: 2px;
    color: var(--white);
}
.vs-badge {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3rem;
    color: var(--gold);
    letter-spacing: 4px;
    text-align: center;
    margin: auto;
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    min-height: 120px;
}

/* ── PROBABILITY BAR ─────────────────────── */
.prob-bar-wrap { margin: 0.5rem 0; }
.prob-label {
    display: flex; justify-content: space-between;
    font-size: 0.82rem; color: var(--grey);
    margin-bottom: 4px;
}
.prob-bar-bg {
    background: rgba(255,255,255,0.06);
    border-radius: 999px;
    height: 12px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.8s cubic-bezier(.4,0,.2,1);
}

/* ── PODIUM ──────────────────────────────── */
.podium-wrap { display: flex; align-items: flex-end; justify-content: center; gap: 1rem; margin: 1.5rem 0; }
.podium-item { text-align: center; flex: 1; }
.podium-block {
    border-radius: 12px 12px 0 0;
    display: flex; flex-direction: column;
    align-items: center; justify-content: flex-end;
    padding: 1rem 0.5rem;
}
.podium-1 { background: linear-gradient(180deg, rgba(255,215,0,0.2), rgba(255,215,0,0.05)); border: 1px solid rgba(255,215,0,0.4); min-height: 180px; }
.podium-2 { background: linear-gradient(180deg, rgba(192,192,192,0.15), rgba(192,192,192,0.03)); border: 1px solid rgba(192,192,192,0.25); min-height: 140px; }
.podium-3 { background: linear-gradient(180deg, rgba(205,127,50,0.15), rgba(205,127,50,0.03)); border: 1px solid rgba(205,127,50,0.25); min-height: 110px; }
.podium-flag { font-size: 2.8rem; }
.podium-name { font-family: 'Bebas Neue', sans-serif; font-size: 1.1rem; letter-spacing: 1px; color: var(--white); margin-top: 0.4rem; }
.podium-prob { font-size: 0.85rem; color: var(--gold); font-weight: 700; }

/* ── NARRATIVE BOX ───────────────────────── */
.narrative-box {
    background: linear-gradient(135deg, rgba(255,215,0,0.07), rgba(0,168,255,0.04));
    border: 1px solid rgba(255,215,0,0.2);
    border-radius: 14px;
    padding: 1.4rem 1.8rem;
    font-size: 1rem;
    line-height: 1.7;
    color: #d0e4f8;
    margin: 1rem 0;
}
.narrative-box .nar-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.1rem;
    letter-spacing: 2px;
    color: var(--gold);
    margin-bottom: 0.5rem;
}

/* ── ABOUT TECH BADGE ────────────────────── */
.tech-badge {
    display: inline-block;
    background: rgba(0,168,255,0.1);
    border: 1px solid rgba(0,168,255,0.25);
    color: var(--blue);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    padding: 4px 12px;
    border-radius: 6px;
    margin: 3px;
}

/* ── APP NAVIGATION ───────────────────────── */
.global-nav-card {
    background: linear-gradient(135deg, rgba(13,31,74,0.9), rgba(3,8,15,0.88));
    border: 1px solid rgba(255,215,0,0.22);
    border-radius: 14px;
    padding: 0.9rem 1.1rem;
    margin: 0 0 1rem 0;
    box-shadow: 0 8px 30px rgba(0,0,0,0.28);
}
.global-nav-eyebrow {
    color: var(--gold);
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
}
.global-nav-copy {
    color: var(--grey);
    font-size: 0.86rem;
}
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,215,0,0.22) !important;
    border-radius: 12px !important;
    min-height: 48px !important;
}
div[data-baseweb="select"] span,
div[data-baseweb="select"] input {
    color: var(--white) !important;
    font-weight: 700 !important;
}

/* ── PLOTLY THEME ────────────────────────── */
.js-plotly-plot .plotly { background: transparent !important; }

/* ── DATAFRAME ───────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--border) !important;
}

/* ── TABS ────────────────────────────────── */
[data-testid="stTabs"] button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--gold) !important;
    border-bottom-color: var(--gold) !important;
}

/* ── BUTTONS ─────────────────────────────── */
.stButton button {
    background: linear-gradient(135deg, #0055aa, #003377) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 10px !important;
    color: var(--white) !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s !important;
}
.stButton button:hover {
    background: linear-gradient(135deg, #0077cc, #0055aa) !important;
    border-color: var(--blue) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(0,168,255,0.25) !important;
}
.stButton button[kind="primary"] {
    background: linear-gradient(135deg, var(--gold), #e6a800) !important;
    color: var(--navy) !important;
    border-color: var(--gold) !important;
}
.stButton button[kind="primary"]:hover {
    box-shadow: 0 4px 20px rgba(255,215,0,0.35) !important;
}

/* ── INPUTS ──────────────────────────────── */
.stSelectbox > div > div,
.stTextInput > div > div > input {
    background: var(--glass2) !important;
    border-color: var(--border) !important;
    color: var(--white) !important;
    border-radius: 10px !important;
}

/* ── TOGGLE ──────────────────────────────── */
[data-testid="stToggle"] { accent-color: var(--gold); }

/* ── WARNING / INFO ──────────────────────── */
[data-testid="stAlert"] {
    background: rgba(0,168,255,0.07) !important;
    border-color: var(--border2) !important;
    border-radius: 10px !important;
    color: #b0ccee !important;
}

/* ── EXPANDER ────────────────────────────── */
[data-testid="stExpander"] {
    background: var(--glass) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

/* ── DIVIDER ─────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid rgba(255,215,0,0.12) !important;
    margin: 1.5rem 0 !important;
}

/* ── CONFED PILL ─────────────────────────── */
.confed-pill {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 2px 10px;
    border-radius: 999px;
    text-transform: uppercase;
}
.confed-UEFA   { background: rgba(0,64,255,0.2); color: #88aaff; border: 1px solid rgba(0,64,255,0.3); }
.confed-CONMEBOL { background: rgba(0,180,60,0.2); color: #80e8a0; border: 1px solid rgba(0,180,60,0.3); }
.confed-CONCACAF { background: rgba(255,120,0,0.2); color: #ffb070; border: 1px solid rgba(255,120,0,0.3); }
.confed-AFC    { background: rgba(255,0,100,0.15); color: #ff88bb; border: 1px solid rgba(255,0,100,0.2); }
.confed-CAF    { background: rgba(180,120,0,0.2); color: #ffd060; border: 1px solid rgba(180,120,0,0.3); }

/* ── RESPONSIVE / MOBILE ─────────────────── */
@media (max-width: 768px) {
    .main .block-container {
        padding: 0.8rem 0.85rem 2rem !important;
    }
    .global-nav-card { padding: 0.8rem 0.9rem !important; }
    .hero-banner { padding: 1.35rem 1.1rem !important; border-radius: 14px !important; }
    .hero-title  { font-size: 2.05rem !important; letter-spacing: 1.5px !important; }
    .hero-subtitle { font-size: 0.9rem !important; }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA & MODEL LOADING
# ─────────────────────────────────────────────
FLAG_MAP = {
    "Argentina": "🇦🇷", "Australia": "🇦🇺", "Belgium": "🇧🇪",
    "Brazil": "🇧🇷", "Canada": "🇨🇦", "Colombia": "🇨🇴",
    "Croatia": "🇭🇷", "Denmark": "🇩🇰", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "France": "🇫🇷", "Germany": "🇩🇪", "Italy": "🇮🇹",
    "Japan": "🇯🇵", "Mexico": "🇲🇽", "Morocco": "🇲🇦",
    "Netherlands": "🇳🇱", "Portugal": "🇵🇹", "Spain": "🇪🇸",
    "United States": "🇺🇸", "South Korea": "🇰🇷", "Uruguay": "🇺🇾",
    "Saudi Arabia": "🇸🇦", "Iran": "🇮🇷", "Nigeria": "🇳🇬",
    "Cameroon": "🇨🇲", "Senegal": "🇸🇳", "Algeria": "🇩🇿",
    "Egypt": "🇪🇬", "Ecuador": "🇪🇨", "Paraguay": "🇵🇾",
    "Switzerland": "🇨🇭", "Poland": "🇵🇱", "Ukraine": "🇺🇦",
    "Serbia": "🇷🇸", "Hungary": "🇭🇺", "Turkey": "🇹🇷",
}

CONFEDERATION_MAP = {
    "Argentina": "CONMEBOL", "Brazil": "CONMEBOL", "Colombia": "CONMEBOL",
    "Uruguay": "CONMEBOL", "Ecuador": "CONMEBOL", "Paraguay": "CONMEBOL",
    "England": "UEFA", "France": "UEFA", "Germany": "UEFA",
    "Spain": "UEFA", "Portugal": "UEFA", "Belgium": "UEFA",
    "Netherlands": "UEFA", "Croatia": "UEFA", "Italy": "UEFA",
    "Denmark": "UEFA", "Switzerland": "UEFA", "Poland": "UEFA",
    "Ukraine": "UEFA", "Serbia": "UEFA", "Hungary": "UEFA", "Turkey": "UEFA",
    "Mexico": "CONCACAF", "United States": "CONCACAF", "Canada": "CONCACAF",
    "Japan": "AFC", "South Korea": "AFC", "Australia": "AFC",
    "Saudi Arabia": "AFC", "Iran": "AFC",
    "Morocco": "CAF", "Nigeria": "CAF", "Cameroon": "CAF",
    "Senegal": "CAF", "Algeria": "CAF", "Egypt": "CAF",
}


def flag(team: str) -> str:
    return FLAG_MAP.get(team, "🏳️")


def confederation_of(team: str) -> str:
    return CONFEDERATION_MAP.get(team, "Other")


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
        name = bundle.get("name", "XGBoost")
    else:
        model = bundle
        features = None
        name = "trained_model"
    if model is None:
        raise ValueError("Bundle has no model.")
    if not features:
        raise ValueError("Bundle missing feature list.")
    return model, list(features), name


@st.cache_data
def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_tables():
    tr = POWERBI_DIR / "team_rankings_dashboard.csv"
    if not tr.exists():
        tr = DASHBOARD_DIR / "team_rankings.csv"
    return {
        "elo":           read_csv(FEATURE_STORE / "elo_ratings.csv"),
        "form":          read_csv(FEATURE_STORE / "team_form.csv"),
        "rank":          read_csv(FEATURE_STORE / "fifa_rankings_latest.csv"),
        "profiles":      read_csv(FEATURE_STORE / "team_profiles.csv"),
        "team_rankings": read_csv(tr),
    }


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["elo", "elo_rating", "form", "form_score", "fifa_rank", "fifa_points",
            "win_pct", "avg_goals_scored", "avg_goals_conceded", "player_strength", "team_strength_score"]
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def normalize_rankings(tables: dict) -> pd.DataFrame:
    tr = tables["team_rankings"].copy()
    if not tr.empty:
        return coerce_numeric(tr)
    profiles = tables["profiles"].copy()
    if not profiles.empty:
        df = profiles.rename(columns={
            "elo_rating": "elo", "form_score": "form", "goals_scored_avg": "avg_goals_scored",
            "goals_conceded_avg": "avg_goals_conceded", "avg_overall": "player_strength"})
        if "player_strength" not in df.columns:
            sc = [c for c in ["avg_overall", "avg_attack_overall",
                              "avg_defense_overall"] if c in df.columns]
            df["player_strength"] = df[sc].mean(
                axis=1) if sc else DEFAULT_PLAYER_STRENGTH
        return coerce_numeric(df)
    elo = tables["elo"].copy()
    form = tables["form"].copy()
    rank = tables["rank"].copy()
    if elo.empty and form.empty:
        return pd.DataFrame()
    df = elo.rename(columns={"elo_rating": "elo"})
    if not form.empty:
        df = df.merge(form.rename(
            columns={"form_score": "form", "win_pct": "win_pct"}), on="team", how="outer")
    if not rank.empty:
        df = df.merge(rank[["team", "fifa_rank"]], on="team", how="outer")
    return coerce_numeric(df)


def available_teams(rankings: pd.DataFrame, tables: dict) -> list:
    teams = set()
    for df in [rankings, tables["elo"], tables["form"], tables["rank"], tables["profiles"]]:
        if not df.empty and "team" in df.columns:
            teams.update(df["team"].dropna().astype(str).tolist())
    return sorted(teams)


def row_for(df: pd.DataFrame, team: str) -> dict:
    if df.empty or "team" not in df.columns:
        return {}
    rows = df[df["team"].astype(str) == team]
    return rows.iloc[0].to_dict() if not rows.empty else {}


def vfrom(row: dict, keys: list, default: float) -> float:
    for k in keys:
        if k in row and pd.notna(row[k]):
            return float(row[k])
    return float(default)


def team_stats(team: str, tables: dict, rankings: pd.DataFrame) -> dict:
    ranking_row = row_for(rankings, team)
    elo_row = row_for(tables["elo"], team)
    form_row = row_for(tables["form"], team)
    rank_row = row_for(tables["rank"], team)
    profile_row = row_for(tables["profiles"], team)
    merged = {}
    for src in [profile_row, form_row, rank_row, elo_row, ranking_row]:
        merged.update({k: v for k, v in src.items() if pd.notna(
            v) if not isinstance(v, float) or not np.isnan(v)})
    sc = ["player_strength", "avg_overall", "team_strength_score"]
    if not any(c in merged for c in sc):
        a = vfrom(merged, ["avg_attack_overall"], DEFAULT_PLAYER_STRENGTH)
        d = vfrom(merged, ["avg_defense_overall"], DEFAULT_PLAYER_STRENGTH)
        merged["player_strength"] = (a + d) / 2.0
    return {
        "elo":               vfrom(merged, ["elo", "elo_rating"], DEFAULT_ELO),
        "form":              vfrom(merged, ["form", "form_score"], DEFAULT_FORM),
        "fifa_rank":         vfrom(merged, ["fifa_rank", "rank"], DEFAULT_RANK),
        "win_pct":           vfrom(merged, ["win_pct", "home_win_percentage"], DEFAULT_WIN_PCT),
        "avg_goals_scored":  vfrom(merged, ["avg_goals_scored", "goals_scored_avg", "home_avg_goals_scored"], DEFAULT_GOALS),
        "avg_goals_conceded": vfrom(merged, ["avg_goals_conceded", "goals_conceded_avg", "home_avg_goals_conceded"], DEFAULT_GOALS),
        "player_strength":   vfrom(merged, sc, DEFAULT_PLAYER_STRENGTH),
    }


def compute_power_index(rankings: pd.DataFrame, probs: pd.DataFrame) -> pd.DataFrame:
    df = rankings.copy()
    if df.empty:
        return df
    df = df.dropna(subset=["elo", "form", "player_strength"]).copy()
    if not probs.empty and "team" in probs.columns and "championship_probability" in probs.columns:
        df["championship_probability"] = df["team"].map(
            probs.set_index("team")["championship_probability"]).fillna(0.0)
    else:
        df["championship_probability"] = 0.0
    df["win_pct"] = df["win_pct"].fillna(
        0.0) if "win_pct" in df.columns else 0.0
    df["fifa_rank"] = df["fifa_rank"].fillna(
        df["fifa_rank"].max() if "fifa_rank" in df.columns else DEFAULT_RANK)
    df["player_strength"] = df["player_strength"].fillna(
        DEFAULT_PLAYER_STRENGTH)

    def norm(s):
        lo, hi = s.min(), s.max()
        return pd.Series(1.0, index=s.index) if lo == hi else (s - lo) / (hi - lo)

    df["_elo_n"] = norm(df["elo"])
    df["_form_n"] = norm(df["form"])
    df["_play_n"] = norm(df["player_strength"])
    df["_win_n"] = norm(df["win_pct"])
    df["_rank_n"] = 1.0 - norm(df["fifa_rank"])
    df["power_index"] = (
        df["_elo_n"] * 0.30 + df["_form_n"] * 0.20 +
        df["_play_n"] * 0.20 + df["_win_n"] * 0.15 + df["_rank_n"] * 0.15
    ) * 100
    df["elo_rank"] = df["elo"].rank(method="min", ascending=False).astype(int)
    df["form_rank"] = df["form"].rank(
        method="min", ascending=False).astype(int)
    df["power_rank"] = df["power_index"].rank(
        method="min", ascending=False).astype(int)
    df["confederation"] = df["team"].map(confederation_of).fillna("Other")
    return df


def build_profile(team: str, tables: dict, rankings: pd.DataFrame, probs: pd.DataFrame) -> dict:
    stats = team_stats(team, tables, rankings)
    profile = row_for(tables["profiles"], team)
    probability = row_for(probs, team).get(
        "championship_probability", 0.0) if not probs.empty else 0.0
    atk = vfrom(profile, ["avg_attack_overall"], stats["player_strength"])
    dfn = vfrom(profile, ["avg_defense_overall"], stats["player_strength"])
    if atk <= 0:
        atk = min(max(stats["avg_goals_scored"] * 35.0, 35.0), 90.0)
    if dfn <= 0:
        dfn = min(max((3.5 - stats["avg_goals_conceded"]) * 22.0, 30.0), 90.0)
    return {
        "team": team, "flag": flag(team), "confederation": confederation_of(team),
        "elo": stats["elo"], "form": stats["form"], "fifa_rank": stats["fifa_rank"],
        "fifa_points": vfrom(profile, ["fifa_points"], 0.0),
        "win_pct": stats["win_pct"],
        "avg_goals_scored": stats["avg_goals_scored"],
        "avg_goals_conceded": stats["avg_goals_conceded"],
        "player_strength": stats["player_strength"],
        "team_strength_score": vfrom(profile, ["team_strength_score"], stats["player_strength"]),
        "attack": atk, "defense": dfn,
        "championship_probability": float(probability),
        "power_index": 0.0, "power_rank": 0, "elo_rank": 0, "form_rank": 0,
    }


def inject_power(profile: dict, power_df: pd.DataFrame) -> dict:
    if profile["team"] in power_df["team"].values:
        row = power_df.set_index("team").loc[profile["team"]]
        profile["power_index"] = float(row["power_index"])
        profile["power_rank"] = int(row["power_rank"])
        profile["elo_rank"] = int(row["elo_rank"])
        profile["form_rank"] = int(row["form_rank"])
    return profile


def construct_feature_row(home: str, away: str, neutral: int,
                          features: list, tables: dict, rankings: pd.DataFrame) -> pd.DataFrame:
    h = team_stats(home, tables, rankings)
    a = team_stats(away, tables, rankings)
    aliases = {
        "home_elo": h["elo"], "away_elo": a["elo"],
        "elo_diff": h["elo"] - a["elo"],
        "home_form": h["form"], "away_form": a["form"],
        "form_diff": h["form"] - a["form"],
        "home_fifa_rank": h["fifa_rank"], "away_fifa_rank": a["fifa_rank"],
        "ranking_diff": a["fifa_rank"] - h["fifa_rank"],
        "home_rank": h["fifa_rank"], "away_rank": a["fifa_rank"],
        "home_win_percentage": h["win_pct"], "away_win_percentage": a["win_pct"],
        "win_percentage_diff": h["win_pct"] - a["win_pct"],
        "home_win_pct": h["win_pct"], "away_win_pct": a["win_pct"],
        "win_pct_diff": h["win_pct"] - a["win_pct"],
        "home_avg_goals_scored": h["avg_goals_scored"], "away_avg_goals_scored": a["avg_goals_scored"],
        "goals_scored_diff": h["avg_goals_scored"] - a["avg_goals_scored"],
        "home_avg_goals": h["avg_goals_scored"], "away_avg_goals": a["avg_goals_scored"],
        "goal_diff": h["avg_goals_scored"] - a["avg_goals_scored"],
        "home_avg_goals_conceded": h["avg_goals_conceded"], "away_avg_goals_conceded": a["avg_goals_conceded"],
        "goals_conceded_diff": h["avg_goals_conceded"] - a["avg_goals_conceded"],
        "home_player_strength": h["player_strength"], "away_player_strength": a["player_strength"],
        "player_strength_diff": h["player_strength"] - a["player_strength"],
        "neutral_match_flag": neutral,
    }
    missing = [f for f in features if f not in aliases]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    return pd.DataFrame([{f: aliases[f] for f in features}])


# ─────────────────────────────────────────────
# PLOTLY DARK THEME
# ─────────────────────────────────────────────
PLOTLY_LAYOUT: dict[str, Any] = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#8899aa", size=12),
    # Plotly font objects only accept a limited set of properties.
    title_font=dict(family="Bebas Neue", color="#ffffff", size=20),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)",
               linecolor="rgba(255,255,255,0.1)", tickfont=dict(color="#8899aa")),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)",
               linecolor="rgba(255,255,255,0.1)", tickfont=dict(color="#8899aa")),
    margin=dict(l=10, r=10, t=55, b=10),
)


def apply_plotly_layout(fig: go.Figure, **overrides: Any) -> go.Figure:
    """Apply the shared Plotly theme with optional per-chart overrides."""
    layout = dict(PLOTLY_LAYOUT)
    layout.update(overrides)
    fig.update_layout(**layout)
    return fig


def select_team(label: str, teams: list[str], index: int) -> str:
    """Wrap Streamlit selectbox so Pylance knows the return value is a string."""
    return cast(str, st.selectbox(label, teams, index=index))


def styled_fig(fig):
    return apply_plotly_layout(fig)

# ─────────────────────────────────────────────
# REUSABLE COMPONENTS
# ─────────────────────────────────────────────


def metric_card(label: str, value: str, sub: str = "", gold: bool = False):
    cls = "metric-card gold" if gold else "metric-card"
    st.markdown(f"""
    <div class="{cls}">
        <div class="m-label">{label}</div>
        <div class="m-value">{value}</div>
        {"<div class='m-sub'>" + sub + "</div>" if sub else ""}
    </div>""", unsafe_allow_html=True)


def section_title(icon: str, text: str, highlight: str = ""):
    full = f"{icon} {text}" + \
        (f" <span>{highlight}</span>" if highlight else "")
    st.markdown(
        f'<div class="section-title">{full}</div>', unsafe_allow_html=True)


def insight(text: str, icon: str = "💡"):
    st.markdown(f'<div class="insight-card"><span class="insight-icon">{icon}</span>{text}</div>',
                unsafe_allow_html=True)


def prob_bar(label: str, pct: float, color: str = "#00A8FF"):
    st.markdown(f"""
    <div class="prob-bar-wrap">
        <div class="prob-label"><span>{label}</span><span>{pct:.1f}%</span></div>
        <div class="prob-bar-bg">
            <div class="prob-bar-fill" style="width:{min(pct, 100):.1f}%;background:{color}"></div>
        </div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE 1 — EXECUTIVE DASHBOARD
# ─────────────────────────────────────────────


def page_executive(tables, rankings):
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge">⚽ WORLD CUP 2026 — USA · CANADA · MEXICO</div>
        <h1 class="hero-title">FIFA WORLD CUP 2026<br>INTELLIGENCE PLATFORM</h1>
        <p class="hero-subtitle">AI-Powered Match Prediction · Team Intelligence · Tournament Simulation</p>
    </div>""", unsafe_allow_html=True)

    probs = read_csv(DASHBOARD_DIR / "championship_probabilities.csv")
    power_df = compute_power_index(rankings, probs)

    # ── KPI Cards ──
    section_title("📊", "PLATFORM", "OVERVIEW")
    k1, k2, k3, k4 = st.columns(4)
    total_teams = len(available_teams(rankings, tables))
    with k1:
        metric_card("Total Teams Analyzed", str(total_teams)
                    if total_teams else "48", "National teams")
    with k2:
        metric_card("Matches in Dataset", "51,440",
                    "Historical records since 1872")
    with k3:
        metric_card("Model Accuracy", "56.6%",
                    "XGBoost — best 3-class", gold=True)
    with k4:
        metric_card("Simulation Runs", "10,000", "Monte Carlo tournament")

    # ── Top Spotlight ──
    if not probs.empty and not power_df.empty:
        section_title("🏆", "TOURNAMENT", "SPOTLIGHT")
        s1, s2, s3 = st.columns(3)
        top_champ = probs.sort_values(
            "championship_probability", ascending=False).iloc[0]
        top_power = power_df.sort_values(
            "power_index", ascending=False).iloc[0]
        top_form = power_df.sort_values("form", ascending=False).iloc[0]
        with s1:
            metric_card("🥇 Top Championship Favorite",
                        f"{flag(top_champ['team'])} {top_champ['team']}",
                        f"{top_champ['championship_probability']:.2f}% win probability", gold=True)
        with s2:
            metric_card("⚡ Highest Power Index",
                        f"{flag(top_power['team'])} {top_power['team']}",
                        f"Power Index: {top_power['power_index']:.1f}")
        with s3:
            metric_card("🔥 Best Recent Form",
                        f"{flag(top_form['team'])} {top_form['team']}",
                        f"Form score: {top_form['form']:.2f}")

        # ── Podium ──
        section_title("🎖️", "CHAMPIONSHIP", "PODIUM")
        top3 = probs.head(3)
        if len(top3) >= 3:
            st.markdown("""<div class="podium-wrap">""",
                        unsafe_allow_html=True)
            items = [
                (top3.iloc[1], "podium-2", "🥈"),
                (top3.iloc[0], "podium-1", "🥇"),
                (top3.iloc[2], "podium-3", "🥉"),
            ]
            cols = st.columns([1, 1, 1])
            for col, (row, cls, medal) in zip(cols, items):
                with col:
                    st.markdown(f"""
                    <div class="podium-item">
                        <div class="podium-block {cls}">
                            <div class="podium-flag">{flag(row['team'])}</div>
                            <div style="font-size:1.5rem">{medal}</div>
                            <div class="podium-name">{row['team']}</div>
                            <div class="podium-prob">{row['championship_probability']:.2f}%</div>
                        </div>
                    </div>""", unsafe_allow_html=True)

        # ── Charts ──
        section_title("📈", "CHAMPIONSHIP", "PROBABILITIES — TOP 20")
        top20 = probs.head(20)
        fig = go.Figure(go.Bar(
            x=top20["team"], y=top20["championship_probability"],
            marker=dict(
                color=top20["championship_probability"],
                colorscale=[[0, "#003377"], [0.5, "#0077cc"], [1, "#FFD700"]],
                showscale=False,
                line=dict(color="rgba(255,255,255,0.05)", width=1)
            ),
            text=[f"{v:.1f}%" for v in top20["championship_probability"]],
            textposition="outside", textfont=dict(color="#8899aa", size=10),
            hovertemplate="<b>%{x}</b><br>%{y:.2f}%<extra></extra>"
        ))
        apply_plotly_layout(fig, yaxis_title="Win Probability (%)")
        st.plotly_chart(fig, use_container_width=True)

        # ── Power index + scatter ──
        c1, c2 = st.columns(2)
        with c1:
            section_title("⚡", "POWER", "INDEX — TOP 20")
            top20p = power_df.sort_values(
                "power_index", ascending=False).head(20)
            fig2 = go.Figure(go.Bar(
                x=top20p["power_index"], y=top20p["team"],
                orientation="h",
                marker=dict(color=top20p["power_index"],
                            colorscale=[[0, "#003377"], [1, "#00A8FF"]], showscale=False),
                hovertemplate="<b>%{y}</b><br>Power Index: %{x:.1f}<extra></extra>"
            ))
            apply_plotly_layout(fig2, yaxis=dict(
                autorange="reversed"), xaxis_title="Power Index")
            st.plotly_chart(fig2, use_container_width=True)
        with c2:
            section_title("🌍", "ELO vs", "CHAMPIONSHIP PROBABILITY")
            fig3 = px.scatter(
                power_df, x="elo", y="championship_probability",
                size="player_strength", color="confederation",
                hover_name="team",
                color_discrete_map={"UEFA": "#4488ff", "CONMEBOL": "#44dd88",
                                    "CONCACAF": "#ff8844", "AFC": "#ff44aa", "CAF": "#ffcc44"},
            )
            fig3.update_traces(opacity=0.85, marker=dict(
                line=dict(width=1, color="rgba(255,255,255,0.2)")))
            apply_plotly_layout(fig3, xaxis_title="ELO Rating",
                                yaxis_title="Championship %")
            st.plotly_chart(fig3, use_container_width=True)

        # ── Auto insights ──
        section_title("💡", "AI-GENERATED", "INSIGHTS")
        if not probs.empty:
            top1 = probs.iloc[0]
            insight(f"<b>{flag(top1['team'])} {top1['team']}</b> leads all teams with a <b>{top1['championship_probability']:.2f}%</b> projected championship probability across 10,000 Monte Carlo simulations.")
        if not power_df.empty:
            pi_top = power_df.sort_values(
                "power_index", ascending=False).iloc[0]
            insight(
                f"<b>{flag(pi_top['team'])} {pi_top['team']}</b> holds the #1 position in the composite World Cup Power Index, factoring ELO, form, squad strength, win rate, and FIFA rank.", "⚡")
            form_top = power_df.sort_values("form", ascending=False).iloc[0]
            insight(
                f"<b>{flag(form_top['team'])} {form_top['team']}</b> enters the tournament with the strongest recent form among all 2026 contenders.", "🔥")
            elo_top = power_df.sort_values("elo", ascending=False).iloc[0]
            insight(
                f"<b>{flag(elo_top['team'])} {elo_top['team']}</b> ranks #1 by ELO rating — a historically reliable predictor of international football dominance.", "🎯")

# ─────────────────────────────────────────────
# PAGE 2 — MATCH PREDICTOR
# ─────────────────────────────────────────────


def page_match_predictor(tables, rankings):
    section_title("🤖", "AI MATCH", "PREDICTOR")
    try:
        model, exp_features, model_name = load_model()
    except Exception as e:
        st.error(f"⚠️ Could not load model: {e}")
        return

    teams = available_teams(rankings, tables)
    if len(teams) < 2:
        st.warning("Not enough team data to run predictions.")
        return

    # ── Team selector ──
    c1, mid, c2 = st.columns([5, 2, 5])
    with c1:
        default_h = teams.index("Argentina") if "Argentina" in teams else 0
        home_team = select_team("🏠 Home Team", teams, default_h)
    with mid:
        st.markdown('<div class="vs-badge">VS</div>', unsafe_allow_html=True)
    with c2:
        default_a = teams.index(
            "France") if "France" in teams else min(1, len(teams)-1)
        away_team = select_team("✈️ Away Team", teams, default_a)

    nv = st.toggle("⚽ Neutral venue", value=True)
    st.caption(f"Model: `{model_name}` · Feature count: `{len(exp_features)}`")

    if home_team == away_team:
        st.warning("Please select two different teams.")
        return

    # ── Team snapshot cards ──
    probs_df = read_csv(DASHBOARD_DIR / "championship_probabilities.csv")
    power_df = compute_power_index(rankings, probs_df)
    pa = inject_power(build_profile(
        home_team, tables, rankings, probs_df), power_df)
    pb = inject_power(build_profile(
        away_team, tables, rankings, probs_df), power_df)

    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown(f"""<div class="team-vs-card">
            <span class="team-flag">{pa['flag']}</span>
            <div class="team-name">{pa['team']}</div>
            <div style="color:var(--grey);font-size:0.8rem;margin-top:0.5rem">Power Index</div>
            <div style="color:var(--gold);font-size:1.4rem;font-weight:800">{pa['power_index']:.1f}</div>
            <div style="color:var(--grey);font-size:0.78rem">FIFA Rank #{int(pa['fifa_rank'])} · ELO {pa['elo']:.0f}</div>
        </div>""", unsafe_allow_html=True)
    with a2:
        st.markdown('<div class="vs-badge">⚔️</div>', unsafe_allow_html=True)
    with a3:
        st.markdown(f"""<div class="team-vs-card">
            <span class="team-flag">{pb['flag']}</span>
            <div class="team-name">{pb['team']}</div>
            <div style="color:var(--grey);font-size:0.8rem;margin-top:0.5rem">Power Index</div>
            <div style="color:var(--gold);font-size:1.4rem;font-weight:800">{pb['power_index']:.1f}</div>
            <div style="color:var(--grey);font-size:0.78rem">FIFA Rank #{int(pb['fifa_rank'])} · ELO {pb['elo']:.0f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("⚡ RUN PREDICTION", type="primary", use_container_width=True):
        try:
            X = construct_feature_row(home_team, away_team, int(
                nv), exp_features, tables, rankings)
            if not hasattr(model, "predict_proba"):
                st.error("Model doesn't support probability predictions.")
                return
            probs = model.predict_proba(X)[0]
            pred = int(model.predict(X)[0])

            class_probs = {0: float(probs[0]) if len(probs) > 0 else 0,
                           1: float(probs[1]) if len(probs) > 1 else 0,
                           2: float(probs[2]) if len(probs) > 2 else 0}
            home_pct = class_probs.get(2, 0) * 100
            draw_pct = class_probs.get(1, 0) * 100
            away_pct = class_probs.get(0, 0) * 100
            pred_label = {0: f"{away_team} Win", 1: "Draw",
                          2: f"{home_team} Win"}.get(pred, "Unknown")

            # ── Outcome banner ──
            if pred == 2:
                color, winner = "var(--gold)", home_team
            elif pred == 0:
                color, winner = "var(--blue)", away_team
            else:
                color, winner = "var(--grey)", "Draw"
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.04);border:1px solid {color};border-radius:16px;
                        padding:1.5rem;text-align:center;margin:1rem 0">
                <div style="color:var(--grey);font-size:0.8rem;letter-spacing:2px;text-transform:uppercase">PREDICTED OUTCOME</div>
                <div style="font-family:'Bebas Neue';font-size:2.8rem;color:{color};letter-spacing:3px;margin:0.2rem 0">{pred_label}</div>
            </div>""", unsafe_allow_html=True)

            # ── Probability bars ──
            st.markdown("<br>", unsafe_allow_html=True)
            prob_bar(f"{flag(home_team)} {home_team} Win", home_pct, "#FFD700")
            prob_bar("Draw", draw_pct, "#8899aa")
            prob_bar(f"{flag(away_team)} {away_team} Win", away_pct, "#00A8FF")

            # ── Gauge chart ──
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=home_pct,
                title=dict(text=f"<b>{home_team}</b> Win Probability",
                           font=dict(color="#ffffff", family="Inter")),
                gauge=dict(
                    axis=dict(range=[0, 100], tickfont=dict(color="#8899aa")),
                    bar=dict(color="#FFD700"),
                    bgcolor="rgba(255,255,255,0.04)",
                    borderwidth=1, bordercolor="rgba(255,255,255,0.1)",
                    steps=[
                        dict(range=[0, 33], color="rgba(0,168,255,0.1)"),
                        dict(range=[33, 66], color="rgba(255,255,255,0.04)"),
                        dict(range=[66, 100], color="rgba(255,215,0,0.08)"),
                    ],
                    threshold=dict(line=dict(color="#00E676",
                                   width=2), thickness=0.75, value=50)
                ),
                number=dict(suffix="%", font=dict(
                    color="#FFD700", family="Bebas Neue", size=40))
            ))
            apply_plotly_layout(fig, height=300)
            st.plotly_chart(fig, use_container_width=True)

            # ── Match narrative ──
            elo_gap = pa["elo"] - pb["elo"]
            form_adv = pa["form"] > pb["form"]
            rank_adv = pa["fifa_rank"] < pb["fifa_rank"]
            stronger = home_team if (
                elo_gap > 0 and home_pct > away_pct) else away_team
            weaker = away_team if stronger == home_team else home_team
            narrative = f"""
            <b>{flag(pa['team'])} {pa['team']}</b> enters this match with a Power Index of <b>{pa['power_index']:.1f}</b>,
            {"ranked higher" if rank_adv else "facing a higher-ranked opponent"} at FIFA #<b>{int(pa['fifa_rank'])}</b>.
            {"Recent form favors " + home_team if form_adv else away_team + " arrives in stronger form"}, while
            the ELO gap of <b>{abs(elo_gap):.0f}</b> points gives {"an edge to " + home_team if elo_gap > 0 else away_team + " the advantage"}.
            The model predicts <b>{pred_label}</b> — with <b>{max(home_pct, away_pct, draw_pct):.1f}%</b> probability assigned to this outcome.
            """
            st.markdown(f'<div class="narrative-box"><div class="nar-title">📝 MATCH NARRATIVE</div>{narrative}</div>',
                        unsafe_allow_html=True)

            with st.expander("🔬 Feature Vector (Model Input)"):
                st.dataframe(X, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # ── Feature importance ──
    section_title("🎯", "MODEL FEATURE", "IMPORTANCE")
    fi = read_csv(DASHBOARD_DIR / "feature_importance.csv")
    if not fi.empty:
        top = fi.sort_values("importance", ascending=False).head(
            15).sort_values("importance")
        fig = go.Figure(go.Bar(
            x=top["importance"], y=top["feature"], orientation="h",
            marker=dict(color=top["importance"],
                        colorscale=[[0, "#003377"], [1, "#00A8FF"]], showscale=False),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
        ))
        apply_plotly_layout(fig, xaxis_title="Feature Importance")
        st.plotly_chart(fig, use_container_width=True)
        insight("ELO difference and ranking difference are the strongest predictors — reflecting real-world football dominance patterns.")
    else:
        st.info(
            "Feature importance file not found at `dashboard/feature_importance.csv`.")

# ─────────────────────────────────────────────
# PAGE 3 — TEAM INTELLIGENCE
# ─────────────────────────────────────────────


def page_team_intelligence(tables, rankings):
    section_title("🌍", "TEAM", "INTELLIGENCE CENTER")
    probs_df = read_csv(DASHBOARD_DIR / "championship_probabilities.csv")
    power_df = compute_power_index(rankings, probs_df)
    teams = available_teams(rankings, tables)
    if not teams:
        st.warning("No team data loaded.")
        return

    mode = st.radio("Mode", ["Single Team Profile",
                    "Head-to-Head Comparison"], horizontal=True)

    if mode == "Single Team Profile":
        selected = select_team("Select Team", teams,
                               teams.index("Brazil") if "Brazil" in teams else 0)
        p = inject_power(build_profile(
            selected, tables, rankings, probs_df), power_df)

        # ── Header ──
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:1.5rem;margin:1.5rem 0;
                    background:var(--glass2);border:1px solid var(--border);
                    border-radius:16px;padding:1.5rem 2rem">
            <div style="font-size:5rem">{p['flag']}</div>
            <div>
                <div style="font-family:'Bebas Neue';font-size:2.5rem;letter-spacing:3px;color:var(--white)">{p['team']}</div>
                <div><span class="confed-pill confed-{p['confederation']}">{p['confederation']}</span></div>
                <div style="color:var(--grey);margin-top:0.5rem;font-size:0.9rem">
                    FIFA Rank #{int(p['fifa_rank'])} · ELO {p['elo']:.0f} · Power Rank #{p['power_rank']}
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        # ── Key metrics ──
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            metric_card("Championship %",
                        f"{p['championship_probability']:.2f}%", gold=True)
        with m2:
            metric_card("Power Index",
                        f"{p['power_index']:.1f}", "Composite score")
        with m3:
            metric_card(
                "ELO Rating", f"{p['elo']:.0f}", f"Rank #{p['elo_rank']}")
        with m4:
            metric_card(
                "Form Score", f"{p['form']:.2f}", f"Rank #{p['form_rank']}")
        with m5:
            metric_card(
                "FIFA Rank", f"#{int(p['fifa_rank'])}", "Current ranking")

        # ── Radar ──
        metrics = [
            ("Attack",    "attack",          30, 90),
            ("Defense",   "defense",         30, 90),
            ("Form",      "form",             0, 20),
            ("ELO",       "elo",          1200, 2100),
            ("Strength",  "player_strength", 50, 90),
            ("FIFA Rank", "fifa_rank",      200,   1),
        ]
        scores, labels = [], []
        for name, key, lo, hi in metrics:
            v = float(p.get(key, 0))
            if lo < hi:
                s = 100 * min(max((v - lo)/(hi - lo), 0), 1)
            else:
                s = 100 * min(max((lo - v)/(lo - hi), 0), 1)
            scores.append(s)
            labels.append(name)
        scores.append(scores[0])
        labels.append(labels[0])  # close polygon

        fig = go.Figure(go.Scatterpolar(
            r=scores, theta=labels, fill="toself",
            fillcolor="rgba(255,215,0,0.1)",
            line=dict(color="#FFD700", width=2),
            name=selected, mode="lines+markers",
            marker=dict(color="#FFD700", size=7)
        ))
        apply_plotly_layout(
            fig,
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(
                    color="#556677"), gridcolor="rgba(255,255,255,0.06)"),
                angularaxis=dict(tickfont=dict(
                    color="#aabbcc", size=12), gridcolor="rgba(255,255,255,0.06)"),
            ),
            showlegend=False, height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Stats grid ──
        section_title("📋", "DETAILED", "STATISTICS")
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            metric_card(
                "Win %",          f"{p['win_pct']:.1f}%",        "All-time record")
        with r1c2:
            metric_card("Goals Scored",
                        f"{p['avg_goals_scored']:.2f}", "Average per match")
        with r1c3:
            metric_card("Goals Conceded",
                        f"{p['avg_goals_conceded']:.2f}", "Average per match")
        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            metric_card("Player Strength",
                        f"{p['player_strength']:.1f}",      "FIFA squad rating")
        with r2c2:
            metric_card("Team Strength",
                        f"{p['team_strength_score']:.1f}",  "Composite score")
        with r2c3:
            metric_card("FIFA Points",
                        f"{p['fifa_points']:.0f}",           "Ranking points")

    else:
        # ── Head-to-Head ──
        c1, c2 = st.columns(2)
        with c1:
            ta = select_team("Team A", teams, teams.index(
                "Argentina") if "Argentina" in teams else 0)
        with c2:
            tb = select_team("Team B", teams, teams.index(
                "France") if "France" in teams else min(1, len(teams)-1))
        if ta == tb:
            st.warning("Select two different teams.")
            return

        pa = inject_power(build_profile(
            ta, tables, rankings, probs_df), power_df)
        pb = inject_power(build_profile(
            tb, tables, rankings, probs_df), power_df)

        # Summary
        ma, mb = st.columns(2)
        with ma:
            st.markdown(f"""<div class="team-vs-card">
                <span class="team-flag">{pa['flag']}</span>
                <div class="team-name">{pa['team']}</div>
                <div style="color:var(--gold);font-size:1.5rem;font-weight:800;margin-top:0.5rem">{pa['championship_probability']:.2f}%</div>
                <div style="color:var(--grey);font-size:0.8rem">Championship probability</div>
                <div style="color:var(--white);font-size:1.2rem;margin-top:0.5rem">Power Index: <b>{pa['power_index']:.1f}</b></div>
            </div>""", unsafe_allow_html=True)
        with mb:
            st.markdown(f"""<div class="team-vs-card">
                <span class="team-flag">{pb['flag']}</span>
                <div class="team-name">{pb['team']}</div>
                <div style="color:var(--gold);font-size:1.5rem;font-weight:800;margin-top:0.5rem">{pb['championship_probability']:.2f}%</div>
                <div style="color:var(--grey);font-size:0.8rem">Championship probability</div>
                <div style="color:var(--white);font-size:1.2rem;margin-top:0.5rem">Power Index: <b>{pb['power_index']:.1f}</b></div>
            </div>""", unsafe_allow_html=True)

        # Radar comparison
        metrics = [
            ("Attack", "attack", 30, 90), ("Defense", "defense", 30, 90),
            ("Form", "form", 0, 20), ("ELO", "elo", 1200, 2100),
            ("Strength", "player_strength", 50,
             90), ("FIFA Rank", "fifa_rank", 200, 1),
        ]
        traces = []
        for p_data, color in [(pa, "#FFD700"), (pb, "#00A8FF")]:
            scores, labels = [], []
            for name, key, lo, hi in metrics:
                v = float(p_data.get(key, 0))
                s = 100 * min(max((v-lo)/(hi-lo) if lo <
                              hi else (lo-v)/(lo-hi), 0), 1)
                scores.append(s)
                labels.append(name)
            scores.append(scores[0])
            labels.append(labels[0])
            traces.append(go.Scatterpolar(
                r=scores, theta=labels, fill="toself",
                fillcolor=color.replace("#", "rgba(").replace("FFD700", "255,215,0,0.1").replace(
                    "00A8FF", "0,168,255,0.1") + ")" if color in ["#FFD700", "#00A8FF"] else "rgba(0,0,0,0.1)",
                line=dict(color=color, width=2),
                name=p_data["team"], mode="lines+markers",
                marker=dict(color=color, size=7)
            ))
        fig = go.Figure(data=traces)
        apply_plotly_layout(
            fig,
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(
                    color="#556677"), gridcolor="rgba(255,255,255,0.06)"),
                angularaxis=dict(tickfont=dict(
                    color="#aabbcc", size=12), gridcolor="rgba(255,255,255,0.06)"),
            ),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#ffffff")),
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Side-by-side stat table
        section_title("📊", "HEAD-TO-HEAD", "STATISTICS")
        metrics_table = [
            ("ELO Rating", "elo"), ("FIFA Rank",
                                    "fifa_rank"), ("Form Score", "form"),
            ("Win %", "win_pct"), ("Avg Goals Scored", "avg_goals_scored"),
            ("Avg Goals Conceded",
             "avg_goals_conceded"), ("Player Strength", "player_strength"),
            ("Championship %", "championship_probability"), ("Power Index", "power_index"),
        ]
        for label, key in metrics_table:
            va, vb = pa.get(key, 0), pb.get(key, 0)
            better_a = (va > vb and key != "fifa_rank") or (
                va < vb and key == "fifa_rank")
            ca, cb = (
                "var(--gold)" if better_a else "var(--white)"), ("var(--gold)" if not better_a else "var(--white)")
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown(
                    f"<div style='color:var(--grey);font-size:0.85rem;padding:0.4rem 0'>{label}</div>", unsafe_allow_html=True)
            with col2:
                st.markdown(
                    f"<div style='color:{ca};font-size:0.95rem;font-weight:700;padding:0.4rem 0'>{va:.2f}</div>", unsafe_allow_html=True)
            with col3:
                st.markdown(
                    f"<div style='color:{cb};font-size:0.95rem;font-weight:700;padding:0.4rem 0'>{vb:.2f}</div>", unsafe_allow_html=True)
            st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE 4 — WORLD CUP SIMULATION
# ─────────────────────────────────────────────


def page_simulation(tables, rankings):
    section_title("🏆", "WORLD CUP 2026", "SIMULATION CENTER")
    probs = read_csv(DASHBOARD_DIR / "championship_probabilities.csv")
    if probs.empty:
        st.warning(
            "No championship probabilities found. Run `simulation/monte_carlo.py` first.")
        return
    power_df = compute_power_index(rankings, probs)
    probs = probs.sort_values("championship_probability", ascending=False)

    # ── Sim controls ──
    st.markdown("""
    <div style="background:var(--glass);border:1px solid var(--border);border-radius:14px;padding:1.2rem 1.6rem;margin-bottom:1.5rem">
        <div style="color:var(--grey);font-size:0.78rem;letter-spacing:2px;text-transform:uppercase;margin-bottom:0.5rem">SIMULATION PARAMETERS</div>
        <div style="color:var(--white);font-size:0.9rem">The probabilities below reflect <b style="color:var(--gold)">10,000 Monte Carlo simulation runs</b> 
        of the complete FIFA World Cup 2026 bracket, including Group Stage (12 groups × 4 teams) → Round of 32 → Knockout rounds.</div>
    </div>""", unsafe_allow_html=True)

    run_sim = st.selectbox("Simulation Run Count (display reference)", [
                           "1,000", "5,000", "10,000 (current)", "50,000"], index=2)

    # ── Podium ──
    section_title("🎖️", "CHAMPIONSHIP", "PODIUM")
    top3 = probs.head(3)
    if len(top3) >= 3:
        cols = st.columns([1, 1.2, 1])
        order = [1, 0, 2]
        cls_map = {0: "podium-1", 1: "podium-2", 2: "podium-3"}
        medals = {0: "🥇", 1: "🥈", 2: "🥉"}
        for i, (col, idx) in enumerate(zip(cols, order)):
            row = top3.iloc[idx]
            with col:
                st.markdown(f"""<div class="podium-item">
                    <div class="podium-block {cls_map[idx]}">
                        <div style="font-size:3rem">{flag(row['team'])}</div>
                        <div style="font-size:2rem">{medals[idx]}</div>
                        <div class="podium-name">{row['team']}</div>
                        <div class="podium-prob">{row['championship_probability']:.2f}%</div>
                        <div style="color:var(--grey);font-size:0.75rem">Win Probability</div>
                    </div>
                </div>""", unsafe_allow_html=True)

    # ── Top 20 bar ──
    section_title("📊", "CHAMPIONSHIP", "PROBABILITIES")
    top20 = probs.head(20)
    fig = go.Figure(go.Bar(
        x=top20["team"], y=top20["championship_probability"],
        marker=dict(color=top20["championship_probability"],
                    colorscale=[[0, "#003377"], [0.6, "#0066cc"], [1, "#FFD700"]], showscale=False,
                    line=dict(color="rgba(255,255,255,0.05)", width=1)),
        text=[f"{v:.1f}%" for v in top20["championship_probability"]],
        textposition="outside", textfont=dict(color="#8899aa", size=10),
        hovertemplate="<b>%{x}</b><br>Win Probability: %{y:.2f}%<extra></extra>"
    ))
    apply_plotly_layout(fig, yaxis_title="Win Probability (%)")
    st.plotly_chart(fig, use_container_width=True)

    # ── Pie + table ──
    col_pie, col_table = st.columns([1, 1])
    with col_pie:
        section_title("🥧", "PROBABILITY", "SHARE")
        pie = go.Figure(go.Pie(
            labels=[f"{flag(r['team'])} {r['team']}" for _,
                    r in top20.iterrows()],
            values=top20["championship_probability"],
            hole=0.5,
            marker=dict(colors=px.colors.qualitative.Bold + px.colors.qualitative.Pastel,
                        line=dict(color="rgba(0,0,0,0.3)", width=1)),
            textfont=dict(color="#ffffff", size=10),
            hovertemplate="<b>%{label}</b><br>%{value:.2f}%<extra></extra>"
        ))
        apply_plotly_layout(pie, showlegend=False, height=400)
        st.plotly_chart(pie, use_container_width=True)
    with col_table:
        section_title("📋", "FULL", "PROBABILITY TABLE")
        display = probs.copy()
        display.insert(0, "Flag", display["team"].map(flag))
        display["championship_probability"] = display["championship_probability"].map(
            lambda v: f"{v:.2f}%")
        st.dataframe(display[["Flag", "team", "championship_probability"]].rename(
            columns={"team": "Team", "championship_probability": "Win Probability"}),
            use_container_width=True, height=400)

    # ── Tournament insights ──
    section_title("💡", "TOURNAMENT", "INSIGHTS")
    top1 = probs.iloc[0] if len(probs) else None
    if top1 is not None:
        insight(f"{flag(top1['team'])} <b>{top1['team']}</b> wins <b>{top1['championship_probability']:.1f}%</b> of simulated tournaments — the model's strongest pick for the 2026 title.")
    if len(probs) >= 3:
        top3names = ", ".join(
            [f"{flag(r['team'])} {r['team']}" for _, r in probs.head(3).iterrows()])
        insight(f"The top 3 favorites — {top3names} — collectively account for "
                f"<b>{probs.head(3)['championship_probability'].sum():.1f}%</b> of all simulated championship wins.", "📊")
    if not power_df.empty:
        eu_teams = power_df[power_df["confederation"] == "UEFA"]
        sa_teams = power_df[power_df["confederation"] == "CONMEBOL"]
        if not eu_teams.empty and not sa_teams.empty:
            eu_pct = probs[probs["team"].isin(
                eu_teams["team"])]["championship_probability"].sum()
            sa_pct = probs[probs["team"].isin(
                sa_teams["team"])]["championship_probability"].sum()
            insight(
                f"UEFA teams hold <b>{eu_pct:.1f}%</b> combined probability; CONMEBOL teams hold <b>{sa_pct:.1f}%</b> — a classic Europe vs South America WC rivalry.", "🌍")
    if not power_df.empty:
        dark_horse = probs[probs["championship_probability"] < 5].sort_values(
            "championship_probability", ascending=False)
        if not dark_horse.empty:
            dh = dark_horse.iloc[0]
            insight(f"{flag(dh['team'])} <b>{dh['team']}</b> is the dark horse to watch — competitive statistics but only <b>{dh['championship_probability']:.1f}%</b> win probability.", "🌙")

    # ── Scatter ──
    if not power_df.empty:
        section_title("🫧", "POWER INDEX vs", "CHAMPIONSHIP PROBABILITY")
        fig2 = px.scatter(
            power_df, x="power_index", y="championship_probability",
            size="player_strength", color="confederation",
            hover_name="team", text="team",
            color_discrete_map={"UEFA": "#4488ff", "CONMEBOL": "#44dd88",
                                "CONCACAF": "#ff8844", "AFC": "#ff44aa", "CAF": "#ffcc44"},
        )
        fig2.update_traces(textposition="top center", textfont=dict(color="#8899aa", size=9),
                           marker=dict(opacity=0.8, line=dict(width=1, color="rgba(255,255,255,0.15)")))
        apply_plotly_layout(fig2, xaxis_title="Power Index",
                            yaxis_title="Championship Probability (%)")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Download ──
    st.markdown("---")
    csv_bytes = probs.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Simulation Results (CSV)", csv_bytes,
                       "wc2026_championship_probabilities.csv", "text/csv", use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 5 — MODEL EXPLAINABILITY
# ─────────────────────────────────────────────


def page_model(tables, rankings):
    section_title("🧠", "MODEL", "EXPLAINABILITY")

    # ── Metrics cards ──
    st.markdown("""
    <div style="background:var(--glass);border:1px solid var(--border);border-radius:14px;padding:1.2rem 1.6rem;margin-bottom:1.5rem">
        <div style="color:var(--grey);font-size:0.78rem;letter-spacing:2px;text-transform:uppercase;margin-bottom:0.5rem">ABOUT THE MODELS</div>
        <div style="color:#c0d4ee;font-size:0.9rem">Three classifiers were trained on 51,440 historical international matches to predict match outcome (home win / draw / away win). 
        Features were engineered from ELO ratings, FIFA rankings, squad strength, form, and historical stats. 
        XGBoost achieved the best accuracy.</div>
    </div>""", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card("XGBoost Accuracy", "56.6%", "Best model", gold=True)
    with m2:
        metric_card("F1 Score (macro)", "0.462", "XGBoost")
    with m3:
        metric_card("CV Accuracy", "53.0%", "Cross-validated")
    with m4:
        metric_card("Baseline", "~43%", "Random 3-class")

    # ── Model comparison ──
    section_title("📊", "MODEL", "COMPARISON")
    model_data = pd.DataFrame({
        "Model":    ["XGBoost", "Random Forest", "Logistic Regression"],
        "Accuracy": [56.6, 54.8, 43.5],
        "F1 Score": [0.462, 0.488, 0.404],
        "CV Accuracy": [53.0, 50.0, 45.5],
    })
    fig = go.Figure()
    colors = ["#FFD700", "#00A8FF", "#aaaaaa"]
    for i, col in enumerate(["Accuracy", "F1 Score (×100)", "CV Accuracy"]):
        y = model_data["Accuracy"] if col == "Accuracy" else \
            model_data["F1 Score"] * \
            100 if "F1" in col else model_data["CV Accuracy"]
        fig.add_trace(go.Bar(
            name=col.replace(" (×100)", ""), x=model_data["Model"], y=y,
            marker_color=colors[i], opacity=0.85,
            text=[f"{v:.1f}" for v in y], textposition="outside",
            textfont=dict(color="#8899aa", size=10)
        ))
    apply_plotly_layout(fig, barmode="group", yaxis_title="Score (%)", legend=dict(
        font=dict(color="#ffffff")))
    st.plotly_chart(fig, use_container_width=True)

    # ── Feature importance ──
    section_title("🎯", "FEATURE", "IMPORTANCE")
    fi = read_csv(DASHBOARD_DIR / "feature_importance.csv")
    if not fi.empty:
        try:
            top = fi.sort_values("importance", ascending=False).head(
                20).sort_values("importance")
            fig2 = go.Figure(go.Bar(
                x=top["importance"], y=top["feature"], orientation="h",
                marker=dict(
                    color=top["importance"],
                    colorscale=[[0, "#002255"], [
                        0.5, "#0055aa"], [1, "#FFD700"]],
                    showscale=True,
                    colorbar=dict(
                        title=dict(text="Importance",
                                   font=dict(color="#8899aa")),
                        tickfont=dict(color="#8899aa"),
                    ),
                ),
                hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
            ))
            apply_plotly_layout(
                fig2, xaxis_title="Feature Importance", height=600)
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.warning(f"Feature importance chart could not be rendered: {e}")
    else:
        st.info("Run model training to generate `dashboard/feature_importance.csv`.")

    # ── Feature explanations ──
    section_title("📖", "FEATURE", "PLAIN-ENGLISH GUIDE")
    explanations = {
        "elo_diff": ("ELO Difference", "The gap in ELO ratings between home and away team. Higher = home team historically stronger. Strongest single predictor."),
        "ranking_diff": ("FIFA Ranking Difference", "Away rank minus home rank. Positive = home team is ranked higher. Reflects official FIFA assessment."),
        "form_diff": ("Form Score Difference", "Points from last 5 matches (W=3,D=1,L=0) — home minus away. Captures recent momentum."),
        "player_strength_diff": ("Player Strength Difference", "Squad overall rating difference from FIFA video game data. Proxy for squad quality."),
        "win_percentage_diff": ("Win % Difference", "All-time win percentage differential. Reflects historical dominance."),
        "goals_scored_diff": ("Goals Scored Difference", "Average goals per match differential. Reflects attacking effectiveness."),
        "neutral_match_flag": ("Neutral Venue Flag", "1 = neutral venue. Reduces home advantage boost, important for tournament matches."),
    }
    for feat, (name, desc) in explanations.items():
        if not fi.empty and "feature" in fi.columns:
            row_fi = fi[fi["feature"] == feat]
            imp = f" — Importance: **{row_fi['importance'].values[0]:.4f}**" if not row_fi.empty else ""
        else:
            imp = ""
        st.markdown(f"""<div class="insight-card">
            <b style="color:var(--white)">{name}</b>
            <span style="color:var(--gold);font-family:'JetBrains Mono';font-size:0.75rem;margin-left:0.5rem">{feat}</span>{imp}<br>
            <span style="color:var(--grey);font-size:0.85rem">{desc}</span>
        </div>""", unsafe_allow_html=True)

    # ── Eval report ──
    if EVAL_REPORT.exists():
        section_title("📁", "EVALUATION", "REPORT")
        with open(EVAL_REPORT) as f:
            report = json.load(f)
        with st.expander("View raw evaluation_report.json"):
            st.json(report)

# ─────────────────────────────────────────────
# PAGE 6 — ABOUT
# ─────────────────────────────────────────────


def page_about():
    section_title("📖", "ABOUT", "THE PROJECT")

    st.markdown(f"""
    <div class="hero-banner">
        <div class="hero-badge">DATA SCIENCE PORTFOLIO PROJECT</div>
        <h1 class="hero-title">FIFA WORLD CUP 2026<br>PREDICTION MODEL</h1>
        <p class="hero-subtitle">End-to-End ML Pipeline · PySpark ETL · XGBoost · Monte Carlo · Power BI · Streamlit</p>
        <div style="margin-top:1rem">
            <a href="https://github.com/Ayush264/FIFA-Prediction-Model" target="_blank"
               style="background:rgba(255,215,0,0.1);border:1px solid rgba(255,215,0,0.3);color:var(--gold);
                      padding:8px 20px;border-radius:8px;text-decoration:none;font-weight:600;font-size:0.9rem">
                🔗 GitHub: Ayush264/FIFA-Prediction-Model
            </a>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Overview ──
    st.markdown("""
    <div class="narrative-box">
        <div class="nar-title">PROJECT OVERVIEW</div>
        An end-to-end machine learning platform to predict the FIFA World Cup 2026 winner. 
        The pipeline ingests 51,440 historical international matches, engineers 9+ predictive features using PySpark, 
        trains and evaluates three ML models via MLflow, runs a 10,000-simulation Monte Carlo tournament bracket, 
        and visualizes results through this Streamlit Intelligence Center and a Power BI dashboard.
    </div>""", unsafe_allow_html=True)

    # ── Architecture ──
    section_title("🏗️", "SYSTEM", "ARCHITECTURE")
    st.markdown("""
    <div style="background:var(--glass);border:1px solid var(--border);border-radius:14px;padding:1.5rem;
                font-family:'JetBrains Mono';font-size:0.85rem;color:#88aabb;letter-spacing:0.5px">
    Raw Datasets (5 CSVs)
         ↓
    PySpark ETL (data_cleaning.py → feature_engineering.py → ranking_pipeline.py)
         ↓
    Feature Store (Apache Parquet — ELO, form, rankings, player strength, profiles)
         ↓
    ML Training (Logistic Regression | Random Forest | XGBoost) + MLflow Tracking
         ↓
    Monte Carlo Simulation (10,000-run full bracket — Group Stage → Final)
         ↓
    Power BI Dashboard + Streamlit Intelligence Center
    </div>""", unsafe_allow_html=True)

    # ── Tech stack ──
    section_title("⚙️", "TECHNOLOGY", "STACK")
    tech_groups = {
        "Data Engineering": ["PySpark", "Apache Parquet", "Pandas", "NumPy"],
        "Machine Learning": ["Scikit-Learn", "XGBoost", "MLflow", "Joblib"],
        "Simulation":       ["Monte Carlo (10K runs)", "Custom bracket logic", "Host advantage model"],
        "Visualization":    ["Streamlit", "Plotly", "Power BI", "Custom CSS"],
        "Infrastructure":   ["Python 3.11", "SQLite (MLflow)", "Git", "GitHub"],
    }
    for group, techs in tech_groups.items():
        st.markdown(
            f"<div style='color:var(--gold);font-size:0.8rem;font-weight:700;letter-spacing:1px;margin:1rem 0 0.5rem'>{group.upper()}</div>", unsafe_allow_html=True)
        badges = " ".join(
            [f'<span class="tech-badge">{t}</span>' for t in techs])
        st.markdown(badges, unsafe_allow_html=True)

    # ── Dataset stats ──
    section_title("📊", "DATASET", "STATISTICS")
    d1, d2, d3, d4, d5 = st.columns(5)
    with d1:
        metric_card("Total Matches", "51,440", "Since 1872", gold=True)
    with d2:
        metric_card("WC Matches", "836", "1930–2022")
    with d3:
        metric_card("FIFA Rankings", "6,675", "Historical rows")
    with d4:
        metric_card("Player Records", "~3,000", "Squad aggregates")
    with d5:
        metric_card("Country Mappings", "36", "Former names")

    # ── Pipeline steps ──
    section_title("🔧", "DATA PIPELINE", "WALKTHROUGH")
    steps = [
        ("1. Data Ingestion", "Five raw CSV datasets loaded via PySpark: international matches, FIFA rankings, World Cup results, player aggregates, and country name mappings."),
        ("2. Data Cleaning", "Standardize country names, remove duplicates, handle nulls, enforce types — producing clean Parquet files in `data/processed/`."),
        ("3. Feature Engineering", "Compute ELO ratings iteratively across 51K matches (K=60 WC, K=20 friendly), rolling form scores, win percentages, and goal averages per team."),
        ("4. Ranking Pipeline", "Merge all feature sources into a unified `team_profiles.csv` — a single source of truth for the model and dashboard."),
        ("5. Model Training", "Train LR, RF, and XGBoost with cross-validation. Log all experiments via MLflow (accuracy, F1, confusion matrix, feature importance). Best model saved as `models/best_model.pkl`."),
        ("6. Monte Carlo Simulation", "Simulate the complete WC bracket 10,000 times: group stage draws, point accumulation, Round of 32, QF, SF, Final. Host bonus (+4%) for USA/Canada/Mexico."),
        ("7. Visualization", "Export probabilities and rankings to CSV. Build Power BI and this Streamlit platform for interactive analytics."),
    ]
    for title, desc in steps:
        st.markdown(f"""<div class="insight-card">
            <b style="color:var(--white)">{title}</b><br>
            <span style="color:var(--grey);font-size:0.88rem">{desc}</span>
        </div>""", unsafe_allow_html=True)

    # ── Author ──
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;padding:2rem">
        <div style="font-family:'Bebas Neue';font-size:2rem;letter-spacing:3px;color:var(--white)">AYUSH</div>
        <div style="color:var(--grey);margin:0.3rem 0">CS Engineering · Chandigarh University</div>
        <a href="https://github.com/Ayush264" target="_blank"
           style="color:var(--gold);text-decoration:none;font-weight:600">
            github.com/Ayush264
        </a>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TEAM RANKINGS PAGE
# ─────────────────────────────────────────────


def page_team_rankings(tables, rankings):
    section_title("🏅", "TEAM", "RANKINGS")
    if rankings.empty:
        st.warning("No rankings data available.")
        return
    probs_df = read_csv(DASHBOARD_DIR / "championship_probabilities.csv")
    power_df = compute_power_index(rankings, probs_df)

    query = st.text_input(
        "🔍 Search team", placeholder="Type a team or confederation...")
    view = power_df.copy() if not power_df.empty else rankings.copy()
    if query:
        view = view[view.apply(lambda r: query.lower()
                               in str(r).lower(), axis=1)]

    # KPI row
    if not rankings.empty:
        m1, m2, m3 = st.columns(3)
        if "elo" in rankings.columns:
            top_e = rankings.sort_values("elo", ascending=False).iloc[0]
            with m1:
                metric_card(
                    "Top ELO", f"{flag(top_e['team'])} {top_e['team']}", f"{top_e['elo']:.0f}")
        if "fifa_rank" in rankings.columns:
            ranked = rankings[rankings["fifa_rank"].notna()
                              ].sort_values("fifa_rank")
            if not ranked.empty:
                top_r = ranked.iloc[0]
                with m2:
                    metric_card(
                        "Best FIFA Rank", f"{flag(top_r['team'])} {top_r['team']}", f"Rank #{int(top_r['fifa_rank'])}")
        if "form" in rankings.columns:
            top_f = rankings.sort_values("form", ascending=False).iloc[0]
            with m3:
                metric_card(
                    "Best Form", f"{flag(top_f['team'])} {top_f['team']}", f"{top_f['form']:.2f}", gold=True)

    # Table
    if not view.empty:
        show_cols = [c for c in ["team", "confederation", "elo", "fifa_rank", "form",
                                 "player_strength", "championship_probability", "power_index"] if c in view.columns]
        st.dataframe(view[show_cols].rename(columns={
            "team": "Team", "confederation": "Confederation", "elo": "ELO", "fifa_rank": "FIFA Rank",
            "form": "Form", "player_strength": "Squad Rating", "championship_probability": "Win %", "power_index": "Power Index"
        }), use_container_width=True, height=400)

    # Charts
    chart_cols = st.columns(3)
    with chart_cols[0]:
        if "elo" in rankings.columns:
            top = rankings.sort_values("elo", ascending=False).head(15)
            fig = go.Figure(go.Bar(x=top["team"], y=top["elo"],
                                   marker=dict(color=top["elo"], colorscale=[[0, "#003377"], [1, "#00A8FF"]], showscale=False)))
            apply_plotly_layout(fig, title_text="TOP TEAMS BY ELO")
            st.plotly_chart(fig, use_container_width=True)
    with chart_cols[1]:
        if "fifa_rank" in rankings.columns:
            top = rankings[rankings["fifa_rank"].notna()].sort_values(
                "fifa_rank").head(15)
            fig = go.Figure(go.Bar(x=top["team"], y=top["fifa_rank"],
                                   marker=dict(color=top["fifa_rank"], colorscale=[[0, "#FFD700"], [1, "#aa7700"]], showscale=False)))
            apply_plotly_layout(
                fig, title_text="BEST FIFA RANKINGS", yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
    with chart_cols[2]:
        if "form" in rankings.columns:
            top = rankings.sort_values("form", ascending=False).head(15)
            fig = go.Figure(go.Bar(x=top["team"], y=top["form"],
                                   marker=dict(color=top["form"], colorscale=[[0, "#004400"], [1, "#00E676"]], showscale=False)))
            apply_plotly_layout(fig, title_text="BEST RECENT FORM")
            st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────


NAV_OPTIONS = [
    "🏠  Executive Dashboard", "🤖  Match Predictor",
    "🌍  Team Intelligence", "🏆  WC Simulation",
    "🧠  Model Explainability", "🏅  Team Rankings", "📖  About",
]


def sidebar_nav():
    """Render reliable in-page navigation.

    Streamlit's native sidebar collapse/reopen button is controlled by frontend
    internals that changed across versions and browsers. For a public launch,
    the safest fix is to put navigation in the main app so it is always visible
    on desktop and mobile, even if the native sidebar is collapsed or hidden.
    """
    current = st.session_state.get("nav_page", NAV_OPTIONS[0])
    if current not in NAV_OPTIONS:
        current = NAV_OPTIONS[0]

    st.markdown("""
    <div class="global-nav-card">
        <div class="global-nav-eyebrow">⚽ FIFA WC 2026 Intelligence Platform</div>
        <div class="global-nav-copy">Choose a module below and dive into Analytics.</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.selectbox(
        "Choose dashboard page",
        NAV_OPTIONS,
        index=NAV_OPTIONS.index(current),
        key="nav_page",
        label_visibility="collapsed",
    )
    return page

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────


def main():
    # In-page navigation is intentionally used instead of Streamlit's native
    # sidebar toggle so the published app remains usable on desktop and mobile.
    page = sidebar_nav()

    tables = load_tables()
    rankings = normalize_rankings(tables)

    if "Executive" in page:
        page_executive(tables, rankings)
    elif "Match Predictor" in page:
        page_match_predictor(tables, rankings)
    elif "Team Intelligence" in page:
        page_team_intelligence(tables, rankings)
    elif "WC Simulation" in page:
        page_simulation(tables, rankings)
    elif "Model" in page:
        page_model(tables, rankings)
    elif "Rankings" in page:
        page_team_rankings(tables, rankings)
    elif "About" in page:
        page_about()


if __name__ == "__main__":
    main()
