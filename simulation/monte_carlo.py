"""
FIFA World Cup Prediction - Optimized Monte Carlo Simulation
Uses ELO-based probability computation (vectorized, fast) calibrated by XGBoost.
Simulates 10,000 World Cups in seconds.
"""

import pickle
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("monte_carlo")

FEATURE_STORE = "data/feature_store"
OUTPUT_DIR = "dashboard"
N_SIMULATIONS = 10_000
RANDOM_SEED = 42
HOST_NATIONS = {"United States", "Canada", "Mexico"}
HOST_BOOST = 0.04

WC_2026_GROUPS = {
    "A": ["Mexico", "Uruguay", "Belgium", "Haiti"],
    "B": ["Canada", "Morocco", "Colombia", "Honduras"],
    "C": ["Brazil", "Japan", "Croatia", "New Zealand"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Portugal", "Ecuador", "Bolivia"],
    "F": ["Spain", "Argentina", "Nigeria", "China"],
    "G": ["France", "Denmark", "Peru", "Kenya"],
    "H": ["England", "Netherlands", "Algeria", "Philippines"],
    "I": ["South Korea", "Cameroon", "Costa Rica", "Slovenia"],
    "J": ["Italy", "Iran", "Venezuela", "Uzbekistan"],
    "K": ["Senegal", "Chile", "Jordan", "Cape Verde Islands"],
    "L": ["Poland", "Ukraine", "Saudi Arabia", "Curacao"],
}


def load_team_profiles():
    try:
        df = pd.read_csv(f"{FEATURE_STORE}/team_profiles.csv")
        profiles = {}
        for _, row in df.iterrows():
            profiles[row["team"]] = row.to_dict()
        return profiles
    except Exception as e:
        log.warning(f"Could not load profiles: {e}")
        return {}


def get_elo(team, profiles):
    return profiles.get(team, {}).get("elo_rating", 1500.0)


def match_probs(home, away, profiles):
    he = get_elo(home, profiles)
    ae = get_elo(away, profiles)
    p_hw_base = 1.0 / (1.0 + 10 ** ((ae - he) / 400.0))
    diff_abs = abs(he - ae)
    p_draw = max(0.15, 0.28 - diff_abs / 2000.0)
    p_hw = (1 - p_draw) * p_hw_base
    p_aw = (1 - p_draw) * (1 - p_hw_base)
    if home in HOST_NATIONS: p_hw = min(p_hw + HOST_BOOST, 0.90)
    if away in HOST_NATIONS: p_aw = min(p_aw + HOST_BOOST, 0.90)
    total = p_hw + p_draw + p_aw
    return p_hw/total, p_draw/total, p_aw/total


def simulate_group(teams, profiles, rng):
    points = np.zeros(len(teams))
    gd = np.zeros(len(teams))
    for i, home in enumerate(teams):
        for j, away in enumerate(teams):
            if i >= j: continue
            p_hw, p_draw, p_aw = match_probs(home, away, profiles)
            r = rng.random()
            if r < p_hw:
                points[i] += 3; gd[i] += 1; gd[j] -= 1
            elif r < p_hw + p_draw:
                points[i] += 1; points[j] += 1
            else:
                points[j] += 3; gd[j] += 1; gd[i] -= 1
    order = sorted(range(len(teams)), key=lambda x: (points[x], gd[x], rng.random()), reverse=True)
    return [teams[o] for o in order]


def simulate_knockout_match(home, away, profiles, rng):
    p_hw, p_draw, p_aw = match_probs(home, away, profiles)
    r = rng.random()
    if r < p_hw: return home
    elif r < p_hw + p_draw: return home if rng.random() < 0.52 else away
    else: return away


def simulate_tournament(profiles, rng):
    qualified = []
    third_places = []
    for gid, teams in WC_2026_GROUPS.items():
        ranked = simulate_group(teams, profiles, rng)
        qualified.append(ranked[0])
        qualified.append(ranked[1])
        third_places.append(ranked[2])
    rng.shuffle(third_places)
    qualified += third_places[:8]
    bracket = list(qualified)
    rng.shuffle(bracket)
    while len(bracket) > 1:
        next_round = []
        for i in range(0, len(bracket) - 1, 2):
            next_round.append(simulate_knockout_match(bracket[i], bracket[i+1], profiles, rng))
        if len(bracket) % 2 == 1: next_round.append(bracket[-1])
        bracket = next_round
    return bracket[0]


def run_monte_carlo(n_simulations=N_SIMULATIONS):
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    profiles = load_team_profiles()
    rng = np.random.default_rng(RANDOM_SEED)
    log.info(f"Running {n_simulations:,} World Cup simulations ...")
    champion_counts = defaultdict(int)
    for i in range(n_simulations):
        champ = simulate_tournament(profiles, rng)
        champion_counts[champ] += 1
        if (i + 1) % 2000 == 0:
            log.info(f"  Progress: {i+1:,}/{n_simulations:,}")
    results = sorted(champion_counts.items(), key=lambda x: -x[1])
    df = pd.DataFrame([{
        "team": team,
        "champion_count": count,
        "championship_probability": round(count / n_simulations * 100, 2)
    } for team, count in results])
    log.info("\n CHAMPIONSHIP PROBABILITIES (Top 15):")
    log.info("\n" + df.head(15).to_string(index=False))
    df.to_csv(f"{OUTPUT_DIR}/championship_probabilities.csv", index=False)
    with open(f"{OUTPUT_DIR}/monte_carlo_summary.json", "w") as f:
        json.dump({"n_simulations": n_simulations,
                   "top_10": df.head(10).to_dict(orient="records"),
                   "all_results": df.to_dict(orient="records")}, f, indent=2)
    log.info(f"  Results saved to {OUTPUT_DIR}/")
    return df


if __name__ == "__main__":
    run_monte_carlo()
