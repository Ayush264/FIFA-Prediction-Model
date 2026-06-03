-- =============================================================
-- FIFA World Cup 2026 Prediction Platform
-- Business Intelligence SQL Queries
-- Target: Power BI via parquet/CSV exports
-- =============================================================


-- ─────────────────────────────────────────────────────────────
-- 1. Championship Win Probability (Monte Carlo Output)
-- =============================================================
SELECT
    team,
    championship_probability        AS win_prob_pct,
    champion_count,
    RANK() OVER (ORDER BY championship_probability DESC) AS rank_in_tournament
FROM championship_probabilities
ORDER BY championship_probability DESC;


-- ─────────────────────────────────────────────────────────────
-- 2. Team Strength Dashboard
--    Composite view of ELO, FIFA rank, form, player rating
-- =============================================================
SELECT
    team,
    ROUND(elo_rating, 1)            AS elo_rating,
    fifa_rank,
    ROUND(fifa_points, 1)           AS fifa_points,
    ROUND(form_score, 2)            AS form_score,
    ROUND(win_pct * 100, 1)         AS win_pct_pct,
    ROUND(goals_scored_avg, 2)      AS avg_goals_scored,
    ROUND(goals_conceded_avg, 2)    AS avg_goals_conceded,
    ROUND(avg_overall, 1)           AS player_overall_rating,
    ROUND(avg_attack_overall, 1)    AS player_attack_rating,
    ROUND(avg_defense_overall, 1)   AS player_defense_rating,
    ROUND(team_strength_score, 2)   AS composite_strength_score
FROM team_profiles
ORDER BY team_strength_score DESC;


-- ─────────────────────────────────────────────────────────────
-- 3. Historical World Cup Performance by Nation
-- =============================================================
SELECT
    home_team                       AS team,
    COUNT(*)                        AS total_wc_matches,
    SUM(CASE WHEN result = 'home_win' THEN 1 ELSE 0 END) AS wins,
    SUM(CASE WHEN result = 'draw' THEN 1 ELSE 0 END)     AS draws,
    SUM(CASE WHEN result = 'away_win' THEN 1 ELSE 0 END) AS losses,
    ROUND(
        SUM(CASE WHEN result = 'home_win' THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100,
    1)                               AS home_win_pct,
    SUM(home_goals)                 AS total_goals_scored,
    SUM(away_goals)                 AS total_goals_conceded,
    SUM(home_goals) - SUM(away_goals) AS goal_difference
FROM world_cup_matches
GROUP BY home_team

UNION ALL

SELECT
    away_team                       AS team,
    COUNT(*)                        AS total_wc_matches,
    SUM(CASE WHEN result = 'away_win' THEN 1 ELSE 0 END) AS wins,
    SUM(CASE WHEN result = 'draw' THEN 1 ELSE 0 END)     AS draws,
    SUM(CASE WHEN result = 'home_win' THEN 1 ELSE 0 END) AS losses,
    ROUND(
        SUM(CASE WHEN result = 'away_win' THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100,
    1)                               AS away_win_pct,
    SUM(away_goals)                 AS total_goals_scored,
    SUM(home_goals)                 AS total_goals_conceded,
    SUM(away_goals) - SUM(home_goals) AS goal_difference
FROM world_cup_matches
GROUP BY away_team;


-- ─────────────────────────────────────────────────────────────
-- 4. ELO Rating Trends Over Time
--    Useful for a line chart per team
-- =============================================================
SELECT
    team,
    date,
    rank                            AS fifa_rank,
    total_points                    AS fifa_points,
    diff_points                     AS points_change
FROM fifa_rankings
WHERE team IN (
    'Spain', 'France', 'Brazil', 'Argentina',
    'England', 'Germany', 'Portugal', 'Netherlands'
)
ORDER BY team, date;


-- ─────────────────────────────────────────────────────────────
-- 5. Goals Scored Distribution by Tournament Type
-- =============================================================
SELECT
    tournament,
    COUNT(*)                        AS match_count,
    ROUND(AVG(home_score + away_score), 2) AS avg_goals_per_match,
    MAX(home_score + away_score)    AS highest_scoring_match,
    SUM(home_score + away_score)    AS total_goals
FROM all_matches
WHERE tournament IN (
    'FIFA World Cup', 'UEFA Euro', 'Copa América',
    'Africa Cup of Nations', 'AFC Asian Cup'
)
GROUP BY tournament
ORDER BY avg_goals_per_match DESC;


-- ─────────────────────────────────────────────────────────────
-- 6. Head-to-Head Record Between Top Teams
-- =============================================================
SELECT
    home_team,
    away_team,
    COUNT(*)                        AS total_matches,
    SUM(CASE WHEN result = 'home_win' THEN 1 ELSE 0 END) AS home_wins,
    SUM(CASE WHEN result = 'draw'     THEN 1 ELSE 0 END) AS draws,
    SUM(CASE WHEN result = 'away_win' THEN 1 ELSE 0 END) AS away_wins,
    ROUND(AVG(home_score), 2)       AS avg_home_goals,
    ROUND(AVG(away_score), 2)       AS avg_away_goals
FROM all_matches
WHERE
    home_team IN ('Spain', 'France', 'Brazil', 'Argentina', 'England', 'Germany')
    AND away_team IN ('Spain', 'France', 'Brazil', 'Argentina', 'England', 'Germany')
GROUP BY home_team, away_team
ORDER BY total_matches DESC;


-- ─────────────────────────────────────────────────────────────
-- 7. Model Evaluation Summary
-- =============================================================
SELECT
    model_name,
    accuracy,
    f1_macro,
    f1_weighted,
    cv_accuracy_mean,
    cv_accuracy_std,
    RANK() OVER (ORDER BY accuracy DESC) AS performance_rank
FROM model_evaluation_results
ORDER BY accuracy DESC;


-- ─────────────────────────────────────────────────────────────
-- 8. World Cup Winners – Historical Roll of Honour
-- =============================================================
SELECT
    home_team                       AS winner,
    Year                            AS wc_year,
    Stage,
    home_goals                      AS home_goals_final,
    away_goals                      AS away_goals_final,
    away_team                       AS runner_up
FROM world_cup_matches
WHERE Stage = 'Final'
ORDER BY Year DESC;


-- ─────────────────────────────────────────────────────────────
-- 9. Group Stage Simulation Result Summary (KPI Card)
-- =============================================================
SELECT
    team,
    championship_probability        AS win_prob_pct,
    CASE
        WHEN championship_probability >= 15  THEN 'Top Favourite'
        WHEN championship_probability >= 8   THEN 'Contender'
        WHEN championship_probability >= 3   THEN 'Dark Horse'
        ELSE 'Underdog'
    END                             AS tier
FROM championship_probabilities
ORDER BY championship_probability DESC;


-- ─────────────────────────────────────────────────────────────
-- 10. Player Rating vs Championship Probability (Scatter)
-- =============================================================
SELECT
    cp.team,
    cp.championship_probability,
    tp.avg_overall                  AS player_overall,
    tp.avg_attack_overall           AS player_attack,
    tp.avg_defense_overall          AS player_defense,
    tp.elo_rating,
    tp.fifa_rank,
    tp.composite_strength_score
FROM championship_probabilities cp
JOIN team_profiles tp ON cp.team = tp.team
ORDER BY cp.championship_probability DESC;
