# Migration Notes: All-Matches Training Dataset

## Summary

The training dataset now comes from `data/raw/all_matches.csv` through the cleaned
`data/processed/all_matches.csv` file instead of `WorldCupMatches.csv`.

This increases `data/feature_store/match_features.csv` from 836 World Cup rows to
approximately 51,440 historical international matches.

## Feature Engineering Changes

- `pyspark_jobs/feature_engineering.py` now builds one row per historical match.
- Match features are computed before updating any rolling state for that match.
- Rolling ELO starts at 1500 for every team.
- ELO K-factors:
  - FIFA World Cup: 60
  - continental tournaments: 40
  - qualifiers: 30
  - friendlies: 20
- Team form uses only the previous 5 matches.
- Average goals and win percentage use only prior matches.
- FIFA ranking uses the latest available ranking before the match date.
- Player strength is a static proxy from `player_aggregates.csv` because that file
  has no match-date history.

## Target Encoding

- `0` = Away Win
- `1` = Draw
- `2` = Home Win

## Outputs

- `data/feature_store/match_features.csv`
- `data/feature_store/elo_ratings.csv`
- `data/feature_store/team_form.csv`
- `data/feature_store/fifa_rankings_latest.csv`
- `dashboard/feature_importance.csv`

## Model Training Changes

- `models/train_model.py` now uses the new feature column names.
- Train/test split is chronological when `date` is available.
- Logistic Regression, Random Forest, and XGBoost are trained.
- Metrics saved:
  - Accuracy
  - F1 macro
  - F1 weighted
  - Confusion matrix
  - Classification report
- Random Forest and XGBoost importances are normalized and averaged into
  `dashboard/feature_importance.csv`.
