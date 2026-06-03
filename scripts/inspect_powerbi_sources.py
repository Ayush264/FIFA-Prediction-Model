import pandas as pd
from pathlib import Path
paths = [
    Path('dashboard/team_rankings.csv'),
    Path('dashboard/championship_probabilities.csv'),
    Path('dashboard/feature_importance.csv'),
    Path('data/processed/world_cup_matches.csv'),
    Path('data/processed/all_matches.csv'),
]
for path in paths:
    print('---', path)
    if path.exists():
        df = pd.read_csv(path)
        print('shape', df.shape)
        print('cols', list(df.columns))
        print(df.head(2).to_string(index=False))
    else:
        print('missing', path)
