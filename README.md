# ⚽ FIFA World Cup 2026 Prediction Platform

An end-to-end machine learning platform to predict the FIFA World Cup 2026 winner using PySpark ETL pipelines, XGBoost, and Monte Carlo simulation.

---

## 🏗️ Architecture

```
Raw Datasets → PySpark ETL → Feature Engineering → ML Models → Monte Carlo → Power BI
```

| Layer                     | Tech                                        |
| ------------------------- | ------------------------------------------- |
| Data Ingestion & Cleaning | PySpark                                     |
| Feature Store             | Apache Parquet                              |
| ML Models                 | Logistic Regression, Random Forest, XGBoost |
| Tournament Simulation     | Monte Carlo (10,000 runs)                   |
| BI Dashboard              | Power BI                                    |

---

## 📁 Project Structure

```
FIFA-WorldCup-Prediction/
├── data/
│   ├── raw/                    # Source CSVs (5 datasets)
│   ├── processed/              # Cleaned Parquet files
│   └── feature_store/          # Feature-engineered Parquet + CSV exports
├── pyspark_jobs/
│   ├── data_cleaning.py        # ETL: cleaning, standardization, dedup
│   ├── feature_engineering.py  # ELO, form, goals, player stats
│   └── ranking_pipeline.py     # Unified team strength profiles
├── models/
│   ├── train_model.py          # LR + RF + XGBoost training + evaluation
│   ├── best_model.pkl          # Best trained model (XGBoost)
│   └── evaluation_report.json  # Accuracy, F1, confusion matrix
├── simulation/
│   └── monte_carlo.py          # 10,000-run WC tournament simulator
├── dashboard/
│   ├── championship_probabilities.csv  # Monte Carlo output (→ Power BI)
│   └── monte_carlo_summary.json
├── sql/
│   └── business_queries.sql    # 10 BI queries for Power BI
└── run_pipeline.py             # Master runner (all steps)
```

---

## 📊 Datasets

| File                    | Description                         | Rows   |
| ----------------------- | ----------------------------------- | ------ |
| `all_matches.csv`       | International matches since 1872    | 51,440 |
| `fifa_mens_rank.csv`    | FIFA ranking history                | 6,675  |
| `WorldCupMatches.csv`   | WC match results 1930–2022          | 836    |
| `player_aggregates.csv` | FIFA game squad strength by country | ~3,000 |
| `former_names.csv`      | Country name mappings               | 36     |

---

## 🔧 Features Engineered

| Feature               | Description                                          |
| --------------------- | ---------------------------------------------------- |
| `elo_diff`            | ELO rating difference (home − away)                  |
| `rank_diff`           | FIFA rank difference (positive = home ranked higher) |
| `form_score`          | Points from last 5 matches (W=3, D=1, L=0)           |
| `goals_scored_avg`    | Historical average goals scored per match            |
| `goals_conceded_avg`  | Historical average goals conceded per match          |
| `win_pct`             | All-time win percentage                              |
| `player_overall_diff` | FIFA game squad rating difference                    |
| `home_is_host`        | Host nation flag (USA / Canada / Mexico)             |
| `elo_rating`          | Final ELO rating after all historical matches        |

---

## 🤖 Model Results

| Model               | Accuracy  | F1 (macro) | CV Accuracy |
| ------------------- | --------- | ---------- | ----------- |
| XGBoost             | **56.6%** | 0.462      | 0.530       |
| Random Forest       | 54.8%     | 0.488      | 0.500       |
| Logistic Regression | 43.5%     | 0.404      | 0.455       |

> ⚠️ Note: Football outcome prediction is inherently noisy. 56% accuracy on 3-class (win/draw/loss) is above the ~43% random baseline and comparable to published football ML research.

---

## 🏆 2026 World Cup Predictions (10,000 Simulations)

| Rank | Team          | Win Probability |
| ---- | ------------- | --------------- |
| 🥇 1 | **Spain**     | 13.86%          |
| 🥈 2 | **Argentina** | 13.20%          |
| 🥉 3 | **France**    | 9.61%           |
| 4    | England       | 5.39%           |
| 5    | Brazil        | 4.77%           |
| 6    | Portugal      | 4.50%           |
| 7    | Colombia      | 4.37%           |
| 8    | Netherlands   | 4.10%           |
| 9    | Germany       | 3.99%           |
| 10   | Japan         | 3.53%           |

---

## 🚀 How to Run

### Prerequisites

```bash
pip install pyspark xgboost scikit-learn pandas numpy pyarrow
```

### Run Full Pipeline

```bash
cd FIFA-WorldCup-Prediction
python run_pipeline.py
```

### Run MLflow UI

```bash
cd FIFA-WorldCup-Prediction
pip install mlflow
# Recommended (Windows): use the helper script which ensures the sqlite backend
# and artifact root are used consistently with training.
# PowerShell
./run_mlflow.ps1

# Or (legacy):
# mlflow ui
```

Then open:

http://localhost:5000

If the UI shows a blank page or refuses to connect, try:

1. Run the helper script and watch the terminal for startup logs:

```powershell
.\run_mlflow.ps1
```

2. If the server starts but the browser refuses to connect, try binding to all interfaces and a different port:

```powershell
.\venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5001
```

3. Verify Windows is listening on the port and the MLflow process is running:

```powershell
netstat -ano | findstr "5000"
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'mlflow' -or $_.CommandLine -match 'venv\\Scripts\\python' } | Select-Object ProcessId,CommandLine
```

4. If the port shows a process but the browser still can't connect, temporarily allow the port in Windows Firewall or test with `curl` from the same machine.

### Run Individual Steps

```bash
# Step 1: Data Cleaning
python pyspark_jobs/data_cleaning.py

# Step 2: Feature Engineering
python pyspark_jobs/feature_engineering.py

# Step 3: Ranking Pipeline
python pyspark_jobs/ranking_pipeline.py

# Step 4: Model Training
python models/train_model.py

# Step 5: Monte Carlo Simulation
python simulation/monte_carlo.py
```

---

## 📈 Power BI Integration

Connect Power BI to:

- `dashboard/championship_probabilities.csv` — Win probabilities (bar/donut chart)
- `data/feature_store/team_profiles.csv` — Team strength matrix (scatter/table)
- `models/evaluation_report.json` — Model comparison table

Use queries from `sql/business_queries.sql` via Power BI's SQL connector or after loading CSVs.

**Recommended Visuals:**

1. **Donut chart** — Top 10 championship probabilities
2. **Scatter plot** — ELO vs. Win probability (coloured by confederation)
3. **Bar chart** — Player rating vs. championship odds
4. **Table** — Full 48-team probability ranking
5. **Line chart** — FIFA ranking trends for top 8 teams
6. **KPI cards** — Top favourite, dark horse, defending champion odds

---

## 🧠 Methodology Notes

- **ELO Ratings** computed iteratively across 51,440 historical matches (K=60 for WC, K=20 for friendlies)
- **Monte Carlo** simulates complete group stage + Round of 32 + knockout bracket
- **Draw resolution** in knockout rounds uses 52/48 home-edge penalty shootout
- **Host advantage** adds +4% win probability boost for USA/Canada/Mexico
- **Feature store** is Parquet-native for fast I/O and Power BI compatibility

---

## 👤 Author

Ayush | CS Engineering | Chandigarh University  
GitHub: [github.com/Ayush264](https://github.com/Ayush264)
