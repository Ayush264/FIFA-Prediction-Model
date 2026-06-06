"""
FIFA World Cup Prediction - Model Training

Trains Logistic Regression, Random Forest, and XGBoost on leakage-aware
features generated from all historical international matches.
"""
import mlflow_tracking
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle
import logging
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_model")

FEATURE_STORE = "data/feature_store"
MODEL_DIR = "models"
DASHBOARD_DIR = "dashboard"
TARGET_COL = "label"

FEATURE_COLS = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_form",
    "away_form",
    "form_diff",
    "home_avg_goals_scored",
    "away_avg_goals_scored",
    "goals_scored_diff",
    "home_avg_goals_conceded",
    "away_avg_goals_conceded",
    "goals_conceded_diff",
    "home_win_percentage",
    "away_win_percentage",
    "win_percentage_diff",
    "home_fifa_rank",
    "away_fifa_rank",
    "ranking_diff",
    "home_player_strength",
    "away_player_strength",
    "player_strength_diff",
    "neutral_match_flag",
]

CLASS_NAMES = ["Away Win", "Draw", "Home Win"]


def load_features() -> pd.DataFrame:
    path = Path(FEATURE_STORE) / "match_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Could not find feature file: {path}")

    df = pd.read_csv(path)
    log.info(
        f"Loaded {len(df):,} rows and {len(df.columns)} columns from {path}")
    return df


def prepare_data(df: pd.DataFrame):
    missing = [col for col in FEATURE_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    df = df.dropna(subset=[TARGET_COL]).copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)

    X = df[FEATURE_COLS].fillna(0)
    y = df[TARGET_COL].astype(int)

    log.info(
        f"Class distribution:\n{y.value_counts().sort_index().to_string()}")
    return X, y, FEATURE_COLS


def chronological_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    split_idx = int(len(X) * (1.0 - test_size))
    return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]


def get_models():
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                C=0.5,
                solver="lbfgs",
                random_state=42,
            )),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        ),
    }


def evaluate_model(name: str, model, X_train, X_test, y_train, y_test) -> dict:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2]).tolist()
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    log.info(f"\n{'=' * 50}")
    log.info(f"Model: {name.upper()}")
    log.info(f"  Test Accuracy : {acc:.4f}")
    log.info(f"  F1 Macro      : {f1_macro:.4f}")
    log.info(f"  F1 Weighted   : {f1_weighted:.4f}")
    log.info(f"\n{report}")

    return {
        "model_name": name,
        "accuracy": round(float(acc), 4),
        "f1_macro": round(float(f1_macro), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "confusion_matrix": cm,
        "classification_report": report,
    }


def model_importance(model) -> np.ndarray | None:
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    if isinstance(model, Pipeline):
        clf = model.named_steps.get("clf")
        if clf is not None and hasattr(clf, "feature_importances_"):
            return clf.feature_importances_
    return None


def write_feature_importance(trained_models: dict, feature_cols):
    importances = []
    for name in ("random_forest", "xgboost"):
        values = model_importance(trained_models.get(name))
        if values is None:
            continue
        series = pd.Series(values, index=feature_cols, dtype=float)
        total = series.sum()
        if total > 0:
            series = series / total
        importances.append(series)

    if not importances:
        log.warning("No feature importances available")
        return

    importance = pd.concat(importances, axis=1).mean(axis=1)
    df = (
        importance
        .reset_index()
        .rename(columns={"index": "feature", 0: "importance"})
        .sort_values("importance", ascending=False)
    )

    Path(DASHBOARD_DIR).mkdir(exist_ok=True)
    df.to_csv(Path(DASHBOARD_DIR) / "feature_importance.csv", index=False)
    log.info("Feature importance written to dashboard/feature_importance.csv")


def train_all():
    Path(MODEL_DIR).mkdir(exist_ok=True)

    df = load_features()
    X, y, feature_cols = prepare_data(df)
    X_train, X_test, y_train, y_test = chronological_split(X, y, test_size=0.2)
    log.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    models = get_models()
    results = []
    trained_models = {}
    best_model = None
    best_acc = -1.0

    for name, model in models.items():
        log.info(f"\nTraining {name} ...")
        metrics = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        results.append(metrics)
        trained_models[name] = model

        model_path = Path(MODEL_DIR) / f"{name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        log.info(f"Saved model to {model_path}")

        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            best_model = (name, model)

    if best_model:
        best_name, best_obj = best_model
        best_model_path = Path(MODEL_DIR) / "best_model.pkl"
        with open(best_model_path, "wb") as f:
            pickle.dump(
                {"model": best_obj, "features": feature_cols, "name": best_name}, f)
        log.info(f"Best model: {best_name} (accuracy: {best_acc:.4f})")
    else:
        best_name = None
        best_model_path = None

    write_feature_importance(trained_models, feature_cols)
    feature_importance_path = Path(DASHBOARD_DIR) / "feature_importance.csv"

    mlflow_tracking.setup_experiment()

    for name, model in trained_models.items():
        metrics = next(
            (item for item in results if item["model_name"] == name), None)
        if metrics is None:
            continue

        params = mlflow_tracking.get_model_params(model)
        model_path = Path(MODEL_DIR) / f"{name}.pkl"
        confusion_matrix_path = Path(
            MODEL_DIR) / f"{name}_confusion_matrix.png"
        mlflow_tracking.save_confusion_matrix(
            metrics["confusion_matrix"], CLASS_NAMES, confusion_matrix_path)

        mlflow_tracking.log_model_run(
            model_name=name,
            model=model,
            params=params,
            metrics=metrics,
            model_path=model_path,
            confusion_matrix_path=confusion_matrix_path,
            feature_importance_path=feature_importance_path,
            best=(name == best_name),
            best_model_path=best_model_path if name == best_name else None,
        )

    with open(Path(MODEL_DIR) / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log.info("Evaluation report saved to models/evaluation_report.json")

    return results


if __name__ == "__main__":
    train_all()
