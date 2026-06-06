"""
FIFA World Cup Prediction - Model Training

Trains Logistic Regression, Random Forest, and XGBoost on leakage-aware
features generated from all historical international matches.
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import json
import logging
import pickle
import sys
from pathlib import Path

# ── third-party ───────────────────────────────────────────────────────────────
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── local ─────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

try:
    import mlflow_tracking
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ── config ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_model")

FEATURE_STORE  = "data/feature_store"
MODEL_DIR      = "models"
DASHBOARD_DIR  = "dashboard"
TARGET_COL     = "label"

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


# ── data loading ──────────────────────────────────────────────────────────────

def load_features() -> pd.DataFrame:
    path = Path(FEATURE_STORE) / "match_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Could not find feature file: {path}")
    df = pd.read_csv(path)
    log.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns from {path}")
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

    log.info(f"Class distribution:\n{y.value_counts().sort_index().to_string()}")
    return X, y, FEATURE_COLS


def chronological_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    split_idx = int(len(X) * (1.0 - test_size))
    return (
        X.iloc[:split_idx],
        X.iloc[split_idx:],
        y.iloc[:split_idx],
        y.iloc[split_idx:],
    )


# ── models ────────────────────────────────────────────────────────────────────

def get_models() -> dict:
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


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(
    name: str, model, X_train, X_test, y_train, y_test
) -> dict:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # FIX 1 & 2 & 3 ─ sklearn metrics return numpy scalars; cast to float
    # before calling round() so Python's builtin is happy (Pylance lines 162-163)
    acc         = float(accuracy_score(y_test, y_pred))
    f1_macro    = float(f1_score(y_test, y_pred, average="macro",    zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    cm     = confusion_matrix(y_test, y_pred, labels=[0, 1, 2]).tolist()
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
        "model_name":            name,
        "accuracy":              round(acc, 4),
        "f1_macro":              round(f1_macro, 4),
        "f1_weighted":           round(f1_weighted, 4),
        "confusion_matrix":      cm,
        "classification_report": report,
    }


# ── feature importance ────────────────────────────────────────────────────────

def _get_importances(model) -> "pd.Series | None":
    """
    FIX 4 ─ the original code called .feature_importances_ on the return value
    of model_importance(), which could be None (Pylance line 175).
    This helper returns a Series or None explicitly, keeping the None-check here.
    """
    if model is None:
        return None

    # Direct attribute (RandomForest, XGBoost)
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_          # type: ignore[return-value]

    # Sklearn Pipeline — look inside the final estimator
    if isinstance(model, Pipeline):
        clf = model.named_steps.get("clf")
        if clf is not None and hasattr(clf, "feature_importances_"):
            return clf.feature_importances_        # type: ignore[return-value]

    return None


def write_feature_importance(trained_models: dict, feature_cols: list) -> None:
    importances = []

    for name in ("random_forest", "xgboost"):
        raw = _get_importances(trained_models.get(name))
        if raw is None:
            continue

        series = pd.Series(raw, index=feature_cols, dtype=float)
        total  = series.sum()
        if total > 0:
            series = series / total
        importances.append(series)

    if not importances:
        log.warning("No feature importances available — skipping export")
        return

    importance = pd.concat(importances, axis=1).mean(axis=1)
    df_imp = (
        importance
        .reset_index()
        .rename(columns={"index": "feature", 0: "importance"})
        .sort_values("importance", ascending=False)
    )

    Path(DASHBOARD_DIR).mkdir(exist_ok=True)
    out_path = Path(DASHBOARD_DIR) / "feature_importance.csv"
    df_imp.to_csv(out_path, index=False)
    log.info(f"Feature importance written to {out_path}")


# ── main training loop ────────────────────────────────────────────────────────

def train_all() -> list:
    Path(MODEL_DIR).mkdir(exist_ok=True)

    df = load_features()
    X, y, feature_cols = prepare_data(df)
    X_train, X_test, y_train, y_test = chronological_split(X, y, test_size=0.2)
    log.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    models         = get_models()
    results        = []
    trained_models = {}
    best_model     = None
    best_acc       = -1.0

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
            best_acc   = metrics["accuracy"]
            best_model = (name, model)

    best_name      = None
    best_model_path: Path | None = None

    if best_model:
        best_name, best_obj = best_model
        best_model_path = Path(MODEL_DIR) / "best_model.pkl"
        with open(best_model_path, "wb") as f:
            pickle.dump(
                {"model": best_obj, "features": feature_cols, "name": best_name},
                f,
            )
        log.info(f"Best model: {best_name} (accuracy: {best_acc:.4f})")

    write_feature_importance(trained_models, feature_cols)
    feature_importance_path = Path(DASHBOARD_DIR) / "feature_importance.csv"

    # ── MLflow logging (FIX 5 — guarded by availability check) ───────────────
    # FIX 5: the original code did `import mlflow_tracking` at the top of the
    # file unconditionally, meaning ANY import error in mlflow_tracking.py
    # (missing mlflow package, wrong path, etc.) would crash train_model.py
    # before training even starts. We now import it lazily at the top with a
    # try/except and only call it when it actually loaded.
    if MLFLOW_AVAILABLE:
        mlflow_tracking.setup_experiment()

        for name, model in trained_models.items():
            metrics = next(
                (item for item in results if item["model_name"] == name), None
            )
            if metrics is None:
                continue

            params                 = mlflow_tracking.get_model_params(model)
            model_path             = Path(MODEL_DIR) / f"{name}.pkl"
            confusion_matrix_path  = Path(MODEL_DIR) / f"{name}_confusion_matrix.png"

            mlflow_tracking.save_confusion_matrix(
                metrics["confusion_matrix"], CLASS_NAMES, confusion_matrix_path
            )
            mlflow_tracking.log_model_run(
                model_name             = name,
                model                  = model,
                params                 = params,
                metrics                = metrics,
                model_path             = model_path,
                confusion_matrix_path  = confusion_matrix_path,
                feature_importance_path= feature_importance_path,
                best                   = (name == best_name),
                best_model_path        = best_model_path if name == best_name else None,
            )
    else:
        log.warning(
            "mlflow_tracking module not found — skipping MLflow logging. "
            "Make sure mlflow_tracking.py is in your project root and "
            "`pip install mlflow` has been run."
        )

    # ── save evaluation report ─────────────────────────────────────────────────
    report_path = Path(MODEL_DIR) / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log.info(f"Evaluation report saved to {report_path}")

    return results


if __name__ == "__main__":
    train_all()