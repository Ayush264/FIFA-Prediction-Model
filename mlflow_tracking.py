import logging
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import os

EXPERIMENT_NAME = "FIFA_World_Cup_Prediction"
REGISTERED_MODEL_NAME = "FIFA_World_Cup_Prediction"

log = logging.getLogger("mlflow_tracking")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")


def setup_experiment() -> None:
    # Ensure the MLflow client/CLI use the project's sqlite DB when no tracking
    # URI is configured in the environment. This makes the UI and training use
    # the same backend by default.
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        log.info("MLflow tracking URI set to sqlite:///mlflow.db")

    mlflow.set_experiment(EXPERIMENT_NAME)
    log.info("MLflow experiment set to '%s'", EXPERIMENT_NAME)


def save_confusion_matrix(cm: Any, labels: list[str], output_path: Path) -> None:
    matrix = np.asarray(cm, dtype=int)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = plt.cm.Blues
    im = ax.imshow(matrix, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(matrix.shape[1]),
        yticks=np.arange(matrix.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="Actual",
        xlabel="Predicted",
        title="Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=45,
             ha="right", rotation_mode="anchor")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, matrix[i, j], ha="center",
                    va="center", color="black")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info("Saved confusion matrix to %s", output_path)


def get_model_params(model: Any) -> Dict[str, Any]:
    if hasattr(model, "get_params"):
        return model.get_params()
    return {}


def log_model_run(
    model_name: str,
    model: Any,
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    model_path: Path,
    confusion_matrix_path: Path,
    feature_importance_path: Optional[Path] = None,
    best: bool = False,
    best_model_path: Optional[Path] = None,
) -> None:
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", float(metrics["accuracy"]))
        mlflow.log_metric("f1_macro", float(metrics["f1_macro"]))
        mlflow.log_metric("f1_weighted", float(metrics["f1_weighted"]))

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(confusion_matrix_path))

        if feature_importance_path is not None and feature_importance_path.exists():
            mlflow.log_artifact(str(feature_importance_path))

        if best_model_path is not None and best_model_path.exists():
            mlflow.log_artifact(str(best_model_path))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="sklearn_model",
            registered_model_name=REGISTERED_MODEL_NAME if best else None,
        )

        log.info("Logged MLflow run for model '%s'", model_name)


def log_final_best_model(run_id: str, model_uri: str) -> None:
    try:
        client = mlflow.tracking.MlflowClient()
        client.create_registered_model(REGISTERED_MODEL_NAME)
    except Exception:
        pass

    try:
        mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
        log.info("Registered best model with name '%s'", REGISTERED_MODEL_NAME)
    except Exception as exc:
        log.warning("Could not register best model automatically: %s", exc)
