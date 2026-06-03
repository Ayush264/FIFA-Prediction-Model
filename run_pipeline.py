"""
FIFA World Cup Prediction - Master Pipeline Runner
Executes the full end-to-end pipeline:
  Step 1: PySpark ETL (Data Cleaning)
  Step 2: PySpark Feature Engineering
  Step 3: PySpark Ranking Pipeline
  Step 4: Model Training (LR / RF / XGBoost)
  Step 5: Monte Carlo Simulation (10,000 runs)
"""

import sys
import time
import logging
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", encoding="utf-8"),
    ]
)
log = logging.getLogger("master_pipeline")


def step(name: str):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            log.info(f"\n{'='*60}")
            log.info(f"  STEP: {name}")
            log.info(f"{'='*60}")
            start = time.time()
            result = fn(*args, **kwargs)
            elapsed = time.time() - start
            log.info(f"  [OK] {name} completed in {elapsed:.1f}s")
            return result
        return wrapper
    return decorator


@step("1 — PySpark ETL: Data Cleaning")
def run_cleaning():
    from pyspark_jobs.data_cleaning import run
    run()


@step("2 — PySpark ETL: Feature Engineering")
def run_feature_engineering():
    from pyspark_jobs.feature_engineering import run
    run()


@step("3 — PySpark ETL: Ranking Pipeline")
def run_ranking():
    from pyspark_jobs.ranking_pipeline import run
    run()


@step("4 — Model Training")
def run_training():
    sys.path.insert(0, "models")
    from models.train_model import train_all
    results = train_all()
    return results


@step("5 — Monte Carlo Simulation (10,000 runs)")
def run_simulation():
    from simulation.monte_carlo import run_monte_carlo
    results = run_monte_carlo(n_simulations=10_000)
    return results


def main():
    log.info("\n" + "=" * 60)
    log.info("  FIFA WORLD CUP 2026 PREDICTION PLATFORM")
    log.info("  End-to-End Pipeline Starting ...")
    log.info("=" * 60)

    overall_start = time.time()

    run_cleaning()
    run_feature_engineering()
    run_ranking()
    run_training()
    results = run_simulation()

    elapsed = time.time() - overall_start

    log.info(f"\n{'='*60}")
    log.info(f"  PIPELINE COMPLETE in {elapsed:.1f}s")
    log.info(f"{'='*60}")
    log.info("\n[RESULTS] Top 5 Championship Favourites:")
    if results is not None:
        print(results.head(5).to_string(index=False))

    log.info("\n[OUTPUTS] Generated files:")
    log.info("  data/processed/*.csv    -> Cleaned datasets")
    log.info("  data/feature_store/*.csv -> Feature store exports")
    log.info("  models/*.pkl            -> Trained models")
    log.info("  models/evaluation_report.json -> Model metrics")
    log.info("  dashboard/championship_probabilities.csv -> Monte Carlo results")
    log.info("  dashboard/monte_carlo_summary.json       -> Full simulation data")
    log.info("\nConnect dashboard/championship_probabilities.csv to Power BI!")


if __name__ == "__main__":
    main()
