import mlflow
import mlflow_tracking
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

print('CALLING setup_experiment()')
mlflow_tracking.setup_experiment()
client = mlflow.tracking.MlflowClient()
exps = client.list_experiments()
print('experiments:', [e.name for e in exps])
