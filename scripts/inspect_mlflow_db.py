import mlflow
from mlflow.tracking import MlflowClient
print('PYTHON', mlflow.__file__)
print('TRACKING_URI default:', mlflow.get_tracking_uri())
client = MlflowClient(tracking_uri='sqlite:///mlflow.db')
exps = client.list_experiments()
print('EXPERIMENTS FOUND:', len(exps))
for e in exps:
    print('-', e.experiment_id, e.name, 'lifecycle_stage', e.lifecycle_stage)
    runs = client.search_runs([e.experiment_id], 'attributes.start_time IS NOT NULL', max_results=1000)
    print('  runs count:', len(runs))

print('\nREGISTERED MODELS:')
for rm in client.list_registered_models():
    print('-', rm.name, 'versions:', [v.version for v in rm.latest_versions])

import sqlite3
con = sqlite3.connect('mlflow.db')
cur = con.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
print('\nTABLES:', [r[0] for r in cur.fetchall()])
try:
    cur.execute('SELECT experiment_id, name FROM experiments')
    rows = cur.fetchall()
    print('\nexperiments table rows:')
    for r in rows:
        print(r)
except Exception as e:
    print('could not read experiments table:', e)

con.close()
