from mlflow.tracking import MlflowClient
c=MlflowClient(tracking_uri='sqlite:///mlflow.db')
print('CLIENT_CLASS', type(c))
print('HAS_METHODS', [m for m in dir(c) if not m.startswith('_')])
# try available experiment listing methods
for name in ('list_experiments','search_experiments','get_experiment','get_experiment_by_name'):
    print('method', name, hasattr(c, name))
# try get by name if available
try:
    exp = c.get_experiment_by_name('FIFA_World_Cup_Prediction')
    print('get_by_name:', exp)
except Exception as e:
    print('get_experiment_by_name error', e)
# try get all experiments via the REST-ish API: call client._get_paginated
try:
    if hasattr(c,'_get_paginated'):
        print('has _get_paginated')
except Exception:
    pass
print('done')
