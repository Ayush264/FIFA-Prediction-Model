import sys
import time
print('START', sys.executable)
print('sys.path[0]=', sys.path[0])
print('IMPORTING mlflow...')
start = time.time()
import mlflow
end = time.time()
print('IMPORTED', mlflow.__version__, 'in', end-start)
print('END')
