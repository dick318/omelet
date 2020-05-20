import subprocess
import sys
import argparse


uuid = sys.argv[1]
host = '192.168.100.172'
port = sys.argv[2]

print('mlruns/1/' + uuid + '/artifacts/model')

subprocess.run('mlflow models serve -m ' + 'mlruns/1/' + uuid + '/artifacts/model ' + '--host ' + host + ' -p ' + port, shell=True)


