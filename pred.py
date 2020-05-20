import subprocess

import pandas as pd
from tabulate import tabulate

import sys
import h2o

h2o.no_progress()

h2o.init()

print('\n' * 3)

model_id = sys.argv[1]
model_path = "/home/yoon/omelet/model/" + model_id + "/model.h2o/" + model_id

observation = h2o.import_file('data/customer_churn_test.csv')
saved_model = h2o.load_model(model_path)


print('Original Data Frame')
print('=' * 19)
print(tabulate(observation.as_data_frame(), headers='keys', tablefmt = 'fancy_gird', floatfmt = '.8f'))

print('\n' * 3)

print('Prediction Results')
print('=' * 18)
predictions = saved_model.predict(observation)
print(tabulate(predictions.as_data_frame(), headers='keys', tablefmt = 'fancy_gird', floatfmt = '.8f'))

