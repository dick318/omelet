import subprocess

import pandas as pd
from tabulate import tabulate


df = pd.read_csv('data/customer_churn_test.csv')

print(tabulate(df, headers='keys', tablefmt = 'fancy_gird', floatfmt = '.8f'))


a = subprocess.run('''curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"],"data":[[714, "Germany", "Female", 36, 2, 155060.41, 2, 1, 0, 167773.55]]}' http://192.168.100.172:9083/invocations''', shell=True)

print(a.predict)
