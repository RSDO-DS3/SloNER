import pandas as pd
import csv

file_names = [
    '05X1015S_20201019-141850', 
    '05X1016S_20201019-142316', 
    '05X1017S_20201019-142341', 
    '05X1018S_20201019-142511'
]

df = pd.DataFrame([], columns=["LAST_NAME"])

for file_name in file_names:
    last_names = pd.read_csv(f'../data/{file_name}.csv')
    df = pd.concat([df, pd.DataFrame(last_names["LAST_NAME"].unique(), columns=["LAST_NAME"])])

df.to_csv(f'../data/last_names.csv', index=False, quoting=csv.QUOTE_ALL)
