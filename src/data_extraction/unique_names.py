import pandas as pd
import csv

file_names = [
    'imena_deckov', 'imena_deklic', 'moska_imena', 'zenska_imena'
]

for file_name in file_names:
    names = pd.read_csv(f'data/{file_name}_all.csv')
    df = pd.DataFrame(names["FIRST_NAME"].unique(), columns=["FIRST_NAME"])
    df.to_csv(f'data/{file_name}.csv', index=False, quoting=csv.QUOTE_ALL)

