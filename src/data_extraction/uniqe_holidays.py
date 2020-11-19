import pandas as pd
import csv

df = pd.read_csv('../data/ner_resources/seznampraznikovindelaprostihdni20002030.csv', delimiter=';')

unique_holidays = df['IME_PRAZNIKA'].unique()

holidays = pd.DataFrame(unique_holidays, columns=["IME_PRAZNIKA"])

holidays.to_csv('data/seznampraznikov.csv', index=False, quoting=csv.QUOTE_ALL)
