import pandas as pd
import numpy as np
import shapefile # not available in python 3.8 (?)
import csv

if __name__ == '__main__':
    with shapefile.Reader("data/ner_resources/REZI_D48/REZI_D48", encoding="ISO8859-1") as shp:
        print(shp.fields)
        field_names = [field[0] for field in shp.fields]
        print(field_names)
        records = shp.records()
        json_records = []
        for record in records:
            json_record = {}
            for field in field_names:
                try:
                    json_record[field] = record[field]
                except:
                    json_record[field] = None
            json_records.append(json_record)
        df = pd.DataFrame(json_records)
        names = np.concatenate([df["BESEDILO"].unique(), df["KRAJSAVA"].unique()])
        names = pd.DataFrame(names, columns=["Name"])
        names.to_csv("data/ner_resources/geo_data.csv", index=False, quoting=csv.QUOTE_ALL)
        df.to_csv("data/ner_resources/geo_data_all.csv", index=False, quoting=csv.QUOTE_ALL)
