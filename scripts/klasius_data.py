import pandas as pd
import requests
import json

from time import sleep
from bs4 import BeautifulSoup

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def convert_file_encoding(fin, fout, from_enc='Windows-1252', to_enc='utf-8'):
    # TODO: č is still encoded in a weird letter
    with open(fin, encoding=from_enc) as in_file, \
            open(fout, 'w', encoding=to_enc) as out_file:
        for line in in_file.readlines():
            print(line.strip(), file=out_file)


def extract_table_content(content):
    soup = BeautifulSoup(content, 'html.parser')
    data = {}
    trows = soup.find_all('tr')
    for trow in trows:
        cells = trow.find_all('td')
        key = cells[0].getText().strip().replace(':', '').lower()
        value = cells[1].getText().strip()
        if value:
            data[key] = value
    return data


if __name__ == '__main__':
    # convert_file_encoding('../csv_data/klasius_win1252-encoding.csv', '../csv_data/klasius.csv')
    srv_base_url = "https://www.stat.si/klasius/SrvDetails.aspx"  # https://www.stat.si/klasius/SrvDetails.aspx?srv=15001
    p16_base_url = "https://www.stat.si/klasius/PDetails.aspx"  # https://www.stat.si/klasius/PDetails.aspx?p=0721

    csv_data = pd.read_csv('../data/klasius.csv', delimiter=';', index_col=False)
    nrows = csv_data.shape[0]
    json_data = []

    csv_data["KLASIUS-P16 koda"] = csv_data["KLASIUS-P16 koda"].map(lambda x: x[2:-1])
    with open('../data/klasius.json', 'w') as ofile:
        for id, row in csv_data.iterrows():
            srv_code = row["KLASIUS-SRV koda"]
            p16_code = row["KLASIUS-P16 koda"]
            row_data = json.loads(row.to_json())

            srv_response = requests.get(srv_base_url, params={"srv": srv_code})
            print(f"[{id + 1}/{nrows}] Accessing {srv_response.url}, status [{srv_response.status_code}]")
            srv_content = extract_table_content(srv_response.content)
            row_data["KLASIUS-SRV"] = srv_content

            sleep(3)

            p16_response = requests.get(p16_base_url, params={"p": p16_code})
            print(f"[{id + 1}/{nrows}] Accessing {p16_response.url}, status [{p16_response.status_code}]")
            p16_content = extract_table_content(p16_response.content)
            row_data["KLASIUS-P16"] = p16_content

            print(json.dumps(row_data), file=ofile)
            sleep(5)
