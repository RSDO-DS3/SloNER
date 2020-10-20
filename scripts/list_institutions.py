import pandas as pd
import requests
import csv

from time import sleep
from bs4 import BeautifulSoup

if __name__ == '__main__':
    base_url = 'https://www.gov.si/zbirke/imenik-institucij/'
    start_id = 0
    institutions = []
    for id in range(10):
        response = requests.get(base_url, params={"nrOfItems": 100, "start": id * 100})
        print(response.url)
        soup = BeautifulSoup(response.content, 'html.parser')
        titles = soup.find_all('div', {'class': 'item-title'})
        for title in titles:
            institutions.append(title.find('h3').get_text().strip())
        sleep(5)
    df = pd.DataFrame(institutions, columns=["InstitutionName"])
    df.to_csv('../data/institution_names.csv', index=False, quoting=csv.QUOTE_ALL)
