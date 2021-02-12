import pandas as pd

from typing import Callable
from pathlib import Path


class UpdateDocuments:
    def __init__(self, path):
        self.path = path


class UpdateBSNLPDocuments(UpdateDocuments):
    def __init__(
        self,
        year: str = 'all',
        lang: str = 'all',
    ) -> None:
        super(UpdateBSNLPDocuments, self).__init__(
            path="./data/datasets/bsnlp"
        )
        datasets = {
            "2017": ["ec", "trump"],
            "2021": ["asia_bibi", "brexit", "nord_stream", "other", "ryanair"],
            "all": ["ec", "trump", "asia_bibi", "brexit", "nord_stream", "other", "ryanair"],
        }
        if year not in datasets:
            raise Exception(f"Invalid subset chosen: {year}")
        self.dirs = datasets[year]
        available_langs = ['bg', 'cs', 'pl', 'ru', 'sl', 'uk']
        if lang in available_langs:
            self.langs = [lang]
        elif lang == 'all':
            self.langs = available_langs
        else:
            raise Exception("Invalid language option.")

    def __update(
        self,
        ftype: str,
        data: dict,
        fun: Callable
    ) -> None:
        for dataset, langs in data.items():
            if dataset not in self.dirs:
                raise Exception(f"Unrecognized dataset: {dataset}")
            for lang, documents in langs.items():
                if lang not in self.langs:
                    raise Exception(f"Unrecognized language: {lang}")
                path = f'{self.path}/{dataset}/{ftype}/{lang}'
                Path(path).mkdir(parents=True, exist_ok=True)
                for docId, content in documents.items():
                    fun(f'{path}/{content["fname"]}', content)

    def update_merged(self, new_data) -> None:
        def update_merged(fpath: str, doc: dict) -> None:
            df = pd.DataFrame(doc['content'])
            df.to_csv(fpath)
        self.__update('predicted', new_data, update_merged)
