import pandas as pd
import pyconll
import numpy as np


class LoadDataset:
    def __init__(self, base_fname: str, format: str):
        self.base_fname = base_fname
        self.data_format = format

    def load(self, set: str) -> pd.DataFrame:
        return pd.DataFrame()

    def train(self, test: bool = False) -> pd.DataFrame:
        return pd.DataFrame()

    def dev(self) -> pd.DataFrame:
        """
            This is the validation data
        """
        return pd.DataFrame()

    def test(self) -> pd.DataFrame:
        return pd.DataFrame()

    def encoding(self) -> (dict, dict):
        return {}, {}


class LoadSSJ500k(LoadDataset):
    def __init__(self):
        super().__init__(
            "data/datasets/ssj500k/",
            "conll"
        )

    def load(self, set: str) -> pd.DataFrame:
        raw_data = pyconll.load_from_file(f"{self.base_fname}{set}_ner.conllu")
        data = []
        for id, sentence in enumerate(raw_data):
            for word in sentence:
                if word.upos == 'PROPN':  # check if the token is a NER
                    annotation = list(word.misc.keys())[0]
                    # some undocumented annotations "O", for now classified as other
                    annotation = annotation if annotation != "O" else "I-othr"
                    data.append({"word": word.form, "sentence": id, "ner": annotation.split("-")[1]})
                else:
                    data.append({"word": word.form, "sentence": id, "ner": "othr"})
        return pd.DataFrame(data)

    def train(self, test: bool = False) -> pd.DataFrame:
        return self.load('train')

    def dev(self) -> pd.DataFrame:
        return self.load('dev')

    def test(self) -> pd.DataFrame:
        return self.load('test')

    def encoding(self, test: bool = False):
        data = self.load('train')
        possible_tags = np.append(data["ner"].unique(), ["PAD"])
        tag2code = {tag: code for code, tag in enumerate(possible_tags)}
        code2tag = {val: key for key, val in tag2code.items()}
        return tag2code, code2tag


if __name__ == '__main__':
    loader = LoadSSJ500k()

    train_data = loader.train()
    print(f"Train data: {train_data.shape[0]}, NERs: {train_data.loc[train_data['ner'] == True].shape[0]}")

    dev_data = loader.dev()
    print(f"Validation data: {dev_data.shape[0]}, NERs: {dev_data.loc[dev_data['ner'] == True].shape[0]}")

    test_data = loader.test()
    print(f"Test data: {test_data.shape[0]}, NERs: {test_data.loc[test_data['ner'] == True].shape[0]}")
