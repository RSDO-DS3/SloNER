import pandas as pd
import pyconll
import numpy as np

from sklearn.model_selection import train_test_split
from src.utils.utils import list_dir

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
                    data.append({"word": word.form, "sentence": id, "ner": annotation.upper()})
                    # NOTE: we cannot use the just <TYPE> annotation without `B-` (begin) or `I-` (inside) `<TYPE>`
                    # because we would not be compliant with the CoNLL format
                    # annotation = annotation if annotation != "O" else "B-O"
                    # data.append({"word": word.form, "sentence": id, "ner": annotation.split("-")[1].upper()})
                else:
                    data.append({"word": word.form, "sentence": id, "ner": "O"})
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


class LoadBSNLP(LoadDataset):
    def __init__(self, lang: str):
        super().__init__(
            "data/datasets/bsnlp",
            "csv"
        )
        self.lang = lang
        self.random_state = 42

    def load(self, set: str) -> pd.DataFrame:
        dirs, _ = list_dir(self.base_fname)
        data = pd.DataFrame()
        for directory in dirs:
            base_path = f"{self.base_fname}/{directory}/{self.lang}"
            _, files = list_dir(base_path)
            for fname in files:
                df = pd.read_csv(f"{base_path}/{fname}")
                df = df[['docId', 'sentenceId', 'text', 'ner']]
                df['sentenceId'] =  df['docId'].astype(str) + '-' + df['sentenceId'].astype('str')
                df = df.drop(columns=['docId'])
                df = df.rename(columns={"sentenceId": "sentence", "text": "word", "ner": "ner"})
                data = pd.concat([data, df])
        train_data, test_data = train_test_split(
                                    data,
                                    test_size=0.2,
                                    random_state=self.random_state,
                                    shuffle=True,
                                    stratify=data['ner']
                                )
        val_data, test_data = train_test_split(
                                    test_data,
                                    test_size=0.5,
                                    random_state=self.random_state,
                                    shuffle=True,
                                    stratify=test_data['ner']
                                )
        return {
            "train": train_data,
            "dev": val_data,
            "test": test_data,
        }[set]

    def train(self, test: bool = False) -> pd.DataFrame:
        return self.load('train')

    def dev(self) -> pd.DataFrame:
        """
            This is the validation data
        """
        return self.load('dev')

    def test(self) -> pd.DataFrame:
        return self.load('test')

    def encoding(self) -> (dict, dict):
        return {}, {}
    

if __name__ == '__main__':
    loader = LoadBSNLP("sl")
    # loader = LoadSSJ500k()

    train_data = loader.train()
    print(f"Train data: {train_data.shape[0]}, NERs: {train_data.loc[train_data['ner'] != 'O'].shape[0]}")
    print(train_data['ner'].value_counts())
    
    dev_data = loader.dev()
    print(f"Validation data: {dev_data.shape[0]}, NERs: {dev_data.loc[dev_data['ner'] != 'O'].shape[0]}")
    print(dev_data['ner'].value_counts())
    
    test_data = loader.test()
    print(f"Test data: {test_data.shape[0]}, NERs: {test_data.loc[test_data['ner'] != 'O'].shape[0]}")
    print(test_data['ner'].value_counts())
