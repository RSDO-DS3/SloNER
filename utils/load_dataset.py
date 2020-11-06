import pandas as pd
import pyconll


class LoadDataset:
    def __init__(self, base_fname: str, format: str):
        self.base_fname = base_fname
        self.data_format = format

    def load(self, set: str) -> pd.DataFrame:
        return pd.DataFrame()

    def train(self) -> pd.DataFrame:
        return pd.DataFrame()

    def dev(self) -> pd.DataFrame:
        """
            This is the validation data
        """
        return pd.DataFrame()

    def test(self) -> pd.DataFrame:
        return pd.DataFrame()


class LoadSSJ500k(LoadDataset):
    def __init__(self):
        super().__init__(
            "data/datasets/ssj500k/ssj500k.conllu/sl_ssj-ud_v2.4",
            "conll"
        )

    def load(self, set: str) -> pd.DataFrame:
        raw_data = pyconll.load_from_file(f"{self.base_fname}-{set}.conllu")
        data = []
        for sentence in raw_data:
            for word in sentence:
                if word.upos == 'PROPN':  # check if the token is a NER
                    data.append({"form": word.form, "upos": word.upos, "ner": True})
                else:
                    data.append({"form": word.form, "upos": word.upos, "ner": False})
        return pd.DataFrame(data)

    def train(self) -> pd.DataFrame:
        return self.load('train')

    def dev(self) -> pd.DataFrame:
        return self.load('dev')

    def test(self) -> pd.DataFrame:
        return self.load('test')


if __name__ == '__main__':
    loader = LoadSSJ500k()

    train_data = loader.train()
    print(f"Train data: {train_data.shape[0]}, NERs: {train_data.loc[train_data['ner'] == True].shape[0]}")

    dev_data = loader.dev()
    print(f"Validation data: {dev_data.shape[0]}, NERs: {dev_data.loc[dev_data['ner'] == True].shape[0]}")

    test_data = loader.test()
    print(f"Test data: {test_data.shape[0]}, NERs: {test_data.loc[test_data['ner'] == True].shape[0]}")
