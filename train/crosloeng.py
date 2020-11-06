import pandas as pd
import torch

from typing import Union
from transformers import BertTokenizer, BertConfig, BertForTokenClassification, AdamW
from train.model import Model
from utils.load_dataset import LoadDataset, LoadSSJ500k


class BertModel(Model):
    def __init__(self, load_dataset: LoadDataset):
        super().__init__(load_dataset)

    def convert_input(self, input_data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame()

    def train(self, train_data: Union[pd.DataFrame, None] = None) -> None:
        if not train_data:
            train_data = self.load_dataset.train()
        # TODO: train the model
        # TODO: save the model!

    def test(self, test_data: pd.DataFrame) -> None:
        pass


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('data/models/cro-slo-eng-bert', from_pt=True, do_lower_case=False)
    dataLoader = LoadSSJ500k()
    bert = BertModel(dataLoader)
    print("Here go the CroSloEng model specifics")
