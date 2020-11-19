from typing import Any

import pandas as pd
from src.utils.load_dataset import LoadDataset


class Model:
    def __init__(self, load_dataset: LoadDataset) -> None:
        self.load_dataset = load_dataset

    def convert_input(self, input_data: pd.DataFrame) -> Any:
        """
            Convert the data to the correct input format for the model
            By default, we assume that it is already in the correct format
        :param input_data:
        :return:
        """
        return input_data

    def train(self, train_data: pd.DataFrame) -> None:
        pass

    def test(self, test_data: pd.DataFrame) -> None:
        pass
