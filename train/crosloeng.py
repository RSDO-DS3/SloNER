import pandas as pd
import numpy as np
import torch

from typing import Union
from transformers import BertTokenizer, BertConfig, BertForTokenClassification, AdamW
from keras.preprocessing.sequence import pad_sequences

from train.model import Model
from utils.load_dataset import LoadDataset, LoadSSJ500k


class BertModel(Model):
    def __init__(self, load_dataset: LoadDataset):
        super().__init__(load_dataset)
        self.tokenizer = BertTokenizer.from_pretrained(
            'data/models/cro-slo-eng-bert',
            from_pt=True,
            do_lower_case=False
        )
        self.MAX_LENGTH = 128  # max input length

    def dataset_encoding(self, data: pd.DataFrame) -> (dict, dict):
        possible_tags = np.append(data["ner"].unique(), ["PAD"])
        tag2code = {tag: code for code, tag in enumerate(possible_tags)}
        code2tag = {val: key for key, val in tag2code.items()}
        return tag2code, code2tag

    def convert_input(self, input_data: pd.DataFrame) -> (np.ndarray, np.ndarray, np.ndarray):
        tag2code, code2tag = self.dataset_encoding(input_data)
        tokens = []
        tags = []  # NER tags

        for sentence, data in input_data.groupby("sentence"):
            sentence_tokens = []
            sentence_tags = []
            for id, word_row in data.iterrows():
                word_tokens = self.tokenizer.tokenize(word_row["word"])
                sentence_tokens.extend(word_tokens)
                sentence_tags.extend([tag2code[word_row["ner"]]] * len(word_tokens))

            sentence_ids = self.tokenizer.convert_tokens_to_ids(sentence_tokens)
            tokens.append(sentence_ids)
            tags.append(sentence_tags)

        tokens = pad_sequences(
            tokens,
            maxlen=self.MAX_LENGTH,
            dtype="long",
            value=0.0,
            truncating="post",
            padding="post"
        )
        tags = pad_sequences(
            tokens,
            maxlen=self.MAX_LENGTH,
            dtype="long",
            value=tag2code["PAD"],
            truncating="post",
            padding="post"
        )
        masks = np.array([[float(token != 0.0) for token in sentence] for sentence in tokens])

        return tokens, tags, masks

    def train(self, train_data: Union[pd.DataFrame, None] = None) -> None:
        if not train_data:
            train_data = self.load_dataset.train()
        tokens, tags, masks = self.convert_input(train_data)

        # model = BertForTokenClassification.from_pretrained(
        #     'data/models/cro-slo-eng-bert',
        #     num_labels=0,
        #     output_attentions=False,
        #     output_hidden_states=False
        # )
        # TODO: train the model
        # TODO: save the model!
        print(train_data.head())

    def test(self, test_data: pd.DataFrame) -> None:
        pass


if __name__ == '__main__':
    dataLoader = LoadSSJ500k()
    bert = BertModel(dataLoader)
    bert.train()
    print("Here go the CroSloEng model specifics")
