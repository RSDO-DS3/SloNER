import pandas as pd
import numpy as np
import torch
import random

from typing import Union
from tqdm import trange, tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertTokenizer, BertForTokenClassification, AdamW, get_linear_schedule_with_warmup
from keras.preprocessing.sequence import pad_sequences

from train.model import Model
from utils.load_dataset import LoadDataset, LoadSSJ500k


class BertModel(Model):
    def __init__(self, load_dataset: LoadDataset, epochs: int = 3, max_grad_norm: float = 1.0):
        super().__init__(load_dataset)
        self.tokenizer = BertTokenizer.from_pretrained(
            'data/models/cro-slo-eng-bert',
            from_pt=True,
            do_lower_case=False
        )
        self.MAX_LENGTH = 128  # max input length
        self.BATCH_SIZE = 32  # max input length
        self.epochs = epochs
        self.max_grad_norm = max_grad_norm
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.tag2code, self.code2tag = self.load_dataset.encoding()

    def convert_input(self, input_data: pd.DataFrame):
        tokens = []
        tags = []  # NER tags

        for sentence, data in input_data.groupby("sentence"):
            sentence_tokens = []
            sentence_tags = []
            for id, word_row in data.iterrows():
                word_tokens = self.tokenizer.tokenize(word_row["word"])
                sentence_tokens.extend(word_tokens)
                sentence_tags.extend([self.tag2code[word_row["ner"]]] * len(word_tokens))

            sentence_ids = self.tokenizer.convert_tokens_to_ids(sentence_tokens)
            tokens.append(sentence_ids)
            tags.append(sentence_tags)
        # padding is required to spill the  in case there are sentences longer than 128 words
        tokens = torch.tensor(pad_sequences(
            tokens,
            maxlen=self.MAX_LENGTH,
            dtype="long",
            value=0.0,
            truncating="post",
            padding="post"
        ))
        tags = torch.tensor(pad_sequences(
            tags,
            maxlen=self.MAX_LENGTH,
            dtype="long",
            value=self.tag2code["PAD"],
            truncating="post",
            padding="post"
        ))
        masks = torch.tensor(np.array([[float(token != 0.0) for token in sentence] for sentence in tokens]))
        data = TensorDataset(tokens, masks, tags)
        sampler = RandomSampler(data)
        return DataLoader(data, sampler=sampler, batch_size=self.BATCH_SIZE)

    def flat_accuracy(self, preds, labels) -> float:
        pred_flat = np.argmax(preds, axis=2).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / float(len(labels_flat))

    def train(
            self,
            train_data: Union[pd.DataFrame, None] = None,
            validation_data: Union[pd.DataFrame, None] = None
    ) -> None:
        if not train_data:
            train_data = self.load_dataset.train(test=False)
        if not validation_data:
            validation_data = self.load_dataset.dev()

        print(self.tag2code)

        train_data = self.convert_input(train_data)
        model = BertForTokenClassification.from_pretrained(
            'data/models/cro-slo-eng-bert',
            num_labels=len(self.tag2code),
            output_attentions=False,
            output_hidden_states=False
        )

        model_parameters = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_parameters = [
            {
                'params': [p for n, p in model_parameters if not any(nd in n for nd in no_decay)],
                'weight_decay_rate': 0.01
            },
            {
                'params': [p for n, p in model_parameters if any(nd in n for nd in no_decay)],
                'weight_decay_rate': 0.0
            }
        ]
        optimizer = AdamW(
            optimizer_parameters,
            lr=3e-5,
            eps=1e-8
        )

        total_steps = len(train_data) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        seed_val = 42

        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        loss_values = []
        model.train()

        for _ in trange(self.epochs, desc="Epoch"):
            total_loss = 0

            for step, batch in tqdm(enumerate(train_data)):
                batch_tokens, batch_masks, batch_tags = tuple(t.to(self.device) for t in batch)

                # reset the grads
                model.zero_grad()

                outputs = model(
                    batch_tokens,
                    token_type_ids=None,
                    attention_mask=batch_masks,
                    labels=batch_tags
                )

                print(outputs)

                # loss = outputs[0]

                # loss.backward()
                break
            break

        # TODO: train the model
        # TODO: save the model!

    def test(self, test_data: pd.DataFrame) -> None:
        pass


if __name__ == '__main__':
    dataLoader = LoadSSJ500k()
    print(torch.__version__)
    bert = BertModel(dataLoader)
    bert.train()
    print("Here go the CroSloEng model specifics")
