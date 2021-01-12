import pandas as pd
import numpy as np
import torch
import transformers
import random
import os
import argparse
import time

from typing import Union
from tqdm import trange, tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertTokenizer, BertForTokenClassification, AdamW
from transformers import get_linear_schedule_with_warmup, PreTrainedModel
from keras.preprocessing.sequence import pad_sequences
from seqeval.metrics import f1_score, accuracy_score, classification_report
from matplotlib import pyplot as plt

from src.train.model import Model
from src.utils.load_dataset import LoadDataset, LoadSSJ500k, LoadBSNLP
from src.utils.utils import list_dir


class BertModel(Model):
    def __init__(
        self, 
        load_dataset: LoadDataset, 
        epochs: int = 3, 
        max_grad_norm: float = 1.0,
        input_model_path: str = 'data/models/cro-slo-eng-bert',  # this is a directory
        output_model_path: str = 'data/models/cro-slo-eng-bert-ssj500k/',  # this is the output dir
        output_model_fname: str = 'cro-slo-eng-bert-ssj500k',  # this is the output file name, the current time is appended automatically and `.pk`
        tune_entire_model: bool = True
    ):
        super().__init__(load_dataset)
        self.input_model_path = input_model_path
        self.output_model_path = output_model_path
        self.output_model_fname = output_model_fname
        print(f"Output model at: {output_model_path}")
        
        print(f"Tuning entire model: {tune_entire_model}")
        self.tune_entire_model = tune_entire_model

        self.tokenizer = BertTokenizer.from_pretrained(
            self.input_model_path,
            from_pt=True,
            do_lower_case=False
        )
        self.MAX_LENGTH = 128  # max input length
        self.BATCH_SIZE = 32  # max input length
        self.epochs = epochs
        self.max_grad_norm = max_grad_norm
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.tag2code, self.code2tag = self.load_dataset.encoding()
        print(f"tags: ", self.tag2code.keys())

    def convert_input(self, input_data: pd.DataFrame):
        tokens = []
        tags = []  # NER tags

        for sentence, data in input_data.groupby("sentence"):
            sentence_tokens = []
            sentence_tags = []
            for id, word_row in data.iterrows():
                word_tokens = self.tokenizer.tokenize(str(word_row["word"]))
                sentence_tokens.extend(word_tokens)
                sentence_tags.extend([self.tag2code[word_row["ner"]]] * len(word_tokens))

            sentence_ids = self.tokenizer.convert_tokens_to_ids(sentence_tokens)
            tokens.append(sentence_ids)
            tags.append(sentence_tags)
        # padding is required to spill the sentence tokens in case there are sentences longer than 128 words
        # or to fill in the missing places to 128 (self.MAX_LENGTH)
        tokens = torch.as_tensor(pad_sequences(
            tokens,
            maxlen=self.MAX_LENGTH,
            dtype="long",
            value=0.0,
            truncating="post",
            padding="post"
        )).to(self.device)
        tags = torch.as_tensor(pad_sequences(
            tags,
            maxlen=self.MAX_LENGTH,
            dtype="long",
            value=self.tag2code["PAD"],
            truncating="post",
            padding="post"
        )).to(self.device)
        masks = torch.as_tensor(np.array([[float(token != 0.0) for token in sentence] for sentence in tokens])).to(self.device)
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

        print("Loading the training data...")
        train_data = self.convert_input(train_data)
        print("Loading the validation data...")
        validation_data = self.convert_input(validation_data)

        print("Loading the pre-trained model...")
        model = BertForTokenClassification.from_pretrained(
            self.input_model_path,
            num_labels=len(self.tag2code),
            output_attentions=False,
            output_hidden_states=False
        )
        model = torch.nn.DataParallel(model.cuda(), device_ids=[0])

        if self.tune_entire_model:
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
        else:
            model_parameters = list(model.named_parameters())
            optimizer_parameters = [{"params": [p for n, p in model_parameters]}]

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

        # ensure reproducibility
        # TODO: try out different seed values
        seed_val = 42
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        training_loss, validation_loss = [], []
        print(f"Training the model for {self.epochs} epochs...")
        for _ in trange(self.epochs, desc="Epoch"):
            model.train()
            total_loss = 0
            # train:
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

                loss = outputs[0]
                loss.backward()
                total_loss += loss.item()

                # preventing exploding gradients
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.max_grad_norm)

                # update the parameters
                optimizer.step()

                # update the learning rate (lr)
                scheduler.step()

            avg_epoch_train_loss = total_loss/len(train_data)
            print(f"Avg train loss = {avg_epoch_train_loss}")
            training_loss.append(avg_epoch_train_loss)

            # validate:
            val_loss, val_acc, val_f1, val_report = self.__test(model, validation_data)
            validation_loss.append(val_loss)
            print(f"Validation loss: {val_loss}")
            print(f"Validation accuracy: {val_acc}")
            print(f"Validation F1 score: {val_f1}")
            print(f"Classification report:")
            print(f"{val_report}")

        out_fname = f"{self.output_model_fname}-{time.time()}"

        fig, ax = plt.subplots()
        ax.plot(training_loss, label="Traing loss")
        ax.plot(validation_loss, label="Validation loss")
        ax.legend()
        ax.set_title("Model Loss")
        fig.savefig(f"./figures/{out_fname}-loss.png")

        out_fname = f"{self.output_model_path}/{out_fname}"
        print(f"Saving the model at: {out_fname}")
        torch.save(model, f"{out_fname}.pk")
        torch.save({
                "epoch": self.epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            }, 
            f"{out_fname}_weights"
        )
        print("Done!")
    
    def translate(self, predictions: list, labels: list) -> (list, list):
        translated_predictions, translated_labels = [], []
        for preds, labs in zip(predictions, labels):
            sentence_predictions, sentence_labels = [], []
            for p, l in zip(preds, labs):
                if l == self.tag2code["PAD"]:
                    continue
                sentence_predictions.append(self.code2tag[p])
                sentence_labels.append(self.code2tag[l])
            translated_predictions.append(sentence_predictions)
            translated_labels.append(sentence_labels)
        return translated_predictions, translated_labels

    def __test(self, model: PreTrainedModel, data: DataLoader) -> (float, float, float, str):
        eval_loss, eval_accuracy = 0., 0.
        eval_steps, eval_examples = 0, 0
        eval_predictions, eval_labels = [], []
        model.eval()
        for batch in tqdm(data):
            batch_tokens, batch_masks, batch_tags = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                outputs = model(
                    batch_tokens,
                    token_type_ids=None,
                    attention_mask=batch_masks,
                    labels=batch_tags
                )
            logits = outputs[1].detach().cpu().numpy()
            label_ids = batch_tags.to('cpu').numpy()

            eval_loss += outputs[0].mean().item()
            eval_accuracy += self.flat_accuracy(logits, label_ids)
            eval_predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            eval_labels.extend(label_ids)

            eval_examples += batch_tokens.size(0)
            eval_steps += 1

        eval_loss = eval_loss / eval_steps
        
        predicted_tags, valid_tags = self.translate(eval_predictions, eval_labels)

        score_acc = accuracy_score(valid_tags, predicted_tags)
        score_f1 = f1_score(valid_tags, predicted_tags)
        report = classification_report(valid_tags, predicted_tags)

        return eval_loss, score_acc, score_f1, report

    def test(self, test_data: Union[pd.DataFrame, None] = None) -> None:
        if test_data is None:
            test_data = self.load_dataset.test()
        if not (os.path.exists(self.output_model_path) and os.path.isdir(self.output_model_path)):
            raise Exception(f"A model with the given parameters has not been trained yet,"\
            f" or is not located at `{self.output_model_path}`.")
        _, models = list_dir(self.output_model_path)
        models = [model_fname for model_fname in models if model_fname.startswith(self.output_model_fname) and model_fname.endswith('.pk')]
        if not models:
            raise Exception(f"There are no trained models with the given criteria: `{self.output_model_fname}`")

        print("Loading the testing data...")
        test_data = self.convert_input(test_data)
        avg_acc, avg_f1, reports = [], [], []
        for model_fname in models:
            print(f"Loading {model_fname}...")
            model = torch.load(
                f"{self.output_model_path}/{model_fname}",
                map_location=torch.device(self.device)
            )
            _, acc, f1, report = self.__test(model, test_data)
            avg_acc.append(acc)
            avg_f1.append(f1)
            print(f"Testing accuracy: {acc}")
            print(f"Testing F1 score: {f1}")
            print(f"Testing classification report:\n{report}")
        print(f"Average accuracy: {np.mean(avg_acc):.4f}")
        print(f"Average f1: {np.mean(avg_f1):.4f}")
        print("Done.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--train-iterations', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--full-finetuning', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Training: {args.train}")
    print(f"Train iterations: {args.train_iterations}")
    print(f"Epochs: {args.epochs}")
    print(f"Full finetuning: {args.full_finetuning}")
    print(f"Testing: {args.test}")
    model_name = "cro-slo-eng-bert"  # "bert-base-multilingual-cased"
    train_dataset = "bsnlp"  # "ssj500k"
    test_dataset = "bsnlp"
    # dataLoader = LoadSSJ500k()
    dataLoader = LoadBSNLP('sl')
    # testDataLoader = LoadBSNLP('sl')
    testDataLoader = LoadSSJ500k()
    bert = BertModel(
        dataLoader,
        epochs=args.epochs,
        input_model_path=f'./data/models/{model_name}',
        output_model_path=f'./data/models/{model_name}-{train_dataset}',
        output_model_fname=f'{model_name}-{train_dataset}'\
                            f"{'-finetuned' if args.full_finetuning else ''}"\
                            f'-{args.epochs}-epochs',
        tune_entire_model=args.full_finetuning
    )
    
    if args.train:
        for i in range(args.train_iterations):
            print(f"Building training model {i + 1}")
            bert.train()
    
    if args.test:
        bert.test(test_data=testDataLoader.test())

if __name__ == '__main__':
    main()
