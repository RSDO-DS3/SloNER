import json
import logging
import sys
import torch
import pandas as pd
import numpy as np

from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import PreTrainedModel, pipeline
from collections import defaultdict
from operator import itemgetter
from tqdm import tqdm

from src.utils.load_dataset import LoadBSNLP


logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s'
)
logger = logging.getLogger('MakePrediction')


class MakePrediction:
    def __init__(
        self,
        model_path: str = f'./data/models/bert-base-multilingual-cased-other',
        use_device: int = 0
    ):
        """
            A class to extract all the NE predictions from a given tokens
        :param model_path: path to a HuggingFace-transformers pre-trained model for the NER task, such as BERT Base Multilingual (Un)Cased
        """
        self.model_path = model_path
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            output_attentions=False,
            output_hidden_states=False
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            from_pt=True,
            do_lower_case=False,
            use_fast=False
        )
        self.ner_pipeline = pipeline(
            'ner',
            model=self.model,
            tokenizer=self.tokenizer,
            framework="pt",
            device=use_device,
        )

    def __merge_tokenized_nes(
        self,
        raw_nes: list
    ) -> list:
        """
            Merges the NE tokens provided by the model, e.g. tokens such as `N`, `##I`, `##J`, `##Z` into `NIJZ`
        :param raw_nes:
        :return:
        """
        nes = []
        for i, ne in enumerate(raw_nes):
            if ne['word'].startswith('##'):
                continue
            j = i + 1
            ne['modelTokens'] = 1
            category = defaultdict(lambda: 0)
            category[ne['entity']] += 1
            while j < len(raw_nes) and raw_nes[j]['word'].startswith('##'):
                if raw_nes[j]['index'] != (raw_nes[j - 1]['index'] + 1):
                    logger.debug("Tokens are not coming one after the other, skipping")
                    break
                ne['word'] += f'{raw_nes[j]["word"][2:]}'
                ne['score'] = (ne['score'] + raw_nes[j]['score']) / 2
                ne['modelTokens'] += 1
                category[raw_nes[j]['entity']] += 1
                j += 1
            ne['word'] = ne['word'].replace('▁', '')
            ne['entity'] = max(category.items(), key=itemgetter(1))[0]  # majority voting
            nes.append(ne)
        return nes

    def __merge_nes(
        self,
        nes: list
    ) -> list:
        """
            Merges the NEs in the form of the expected output
        :param nes:
        :return:
        """
        merged = []
        for i, ne in enumerate(nes):
            if ne['entity'].startswith('I-'):
                continue
            j = i + 1
            ne['numTokens'] = 1
            while j < len(nes) and not nes[j]['entity'].startswith('B-'):
                ne['word'] += f' {nes[j]["word"]}'
                ne['score'] = (ne['score'] + nes[j]['score']) / 2
                ne['modelTokens'] += nes[j]['modelTokens']
                ne['numTokens'] += 1
                j += 1
            ne['entity'] = ne['entity'][2:]
            del ne['index']
            merged.append(ne)
        return merged

    def get_ners(
        self,
        data: str,
        merge_nes: bool = False,
    ) -> list:
        """
            Get the NEs from a particular tokens [stored in data], provided as a whole string.
        :param data: The input data from which the NEs are extracted
        :param merge_nes: Indicator to merge the entities together, i.e. B-XXX and I-XXX into XXX
        :return:
        """
        raw_ners = self.ner_pipeline(data)
        nes = self.__merge_tokenized_nes(raw_ners)
        if merge_nes:
            return self.__merge_nes(nes)
        return nes


class ExtractPredictions:
    def __init__(
        self,
        model_path: str = f'./data/models/bert-base-multilingual-cased-other',
    ):
        """
            A class to extract all the NE predictions from a given tokens
        :param model_path: path to a HuggingFace-transformers pre-trained model for the NER task, such as BERT Base Multilingual (Un)Cased
        """
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            output_attentions=False,
            output_hidden_states=False
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            from_pt=True,
            do_lower_case=False,
            use_fast=False
        )
        self.BATCH_SIZE = 32
        self.MAX_LENGTH = 128

    def convert_input(
        self,
        input_data: pd.DataFrame,
        tag2code: dict,
    ) -> (DataLoader, list):
        all_ids = []
        ids = []  # sentence ids
        tokens = []  # sentence tokens
        token_ids = []  # converted sentence tokens
        tags = []  # NER tags

        for sentence, data in input_data.groupby("sentence"):
            sentence_tokens = []
            sentence_tags = []
            sentence_ids = []
            for id, word_row in data.iterrows():
                word_tokens = self.tokenizer.tokenize(str(word_row["word"]))
                sentence_tokens.extend(word_tokens)
                sentence_tags.extend([tag2code[word_row["ner"]]] * len(word_tokens))
                token_id_str = f'{sentence};{word_row["tokenId"]}'
                all_ids.append(token_id_str)
                token_id = len(all_ids) - 1
                sentence_ids.extend([token_id] * len(word_tokens))
            if len(sentence_tokens) != len(sentence_tags) != len(sentence_ids):
                raise Exception("Inconsistent output!")
            ids.append(sentence_ids)
            tokens.append(sentence_tokens)
            sentence_token_ids = self.tokenizer.convert_tokens_to_ids(sentence_tokens)
            token_ids.append(sentence_token_ids)
            tags.append(sentence_tags)
        # padding is required to spill the sentence tokens in case there are sentences longer than 128 words
        # or to fill in the missing places to 128 (self.MAX_LENGTH)
        ids = torch.as_tensor(pad_sequences(
            ids,
            maxlen=self.MAX_LENGTH,
            dtype="long",
            value=-1,
            truncating="post",
            padding="post"
        )).to(self.device)
        token_ids = torch.as_tensor(pad_sequences(
            token_ids,
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
            value=tag2code["PAD"],
            truncating="post",
            padding="post"
        )).to(self.device)
        masks = torch.as_tensor(np.array([[float(token != 0.0) for token in sentence] for sentence in token_ids])).to(self.device)
        data = TensorDataset(ids, token_ids, masks, tags)
        sampler = RandomSampler(data)
        return DataLoader(data, sampler=sampler, batch_size=self.BATCH_SIZE), all_ids

    def translate(
        self,
        predictions: list,
        labels: list,
        tokens: list,
        sent_ids: list,
        tag2code: dict,
        code2tag: dict,
        all_ids: list
    ) -> (list, list, list, list):
        translated_predictions, translated_labels, translated_tokens, translated_sentences = [], [], [], []
        for preds, labs, toks, ids in zip(predictions, labels, tokens, sent_ids):
            sentence_predictions, sentence_labels, sentence_tokens, sentence_ids = [], [], [], []
            for p, l, t, i in zip(preds, labs, toks, ids):
                if l == tag2code["PAD"]:
                    continue
                sentence_tokens.append(t)
                sentence_predictions.append(code2tag[p])
                sentence_labels.append(code2tag[l])
                sentence_ids.append(all_ids[i])
            translated_tokens.append(sentence_tokens)
            translated_predictions.append(sentence_predictions)
            translated_labels.append(sentence_labels)
            translated_sentences.append(sentence_ids)
        return translated_predictions, translated_labels, translated_tokens, translated_sentences

    def test(
        self,
        data: DataLoader,
        all_ids: list,
        tag2code: dict,
        code2tag: dict,
    ) -> (dict, pd.DataFrame):
        eval_loss = 0.
        eval_steps, eval_examples = 0, 0
        eval_ids, eval_tokens, eval_predictions, eval_labels = [], [], [], []
        self.model.eval()
        for batch in tqdm(data):
            batch_ids, batch_tokens, batch_masks, batch_tags = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                outputs = self.model(
                    batch_tokens,
                    attention_mask=batch_masks,
                    labels=batch_tags
                )
            logits = outputs[1].detach().cpu().numpy()
            label_ids = batch_tags.to('cpu').numpy()
            toks = batch_tokens.to('cpu').numpy()
            sentence_ids = batch_ids.to('cpu').numpy()

            eval_loss += outputs[0].mean().item()
            toks = [self.tokenizer.convert_ids_to_tokens(sentence) for sentence in toks]
            eval_tokens.extend(toks)
            eval_predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            eval_labels.extend(label_ids)
            eval_ids.extend(sentence_ids)

            eval_examples += batch_tokens.size(0)
            eval_steps += 1
            break
        eval_loss = eval_loss / eval_steps
        flatten = lambda x: [j for i in x for j in i]

        predicted_tags, valid_tags, tokens, sentence_ids = self.translate(eval_predictions, eval_labels, eval_tokens, eval_ids, tag2code, code2tag, all_ids)

        # for st, sp, sv, vi in zip(tokens, predicted_tags, valid_tags, sentence_ids):
        #     for t, p, v, i in zip(st, sp, sv, vi):
        #         logger.info(f"row = {t}, {p}, {v}, {i}")

        predicted_data = pd.DataFrame(data={
            'sentence_ids': flatten(sentence_ids),
            'tokens': flatten(tokens),
            'predicted_tags': flatten(predicted_tags),
            'valid_tags': flatten(valid_tags),
        })

        scores = {
            "loss": eval_loss,
            "acc": accuracy_score(valid_tags, predicted_tags),
            "f1": f1_score(valid_tags, predicted_tags),
            "p": precision_score(valid_tags, predicted_tags),
            "r": recall_score(valid_tags, predicted_tags),
            "report": classification_report(valid_tags, predicted_tags),
        }

        return scores, predicted_data


if __name__ == '__main__':
    # model_path = f'./data/models/bert-base-multilingual-cased-other'
    model_path = './data/runs/run_2021-02-17T11:42:19_slo-models/models/sloberta-1.0-bsnlp-2021-5-epochs'
    tag2code, code2tag = LoadBSNLP(lang='sl', year='2021').encoding()
    loader = LoadBSNLP(lang="sl", year='2021', data_set='asia_bibi', merge_misc=False)
    predictor = ExtractPredictions(model_path)
    data, ids = predictor.convert_input(loader.test(), tag2code)
    scores, pred_data = predictor.test(data, ids, tag2code, code2tag)
    logger.info(f'{json.dumps(scores, indent=4)}')
    logger.info(f'\n{scores["report"]}')
    logger.info(f'\n{pred_data}')

    # predictor = MakePrediction(model_path=model_path, use_device=-1)

    # res = predictor.get_ners(
    #     "Irena Grmek Košnik iz kranjske območne enote Nacionalnega inštituta za javno zdravje (NIJZ) je povedala, da so bili izvidi torkovega ponovnega testiranja popolnoma drugačni od ponedeljkovega, ko je bilo pozitivnih kar 146 od 1090 testiranih."
    #     "Italijanski predsednik Sergio Mattarella Mariu Draghiju podelil mandat za sestavo vlade"
    # )

    # logger.info(f'{json.dumps(res)}')  # does not pretty-print UTF-8 chars
    # logger.info(f'{res}')
