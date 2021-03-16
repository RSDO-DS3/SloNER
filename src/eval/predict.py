import logging
import sys
from typing import List, Dict
from collections import defaultdict
from operator import itemgetter
import re

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline
)


class MakePrediction:
    def __init__(
        self,
        model_path: str = f'./data/models/bert-base-multilingual-cased-other',
        use_device: int = None
    ):
        """
            A class to extract all the NE predictions from a given text
        :param model_path: path to a HuggingFace-transformers pre-trained model for the NER task, such as BERT Base Multilingual (Un)Cased
        """

        if use_device is None:
            use_device = torch.cuda.device_count() - 1

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
            ignore_labels=[]
        )

    @staticmethod
    def __merge_tokenized_nes(raw_nes: List[Dict]) -> List[Dict]:
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


    @staticmethod
    def __merge_nes(nes: List[Dict]) -> List[Dict]:
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


    @staticmethod
    def _add_other(text: str, nes: List[Dict]) -> List[Dict]:
        """
            Add entity O (other) for all untagged words in a given text.
        :param text: The input text from which the NEs have been extracted
        :param nes: A list of named entities, originating from a given text
        :return: A list of named entities, where no word or punctuation remains unlabeled
        """
        j = 0
        out_nes = []
        for word in re.split(r'(\W+)', text):
            word = word.strip()
            if not word:
                continue
            if j < len(nes) and nes[j]['word'] == word:
                out_nes.append({'word': word, 'entity': nes[j]['entity'], 'score': nes[j]['score']})
                j += 1
            else:
                out_nes.append({'word': word, 'entity': 'O', 'score': 0.0}) # score is unknown
        assert j == len(nes)

        return out_nes


    def get_ners(self, text: str, merge_nes: bool = False, add_other: bool = False) -> List[Dict]:
        """
            Extract the NEs from a given text.
        :param text: The input text from which the NEs are extracted
        :param merge_nes: Indicator to merge the entities together, i.e. B-XXX and I-XXX into XXX
        :param add_other: Add named entity O (other) for all unclassified words.
        :return: A list of words, annotated with recognised named entities
        """
        raw_ners = self.ner_pipeline(text)
        nes = self.__merge_tokenized_nes(raw_ners)

        if add_other:
            nes = self._add_other(text, nes)

        # Extract only the relevant fields
        nes = [{'entity': ne['entity'], 'score': ne['score'], 'word': ne['word']} for ne in nes]

        if merge_nes:
            return self.__merge_nes(nes)
        return nes


def predict_from_string(model_path: str, sentences: List[str]) -> None:
    predictor = MakePrediction(model_path=model_path)

    for sentence in sentences:
        res = predictor.get_ners(sentence)
        logger.info(f'{res}')

    #logger.info(f'{json.dumps(res)}')  # does not pretty-print UTF-8 chars


if __name__ == '__main__':
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('MakePrediction')

    model_path = os.environ['NER_MODEL_PATH']
    
    text = ["Irena Grmek Košnik iz kranjske območne enote Nacionalnega inštituta za javno zdravje (NIJZ) je povedala, da so bili izvidi torkovega ponovnega testiranja popolnoma drugačni od ponedeljkovega, ko je bilo pozitivnih kar 146 od 1090 testiranih.", 
    "Italijanski predsednik Sergio Mattarella Mariu Draghiju podelil mandat za sestavo vlade."]

    predict_from_string(model_path, text)
