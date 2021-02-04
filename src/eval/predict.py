import json

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

from src.utils.load_dataset import LoadBSNLP


class MakePrediction:
    def __init__(
        self,
        model_path: str = f'./data/models/bert-base-multilingual-cased-other',
    ):
        """
            A class to extract all the NE predictions from a given document
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
            while j < len(raw_nes) and raw_nes[j]['word'].startswith('##'):
                if raw_nes[j]['index'] != (raw_nes[j - 1]['index'] + 1):
                    raise Exception("Tokens are not coming one after the other")
                if raw_nes[j]['entity'] != raw_nes[j - 1]['entity']:
                    raise Exception("Tokens not the same type")
                ne['word'] += f'{raw_nes[j]["word"][2:]}'
                ne['score'] = (ne['score'] + raw_nes[j]['score']) / 2
                ne['modelTokens'] += 1
                j += 1
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
            Get the NEs from a particular document [stored in data], provided as a whole string.
        :param data: The input data from which the NEs are extracted
        :param merge_nes: Indicator to merge the entities together, i.e. B-XXX and I-XXX into XXX
        :return:
        """
        raw_ners = self.ner_pipeline(data)
        nes = self.__merge_tokenized_nes(raw_ners)
        if merge_nes:
            return self.__merge_nes(nes)
        return nes


if __name__ == '__main__':
    tag2code, code2tag = LoadBSNLP("sl").encoding()
    model_path = f'./data/models/bert-base-multilingual-cased-other'
    predictor = MakePrediction(model_path=model_path)

    res = predictor.get_ners(
        "Irena Grmek Košnik iz kranjske območne enote Nacionalnega inštituta za javno zdravje (NIJZ) je povedala, da so bili izvidi torkovega ponovnega testiranja popolnoma drugačni od ponedeljkovega, ko je bilo pozitivnih kar 146 od 1090 testiranih."
        "Italijanski predsednik Sergio Mattarella Mariu Draghiju podelil mandat za sestavo vlade"
    )
    print(json.dumps(res))  # does not pretty-print UTF-8 chars
    print(res)
