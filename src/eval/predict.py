import json
import torch

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from transformers.configuration_utils import PretrainedConfig

from src.utils.load_dataset import LoadBSNLP


if __name__ == '__main__':
    tag2code, code2tag = LoadBSNLP("sl").encoding()
    model_path = f'./data/models/bert-base-multilingual-cased-other'
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        # num_labels=len(tag2code),
        # id2label=code2tag,
        # label2id=tag2code,
        output_attentions=False,
        output_hidden_states=False
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        from_pt=True,
        do_lower_case=False,
        use_fast=False
    )
    ner = pipeline(
        'ner',
        model=model,
        tokenizer=tokenizer,
        framework="pt",
    )

    res = ner(
        "Irena Grmek Košnik iz kranjske območne enote Nacionalnega inštituta za javno zdravje (NIJZ) je povedala, da so bili izvidi torkovega ponovnega testiranja popolnoma drugačni od ponedeljkovega, ko je bilo pozitivnih kar 146 od 1090 testiranih."
        "Italijanski predsednik Sergio Mattarella Mariu Draghiju podelil mandat za sestavo vlade"
    )
    print(json.dumps(res, indent=4))
