import json
import argparse
import tqdm
import logging
import sys
import multiprocessing
import pandas as pd

from torch.cuda import device_count
from collections import defaultdict

from src.eval.predict import MakePrediction
from src.utils.load_documents import LoadBSNLPDocuments
from src.utils.update_documents import UpdateBSNLPDocuments
from src.utils.utils import list_dir


logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s'
)
logger = logging.getLogger('TrainEvalModels')

DEBUG = False
warnings = []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, default='all')
    parser.add_argument('--run-path', type=str, default=None)
    return parser.parse_args()


def group_sentences(document: list) -> dict:
    sentences = defaultdict(lambda: "")
    for token in document:
        sentences[token['sentenceId']] = f"{sentences[token['sentenceId']]} {token['text']}"
    return dict(sentences)


def ungroup_sentences(
    tags: list,
    tokens: list,
    pred_key: str = 'calcNer',
) -> list:
    for token in tokens:
        token[pred_key] = 'O'
    used = 0
    tags = sorted(tags, key=lambda x: len(x['word']), )
    unused_tags = []

    for tag in tags:
        updated = 0
        for token in tokens:
            if f"{token['text']}" == f"{tag['word']}":
                token[pred_key] = tag['entity']
                updated += 1
            elif updated == 0 and f"{token['text']}".startswith(f"{tag['word']}"):
                warn = f"[WARNING] PARTIAL MATCH: {tag['word']} ({tag['entity']}) -> {token['text']} {token['ner']}"
                warnings.append(warn)
                if DEBUG: logger.info(warn)
                token[pred_key] = tag['entity']
                updated += 1
        used += 1 if updated > 0 else 0
        if updated == 0:
            unused_tags.append(tag)

    if len(unused_tags) > 0:
        warn = f"Unused tags: {[(tag['word'], tag['entity']) for tag in unused_tags]}"
        warnings.append(warn)
        if DEBUG: logger.info(warn)
    return tokens


def merge_misc_tokens(
    sentence_tokens: list,
    misc_tokens: list,
    classified_tokens: list,
) -> list:

    return sentence_tokens


def looper(
    run_path: str,
    clang: str,
    model: str,
) -> dict:
    loader = LoadBSNLPDocuments(lang=clang)

    model_name = model.split('/')[-1]
    model_path = f'{run_path}/models/{model}'

    predictor = MakePrediction(model_path=model_path, use_device=0 if device_count() > 0 else -1)
    pred_misc = None  # if clang != 'sl' else MakePrediction(model_path='TODO', use_device=1 if device_count() > 1 else -1)
    logger.info(f"Predicting for {model_name}")

    updater = UpdateBSNLPDocuments(lang=clang, path=f'{run_path}/predictions/bsnlp/{model_name}')
    predictions = {}
    data = loader.load_merged()
    tdset = tqdm.tqdm(data.items(), desc="Dataset")
    for dataset, langs in tdset:
        tdset.set_description(f'Dataset: {dataset}')
        tlang = tqdm.tqdm(langs.items(), desc="Language")
        predictions[dataset] = {}
        for lang, docs in tlang:
            predictions[dataset][lang] = {}
            tlang.set_description(f'Lang: {tlang}')
            for docId, doc in tqdm.tqdm(docs.items(), desc="Docs"):
                tokens = []
                for sentence in group_sentences(doc['content']).values():
                    sentence_tokens = predictor.get_ners(sentence)
                    if pred_misc is not None:
                        misc = list(filter(lambda x: x in ['B-MISC', 'I-MISC'], tokens))
                        if len(misc) > 0:
                            misc_tokens = pred_misc.get_ners(sentence)
                            sentence_tokens = merge_misc_tokens(sentence_tokens, misc, misc_tokens)
                    tokens.extend(sentence_tokens)

                doc['content'] = ungroup_sentences(tokens, doc['content'])  # , pred_key=f'{model_name}-NER')
                predictions[dataset][lang][docId] = tokens
    updater.update_merged(data)
    logger.info(f"Done predicting for {model_name}")
    return {
        'model': model_name,
        'preds': predictions
    }


def main():
    args = parse_args()
    run_path = args.run_path if args.run_path is not None else "./data/models/"
    lang = args.lang

    print(f"Run path: {run_path}")
    print(f"Langs: {lang}")

    models, _ = list_dir(f'{run_path}/models')
    logger.info(f"Models to predict: {models}")
    pool = multiprocessing.Pool(8)

    # tmodel = tqdm.tqdm(list(map(lambda x: (run_path, lang, x), models)), desc="Model")
    # predictions = pool.map(looper, tmodel)
    # predictions = list(map(looper, tmodel))
    predictions = []
    for model in tqdm.tqdm(models, desc="Model"):
        predictions.append(looper(run_path, lang, model))
    logger.info(predictions)
    
    with open(f'{run_path}/all_predictions.json', 'w') as f:
        json.dump(predictions, f)
    
    logger.info("Warnings that occcured:")
    logger.info(f"{warnings}")
    with open(f'{run_path}/pred_warnings.json', 'w') as f:
        json.dump(warnings, f, indent=4)
    logger.info("Done.")


if __name__ == '__main__':
    main()
