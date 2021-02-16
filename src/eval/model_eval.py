import json
import argparse
import tqdm
import logging
import sys

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
    tag_i = 0
    for token in tokens:
        token[pred_key] = 'O'
    used = 0
    tags = sorted(tags, key=lambda x: len(x['word']), )
    unused_tags = []

    for tag in tags:
        updated = 0
        for token in tokens:
            if f"{token['text']}" == f"{tag['word']}":
                token[pred_key] = tags[tag_i]['entity']
                updated += 1
            elif updated == 0 and f"{token['text']}".startswith(f"{tag['word']}"):
                warn = f"[WARNING] PARTIAL MATCH: {tag['word']} ({tag['entity']}) -> {token['text']} {token['ner']}"
                warnings.append(warn)
                if DEBUG: logger.info(warn)
                token[pred_key] = tags[tag_i]['entity']
                updated += 1
        used += 1 if updated > 0 else 0
        if updated == 0:
            unused_tags.append(tag)

    if len(unused_tags) > 0:
        warn = f"Unused tags: {[(tag['word'], tag['entity']) for tag in unused_tags]}"
        warnings.append(warn)
        if DEBUG: logger.info(warn)
    return tokens


def main():
    args = parse_args()
    run_path = args.run_path if args.run_path is not None else "./data/models/"
    lang = args.lang

    models, _ = list_dir(run_path)
    logger.info(f"Models to predict: {models}")

    loader = LoadBSNLPDocuments(lang=lang)
    updater = UpdateBSNLPDocuments(lang=lang)
    data = loader.load_merged()

    predictions = {}
    tmodel = tqdm.tqdm(models, desc="Model")
    for model in tmodel:
        model_name = model.split('/')[-1]
        model_path = f'{run_path}/{model}'
        tmodel.set_description(f'Model: {model_name}')
        predictor = MakePrediction(model_path=model_path)
        predictions[model_name] = {}
        tdset = tqdm.tqdm(data.items(), desc="Dataset")
        for dataset, langs in tdset:
            tdset.set_description(f'Dataset: {dataset}')
            predictions[model_name][dataset] = {}
            tlang = tqdm.tqdm(langs.items(), desc="Language")
            for lang, docs in tlang:
                tlang.set_description(f'Lang: {tlang}')
                predictions[model_name][dataset][lang] = {}
                for docId, doc in tqdm.tqdm(docs.items(), desc="Docs"):
                    tokens = []
                    for sentence in tqdm.tqdm(group_sentences(doc['content']).values(), f"Sentences {docId}"):
                        tokens.extend(predictor.get_ners(sentence))
                    predictions[model_name][dataset][lang][docId] = tokens
                    doc['content'] = ungroup_sentences(tokens, doc['content'], pred_key=f'{model_name}-NER')
    updater.update_merged(data)
    logger.info(predictions)
    with open(f'{run_path}/all_predictions.json', 'w') as f:
        json.dump(predictions, f)
    logger.info("Warnings that occcured:")
    logger.info(f"{warnings}")
    logger.info("Done.")


if __name__ == '__main__':
    main()
