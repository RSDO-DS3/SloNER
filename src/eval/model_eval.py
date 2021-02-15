import json
import argparse

from collections import defaultdict

from src.eval.predict import MakePrediction
from src.utils.load_documents import LoadBSNLPDocuments
from src.utils.update_documents import UpdateBSNLPDocuments
from src.utils.utils import list_dir


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
                print(f"[WARNING] PARTIAL MATCH: {tag['word']} ({tag['entity']}) -> {token['text']} {token['ner']}")
                token[pred_key] = tags[tag_i]['entity']
                updated += 1
        used += 1 if updated > 0 else 0
        if updated == 0:
            unused_tags.append(tag)

    if len(unused_tags) > 0:
        print(f"Unused tags: {[(tag['word'], tag['entity']) for tag in unused_tags]}")

    return tokens


def main():
    args = parse_args()
    run_path = args.run_path if args.run_path is not None else "./data/models/"
    lang = args.lang

    models, _ = list_dir(run_path)
    print(f"Models to predict: {models}")

    loader = LoadBSNLPDocuments(lang=lang)
    updater = UpdateBSNLPDocuments(lang=lang)
    data = loader.load_merged()

    predictions = {}
    for model in models:
        model_name = model.split('/')[-1]
        model_path = f'{run_path}/{model}'
        print(f"Working on `{model_path}`...")
        predictor = MakePrediction(model_path=model_path)
        predictions[model_name] = {}
        for dataset, langs in data.items():
            predictions[model_name][dataset] = {}
            for lang, docs in langs.items():
                predictions[model_name][dataset][lang] = {}
                for docId, doc in docs.items():
                    tokens = []
                    for sentence in group_sentences(doc['content']).values():
                        tokens.extend(predictor.get_ners(sentence))
                    predictions[model_name][dataset][lang][docId] = tokens
                    doc['content'] = ungroup_sentences(tokens, doc['content'], pred_key=f'{model_name}-NER')
    updater.update_merged(data)
    print(predictions)
    with open(f'{run_path}/all_predictions.json', 'w') as f:
        json.dump(predictions, f)


if __name__ == '__main__':
    main()
