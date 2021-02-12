import json

from collections import defaultdict

from src.eval.predict import MakePrediction
from src.utils.load_documents import LoadBSNLPDocuments
from src.utils.update_documents import UpdateBSNLPDocuments


def group_sentences(document: list) -> dict:
    sentences = defaultdict(lambda: "")
    for token in document:
        sentences[token['sentenceId']] = f"{sentences[token['sentenceId']]} {token['text']}"
    return dict(sentences)


def ungroup_sentences(tags: list, tokens: list) -> list:
    tag_i = 0
    if tokens[0]['docId'] == 325:
        print("Here we go")
    for token in tokens:
        token['calcNER'] = 'O'
    used = 0
    tags = sorted(tags, key=lambda x: len(x['word']), )
    unused_tags = []

    for tag in tags:
        updated = 0
        for token in tokens:
            if f"{token['text']}" == f"{tag['word']}":
                token['calcNER'] = tags[tag_i]['entity']
                updated += 1
            elif updated == 0 and f"{token['text']}".startswith(f"{tag['word']}"):
                print(f"[WARNING] PARTIAL MATCH: {tag['word']} ({tag['entity']}) -> {token['text']} {token['ner']}")
                token['calcNER'] = tags[tag_i]['entity']
                updated += 1
        used += 1 if updated > 0 else 0
        if updated == 0:
            unused_tags.append(tag)

    if len(unused_tags) > 0:
        print(f"Unused tags: {[(tag['word'], tag['entity']) for tag in unused_tags]}")

    return tokens


def main():
    loader = LoadBSNLPDocuments(lang='sl')
    updater = UpdateBSNLPDocuments(lang='sl')
    predictor = MakePrediction()
    data = loader.load_merged()
    predictions = {}
    for dataset, langs in data.items():
        predictions[dataset] = {}
        for lang, docs in langs.items():
            predictions[dataset][lang] = {}
            for docId, doc in docs.items():
                tokens = []
                for sentence in group_sentences(doc['content']).values():
                    tokens.extend(predictor.get_ners(sentence))
                predictions[dataset][lang][docId] = tokens
                doc['content'] = ungroup_sentences(tokens, doc['content'])
    updater.update_merged(data)


if __name__ == '__main__':
    main()
