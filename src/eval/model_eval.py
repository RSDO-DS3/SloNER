import json

from src.eval.predict import MakePrediction
from src.utils.load_documents import LoadBSNLPDocuments


def main():
    loader = LoadBSNLPDocuments(lang='sl')
    predictor = MakePrediction()
    data = loader.load_raw()
    predictions = {}
    truncate_length = 1400
    for dataset, langs in data.items():
        predictions[dataset] = {}
        for lang, docs in langs.items():
            predictions[dataset][lang] = {}
            for docId, doc in docs.items():
                content = doc['content']
                tokens = []
                for i in range(0, len(content), truncate_length):
                    iter_toks = predictor.get_ners(content[i:i + min(truncate_length, len(content) - i)], merge_nes=True)
                    tokens.extend(iter_toks)
                predictions[dataset][lang][docId] = tokens
    print(json.dumps(predictions, indent=4))


if __name__ == '__main__':
    main()
