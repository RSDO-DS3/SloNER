import json
import argparse
import tqdm
import logging
import sys
import multiprocessing
import pandas as pd
import random

from torch.cuda import device_count
from collections import defaultdict

from src.eval.predict import MakePrediction, ExtractPredictions
from src.utils.load_documents import LoadBSNLPDocuments
from src.utils.load_dataset import LoadBSNLP
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


def looper(
    run_path: str,
    clang: str,
    model: str,
    categorize_misc: bool = True,
) -> dict:
    loader = LoadBSNLPDocuments(lang=clang, year='2021')
    tag2code, code2tag = LoadBSNLP(lang=clang, year='2021', merge_misc=True).encoding()
    misctag2code, misccode2tag = LoadBSNLP(lang='sl', year='2021', merge_misc=False, misc_data_only=True).encoding()

    model_name = model.split('/')[-1]
    model_path = f'{run_path}/models/{model}'
    misc_model, _ = list_dir(f'{run_path}/misc_models')

    predictor = ExtractPredictions(model_path=model_path)
    pred_misc = None if not categorize_misc else ExtractPredictions(model_path=f'./{run_path}/misc_models/{misc_model[0]}')
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
                to_pred = pd.DataFrame(doc['content'])
                if categorize_misc:
                    # randomly choose a category for (B|I)-MISC category
                    # cat = random.choice(['PRO', 'EVT'])
                    # to_pred.loc[(to_pred['ner'] == 'B-MISC'), 'ner'] = f'B-{cat}'
                    # to_pred.loc[(to_pred['ner'] == 'I-MISC'), 'ner'] = f'I-{cat}'
                    # categorize the PRO and EVT to MISC, as the model only knows about it
                    to_pred.loc[to_pred['ner'].isin(['B-PRO', 'B-EVT']), 'ner'] = f'B-MISC'
                    to_pred.loc[to_pred['ner'].isin(['I-PRO', 'I-EVT']), 'ner'] = f'I-MISC'
                scores, pred_data = predictor.predict(to_pred, tag2code, code2tag)
                logger.info(f'\n{scores["report"]}')
                if pred_misc is not None and len(pred_data.loc[pred_data['ner'].isin(['B-MISC', 'I-MISC'])]) > 0:
                # if pred_misc is not None and len(pred_data.loc[pred_data['ner'].isin(['B-PRO', 'I-PRO', 'B-EVT', 'I-EVT'])]) > 0:
                    misc_data = pd.DataFrame(doc['content'])
                    misc_data.loc[~(misc_data['ner'].isin(['B-PRO', 'B-EVT', 'I-PRO', 'I-EVT'])), 'ner'] = 'O'
                    scores_misc, misc_pred = pred_misc.predict(misc_data, misctag2code, misccode2tag)
                    pred_data['ner'] = pd.DataFrame(doc['content'])['ner']
                    # update the entries
                    pred_data.loc[misc_pred['calcNER'].isin(['B-PRO', 'B-EVT', 'I-PRO', 'I-EVT']), 'calcNER'] = misc_pred.loc[misc_pred['calcNER'].isin(['B-PRO', 'B-EVT', 'I-PRO', 'I-EVT']), 'calcNER']
                doc['content'] = pred_data.to_dict(orient='records')
                predictions[dataset][lang][docId] = pred_data.loc[~(pred_data['calcNER'] == 'O')].to_dict(orient='records')
        break
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
