from src.eval.predict import MakePrediction
import sys
import logging
import argparse
import transformers
import pathlib
from datetime import datetime
from itertools import product

import pandas as pd
import torch

from src.utils.load_dataset import LoadBSNLP
from src.model.bert import BertModel


logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s'
)
logger = logging.getLogger('TrainEvalModels')

def read_from_file(path: str) -> str:
    with open(input_text_path) as f:
        return f.read()

def predict(input_text_path: str, model_path: str, device: int=0) -> None:
    text = read_from_file(input_text_path)

    predictor = MakePrediction(model_path, device)
    ners = predictor.get_ners()

    sys.exit(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--train-iterations', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--run-path', type=str, default=None)
    parser.add_argument('--full-finetuning', action='store_true')

    parser.add_argument('--predict', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(f"Training: {args.train}")
    logger.info(f"Train iterations: {args.train_iterations}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Full finetuning: {args.full_finetuning}")
    logger.info(f"Testing: {args.test}")
    logger.info(f"Predict: {args.predict}")
    logger.info(f"Torch version {torch.__version__}")
    logger.info(f"Transformers version {transformers.__version__}")

    if not args.run_path:
        run_time = datetime.now().isoformat()[:-7]  # exclude the ms
        run_path = f'./data/runs/run_{run_time}'
    else:
        run_path = args.run_path
        run_time = run_path.split('/')[-1][4:]

    pathlib.Path(run_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{run_path}/models').mkdir(parents=True, exist_ok=True)
    logger.info(f'Running path: `{run_path}`, run time: `{run_time}`')

    model_names = [
        #"cro-slo-eng-bert",
        "bert-base-multilingual-cased",
        #"bert-base-multilingual-uncased",
        #"sloberta-1.0",
        #"sloberta-2.0",
    ]

    # slo_ssj_train_datasets = {
    #     "ssj500k-bsnlp2017-iterative": {
    #         "ssj500k": LoadSSJ500k(),
    #         "bsnlp-2017": LoadBSNLP(lang='sl', year='2017'),
    #     },
    #     "ssj500k-bsnlp-2017-combined": {
    #         "combined": LoadCombined([LoadSSJ500k(), LoadBSNLP(lang='sl', year='2017')]),
    #     },
    #     "ssj500k-bsnlp-2021-iterative": {
    #         "ssj500k": LoadSSJ500k(),
    #         "bsnlp2021": LoadBSNLP(lang='sl', year='2021'),
    #     },
    #     "ssj500k-bsnlp-2021-combined": {
    #         "combined": LoadCombined([LoadSSJ500k(), LoadBSNLP(lang='sl', year='2021')]),
    #     },
    #     "ssj500k-bsnlp-all-iterative": {
    #         "ssj500k": LoadSSJ500k(),
    #         "bsnlp2017": LoadBSNLP(lang='sl', year='all'),
    #     },
    #     "ssj500k-bsnlp-all-combined": {
    #         "combined": LoadCombined([LoadSSJ500k(), LoadBSNLP(lang='sl', year='all')]),
    #     },
    #     "ssj500k": {
    #         "ssj500k": LoadSSJ500k(),
    #     },
    #     "bsnlp-2017": {
    #         "bsnlp-2017": LoadBSNLP(lang='sl', year='2017'),
    #     },
    #     "bsnlp-2021": {
    #         "bsnlp-2021": LoadBSNLP(lang='sl', year='2021'),
    #     },
    #     "bsnlp-all": {
    #         "bsnlp-all": LoadBSNLP(lang='sl', year='all'),
    #     },
    # }
    # slo_ssj_test_datasets = {
    #     "ssj500k": LoadSSJ500k(),
    #     "bsnlp-2017": LoadBSNLP(lang='sl', year='2017'),
    #     "bsnlp-2021": LoadBSNLP(lang='sl', year='2021'),
    #     "bsnlp-all": LoadBSNLP(lang='sl', year='all')
    # }

    # slo_ssj_train_misc_datasets = {
    #     "bsnlp-2021": {
    #         "bsnlp-2021": LoadBSNLP(lang='sl', year='2021', merge_misc=False, misc_data_only=True),
    #     }
    # }
    # slo_ssj_test_misc_datasets = {
    #     "bsnlp-2021": LoadBSNLP(lang='sl', year='2021', merge_misc=False, misc_data_only=True),
    # }

    # TODO: Fix this
    bsnlp2021 = LoadBSNLP(lang='sl', year='2021', merge_misc=False)
    
    slo_train_datasets = {
        "bsnlp-2021": {
            "bsnlp-2021": bsnlp2021,
        }
    }

    slo_test_datasets = {
        "bsnlp-2021": bsnlp2021,
    }
    tag2code, code2tag = bsnlp2021.encoding()

    # multi_lang_train_datasets = {
    #     'bsnlp-2021-bg': {'bsnlp-2021-bg': LoadBSNLP(lang='bg', year='2021', merge_misc=False)},
    #     'bsnlp-2021-cs': {'bsnlp-2021-cs': LoadBSNLP(lang='cs', year='2021', merge_misc=False)},
    #     'bsnlp-2021-pl': {'bsnlp-2021-pl': LoadBSNLP(lang='pl', year='2021', merge_misc=False)},
    #     'bsnlp-2021-ru': {'bsnlp-2021-ru': LoadBSNLP(lang='ru', year='2021', merge_misc=False)},
    #     'bsnlp-2021-sl': {'bsnlp-2021-sl': LoadBSNLP(lang='sl', year='2021', merge_misc=False)},
    #     'bsnlp-2021-uk': {'bsnlp-2021-uk': LoadBSNLP(lang='uk', year='2021', merge_misc=False)},
    #     'bsnlp-2021-all': {'bsnlp-2021-all': LoadBSNLP(lang='all', year='2021', merge_misc=False)},
    # }
    # multi_lang_test_datasets = {
    #     "bsnlp-2021-bg": LoadBSNLP(lang='bg', year='2021', merge_misc=False),
    #     "bsnlp-2021-cs": LoadBSNLP(lang='cs', year='2021', merge_misc=False),
    #     "bsnlp-2021-pl": LoadBSNLP(lang='pl', year='2021', merge_misc=False),
    #     "bsnlp-2021-ru": LoadBSNLP(lang='ru', year='2021', merge_misc=False),
    #     "bsnlp-2021-sl": LoadBSNLP(lang='sl', year='2021', merge_misc=False),
    #     "bsnlp-2021-uk": LoadBSNLP(lang='uk', year='2021', merge_misc=False),
    #     "bsnlp-2021-all": LoadBSNLP(lang='all', year='2021', merge_misc=False),
    # }

    if args.predict:
        predict(args.predict)

    
    test_f1_scores = []
    for model_name, fine_tuning in product(model_names, [True, False]):
        logger.info(f"Working on model: `{model_name}`...")
        for train_bundle, loaders in slo_train_datasets.items():
            bert = BertModel(
                tag2code=tag2code,
                code2tag=code2tag,
                epochs=args.epochs,
                input_model_path=f'./data/models/{model_name}',
                output_model_path=f'./data/runs/{run_path}',
                output_model_fname=f'{model_name}-{train_bundle}'
                                   f"{'-finetuned' if fine_tuning else ''}"
                                   f'-{args.epochs}-epochs',
                tune_entire_model=fine_tuning,
            )

            if args.train:
                logger.info(f"Training data bundle: `{train_bundle}`")
                bert.train(loaders)

            if args.test:
                for test_dataset, dataloader in slo_test_datasets.items():
                    logger.info(f"Testing on `{test_dataset}`")
                    p, r, f1 = bert.test(test_data=dataloader.test())
                    test_f1_scores.append({
                        "model_name": model_name,
                        "fine_tuned": fine_tuning,
                        "train_bundle": train_bundle,
                        "epochs": args.epochs,
                        "test_dataset": test_dataset,
                        "precision_score": p,
                        "recall_score": r,
                        "f1_score": f1
                    })
                    logger.info(f"[{train_bundle}][{test_dataset}] P = {p:.4f}, R = {r:.4f}, F1 = {f1:.4f}")
    if args.test:
        scores = pd.DataFrame(test_f1_scores)
        scores.to_csv(f'{run_path}/training_scores-{run_time}.csv', index=False)
    logger.info(f'Entire training suite is done.')


if __name__ == '__main__':
    main()
