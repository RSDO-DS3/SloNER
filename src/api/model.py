from flask import current_app

from src.eval.predict import MakePrediction

predictor = None

def get_predictor() -> MakePrediction:
    global predictor
    if not predictor:
        model_path = current_app.config['NER_MODEL']
        current_app.logger.info(f'Loading NER model from path "{model_path}"')
        predictor = MakePrediction(model_path=model_path)

    return predictor
