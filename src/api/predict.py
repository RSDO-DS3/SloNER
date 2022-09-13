from flask import (
    Blueprint, request, jsonify, current_app, abort
)

from flasgger import swag_from

from src.api.model import get_predictor

bp = Blueprint('predict', __name__, url_prefix='/predict')


@bp.route('/ner', methods=['POST'])
@swag_from('docs/predict.yml')
def predict_ner():
    """For a given text (usually one sentence), this endpoint returns annotated text with named entities, word by word.
    """

    text = request.get_data(as_text=True)

    if len(text) > 10000: # Limit to 10k characters
        return abort(413)

    current_app.logger.info(f'text: "{text}"')

    res = get_predictor().get_ners(text)
    current_app.logger.info(f'prediction: "{res}"')

    return jsonify(named_entities=res)
