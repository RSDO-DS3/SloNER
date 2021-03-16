from flask import (
    Blueprint, request, jsonify, current_app, abort
)

from flasgger import swag_from

from src.api.model import get_predictor

bp = Blueprint('predict', __name__, url_prefix='/predict')


@bp.route('/ner', methods=('PUT', 'POST', 'PATCH'))
@swag_from('docs/predict.yml')
def predict_ner():
    """For a given text (usually one sentence), this endpoint returns annotated text with named entities, word by word.
    """

    if request.content_length > 1<<20:  # Limit to 1 MiB
        return abort(413) #, 'Content length exceeds 1 MB limit'

    text = request.get_data(as_text=True)
    current_app.logger.info(f'text: "{text}"')

    res = get_predictor().get_ners(text)
    current_app.logger.info(f'prediction: "{res}"')

    return jsonify(named_entities=res)
