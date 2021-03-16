import os

from flask import Flask, request
from flasgger import Swagger, LazyString, LazyJSONEncoder

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.json_encoder = LazyJSONEncoder
    
    app.config.from_mapping(
        NER_MODEL=os.environ['NER_MODEL_PATH'], #"./data/runs/run_2021-03-05T12:39:26/bert-base-multilingual-cased-bsnlp-2021-finetuned-5-epochs",
        SWAGGER={
            'openapi': '3.0.2',
            'uiversion': '3',
            'info': {
                'title': 'Named Entity Recognition API',
                'description': 'RESTful Named Entity Recognition API For Slovenian language.',
                'version': '1.0.0',
                'contact': {
                    'responsibleOrganization': 'University of Ljubljana, Faculty of Computer and Information Science'
                },
                'termsOfService': None,
            },
            'basePath': '/api',
            'schemes': [LazyString(lambda: 'https' if request.is_secure else 'http')]
        },
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # Configure Swagger
    swagger_config = {
        "headers": [],
        "specs": [
            {
                "endpoint": 'predict',
                "route": '/predict.json',
                "rule_filter": lambda rule: True,  # all in
                "model_filter": lambda tag: True,  # all in
            }
        ],
        "static_url_path": "/flasgger_static",
        # "static_folder": "static",  # must be set by user
        "swagger_ui": True,
        "specs_route": "/apidocs/"
    }

    # Initialise Swagger
    Swagger(app, config=swagger_config)

    # Initialise a model
    with app.app_context():
        from src.api.model import get_predictor
        get_predictor()

    # Register a route
    import src.api.predict as predict
    app.register_blueprint(predict.bp)

    return app
