import sys 
import os
sys.path.append(os.path.abspath("models"))


from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
from flask_cors import CORS

from models import load_models, predict

### LOAD APP

application = Flask(__name__)
CORS(application)
api = Api(application)

### LOAD MODEL

load_models()


### LOAD PARSER

parser = reqparse.RequestParser()
parser.add_argument('query')
parser.add_argument('model')
parser.add_argument('lang')


class PredictHateSpeech(Resource):
    def get(self):

        # GET PARAMS
        args = parser.parse_args()
        user_query = args['query']
        user_model = args['model']
        user_lang = args['lang']

        application.logger.info('[LOG] Query: ', user_query)

        # PREDICT 

        result_prediction = predict(user_query,user_lang,user_model)

        return result_prediction


api.add_resource(PredictHateSpeech, '/')


if __name__ == '__main__':
    application.run(debug=True)
  