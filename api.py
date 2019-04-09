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
parser.add_argument('content')
parser.add_argument('news_title')
parser.add_argument('headline')
parser.add_argument('model')
parser.add_argument('lang')


class PredictHateSpeech(Resource):
    def get(self):

        # GET PARAMS
        args = parser.parse_args()
        
        ## General
        user_query = args['content']
        user_model = args['model']
        user_lang = args['lang']

        print('[INFO REQUEST] Received Parameters')
        print('[INFO REQUEST - content ] ', user_query)
        print('[INFO REQUEST - Model ] ', user_model)
        print('[INFO REQUEST - Lang ] ', user_lang)

        application.logger.info('[LOG] Query: ', user_query)

        # PREDICT 

        result_prediction = predict(user_query,user_lang,user_model)

        return result_prediction


api.add_resource(PredictHateSpeech, '/comments_api')


class PredictNewsHateSpeech(Resource):
    def get(self):

        # GET PARAMS
        args = parser.parse_args()
        
        ## Specifc News
        news_headline = args['headline']
        news_title = args['news_title']
        
        ## General
        user_query = args['content']
        user_model = args['model']
        user_lang = args['lang']

        print('[INFO REQUEST] Received Parameters')
        print('[INFO REQUEST - content ] ', user_query)
        print('[INFO REQUEST - headline ] ', news_headline)
        print('[INFO REQUEST - news_title ] ', news_title)
        print('[INFO REQUEST - Model ] ', user_model)
        print('[INFO REQUEST - Lang ] ', user_lang)

        application.logger.info('[LOG] Query: ', user_query)

        # PREDICT 

        result_prediction = predict(user_query,user_lang,user_model)

        return result_prediction


api.add_resource(PredictNewsHateSpeech, '/news_api')



if __name__ == '__main__':
    application.run(debug=True)
  