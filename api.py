import sys 
import os
sys.path.append(os.path.abspath("models"))


from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
from flask_cors import CORS

from models import load_models, predict, predict_news_hatespeech

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


parser.add_argument('title_news')
parser.add_argument('lead_news')
parser.add_argument('text_news')
parser.add_argument('authors_news')
parser.add_argument('source_news')
parser.add_argument('weekday_news')
parser.add_argument('period_day_news')



class PredictHateSpeech(Resource):
    ### OLD MODELS BY PAULA
    def get(self):

        # GET PARAMS
        args = parser.parse_args()
        
        ## General
        user_query = args['content']
        user_model = args['model']
        user_lang = args['lang']


        if (args['content']):

            print('[INFO REQUEST] Received Parameters')
            print('[INFO REQUEST - content ] ', user_query)
            print('[INFO REQUEST - Model ] ', user_model)
            print('[INFO REQUEST - Lang ] ', user_lang)

            application.logger.info('[LOG] Query: ', user_query)

            # PREDICT 

            result_prediction = predict(user_query,user_lang,user_model)

            print(result_prediction)

            return result_prediction
        else:
            return  "message : Missing Parameters"



api.add_resource(PredictHateSpeech, '/comments')


class PredictNewsHateSpeech(Resource):
    def get(self):

        # GET PARAMS
        args = parser.parse_args()
        
        ## Specifc News
        title_news = args['title_news']
        lead_news = args['lead_news']
        text_news = args['text_news']
        authors_news  = args['authors_news']
        source_news  = args['source_news']
        week_news = args['weekday_news']
        period_news = args['period_day_news']

        if not source_news:
            source_news = 14
        
        if not week_news:
            #unknown
            week_news = 2

        if not period_news:
            #unknown
            period_news = 4

        
  
        print('[INFO REQUEST] Received Parameters')
        print('[INFO REQUEST - title ] ', title_news)
        print('[---]')
        print('[INFO REQUEST - lead_news ] ', lead_news)
        print('[---]')
        print('[INFO REQUEST - text_news ] ', text_news)
        #print('[INFO REQUEST - Model ] ', user_model)
        #print('[INFO REQUEST - Lang ] ', user_lang)

        #application.logger.info('[LOG] Query: ', user_query)

        # PREDICT 
        if (title_news and lead_news and text_news ):

            if (title_news is not "") & (lead_news is not "") & (text_news is not ""):
                result_prediction = predict_news_hatespeech(title_news , lead_news , text_news , authors_news , source_news , week_news , period_news )
                print(result_prediction)
                return result_prediction
            else:
                return  "message : Empty Parameters"  

        else:
            return  "message : Missing Parameters"


api.add_resource(PredictNewsHateSpeech, '/news')



if __name__ == '__main__':
    application.run(debug=True)
  