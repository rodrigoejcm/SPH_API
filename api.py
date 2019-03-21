from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import load_model

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

import numpy

import sys 
import os
sys.path.append(os.path.abspath("include"))
import clean_text_en
import nlp_functions


vocabulary_size = 1000


### LOAD APP

application = Flask(__name__)
api = Api(application)

### LOAD MODEL

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


### LOAD PARSER

parser = reqparse.RequestParser()
parser.add_argument('query')


class PredictHateSpeech(Resource):
    def get(self):

        # GET PARAMS
        args = parser.parse_args()
        user_query = args['query']

        # PREICT 


        print("TEXT TO PREDICT: ", user_query )
        print("MODEL TO USe: ", loaded_model )

        ##############
        user_query = clean_text_en.clean_text_full(user_query)
        print("FASE 1")
        if user_query  == "EMPTY STRING":
            print("FASE EMPTY")
            return "Empty String"
        else:
            user_query = nlp_functions.steam_text(user_query)
            print("FASE 2")
            #print("Steamer -> ",text_1 )

            tokenizer_test = Tokenizer(num_words= vocabulary_size)
            tokenizer_test.fit_on_texts(user_query)
            #print("Tokenizer -> ", tokenizer_test)
            print("FASE 3")

            sequences_test = tokenizer_test.texts_to_sequences(user_query)
            #print("Sequence -> ", sequences_test)
            print("FASE 4")

            data_test = pad_sequences(sequences_test, maxlen=200)
            print("FASE 5")
            

            ################ EVALAUTE

            loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            print("FASE 6")
            #print(data_test)
            score = loaded_model.predict_classes(data_test)
            print("FASE 7")
            #print(score)

            if score[0] == 0:
                result = "Hate Speech"
            elif score[0] == 1:
                result = "Abusive Language"
            else:
                result = "Normal Language"

            # CREATE JSON TO OUTPUT
            output = {'prediction': result}

            return output


api.add_resource(PredictHateSpeech, '/')


if __name__ == '__main__':
    application.run(debug=True)
  