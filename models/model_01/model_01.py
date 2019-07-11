from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

import clean_text_en
import nlp_functions

NAME = "Model Inicial Keras Tetse"
ORG = "model_01"
DATA_DIRECTORY = "models/model_01/data/"

loaded_model = None
VOCABULARY_SIZE = 1000

def load():
    global loaded_model

    print('[INFO] Loading Model (%s) : %s' % (ORG, NAME))
    
    json_file = open(DATA_DIRECTORY+'model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(DATA_DIRECTORY+"model.h5")

    print(loaded_model.summary())
    
    print('[INFO] Model %s loaded' % (ORG))
    


def predict(user_query,lang):
    if lang == "en":
        user_query = clean_text_en.clean_text_full(user_query)
        if user_query  == "EMPTY STRING":
            return  'message : Empty Parameters'
        else:
            user_query = nlp_functions.steam_text(user_query)
            
            ##########3 LOAD PREVIOUS
            tokenizer_test = Tokenizer(num_words= VOCABULARY_SIZE)
            tokenizer_test.fit_on_texts(user_query)
            ##########3 LOAD PREVIOUS
            
            sequences_test = tokenizer_test.texts_to_sequences(user_query)

            data_test = pad_sequences(sequences_test, maxlen=200)
            

            ################ EVALAUTE

            #loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            #print(data_test)
            score = loaded_model.predict_classes(data_test)
            score2 = loaded_model.predict(data_test)
            #print(score)
            print(score)
            print(score2)
            print("----------------")
            if score[0] == 0:
                result = "Hate Speech"
            elif score[0] == 1:
                result = "Abusive Language"
            else:
                result = "Normal Language"

            # CREATE JSON TO OUTPUT
            return result
            
    else:
        return 'message : Language not supported'
