
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


################ LOAD PRESAVED MODEL

def load():

    # load json to create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    return loaded_model


################# LOAD DATA TO PREDICT

## EXAMPLES
text_1 = "@iloveyou"
text_0 = "I love You"
text_2 = "@CNN @W_McPrivilege Hahahahahahahahahahahahahahahahahahahahaha"
text_3 = "@CNN please stop framing in this in a way that pits strong women (both highly admirable) against each other.. clinton and obama’s publicly support each other, what point are u trying to make here??"



def predict(text,model_load):

    print("TEXT TO PREDICT: ", text )
    print("MODEL TO USe: ", model_load )

    ##############
    text = clean_text_en.clean_text_full(text)
    print("FASE 1")
    if text  == "EMPTY STRING":
       print("FASE EMPTY")
       return "Empty String"
    else:
        text = nlp_functions.steam_text(text)
        print("FASE 2")
        #print("Steamer -> ",text_1 )

        tokenizer_test = Tokenizer(num_words= vocabulary_size)
        tokenizer_test.fit_on_texts(text)
        #print("Tokenizer -> ", tokenizer_test)
        print("FASE 3")

        sequences_test = tokenizer_test.texts_to_sequences(text)
        #print("Sequence -> ", sequences_test)
        print("FASE 4")

        data_test = pad_sequences(sequences_test, maxlen=200)
        print("FASE 5")
        

        ################ EVALAUTE

        #model_load.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print("FASE 6")
        #print(data_test)
        score = model_load.predict_classes(data_test)
        print("FASE 7")
        #print(score)

        if score[0] == 0:
            result = "Hate Speech"
        elif score[0] == 1:
            result = "Abusive Language"
        else:
            result = "Normal Language"

        return result
        #print("Prediction: ", result )
        #print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
