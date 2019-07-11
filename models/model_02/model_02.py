#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
#from keras.layers.embeddings import Embedding
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.models import model_from_json
import pickle 

from keras.models import load_model
#from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Activation

import numpy as np


NAME = "Model Paula Hate ( EN & PT )"
ORG = "model_02"
DATA_DIRECTORY = "models/model_02/data/"

loaded_model_en = None
loaded_model_pt = None

VOCABULARY_SIZE = 1000
MAX_WORDS = 1000

def load():
    global loaded_model_en,loaded_model_pt

    print('[INFO] Loading Model (%s) : %s' % (ORG, NAME))

    loaded_model_en = load_model(DATA_DIRECTORY+'hate_en.h5')
    loaded_model_pt = load_model(DATA_DIRECTORY+'hate_pt.h5')

    print('[INFO] Model %s loaded' % (ORG))

def predict(user_query,lang):
    global loaded_model_en,loaded_model_pt

    if lang == 'en':
        tokenizer = pickle.load( open( DATA_DIRECTORY+'tokenizer_en', "rb" )  )

        #tokenizer = Tokenizer(num_words= VOCABULARY_SIZE)
        #text_to_token = []
        #text_to_token.append(user_query)

        #tokenizer.fit_on_texts(text_to_token)
        #print("---> ",tokenizer.word_counts)
        #print("---> ",tokenizer.document_count)


        #tokens = text_to_word_sequence(user_query)
        #print(tokens)
        sequences = tokenizer.texts_to_sequences([user_query,""])
        #print(sequences)
        data = pad_sequences(sequences, maxlen= 100) 
        #print(data)



        result = loaded_model_en.predict_proba(data)
        #print("----->", result)
        #result = loaded_model_en.predict(data)
        #print("--------->",result)
        #result = loaded_model_en.predict_classes(data)
        #print("--------->",result)


        #result = np.round(result)
        #print(np.argmax(result, axis=0))
        #print(len(result))
        
        if result[0] == 0:
            pred = "No Hate"
        else:
            pred = "Hate"

        return pred
        
        ############ TODO fazer pop da ultima layer e trocar ativação para softma
        ### https://stackoverflow.com/questions/43083381/keras-predict-proba-does-not-add-up-to-1

        #sequences <- keras::texts_to_sequences(tokenizer, data_frame$text)
        
        #x_test <- keras::pad_sequences(sequences, maxlen = 100)


        
    elif lang == 'pt':
        pass
    else:
        return 'message : Language not supported'


