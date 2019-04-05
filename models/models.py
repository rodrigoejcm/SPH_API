import sys 
import os

### LOAD FOLDER PATH DOS MODELOS AND IMPORT
sys.path.append(os.path.abspath("models/include"))
sys.path.append(os.path.abspath("models/model_01"))
sys.path.append(os.path.abspath("models/model_02"))

#print(sys.path)

import model_01 as m01
import model_02 as m02


###


def load_models():
    print("[INFO] Loading Models")

    m01.load()
    m02.load()
    #loaded_model_02 = m02.load()
    

def predict(user_query,lang,model):
    print("[INFO] Prediction Using model ", model)

    if model == '01':
        result_query = m01.predict(user_query,lang)
        print('[INFO] FINISHED PREDICTION')
    elif model == '02':
        result_query = m02.predict(user_query,lang)
        print('[INFO] FINISHED PREDICTION')

    return result_query    