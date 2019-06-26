import sys 
import os

### LOAD FOLDER PATH DOS MODELOS AND IMPORT
sys.path.append(os.path.abspath("models/include"))
sys.path.append(os.path.abspath("models/model_01"))
sys.path.append(os.path.abspath("models/model_02"))
sys.path.append(os.path.abspath("models/models_simple_features_news"))

#print(sys.path)

import model_01 as m01
import model_02 as m02
import models_sf as msf01



###


def load_models():
    print("[INFO] Loading Models")

    ### COMMENTS
    m01.load()
    m02.load()

    ### NEWS
    msf01.load_svmv01()
    msf01.load_lgbmv01()

def predict(user_query,lang,model):
    print("[INFO] Prediction Using model ", model)

    if model == '01':
        result_query = m01.predict(user_query,lang)
        print('[INFO] FINISHED PREDICTION')
    elif model == '02':
        result_query = m02.predict(user_query,lang)
        print('[INFO] FINISHED PREDICTION')

    return result_query    

def predict_news_hatespeech(title_news,lead_news,text_news,authors_news,source_news,week_news,period_news,model="svm_v01"):

    if model:
        print("[INFO] Prediction Using model ", model)

        if model == 'svm_v01':
            print("1")
            result_query = msf01.predic_svmv01(title_news,lead_news,text_news,authors_news,source_news,week_news,period_news)
            print('[INFO] FINISHED PREDICTION')
        elif model == 'lgbm_v01':
            print("2")
            result_query = msf01.predic_lgbmv01(title_news,lead_news,text_news,authors_news,source_news,week_news,period_news)
            print('[INFO] FINISHED PREDICTION')
    else:
        ### DEFAULT MODEL
        print("3")
        model == 'svm_v01'
        print("[INFO] Prediction Using model ", model)
        result_query = msf01.predic_svmv01(title_news,lead_news,text_news,authors_news,source_news,week_news,period_news)
        print('[INFO] FINISHED PREDICTION')

    return result_query    