
import sys 
import os

sys.path.append(os.path.abspath("models/include"))

import pickle
from sklearn import svm
import lightgbm as lgb

from simple_feature_extraction import extract_features

NAME = "SVM Classification Model news hatespeech - V01 "
ORG = "svm_v01"
DATA_DIRECTORY = "models/models_simple_features_news/data/"

def load_svmv01():
    global loaded_model_svm_v01
    filename = DATA_DIRECTORY+"SVM_v01.pkl"
    loaded_model_svm_v01 = pickle.load(open(filename, 'rb'))
    print("loaded SVM")

def load_lgbmv01():
    global loaded_model_lgbm_v01
    filename = DATA_DIRECTORY+"LG_v01.pkl"
    loaded_model_lgbm_v01 = pickle.load(open(filename, 'rb'))
    print("loaded LG")

def predic_svmv01(title_news,lead_news,text_news,authors_news,source_news,week_news,period_news):
    global loaded_model_svm_v01
    ds = extract_features(title_news,lead_news,text_news,authors_news,period_news,week_news,source_news)
    print("[INFO] Extracted Features")
    result = loaded_model_svm_v01.predict(ds)
    if result[0] == 0: 
        pred = "No Hate"
    else:
        pred = "Hate"

    return pred

def predic_lgbmv01(title_news,lead_news,text_news,authors_news,source_news,week_news,period_news):
    global loaded_model_lgbm_v01
    ds = extract_features(title_news,lead_news,text_news,authors_news,period_news,week_news,source_news)
    result = loaded_model_lgbm_v01.predict(ds)
    if result[0] == 0:
        pred = "No Hate"
    else:
        pred = "Hate"

    return  pred

