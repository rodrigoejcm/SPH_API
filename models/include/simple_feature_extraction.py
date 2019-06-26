import pandas as pd
import pickle
import json
from scipy.sparse import coo_matrix, hstack,vstack
import gender_guesser.detector as gender
import pandas,numpy, textblob, string

from gensim.corpora import Dictionary
import ast

import spacy
from spacy import displacy
import en_core_web_sm

import gensim
from gensim import corpora, models

from datetime import datetime

import numpy as np
np.random.seed(42)

import nltk


from nltk.corpus import stopwords,brown
from nltk.stem.snowball import SnowballStemmer
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer


from collections import Counter


DATA_DIRECTORY = "models/include/shared_data/"

target_columns_new_feats  = ['authors_count', 'authors_male', 'authors_female', 'authors_unknown',
'char_count_text', 'word_count_text', 'word_density_text',
'punctuation_count_text', 'title_word_count_text',
'upper_case_word_count_text', 'char_count_lead', 'word_count_lead',
'word_density_lead', 'punctuation_count_lead', 'title_word_count_lead',
'upper_case_word_count_lead', 'char_count_title', 'word_count_title',
'word_density_title', 'punctuation_count_title',
'title_word_count_title', 'upper_case_word_count_title',
'noun_count_title', 'verb_count_title', 'adj_count_title',
'adv_count_title', 'pron_count_title', 'noun_count_lead',
'verb_count_lead', 'adj_count_lead', 'adv_count_lead',
'pron_count_lead', 'noun_count_text', 'verb_count_text',
'adj_count_text', 'adv_count_text', 'pron_count_text',
'part_of_the_day', 'weekday', 'ent_CARDINAL_count_title',
'ent_DATE_count_title', 'ent_GPE_count_title', 'ent_MONEY_count_title',
'ent_NORP_count_title', 'ent_ORG_count_title', 'ent_PERSON_count_title',
'ent_TIME_count_title', 'ent_CARDINAL_prop_title', 'ent_DATE_prop_title',
'ent_GPE_prop_title', 'ent_MONEY_prop_title', 'ent_NORP_prop_title',
'ent_ORG_prop_title', 'ent_PERSON_prop_title', 'ent_TIME_prop_title',
'ent_CARDINAL_count_text','ent_DATE_count_text','ent_GPE_count_text',
'ent_MONEY_count_text', 'ent_NORP_count_text', 'ent_ORG_count_text',
'ent_PERSON_count_text','ent_TIME_count_text','ent_CARDINAL_prop_text',
'ent_DATE_prop_text','ent_GPE_prop_text','ent_MONEY_prop_text',
'ent_NORP_prop_text','ent_ORG_prop_text','ent_PERSON_prop_text',
'ent_TIME_prop_text','ent_CARDINAL_count_lead','ent_DATE_count_lead',
'ent_GPE_count_lead','ent_MONEY_count_lead','ent_NORP_count_lead',
'ent_ORG_count_lead','ent_PERSON_count_lead','ent_TIME_count_lead',
'ent_CARDINAL_prop_lead','ent_DATE_prop_lead','ent_GPE_prop_lead',
'ent_MONEY_prop_lead','ent_NORP_prop_lead','ent_ORG_prop_lead','ent_PERSON_prop_lead',
'ent_TIME_prop_lead','topic','name_usuario_tweet']

############ DOWNLOAD NLTK RESOURCER

FIRST_TIME_LOADING = False

nltk_download = ['punkt', 'averaged_perceptron_tagger','wordnet', 'stopwords', 'tagsets', 'universal_tagset']

if FIRST_TIME_LOADING:
    for rcr in nltk_download:
        nltk.download(rcr)

d = gender.Detector(case_sensitive=False)

def author_presence(a):
  a = a
  return len(a)

def author_genre(a):
  auts = a
  count_male = 0
  count_female = 0
  count_unknown = 0
  
  if auts[0].strip():

    if auts:
      for aut in auts:
        name_gender = d.get_gender(aut.split()[0])
        if (name_gender == "male" or name_gender == "mostly_male" ):
          count_male+=1
        elif (name_gender == "female" or name_gender == "mostly_female" ):
          count_female+=1
        elif (name_gender == "unknown" or name_gender == "andy" ):
          count_unknown+=1
  else:
    count_male = 0
    count_female = 0
    count_unknown = 1


  return count_male,count_female,count_unknown


### POS FEATURES

pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
}

# function to check and get the part of speech tag count of a words in a given sentence
def check_pos_tag(x, flag):
    cnt = 0
    try:
        wiki = textblob.TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in pos_family[flag]:
                cnt += 1
    except:
        pass
    return cnt
  
nlp = en_core_web_sm.load()

def remove_stop_words(text,lang="english",min_word_length = 3):
  if lang == "english":
    stops = set(stopwords.words("english"))
  if lang == "pt-br":
    pass

  text = text.split()
  text = [w for w in text if not w in stops and len(w) >= min_word_length]
  text = " ".join(text)
  
  return text

def extract_entitie(text):
  text = nlp(text)
  return [(X.text, X.label_) for X in text.ents]

def extract_entitie_count(text , ent = None , check_existence = False):
  text = remove_stop_words(text)
  text = nlp(text)
  labels = [x.label_ for x in text.ents]
  counts = dict(Counter(labels))
  
  if ent and ent in counts :
    if check_existence:
      return True
    else:
      return counts[ent]
  else:
    if check_existence:
        if counts:
          return True
        else:
          return False
    else:
      return 0
    
stemmer = SnowballStemmer("english") # Choose a language
    
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result    
    

   
lda_model = models.LdaModel.load(DATA_DIRECTORY+'lda.model')
dictionary = Dictionary.load_from_text(DATA_DIRECTORY+"lda_dict")
    
def topic_text(text):
  bow_vector = dictionary.doc2bow(preprocess(text))
  return sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1])[0][0]

### FEATURE EXTRACTOR

def extract_features(title,lead,body,authors,potd,weekday,source):

  dataset = pd.DataFrame(columns=['title_news', 'lead_news', 'text_news', 'authors_rev', 'part_of_the_day',	'weekday'	,'name_usuario_tweet'])

  dataset = dataset.append({'title_news': title,
                            'lead_news': lead,
                            'text_news': body,
                            'authors_rev': authors.split(','),
                            'part_of_the_day': int(potd),
                            'weekday': int(weekday),
                            'name_usuario_tweet': int(source)}, ignore_index=True)

  dataset['part_of_the_day'] = dataset['part_of_the_day'].astype('int64')
  dataset['weekday'] = dataset['weekday'].astype('int64')
  dataset['name_usuario_tweet'] = dataset['name_usuario_tweet'].astype('int64')
  
  ### AUTHORS 
  dataset['authors_count'] = dataset['authors_rev'].apply(author_presence)
  dataset['authors_male'],dataset['authors_female'],dataset['authors_unknown'] = zip(*dataset['authors_rev'].apply(author_genre))
  
  
  ## NLP FEATURES

  ### COUNT FEATURES

  #### TEXT COUNT

  dataset['char_count_text'] = dataset['text_news'].apply(len)
  dataset['word_count_text'] = dataset['text_news'].apply(lambda x: len(x.split()))
  dataset['word_density_text'] = dataset['char_count_text'] / (dataset['word_count_text']+1)
  dataset['punctuation_count_text'] = dataset['text_news'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 
  dataset['title_word_count_text'] = dataset['text_news'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
  dataset['upper_case_word_count_text'] = dataset['text_news'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))


  #### LEAD COUNT

  dataset['char_count_lead'] = dataset['lead_news'].apply(len)
  dataset['word_count_lead'] = dataset['lead_news'].apply(lambda x: len(x.split()))
  dataset['word_density_lead'] = dataset['char_count_lead'] / (dataset['word_count_lead']+1)
  dataset['punctuation_count_lead'] = dataset['lead_news'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 
  dataset['title_word_count_lead'] = dataset['lead_news'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
  dataset['upper_case_word_count_lead'] = dataset['lead_news'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))



  #### TITLEDY COUNT

  dataset['char_count_title'] = dataset['title_news'].apply(len)
  dataset['word_count_title'] = dataset['title_news'].apply(lambda x: len(x.split()))
  dataset['word_density_title'] = dataset['char_count_title'] / (dataset['word_count_title']+1)
  dataset['punctuation_count_title'] = dataset['title_news'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 
  dataset['title_word_count_title'] = dataset['title_news'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
  dataset['upper_case_word_count_title'] = dataset['title_news'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))
  
  
  ## POS FEATURES 
  
  #### TITLE COUNT

  dataset['noun_count_title'] = dataset['title_news'].apply(lambda x: check_pos_tag(x, 'noun'))
  dataset['verb_count_title'] = dataset['title_news'].apply(lambda x: check_pos_tag(x, 'verb'))
  dataset['adj_count_title'] = dataset['title_news'].apply(lambda x: check_pos_tag(x, 'adj'))
  dataset['adv_count_title'] = dataset['title_news'].apply(lambda x: check_pos_tag(x, 'adv'))
  dataset['pron_count_title'] = dataset['title_news'].apply(lambda x: check_pos_tag(x, 'pron'))


  #### LEAD COUNT

  dataset['noun_count_lead'] = dataset['lead_news'].apply(lambda x: check_pos_tag(x, 'noun'))
  dataset['verb_count_lead'] = dataset['lead_news'].apply(lambda x: check_pos_tag(x, 'verb'))
  dataset['adj_count_lead'] = dataset['lead_news'].apply(lambda x: check_pos_tag(x, 'adj'))
  dataset['adv_count_lead'] = dataset['lead_news'].apply(lambda x: check_pos_tag(x, 'adv'))
  dataset['pron_count_lead'] = dataset['lead_news'].apply(lambda x: check_pos_tag(x, 'pron'))


  #### TEXT COUNT

  dataset['noun_count_text'] = dataset['text_news'].apply(lambda x: check_pos_tag(x, 'noun'))
  dataset['verb_count_text'] = dataset['text_news'].apply(lambda x: check_pos_tag(x, 'verb'))
  dataset['adj_count_text'] = dataset['text_news'].apply(lambda x: check_pos_tag(x, 'adj'))
  dataset['adv_count_text'] = dataset['text_news'].apply(lambda x: check_pos_tag(x, 'adv'))
  dataset['pron_count_text'] = dataset['text_news'].apply(lambda x: check_pos_tag(x, 'pron'))
  
  
  ## ENTITY FEATURES
  
  #### TITLE ENTITY COUNT

  dataset['ent_CARDINAL_count_title'] = dataset['title_news'].apply(lambda x: extract_entitie_count(x, 'CARDINAL'))
  dataset['ent_DATE_count_title'] = dataset['title_news'].apply(lambda x: extract_entitie_count(x, 'DATE'))
  dataset['ent_GPE_count_title'] = dataset['title_news'].apply(lambda x: extract_entitie_count(x, 'GPE'))
  dataset['ent_MONEY_count_title'] = dataset['title_news'].apply(lambda x: extract_entitie_count(x, 'MONEY'))
  dataset['ent_NORP_count_title'] = dataset['title_news'].apply(lambda x: extract_entitie_count(x, 'NORP'))
  dataset['ent_ORG_count_title'] = dataset['title_news'].apply(lambda x: extract_entitie_count(x, 'ORG'))
  dataset['ent_PERSON_count_title'] = dataset['title_news'].apply(lambda x: extract_entitie_count(x, 'PERSON'))
  dataset['ent_TIME_count_title'] = dataset['title_news'].apply(lambda x: extract_entitie_count(x, 'TIME'))


  #### TITLE ENTITY PROPORTION

  dataset['ent_CARDINAL_prop_title'] = dataset['ent_CARDINAL_count_title'] / dataset['word_count_title']
  dataset['ent_DATE_prop_title'] = dataset['ent_DATE_count_title'] / dataset['word_count_title']
  dataset['ent_GPE_prop_title'] = dataset['ent_GPE_count_title'] / dataset['word_count_title']
  dataset['ent_MONEY_prop_title'] = dataset['ent_MONEY_count_title'] / dataset['word_count_title']
  dataset['ent_NORP_prop_title'] = dataset['ent_NORP_count_title'] / dataset['word_count_title']
  dataset['ent_ORG_prop_title'] = dataset['ent_ORG_count_title']/ dataset['word_count_title']
  dataset['ent_PERSON_prop_title'] = dataset['ent_PERSON_count_title'] / dataset['word_count_title']
  dataset['ent_TIME_prop_title'] = dataset['ent_TIME_count_title'] / dataset['word_count_title']
  
  
  #### text ENTITY COUNT

  dataset['ent_CARDINAL_count_text'] = dataset['text_news'].apply(lambda x: extract_entitie_count(x, 'CARDINAL'))
  dataset['ent_DATE_count_text'] = dataset['text_news'].apply(lambda x: extract_entitie_count(x, 'DATE'))
  dataset['ent_GPE_count_text'] = dataset['text_news'].apply(lambda x: extract_entitie_count(x, 'GPE'))
  dataset['ent_MONEY_count_text'] = dataset['text_news'].apply(lambda x: extract_entitie_count(x, 'MONEY'))
  dataset['ent_NORP_count_text'] = dataset['text_news'].apply(lambda x: extract_entitie_count(x, 'NORP'))
  dataset['ent_ORG_count_text'] = dataset['text_news'].apply(lambda x: extract_entitie_count(x, 'ORG'))
  dataset['ent_PERSON_count_text'] = dataset['text_news'].apply(lambda x: extract_entitie_count(x, 'PERSON'))
  dataset['ent_TIME_count_text'] = dataset['text_news'].apply(lambda x: extract_entitie_count(x, 'TIME'))


  #### text ENTITY PROPORTION

  dataset['ent_CARDINAL_prop_text'] = dataset['ent_CARDINAL_count_text'] / dataset['word_count_text']
  dataset['ent_DATE_prop_text'] = dataset['ent_DATE_count_text'] / dataset['word_count_text']
  dataset['ent_GPE_prop_text'] = dataset['ent_GPE_count_text'] / dataset['word_count_text']
  dataset['ent_MONEY_prop_text'] = dataset['ent_MONEY_count_text'] / dataset['word_count_text']
  dataset['ent_NORP_prop_text'] = dataset['ent_NORP_count_text'] / dataset['word_count_text']
  dataset['ent_ORG_prop_text'] = dataset['ent_ORG_count_text']/ dataset['word_count_text']
  dataset['ent_PERSON_prop_text'] = dataset['ent_PERSON_count_text'] / dataset['word_count_text']
  dataset['ent_TIME_prop_text'] = dataset['ent_TIME_count_text'] / dataset['word_count_text']
  
  #### lead ENTITY COUNT
  
  dataset['ent_CARDINAL_count_lead'] = dataset['lead_news'].apply(lambda x: extract_entitie_count(x, 'CARDINAL'))
  dataset['ent_DATE_count_lead'] = dataset['lead_news'].apply(lambda x: extract_entitie_count(x, 'DATE'))
  dataset['ent_GPE_count_lead'] = dataset['lead_news'].apply(lambda x: extract_entitie_count(x, 'GPE'))
  dataset['ent_MONEY_count_lead'] = dataset['lead_news'].apply(lambda x: extract_entitie_count(x, 'MONEY'))
  dataset['ent_NORP_count_lead'] = dataset['lead_news'].apply(lambda x: extract_entitie_count(x, 'NORP'))
  dataset['ent_ORG_count_lead'] = dataset['lead_news'].apply(lambda x: extract_entitie_count(x, 'ORG'))
  dataset['ent_PERSON_count_lead'] = dataset['lead_news'].apply(lambda x: extract_entitie_count(x, 'PERSON'))
  dataset['ent_TIME_count_lead'] = dataset['lead_news'].apply(lambda x: extract_entitie_count(x, 'TIME'))


  #### lead ENTITY PROPORTION

  dataset['ent_CARDINAL_prop_lead'] = dataset['ent_CARDINAL_count_lead'] / dataset['word_count_lead']
  dataset['ent_DATE_prop_lead'] = dataset['ent_DATE_count_lead'] / dataset['word_count_lead']
  dataset['ent_GPE_prop_lead'] = dataset['ent_GPE_count_lead'] / dataset['word_count_lead']
  dataset['ent_MONEY_prop_lead'] = dataset['ent_MONEY_count_lead'] / dataset['word_count_lead']
  dataset['ent_NORP_prop_lead'] = dataset['ent_NORP_count_lead'] / dataset['word_count_lead']
  dataset['ent_ORG_prop_lead'] = dataset['ent_ORG_count_lead']/ dataset['word_count_lead']
  dataset['ent_PERSON_prop_lead'] = dataset['ent_PERSON_count_lead'] / dataset['word_count_lead']
  dataset['ent_TIME_prop_lead'] = dataset['ent_TIME_count_lead'] / dataset['word_count_lead']
  
  
  ## TOPIC
  
  dataset['topic'] = dataset['text_news'].apply(topic_text)
  

  return dataset[target_columns_new_feats]

#test = extract_features(dataset)

#test

#loaded_model_lg = pickle.load(open('SVM_v01.pkl', 'rb'))
#loaded_model_svm = pickle.load(open('LG_v01.pkl', 'rb'))
#result_lg = loaded_model_lg.predict(test)
#result_svm = loaded_model_svm.predict(test)

#result_lg[0]

#result_svm[0]

#print(dataset.dtypes)