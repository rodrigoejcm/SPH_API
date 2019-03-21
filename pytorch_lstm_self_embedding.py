


# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import keras_metrics

import tensorflow as tf

from keras import backend as K
## Plotly
import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt

# Others
import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


#sklearn
from sklearn.model_selection import KFold, cross_val_score
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.utils import class_weight
from sklearn.metrics import classification_report

import sys 
import os
sys.path.append(os.path.abspath("include"))
import clean_text_en



### DOWNLOAD NLTK RESOURCES

#####
print("Dowloading NLTK resources ")
#####

nltk.download('stopwords')



##### DATA DICTIONARY
###  0 -->> hate speech 
###  1 -->> offensive language 
###  2 -->> neither

#####
print("Loading Training Dataset")
#####


df_train = pd.read_csv("data_model/train_clean.csv")
df_train['label'] = df_train['label'].map(lambda x : 0 if int(x) == 0 else 1)
df_train = df_train[['text','label']]


### CLASSES INTANCES
#0     1427
#1    23340

#####
print("Loading Testing Dataset")
#####



df_test = pd.read_csv("data_model/validation_SN.tsv" , sep='\t')
df_test = df_test[df_test['HS'].isin(['S','N'])]
df_test['HS'] = df_test['HS'].map(lambda x : 0 if x == "S" else 1)
df_test = df_test[['text_full_reply', 'HS']]
df_test.rename(columns={'text_full_reply':'text','HS':'label'}, inplace=True)

### CLASSES INTANCES
#0    116
#1    623

"""### Clean Data to train and test
Clean, Remove Stop Words, Steaming
"""


#####
print("Cleaning Train and Test Data")
#####


df_test['text'] = df_test['text'].apply(clean_text_en.clean_text_full)
df_train['text'] = df_train['text'].apply(clean_text_en.clean_text_full)

### CHECK EMPTY STRING AFTER CLEANING AND DROP

#####
print("Droping Empty instances")
#####


ids_to_remove_from_train = df_train[df_train['text'] == "EMPTY STRING"].index
ids_to_remove_from_test = df_test[df_test['text'] == "EMPTY STRING"].index

print("removendo ", len(ids_to_remove_from_train),len(ids_to_remove_from_test) )

df_train = df_train.drop(ids_to_remove_from_train)
df_test = df_test.drop(ids_to_remove_from_test)


#####
print("Steaming Dataset")
#####


def steam_text(text):
    text = text.split()
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    return " ".join(stemmed_words)

df_test['text'] = df_test['text'].apply(steam_text)
df_train['text'] = df_train['text'].apply(steam_text)


#####
print("Configuring Model")
#####

"""### Data and Model"""



label_train = df_train['label']
label_test = df_test['label']

y_train = to_categorical(label_train, num_classes=2)
y_test = to_categorical(label_test)

#####
print("Configuring Embedding Layer")
#####

## Create sequence
vocabulary_size = 20000
tokenizer_train = Tokenizer(num_words= vocabulary_size)
tokenizer_train.fit_on_texts(df_train['text'])
sequences_train = tokenizer_train.texts_to_sequences(df_train['text'])
data_train = pad_sequences(sequences_train, maxlen=200)

## Create sequence
tokenizer_test = Tokenizer(num_words= vocabulary_size)
tokenizer_test.fit_on_texts(df_test['text'])
sequences_test = tokenizer_test.texts_to_sequences(df_test['text'])
data_test = pad_sequences(sequences_test, maxlen=200)

### CLASS WEIGHT

#####
print("Setting Class weights")
#####

cweight = class_weight.compute_class_weight("balanced" ,np.unique(label_train) ,label_train)
cweight = {0:1, 1:16}

"""#### Network architecture"""



#####
print("Configuring Metrics")
#####

# Calculate precision for the second label.
precision = keras_metrics.precision(label=0)

# Calculate recall for the first label.
recall = keras_metrics.recall()

precision_01 = keras_metrics.precision(label=1)

# Calculate recall for the first label.
recall_01 = keras_metrics.recall(label=1)


#####
print("Configuring DL network architecture")
#####


model = Sequential()
model.add(Embedding(20000, 100, input_length=200))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#precision,recall, precision_01, recall_01

model.summary()

"""#### Train model"""



#from tensorflow.python.client import device_lib
#device_lib.list_local_devices()

#####
print("Training Model")
#####


early_stopping_monitor = EarlyStopping(patience=3)
model.fit(data_train, y_train, validation_data=(data_test,y_test), epochs=3, batch_size=50, callbacks=[early_stopping_monitor], class_weight = cweight)


#####
print('serialize model to JSON')
#####

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")

#####
print("Saved model to disk")
#####


#score = model.evaluate(data_test,np.array(label_test), verbose=1)

#model.load_weights("weights.hdf5")
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
y_pred = model.predict_classes(data_test)



print(classification_report(label_test, y_pred))

cfsm = confusion_matrix(label_test, y_pred)

trace = go.Heatmap(z=cfsm,
                  x=[0, 1],
                  y=[0, 1])
data=[trace]
py.iplot(data, filename='basic-heatmap')


fig = ff.create_annotated_heatmap(z=cfsm,
                  x=[0, 1],
                  y=[1, 0])
py.iplot(fig, filename='annotated_heatmap')







# Function to create model, required for KerasClassifier
#def create_model(optimizer='adam'):
#  model = Sequential()
#  model.add(Embedding(20000, 100, input_length=200))
#  model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
#  model.add(Dense(1, activation='sigmoid'))
#  model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#  return model

#model = KerasClassifier(build_fn=create_model, verbose=50)

#batch_size = [10]
#epochs = [1]
#param_grid = dict(batch_size=batch_size, epochs=epochs)
#grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, scoring="f1_micro", cv=3, verbose=60)
#grid_result = grid.fit(data_train, np.array(label_train))

# summarize results
#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#means = grid_result.cv_results_['mean_test_score']
#stds = grid_result.cv_results_['std_test_score']
#params = grid_result.cv_results_['params']
#for mean, stdev, param in zip(means, stds, params):
#    print("%f (%f) with: %r" % (mean, stdev, param))