#!/usr/bin/env python
# coding: utf-8

# In[14]:


#Numerical Python for array manipulations.
import numpy as np
#Pandas for reading the dataset and converting it into dataframes.
import pandas as pd
#Math library for rounding off to next whole number.
import math
#os module to specify the directory path.
import os
import random

#Disables warnings
import warnings
warnings.filterwarnings("ignore")

#Natural Language Toolkit
import nltk
#stopwords - words that do not add much meaning to the sentence.
from nltk.corpus import stopwords
#PorterStemmer - removes common morphological endings from words (tense, number, plural, etc.)
from nltk.stem import PorterStemmer
#SentimentIntensityAnalyzer - Implements and facilitates sentiment analysis tasks.
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#nltk.download('stopwords')
#nltk.download('vader_lexicon')     #Model for text sentiment analysis
#nltk.download('punkt')             #Sentence tokenizer

#CountVectorizer - Tokenizes collection of text and builds vocabulary of known words.
from sklearn.feature_extraction.text import CountVectorizer
#TfidfVectorizer - Highlight interesting words.
from sklearn.feature_extraction.text import TfidfVectorizer
#Provides train/text split for training.
from sklearn.model_selection import KFold
#Used to fit a linear model.
from sklearn.linear_model import LinearRegression
#cohen-kappa-score is used to measure agreement between two raters.
from sklearn.metrics import cohen_kappa_score

#Word2Vec creates word embeddings (Creates word vector for each word).
from gensim.models import Word2Vec
#KeyedVectors generates mapping between keys and vectors.
from gensim.models import KeyedVectors

#Embedding translates high dimensional vectors and makes it easy to do ML on large inputs.
from tensorflow.keras.layers import Embedding
#pad-sequences ensures all sequences in a list have same length.
from tensorflow.keras.preprocessing.sequence import pad_sequences
#A sequential model is used.
from tensorflow.keras.models import Sequential
#one-hot is used to one hot encode categorical values.
from tensorflow.keras.preprocessing.text import one_hot
#LSTM layers for building the model.
#Dropout layer to prevent overfitting.
#Dense layers as the output layer.
from tensorflow.keras.layers import LSTM, Dropout, Dense

#Lambda (Creates a nameless fuction for a short period of time).
#Flatten layer is used to compress the input into 1D vector.
from keras.layers import Lambda, Flatten
#load_model is used to load a saved model.
#model_from_config instantiates a keras model from its config.
from keras.models import load_model, model_from_config
#Keras backend API.
import keras.backend as K

#Regular Expressions.
import re, string

#python spell checker.
#from spellchecker import SpellChecker

#Python grammar checker.
import language_tool_python

#Allows to send HTTP requests.
import requests
#Helps to fetch data from XML and HTML files.
from bs4 import BeautifulSoup as bs

#Used to compare a pair of inputs.
from difflib import SequenceMatcher


# In[15]:


df = pd.read_csv("/home/guru/fyp/training_set_rel3.tsv", sep='\t', encoding='ISO-8859-1')

def define_model():
    #Declare a sequential model.
    model = Sequential()
    #Add two LSTM layers a dropout layer and a dense layer with rectified linear unit as the activation function and a single output unit.
    model.add(LSTM(300, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, 300], return_sequences=True))
    model.add(LSTM(64, recurrent_dropout=0.3))
    model.add(Dropout(0.5))
    #model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='relu'))

    #Compile the model
    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy','mae'])
    #model.summary()
    
    #Return the defined model.
    return model
X=df
y = df['domain1_score']


# In[16]:


def essay_wordlist(essay_1, rem_stopwords):
    #match all strings without a letter and replace it with a white space character in the essay.
    essay_1 = re.sub("[^A-Za-z]", " ", essay_1)
    #Convert the essay into all lower case characters. 
    words = essay_1.lower().split()
    #Removing stop words from the essay.
    if rem_stopwords:
        #creates a set of stopwords in english.
        stop = set(stopwords.words("english"))
        #reassigns an essay containing no stop words.
        words = [word1 for word1 in words if not word1 in stop]
        
    #return the words list.            
    return (words)


# In[17]:


def essay_sentences(essay_1, rem_stopwords):
    #Load the pre-trained punkt tokenizer for English.
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    #Tokenizing the essay.
    sentence = tokenizer.tokenize(essay_1.strip())
    sentences = []
    #Generate word list for the tokenizer sentences.
    for sentence1 in sentence:
        if len(sentence1) > 0:
            sentences.append(essay_wordlist(sentence1, rem_stopwords))
    #Return the sentence list.
    return sentences


# In[19]:


def FeatureVector(words, model, no_feat):
    #Create an array filled with zeroes.
    FeatureVector = np.zeros((no_feat,),dtype="float32")
    no_words = 0.
    #Convert the list of names in the vocabulary into a set.
    indextoword_set = set(model.wv.index2word)
    #Calculate the word count and if a word is present in the vocabulary add it to the overall feature vector.
    for x in words:
        if x in indextoword_set:
            no_words  = no_words + 1
            FeatureVector = np.add(FeatureVector,model[x])
    #Calculate the average.       
    FeatureVector = np.divide(FeatureVector,no_words)
    #Return the feature vector.
    return FeatureVector


# In[20]:


def AvgFeatureVectors(essays, model, no_feat):
    flag = 0
    #Create another array with dimensions length of essay and number of features filled with zeroes.
    FeatureVectors = np.zeros((len(essays),no_feat),dtype="float32")
    #For each essay append the average feature vector into the FeatureVector array.
    for x in essays:
        FeatureVectors[flag] = FeatureVector(x, model, no_feat)
        flag = flag + 1
    #Return the total average feature vector.
    return FeatureVectors


# In[21]:


#Define 5 splits for KFOLD training.
x = KFold(n_splits = 5, shuffle = True)
output = []
y_pred1 = []

fold = 1
#Perform training by creating a list from the dataset for each train and test datasets for 5 folds.
for train, test in x.split(X):
    #print("\nFold {}\n".format(fold))
    #Declare test and train sets for each fold.
    x_train, x_test, y_train, y_test = X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test]
    
    #Define the test and train essays from the 'essay' column of the dataset.
    training_essays = x_train['essay']
    testing_essays = x_test['essay']
    
    a = []
    
    #Sentence tokenize each training essay.
    for essay in training_essays:
            a = a + essay_sentences(essay, rem_stopwords = True)
            
    no_feat = 300 
    word_count = 40
    no_workers = 4
    cont = 10
    sample = 1e-3

    #Predict the nearby words for each word in the sentence.
    model = Word2Vec(a, workers=no_workers, size=no_feat, min_count = word_count, window = cont, sample = sample)

    #Normalize vectors (Equal length)
    model.init_sims(replace=True)
    #Save the model.
    model.wv.save_word2vec_format('word2vecmodel.bin', binary=True)

    cleaning_train_essays = []
    
    #For each training essay generate a word list.
    for essay_1 in training_essays:
        cleaning_train_essays.append(essay_wordlist(essay_1, rem_stopwords=True))
    #Generate average feature vectors for the word lists.
    Vectors_train = AvgFeatureVectors(cleaning_train_essays, model, no_feat)
    
    #Similarly for the test essays generate word lists and average feature vectors.
    cleaning_test_essays = []
    for essay_1 in testing_essays:
        cleaning_test_essays.append(essay_wordlist( essay_1, rem_stopwords=True ))
    Vectors_test = AvgFeatureVectors( cleaning_test_essays, model, no_feat )
    
    #Reshape the average feature vectors of test and train datasets to the shape of first dimension of the respective data vectors.
    Vectors_train = np.array(Vectors_train)
    Vectors_test = np.array(Vectors_test)
    Vectors_train = np.reshape(Vectors_train, (Vectors_train.shape[0], 1, Vectors_train.shape[1]))
    Vectors_test = np.reshape(Vectors_test, (Vectors_test.shape[0], 1, Vectors_test.shape[1]))
    
    #Assign the defined model.
    lstm_model = define_model()
    #Fit the model.
    lstm_model.fit(Vectors_train, y_train, batch_size=64, epochs=20, verbose = 0)
    #Load the model weights.
    #lstm_model.load_weights('./fyp/model.h5')
    y_predict = lstm_model.predict(Vectors_test)
    
    #Save the model when all the folds are completed.
    if fold == 5:
         #lstm_model.save('./fyp/model.h5')
        model_json = lstm_model.to_json()
        with open("model1.json", "w") as json_file:
            json_file.write(model_json)
# serialize weights to HDF5
        lstm_model.save_weights("model1.h5")
        print("Saved model to disk")
        
    
    #Round off the predicted value.
    y_predict = np.around(y_predict)
    
    #Generate a kappa score for each fold.
    result = cohen_kappa_score(y_test.values,y_predict,weights='quadratic')
    #print("Kappa Score for fold {fold} is {score}".format(fold = fold, score = result))
    #Add each kappa score to the overall score.
    output.append(result)

    #Increment the value of fold.
    fold = fold + 1


# In[ ]:




