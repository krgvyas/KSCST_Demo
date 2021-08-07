#!/usr/bin/env python
# coding: utf-8

# In[28]:
#Flask framework for creatinga POST API
from flask import Flask, redirect, url_for, request

#Json for receiving and sending data in json format.
import json
from json import dumps, loads, JSONEncoder, JSONDecoder

#os module to specify the directory path.
import os
#Disables warnings
import warnings
warnings.filterwarnings("ignore")

#To avoid displaying unwanted text messages.
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
#Keras framework
import keras
sys.stderr = stderr

#Numerical Python for array manipulations.
import numpy as np
#Pandas for reading the dataset and converting it into dataframes.
import pandas as pd
#Math library for rounding off to next whole number.
import math
#Randomization
import random

#Natural Language Toolkit
import nltk
#stopwords - words that do not add much meaning to the sentence.
from nltk.corpus import stopwords
#PorterStemmer - removes common morphological endings from words (tense, number, plural, etc.)
from nltk.stem import PorterStemmer
#SentimentIntensityAnalyzer - Implements and facilitates sentiment analysis tasks.
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#Twitter samles for training a sentiment analysis model.
from nltk.corpus import twitter_samples
#pos_tag is used to tag tokens.
from nltk.tag import pos_tag
#Groups different forms of words.
from nltk.stem.wordnet import WordNetLemmatizer
#Frequency of various kinds of words.
from nltk import FreqDist
#Classifying a text as positive or negative.
from nltk import classify
from nltk import NaiveBayesClassifier
#Extract tokens from string of characters.
from nltk.tokenize import word_tokenize

#nltk.download('stopwords')	    #English Stop words
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
from keras.models import model_from_json

#Regular Expressions.
import re, string

#Python grammar checker.
import language_tool_python

#Allows to send HTTP requests.
import requests
#Helps to fetch data from XML and HTML files.
from bs4 import BeautifulSoup as bs

#Used to compare a pair of inputs.
from difflib import SequenceMatcher

#Importing classifier from sentiment analysis python file.
from sentiment import classifier
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
#Creating a flask API.
#app = Flask(__name__)
#creating an API URL and using POST request.
@app.route('/essay_grading', methods = ['POST'])
#@cross_origin()

#The main function
def original_func():
	if request.method=='POST':
		#Fetching the requested data(essay and expected sentiment analysis)
		posted_data = request.get_json()
		content = posted_data['content']
		expected_sent = posted_data['expected_sent']
		
        #return jsonify(str("Successfully stored  " + str(data)))
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

	no_feat = 300 

	json_file = open('model1.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("model1.h5")

	#Load the saved word2vec model.
	model = KeyedVectors.load_word2vec_format( "./word2vecmodel.bin", binary=True)
	test_essays = []
	#Create a word list, average input features and reshape the input essay.
	test_essays.append(essay_wordlist( content, rem_stopwords=True ))
	test_vectors = AvgFeatureVectors( test_essays, model, no_feat )
	test_vectors = np.array(test_vectors)
	test_vectors = np.reshape(test_vectors, (test_vectors.shape[0], 1, test_vectors.shape[1]))

	#Generate grade prediction for the input essay.
	preds = loaded_model.predict(test_vectors)

	tool = language_tool_python.LanguageTool('en-US')
	matches1 = []
	
	#Parsing the input essay to detect syntactic errors.
	matches = tool.check(content)
	#matches1  = list(matches)

	#Removing noise from the text.
	def remove_noise(tweet_tokens, stop_words = ()):
		cleaned_tokens = []
		#Replacing the substrings with appropriate tokens.
		for token, tag in pos_tag(tweet_tokens):
			token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
			token = re.sub("(@[A-Za-z0-9_]+)","", token)
			
			#If the tag is a noun.
			if tag.startswith("NN"):
		    		pos = 'n'
			#If tag is a verb.
			elif tag.startswith('VB'):
		    		pos = 'v'
			else:
		    		pos = 'a'
			
			#Lemmatizer converts the word to its meaningful form i.e to its base form (Caring to care).
			lemmatizer = WordNetLemmatizer()
			token = lemmatizer.lemmatize(token, pos)

			#populate the cleaned tokens list with valid tokens(no punctuations, no stopwords).
			if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
		    		cleaned_tokens.append(token.lower())
		#Return the cleaned tokens list.
		return cleaned_tokens

	#Create custom tokes for the input essay, by removing noise.
	custom_tokens = remove_noise(word_tokenize(content))

	#Obtain the sentiment of the essay by using the classifier.
	sent1 = classifier.classify(dict([token, True] for token in custom_tokens))
	#Positive sentiment is considered 1.
	if sent1 == 'Positive':
		sent = 1
	#Negative sentiment is considered 0.
	elif sent1 == 'Negative':
		sent = 0
	    
	def search(query, num):
	    
	    #Define a URL to perform searching
		url = 'https://www.bing.com/search?q=' + query
		url1 = []
	    
	    #Generae a HTTP request to the URL
		x = requests.get(url, headers = {'User-agent': 'John Doe'})
	    #Fetch the data in the site using beautifulsoup.
		y = bs(x.text, 'html.parser')
	    
	    #Append the current examined URL into a list.
		for a in y.find_all('a'):
			url = str(a.get('href'))
			if url.startswith('http'):
		    		if not url.startswith('http://go.m') and not url.startswith('https://go.m'):
		        		url1.append(url)
	    
		return url1[:num]

	def extract(url):
		x = requests.get(url)
		y = bs(x.text, 'html.parser')
	    #Return the text from the site.
		return y.get_text()
	#Define Stopping words.
	stopping_words = set(nltk.corpus.stopwords.words('english'))

	def TokenizeText(string):
	    #Tokenize the string in the site.
		words = nltk.word_tokenize(string)
	    #Return all the non stopping words.
		return (" ".join([word for word in words if word not in stopping_words]))

	def Verify(string, results_per_sentence):
	    #Sentence tokenize the string in the site.
		sentences = nltk.sent_tokenize(string)
		matching_sites = []
	    #Detect URLs where similar content is found.
		for url in search(query=string, num=results_per_sentence):
			matching_sites.append(url)
	    #Detct the sentences in the URL.
		for sentence in sentences:
			for url in search(query = sentence, num = results_per_sentence):
				matching_sites.append(url)
	    
	    #Return the URLs
		return (list(set(matching_sites)))

	def similarity(str1, str2):
	    #Match the entire (100%) two contents and return the ratio of similarities between the two texts.
		return (SequenceMatcher(None,str1,str2).ratio())*100


	def result(text):
	    
	    #Copare the two texts.
		matching_sites = Verify(TokenizeText(text), 2)
		matches = {}

	    #For each matching site determine the amount of similarity.
		for i in range(len(matching_sites)):
			matches[matching_sites[i]] = similarity(text, extract(matching_sites[i]))

	    #Sort the similarities in descending order
		matches = {k: v for k, v in sorted(matches.items(), key=lambda item: item[1], reverse=True)}

	    #Return the URLs and their corresponding plagiarized percentage score as a dictionary.
		return matches

	plag1 = {}
	total = 0
	plag1 = result(content)
	for key, value in plag1.items():
	    #Calculate the total plagiarism percentage.
		total = total+value
	plag = dict(zip(plag1.values(), plag1.keys()))
	#For each key, value pair in the dictionary, display the URL and the amount of % plagiarism.
	

	#Obtaining grade for the quality of the essay and reducing it to the ratio of 3 from ratio of 10.
	x = math.ceil(preds)
	a = (x * 3)/10
	#Determining the number of grammatical and spelling mistakes, and 0.25 marks is deducted for each one.
	b = 0.25 * len(matches)
	if b > 3:
		b = 3
	#if plagiarism is less than 15%, no marks are deducted for plagiarism.
	if total <= 15:
		c = 0
	#If plagiarism is between 15-25%, 0.5 marks are deducted.
	elif total > 15 and total <= 25:
		c = 0.5
	#If plagiarism is between 25-50%, 1 mark is deducted.
	elif total > 25 and total <= 50:
		c = 1
	#If plagiarism is between 50-75%, 1,5 marks are deducted.
	elif total > 50 and total <= 75:
		c = 1.5
	#If it is between 75-85%, 2 marks are deducted.
	elif total > 75 and total <= 85:
		c = 2
	#If it is between 85-95, then 2.5 marks are deducted.
	elif total > 85 and total <= 95:
		c = 2.5
	#If is is greater than 95%, then complete 3 marks are deducted.
	elif total > 95:
		c = 3

	#If the obtained sentiment matches the expected sentiment for the topic, then no marks are deducted.
	if sent == expected_sent:
		d = 0
	#If it is not similar, 1 mark is deducted.
	else:
		d = 1
	#If the expected sentiment is neutral, then no marks are deducted for either positive or negative sentiment of the essay.
	if expected_sent == 0.5:
		d = 0
	    
	sent_op = ""

	#Rounding up the marks for quality of the essay upto 2 decimal places.
	a1 = np.around(a, decimals = 2)
	#Displaying comments for the coresponding sentiments of the essay.
	if expected_sent == sent:
		sent_op = "Your approach towards the topic is acceptable and relevant\t1"
	elif expected_sent == 1 and sent == 0:
		sent_op="The expected attitude is considered to be in support of the topic but, Your approach is against and not relevenant\t0"
	elif expected_sent == 0 and sent == 1:
		sent_op = "The expected attitude is considered not to be in support of the topic but, Your approach is supporting and not relevenant\t0"

	#Calculatig the score.	    
	score = 10 - (3-a) - b - c - d

	#Determining the final score by rounding up the score to the next whole number.
	final_score = math.ceil(score)

	#Displaying comments for corresponding scores obtained.
	if final_score < 5:
		y = "Poor"
	#Else if the predicted score is between 5 and 8 the grade is average.
	elif final_score >= 5 and final_score < 8:
		y = "Average"
	#If the predicted score is greater than or equal to 8 then the grade is Excellent.
	else:
		y = "Excellent"

	mistakes = []
	corrections = []
	positions1 = []
	positions2 = []

	#For each syntactical mistake in the essay, replace the mistake with the appropriate correction at the corresponding offset value and the error length in a new list.
	for a in matches:
		if len(a.replacements)>0:
			positions1.append(a.offset)
			positions2.append(a.errorLength+a.offset)
			mistakes.append(content[a.offset:a.errorLength+a.offset])
			corrections.append(a.replacements[0])
	     
	 
	#Create a list of the input essay.
	new_text = list(content)

	#Create a new string of text based on the values in the mistakes list and the original list by joining the two lists appropriately.
	for m in range(len(positions1)):
		for i in range(len(content)):
			new_text[positions1[m]] = corrections[m]
			if (i>positions1[m] and i<positions2[m]):
		    		new_text[i]=""
	     
	new_text = "".join(new_text)
	new_text

	#Fucntion to return json serilizable data.
	def myconverter(o):
        	return o.__str__()
	#json serializing the Match data type.
	matches1 = json.dumps(matches, default = myconverter)

	#Creating a dictionary of all the items to be diaplyed as part of result to the student.		
	dict1 = {"errors" : matches1 , "sentiment" : sent1, "plagiarism" : plag, "total_plagiarism" : total, "Quality_of_Essay" : a1, "spell_error" : b, "plagiarism_marks_lost" : c, "sentiment_op" : sent_op, "final_grade" : final_score, "comment" : y, "corrected_essay" : new_text}
	#Clearing the keras backed session.
	K.clear_session()
	
	#Returning the dictionary.
	return dict1

# In[ ]:
#Running the application.
app.run(debug = True)


