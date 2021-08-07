#!/usr/bin/env python
# coding: utf-8

# In[2]:

#Twitter dataset for training.
from nltk.corpus import twitter_samples
#pos_tag is used to tag tokens.
from nltk.tag import pos_tag
#Groups different forms of words.
from nltk.stem.wordnet import WordNetLemmatizer
#frequency of various forms of words.
from nltk import FreqDist
#Classifying a text as positive or negative.
from nltk import classify
from nltk import NaiveBayesClassifier
#Extract tokens from string of characters.
from nltk.tokenize import word_tokenize
#English stopwords
from nltk.corpus import stopwords
#Regular expression, string and randomization.
import re, string, random
#Natural Language toolkit
import nltk

#Fetching the positive and negative tweets from the respective json files.
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')
text = twitter_samples.strings('tweets.20150430-223406.json')
tweet_tokens = twitter_samples.tokenized('positive_tweets.json')

#lemmatizing a sentence based on whether they are nouns, verbs or others.
def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence


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

#Initializing english Stop words
stop_words = stopwords.words('english')

#Initializing tokenized positive and negative tweets respectively.
positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

positive_cleaned_tokens_list = []
negative_cleaned_tokens_list = []

#Removing noise from positive and negative tweets.
for tokens in positive_tweet_tokens:
    positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

for tokens in negative_tweet_tokens:
    negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))
#Fetching all the words from the tokens list.
def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token
#Fetching all the positive words and determining the frequency of them.
all_pos_words = get_all_words(positive_cleaned_tokens_list)
freq_dist_pos = FreqDist(all_pos_words)

#Fetching tokens from cleaned tokens list for the model.
def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

#Fetching the positive and negaive tokens for the model
positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

#Deterining the positive and negative tweets dataset.
positive_dataset = [(tweet_dict, "Positive")
                     for tweet_dict in positive_tokens_for_model]

negative_dataset = [(tweet_dict, "Negative")
                     for tweet_dict in negative_tokens_for_model]

#combining both into a common dataset.
dataset = positive_dataset + negative_dataset

#Randolmy shuffling the dataset.
random.shuffle(dataset)

#Splitting train and test datasets.
train_data = dataset[:7000]
test_data = dataset[7000:]

#Defining a naive bayes classifier.
classifier = NaiveBayesClassifier.train(train_data)


# In[ ]:




