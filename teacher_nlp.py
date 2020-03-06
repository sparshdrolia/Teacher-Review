 from __future__ import print_function, division
import nltk
import os
import random
from collections import Counter
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import NaiveBayesClassifier, classify
import pandas as pd
#%%
stoplist = stopwords.words('english')


def preprocess(sentence):
    lemmatizer = WordNetLemmatizer()#some words hataichi
    return [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(str(sentence))]#different words banaichi

def get_features(text, setting):
    if setting=='bow':#random words hoichi
        return {word: count for word, count in Counter(preprocess(text)).items() if not word in stoplist}
    else:
        return {word: True for word in preprocess(text) if not word in stoplist}

def train(features, samples_proportion):
    train_size = int(len(features) * samples_proportion)
    # training ani test initialize karaichi
    train_set, test_set = features[:train_size], features[train_size:]
    print ('Training set size = ' + str(len(train_set)) + ' reviews')
    print ('Test set size = ' + str(len(test_set)) + ' reviews')
    # trainaichi the classifier
    classifier = NaiveBayesClassifier.train(train_set)
    return train_set, test_set, classifier

def evaluate(train_set, test_set, classifier):
    
    print ('hey bhagwan accuracy niklaichi (train) = ' + str(classify.accuracy(classifier, train_set)))
    print ('sem as above but test= ' + str(classify.accuracy(classifier, test_set)))
   
    classifier.show_most_informative_features(20)
#%%
df = pd.read_csv('C:/Users/hp/Desktop/nlp_data.csv')
data = []
# arrange data in  format (review,label)
for index,rows in df.iterrows():
    a = (rows['Review'],rows['Useful'])
    data.append(a)
# data
# for (each,label) in data:
#     print(each,label)
#%%
    
# feature extraction
corpus_features = [(get_features(each,''),label) for (each,label) in data]
print ('Collected ' + str(len(corpus_features)) + ' feature sets')

# training the classifier
train_set, test_set, classifier = train(corpus_features, 0.6)


evaluate(train_set, test_set, classifier)
#%%
lineList = [line.rstrip('\n') for line in open(r'C:\Users\hp\Desktop\sentiment.txt',)]
#print(lineList)
#test1=preprocess(lineList[0])
#print(test1)
feature_test=get_features(lineList[0],"bow")
print(feature_test)
print(classifier.classify(feature_test))
