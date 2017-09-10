# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 17:49:57 2017

@author: vpandrap
"""

import sys
import collections
import sklearn.naive_bayes as NB
import sklearn.linear_model as LR
import nltk
import random
import numpy as np
random.seed(0)
import os
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.metrics import confusion_matrix
#nltk.download("stopwords")          # Download the stop words from nltk


# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets

def main(method, input_type):
    path_to_data = os.getcwd()+"\\data\\" + input_type + "\\"
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)
    
    if method == "nlp":
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
  #  if method == "d2v":
      #  train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
        #nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
    print ("Naive Bayes")
    print ("-----------")
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print ("")
    print ("Logistic Regression")
    print ("-------------------")
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)
    



def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i,line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_neg.append(words)

    return train_pos, train_neg, test_pos, test_neg



def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # English stopwords from nltk
    stopwords = set(nltk.corpus.stopwords.words('english'))
       
    # Determine a list of words that will be used as features. 
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.
    # YOUR CODE HERE
    
    train_pos_clean = []
    train_neg_clean = []
    test_pos_clean = []
    test_neg_clean = []    
    
    # Text Cleansing: Convert to lower case, remove stop words, remove special characters
    for idx in range(len(train_pos)):
        words = [word for word in train_pos[idx] if word not in stopwords]
        train_pos_clean.append(words)
    for idx in range(len(train_neg)):
        words = [word for word in train_neg[idx] if word not in stopwords]
        train_neg_clean.append(words)
    for idx in range(len(test_pos)):
        words = [word for word in test_pos[idx] if word not in stopwords]
        test_pos_clean.append(words)
    for idx in range(len(test_neg)):
        words = [word for word in test_neg[idx] if word not in stopwords]
        test_neg_clean.append(words)
        
    
    all_words = set()
    for word_list in (train_pos_clean):
        for word in word_list:
            all_words.add(word)
    for word_list in (train_neg_clean):
        for word in word_list:
            all_words.add(word)                                 
    
    all_words = list(all_words)
    word_index = {}
    for i, word in enumerate(all_words):
        word_index[word] = i   
       
    train_pos_vec = np.zeros((len(train_pos_clean), len(all_words)))
    train_neg_vec = np.zeros((len(train_neg_clean), len(all_words)))
    
    test_pos_vec = np.zeros((len(test_pos_clean), len(all_words)))
    test_neg_vec = np.zeros((len(test_neg_clean), len(all_words)))
    
    for idx in range(len(train_pos_clean)):
         indexes = set([word_index[word] for word in train_pos_clean[idx]])                 
         for index, replacement in zip(indexes, [1] * len(indexes)):
             train_pos_vec[idx][index] = replacement
    
    for idx in range(len(train_neg_clean)):
         indexes = set([word_index[word] for word in train_neg_clean[idx]])                 
         for index, replacement in zip(indexes, [1] * len(indexes)):
             train_neg_vec[idx][index] = replacement
                          
    for idx in range(len(test_pos_clean)):
         indexes = set([word_index[word] if word in word_index.keys() else None for word in test_pos_clean[idx]]) 
         indexes = [idx for idx in indexes if idx != None]
         for index, replacement in zip(indexes, [1] * len(indexes)):
             test_pos_vec[idx][index] = replacement 
    
    for idx in range(len(test_neg_clean)):
         indexes = set([word_index[word] if word in word_index.keys() else None for word in test_neg_clean[idx]])                 
         indexes = [idx for idx in indexes if idx != None]
         for index, replacement in zip(indexes, [1] * len(indexes)):
             test_neg_vec[idx][index] = replacement     
        
    
    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.
    

    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.
    # YOUR CODE HERE
    def labelizeReviews(reviews, label_type):
        labelized = []
        for i,v in enumerate(reviews):
            label = '%s_%s'%(label_type,i)            
            labelized.append(LabeledSentence(v, [label]))
        return labelized
    
    labeled_train_pos = labelizeReviews(train_pos, 'TRAIN_POS')
    labeled_train_neg = labelizeReviews(train_neg, 'TRAIN_NEG')
    labeled_test_pos =  labelizeReviews(test_pos, 'TEST_POS')
    labeled_test_neg =  labelizeReviews(test_neg, 'TEST_NEG')

    # Initialize model
    model = Doc2Vec(min_count=1, alpha =  0.05, min_alpha=0.025, window=10, size=100, sample=1e-4, negative=5, workers=4)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)

    # Train the model
    # This may take a bit to run 
    for i in range(5):
        print ("Training iteration %d" % (i))
        random.shuffle(sentences)
        model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
        model.alpha -= 0.002  # decrease the learning rate
        model.min_alpha = model.alpha
        
    for i in range(len(labeled_train_pos)):
        inf_vector = np.array(model.infer_vector(labeled_train_pos[i].words))
        inf_vector = inf_vector.reshape((1,len(inf_vector)))
        
        
    
    
       

    # Use the docvecs function to extract the feature vectors for the training and test data
    # YOUR CODE HERE
    
    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """
    X = np.concatenate((train_pos_vec,train_neg_vec),axis =0)
    Y = np.array(["POSITIVE"]*len(train_pos_vec) + ["NEGATIVE"]*len(train_neg_vec))
    
    assert(len(X) == len(Y))
    
    nb_model = NB.BernoulliNB(alpha = 1.0, binarize = None)
    nb_model.fit(X , Y)

    # For LogisticRegression, pass no parameters
    
    lr_model = LR.SGDClassifier()
    lr_model.fit(X , Y)
    
    # YOUR CODE HERE
    
    return nb_model, lr_model



def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    
    return nb_model, lr_model



def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.
    test_X = np.concatenate((test_pos_vec,test_neg_vec),axis =0)
    test_Y = np.array(["POSITIVE"]*len(test_pos_vec) + ["NEGATIVE"]*len(test_neg_vec))
    results = model.predict(test_X)
    cmatrix = confusion_matrix(test_Y, results)
    
    accuracy = (cmatrix[0][0] + cmatrix[1][1])/len(results)
    tp  = cmatrix[0][0]
    fn =  cmatrix[0][1]
    fp =  cmatrix[1][0]
    tn =  cmatrix[1][1]
    
    
    if print_confusion:
        print ("predicted:\tpos\tneg")
        print ("actual:")
        print ("pos\t\t%d\t%d" % (tp, fn))
        print ("neg\t\t%d\t%d" % (fp, tn))
    print ("accuracy: %f" % (accuracy))


if __name__ == "__main__":
    main(method ="nlp", input_type = "imdb")
