import sklearn.naive_bayes as NB
import sklearn.linear_model as LR
import nltk
import random
import numpy as np
import pandas as pd
import pyprind
random.seed(0)
import os
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
import tensorflow as tf
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfTransformer
import itertools
import matplotlib.pyplot as plt
import multiprocessing

#nltk.download("stopwords")          # Download the stop words from nltk


def main(method, input_type, min_word_count = None, polarity_cutoff = None):
    
    if method == "nlp":
        train_pos, train_neg, test_pos, test_neg = load_data(input_type, method)
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP (train_pos, train_neg, test_pos, test_neg, min_word_count, polarity_cutoff)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    elif method == "d2v":
        train_pos, train_neg, test_pos, test_neg = load_data(input_type, method)
        model, train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC (train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model, test_predictions = build_models_DOC(train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec)
    
    print ('\033[1m' + 'Evaluate Naive Bayes Model')
    print ("-----------")
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True, model_type = 'NB')
    print ("")
    print ('\033[1m' + 'Evaluate Logistic Regression Model')
    print ("-------------------")
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True, model_type = 'LR')
    
    if method == "d2v":
        print ("")
        print ('\033[1m' + "4 Layer Feed forward Neural Network model for Doc2Vec")
        print ("-------------------")
        evaluate_model(None, test_pos_vec, test_neg_vec, True, 'NN', test_predictions)

def load_data(input_type, model_type):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    path_to_dir = os.getcwd()+"\\data\\" + input_type + "\\"
    if (input_type == 'imdb'):
        prog_bar_train = 12500
        prog_bar_test = 12500
    if (input_type == 'twitter'):
        prog_bar_train = 375000
        prog_bar_test = 75000
        
    def text_cleasing(line, model_type):
        if (model_type == 'nlp'):
            emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', line.strip())
            line = re.sub('[\W]+',' ',line.lower()) + ' '.join(emoticons).replace('-','')
        
        #Remove URLs for both nlp and d2v
        line = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', line.lower())
        words = [w for w in line.strip().split() if len(w)>=3]
        return words
    
      
    print("Reading " + input_type + " Data Set")
    with open(path_to_dir+"train-pos.txt", "r", encoding="utf8") as f:
        pbar = pyprind.ProgBar(prog_bar_train)
        for i,line in enumerate(f):
            train_pos.append(text_cleasing(line, model_type))
            pbar.update()
    with open(path_to_dir+"train-neg.txt", "r", encoding="utf8") as f:
        pbar = pyprind.ProgBar(prog_bar_train)
        for line in f:
            train_neg.append(text_cleasing(line, model_type))
            pbar.update()
    with open(path_to_dir+"test-pos.txt", "r", encoding="utf8") as f:
        pbar = pyprind.ProgBar(prog_bar_test)
        for line in f:
            test_pos.append(text_cleasing(line, model_type))
            pbar.update()
    with open(path_to_dir+"test-neg.txt", "r", encoding="utf8") as f:
        pbar = pyprind.ProgBar(prog_bar_test)
        for line in f:
            test_neg.append(text_cleasing(line, model_type))          
            pbar.update()

    return train_pos, train_neg, test_pos, test_neg


def remove_stopwords(train_pos, train_neg, test_pos, test_neg):
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
        
    return train_pos_clean,train_neg_clean, test_pos_clean, test_neg_clean 

def pre_process_data(train_pos_clean,train_neg_clean):
    
    positive_counts = Counter()
    negative_counts = Counter()
    total_counts = Counter()
    pos_neg_ratios = Counter()
    
    for word_list in (train_pos_clean):
        for word in word_list:
            positive_counts[word] += 1
            total_counts[word] += 1               
    for word_list in (train_neg_clean):
        for word in word_list:
            negative_counts[word] += 1
            total_counts[word] += 1   
        
    for word,cnt in list(total_counts.most_common()):
        # Consider only if the word appears for more than 50 times
        if(cnt > 50):
            pos_neg_ratio = positive_counts[word] / float(negative_counts[word]+1)
            pos_neg_ratios[word] = pos_neg_ratio

    for word,ratio in pos_neg_ratios.most_common():
        if(ratio > 1):
            pos_neg_ratios[word] = np.log(ratio)
        else:
            pos_neg_ratios[word] = -np.log((1 / (ratio+0.01)))         
  
    return total_counts,positive_counts, negative_counts, pos_neg_ratios




def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg, min_count = 10, polarity_cutoff = 0.1):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    print("Removing Stop words")
    train_pos_clean,train_neg_clean, test_pos_clean, test_neg_clean  = remove_stopwords(train_pos, train_neg, test_pos, test_neg)
    print("Gathering feature vectors. Using Minimum Word Count %f and Polarity Cut off %f" % (min_count,polarity_cutoff))
    total_counts,positive_counts, negative_counts, pos_neg_ratios = pre_process_data(train_pos_clean,train_neg_clean)
    
   
    all_words = set()
    for word_list in (train_pos_clean + train_neg_clean):
        for word in word_list:
            if(total_counts[word] > min_count):
                if(word in pos_neg_ratios.keys()):
                    if((pos_neg_ratios[word] >= polarity_cutoff) or (pos_neg_ratios[word] <= -polarity_cutoff)):
                        all_words.add(word)
                else:
                    all_words.add(word)
            
            
    all_words = list(all_words)
    print("Vocabulary Size (Unique Word List): %f" %(len(all_words)))
    word_index = {}
    for i, word in enumerate(all_words):
        word_index[word] = i   
       
    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.
    
    train_pos_vec = np.zeros((len(train_pos_clean), len(all_words)))
    train_neg_vec = np.zeros((len(train_neg_clean), len(all_words)))
    
    test_pos_vec = np.zeros((len(test_pos_clean), len(all_words)))
    test_neg_vec = np.zeros((len(test_neg_clean), len(all_words)))
    
    for idx in range(len(train_pos_clean)):
        indexes = set([word_index[word] if word in word_index.keys() else None for word in train_pos_clean[idx]])   
        indexes = [idx for idx in indexes if idx != None]
        for index, replacement in zip(indexes, [1] * len(indexes)):
            train_pos_vec[idx][index] = replacement
    
    for idx in range(len(train_neg_clean)):
        indexes = set([word_index[word] if word in word_index.keys() else None for word in train_neg_clean[idx]])  
        indexes = [idx for idx in indexes if idx != None]
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
        
    
    # Convert the feature vectors to weights using TF-IDF transformer
    #tfidf= TfidfTransformer(norm='l2')
    #train_vec = np.concatenate((train_pos_vec,train_neg_vec),axis=0)   
    #test_vec = np.concatenate((test_pos_vec,test_neg_vec),axis=0)   
    
    #train_vec = tfidf.fit_transform(train_vec)
    #test_vec = tfidf.fit_transform(test_vec)
    

    # Return the four feature vectors
    #return train_vec.toarray(), test_vec.toarray()
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.
    
    #print("Removing Stop words")
    #train_pos_clean,train_neg_clean, test_pos_clean, test_neg_clean  = remove_stopwords(train_pos, train_neg, test_pos, test_neg)
    
    def labelizeReviews(reviews, label_type):
        labelized = []
        for i,v in enumerate(reviews):
            label = '%s_%s'%(label_type,i)            
            labelized.append(LabeledSentence(v, [label]))
        return labelized
    
 
    # Labelize Reviews/ Tweets
    labeled_train_pos = labelizeReviews(train_pos, 'TRAIN_POS')
    labeled_train_neg = labelizeReviews(train_neg, 'TRAIN_NEG')
    labeled_test_pos =  labelizeReviews(test_pos, 'TEST_POS')
    labeled_test_neg =  labelizeReviews(test_neg, 'TEST_NEG')

    # Initialize model
    NUM_WORKERS = multiprocessing.cpu_count()
    model = Doc2Vec(min_count=1, alpha =  0.025, min_alpha=0.025, window=10, size=100, sample=1e-4, negative=5, workers=NUM_WORKERS)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)
    
    train_pos_vec = []
    train_neg_vec = []
    test_pos_vec = []
    test_neg_vec = []
    
    
    # Train the model
    # This may take a bit to run 
    print("Training Doc2Vec using Distributed Memory and Negative Sampling")
    for i in range(5):
        print ("Training iteration %d" % (i))
        random.shuffle(sentences)
        model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
        model.alpha -= 0.002  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay
        
    
    print("Generating Feature Vectors for Train POSITIVE")
    # Use the docvecs function to extract the feature vectors for the training and test data
    for i in range(len(labeled_train_pos)):
        inf_vector = model.docvecs[labeled_train_pos[i].tags[0]]
        train_pos_vec.append(inf_vector)
    train_pos_vec= np.array(train_pos_vec)
    
            
    print("Generating Feature Vectors for Train NEGATIVE")
    for i in range(len(labeled_train_neg)):
        inf_vector = model.docvecs[labeled_train_neg[i].tags[0]]        
        train_neg_vec.append(inf_vector)
    train_neg_vec = np.array(train_neg_vec)
    
            
    print("Generating Feature Vectors for Test POSITIVE")
    for i in range(len(labeled_test_pos)):
        inf_vector = model.docvecs[labeled_test_pos[i].tags[0]]        
        test_pos_vec.append(inf_vector)
    test_pos_vec = np.array(test_pos_vec)
    
            
    print("Generating Feature Vectors for Test NEGATIVE")
    for i in range(len(labeled_test_neg)):
        inf_vector = model.docvecs[labeled_test_neg[i].tags[0]]        
        test_neg_vec.append(inf_vector)
    test_neg_vec = np.array(test_neg_vec)
    
        
    print("Done")    
    # Return the four feature vectors
    return model, train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """
    X = np.concatenate((train_pos_vec,train_neg_vec),axis =0)
    Y = np.array(["POSITIVE"]*len(train_pos_vec) + ["NEGATIVE"]*len(train_neg_vec))
    
    # Check is Features and Labels are of Same size
    assert(len(X) == len(Y))
    print("Training Naive Bayes")
    
    nb_model = NB.BernoulliNB(alpha = 1.0, binarize = None)
    nb_model.fit(X , Y)
    
    
    print("Training Logistic Regression")

    # For LogisticRegression, pass loss paramter as log    
    lr_model = LR.SGDClassifier(loss = "log", penalty='l1')
    lr_model.fit(X , Y)
    
    print ("Training Complete")
    
    return nb_model, lr_model



def build_models_DOC(train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    X = np.concatenate((train_pos_vec,train_neg_vec),axis =0)
    Y = np.array(["POSITIVE"]*len(train_pos_vec) + ["NEGATIVE"]*len(train_neg_vec))
    
    assert(len(X) == len(Y))

    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
    print("Training Gaussian Naive Bayes")
    nb_model = NB.GaussianNB()
    nb_model.fit(X, Y)
    
    print("Training Logistic Regression")
    lr_model = LR.SGDClassifier()
    lr_model.fit(X , Y)
    
    print("Training Neural Network using Tensorflow")
    test_predictions = build_models_Neuralnet(train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec)
    
    print("Training Complete")
    
    return nb_model, lr_model, test_predictions


def build_models_Neuralnet(train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec):
    
    def batches(batch_size, features, labels):
        n_batches = len(features)//batch_size
    
    # only full batches    
        features = features[:n_batches*batch_size]
        for i in range(0, len(features), batch_size):
            batch_X = features[i:i + batch_size]
            batch_Y = labels[i:i + batch_size]
            yield batch_X, batch_Y
        
    def print_epoch_stats(epoch_i, sess, last_features, last_labels):
        """
        Print cost and validation accuracy of an epoch
        """
        current_cost = sess.run(cost,feed_dict={features: last_features, labels: last_labels})
        training_accuracy = sess.run(accuracy,feed_dict={features: last_features, labels: last_labels})
        valid_accuracy = sess.run(accuracy,feed_dict={features: valid_features, labels: valid_labels})
        print('Epoch: {:<4} - Cost: {:<8.3} Training Accuracy: {:<5.3} Validation Accuracy: {:<5.3}'.format(epoch_i,current_cost, training_accuracy, valid_accuracy))
        
    def one_hot_encode(x):
        """
        One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
        : x: List of sample Labels
        : return: Numpy array of one-hot encoded labels
        """
        one_hot_array=[]
        for l in x:
            holder = np.repeat(0,2)
            np.put(holder,l,1)
            one_hot_array.append(holder)
    
        return np.array(one_hot_array)
    
    def fully_connected(features_tensor, num_outputs, num_inputs = None):
        """
        Apply a fully connected layer to x_tensor using weight and bias
        : x_tensor: A 2-D tensor where the first dimension is batch size.
        : num_outputs: The number of output that the new tensor should be.
        : return: A 2-D tensor where the second dimension is num_outputs.
        """
        if (num_inputs != None):
            inputs = num_inputs
        else:
            inputs = features_tensor.get_shape().as_list()[1]
        weights = tf.Variable(tf.truncated_normal([inputs, num_outputs],stddev=0.1))
        bias = tf.Variable(tf.truncated_normal([num_outputs],stddev=0.1))        
        fc = tf.add(tf.matmul(features_tensor,weights),bias)
        fc = tf.nn.sigmoid(fc)
        #fc = tf.nn.relu(fc)
        #fc = tf.nn.dropout(fc, keep_prob = 0.70)
        return fc
        
    
    n_input = 100
    n_classes = 2
    ip_features = np.concatenate((train_pos_vec,train_neg_vec),axis =0)
    ip_labels = np.array([1]*len(train_pos_vec) + [0]*len(train_neg_vec))
    
    ip_labels = one_hot_encode(ip_labels)
    
    
    train_features, valid_features, train_labels, valid_labels = train_test_split(ip_features, ip_labels, test_size=0.2)

    
    test_features = np.concatenate((test_pos_vec,test_neg_vec),axis =0)
    test_labels = np.array([1]*len(test_pos_vec) + [0]*len(test_neg_vec))
    test_labels = one_hot_encode(test_labels)
    
    # Features and Labels
    features = tf.placeholder(tf.float32, [None, n_input])
    labels = tf.placeholder(tf.float32, [None, n_classes])
    
    layer_1 = fully_connected(features, num_outputs = 256, num_inputs = n_input)
    layer_2 = fully_connected(layer_1, num_outputs = 512)
    logits = fully_connected(layer_2, num_outputs = n_classes)

     
    # Define loss and optimizer
    learning_rate = tf.placeholder(tf.float32)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Calculate accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.global_variables_initializer()

    batch_size = 128
    epochs = 250
    learn_rate = 0.001
        
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        print("Training 4 layer Feed forward Neural Network classifier over Do2Vec Vectors")
        for epoch_i in range(epochs+1):
            train_batches = batches(batch_size, train_features, train_labels)
            # Loop over all batches
            for batch_features, batch_labels in train_batches:
                train_feed_dict = {
                        features: batch_features,
                        labels: batch_labels,
                        learning_rate: learn_rate}
                sess.run(optimizer, feed_dict=train_feed_dict)                

            # Print cost and validation accuracy for every 10 iterations
            if (epoch_i%25 == 0):
                print_epoch_stats(epoch_i, sess, batch_features, batch_labels)

        # Calculate accuracy for test dataset
        test_accuracy = sess.run(
                accuracy,
                feed_dict={features: test_features, labels: test_labels})
        
        pred =  logits
        test_feed_dict = { features: test_features}
        prediction = sess.run(pred,feed_dict = test_feed_dict)
        

        print('Test Accuracy: {}'.format(test_accuracy))
    
    
    test_predictions = []
    for i in range(len(prediction)):
        if prediction[i][1] >= 0.5:
            test_predictions.append('POSITIVE')
        else:
            test_predictions.append('NEGATIVE')
    test_predictions =np.array(test_predictions)
    
    return test_predictions
    
   
def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False, model_type = None, NN_predictions = None):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    def plot_confusion_matrix(cm, text, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues
                          ):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
           
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]            
        
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.title(text)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
            
    
        # Use the predict function and calculate the true/false positives and true/false negative.
    if (model_type in ('LR','NB')):
        test_X = np.concatenate((test_pos_vec,test_neg_vec),axis =0)
        test_Y = np.array(["POSITIVE"]*len(test_pos_vec) + ["NEGATIVE"]*len(test_neg_vec))
        results = model.predict(test_X)
        cmatrix = confusion_matrix(test_Y, results)
    elif (model_type =='NN'):
        test_Y = np.array(["POSITIVE"]*len(test_pos_vec) + ["NEGATIVE"]*len(test_neg_vec))
        cmatrix = confusion_matrix(test_Y, NN_predictions)
        results = NN_predictions
        
   
    accuracy = (cmatrix[0][0] + cmatrix[1][1])/len(results)
    precision = cmatrix[0][0] / (cmatrix[0][0] + cmatrix[1][0])
    recall = cmatrix[0][0] / (cmatrix[0][0] + cmatrix[0][1])
    sensitivity = recall
    specificity = cmatrix[1][1] / (cmatrix[1][1] + cmatrix[1][0])
    
    if (model_type == 'NB'):
        text = 'Naive Bayes'
    elif (model_type == 'LR'):
        text = 'Logistic Regression'
    elif (model_type == 'NN'):
        text = 'Neural Network'
    
    if print_confusion:
        plt.figure()
        plot_confusion_matrix(cmatrix, text, classes=np.array(['POSITIVE','NEGATIVE']),title='Confusion matrix')
        
    print (text + " Accuracy: %f" % (accuracy))
    print (text + " Precision: %f" % (precision))
    print (text + " Recall: %f" % (recall))
    print (text + " Sensitivity: %f" % (sensitivity))
    print (text + " Specificity: %f" % (specificity))

if __name__ == "__main__":
    main(method ="nlp", input_type = "imdb", min_word_count = 50, polarity_cutoff = 0.1)
    main(method ="nlp", input_type = "twitter", min_word_count = 75, polarity_cutoff = 0.1)
    
    main(method ="d2v", input_type = "imdb")
    main(method ="d2v", input_type = "twitter")

