# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 22:43:29 2017

@author: vpandrap
"""

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
        
    
    # Quick Theory Validation
    positive_counts = collections.Counter()
    negative_counts = collections.Counter()
    total_counts = collections.Counter()
    pos_neg_ratios = collections.Counter()
    
    for word_list in (train_pos_clean):
        for word in word_list:
            positive_counts[word] += 1
            total_counts[word] += 1               
    for word_list in (train_neg_clean):
        for word in word_list:
            negative_counts[word] += 1
            total_counts[word] += 1                                         
    
    

    for word,cnt in list(total_counts.most_common()):
        # Consider only if the word appears for more than 100 times
        if(cnt > 100):
            pos_neg_ratio = positive_counts[word] / float(negative_counts[word]+1)
            pos_neg_ratios[word] = pos_neg_ratio

    for word,ratio in pos_neg_ratios.most_common():
        if(ratio > 1):
            pos_neg_ratios[word] = np.log(ratio)
        else:
            pos_neg_ratios[word] = -np.log((1 / (ratio+0.01))) 

    
      
    all_word_list =  list(total_counts.keys())
    random.shuffle(all_word_list)
    
    train_pos_vec = np.zeros((len(train_pos_clean), len(all_word_list)))
    train_neg_vec = np.zeros((len(train_neg_clean), len(all_word_list)))
    
    test_pos_vec = np.zeros((len(test_pos_clean), len(all_word_list)))
    test_neg_vec = np.zeros((len(test_neg_clean), len(all_word_list)))
    
    
    
    for idx in range(len(train_pos_clean)):
         indexes = set([all_word_list.index(word) for word in train_pos_clean[idx]])         
         
         for index, replacement in zip(indexes, [1] * len(indexes)):
             train_pos_vec[idx][index] = replacement
         
        
    
         
    print(len(train_pos_clean[0]))
    print(len(train_pos_vec[0]))
         
        
    
        
    

    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.
    # YOUR CODE HERE

    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec