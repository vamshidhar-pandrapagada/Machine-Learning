##Supervised Learning Techniques for Sentiment Analytics

In this project, we will perform sentiment analysis over IMDB movie reviews and Twitter data. The goal will be to classify tweets or movie reviews as either positive or negative. For classification, we'll experiment with logistic regression as well as a Naive Bayes classifier from python’s well-regarded machine learning package scikit-learn .
A major part of this project is the task of generating feature vectors for use in these classifiers.

###Explore two methods:  

1. A more traditional NLP technique where the features are simply “important” words and the feature vectors are simple binary vectors.
2. Doc2Vec technique where document vectors are learned via artificial neural networks.

### Datasets
The IMDB reviews and tweets can be found in the data folder. These have already been divided into train and test sets.  
1. The IMDB dataset, originally found here, that contains 50,000 reviews split evenly into 25k train and 25k test sets. Overall, there are 25k pos and 25k neg reviews. In the
labeled train/test sets, a negative review has a score <= 4 out of 10, and a positive review has a score >= 7 out of 10. Thus reviews with more neutral ratings are not
included in the train/test sets.
2. The Twitter Dataset, taken from here , contains 900,000 classified tweets split into 750k train and 150k test sets. The overall distribution of labels is balanced (450k pos and 450k neg).