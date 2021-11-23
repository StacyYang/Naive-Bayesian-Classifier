# Movie Review Classification using Naïve Bayes
This repository implements in Python a Naïve Bayes classifier with picked features(based on mutual information) and Add-one smoothing. It implements the algorithm from scratch and does not use off-the-shelf software.


## How to build naive_bayesian_classifier: 
### files: preprocess.py, wordcounter.py, main.py
#### preprocess.py

Takes as input the paths of two folders which contain movie reviews judged as positive and negative, respectively.  
Processes(lowercase-->remove stopwords-->stem words) the movie reviews and push them to a dataframe then write to pos_train.csv and neg_train.csv file which will be used for training a Bayesian classifier OR evaluating it.

dataframe/.csv file format:
1.one document per line
2.three columns in total: id, sentiment(1 for positive and 0 for negative), reviews(a list of meaningful words).

#### wordcounter.py
Takes pos_train.csv and neg_train.csv, then collect info. needed. positive review number, negative review number, positive word number, negative word number, unique words, freq_dict, pos_dict and neg_dict.
       
#### main.py
1. pick features based on abs(I(A,pos) - I(A,neg))
2. Train: calculate logprior and (loglikelihood for picked features).
3. Test: a. cal sum_of_log_prob for each movie review.   b. predict, review is positive if prob>1.



## Implementation 2:
### file: run.py  (run_nb_classifier.py)
Takes as input the path of a folder which contains multiple files, each containing a movie review, AND the path to a file that stores a Bayesian classifier for movie reviews, 
Loads the Bayesian classifier stored in the file,
Processes each of the movie reviews and convert them into a form which can be classified using the classifier, AND
Uses the classifier loaded to classify each of the movie reviews as positive or negative, and output the results in a .csv file, with each line showing the name of the file containing the movie review, followed by a comma, followed by either "1" or "0" (meaning pos and neg, respectively).
Your .csv file should contain results like this:
aa101.txt,0
aa102.txt,1
aa103.txt,1

