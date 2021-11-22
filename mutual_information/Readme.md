### Foundation work(use mutual information to pick out features) for a Naive Bayesian Classifier to classify the sentiments of moive reviews.


### 1. You should find on the “Files” page a folder called "Movie Reviews". In it, you will find two sub-folders called "pos" and "neg", containing movie reviews which have been judged as positive and negative, respectively.
### 2. Download the data and divide the data into 2 subsets: training data and testing data.
### 3. Write a well documented program in python which
####      Takes as input the paths of two folders which contain movie reviews judged as positive and negative, respectively.  
####      Processes the movie reviews and convert them into a form which can be used for training a Bayesian classifier OR evaluating it.

####      Produces as output the top 5 positive and top 5 negative evidences in your model (i.e. the features/attributesAi which have the highest I(Ai , pos) and I(Ai ,neg) values) in a human readable form (e.g. P1: Word:Amazing, P2: Word: Good, .... N1: Disappointing, N2: Forgettable,... ).
####      You can use imported libraries for preprocessing the movie reviews (e.g. tokenizers). But you cannot use libraries for other purposes.

#####     Useful files: preprocess.py  wordcounter.py  main.py   output.txt

## Step by step:
## 1. Use pandas to read all data into a data frame (tokenize the reivews to a list of words), then split the data into training set and test set. 
## 2. Use set to get vocabulary of whole dataset, positive review train set, negative review train set respectively.
## 3. Create dictionary to map words to their frequencies in whole dataset, positive review train set, negative review train set respectively.
## 4. Calculate mutual information.
