# Movie Review Classification using Naïve Bayes
This repository implements in Python a Naïve Bayes classifier with picked features(based on mutual information) and Add-one smoothing. It implements the algorithm from scratch and does not use off-the-shelf software.


## How to build naive_bayesian_classifier: 
### files: preprocess.py, wordcounter.py, main.py
preprocess.py

Takes the paths of two folders containing training movie reviews, pos and neg.

Prior to building feature vectors, I separated punctuation from words and lowercased the words in the reviews.
output the files in the vector format to be used by NB.py.
The training and the test files which are output from pre-process.py have the following format:

one document per line
each line corresponds to a document
first column is the label
the other columns are feature values.
NB.py takes the following parameters:

the training file output from pre-process.py
the test file output from pre-process.py
the file where the parameters of the resulting model will be saved
the output file which stores predictions made by the classifier on the test data (one document per line).
The last line in the output file lists the overall accuracy of the classifier on the test data.


## Implementation 2:
### file: run.py
Takes as input the path of a folder which contains multiple files, each containing a movie review, AND the path to a file that stores a Bayesian classifier for movie reviews, 
Loads the Bayesian classifier stored in the file,
Processes each of the movie reviews and convert them into a form which can be classified using the classifier, AND
Uses the classifier loaded to classify each of the movie reviews as positive or negative, and output the results in a .csv file, with each line showing the name of the file containing the movie review, followed by a comma, followed by either "1" or "0" (meaning pos and neg, respectively).
Your .csv file should contain results like this:
aa101.txt,0
aa102.txt,1
aa103.txt,1

