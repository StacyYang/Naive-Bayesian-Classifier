import pandas as pd
import os
import glob
import re
import json
import csv
from nltk import PorterStemmer
from nltk.corpus import stopwords


def tokenize(text):
    """
    :param text: movie review(string) from a .txt file
    :return: a list of important words
    """
    token_words = list()

    text = text.lower()  # turn all characters to lowercase.
    clean_text = re.sub("[^a-zA-Z]+", ' ', text)  # replace every non-alphanumeric character with a whitespace
    stopwords_en = stopwords.words("english")  # remove stop words using those defined(for example: a the this and then) in the NLTK library
    stemmer = PorterStemmer()  # apply stemming: For example: (beautiful, beauty --> beauti) (gets --> get)

    for word in re.split("\W+", clean_text):
        if word not in stopwords_en and word != '':
            stem_word = stemmer.stem(word)
            token_words.append(stem_word)

    return token_words


def token_parser(text):
    parsed = re.split(r'[^\w]', text)
    return [w for w in parsed if w != '']


def process_test_data(path):
    """
    Take the path of the folder which contains movie reviews
    :param path: the path
    :return:
    """
    review_folder = os.path.join(path)
    review_files = os.path.join(review_folder, "*.txt")

    # getting a list of name reviews
    review_file_names = [os.path.basename(x) for x in glob.glob(review_files)]

    # setting result list
    review_li = []

    # for each file, read the line, then push [filename, output, line] into a list
    for file_name in review_file_names:
        file_path = os.path.join(review_folder, file_name)
        text_file = open(file_path, "r")
        data = text_file.read()
        clean_data = tokenize(data)
        review_li.append([file_name, clean_data])
        text_file.close()

    # use the list to create dataframe
    df = pd.DataFrame(review_li, columns=['id','review'])
    # write all data to csvfile
    df.to_csv('test_set.csv', index=False)


def fetch_nb(classifier_path):
    """
    Read classifier from the json file and create the dictionary of loglikelihood.
    :return: loglikelihood dictionary
    """
    classifier = os.path.join(classifier_path)
    with open(classifier, "r") as json_file:
        mydict = json.load(json_file)
    return mydict


def get_sum_log_prob(review_wordlist, logprior, loglikelihood):
    """
    Note: prob = logprior + sum(loglikelihood), the review is positive if prob>1
    """
    # initialize probability to zero
    sum_of_log_probs = 0

    # add the logprior
    sum_of_log_probs += logprior

    for word in review_wordlist:
        # check if the word exists in the loglikelihood dictionary
        if word in loglikelihood:
            # add the log likelihood of that word to the probability
            sum_of_log_probs += loglikelihood[word]

    return sum_of_log_probs


def predict_and_output(testcsvfile, logprior, loglikelihood):
    """
    Predict the sentiment of test reviews and output the result to .csv file.
    """
    with open(testcsvfile, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        result = []
        ####
        correct_classify_num = 0
        total_test_number = 0

        for row in reader:
            wordlist = token_parser(row[1])
            prob = get_sum_log_prob(wordlist, logprior, loglikelihood)
            ######
            total_test_number += 1


            # predict: review is positive if prob>1
            if prob >= 1:
                predict_sentiment = 1
            else:
                predict_sentiment = 0
            result.append([row[0], predict_sentiment])


            #####
            if predict_sentiment == 0:
                correct_classify_num += 1

    result_df = pd.DataFrame(result)
    result_df.to_csv('output.csv', index=False, header=False)

    f.close()
    #####
    return correct_classify_num, total_test_number


def main():
    print(f"\nYour current location: {os.getcwd()}")
    print(f"\nPlease enter the path needed.\nFor example D:./reviews/test")

    #folder_path = input("Enter the path of your folder: ")
    #classifier_path = input("Enter the path of your classifier: ")
    folder_path = "./testneg"
    classifier_path = "./merge3.json"
    process_test_data(folder_path)

    loglikelihood = fetch_nb(classifier_path)
    logprior = 0

    # predict_and_output("test_set.csv", logprior, loglikelihood)


    correct_num, total_test_number = predict_and_output("test_set.csv", logprior, loglikelihood)
    print(f"correct_num is: {correct_num}")

    accuracy = correct_num/total_test_number
    print(f"accuracy is:  {accuracy}")
    print(f"total is :{total_test_number}")




main()