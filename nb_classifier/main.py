import json
import math
from wordcounter import WordCounter
from preprocess import pre_process
import csv
import re


def mutual_info(_fxy, _N, _fx, _fy):
    """
    Calculate mutual information.
    """
    a = _fxy * _N / (_fx * _fy)
    m_info = math.log(a, 2)
    return m_info


def pick_features(word_counter):
    """
    Calculate abs(I(A,pos) - I(A,neg)), pick words which have higher value as our features.
    """
    abs_mi = {}  # dictionary to map words to their abs(mutual info)
    for word in word_counter.vocab:
        if word_counter.pos_dict.get(word) is None:
            continue
        else:
            f_word_pos = word_counter.pos_dict.get(word)

        if word_counter.neg_dict.get(word) is None:
            continue
        else:
            f_word_neg = word_counter.neg_dict.get(word)

        num = f_word_pos / f_word_neg
        a_mi = math.log(num, 2)
        a_mi = abs(a_mi)
        abs_mi.update({word: a_mi})

    sorted_abs_mi = sorted(abs_mi.items(), key=lambda x: x[1], reverse=True)

    picked_abs_mi = [feature for feature in sorted_abs_mi if feature[1] > 0.2]

    feature_words = list()
    for feature in picked_abs_mi:
        feature_words.append(feature[0])

    return feature_words


def token_parser(text):
    parsed = re.split(r'[^\w]', text)
    return [w for w in parsed if w != '']


def train_naive_bayes(word_counter, features):
# def train_naive_bayes(word_counter):
    """
    Calculate logprior and loglikelihood.
    logprior = log(Px) = log(num of pos reviews/num of neg reviews)
    loglikelihood = log(P(Word|pos))/log(P(Word|neg))
              --->  P(Word|class) =  (word freq being pos or neg + 1) / (number of pos or neg words + number of unique words in the whole dataset.
    """
    # Decided to use picked features based on mutual information instead of using the whole vocab.
    # v = len(word_counter.vocab)  # the number of unique words in the whole dataset.
    v = len(features)
    d_pos = word_counter.pos_review_num  # the number of positive reviews
    d_neg = word_counter.neg_review_num  # the number of negative reviews

    n_pos = word_counter.pos_word_num  # the number of positive words.
    n_neg = word_counter.neg_word_num  # the number of negative words.

    logprior = math.log(d_pos / d_neg)

    loglikelihood = {}
    for word in features:
    # for word in word_counter.vocab:
        # get the positive and negative frequency of the word
        freq_pos = word_counter.freq_dict.get((word, 1), 0)
        freq_neg = word_counter.freq_dict.get((word, 0), 0)

        # calculate the probability that each word is positive, and negative
        # add the "+1" in the numerator for additive smoothing
        p_w_pos = (freq_pos + 1) / (n_pos + v)
        p_w_neg = (freq_neg + 1) / (n_neg + v)

        # calculate the log likelihood of the word
        loglikelihood[word] = math.log(p_w_pos/p_w_neg)

    return logprior, loglikelihood


def predict_naive_bayes(review_wordlist, logprior, loglikelihood):
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


def test_naive_bayes(testcsvfile, logprior, loglikelihood):
    """
    Test nb classifier, to check the accuracy and improve the model.
    """
    #print(f"review id       true_sentiment          predict_sentiment")
    with open(testcsvfile, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        correct_classify_num = 0
        total_test_number = 0
        for row in reader:
            wordlist = token_parser(row[2])
            true_sentiment = row[1]
            prob = predict_naive_bayes(wordlist, logprior, loglikelihood)
            total_test_number += 1
            # predict: review is positive if prob>1
            if prob >= 1:
                predict_sentiment = 1
            else:
                predict_sentiment = 0
            # check whether classify correctly
            if str(predict_sentiment) == true_sentiment:
                correct_classify_num += 1

            #print(f"{row[0]}           {true_sentiment}              {predict_sentiment}")

    f.close()
    return correct_classify_num, total_test_number


def main():
    pre_process()

    train_nb = WordCounter("./pos_train.csv", "./neg_train.csv")
    train_nb.prepare_data()

    features = pick_features(train_nb)
    logprior, loglikelihood = train_naive_bayes(train_nb, features)
    # logprior, loglikelihood = train_naive_bayes(train_nb)
    testfile = "val.csv"
    correct_num, total_test_number = test_naive_bayes(testfile, logprior, loglikelihood)
    print(f"correct_num is: {correct_num}")
    accuracy = correct_num/total_test_number
    print(f"accuracy is:  {accuracy}")
    print(f"len of features   {len(features)}")
    print(f"len of vocab    {len(train_nb.vocab)}")
    print(f"len of loglikelihood     {len(loglikelihood)}")

    # save loglikelihood(nb classifier) into a json file
    with open('ex.json', "w") as outfile:
        json.dump(loglikelihood, outfile)


if __name__ == '__main__':
    main()


