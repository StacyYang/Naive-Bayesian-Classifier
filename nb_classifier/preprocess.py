import pandas as pd
import os
import glob
from nltk import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import re


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
    #lemmatizer = WordNetLemmatizer()  # apply lemmatization ## Finally decided to use stem, not lemma.

    for word in re.split("\W+", clean_text):
        if word not in stopwords_en and word != '':
            stem_word = stemmer.stem(word)
            token_words.append(stem_word)
            #lem_words = lemmatizer.lemmatize(word)
            #token_words.append(lem_words)

    return token_words


def pre_process():
    # set this to print df
    # setting folders and files path
    pos_folder = os.path.join(os.getcwd(), "./Movie_Reviews", "pos")
    pos_files = os.path.join(pos_folder, "*.txt")
    neg_folder = os.path.join(os.getcwd(), "./Movie_Reviews", "neg")
    neg_files = os.path.join(neg_folder, "*.txt")

    # getting a list of name for both pos and neg folder
    pos_file_names = [os.path.basename(x) for x in glob.glob(pos_files)]
    neg_file_names = [os.path.basename(x) for x in glob.glob(neg_files)]
    # setting result list
    pos_li = []
    neg_li = []

    # for each file, read lines, then push [filename(id), output(sentiment), a list of review words] into a list
    for file_name in pos_file_names:
        file_path = os.path.join(pos_folder, file_name)
        text_file = open(file_path, "r")
        data = text_file.read()
        clean_data = tokenize(data)
        pos_li.append([file_name[2:-4], 1, clean_data])
        text_file.close()

    # for each file, read line, then push [filename(id), output(sentiment), a list of review words] into a list
    for file_name in neg_file_names:
        file_path = os.path.join(neg_folder, file_name)
        text_file = open(file_path, "r")
        data = text_file.read()
        clean_data = tokenize(data)
        neg_li.append([file_name[2:-4], 0, clean_data])
        text_file.close()

    # use the list to create dataframe
    pos_df = pd.DataFrame(pos_li, columns=['id', 'sentiment', 'review'])
    neg_df = pd.DataFrame(neg_li, columns=['id', 'sentiment', 'review'])

    ##split pos/neg dataframe to 1:9, 1 for validation, 9 for training
    pos_val = pos_df.sample(frac=0.1)
    pos_train = pos_df.drop(pos_val.index)

    neg_val = neg_df.sample(frac=0.1)
    neg_train = neg_df.drop(neg_val.index)

    ##writing validation set
    # pos_val.to_csv("pos_val.csv", index=False)
    # neg_val.to_csv("neg_val.csv", index=False)

    ##writing training set
    pos_train.to_csv("pos_train.csv", index=False)
    neg_train.to_csv("neg_train.csv", index=False)

    ##validation
    val_df = pd.concat([pos_val, neg_val])
    #  val_df = val_df.drop(['id', 'sentiment'], axis=1)   # drop id and sentiment column for validation dataset.
    train_df = pd.concat([pos_train, neg_train])

    val_df.to_csv('val.csv', index=False)
    train_df.to_csv("train.csv", index=False)

    """
    print("pos_val has {} rows, neg_val has {} rows".format(len(pos_val.loc[pos_val['sentiment'] == 1]),
                                                            len(neg_val.loc[neg_val['sentiment'] == 0])))
    print("pos_train has {} rows, neg_train has {} rows".format(len(pos_train.loc[pos_train['sentiment'] == 1]),
                                                                len(neg_train.loc[neg_train['sentiment'] == 0])))
    print("val_df has {} rows, {} pos vs {} neg".format(len(val_df), len(val_df.loc[val_df['sentiment'] == 1]),
                                                        len(val_df.loc[val_df['sentiment'] == 0])))
    print("train_df has {} rows, {} pos vs {} neg".format(len(train_df), len(train_df.loc[train_df['sentiment'] == 1]),
                                                          len(train_df.loc[train_df['sentiment'] == 0])))
    print("overlapping row : {}".format(len(val_df.merge(train_df, on='id'))))
    """

# pre_process()







