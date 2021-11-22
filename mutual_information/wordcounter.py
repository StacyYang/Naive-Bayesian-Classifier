import csv
import re


class WordCounter:

    def __init__(self, pos_filename, neg_filename):
        self.pos_file = open(pos_filename)
        self.neg_file = open(neg_filename)
        self.pos_dict = {}  # dictionary to map words from all positive reviews to their frequencies
        self.neg_dict = {}  # dictionary to map words from all negative reviews to their frequencies
        self.freq_dict = {}  # dictionary to map words from dataset(both pos and neg) to their frequencies
        self.vocab = set()  # a set of unique words from dataset(both pos and neg).
        self.pos_vocab = set()  # a set of unique words from positive reviews.
        self.neg_vocab = set()  # a set of unique words from negative reviews.
        self.pos_review_num = 0  # positive review number
        self.neg_review_num = 0  # negative review number

    @staticmethod
    def text_parser(text):
        parsed = re.split(r'[^\w]', text)
        return [w for w in parsed if w != '']

    def prepare_data(self):
        """
        Use csv.reader to read files.  Row[2] corresponds to column['review'].
        Create vocab set for whole dataset, positive reviews and negative reviews respectively.
        Count positive review number and negative review number.
        Create dictionaries mapping word to its frequency.
        """
        pos_reader = csv.reader(self.pos_file)
        neg_reader = csv.reader(self.neg_file)
        ## skipping header
        next(pos_reader)
        next(neg_reader)

        for row in pos_reader:
            wordlist = self.text_parser(row[2])
            single_review_vocab = set(wordlist)
            self.pos_review_num += 1
            for word in single_review_vocab:
                self.vocab.add(word)
                self.pos_vocab.add(word)
                self.pos_dict[word] = self.pos_dict.get(word, 0) + 1
                self.freq_dict[word] = self.freq_dict.get(word, 0) + 1

        for row in neg_reader:
            wordlist = self.text_parser(row[2])
            single_review_vocab = set(wordlist)
            self.neg_review_num += 1
            for word in single_review_vocab:
                self.vocab.add(word)
                self.neg_vocab.add(word)
                self.neg_dict[word] = self.neg_dict.get(word, 0) + 1
                self.freq_dict[word] = self.freq_dict.get(word, 0) + 1






