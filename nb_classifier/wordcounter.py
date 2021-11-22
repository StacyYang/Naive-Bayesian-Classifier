import csv
import re


class WordCounter:

    def __init__(self, pos_filename, neg_filename):
        self.pos_file = open(pos_filename)
        self.neg_file = open(neg_filename)

        # These two dictionares are used to calculate mutual info as it counts word freq only once per review.
        # For example, "great" appears three time in one review, but we only count once.
        self.pos_dict = {}  # dictionary to map words from all positive reviews to their frequencies
        self.neg_dict = {}  # dictionary to map words from all negative reviews to their frequencies

        self.freq_dict = {}  # dictionary to map words from dataset(both pos and neg) to their frequencies.
                             # For example, {('great', 1): 3, ('bad', 0): 5}
        self.vocab = set()  # a set of unique words from dataset(both pos and neg).
        self.pos_review_num = 0  # positive review number
        self.neg_review_num = 0  # negative review number
        self.pos_word_num = 0    # total number of words for all positive reviews
        self.neg_word_num = 0  # total number of words for all negative reviews

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
        next(pos_reader)
        next(neg_reader)

        for row in pos_reader:
            wordlist = self.text_parser(row[2])
            single_review_vocab = set(wordlist)
            self.pos_review_num += 1
            for word in single_review_vocab:
                self.vocab.add(word)
                self.pos_dict[word] = self.pos_dict.get(word, 0) + 1

            for word in wordlist:
                self.freq_dict[(word, 1)] = self.freq_dict.get((word, 1), 0) + 1
                self.pos_word_num += 1

        for row in neg_reader:
            wordlist = self.text_parser(row[2])
            single_review_vocab = set(wordlist)
            self.neg_review_num += 1
            for word in single_review_vocab:
                self.vocab.add(word)
                self.neg_dict[word] = self.neg_dict.get(word, 0) + 1

            for word in wordlist:
                self.freq_dict[(word, 0)] = self.freq_dict.get((word, 0), 0) + 1
                self.neg_word_num += 1

        self.pos_file.close()
        self.neg_file.close()
