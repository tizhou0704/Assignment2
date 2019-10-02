"""main.py

Code scaffolding

"""

import os
import nltk
from nltk.corpus import brown
from nltk.corpus import wordnet as wn
from nltk.corpus import PlaintextCorpusReader
from nltk.probability import FreqDist
from nltk.text import Text
from nltk.corpus import stopwords
import numpy as np

stop_words = set(stopwords.words('english'))
english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '--', "''", '``', '\'s']


def read_text(path):
    raw = ""

    # if file
    if os.path.isfile(path):
        f = open(path)
        raw = f.read()
    # if directory
    elif os.path.isdir(path):
        files = os.listdir(path)
        files.sort()
        for file in files:
            f = open(os.path.join(path, file))
            raw = raw + f.read()
    tokens = nltk.word_tokenize(raw)
    text = nltk.Text(tokens)
    return text


def token_count(text):
    res = len(text)
    return res


def type_count(text):
    s = set(text)
    # use lower case
    s_new = {w.lower() for w in s}
    return len(s_new)


def sentence_count(text):
    res = text.count(".")
    return res


def most_frequent_content_words(text):
    fdist = FreqDist(text)
    list1 = fdist.most_common()

    # remove stop words
    # remove punctuations and 's
    filtered = [(w, count) for (w, count) in list1 if w not in stop_words and w not in english_punctuations]
    return filtered[:25]


def most_frequent_bigrams(text):
    list1 = text[:]

    # remove stop words
    # remove punctuations and 's
    list_new = [i for i in list1 if i not in stop_words and i not in english_punctuations]

    blist = list(nltk.bigrams(list_new))
    fdist = FreqDist(blist)
    return fdist.most_common(25)


class Vocabulary():

    def __init__(self, text):
        self.text = text

    def frequency(self, word):
        return self.text.count(word)

    def pos(self, word):
        syn = wn.synsets(word)[0]
        return syn.pos()

    def gloss(self, word):
        syn = wn.synsets(word)[0]
        return syn.definition()

    def quick(self, word):
        return self.text.concordance(word)


categories = ('adventure', 'fiction', 'government', 'humor', 'news')


def compare_to_brown(text):
    genre_arr = []
    for i in range(0, len(categories)):
        genre = categories[i]
        # get all words in this category
        genre_text = brown.words(categories=genre)
        genre_set = set(genre_text)
        # save words to array except stop words and punctuations
        genre_arr.append({i for i in genre_set if i not in stop_words and i not in english_punctuations})

    for i in range(0, len(categories)):
        genre = categories[i]
        genre_text = brown.words(categories=genre)
        # too many words, only calculate common words
        common = set(text[:]) & genre_arr[i]
        vector1 = get_frequncy_np_array(genre_text, common)
        vector2 = get_frequncy_np_array(text, common)
        # build cos similarity model
        op = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * (np.linalg.norm(vector2)))
        print(genre, ' ', round(op, 2))


def get_frequncy_np_array(text, common):
    flist = []
    for word in common:
        flist.append(text.count(word))
    return np.array(flist)


if __name__ == '__main__':

    # text = read_text('data/emma.txt')
    # token_count(text)
    # print(type_count(text))
    # sentence_count(text)
    # print(most_frequent_content_words(text))
    # print(type_count(text))
    # print(most_frequent_bigrams(text))
    # vocab = Vocabulary(read_text('data/grail.txt'))
    # print(vocab.frequency('swallow'))
    # print(vocab.pos('swallow'))
    # print(vocab.gloss('swallow'))
    grail = read_text('data/grail.txt')
    compare_to_brown(grail)
