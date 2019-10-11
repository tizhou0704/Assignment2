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
        # combine contents of files under this dir
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
    # checking the token is one of '.' '?' '!' or '...', it's an end of sentence
    res = text.count(".") + text.count('?') + text.count('!') + text.count('...')
    return res


def most_frequent_content_words(text):
    fdist = FreqDist(text)
    list1 = fdist.most_common()

    # remove stop words
    # remove punctuations and 's
    filtered = [(w, count) for (w, count) in list1 if w not in stop_words and w not in english_punctuations]
    # get the most 25 frequent content words
    return filtered[:25]


def most_frequent_bigrams(text):
    list1 = text[:]

    # remove stop words
    # remove punctuations and 's
    list_new = [i for i in list1 if i not in stop_words and i not in english_punctuations]

    blist = list(nltk.bigrams(list_new))
    fdist = FreqDist(blist)
    # get the most 25 frequent biagrams
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

    def kwic(self, word):
        return self.text.concordance(word)


categories = ('adventure', 'fiction', 'government', 'humor', 'news')


def compare_to_brown(text):
    genre_arr = []
    for i in range(0, len(categories)):
        genre = categories[i]
        # get all words in this category
        genre_text = brown.words(categories=genre)
        # convert to set format and eliminate duplicated words
        genre_set = set(genre_text)
        # save words to array except stop words and punctuations
        # genre_arr.append({i for i in genre_set if i not in stop_words and i not in english_punctuations})
        genre_arr.append(genre_set)

    for i in range(0, len(categories)):
        genre = categories[i]
        # get the set of each category
        genre_text = brown.words(categories=genre)

        s = set(text[:])
        s_new = {i for i in s if i not in stop_words and i not in english_punctuations}

        # too many words, only calculate common words
        common = s_new & genre_arr[i]

        # convert count numbers to numpy array
        vector1 = get_frequncy_np_array(genre_text, common)
        vector2 = get_frequncy_np_array(text, common)

        # build cos similarity model
        cos = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * (np.linalg.norm(vector2)))
        print(genre,': ',round(cos, 2))

        # running time is very long
        # this is the running result for grail.txt:
        # adventure :  0.64
        # fiction :  0.62
        # government :  0.38
        # humor :  0.74
        # news :  0.37



def get_frequncy_np_array(text, common):
    flist = []
    for word in common:
        # add each word's count to the list
        flist.append(text.count(word))
    # convert to numpy array
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
