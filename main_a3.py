"""main_3a.py

An instance of the Text class should be initialized with a file path (a file or
directory). The example here uses object as the super class, but you may use
nltk.text.Text as the super class.

An instance of the Vocabulary class should be initialized with an instance of
Text (not nltk.text.Text).

"""

import os
import nltk
import re
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn

stop_words = set(stopwords.words('english'))
english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '--', "''", '``', '\'s']


class Text(object):

    def __init__(self, path):
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

        self.raw = raw
        self.tokens = tokens

    def token_count(self):
        return len(self.tokens)

    def type_count(self):
        s = set(self.tokens)
        # use lower case
        s_new = {w.lower() for w in s}
        return len(s_new)

    def sentence_count(self):
        # checking the token is one of '.' '?' '!' or '...', it's an end of sentence
        res = self.tokens.count(".") + self.tokens.count('?') + self.tokens.count('!') + self.tokens.count('...')
        return res

    def most_frequent_content_words(self):
        fdist = nltk.FreqDist(self.tokens)
        list1 = fdist.most_common()

        # remove stop words
        # remove punctuations and 's
        filtered = [(w, count) for (w, count) in list1 if w.lower() not in stop_words and w not in english_punctuations]
        # get the most 25 frequent content words
        return filtered[:25]

    def most_frequent_bigrams(self):
        # remove stop words
        # remove punctuations and 's
        list_new = [i for i in self.tokens if i.lower() not in stop_words and i not in english_punctuations]

        blist = list(nltk.bigrams(list_new))
        fdist = nltk.FreqDist(blist)

        # get the most 25 frequent biagrams
        return fdist.most_common(25)

    def find_sirs(self):
        p = re.compile(r"Sir [\w|-]+")
        s_list = list(set(p.findall(self.raw)))
        s_list.sort()
        return s_list

    def find_brackets(self):
        b_list1 = re.findall(r"[(].*?[)]", self.raw)
        b_list2 = re.findall(r"[\[].*?[\]]", self.raw)
        return b_list1 + b_list2

    def find_roles(self):

        # find the text before ':' at the beginning of each line
        r_list = re.findall(r"^(.*?): ", self.raw, re.MULTILINE)

        # eliminate SCENE and remove duplicated roles
        list_new = list(set([r for r in r_list if (len(r) > 5 and r[:5] != 'SCENE') or len(r) <= 5]))
        list_new.sort()
        return list_new

    def find_repeated_words(self):

        r_list = self.tokens
        count = 0            # record count numbers
        str = ''
        res = []
        # use Math methods
        for index in range(len(r_list)):
            if (index == 0) :
                count = count + 1
                str = str + r_list[index] + ' '
                continue
            if (r_list[index] == r_list[index-1]):
                count = count + 1
                str = str + r_list[index] + ' '
            else:
                if (count >= 3):
                    str = str[:len(str)-1]
                    res.append(str)
                str = r_list[index] + ' '
                count = 1
        res = list(set(res))
        res.sort()
        return res


class Vocabulary(object):

    def __init__(self, text):
        self.raw = text.raw
        self.tokens = text.tokens

    def frequency(self, word):
        return self.tokens.count(word)

    def pos(self, word):
        syn = wn.synsets(word)[0]
        return syn.pos()

    def gloss(self, word):
        syn = wn.synsets(word)[0]
        return syn.definition()

    def kwic(self, word):
        txt = nltk.Text(self.tokens)
        return txt.concordance(word)


if __name__ == '__main__':

    grail = Text('data/grail.txt')
    print(grail.find_repeated_words())
