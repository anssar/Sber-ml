import re

import snowballstemmer

from sources.preprocessing import loader


def splitter(X):
    ret = []
    for x in X:
        retx = re.split('[\s\,\.\(\)\!\?]', x)
        ret.append(retx)
    return ret


def stemmer(X):
    stemmer_obj = snowballstemmer.RussianStemmer()
    ret = []
    for x in X:
        retx = []
        for word in x:
            retxx = stemmer_obj.stemWord(word)
            retx.append(retxx)
        ret.append(retx)
    return ret


def joiner(X):
    ret = []
    for x in X:
        retx = ' '.join(x)
        ret.append(retx)
    return ret


def tolower(X):
    ret = []
    for x in X:
        retx = x.lower()
        ret.append(retx)
    return ret


def y_to_vec(y):
    vec = [0 for _ in range(loader.NB_CLASSES)]
    vec[y - 1] = 1
    return vec