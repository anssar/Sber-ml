import re

import pandas as pd


DATA_PATH = '/Users/andreyrumyantsev/PycharmProjects/ml/compliance/ml_rum/data.csv'
ENCODING = 'maccyrillic'
DELIMITER = ';'


class Loader():
    def __init__(self):
        self.NB_CLASSES = 0
        self.NB_WORDS = 0
        self.ALPHABET = []
        self.MAX_LENGTH = 0

    def load(self):
        data = pd.read_csv(DATA_PATH, encoding=ENCODING, delimiter=DELIMITER)
        X = data['текст'].tolist()
        y = data['класс'].tolist()
        data['words'] = data['текст'].apply(lambda x: re.split('[\s\,\.\(\)\!\?]', x))
        self.letters = set()
        def add_text_to_set(text):
            self.letters = self.letters.union(set(text))
        data['текст'].apply(lambda text: add_text_to_set(text))
        data['len'] = data['текст'].apply(lambda x: len(x))
        self.MAX_LENGTH = data['len'].max()
        self.ALPHABET = sorted(self.letters)
        self.NB_CLASSES = data['класс'].max()
        self.NB_WORDS = data['words'].apply(lambda x: len(x)).max()
        return X, y
