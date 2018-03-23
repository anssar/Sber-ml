import numpy as np

from sources.base_source import BaseSource
from sources.preprocessing.util import y_to_vec
from sources.preprocessing import loader


class CharTableSource(BaseSource):
    def get_generator(self, X, y, batch_size=32):
        def generator():
            idx = 0
            batch_x = []
            batch_y = []
            while True:
                x_current = X[idx]
                y_current = y[idx]
                y_vec = y_to_vec(y_current)
                x_table_t = []
                for letter in x_current:
                    x_table_t.append([1 if letter == x else 0 for x in loader.ALPHABET])
                while len(x_table_t) < loader.MAX_LENGTH:
                    x_table_t.append([0] * len(loader.ALPHABET))
                x_table = []
                for _ in loader.ALPHABET:
                    x_table.append([])
                for letter in x_table_t:
                    for idx, value in enumerate(letter):
                        x_table[idx].append([value])
                batch_x.append(x_table)
                batch_y.append(y_vec)
                if len(batch_x) == batch_size:
                    yield np.array(batch_x), np.array(batch_y)
                    batch_x = []
                    batch_y = []
                idx += 1
                if idx == len(X):
                    idx = 0
        return generator