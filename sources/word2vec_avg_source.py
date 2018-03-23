import numpy as np

from sources.base_source import BaseSource
from models import modelw2v, W2V_SIZE

class Word2VecAvgSource(BaseSource):
    def get_generator(self, X, y, batch_size=32):
        def generator():
            idx = 0
            batch_x = []
            batch_y = []
            while True:
                x_current = X[idx]
                y_current = y[idx]
                y_vec = y_current
                x_vec = np.zeros(shape=(W2V_SIZE,))
                word_cnt = 0
                for word in x_current:
                    if word in modelw2v:
                        x_vec += modelw2v[word]
                        word_cnt += 1
                if word_cnt != 0:
                    x_vec /= word_cnt
                batch_x.append(x_vec)
                batch_y.append(y_vec)
                if len(batch_x) == batch_size:
                    yield np.array(batch_x), np.array(batch_y)
                    batch_x = []
                    batch_y = []
                idx += 1
                if idx == len(X):
                    idx = 0
        return generator