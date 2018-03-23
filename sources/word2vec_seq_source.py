import numpy as np

from sources.base_source import BaseSource
from sources.preprocessing.util import y_to_vec
from sources.preprocessing import loader
from models import modelw2v, W2V_SIZE

class Word2VecSeqSource(BaseSource):
    def get_generator(self, X, y, batch_size=32):
        def generator():
            idx = 0
            batch_x = []
            batch_y = []
            while True:
                x_current = X[idx]
                y_current = y[idx]
                y_vec = y_to_vec(y_current)
                x_vec = []
                for word in x_current:
                    if word in modelw2v:
                        x_vec.extend(modelw2v[word])
                    else:
                        x_vec.extend([0] * W2V_SIZE)
                while len(x_vec) < W2V_SIZE * loader.NB_WORDS:
                    x_vec.extend([0] * W2V_SIZE)
                x_vec = [[t] for t in x_vec]
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