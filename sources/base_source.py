import numpy as np
from sklearn.model_selection import train_test_split


class BaseSource():
    def get_generator(self, X, y, batch_size=32):
        raise NotImplementedError

    def get_train_test(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        generator = self.get_generator(X_train, y_train, batch_size=1)
        X_train_ret = []
        y_train_ret = []
        for X_elem, y_elem in generator():
            if len(X_train_ret) == len(X_train):
                break
            X_train_ret.append(X_elem.tolist()[0])
            y_train_ret.append(y_elem.tolist()[0])
        generator = self.get_generator(X_test, y_test, batch_size=1)
        X_test_ret = []
        y_test_ret = []
        for X_elem, y_elem in generator():
            if len(X_test_ret) == len(X_test):
                break
            X_test_ret.append(X_elem.tolist()[0])
            y_test_ret.append(y_elem.tolist()[0])
        X_train_ret = np.array(X_train_ret)
        X_test_ret = np.array(X_test_ret)
        y_train_ret = np.array(y_train_ret)
        y_test_ret = np.array(y_test_ret)
        return X_train_ret, X_test_ret, y_train_ret, y_test_ret