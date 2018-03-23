from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from sources.word2vec_avg_source import Word2VecAvgSource
from sources.preprocessing.pipeline import Pipeline
from sources.preprocessing.util import splitter, tolower
from sources.preprocessing import loader, X, y


class Word2VecAvgSGD():
    def run(self):
        X_transformed = Pipeline((tolower, splitter)).transform(X)
        clf = GridSearchCV(SGDClassifier(), {
            'penalty': ['l1', 'l2', 'elasticnet'],
            'alpha': [0.01, 0.001, 0.0001, 0.00001, 0.000001]
        }, cv=5)
        X_train, X_test, y_train, y_test = Word2VecAvgSource().get_train_test(X_transformed, y)
        clf = clf.fit(X_train, y_train)
        print('Test accuracy:', accuracy_score(y_test, clf.predict(X_test)))
        return clf