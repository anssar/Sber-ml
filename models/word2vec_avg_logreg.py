from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from sources.word2vec_avg_source import Word2VecAvgSource
from sources.preprocessing.pipeline import Pipeline
from sources.preprocessing.util import splitter, tolower
from sources.preprocessing import loader, X, y


class Word2VecAvgLogReg():
    def run(self):
        X_transformed = Pipeline((tolower, splitter)).transform(X)
        clf = GridSearchCV(LogisticRegression(multi_class='multinomial', solver='lbfgs'), {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
        }, cv=5)
        X_train, X_test, y_train, y_test = Word2VecAvgSource().get_train_test(X_transformed, y)
        clf = clf.fit(X_train, y_train)
        print('Test accuracy:', accuracy_score(y_test, clf.predict(X_test)))
        return clf