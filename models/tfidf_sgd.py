from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sources.preprocessing.pipeline import Pipeline
from sources.preprocessing.util import splitter, tolower, stemmer, joiner
from sources.preprocessing import loader, X, y


class TfIdfSGD():
    def run(self):
        X_transformed = Pipeline((tolower, splitter, stemmer, joiner)).transform(X)
        clf = GridSearchCV(SGDClassifier(), {
            'penalty': ['l1', 'l2', 'elasticnet'],
            'alpha': [0.01, 0.001, 0.0001, 0.00001, 0.000001]
        }, cv=5)
        tf = TfidfVectorizer().fit_transform(X_transformed)
        X_train, X_test, y_train, y_test = train_test_split(tf, y)
        clf = clf.fit(X_train, y_train)
        print('Test accuracy:', accuracy_score(y_test, clf.predict(X_test)))
        return clf