class Pipeline():
    def __init__(self, handlers):
        self.handlers = handlers

    def transform(self, X):
        for handler in self.handlers:
            X = handler(X)
        return X