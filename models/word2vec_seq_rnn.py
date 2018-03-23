from keras.layers import LSTM
from keras.models import Sequential
from keras.constraints import maxnorm
from keras.layers import Dense, Dropout, Convolution1D, MaxPooling1D
from keras.optimizers import Adamax

from sources.word2vec_seq_source import Word2VecSeqSource
from sources.preprocessing.pipeline import Pipeline
from sources.preprocessing.util import splitter, tolower
from sources.preprocessing import loader, X, y
from models import W2V_SIZE


class Word2VecSeqRNN():
    def run(self):
        X_transformed = Pipeline((tolower, splitter)).transform(X)
        model = Sequential()
        model.add(Convolution1D(filters=32, kernel_size=3, padding='same', activation='relu',
                                input_shape=(loader.NB_WORDS * W2V_SIZE, 1)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(100))
        model.add(Dropout(0.7))
        model.add(Dense(256, activation='relu', W_constraint=maxnorm(3)))
        model.add(Dense(loader.NB_CLASSES, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=Adamax(), metrics=['accuracy'])
        print(model.summary())
        X_train, X_test, y_train, y_test = Word2VecSeqSource().get_train_test(X_transformed, y)
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=64)
        score = model.evaluate(X_test, y_test, verbose=0)
        print('Test accuracy:', score[1])
        return model