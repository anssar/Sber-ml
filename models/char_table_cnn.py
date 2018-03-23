from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.constraints import maxnorm
from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adamax

from sources.char_table_source import CharTableSource
from sources.preprocessing.pipeline import Pipeline
from sources.preprocessing.util import splitter, tolower, joiner
from sources.preprocessing import loader, X, y


class CharTableCNN():
    def run(self):
        X_transformed = Pipeline((tolower, splitter, joiner)).transform(X)
        model = Sequential()
        model.add(Convolution2D(32, (len(loader.ALPHABET), 4),
        input_shape=(len(loader.ALPHABET), loader.MAX_LENGTH, 1), border_mode='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=(1, 3)))
        model.add(Convolution2D(32, 1, 3, border_mode='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=(1, 3)))
        model.add(Flatten())
        model.add(Dropout(0.7))
        model.add(Dense(256, activation='relu', W_constraint=maxnorm(3)))
        model.add(Dense(loader.NB_CLASSES, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=Adamax(), metrics=['accuracy'])
        print(model.summary())
        X_train, X_test, Y_train, Y_test = train_test_split(X_transformed, y)
        model.fit_generator(CharTableSource().get_generator(X_train, Y_train)(), steps_per_epoch=16, validation_steps=16,
                            epochs=1, verbose=1, validation_data=CharTableSource().get_generator(X_test, Y_test)())
        score = model.evaluate_generator(CharTableSource().get_generator(X_test, Y_test)(), steps=32)
        print('Test accuracy:', score[1])
        return model