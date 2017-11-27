import numpy
import math
import pandas as pd
from keras import optimizers
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import LSTM
from keras.layers import TimeDistributed, Bidirectional
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers.convolutional import MaxPooling1D, MaxPooling2D, AveragePooling1D
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

def step_decay(epoch):
    initial_lrate = 0.002
    drop = 0.5
    epochs_drop = 5.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate
lrate = LearningRateScheduler(step_decay)

# fix random seed for reproducibility
numpy.random.seed(7)


triPosCsv = pd.read_csv('convDiPos.csv', header=None)
triNegCsv = pd.read_csv('convDiNeg.csv', header=None)

labels = pd.DataFrame([1 for x in range(len(triPosCsv))] + [0 for x in range(len(triNegCsv))] )
data = pd.concat([triPosCsv, triNegCsv])

assert len(labels)==len(data)

embedding_vector_length = 64
numsamples = len(data)
max_sample_length = len(data.columns)



kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1234)
kf.get_n_splits(data, labels)

cvscores = []

earlyStopping=EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', epsilon=0.001, cooldown=0, min_lr=0)

for train_index, test_index in kf.split(data, labels):
    X_train = data.iloc[train_index,:].values
    X_train = X_train[:, 1::3]
    max_sample_length = len(X_train[1,:])
    y_train = labels.iloc[train_index,:].values

    X_test = data.iloc[test_index,:].values
    X_test = X_test[:, 1::3]
    y_test = labels.iloc[test_index,:].values

    # create the model
    model = Sequential()

    model.add(Embedding(numsamples, embedding_vector_length, input_length=max_sample_length))

    model.add(Conv1D(filters=100, kernel_size=9, padding='same', activation='relu', strides=1))
    # model.add(AveragePooling1D(pool_size=3, strides=1))
    model.add(MaxPooling1D(pool_size=3, strides=3))

    # model.add(Conv1D(filters=250, kernel_size=17, padding='same', activation='relu', strides=1))
    # model.add(MaxPooling1D(pool_size=2, strides=2))


    model.add(Bidirectional(LSTM(32, dropout=0.3, recurrent_dropout=0.2, return_sequences=True)))
    model.add(Bidirectional(LSTM(16, dropout=0.1, recurrent_dropout=0.1, return_sequences=True)))
    model.add(LSTM(  8, dropout=0.1, recurrent_dropout=0.1))
    # model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))

    model.add(Dense(128, activation='sigmoid'))

    # model.add(Activation('softmax'))
    model.add(Dense(1, activation='sigmoid'))

    opt = []
    opt .append(optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0))
    opt .append(optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004))


    model.compile(loss='binary_crossentropy', optimizer=opt[1], metrics=['accuracy'])
    print(model.summary())
    history = model.fit(X_train, y_train, validation_split=0.33, epochs=100, batch_size=64, callbacks=[earlyStopping,lrate])

    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    cvscores.append(scores[1] * 100)
    print scores

    print '='*30

print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))

print('END')