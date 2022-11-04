from configparser import ConfigParser

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

from dataPrepare import getData
from sklearn import metrics

conf = ConfigParser()
conf.read("config.ini", encoding='UTF-8')
maxSeqLength = conf.getint("model", "maxSeqLength")
vectorLength = conf.getint("model", "vectorLength")

model = Sequential()
model.add(LSTM(128, input_shape=([maxSeqLength, vectorLength])))
model.add(Dropout(0.2))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

x, y = getData(conf.get("data", "textPath"))
model.fit(np.array(x), np_utils.to_categorical(y), epochs=20, batch_size=64)

y_pred = []
for i in model.predict(np.array(x)):
    y_pred.append(i.argmax())

print(f"accuracy_score: {metrics.accuracy_score(y, y_pred)}"
      , f"precision_score: {metrics.precision_score(y, y_pred)}"
      , f"recall_score: {metrics.recall_score(y, y_pred)}"
      )
