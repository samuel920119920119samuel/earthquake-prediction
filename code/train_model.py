import os
import pickle
from glob import glob
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import Sequential
from keras.layers import Conv1D,MaxPooling1D,Dense,Flatten,Reshape,Activation,Dropout,PReLU,BatchNormalization,ReLU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import LeakyReLU
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import gen_util

#reduce gram usage
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.keras.backend.set_session(sess)

train_x,train_y=pickle.load(open('raw_data\\train_data.pickle','rb'))

model = Sequential()
model.add(Conv1D(64, 3, input_shape=(150, 6)))
model.add(ReLU())
model.add(MaxPooling1D(pool_size=3))
model.add(Conv1D(32, 3))
model.add(ReLU())
model.add(Flatten())
#model.add(Dropout(0.5))
model.add(Dense(100))
model.add(ReLU())
#
#model.add(Flatten())
#model.add(Dense(100, input_shape=[900]))
#model.add(ReLU())
#model.add(Dense(100))
#model.add(ReLU())
model.add(Dense(1))

model.compile(loss='mae',optimizer='adam')#,metrics=['mse'])
model.summary()
history=model.fit(train_x,train_y,epochs=100,batch_size=1000,validation_split=0.1)
#history=model.fit_generator(gen_util.data_gen(batch_size=1000),epochs=10,steps_per_epoch=100000000/(1000*100))#,validation_split=0.1)
model.save('md_tmp.h5')
with open('hist_tmp.pickle', 'wb') as file_pi:
    pickle.dump(history.history,file_pi)