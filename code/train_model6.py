import os
import pickle
from keras import Sequential
from keras.layers import Dense,Flatten,Reshape,ReLU,CuDNNGRU
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import numpy as np
import random
import tensorflow as tf

random.seed(7951)

#reduce gram usage
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.keras.backend.set_session(sess)

import gc
gc.disable()
train_x,train_y=pickle.load(open('raw_data\\train_data_fft.pickle','rb'))
gc.enable()

model = Sequential()
model.add(CuDNNGRU(50,return_sequences=True,input_shape=(150, 24)))
model.add(CuDNNGRU(25,return_sequences=True))
model.add(Flatten())
model.add(Dense(10))
model.add(ReLU())
model.add(Dense(1))

model.compile(loss='mean_squared_logarithmic_error',optimizer=Adam(lr=0.0001),metrics=['mae'])
#model.compile(loss='mae',optimizer=Adam(lr=0.0001))#,metrics=['mae'])
model.summary()
history=model.fit(train_x,train_y,epochs=50,batch_size=10000,validation_split=0.1,callbacks=[ModelCheckpoint('md_tmp\\md_tmp_{epoch:03d}.h5',monitor='val_categorical_accuracy', period=1)])
#history=model.fit_generator(gen_util.data_gen(batch_size=1000),epochs=10,steps_per_epoch=100000000/(1000*100))#,validation_split=0.1)
model.save('md.h5')
with open('hist.pickle', 'wb') as file_pi:
    pickle.dump(history.history,file_pi)