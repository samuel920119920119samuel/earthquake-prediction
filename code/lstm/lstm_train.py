import shutil
import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, LSTM, Flatten
from tensorflow.keras.layers import Dense, CuDNNGRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from generator import gen

# use all time series data 104677355

n_features = 20

# delete old checkpoints and logs
shutil.rmtree('logs')
shutil.rmtree('model')
os.mkdir('logs')
os.mkdir('model')

# argument
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--partial", help="Use only a part of dataset", action="store_true")
args = parser.parse_args()
if args.partial:
    print("Reading 30% of dataset...")
    float_data = pd.read_csv("raw_data/train.csv", \
                             dtype={"acoustic_data": np.float32, "time_to_failure": np.float32}, \
                             nrows=204677355).to_numpy()
else:
    print("Reading full dataset...")
    float_data = pd.read_csv("raw_data/train.csv", \
                             dtype={"acoustic_data": np.float32, "time_to_failure": np.float32}).to_numpy()

print('read raw data complete')
train_gen = gen(float_data) # samples, targets
valid_gen = gen(float_data, max_index=50085877) # second

callbacks = [
    ModelCheckpoint("model/lstm-{epoch:02d}-{val_loss:.2f}.hdf5"),
    TensorBoard(log_dir='logs')
]

model = Sequential()
# model.add(LSTM(150, input_shape=(150, n_features), return_sequences=True, dropout=0.2))
model.add(CuDNNGRU(48, input_shape=(None, n_features)))
model.add(Dense(10, activation='relu'))
# model.add(Flatten())
model.add(Dense(1))
model.summary()

model.compile(optimizer=Adam(lr=0.0005), loss="mae")
history = model.fit_generator(train_gen,
                              steps_per_epoch=1000,
                              epochs=30,
                              verbose=1,
                              callbacks=callbacks,
                              validation_data=valid_gen,
                              validation_steps=200)
