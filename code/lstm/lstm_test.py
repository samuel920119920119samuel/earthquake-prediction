import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from tensorflow.keras.models import load_model

from generator import create_X

parser = argparse.ArgumentParser()
parser.add_argument("model", help="Choose a model checkpoint for prediction",
                    type=str)
args = parser.parse_args()
model_path = 'model/' + args.model

submission = pd.read_csv('raw_data/sample_submission.csv',
                          index_col='seg_id',
                          dtype={"time_to_failure": np.float32})

# load model
model = load_model(model_path)

for i, seg_id in enumerate(tqdm(submission.index)):
    seg = pd.read_csv('raw_data/test/' + seg_id + '.csv')
    test_x = seg['acoustic_data'].to_numpy()
    submission.time_to_failure[i] = model.predict(np.expand_dims(create_X(test_x), 0))

if not os.path.isdir('submission'):
    os.mkdir('submission')
submission.to_csv('submission/lstm.csv')
print(submission.head())
