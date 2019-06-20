import os
import pickle
import numpy as np
import random
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

x,y=pickle.load(open('raw_data\\train_data.pickle','rb'))
x=x.reshape(-1,900)
y=y.squeeze()
print('A')
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1, random_state=0)
del x
del y
dtrain = xgb.DMatrix(train_x, label=train_y)
dtest = xgb.DMatrix(test_x, label=test_y)
del train_x
del train_y
print('B')
param = {'tree_method':'gpu_hist', 'eval_metric':'mae', 'nthread':4}
evallist = [(dtrain, 'train'), (dtest, 'eval')]
num_round = 1000

bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=10)

bst.save_model('xgb_md.model')
