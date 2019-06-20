import os
import pickle
import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

x,y=pickle.load(open('raw_data\\train_data_sparse.pickle','rb'))
x=x.reshape(-1,900)
y=y.squeeze()

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1, random_state=0)

parameters = {'n_estimators': 1000,
              'n_jobs':-1,
              'verbose':1}
print('A')
RF_model = RandomForestRegressor(**parameters)
print('B')
RF_model.fit(train_x,train_y)
print('C')

RF_predictions = RF_model.predict(test_x)
score = mean_absolute_error(test_y ,RF_predictions)
print(score)

with open('rf_md.pickle', 'wb') as file_pi:
    pickle.dump(RF_model,file_pi)