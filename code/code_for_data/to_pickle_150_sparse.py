import pickle
import time
from scipy.stats import kurtosis
from collections import deque
import numpy as np

path='train_data_raw.pickle'

start_time=time.time()
print('Stage:init '+str(time.time()-start_time))
x_raw,y_raw=pickle.load(open(path,'rb'))
print('Stage:loaded '+str(time.time()-start_time))
x_raw=np.array(x_raw).reshape((-1,1000))
y_raw=np.array(y_raw)
x=list()
y=list()
print('Stage:reshaped '+str(time.time()-start_time))
for i in range(x_raw.shape[0]):
    x.append([x_raw[i].mean(),x_raw[i].max(),x_raw[i].min(),x_raw[i].std(),np.diff(x_raw[i]).mean(),kurtosis(x_raw[i])])
    y.append([y_raw[i]])

print('Stage:deleted '+str(time.time()-start_time))
del x_raw
del y_raw
print('Stage:preprocessed '+str(time.time()-start_time))
batch_x=list()
batch_y=list()
full_len=(len(x)//150)*150
for index in range(0,full_len-150+1,15):
    batch_x.append(x[index:index+150])
    batch_y.append(y[index+149])
    
    if(index%1000==0):
        print(str(index)+':'+str(time.time()-start_time))
print('Stage:processed '+str(time.time()-start_time))
batch_x=np.array(batch_x,dtype='float32')
batch_y=np.array(batch_y,dtype='float32')
print('Stage:converted '+str(time.time()-start_time))
with open('train_data_sparse.pickle', 'wb') as file_pi:
    pickle.dump((batch_x,batch_y),file_pi)