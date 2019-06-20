import pickle
import time
from scipy.stats import kurtosis
from collections import deque
import numpy as np

def fftri(x):
    x=np.fft.fft(x)
    x1=np.array([n.real for n in x],dtype='float32')
    x2=np.array([n.imag for n in x],dtype='float32')
    x3=np.array([n.real for n in np.fft.ifft(x[:500])],dtype='float32')
    return x1,x2,x3

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
    x1,x2,x3=fftri(x_raw[i])
    x.append([x_raw[i].mean(),x_raw[i].max(),x_raw[i].min(),x_raw[i].std(),np.diff(x_raw[i]).mean(),kurtosis(x_raw[i]),
        x1.mean(),x1.max(),x1.min(),x1.std(),np.diff(x1).mean(),kurtosis(x1),
        x2.mean(),x2.max(),x2.min(),x2.std(),np.diff(x2).mean(),kurtosis(x2),
        x3.mean(),x3.max(),x3.min(),x3.std(),np.diff(x3).mean(),kurtosis(x3)])
    y.append([y_raw[i]])
    if(i%1000==0):
        print(str(i)+'/'+str(x_raw.shape[0])+':'+str(time.time()-start_time))

print('Stage:deleted '+str(time.time()-start_time))
del x_raw
del y_raw
print('Stage:preprocessed '+str(time.time()-start_time))
batch_x=list()
batch_y=list()
for index in range(len(x)-150+1):
    batch_x.append(x[index:index+150])
    batch_y.append(y[index+149])
    
    if(index%1000==0):
        print(str(index)+':'+str(time.time()-start_time))
print('Stage:processed '+str(time.time()-start_time))
batch_x=np.array(batch_x,dtype='float32')
batch_y=np.array(batch_y,dtype='float32')
print('Stage:converted '+str(time.time()-start_time))

with open('train_data_fft.pickle', 'wb') as file_pi:
    pickle.dump((batch_x,batch_y),file_pi,protocol=4)