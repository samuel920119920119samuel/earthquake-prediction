import pickle
import time
from scipy.stats import kurtosis
from collections import deque
import numpy as np

def fftri(x):
    y=np.fft.fft(x)
    return y.real,y.imag

def feature(x):
    return [*x.mean(axis=-1),*x.max(axis=-1),*x.min(axis=-1),*x.std(axis=-1),*np.diff(x,axis=-1).mean(axis=-1),*kurtosis(x,axis=-1)]

def full_feature(x):
    #print(x.shape)
    x1,x2=fftri(x)
    return [*feature(x),*feature(x1),*feature(x2)]

path='train_data_raw.pickle'

start_time=time.time()
print('Stage:init '+str(time.time()-start_time))
x_raw,y_raw=pickle.load(open(path,'rb'))
print('Stage:loaded '+str(time.time()-start_time))
x_raw=np.array(x_raw)#.reshape((-1,150000))
x_raw.resize(x_raw.shape[0]//150000,150000)
#x_raw=np.array(x_raw)
#x_raw.resize((x_raw.shape[0]//1000,1000))
#y_raw=np.array(y_raw)
#y_raw.resize(y_raw.shape[0],1)
x=list()
y=list()
print('Stage:reshaped '+str(time.time()-start_time))
for i in range(x_raw.shape[0]):
    x1=x_raw[i].reshape(-1,1000)#150*1000
    x2=x_raw[i].reshape(-1,5000)#30*5000
    x3=x_raw[i].reshape(-1,10000)#15*10000
    x4=x_raw[i].reshape(-1,50000)#3*50000
    x5=x_raw[i].reshape(-1,150000)#1*150000
    x.append([*full_feature(x1),*full_feature(x2),*full_feature(x3),*full_feature(x4),*full_feature(x5)])#[None,(199*18)]
    y.append([y_raw[i*150+149]])
    if(i%100==0):
        print(str(i)+'/'+str(x_raw.shape[0])+':'+str(time.time()-start_time))

#from IPython import embed
#embed()

print('Stage:deleted '+str(time.time()-start_time))
del x_raw
del y_raw

print('Stage:processed '+str(time.time()-start_time))
batch_x=np.array(x,dtype='float32')
batch_y=np.array(y,dtype='float32')

#print('Stage:preprocessed '+str(time.time()-start_time))
#batch_x.resize((batch_x.shape[0]//150,150,24))
#batch_y.resize((batch_x.shape[0]//150,150,1))

print(batch_x.shape)
print(batch_y.shape)

print('Stage:converted '+str(time.time()-start_time))
with open('train_data_fft_sparse_final.pickle', 'wb') as file_pi:
    pickle.dump((batch_x,batch_y),file_pi,protocol=4)