import pickle
import time
from scipy.stats import kurtosis
from collections import deque
import numpy as np
from itertools import islice

def fftri(x):
    y=np.fft.fft(x)
    return y.real,y.imag

def feature(x):
    return [*x.mean(axis=-1),*x.max(axis=-1),*x.min(axis=-1),*x.std(axis=-1),*np.diff(x,axis=-1).mean(axis=-1),*kurtosis(x,axis=-1)]

def full_feature(x):
    #print(x.shape)
    x1,x2=fftri(x)
    return [*feature(x),*feature(x1),*feature(x2)]

start_time=time.time()

test_data_prefix='test\\'
test_data_name=list()
test_data_suffix='.csv'
with open('sample_submission.csv') as f:
    for line in f.readlines()[1:]:#islice(f,1,2624):
        test_data_name.append(line.split(',')[0])

#from IPython import embed
#embed()
test_data=list()
tmp=list()
for index,f_name in enumerate(test_data_name):
    with open(test_data_prefix+f_name+test_data_suffix) as f:
        tmp=list()
        #buffer=list()
        for line in f.readlines()[1:]:#islice(f,1,150001):
            tmp.append(float(line))
        x=np.array(tmp,dtype='float32').reshape(1,150000)
        x1=x.reshape(-1,1000)#150*1000
        x2=x.reshape(-1,5000)#30*5000
        x3=x.reshape(-1,10000)#15*10000
        x4=x.reshape(-1,50000)#3*50000
        x5=x.reshape(-1,150000)#1*150000
        buffer=[*full_feature(x1),*full_feature(x2),*full_feature(x3),*full_feature(x4),*full_feature(x5)]#[None,(199*18)]
    test_data.append(np.array(buffer,dtype='float32'))
    print(f_name+' '+str(index+1)+'/'+str(len(test_data_name))+' finish')

test_data=np.array(test_data,dtype='float32')
print(test_data.shape)
with open('test_name_data_fft_sparse_final.pickle', 'wb') as file_pi:
    pickle.dump((test_data_name,test_data),file_pi)