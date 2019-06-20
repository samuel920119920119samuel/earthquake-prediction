import pickle
import time
from scipy.stats import kurtosis
from collections import deque
import numpy as np
from itertools import islice

def fftri(x):
    x=np.fft.fft(x)
    x1=np.array([n.real for n in x],dtype='float32')
    x2=np.array([n.imag for n in x],dtype='float32')
    x3=np.array([n.real for n in np.fft.ifft(x[:500])],dtype='float32')
    return x1,x2,x3

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
        buffer=list()
        for line in f.readlines()[1:]:#islice(f,1,150001):
            tmp.append(float(line))
        tmp=np.array(tmp,dtype='float32').reshape((-1,1000))
        for i in range(tmp.shape[0]):
            x1,x2,x3=fftri(tmp[i])
            buffer.append([tmp[i].mean(),tmp[i].max(),tmp[i].min(),tmp[i].std(),np.diff(tmp[i]).mean(),kurtosis(tmp[i]),
                x1.mean(),x1.max(),x1.min(),x1.std(),np.diff(x1).mean(),kurtosis(x1),
                x2.mean(),x2.max(),x2.min(),x2.std(),np.diff(x2).mean(),kurtosis(x2),
                x3.mean(),x3.max(),x3.min(),x3.std(),np.diff(x3).mean(),kurtosis(x3)])
    test_data.append(np.array(list(buffer),dtype='float32'))
    print(f_name+' '+str(index+1)+'/'+str(len(test_data_name))+' finish')

test_data=np.array(test_data,dtype='float32')
with open('test_name_data_fft.pickle', 'wb') as file_pi:
    pickle.dump((test_data_name,test_data),file_pi)