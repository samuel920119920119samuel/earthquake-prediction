import numpy as np
from scipy.stats import kurtosis
from collections import deque
import pickle

def compute_features(x):
    tmp=np.array(x)
    return [tmp.mean(),tmp.max(),tmp.min(),tmp.std(),np.diff(tmp).mean(),kurtosis(x)]
    #return [0,1]
    pass

def data_gen(path='raw_data\\train_data.pickle', window_size=1000, output_len=150, step=1):#will output ([output_len,n_feature],y)
    x_raw,y_raw=pickle.load(open(path,'rb'))
    row=deque(maxlen=output_len)
    current=list()
    features=list()
    for start,end in zip(range(len(x_raw)),range(window_size,len(x_raw)+window_size)):
        current=x_raw[start:end]
        target=y_raw[end]
        features=compute_features(current)
        row.append(features)
        if(len(row)>=output_len):
            yield np.array(row,dtype='float32'),target

#a=data_gen()
#from IPython import embed
#embed()
#next(a)
#import time
#tt=time.time()
#for i in range(100000000):
#    next(a)
#    if(i%1000==999):
#        print(time.time()-tt)