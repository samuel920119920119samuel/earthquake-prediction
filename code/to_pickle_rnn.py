import pickle
import numpy as np

x,y=pickle.load(open('train_data_raw.pickle','rb'))

x=np.array(x,dtype='float32')
x=x.reshape([-1,1000])
x=np.expand_dims(x,axis=2)
y=np.array(y,dtype='float32')

with open('train_data_rnn.pickle', 'wb') as file_pi:
    pickle.dump((x,y),file_pi)