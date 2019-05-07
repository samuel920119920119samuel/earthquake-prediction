import os
import pickle
#from keras.models import load_model
import numpy as np
#import random
#import tensorflow as tf

#reduce gram usage
#gpu_options = tf.GPUOptions(allow_growth=True)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#tf.keras.backend.set_session(sess)

test_data_name,test_data=pickle.load(open('raw_data\\test_name_data.pickle','rb'))
test_data=test_data.reshape(-1,900)

model=pickle.load(open('rf_md.pickle', 'rb'))
pred=model.predict(test_data)

#from IPython import embed
#embed()
with open('output.csv','w',encoding='utf-8') as f:
    f.write('seg_id,time_to_failure\n')
    for f_name,y in zip(test_data_name,pred):
        #f.write(f_name+','+str(y[0])+'\n')
        f.write(f_name+','+str('{0:.16f}'.format(y))+'\n')