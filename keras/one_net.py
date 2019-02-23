

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import os
np.random.seed(1337)  # for reproducibility

print('start1')

from keras.models import Sequential ,load_model
from keras.layers import Dense, Dropout, Activation,BatchNormalization
from keras.optimizers import SGD, Adam, RMSprop ,Adadelta
from keras.utils import np_utils


main_path='C:/Users/halleas/database_project/single_net_output'

        
Y_train= np.genfromtxt(
   main_path+'/map_train_output_new.csv',       # file name
   delimiter=',',          # column delimiter
    )
        
X_train = np.genfromtxt(
   main_path+'/Z_train_output_new.csv',         # file name
          # file name
    delimiter=',',          # column delimiter
    )   

batch_size = 128
nb_epoch = 15
    
num_input = X_train.shape[1]
num_ficher =  Y_train.shape[1] #number of feacher 
N =  Y_train.shape[0] #number of example    
        
model = Sequential()
model.add(Dense(512, input_shape=(num_input,)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_ficher))
model.add(Activation('sigmoid'))
model.summary()

model.compile(loss='mse',
                  optimizer=Adadelta()
                  )
    # the data, shuffled and split between train and test sets
    
print(X_train.shape[0], 'train samples')
    # print(X_test.shape[0], 'test samples')
    
    # convert class vectors to binary class matrices
    
history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch)



model.save('C:/Users/halleas/Google_Drive/project_root/nm_models_new/my_NM_single_net_oppsite.h5')
model.save('my_NM_single_net.h5')

print('end')

