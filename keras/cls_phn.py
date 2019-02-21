import numpy as np
import matplotlib.pyplot as plt
import os
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential ,load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import BatchNormalization
from keras.optimizers import SGD, Adam, RMSprop, Adadelta

path_main_project='C:/Users/halleas/Google_Drive/project_root/'
main_path='C:/Users/halleas/database_project/'

X_train = np.genfromtxt(
            main_path+'output_mfcc_new.csv',           # file name
            delimiter=',',          # column delimiter
            )  
    

Y_train= np.genfromtxt(
            main_path+'output_onehot_new.csv',       # file name
            delimiter=',',          # column delimiter
            )
            
batch_size = 128
nb_epoch = 15

num_input =   X_train.shape[1]
num_ficher =  Y_train.shape[1] #number of feacher 
N =  Y_train.shape[0] #number of example    
    
model = Sequential()
model.add(Dense(512, input_shape=(num_input,)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_ficher))
model.add(Activation('softmax'))
model.summary()
 
model.compile(loss='mse',
                      optimizer=Adadelta()
                      )   

        
        
history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch)
model.save(path_main_project+'/nm_models_new/mfcc/my_nm_by_mcc_onehot_ad_mse_8vs39_matlab.h5')


#model=load_model(path_main_project+'/nm_models_new/mfcc/my_nm_by_mcc_onehot_ad_mse_8vs39_matlab.h5')
model=load_model(path_main_project+'/nm_models_new/mfcc/my_nm_by_mcc_onehot_ad_mse_8vs39_new.h5')

probability_hotmap=model.predict(X_train)
index_max_col=np.asarray(np.argmax(probability_hotmap,axis=1))

output_hotmap=np.asmatrix(np.zeros((np.shape(probability_hotmap))))
output_hotmap[np.arange(0,np.shape(probability_hotmap)[0]),index_max_col]=1

# %success rate

p=abs(np.subtract(output_hotmap,Y_train))

error=(np.sum(np.sum(p,axis=1),axis=0))/(2*np.shape(probability_hotmap)[0])
success=100*(1-error)
print(success)
