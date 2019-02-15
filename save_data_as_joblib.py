import scipy.io as sio
import numpy as np
import joblib
import os
import random

batch_size = 20000
for NAME in ['TRAIN', 'TEST']:
    dict_mat = {}
    X_PATH = '/media/ophir/DATA1/Asaf/dataset_deep_project/database_joblib/{}'.format(NAME)
    files_train_x = os.listdir(X_PATH)
    list_file = [os.path.join(X_PATH, x) for x in files_train_x]
    list_file_tmp = ['_'.join(x.split('_')[:-1]) for x in list_file]
    list_file_tmp_un = set(list_file_tmp)
    # random.shuffle(list_file)


    for j, file in enumerate(list_file_tmp_un):
        # file_list = [x for x in list_file if file in x]
        file_list_sorted  = sorted(['_'.join(x.split('_')[:-1])+'_{:03d}.mat'.format(int(x.split('_')[-1].split('.')[0])) for x in list_file if file in x])
        stft = np.zeros((len(file_list_sorted), 257))
        mfcc = np.zeros((len(file_list_sorted), 39 ))
        map = np.zeros((len(file_list_sorted), 257))
        for i, one_file in enumerate(file_list_sorted):
            # one_file.split('.')[0][-3:].lstrip('0')
            one_file_new = one_file.split('.')[0][:-3]+one_file.split('.')[0][-3:].lstrip('0')+'.mat'
            image3d_mat = sio.loadmat(one_file_new, squeeze_me=True)
            stft[i, :] = image3d_mat['Z_to_write_one'][257*4:5*257]
            mfcc[i, :] = image3d_mat['mfcc_to_write_one'][39*4:5*39]
            map[i, :] = image3d_mat['map_to_write_one']
        data = {'stft': stft, 'mfcc': mfcc, 'map': map}
        joblib.dump(data, '/media/ophir/DATA1/Ophir/DeepLearning/project/data_for_lstm/{}/{}'.format(NAME, os.path.basename(file)))



        print('done batch_{}_#{}'.format(os.path.basename(file),j))
    print('FINISH {}'.format(NAME))




    # for j in range(int(len(list_file)/batch_size)):
    #     list_batch = list_file[j:j+batch_size]
    #     stft = np.zeros((batch_size, 257 * 9))
    #     mfcc = np.zeros((batch_size, 39 * 9))
    #     map = np.zeros((batch_size, 257))
    #     for i, file_name in enumerate(list_batch):
    #         print(i)
    #         image3d_mat = sio.loadmat(file_name, squeeze_me=True)
    #         stft[i, :] = image3d_mat['Z_to_write_one']
    #         mfcc[i, :] = image3d_mat['mfcc_to_write_one']
    #         map[i, :] = image3d_mat['map_to_write_one']
    #     data = {'stft': stft, 'mfcc': mfcc, 'map': map}
    #     joblib.dump(data, '/media/ophir/DATA1/Ophir/DeepLearning/project/data_in_batchs/{}/batch_{}'.format(NAME, j))
    #     print('done batch_{}'.format(j))
    # print('FINISH {}'.format(NAME))