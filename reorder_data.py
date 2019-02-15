import scipy.io as sio
import numpy as np
import joblib
import os


batch_size = 300

if 0:
    for NAME in ['TRAIN', 'TEST']:
        dict_mat = {}
        X_PATH = '/media/ophir/DATA1/Asaf/dataset_deep_project/database_joblib/{}'.format(NAME)
        files_train_x = os.listdir(X_PATH)
        list_file = [os.path.join(X_PATH,x) for x in files_train_x]
        list_file = list_file[:batch_size]

        stft = np.zeros((batch_size,257*9))
        mfcc = np.zeros((batch_size,39*9))
        map = np.zeros((batch_size,257))
        for i ,file_name in enumerate(list_file):
            image3d_mat = sio.loadmat(file_name, squeeze_me=True)
            stft[i, :] = image3d_mat['Z_to_write_one']
            mfcc[i, :] = image3d_mat['mfcc_to_write_one']
            map[i, :]  = image3d_mat['map_to_write_one']
        data = {'stft':stft, 'mfcc':mfcc, 'map':map}
        joblib.dump(data,'data_tmp')


        new_mat_input = np.zeros((len(list_file), 257*9))
        for i ,file_name in enumerate(list_file):
            image3d_mat = sio.loadmat(file_name, squeeze_me=True)
            new_mat_input[i, :] = image3d_mat['Z_to_write_one']

        files_train_x = os.listdir(c_PATH)
        list_file = [os.path.join(c_PATH,x) for x in files_train_x]
        # list_file = list_file[:20]
        new_mat_label = np.zeros((len(list_file), 257))
        for i ,file_name in enumerate(list_file):
            image3d_mat = sio.loadmat(file_name, squeeze_me=True)
            new_mat_label[i, :] = image3d_mat['map_to_write_one']

        path_save = os.path.dirname(X_PATH.replace('database_phn', 'database_phn_joblib'))
        os.makedirs(path_save , exist_ok=True)


        dict_mat['input'] = new_mat_input
        dict_mat['label'] = new_mat_label

        joblib.dump(dict_mat, path_save+'/'+'data')
        print('FINISH {} {}'.format(NAME, id_phn))

        # dict2 = joblib.load('/media/ophir/DATA1/Asaf/sakranot/dataset/database_phn_joblib/TRAIN/7'+'/'+'data')


if 1:
    for NAME in ['TRAIN', 'TEST']:
        if NAME == 'TRAIN':
            num_exapmle = 800000
        else:
            num_exapmle = 200000

        dict_mat = {}
        X_PATH = '/media/ophir/DATA1/Asaf/sakranot/dataset/database_mfcc_matlab/{}/input'.format(NAME)
        c_PATH = '/media/ophir/DATA1/Asaf/sakranot/dataset/database_mfcc_matlab/{}/label'.format(NAME)
        files_train_x = os.listdir(X_PATH)
        list_file = [os.path.join(X_PATH,x) for x in files_train_x]

        list_file = list_file[:num_exapmle]
        new_mat_input = np.zeros((len(list_file), 39*9))
        for i ,file_name in enumerate(list_file):
            image3d_mat = sio.loadmat(file_name, squeeze_me=True)
            new_mat_input[i, :] = image3d_mat['mfcc_39_9']
        print('FINISH mfcc {} '.format(NAME))

        files_train_x = os.listdir(c_PATH)
        list_file = [os.path.join(c_PATH,x) for x in files_train_x]
        list_file = list_file[:num_exapmle]
        new_mat_label = np.zeros((len(list_file), 39))
        for i ,file_name in enumerate(list_file):
            image3d_mat = sio.loadmat(file_name, squeeze_me=True)
            new_mat_label[i, :] = image3d_mat['onehot_vec']
        print('FINISH onehot {} '.format(NAME))

        path_save = os.path.dirname(X_PATH.replace('database_mfcc_matlab', 'database_mfcc_matlab_joblib'))
        os.makedirs(path_save , exist_ok=True)


        dict_mat['input'] = new_mat_input
        dict_mat['label'] = new_mat_label

        joblib.dump(dict_mat, path_save+'/'+'data4')
        print('FINISH {}'.format(NAME))
