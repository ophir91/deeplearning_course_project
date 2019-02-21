import os
import scipy.io as sio
from torch.utils.data import Dataset
import torch
import joblib

class MyDataset(Dataset):  # TODO: need to write the data loader
    def __init__(self, main_path):
        # self.mfcc_path = mfcc_path
        # self.stft_path = stft_path
        # self.map_path = map_path
        self.main_path = main_path
        self.files_names = os.listdir(self.main_path)
        pass

    def __len__(self):
        return len(self.files_names)

    def __getitem__(self, idx):
        # mfcc = sio.loadmat(os.path.join(self.mfcc_path, self.files_names[idx]))['mfcc_to_write_one']
        # stft = sio.loadmat(os.path.join(self.stft_path, self.files_names[idx]))['Z_to_write_one']
        # map = sio.loadmat(os.path.join(self.map_path, self.files_names[idx]))['map_to_write_one']

        # data = sio.loadmat(os.path.join(self.main_path, self.files_names[idx]))

        data = joblib.load(os.path.join(self.main_path, self.files_names[idx]))
        #sample = {'mfcc': torch.rand(1,39*9), 'stft': torch.rand(1,257*9), 'ground_truth': torch.rand(1,257)}
        # sample = {'mfcc': torch.from_numpy(mfcc), 'stft': torch.from_numpy(stft), 'ground_truth': torch.from_numpy(map)}

        sample = {'mfcc': torch.from_numpy(data['mfcc']), 'stft': torch.from_numpy(data['stft']), 'ground_truth': torch.from_numpy(data['map'])}
        return sample  # the sample the will be return nee to be in the above format
        pass



if __name__ == '__main__':
    root_path = '/media/ophir/DATA1/Asaf/dataset_deep_project/database/TRAIN'
    mfcc_path = os.path.join(root_path,'MFCC')
    stft_path = os.path.join(root_path,'STFT')
    map_path = os.path.join(root_path,'MAP')

    triandataset = MyDataset(mfcc_path,stft_path,map_path)
    for s in triandataset:
        print(s)