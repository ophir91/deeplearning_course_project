from torch.utils.data import Dataset
import torch


class MyDataset(Dataset):  # TODO: need to write the data loader
    def __init__(self):
        pass

    def __len__(self):
        return 2000

    def __getitem__(self, item):

        sample = {'mfcc': torch.rand(1,39*9), 'stft': torch.rand(1,257*9), 'ground_truth': torch.rand(1,257)}
        return sample  # the sample the will be return nee to be in the above format
        pass



