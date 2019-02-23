import os

# torch imports
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

#local imports
from model import model
from dataloader import MyDataset

package_name = 'net'
model_name = 'ClassifaierAndDenoise'
model_arg = 6
# batch_size = 10000
batch_size = 1
optimazier = 'Adam'
lr = 0.001

root_path = '/media/ophir/DATA1/Ophir/DeepLearning/project/data_in_batchs'
# mfcc_path = os.path.join(root_path, 'TRAIN', 'MFCC')
# stft_path = os.path.join(root_path, 'TRAIN', 'STFT')
# map_path = os.path.join(root_path, 'TRAIN', 'MAP')
main_path_train = os.path.join(root_path, 'TRAIN')

# main_path_train = '/media/ophir/DATA1/Ophir/DeepLearning/project/data_in_batchs'
train_data = MyDataset(main_path_train)
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)

# valid_mfcc_path = os.path.join(root_path, 'TEST', 'MFCC')
# valid_stft_path = os.path.join(root_path, 'TEST', 'STFT')
# valid_map_path = os.path.join(root_path, 'TEST', 'MAP')
main_path_val = os.path.join(root_path, 'TEST')

valid_data = MyDataset(main_path_val)
validloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=0)

my_model = model(package_name=package_name, model_name=model_name, args=model_arg, description='6_experts', model_path = 'saved_models/Feb20_17-55_LSTMClassifaierAndDenoise_3_LSTM_39_experts_0_Adam_BCELoss.pth.tar')
my_model.fit(loss='MSELoss', accuracy_name='count_success', optimizer_name=optimazier, lr=lr)
# my_model.print_summary()
# my_model.model.double()
my_model.train(num_epochs=500, trainloader=trainloader,valloader=None, epochs_per_save=1, valloader=validloader)
