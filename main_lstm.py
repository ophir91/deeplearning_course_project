import os

# torch imports
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

#local imports
from model import model
from dataloader import MyDataset

package_name = 'net_lstm'
model_name = 'LSTMClassifaierAndDenoise'
model_arg = 39
# batch_size = 10000
batch_size = 1
optimazier = 'Adam'
lr = 0.001

root_path = '/media/ophir/DATA1/Ophir/DeepLearning/project/data_for_lstm'
# mfcc_path = os.path.join(root_path, 'TRAIN', 'MFCC')
# stft_path = os.path.join(root_path, 'TRAIN', 'STFT')
# map_path = os.path.join(root_path, 'TRAIN', 'MAP')
main_path_train = os.path.join(root_path, 'TRAIN')

# main_path_train = '/media/ophir/DATA1/Ophir/DeepLearning/project/data_in_batchs'
train_data = MyDataset(main_path_train)
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=0)

# valid_mfcc_path = os.path.join(root_path, 'TEST', 'MFCC')
# valid_stft_path = os.path.join(root_path, 'TEST', 'STFT')
# valid_map_path = os.path.join(root_path, 'TEST', 'MAP')
main_path_val = os.path.join(root_path, 'TEST')

valid_data = MyDataset(main_path_val)
validloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=0)

my_model = model(package_name=package_name, model_name=model_name, args=model_arg, description='3_LSTM_39_experts')
my_model.fit(loss='BCELoss', accuracy_name='count_success', optimizer_name=optimazier, lr=lr)
# my_model.print_summary()
# my_model.model.double()
my_model.train(num_epochs=250, trainloader=trainloader, epochs_per_save=5, valloader=validloader)
