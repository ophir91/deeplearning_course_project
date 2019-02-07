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
batch_size = 1000
optimazier = 'Adam'
lr = 0.001

root_path = '/media/ophir/DATA1/Asaf/dataset_deep_project/database'
mfcc_path = os.path.join(root_path, 'TRAIN', 'MFCC')
stft_path = os.path.join(root_path, 'TRAIN', 'STFT')
map_path = os.path.join(root_path, 'TRAIN', 'MAP')

train_data = MyDataset(mfcc_path,stft_path,map_path)
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)

valid_mfcc_path = os.path.join(root_path, 'TEST', 'MFCC')
valid_stft_path = os.path.join(root_path, 'TEST', 'STFT')
valid_map_path = os.path.join(root_path, 'TEST', 'MAP')

valid_data = MyDataset(valid_mfcc_path,valid_stft_path,valid_map_path)
validloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=0)

my_model = model(package_name=package_name, model_name=model_name, args=model_arg, description='6_experts')
my_model.fit(loss='MSELoss', accuracy_name='argmax', optimizer_name=optimazier, lr=lr)
# my_model.print_summary()
# my_model.model.double()
my_model.train(num_epochs=100, trainloader=trainloader, epochs_per_save=5, valloader=validloader)
