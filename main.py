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

# transform = transforms.Compose()

train_data = MyDataset()
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)


valid_data = MyDataset()
validloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=0)

my_model = model(package_name=package_name, model_name=model_name, args=model_arg, description='6_experts')
my_model.fit(loss='MSELoss', accuracy_name='MyMse', optimizer_name=optimazier, lr=lr)
# my_model.print_summary()
my_model.train(num_epochs=100, trainloader=trainloader, epochs_per_save=5, valloader=validloader)
