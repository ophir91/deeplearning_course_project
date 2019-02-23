import datetime
import os
import numpy as np
import time
import scipy.ndimage

# torch imports
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from tensorboardX import SummaryWriter
from torchsummary import summary

# local imports


class model:
    def __init__(self, package_name, model_name, description='', model_path=None, args=None):

        self.package_name = __import__(package_name)
        self.model_name = model_name
        self.model_path = model_path
        self.model = None
        self.writer = None
        self.loss = ''
        self.optimizer_name = ''
        self.accuracy_name = ''
        self.checkpoint_name = ''
        self.optimizer = None
        self.accuracy = None
        self.criterion = None
        self.epoch = 0
        self.description = description
        self.create_model(args=args)
        # self.create_writer()

    def create_model(self, args):
        if self.model_path is not None:
            print("=> Loading checkpoint '{}'".format(self.model_path))
            self.load_checkpoint(self.model_path, args=args)
        else:
            print("=> Creating new model")
            self.model = getattr(self.package_name, self.model_name)(args)
        if torch.cuda.is_available():
            print("Using GPU")
            # self.model = nn.DataParallel(self.model,  device_ids=[0])
            self.model = self.model.cuda()

    def create_writer(self, checkpoint_name=''):
        if checkpoint_name is '':
            self.checkpoint_name = datetime.datetime.now().strftime('%b%d_%H-%M') + '_' + self.model_name + '_' + self.description
        else:
            self.checkpoint_name = checkpoint_name
        writer_name = '{checkpoint_name}_{optimizer}_{loss_name}.pth.tar'\
            .format(checkpoint_name=self.checkpoint_name, optimizer=self.optimizer_name, loss_name=self.loss)
        writer_dir = os.path.join('runs', writer_name)
        self.writer = SummaryWriter(log_dir=writer_dir)

    def fit(self, loss='MSELoss', optimizer_name='Adam', lr=0.01, weight_decay=0, accuracy_name='',
            create_writer=True):
        self.loss = loss
        self.optimizer_name = optimizer_name
        self.accuracy_name = accuracy_name
        self.criterion = getattr(nn, loss)()
        # self.criterion = getattr(self.package_name, loss)()
        if optimizer_name == 'SGD':
            self.optimizer = getattr(optim, optimizer_name)(self.model.parameters(), lr=lr, momentum=0.9)
        else:
            self.optimizer = getattr(optim, optimizer_name)(self.model.parameters(), lr=lr)
        if accuracy_name == 'argmax':
            self.accuracy = argmax
        elif accuracy_name == 'count_success':
            self.accuracy = count_success
        elif accuracy_name != '':
            self.accuracy = getattr(nn, accuracy_name)()
            # self.accuracy = getattr(self.package_name, accuracy_name)()
        if create_writer:
            self.create_writer(self.checkpoint_name)

    def save_checkpoint(self, save_dir='', epochs_per_save=1):
        # TODO: add is_best to save, to save the best model in the training
        filename = '{checkpoint_name}_{epoch_num}_{optimizer}_{loss_name}.pth.tar'\
            .format(checkpoint_name=self.checkpoint_name, epoch_num=self.epoch,
                    optimizer=self.optimizer_name, loss_name=self.loss)
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss,
            'optimizer_name': self.optimizer_name,
            'accuracy_name': self.accuracy_name,
            'checkpoint_name': self.checkpoint_name
        }, os.path.join(save_dir, filename))

        print("Saved checkpoint as: {}".format(os.path.join(save_dir, filename)))

        # removing the old checkpoint:
        # TODO: need to check if the remove is working
        old_filename = '{checkpoint_name}_{epoch_num}_{optimizer}_{loss_name}.pth.tar'\
            .format(checkpoint_name=self.checkpoint_name, epoch_num=self.epoch-epochs_per_save,
                    optimizer=self.optimizer_name, loss_name=self.loss)
        #TODO check it
        # if os.path.exists(os.path.join(save_dir, old_filename)):
        #     os.remove(os.path.join(save_dir, old_filename))

    def load_checkpoint(self, filename, args):
        """
        loads checkpoint (that was save with save_checkpoint)
        No need to do .fit after
        :param filename: path to the checkpoint
        :return:
        """
        self.model = getattr(self.package_name, self.model_name)(args)

        checkpoint = torch.load(filename)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.epoch = checkpoint['epoch']
        self.loss = checkpoint['loss']
        self.optimizer_name = checkpoint['optimizer_name']
        self.accuracy_name = checkpoint['accuracy_name']
        # self.checkpoint_name = None
        self.checkpoint_name = checkpoint['checkpoint_name']

        self.fit(self.loss, self.optimizer_name, accuracy_name=self.accuracy_name)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print("Loaded checkpoint as: {}".format(filename))

    def print_summary(self, input_size=(1, 32, 128, 128)):
        summ = summary(self.model, input_size=input_size)
        # self.writer.add_text('Summary', summ)

    def print_graph(self, dummy_input):
        # dummy_input = Variable(torch.rand(1, 1, 32, 64, 64))
        if self.writer is not None:
            self.writer.add_graph(model=self.model, input_to_model=(dummy_input,))

    def print_epoch_statistics(self, epoch, epoch_time, running_loss, running_accuracy, validation_accuracy=None):
        """
        :param epoch: number of epoch this results from
        :param running_loss: array of all the losses in this epoch
        :param running_accuracy: array of all the training accuracies in this epoch
        :param validation_accuracy: array of all the validation accuracies in this epoch
        :return: print on the stdout the results and log in tensorboard if defined
        """
        if validation_accuracy is None:
            print("End of epoch {:3d} in {:3d} sec | Training loss = {:5.4f} | Training acc = {:5.4f}"
                  .format(epoch, int(epoch_time), np.mean(running_loss), np.mean(running_accuracy)))
        else:
            print("End of epoch {:3d} in {:3d} sec | Training loss = {:5.4f} | Training acc = {:5.4f} | Valid acc =  {:5.4f}"
                  .format(epoch, int(epoch_time), np.mean(running_loss), np.mean(running_accuracy), np.mean(validation_accuracy)))
        if self.writer is not None:
            self.writer.add_scalar('Train/Loss', float(np.mean(running_loss)), epoch)
            self.writer.add_scalar('Train/accuracy', float(np.mean(running_accuracy)), epoch)
            if validation_accuracy is not None:
                self.writer.add_scalar('Validation/accuracy', float(np.mean(validation_accuracy)), epoch)

    def add_images_tensorboard(self, inputs, labels, outputs):  # TODO: check this function
        """

        :param inputs: the net input, a 5 dim tensor shape: [batch, channels, z, x, y]
        :param labels: the ground truth, a 5 dim tensor shape: [batch, channels, z, x, y]
        :param outputs: the net output, a 5 dim tensor shape: [batch, channels, z, x, y]
        :return: add images to tensorboard
        """


    def test_validation(self, validationloader=None):
        validation_accuracy = None
        if validationloader is not None:
            self.model.eval()  # changing to eval mode
            valid_running_accuracy = []
            with torch.no_grad():
                for k, sample in enumerate(validationloader, 0):
                    if isinstance(sample, dict):
                        if self.model_name == 'LSTMClassifaierAndDenoise':
                            valid_mfcc = sample['mfcc'].reshape(-1, 1, 39)
                            valid_stft = sample['stft'].reshape(-1, 1, 257)
                            valid_labels = sample['ground_truth'].reshape(-1, 257)
                        else:
                            valid_mfcc = sample['mfcc'].reshape(-1, 1, 351)
                            valid_stft = sample['stft'].reshape(-1, 1, 2313)
                            valid_labels = sample['ground_truth'].reshape(-1, 1, 257)
                    else:
                        valid_mfcc, valid_stft, valid_labels = sample

                    # wrap them in Variable
                    if torch.cuda.is_available():
                        valid_mfcc,valid_stft, valid_labels = Variable(valid_mfcc.cuda()).float(),\
                                                              Variable(valid_stft.cuda()).float(),\
                                                              Variable(valid_labels.cuda()).float()
                    else:
                        valid_mfcc,valid_stft, valid_labels = Variable(valid_mfcc), Variable(valid_stft), \
                                                     Variable(valid_labels)

                    valid_outputs = self.model(valid_stft, valid_mfcc).cuda()
                    acc = self.accuracy(valid_outputs.cpu().data, valid_labels.cpu().data)
                    valid_running_accuracy.append(acc)
            validation_accuracy = valid_running_accuracy
            self.model.train()  # back to train mode
        return validation_accuracy

    def train(self, num_epochs, trainloader, valloader=None, epochs_per_save=10):
        print("Start training")
        start_train_time = time.time()
        for epoch in range(self.epoch,num_epochs):  # loop over the dataset multiple times # adds that aaafter loadndigng the epochs will start for the last one
            start_epoch_time = time.time()
            self.epoch = epoch
            running_loss = []
            running_accuracy = []
            for i, sample in enumerate(trainloader, 0):
                # print(i)
                if isinstance(sample, dict):
                    if self.model_name =='LSTMClassifaierAndDenoise':
                        mfcc = sample['mfcc'].reshape(-1, 1, 39)
                        stft = sample['stft'].reshape(-1, 1, 257)
                        labels = sample['ground_truth'].reshape(-1, 257)
                    else:
                        # reshape because we entered batch as one sample
                        mfcc = sample['mfcc'].reshape(-1,1,351)
                        stft = sample['stft'].reshape(-1,1,2313)
                        labels = sample['ground_truth'].reshape(-1, 1,257)
                else:
                    inputs, labels = sample

                # wrap them in Variable
                if torch.cuda.is_available():
                    mfcc, stft, labels = Variable(mfcc.cuda()).float(), Variable(stft.cuda()).float(), Variable(labels.cuda()).float()
                else:
                    mfcc, stft, labels = Variable(mfcc), Variable(stft), Variable(labels)

                # forward + backward + optimize
                outputs = self.model(stft, mfcc).cuda()
                loss = self.criterion(outputs, labels)
                # zero the parameter gradients
                self.optimizer.zero_grad()
                loss.backward()
                # if i >0:
                #     loss.backward()
                # else:
                #     loss.backward(retain_graph=True)
                self.optimizer.step()

                # for loss per epoch
                running_loss.append(loss.item())
                if self.accuracy is not None:
                    # for accuracy per epoch
                    running_accuracy.append(self.accuracy(outputs.cpu().data, labels.cpu().data))
                    if i%10 == 0 :
                        print('tmp accuracy {} in i = {} in epoch {}'.format(np.mean(running_accuracy),i,epoch))
            validation_accuracy = self.test_validation(valloader)
            self.print_epoch_statistics(epoch, int(time.time() - start_epoch_time), running_loss, running_accuracy, validation_accuracy, )
            if epoch % epochs_per_save == 0:
                self.save_checkpoint('saved_models', epochs_per_save)
                # self.add_images_tensorboard(inputs, labels, outputs)
        self.save_checkpoint('saved_models')
        print('='*89)
        print("Finish Training, {} epochs in {} seconds".format(num_epochs, int(time.time() - start_train_time)))
        print('='*89)

def MyMse(outputs, labels):
    return torch.sqrt(torch.sum((outputs-labels)**2))

def argmax(predict, labels): #TODO: not working yet.. need to change it
    pred = predict.data.max(1)[1].long()
    return float(pred.eq(labels.data.view_as(pred).long()).sum()) / float(pred.shape[0])

def count_success(predict, labels):
    pred = np.array(([predict > 0.5] * 1)[0], dtype='float')
    a = pred.flatten()==labels.view(-1).numpy()
    return int(a.sum())/a.shape[0]

if __name__ == '__main__':

    pass