import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init
from torchvision import models
from collections import OrderedDict


class ClassifaierAndDenoise(nn.Module):
    def __init__(self, M):
        super(ClassifaierAndDenoise, self).__init__()
        self.M = M
        self.d = torch.zeros(self.M, 257)
        self.oneDenoise = Denoise()
        self.MDenoise = nn.ModuleList()
        for i in range(M):
            self.MDenoise.append(self.oneDenoise)

        self.classifier = Classifier(M=self.M)

    def forward(self, stft, mfcc):
        c = self.classifier(mfcc)
        d = torch.zeros(mfcc.shape[0], self.M, 257)
        for i, one_denoise in enumerate(self.MDenoise):
            d[:, i:i+1, :] = one_denoise(stft) * c[..., i].unsqueeze(1)

        return torch.sum(d, dim=1).unsqueeze(1)


class Denoise(nn.Module):
    def __init__(self, hiddenlayers=1, sizes=[512], input_size=257*9, output_size=257):
        super(Denoise, self).__init__()
        layers = []
        i=0
        for i in range(hiddenlayers):
            layers.append(nn.Linear(input_size, sizes[i]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=0.5))
        layers.append(nn.Linear(sizes[i], output_size))
        layers.append(nn.Sigmoid())
        self.oneDenosie = nn.Sequential(*layers)

    def forward(self, stft):
        return self.oneDenosie(stft)


class Classifier(nn.Module):
    def __init__(self, input_size=39*9, M=39):
        super(Classifier, self).__init__()
        self.M = M
        self.l1 = nn.Linear(input_size, 512)
        self.bn = nn.BatchNorm1d(1)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.l2 = nn.Linear(512, M)
        self.outactivate = nn.Softmax(dim=2)

    def forward(self, mfcc):
        x = self.l1(mfcc)
        x = self.bn(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.l2(x)
        x = self.outactivate(x)
        return x


if __name__ == '__main__':
    stft = torch.rand(1, 1, 257*9)
    mfcc = torch.rand(1, 1, 39*9)
    net = ClassifaierAndDenoise(6)
    out = net(stft, mfcc)
    pass