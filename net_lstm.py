import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init
from torchvision import models
from collections import OrderedDict


class LSTMClassifaierAndDenoise(nn.Module):
    def __init__(self, M):
        super(LSTMClassifaierAndDenoise, self).__init__()
        self.M = M

        self.d = torch.zeros(self.M, 257)
        self.oneDenoise = LSTMblock(input_dim=257, hidden_dim=512, output_size=257, num_of_deep_lstm=1)
        self.MDenoise = nn.ModuleList()
        for i in range(M):
            self.MDenoise.append(self.oneDenoise)

        self.classifier = LSTMblock(input_dim=39, hidden_dim=512, output_size=M, num_of_deep_lstm=1)

    def forward(self, stft, mfcc):
        N = mfcc.shape[0]  # number of frames
        c = self.classifier(mfcc)
        d = torch.zeros(self.M, N, 257)
        for i, one_denoise in enumerate(self.MDenoise):
            one_denoise_output = one_denoise(stft)
            d[i, :, :] = one_denoise_output

        maps = torch.zeros(N, 257)
        for j in range(N):
            maps[j:j+1] = torch.matmul(c[j:j+1, :], d[:, j, :])
        return maps


class LSTMblock(nn.Module):

    def __init__(self, input_dim=257, hidden_dim=512, output_size=257, num_of_deep_lstm=1):
        super(LSTMblock, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes stft as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim, hidden_dim,num_layers=num_of_deep_lstm)

        # The linear layer that maps from hidden state space to output space
        self.hidden2out = nn.Linear(hidden_dim, output_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, stft):
        lstm_out, self.hidden = self.lstm(
            stft.view(len(stft), 1, -1), self.hidden)
        out_space = self.hidden2out(lstm_out.view(len(stft), -1))
        out_scores = F.log_softmax(out_space, dim=1)
        return out_scores


if __name__ == '__main__':
    N = 80
    stft = torch.rand(N, 1, 257)  # LSTM input of shape (seq_len, batch, input_size)
    mfcc = torch.rand(N, 1, 39)  # LSTM input of shape (seq_len, batch, input_size)
    net = LSTMClassifaierAndDenoise(6)
    out = net(stft, mfcc)
    pass