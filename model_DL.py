from ast import mod
import math
from pyexpat import model
from threading import local
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ExpActivation(nn.Module):
    """
    Exponential activation function from Koo & Ploenzke, 2021 (PMID: 34322657)
    """
    def __init__(self):
        super(ExpActivation, self).__init__()

    def forward(self, x):
        return torch.exp(x)

class Unsqueeze(torch.nn.Module):
    """
    Unsqueeze for sequential models
    """
    def forward(self, x):
        return x.unsqueeze(-1)


class DeepSEA(nn.Module):
    def __init__(self, classes, linear_units, activate):
        super().__init__()

        conv_kernel_size = 8
        pool_kernel_size = 2

        # 选择激活函数
        activation = nn.ReLU() if activate == 'relu' else ExpActivation()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=32, kernel_size=conv_kernel_size, stride=1, padding=0),
            activation,
            nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=conv_kernel_size, stride=1, padding=0),
            activation,
            nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=conv_kernel_size, stride=1, padding=0),
            activation,
            nn.Dropout(0.1)  
        )

        self.fc = nn.Sequential(
            nn.Linear(linear_units, 64),
            activation,
            nn.Linear(64, classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class DanQ(nn.Module):
    def __init__(self,classes,linear_units,activate,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if activate == 'relu':
            activation = nn.ReLU()
        elif activate == 'exp':
            activation = ExpActivation()

        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=4,out_channels=128,kernel_size=19,padding=9),
            activation,
            nn.MaxPool1d(kernel_size=13, stride=13),
            nn.Dropout(0.5)
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)

        self.fc = nn.Sequential(
            nn.Linear(linear_units, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, classes)
        )
    
    def forward(self, x):
        x = self.conv1d(x)
        x = x.permute(0,2,1)
        out,(hn,cn) = self.lstm(x)
        out = out.transpose(1,2)
        out = out.contiguous().view(x.size()[0],-1)
        out = self.fc(out)
        return out


class CNN_Attention(nn.Module):
    def __init__(self, classes, linear_units, activate, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if activate == 'relu':
            activation = nn.ReLU()
        elif activate == 'exp':
            activation = ExpActivation()
        else:
            raise ValueError("activate must be 'relu' or 'exp'")

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=100, kernel_size=19, padding=9),
            nn.BatchNorm1d(100),
            activation,
            nn.MaxPool1d(10)  
        )

        self.multiatten = nn.MultiheadAttention(embed_dim=100, num_heads=4, batch_first=True)

        self.linear = nn.Sequential(
            nn.Linear(linear_units, 1000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, classes),
        )

    def forward(self, x):
        x = self.conv(x)  
        x = x.transpose(1, 2)  
        x, _ = self.multiatten(x, x, x)  # 注意力层
        x = x.transpose(1, 2).contiguous().view(x.size(0), -1)  
        x = self.linear(x)
        return x


class CNN_Transformer(nn.Module):
    def __init__(self, classes, linear_units, activate) -> None:
        super().__init__()

        # 选择激活函数
        if activate == 'relu':
            activation = nn.ReLU()
        elif activate == 'exp':
            activation = ExpActivation()
        else:
            raise ValueError("activate 必须是 'relu' 或 'exp'")

        # CNN 部分
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=60, kernel_size=19, padding=9),  
            nn.BatchNorm1d(60),  # 更新 BatchNorm 输入通道数
            activation,
            nn.MaxPool1d(kernel_size=10),
            nn.Dropout(0.5)
        )

        # Transformer 部分
        transformer_layer = nn.TransformerEncoderLayer(d_model=60, nhead=6, batch_first=True)  
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=1)  

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(60, 925), 
            nn.ReLU(),
            nn.Linear(925, classes)
        )

    def forward(self, x):
        x = self.conv1d(x)  
        x = x.permute(0, 2, 1)  

        x = self.transformer(x)  
        x = x.mean(dim=1)  

        out = self.fc(x)
        return out



















