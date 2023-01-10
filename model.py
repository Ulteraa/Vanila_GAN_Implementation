import torch
import torchvision.utils
from torchvision import  transforms, datasets
from torch.utils.data import  Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import  SummaryWriter
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from PIL import  Image

class Discriminator(nn.Module):
    def __init__(self,img_dim):
        super(Discriminator, self).__init__()
        self.fc1=nn.Linear(img_dim,256)
        self.relu=nn.LeakyReLU(0.1)
        self.fc2=nn.Linear(256,256)
        self.out=nn.Linear(256,1)
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        x=self.fc1(x)
        x=self.relu(x)
        x=self.fc2(x)
        x=self.relu(x)
        x=self.out(x)
        x=self.sigmoid(x)
        return x
class Generator(nn.Module):
    def __init__(self,noize_dim,img_dim):
        super(Generator, self).__init__()
        self.fc1=nn.Linear(noize_dim,256)
        self.relu=nn.LeakyReLU(0.1)
        self.fc2=nn.Linear(256,256)
        self.out=nn.Linear(256,img_dim)
        self.tanh=nn.Tanh()
    def forward(self,x):
        x=self.fc1(x)
        x=self.relu(x)
        x=self.fc2(x)
        x=self.relu(x)
        x=self.out(x)
        x=self.tanh(x)
        return x





