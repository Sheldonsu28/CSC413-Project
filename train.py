from ESRGAN import Generator, Discriminator
from SRCNN import SRCNN

import os
import numpy as np
from numpy import genfromtxt
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision import transforms
from torch.optim import Adam
from Utils import to_var

random.seed(42)


class CustomDataSRCNN(Dataset):
    def __init__(self):
        self.names = np.loadtxt('DIV2K_train_HR/DIV2K_train_HR.csv', dtype='str', delimiter=", ",
                                encoding="utf-8")
        self.hr_transform = transforms.Compose([transforms.CenterCrop((256, 256)),
                                                ])

        self.lr_transform = transforms.Compose([transforms.CenterCrop((256, 256)),
                                                transforms.Resize((128, 128)),
                                                transforms.Resize((256, 256))])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        img_name = os.path.join('DIV2K_train_HR',
                                self.names[idx])
        hr = read_image(img_name)
        return self.lr_transform(hr), self.hr_transform(hr)


def train_SRCNN():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)

    srcnn = SRCNN(3, device)
    if torch.cuda.is_available():
        srcnn.cuda(device)
    params = srcnn.parameters()
    optimizer = Adam(params, 10e-4)
    loss = nn.MSELoss()
    dataloader = DataLoader(CustomDataSRCNN(), batch_size=100, shuffle=True, num_workers=0)
    train_iter = iter(dataloader)

    for i in range(100):
        print(i)
        try:
            lr, hr = train_iter.next()
            lr, hr = to_var(lr, device), to_var(hr, device)
        except StopIteration:
            train_iter = iter(dataloader)
            lr, hr = train_iter.next()
            lr, hr = to_var(lr, device), to_var(hr, device)
        print('f')
        optimizer.zero_grad()
        g = srcnn(lr)
        l = loss(g, hr)
        l.backward()
        optimizer.step()


if __name__ == "__main__":
    train_SRCNN()
