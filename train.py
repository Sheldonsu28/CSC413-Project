from ESRGAN import Generator, Discriminator
from matplotlib import pyplot as plt
from SRCNN import SRCNN
import os
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision import transforms
from torch.optim import Adam
from torch.autograd import Variable
from torchvision.utils import save_image
from Utils import to_var

random.seed(42)


class CustomData(Dataset):
    def __init__(self):
        self.names = np.loadtxt('hr/hr.csv', dtype='str', delimiter=", ",
                                encoding="utf-8")
        self.hrs = [read_image(os.path.join('hr', name)) for name in self.names]

        self.lr_transform = transforms.Compose([transforms.Resize((64, 64)), transforms.Resize((128, 128))])

        self.lrs = [self.lr_transform(img) for img in self.hrs]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        return self.lrs[idx], self.hrs[idx]


def train_SRCNN(args):
    """
    Train SRCNN model
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)

    model = SRCNN(args['out_channels'])
    if torch.cuda.is_available():
        model.cuda(device)

    model.train()
    params = model.parameters()
    optimizer = Adam(params, args['learning_rate'], args['betas'])
    # train with mse loss
    loss = nn.MSELoss()
    dataloader = DataLoader(CustomData(), batch_size=args['batch_size'], shuffle=True, num_workers=0,
                            pin_memory=True)
    train_iter = iter(dataloader)
    train_loss = []

    for i in range(args['epochs']):
        try:
            lr, hr = train_iter.next()
            lr, hr = to_var(lr, device), to_var(hr, device)
        except StopIteration:
            train_iter = iter(dataloader)
            lr, hr = train_iter.next()
            lr, hr = to_var(lr, device), to_var(hr, device)
        optimizer.zero_grad()
        g = model(lr)
        l = loss(g, hr)
        l.backward()
        optimizer.step()
        train_loss.append(l.item())
        if i % 100 == 0:
            print('Iteration {}/{}, training loss: {}'.format(i, args['epochs'], l))
    plt.plot([i for i in range(args['epochs'])], train_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('SRCNN.png')
    model.eval()
    with torch.no_grad():
        img = read_image('1.png')
        tensor = torch.tensor(img)
        out = model(to_var(tensor[None, :, :, :], device))
        out = out.squeeze(0).cpu().detach()
        save_image(out, "SRCNN_out.png")


def train_ESRGAN(args):
    """
    Train ESRGAN model
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)

    G = Generator(args['out_channels'])
    D = Discriminator(args['out_channels'])

    if torch.cuda.is_available():
        G.cuda(device)
        D.cuda(device)

    G.train()
    D.train()

    params_G = G.parameters()
    params_D = D.parameters()
    num_params_G = sum(p.numel() for p in G.parameters() if p.requires_grad)
    num_params_D = sum(p.numel() for p in D.parameters() if p.requires_grad)
    print("Number of parameters in G: {}".format(num_params_G))
    print("Number of parameters in D: {}".format(num_params_D))

    optimizer_G = Adam(params_G, args['learning_rate'], args['betas'])
    optimizer_D = Adam(params_D, args['learning_rate'], args['betas'])

    D_loss_func = nn.BCEWithLogitsLoss()
    G_loss_func = nn.MSELoss()

    dataloader = DataLoader(CustomData(), batch_size=args['batch_size'], shuffle=True, num_workers=0, pin_memory=True)

    train_iter = iter(dataloader)

    G_losses = []
    D_losses = []

    for i in range(args['epochs']):
        try:
            lr, hr = train_iter.next()
            lr, hr = to_var(lr, device), to_var(hr, device)
        except StopIteration:
            train_iter = iter(dataloader)
            lr, hr = train_iter.next()
            lr, hr = to_var(lr, device), to_var(hr, device)
        optimizer_G.zero_grad()
        optimizer_D.zero_grad()

        # G loss
        fake_img = G(lr)
        g_loss = G_loss_func(fake_img, hr)
        g_loss.backward()
        optimizer_G.step()

        # D loss (Relativistic Loss)
        predict_fake = D(fake_img.detach())
        predict_real = D(hr)

        fake_loss = D_loss_func(predict_real - torch.mean(predict_fake), torch.ones_like(predict_real))
        real_loss = D_loss_func(predict_fake - torch.mean(predict_real), torch.zeros_like(predict_fake))

        # Code segment from PA4 DCGAN
        # ---- Gradient Penalty ----
        if args['gradient_penalty']:
            alpha = torch.rand(hr.shape[0], 1, 1, 1)
            alpha = alpha.expand_as(hr).cuda()
            interp_images = Variable(alpha * hr.data + (1 - alpha) * fake_img.data,
                                     requires_grad=True).cuda()
            D_interp_output = D(interp_images)

            gradients = torch.autograd.grad(outputs=D_interp_output, inputs=interp_images,
                                            grad_outputs=torch.ones(D_interp_output.size()).cuda(),
                                            create_graph=True, retain_graph=True)[0]
            gradients = gradients.view(hr.shape[0], -1)
            gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

            gp = gradients_norm.mean()
        else:
            gp = 0.0

        d_loss = fake_loss + real_loss + gp
        d_loss.backward()

        optimizer_D.step()

        G_losses.append(g_loss.item())
        D_losses.append(d_loss.item())
        if i % 200 == 0:
            print('Iteration {}/{}, G loss: {}, D loss: {}'.format(i, args['epochs'], g_loss, d_loss))
    x_axis = [i for i in range(args['epochs'])]
    plt.plot(x_axis, G_losses, label='G loss')
    plt.plot(x_axis, D_losses, label='D loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('ESRGAN.png')


if __name__ == "__main__":
    args = {'out_channels': 3,
            'batch_size': 100,
            'learning_rate': 10e-4,
            'epochs': 5000,
            'betas': [0.9, 0.999],
            'gradient_penalty': True  # This field is not used in SRCNN training
            }
    train_SRCNN(args)
    # train_ESRGAN(args)
