from torch import nn

"""
Reference:
https://arxiv.org/pdf/1501.00092.pdf
"""


class SRCNN(nn.Module):
    def __init__(self, channel, device, f1=9, f2=5, f3=5, n1=64, n2=32):

        super(SRCNN, self).__init__()
        self.channel = channel
        self.block1 = nn.Conv2d(channel, n1, kernel_size=(f1, f1), bias=True, padding=f1 // 2, device=device)
        self.block2 = nn.Conv2d(n1, n2, kernel_size=(f2, f2), bias=True, padding=f2 // 2, device=device)
        self.block3 = nn.Conv2d(n2, channel, kernel_size=(f3, f3), bias=True, padding=f3 // 2, device=device)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.activation(self.block1(x))
        x2 = self.activation(self.block2(x1))
        return self.block3(x2)
