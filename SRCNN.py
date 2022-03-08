from torch import nn

"""
Reference:
https://arxiv.org/pdf/1501.00092.pdf
"""


class SRCNN(nn.Module):
    def __init__(self, channel, f1=9, f2=5, f3=5, n1=64, n2=32):
        super(SRCNN, self).__init__()
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.channel = channel
        self.block1 = nn.Conv2d(channel, n1, f1, bias=True, padding=f1 // 2)
        self.block2 = nn.Conv2d(n1, n2, f2, bias=True, padding=f2 // 2)
        self.block3 = nn.Conv2d(n2, channel, f3, bias=True, padding=f3 // 2)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.activation(self.block1(x))
        x2 = self.activation(self.block2(x1))
        return self.block3(x2)


def calc_size(H, kernel_size, padding=0, dilation=1, stride=1):
    return ((H + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1


if __name__ == "__main__":
    a = 1080
    a = calc_size(a, 3, stride=1)
    print(a)
