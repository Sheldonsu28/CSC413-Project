from torch import cat
from torch import nn

"""
Reference:
https://arxiv.org/pdf/1809.00219v2.pdf
"""
global_beta = 0.2


class Generator(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

        self.conv0 = nn.Sequential(
            nn.Conv2d(channels, 64, 9, stride=stride, padding=padding),
            self.relu
        )

        self.RRDB_layers = nn.Sequential(*([RRDB(64, 32, global_beta)] * 23))
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size, stride=stride, padding=padding),
            self.relu
        )

        self.upSample0 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size, stride=stride, padding=padding),
            nn.PixelShuffle(2),
            self.relu
        )

        self.upSample1 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size, stride=stride, padding=padding),
            nn.PixelShuffle(2),
            self.relu
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size, stride=stride, padding=padding),
            self.relu
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, channels, kernel_size, stride=stride, padding=padding),
            self.relu
        )

    def forward(self, x):
        x1 = self.conv0(x)
        x2 = self.RRDB_layers(x1)
        x3 = self.conv1(x2) + x1
        x4 = self.upSample0(x3)
        x5 = self.upSample1(x4)
        x6 = self.conv2(x5)

        return self.conv3(x6)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()


class RRDB(nn.Module):
    def __init__(self, channels, growth_rate, beta):
        super().__init__()

        self.block0 = DenseBlock(channels, growth_rate)
        self.block1 = DenseBlock(channels, growth_rate)
        self.block2 = DenseBlock(channels, growth_rate)
        self.beta = beta

    def forward(self, x):
        accumulator = x

        x1 = self.block0(x)
        accumulator += self.beta * x1

        x2 = self.block1(accumulator)
        accumulator += self.beta * x2

        x3 = self.block2(accumulator)

        return (accumulator + self.beta * x3) * self.beta + x


class DenseBlock(nn.Module):
    """
    Follows the design used in: https://arxiv.org/pdf/1809.00219v2.pdf
    Original design: https://arxiv.org/pdf/1608.06993v5.pdf
    """

    def __init__(self, channels, growth_rate, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.conv0 = nn.Conv2d(channels, growth_rate, kernel_size, stride, padding)
        self.conv1 = nn.Conv2d(channels + growth_rate, growth_rate, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(channels + 2 * growth_rate, growth_rate, kernel_size, stride, padding)
        self.conv3 = nn.Conv2d(channels + 3 * growth_rate, growth_rate, kernel_size, stride, padding)
        self.conv4 = nn.Conv2d(channels + 3 * growth_rate, channels, kernel_size, stride, padding)

        self.LReLu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.LReLu(self.conv0(x))
        x2 = self.LReLu(self.conv1(cat((x, x1), 1)))
        x3 = self.LReLu(self.conv2(cat((x, x1, x2), 1)))
        x4 = self.LReLu(self.conv3(cat((x, x1, x2, x3), 1)))
        return self.conv4(cat((x, x1, x2, x4), 1))
