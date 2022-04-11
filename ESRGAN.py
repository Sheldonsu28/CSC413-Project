from torch import cat
from torch import nn
from torch import flatten
from torch import add

"""
Reference:
https://arxiv.org/pdf/1809.00219v2.pdf
"""
global_beta = 0.2


class Generator(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(channels, 64, (3, 3), stride=(stride, stride), padding=padding),
            nn.ReLU()
        )

        self.RRDB_layers = nn.Sequential(*[RRDB(64, 32, global_beta) for i in range(16)])

        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, (kernel_size, kernel_size), stride=(stride, stride), padding=padding),
            nn.ReLU(inplace=True)
        )

        self.upSample0 = nn.Sequential(
            nn.Conv2d(64, 64 * 4, (1, 1), stride=(stride, stride)),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, (kernel_size, kernel_size), stride=(stride, stride), padding=padding),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, channels, (kernel_size, kernel_size), stride=(stride, stride), padding=padding),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.conv0(x)
        x2 = self.RRDB_layers(x1)
        x3 = add(self.conv1(x2), x1)
        x4 = self.upSample0(x3)
        x5 = self.conv2(x4)
        return self.conv3(x5)


class Discriminator(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=(kernel_size, kernel_size), stride=(stride, stride), padding=padding,
                      bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.basicBlocks = nn.Sequential(
            DiscriminatorBlock(64, 64, kernel_size=4, stride=2, padding=padding),
            DiscriminatorBlock(64, 128, kernel_size=3, stride=stride, padding=padding),
            DiscriminatorBlock(128, 128, kernel_size=4, stride=2, padding=padding),
            DiscriminatorBlock(128, 256, kernel_size=3, stride=stride, padding=padding),
            DiscriminatorBlock(256, 256, kernel_size=4, stride=2, padding=padding),
            DiscriminatorBlock(256, 512, kernel_size=3, stride=stride, padding=padding),
            DiscriminatorBlock(512, 512, kernel_size=4, stride=2, padding=padding),
            DiscriminatorBlock(512, 512, kernel_size=3, stride=stride, padding=padding),
            DiscriminatorBlock(512, 512, kernel_size=4, stride=2, padding=padding),
        )

        self.block2 = nn.Sequential(
            nn.Linear(512 * 4 * 4, 100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        layer1 = self.block1(x)
        block_out = self.basicBlocks(layer1)
        flattened = flatten(block_out, 1)
        return self.block2(flattened)


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size),
                      stride=(stride, stride), padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.block1(x)


class RRDB(nn.Module):
    def __init__(self, channels, growth_rate, beta):
        super().__init__()

        self.block0 = DenseBlock(channels, growth_rate)
        self.block1 = DenseBlock(channels, growth_rate)
        self.block2 = DenseBlock(channels, growth_rate)
        self.beta = beta

    def forward(self, x):
        x1 = self.block0(x)
        skip0 = self.beta * x1 + x

        x2 = self.block1(skip0)
        skip1 = self.beta * x2 + skip0

        x3 = self.block2(skip1)
        return (skip1 + self.beta * x3) * self.beta + x


class DenseBlock(nn.Module):
    """
    Follows the design used in: https://arxiv.org/pdf/1809.00219v2.pdf
    Original design: https://arxiv.org/pdf/1608.06993v5.pdf
    """

    def __init__(self, channels, growth_rate, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.conv0 = nn.Conv2d(channels, growth_rate, (kernel_size, kernel_size),
                               (stride, stride), padding)
        self.conv1 = nn.Conv2d(channels + growth_rate, growth_rate, (kernel_size, kernel_size),
                               (stride, stride), padding)
        self.conv2 = nn.Conv2d(channels + 2 * growth_rate, growth_rate, (kernel_size, kernel_size),
                               (stride, stride), padding)
        self.conv3 = nn.Conv2d(channels + 3 * growth_rate, growth_rate, (kernel_size, kernel_size),
                               (stride, stride), padding)
        self.conv4 = nn.Conv2d(channels + 3 * growth_rate, channels, (kernel_size, kernel_size),
                               (stride, stride), padding)

        self.LReLu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.LReLu(self.conv0(x))
        x2 = self.LReLu(self.conv1(cat((x, x1), 1)))
        x3 = self.LReLu(self.conv2(cat((x, x1, x2), 1)))
        x4 = self.LReLu(self.conv3(cat((x, x1, x2, x3), 1)))
        return self.conv4(cat((x, x1, x2, x4), 1))
