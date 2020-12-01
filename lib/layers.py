import torch.nn as nn


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

class GatedConv2dBN(nn.Module):
    """
    Based on https://github.com/riannevdberg/sylvester-flows/blob/master/models/layers.py
    """
    def __init__(self, input_channels, output_channels, stride, padding, kernel_size=5, dilation=1, activation=None):
        super(GatedConv2dBN, self).__init__()

        self.activation = activation
        self.sigmoid = nn.Sigmoid()

        self.h = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation)
        self.g = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation)
        self.bn_h = nn.BatchNorm2d(output_channels)
        self.bn_g = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        """
        BN(h(x)) * \sigmoid(BN(g(x)))
        """
        if self.activation is None:
            h = self.bn_h(self.h(x))
        else:
            h = self.activation(self.bn_h(self.h(x)))
        g = self.sigmoid(self.bn_g(self.g(x)))

        return h * g

class GatedConvTranspose2dBN(nn.Module):
    def __init__(self, input_channels, output_channels, stride, padding, kernel_size=5, output_padding=0, dilation=1,
                 activation=None):
        super(GatedConvTranspose2dBN, self).__init__()

        self.activation = activation
        self.sigmoid = nn.Sigmoid()

        self.h = nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding, output_padding,
                                     dilation=dilation)
        self.g = nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding, output_padding,
                                     dilation=dilation)
        self.bn_h = nn.BatchNorm2d(output_channels)
        self.bn_g = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        if self.activation is None:
            h = self.bn_h(self.h(x))
        else:
            h = self.activation(self.bn_h(self.h(x)))
        g = self.sigmoid(self.bn_g(self.g(x)))

        return h * g
