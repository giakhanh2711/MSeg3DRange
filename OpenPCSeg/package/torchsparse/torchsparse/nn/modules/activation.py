from torch import nn

from OpenPCSeg.package.torchsparse.torchsparse.tensor import SparseTensor
from OpenPCSeg.package.torchsparse.torchsparse.nn.utils import fapply

__all__ = ['ReLU', 'LeakyReLU']


class ReLU(nn.ReLU):

    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)


class LeakyReLU(nn.LeakyReLU):

    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)
