from torch.nn import functional as F

from OpenPCSeg.package.torchsparse.torchsparse.tensor import SparseTensor
from OpenPCSeg.package.torchsparse.torchsparse.nn.utils import fapply

__all__ = ['relu', 'leaky_relu']


def relu(input: SparseTensor, inplace: bool = True) -> SparseTensor:
    return fapply(input, F.relu, inplace=inplace)


def leaky_relu(input: SparseTensor,
               negative_slope: float = 0.1,
               inplace: bool = True) -> SparseTensor:
    return fapply(input,
                  F.leaky_relu,
                  negative_slope=negative_slope,
                  inplace=inplace)
