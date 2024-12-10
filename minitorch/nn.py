from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off

max_reduce = FastOps.reduce(operators.max, -1e9)

def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.
    
    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    
    new_height = height // kh
    new_width = width // kw
    
    # Make input contiguous before reshaping
    input = input.contiguous()
    
    # Reshape the input tensor to create tiles
    input = input.view(batch, channel, new_height, kh, new_width, kw)
    
    # Reorder dimensions to get the desired output format
    # From: batch x channel x new_height x kh x new_width x kw
    # To: batch x channel x new_height x new_width x (kh * kw)
    input = input.permute(0, 1, 2, 4, 3, 5)
    input = input.contiguous()
    out = input.view(batch, channel, new_height, new_width, kh * kw)
    
    return out, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D.

    Args:
        input: Tensor of shape (batch, channel, height, width)
        kernel: Tuple of (kernel_height, kernel_width)

    Returns:
        Tensor of shape (batch, channel, new_height, new_width)

    """
    batch, channel, height, width = input.shape
    input, h, w = tile(input, kernel)
    input = input.mean(4)
    out = input.view(batch, channel, h, w)
    return out

def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor.

    Args:
        input: input tensor
        dim: dimension to reduce

    Returns:
        A 1-hot tensor with 1 in the position of the maximum value
    
    """
    # Get the maximum value along the specified dimension
    out = max_reduce(input, dim)
    return input == out

class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: int) -> Tensor:
        """Max forward pass"""
        dims = int(dim.item())
        ctx.save_for_backward(input, dim)
        return max_reduce(input, dims)
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, None]:
        """Max backward pass"""
        input, dim = ctx.saved_values
        dims = int(dim.item())
        return grad_output * argmax(input, dims), 0.0

def max(input: Tensor, dim: int) -> Tensor:
    """Apply max reduction along a dimension.

    Args:
        input: input tensor
        dim: dimension to reduce

    Returns:
        Tensor containing max values along dim

    """
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax as a tensor.

    Args:
        input: input tensor
        dim: dimension to compute softmax over

    Returns:
        Tensor of same shape with softmax applied
    """
    input = input.exp()
    return input / input.sum(dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax as a tensor.

    Args:
        input: input tensor
        dim: dimension to compute log softmax over

    Returns:
        Tensor of same shape with log softmax applied
    """
    return softmax(input, dim).log()


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D.

    Args:
        input: Tensor of shape (batch, channel, height, width)
        kernel: height x width of pooling

    Returns:
        Tensor of shape (batch, channel, new_height, new_width)
    """
    batch, channel, height, width = input.shape
    input, h, w = tile(input, kernel)
    input = max(input, 4)
    out = input.view(batch, channel, h, w)
    return out


def dropout(input: Tensor, rate: float, is_training: bool = False) -> Tensor:
    """Dropout positions based on random noise.

    Args:
        input: input tensor
        rate: probability of dropping a position
        is_training: whether to apply dropout (True) or not (False)

    Returns:
        Tensor with dropout applied if is_training is True
    """
    if not is_training:
        drop = rand(input.shape, input.backend) > rate
        input = drop * input
    return input
