from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


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
    reshaped = input.view(batch, channel, new_height, kh, new_width, kw)
    
    # Reorder dimensions to get the desired output format
    # From: batch x channel x new_height x kh x new_width x kw
    # To: batch x channel x new_height x new_width x (kh * kw)
    out = reshaped.permute(0, 1, 2, 4, 3, 5)
    out = out.contiguous()
    out = out.view(batch, channel, new_height, new_width, kh * kw)
    
    return out, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    Tiled average pooling 2D

    Args:
        input: Tensor of shape (batch, channel, height, width)
        kernel: Tuple of (kernel_height, kernel_width)

    Returns:
        Tensor of shape (batch, channel, new_height, new_width)
    """
    # Use tile to reshape the input into tiles
    tiled, new_height, new_width = tile(input, kernel)
    
    # Calculate the mean over the last dimension (dim=4)
    # This will average over the kernel_height * kernel_width values
    pooled = tiled.mean(dim=4)
    
    # Ensure the output has the correct shape (batch, channel, new_height, new_width)
    return pooled.contiguous().view(input.shape[0], input.shape[1], new_height, new_width)


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor.

    Args:
        input: input tensor
        dim: dimension to reduce

    Returns:
        A 1-hot tensor with 1 in the position of the maximum value
    """
    # Get the maximum value along the specified dimension
    max_vals = input.max(dim)[0]
    
    # Create a mask that's 1 where input equals max_val, 0 elsewhere
    return (input == max_vals).float()


def max(input: Tensor, dim: int) -> Tensor:
    """Apply max reduction along a dimension.

    Args:
        input: input tensor
        dim: dimension to reduce

    Returns:
        Tensor containing max values along dim
    """
    # Make input contiguous
    input = input.contiguous()
    
    # Get the shape
    shape = list(input.shape)
    
    # Initialize with negative infinity
    out = input - input  # Creates tensor of zeros
    
    # For each position along the reduction dimension
    for i in range(shape[dim]):
        # Update out with larger values
        out = out + input * (tensor(1.0) - (input <= out))
    
    # Return the result with the correct shape
    return out.sum(dim) / tensor(shape[dim])


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax as a tensor.

    Args:
        input: input tensor
        dim: dimension to compute softmax over

    Returns:
        Tensor of same shape with softmax applied
    """
    # Subtract max for numerical stability
    max_val = input.max(dim)
    shifted = input - max_val
    
    # Compute exp
    exp_x = shifted.exp()
    
    # Normalize by sum
    return exp_x / exp_x.sum(dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax as a tensor.

    Args:
        input: input tensor
        dim: dimension to compute log softmax over

    Returns:
        Tensor of same shape with log softmax applied
    """
    # Use the log-sum-exp trick for numerical stability
    max_val = input.max(dim)
    shifted = input - max_val
    
    # Log-sum-exp trick: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    exp_shifted = shifted.exp()
    sum_exp = exp_shifted.sum(dim)
    
    return shifted - sum_exp.log()


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D.

    Args:
        input: Tensor of shape (batch, channel, height, width)
        kernel: height x width of pooling

    Returns:
        Tensor of shape (batch, channel, new_height, new_width)
    """
    # Use tile to reshape the input into tiles
    tiled, new_height, new_width = tile(input, kernel)
    
    # Take max over the kernel dimension (last dimension)
    pooled = tiled.max(dim=4)
    
    return pooled.view(input.shape[0], input.shape[1], new_height, new_width)


def dropout(input: Tensor, rate: float, is_training: bool = True) -> Tensor:
    """Dropout positions based on random noise.

    Args:
        input: input tensor
        rate: probability of dropping a position
        is_training: whether to apply dropout (True) or not (False)

    Returns:
        Tensor with dropout applied if is_training is True
    """
    if not is_training or rate == 0:
        return input
    
    # Generate random mask
    mask = rand(input.shape) > rate
    
    # Scale by 1/(1-rate) to maintain expected value
    scale = 1.0 / (1.0 - rate)
    return input * mask * scale


# TODO: Implement for Task 4.3.
