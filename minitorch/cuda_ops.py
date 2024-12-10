# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """JIT a function for the device.

    Args:
    ----
        fn (Fn): Function to be JIT compiled
        **kwargs (Any): Additional arguments to pass to the JIT compiler

    Returns:
    -------
        Fn: JIT compiled function for the device

    """
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Fn, **kwargs: Any) -> FakeCUDAKernel:
    """JIT a function for the host.

    Args:
    ----
        fn (Fn): Function to be JIT compiled
        **kwargs (Any): Additional arguments to pass to the JIT compiler

    Returns:
    -------
        FakeCUDAKernel: JIT compiled function

    """
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """Creates a CUDA-accelerated function that applies a binary operation element-wise to two tensors.

        This method takes a binary function that operates on scalar float values and converts it into a
        CUDA-optimized function that can operate on entire tensors in parallel. The resulting function
        handles broadcasting between tensors of different shapes.

        Args:
        ----
            fn (Callable[[float, float], float]): A binary function that takes two float values
                and returns a float value. This is the operation to be applied element-wise.

        Returns:
        -------
            Callable[[Tensor, Tensor], Tensor]: A function that takes two tensors as input and returns
                a new tensor containing the element-wise application of the original function.
                The output tensor's shape is determined by broadcasting rules between the input tensors.

        Note:
        ----
            - The function automatically handles CUDA memory management and kernel launching.
            - The operation is parallelized using CUDA threads and blocks.
            - Input tensors are automatically broadcast to compatible shapes if necessary.
            - The implementation uses the THREADS_PER_BLOCK constant for CUDA thread organization.

        """
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Creates a CUDA-accelerated reduction function that operates along a specified dimension of a tensor.

        This method takes a binary reduction function and converts it into a CUDA-optimized function that can
        perform reduction operations (like sum, max, etc.) along a specified dimension of a tensor. The operation
        is parallelized using CUDA threads for efficient computation.

        Args:
        ----
            fn (Callable[[float, float], float]): A binary reduction function that takes two float values
                and returns a float value. This function should be associative and commutative for correct results
                (e.g., addition, multiplication, max, min).
            start (float, optional): The initial value for the reduction operation. Defaults to 0.0.
                This value should be the identity element for the reduction operation
                (e.g., 0 for addition, 1 for multiplication, -inf for max, inf for min).

        Returns:
        -------
            Callable[[Tensor, int], Tensor]: A function that takes a tensor and a dimension index as input
                and returns a new tensor with the reduction operation applied along the specified dimension.
                The output tensor's shape will have the specified dimension reduced to ceil(original_dim/1024).

        Note:
        ----
            - The implementation uses a fixed thread block size of 1024 for parallel reduction.
            - The reduction is performed in parallel across CUDA threads within each block.
            - The output tensor's size along the reduced dimension is ceil(input_dim/1024).
            - Memory management and kernel launching are handled automatically.

        """
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Performs matrix multiplication between two tensors with CUDA optimization.

        Handles both 2D and 3D tensor inputs by automatically promoting 2D tensors to 3D.
        Supports broadcasting of batch dimensions for tensors with different batch sizes.

        Args:
        ----
            a (Tensor): First input tensor of shape (..., n, m) or (n, m)
            b (Tensor): Second input tensor of shape (..., m, p) or (m, p)

        Returns:
        -------
            Tensor: Output tensor of shape (..., n, p), where ... represents broadcast batch dimensions.
                If both inputs are 2D, output will be 2D of shape (n, p)

        Raises:
        ------
            AssertionError: If the inner dimensions don't match (a.shape[-1] != b.shape[-2])

        Note:
        ----
            - Uses CUDA parallelization with block size defined by THREADS_PER_BLOCK
            - Automatically handles dimension promotion and broadcasting
            - Preserves input dimensionality in output (2D inputs → 2D output)

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if i < out_size:
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            o = index_to_position(out_index, out_strides)
            j = index_to_position(in_index, in_strides)
            out[o] = fn(in_storage[j])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # Local memory for indices
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)

        # Calculate thread position
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # Check bounds before computation
        if i < out_size:
            # Convert position to indices
            to_index(i, out_shape, out_index)

            # Handle broadcasting for both inputs
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)

            # Calculate storage positions
            out_pos = index_to_position(out_index, out_strides)
            a_pos = index_to_position(a_index, a_strides)
            b_pos = index_to_position(b_index, b_strides)

            # Apply binary function
            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    r"""Practice sum kernel that sums elements in blocks using shared memory.

    Given an array of length n and out of size n // blockDIM, it sums up each
    blockDim values into an out cell.

    Args:
    ----
        out (Storage): Output storage array to store block-wise sums
        a (Storage): Input storage array to be summed
        size (int): Size of the input array

    Note:
    ----
        - Uses shared memory within each block for efficient reduction
        - Each block computes a partial sum of blockDim elements
        - Final output contains one sum per block

    """
    BLOCK_DIM = 32
    cache = cuda.shared.array(BLOCK_DIM, numba.float64)

    # Calculate thread indices
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    # Load data into shared memory with bounds check
    if i < size:
        val = float(a[i])
        cache[pos] = val
        cuda.syncthreads()
    else:
        cache[pos] = 0

    if i < size:
        for j in [1, 2, 4, 8, 16]:
            if pos % (j * 2) == 0:
                cache[pos] += cache[pos + j]
                cuda.syncthreads()
        if pos == 0:
            out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """CUDA sum practice.

    This function serves as an example of how to implement a simple CUDA kernel
    in Python. It takes a single tensor as input and returns a tensor with a
    single element containing the sum of all elements in the input tensor.

    The function first calculates the number of blocks needed based on the size
    of the input tensor and the number of threads per block. It then allocates a
    tensor with a single element and copies it to the CUDA device. The CUDA
    kernel is then launched with the specified number of blocks and threads per
    block. The kernel performs a parallel reduction of the input tensor and
    stores the result in the output tensor.

    Args:
    ----
        a (Tensor): The input tensor to be summed.

    Returns:
    -------
        TensorData: A tensor containing the sum of all elements in the input
            tensor.

    Raises:
    ------
        ValueError: If the input tensor is empty or has more than one element.

    """
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x
        cache[pos] = reduce_value

        if out_pos < out_size:
            to_index(out_pos, out_shape, out_index)
            o = index_to_position(out_index, out_strides)

            out_index[reduce_dim] = out_index[reduce_dim] * BLOCK_DIM + pos
            if out_index[reduce_dim] < a_shape[reduce_dim]:
                in_a = index_to_position(out_index, a_strides)
                cache[pos] = a_storage[in_a]
                cuda.syncthreads()
                x = 0
                while 2**x < BLOCK_DIM:
                    j = 2**x
                    if pos % (j * 2) == 0:
                        cache[pos] = fn(cache[pos], cache[pos + j])
                    cuda.syncthreads()
                    x += 1
            if pos == 0:
                out[o] = cache[0]

    return cuda.jit()(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """Generates a practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32

    # Shared memory for tiles
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Thread indices
    i = cuda.threadIdx.x
    j = cuda.threadIdx.y

    # Bounds check
    if i >= size or j >= size:
        return

    # Load matrices into shared memory
    a_shared[i, j] = a[size * i + j]
    b_shared[i, j] = b[size * i + j]
    cuda.syncthreads()

    # Compute matrix multiplication
    accum = 0.0
    for k in range(size):
        accum += a_shared[i, k] * b_shared[k, j]

    # Write result
    out[size * i + j] = accum


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Practice matrix multiplication using a single kernel launch.

    Given two tensors `a` and `b` of shape `(size, size)`, compute the matrix
    multiplication of `a` and `b` using a single kernel launch. The result is
    returned as a new tensor of the same shape.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Args:
    ----
        a (Tensor): First matrix to multiply.
        b (Tensor): Second matrix to multiply.

    Returns:
    -------
        TensorData: Result of the matrix multiplication.

    Notes:
    -----
        * This is a practice function to help you prepare for the matrix
          multiplication assignment.
        * This function will be used as a reference implementation to validate
          your solution.

    """
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    # Block and grid configuration
    BLOCK_DIM = 32

    # Handle batch dimension with broadcasting
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    batch = cuda.blockIdx.z

    # Shared memory allocation for tiles
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Global indices (position in output matrix)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # Local thread indices (position in shared memory tile)
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Accumulator for dot product
    accum = 0.0

    # Iterate over tiles in the shared dimension
    for phase in range(0, a_shape[2], BLOCK_DIM):
        # Load tile from matrix A into shared memory
        k = phase + pj
        if i < a_shape[1] and k < a_shape[2]:
            a_pos = batch * a_batch_stride + i * a_strides[1] + k * a_strides[2]
            a_shared[pi, pj] = a_storage[a_pos]
        else:
            a_shared[pi, pj] = 0.0

        # Load tile from matrix B into shared memory
        k = phase + pi
        if k < b_shape[1] and j < b_shape[2]:
            b_pos = batch * b_batch_stride + k * b_strides[1] + j * b_strides[2]
            b_shared[pi, pj] = b_storage[b_pos]
        else:
            b_shared[pi, pj] = 0.0

        # Ensure all threads have loaded their data
        cuda.syncthreads()

        # Compute dot product for this tile
        if i < out_shape[1] and j < out_shape[2]:
            for k in range(min(BLOCK_DIM, a_shape[2] - phase)):
                accum += a_shared[pi, k] * b_shared[k, pj]

        # Sync before next iteration
        cuda.syncthreads()

    # Write result to global memory
    if i < out_shape[1] and j < out_shape[2]:
        out_pos = batch * out_strides[0] + i * out_strides[1] + j * out_strides[2]
        out[out_pos] = accum


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
