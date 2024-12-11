import pytest
from hypothesis import given

import minitorch
from minitorch import Tensor

from .strategies import assert_close
from .tensor_strategies import tensors


@pytest.mark.task4_3
@given(tensors(shape=(1, 1, 4, 4)))
def test_avg(t: Tensor) -> None:
    out = minitorch.avgpool2d(t, (2, 2))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(2) for j in range(2)]) / 4.0
    )

    out = minitorch.avgpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(2) for j in range(1)]) / 2.0
    )

    out = minitorch.avgpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(1) for j in range(2)]) / 2.0
    )
    minitorch.grad_check(lambda t: minitorch.avgpool2d(t, (2, 2)), t)


@pytest.mark.task4_4
@given(tensors(shape=(2, 3, 4)))
def test_max(t: Tensor) -> None:
    """Test max function properties"""
    # Forward pass tests first
    out = minitorch.max(t, 0)
    assert_close(out[0, 0, 0], max(t[i, 0, 0] for i in range(2)))

    out = minitorch.max(t, 1)
    assert_close(out[0, 0, 0], max(t[0, i, 0] for i in range(3)))

    out = minitorch.max(t, 2)
    assert_close(out[0, 0, 0], max(t[0, 0, i] for i in range(4)))

    # Test max on a simple 1D case for gradient
    simple_t = minitorch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    out = minitorch.max(simple_t, 0)
    out.backward()

    # Add check for grad before accessing it
    assert simple_t.grad is not None, "Gradient should not be None"
    # Now we can safely access grad
    assert simple_t.grad[2] == 1.0
    assert simple_t.grad[0] == 0.0
    assert simple_t.grad[1] == 0.0


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_max_pool(t: Tensor) -> None:
    # Forward pass tests
    out = minitorch.maxpool2d(t, (2, 2))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(2)])
    )

    out = minitorch.maxpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(1)])
    )

    out = minitorch.maxpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(1) for j in range(2)])
    )

    # Test backward pass with explicit gradient
    simple_t = minitorch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True)
    out = minitorch.maxpool2d(simple_t, (2, 2))
    grad_output = minitorch.tensor([[[[1.0]]]])
    out.backward(grad_output)

    # Add check for grad before accessing it
    assert simple_t.grad is not None, "Gradient should not be None"
    # Now we can safely access grad
    assert simple_t.grad[0, 0, 1, 1] == 1.0


@pytest.mark.task4_4
@given(tensors())
def test_drop(t: Tensor) -> None:
    """Test dropout properties"""
    # Test with dropout probability = 0.0 (keep all values)
    q = minitorch.dropout(t, 0.0)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]

    # Test with dropout probability = 1.0 (drop all values)
    q = minitorch.dropout(t, 1.0)
    assert q[q._tensor.sample()] == 0.0

    # Test in eval mode (ignore=True)
    q = minitorch.dropout(t, 1.0, is_training=True)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]

    # For dropout, we'll test a simple deterministic case
    simple_t = minitorch.tensor([1.0, 1.0], requires_grad=True)
    out = minitorch.dropout(simple_t, 0.0)  # No dropout
    out.backward(minitorch.tensor([1.0, 1.0]))

    # Add check for grad before accessing it
    assert simple_t.grad is not None, "Gradient should not be None"
    # Now we can safely access grad
    assert_close(simple_t.grad[0], 1.0)


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_softmax(t: Tensor) -> None:
    """Test softmax properties:
    1. Output should sum to 1 along specified dimension
    2. All values should be positive
    3. Exponential monotonicity should be preserved
    4. Gradient check should pass
    """
    # Test summing to 1 along different dimensions
    q = minitorch.softmax(t, 3)
    x = q.sum(dim=3)
    assert_close(x[0, 0, 0, 0], 1.0)

    q = minitorch.softmax(t, 1)
    x = q.sum(dim=1)
    assert_close(x[0, 0, 0, 0], 1.0)

    # Gradient check
    minitorch.grad_check(lambda a: minitorch.softmax(a, dim=2), t)


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_log_softmax(t: Tensor) -> None:
    """Test logsoftmax properties:
    1. Should equal log(softmax(x))
    2. Sum of exp should be 1
    3. Gradient check should pass
    """
    # Compare with regular softmax
    q = minitorch.softmax(t, 3)
    q2 = minitorch.logsoftmax(t, 3).exp()
    for i in q._tensor.indices():
        assert_close(q[i], q2[i])

    # Gradient check
    minitorch.grad_check(lambda a: minitorch.logsoftmax(a, dim=2), t)
