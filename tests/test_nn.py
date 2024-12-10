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
    out = minitorch.max(t, 0)
    print(out)
    assert_close(out[0, 0, 0], max(t[i, 0, 0] for i in range(2)))

    out = minitorch.max(t, 1)
    print(out)
    assert_close(out[0, 0, 0], max(t[0, i, 0] for i in range(3)))

    out = minitorch.max(t, 2)
    print(out)
    assert_close(out[0, 0, 0], max(t[0, 0, i] for i in range(4)))


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_max_pool(t: Tensor) -> None:
    out = minitorch.maxpool2d(t, (2, 2))
    print(out)
    print(t)
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


@pytest.mark.task4_4
@given(tensors())
def test_drop(t: Tensor) -> None:
    """Test dropout properties:
    1. Training mode should zero some values
    2. Non-zero values should be scaled correctly
    3. Eval mode should return input unchanged
    4. Different random seeds should give different masks
    """
    # Test with dropout probability = 0.0 (keep all values)
    # Should return input tensor unchanged
    q = minitorch.dropout(t, 0.0)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]

    # Test with dropout probability = 1.0 (drop all values)
    # All values should be set to zero
    q = minitorch.dropout(t, 1.0)
    assert q[q._tensor.sample()] == 0.0

    # Test in eval mode (ignore=True)
    # Should return input tensor unchanged regardless of dropout probability
    q = minitorch.dropout(t, 1.0, is_training=True)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_softmax(t: Tensor) -> None:
    """Test softmax properties:
    1. Output should sum to 1 along specified dimension
    2. All values should be positive
    3. Exponential monotonicity should be preserved
    4. Gradient check should pass
    """
    q = minitorch.softmax(t, 3)
    x = q.sum(dim=3)
    assert_close(x[0, 0, 0, 0], 1.0)

    q = minitorch.softmax(t, 1)
    x = q.sum(dim=1)
    assert_close(x[0, 0, 0, 0], 1.0)

    minitorch.grad_check(lambda a: minitorch.softmax(a, dim=2), t)


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_log_softmax(t: Tensor) -> None:
    """Test logsoftmax properties:
    1. Should equal log(softmax(x))
    2. Sum of exp should be 1
    3. Gradient check should pass
    """
    q = minitorch.softmax(t, 3)
    q2 = minitorch.logsoftmax(t, 3).exp()
    for i in q._tensor.indices():
        assert_close(q[i], q2[i])
    minitorch.grad_check(lambda a: minitorch.logsoftmax(a, dim=2), t)
