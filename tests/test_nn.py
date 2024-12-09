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
    dim = 0  # Test just one dimension for now
    max_val = minitorch.max(t, dim)
    
    # Convert to Python values for easier debugging
    t_vals = t.to_numpy()
    max_vals = max_val.to_numpy()
    
    # Check max is correct
    for i in range(t.shape[1]):
        for j in range(t.shape[2]):
            expected = max(t_vals[k][i][j] for k in range(t.shape[0]))
            # Use numpy's item() to get a scalar value
            assert abs(float(max_vals[i][j]) - expected) < 1e-5


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
    rate = 0.5
    
    # Test training mode
    out1 = minitorch.dropout(t, rate, is_training=True)
    
    # Some values should be 0
    assert (out1 == 0).sum() > 0
    
    # Non-zero values should be scaled by 1/(1-rate)
    scale = 1.0 / (1-rate)
    nonzero_mask = out1 != 0
    assert_close(out1[nonzero_mask] / t[nonzero_mask], tensor([scale] * nonzero_mask.sum()))
    
    # Test eval mode
    out2 = minitorch.dropout(t, rate, is_training=False)
    assert_close(out2, t)
    
    # Different calls should give different masks
    out3 = minitorch.dropout(t, rate, is_training=True)
    assert not (out1 == out3).all()


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_softmax(t: Tensor) -> None:
    """Test softmax properties:
    1. Output should sum to 1 along specified dimension
    2. All values should be positive
    3. Exponential monotonicity should be preserved
    4. Gradient check should pass
    """
    for dim in range(len(t.shape)):
        s = minitorch.softmax(t, dim)
        
        # Sum should be close to 1
        sum_along_dim = s.sum(dim)
        assert_close(sum_along_dim, tensor([1.0] * sum_along_dim.size))
        
        # All values should be positive
        assert (s >= 0).all()
        
        # Check monotonicity: if a > b then softmax(a) > softmax(b)
        original = t.sum(dim)
        softmaxed = s.sum(dim)
        for i in range(original.size - 1):
            if original[i] > original[i + 1]:
                assert softmaxed[i] > softmaxed[i + 1]
    
    # Gradient check
    minitorch.grad_check(lambda t: minitorch.softmax(t, dim=1), t)


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_log_softmax(t: Tensor) -> None:
    """Test logsoftmax properties:
    1. Should equal log(softmax(x))
    2. Sum of exp should be 1
    3. Gradient check should pass
    """
    for dim in range(len(t.shape)):
        # Test that logsoftmax equals log(softmax)
        s = minitorch.softmax(t, dim)
        ls = minitorch.logsoftmax(t, dim)
        assert_close(ls, s.log())
        
        # Test that exp(logsoftmax) sums to 1
        sum_exp = ls.exp().sum(dim)
        assert_close(sum_exp, tensor([1.0] * sum_exp.size))
    
    # Gradient check
    minitorch.grad_check(lambda t: minitorch.logsoftmax(t, dim=1), t)


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_maxpool2d(t: Tensor) -> None:
    """Test maxpool2d properties:
    1. Output shape should be correct
    2. Max values should match manual computation
    3. Gradient check should pass
    """
    # Test with different kernel sizes
    for kernel in [(2, 2), (2, 1), (1, 2)]:
        out = minitorch.maxpool2d(t, kernel)
        
        # Check output shape
        kh, kw = kernel
        expected_h = t.shape[2] // kh
        expected_w = t.shape[3] // kw
        assert out.shape == (t.shape[0], t.shape[1], expected_h, expected_w)
        
        # Check first pooling window manually
        assert_close(
            out[0, 0, 0, 0],
            max([t[0, 0, i, j] for i in range(kh) for j in range(kw)])
        )
    
    # Gradient check
    minitorch.grad_check(lambda t: minitorch.maxpool2d(t, (2, 2)), t)
