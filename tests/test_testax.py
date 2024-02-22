import jax
from jax import numpy as jnp
from jax.experimental import checkify
import numpy as np
import operator
import pytest
import testax


def test_logical_expressions_on_shapes():
    @jax.jit
    def assert_same_shape(x, y):
        assert x.shape == y.shape

    assert_same_shape(jnp.arange(3), jnp.arange(3))
    with pytest.raises(AssertionError):
        assert_same_shape(jnp.arange(3), jnp.arange(4))


def test_logical_expressions_on_dtypes():
    @jax.jit
    def assert_same_shape(x, y):
        assert x.dtype == y.dtype

    assert_same_shape(jnp.arange(3), jnp.arange(3))
    with pytest.raises(AssertionError):
        assert_same_shape(jnp.arange(3), jnp.arange(3).astype(float))


def assert_equal(x: jnp.ndarray, y: jnp.ndarray) -> None:
    cond = (x == y).all()
    a = jax.lax.cond(cond, lambda: x, lambda: jnp.empty([]))
    checkify.check(cond, "not equal", a=a)


@pytest.mark.parametrize(
    "comparison, x, y, reason",
    [
        (operator.eq, jnp.arange(5), jnp.arange(5), None),
        (operator.eq, jnp.arange(5), jnp.arange(6), None),
    ],
)
def test_assert_array_compare(comparison, x, y, reason) -> None:
    @checkify.checkify
    def target(a, b):
        testax.assert_array_compare(comparison, a, b)

    numpy_ex = None
    try:
        np.testing.assert_array_compare(comparison, x, y)
    except AssertionError as ex:
        numpy_ex = ex

    testax_ex = None
    try:
        err, _ = target(x, y)
        err.throw()
    except AssertionError as ex:
        testax_ex = ex

    assert (numpy_ex is None) == (testax_ex is None)


def test_custom_check() -> None:
    f = checkify.checkify(
        lambda: testax.check(False, "reason", {"a": 1}), {testax.TestaxError}
    )
    err, out = f()
    with pytest.raises(Exception):
        err.throw()
