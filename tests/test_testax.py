from functools import partial
import jax
from jax import numpy as jnp
from jax.experimental import checkify
import numpy as np
import operator
import pytest
import testax
from typing import Callable, NamedTuple, Optional


@pytest.fixture(params=[False, True])
def maybe_jit(request: pytest.FixtureRequest) -> Callable:
    if request.param:
        return jax.jit
    else:
        return lambda x, *args, **kwargs: x


def test_logical_expressions_on_shapes(maybe_jit: Callable) -> None:
    @maybe_jit
    def assert_same_shape(x, y):
        assert x.shape == y.shape

    assert_same_shape(jnp.arange(3), jnp.arange(3))
    with pytest.raises(AssertionError):
        assert_same_shape(jnp.arange(3), jnp.arange(4))


def test_logical_expressions_on_dtypes(maybe_jit: Callable) -> None:
    @maybe_jit
    def assert_same_shape(x, y):
        assert x.dtype == y.dtype

    assert_same_shape(jnp.arange(3), jnp.arange(3))
    with pytest.raises(AssertionError):
        assert_same_shape(jnp.arange(3), jnp.arange(3).astype(float))


class Configuration(NamedTuple):
    fail: bool
    x: jnp.ndarray
    y: jnp.ndarray
    testax_func: Callable
    numpy_func: Optional[Callable] = None
    kwargs: Optional[dict] = None


CONFIGURATIONS = [
    Configuration(
        fail=False,
        x=jnp.arange(5),
        y=jnp.arange(5),
        testax_func=partial(testax.assert_array_compare, operator.eq),
        numpy_func=partial(np.testing.assert_array_compare, operator.eq),
    ),
    Configuration(
        fail=True,
        x=jnp.arange(5),
        y=jnp.arange(6),
        testax_func=partial(testax.assert_array_compare, operator.eq),
        numpy_func=partial(np.testing.assert_array_compare, operator.eq),
    ),
    Configuration(
        fail=True,
        x=jnp.arange(5),
        y=jnp.arange(5).astype(float),
        testax_func=partial(testax.assert_array_compare, operator.eq),
        numpy_func=partial(np.testing.assert_array_compare, operator.eq),
        kwargs={"strict": True},
    ),
    Configuration(
        fail=False,
        x=jnp.linspace(0, 1),
        y=jnp.linspace(0, 1) + 0.1,
        testax_func=testax.assert_allclose,
        kwargs={"atol": 0.11},
    ),
    Configuration(
        fail=True,
        x=jnp.zeros(3),
        y=jnp.zeros(3) + 0.1,
        testax_func=testax.assert_allclose,
        kwargs={"atol": 0.09},
    ),
    Configuration(
        fail=False,
        x=jnp.ones(3),
        y=jnp.ones(3) * 1.2,
        testax_func=testax.assert_allclose,
        kwargs={"rtol": 0.2},
    ),
]


@pytest.mark.parametrize("configuration", CONFIGURATIONS)
def test_assert_xyz(configuration: Configuration) -> None:
    kwargs = configuration.kwargs or {}

    @testax.checkify
    def target(x, y):
        configuration.testax_func(x, y, **kwargs)

    numpy_func = configuration.numpy_func or getattr(
        np.testing, configuration.testax_func.__name__
    )

    if not configuration.fail:
        err, _ = target(configuration.x, configuration.y)
        err.throw()
        numpy_func(configuration.x, configuration.y, **kwargs)
        return

    with pytest.raises((testax.TestaxError, checkify.JaxRuntimeError)) as testax_ex:
        err, _ = target(configuration.x, configuration.y)
        err.throw()

    with pytest.raises(AssertionError) as numpy_ex:
        numpy_func(configuration.x, configuration.y, **kwargs)

    assert testax_ex.exconly()
    assert numpy_ex.exconly()


def test_nan_mismatch() -> None:
    with pytest.raises(checkify.JaxRuntimeError):
        testax.assert_allclose(jnp.zeros(1), jnp.nan * jnp.zeros(1))


def test_rel_err_zero():
    with pytest.raises(checkify.JaxRuntimeError):
        testax.assert_allclose(jnp.ones(1), jnp.zeros(1))
