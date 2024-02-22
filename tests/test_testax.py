import contextlib
import functools
import jax
from jax import numpy as jnp
from jax.experimental import checkify
import numpy as np
import operator
import pytest
import testax
from typing import Callable, NamedTuple


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
    x: jnp.ndarray
    y: jnp.ndarray
    func: Callable
    kwargs: dict
    fail: bool


CONFIGURATIONS = [
    Configuration(
        x=jnp.arange(5),
        y=jnp.arange(5),
        func=testax.assert_array_compare,
        kwargs={"comparison": operator.eq},
        fail=False,
    ),
    Configuration(
        x=jnp.arange(5),
        y=jnp.arange(6),
        func=testax.assert_array_compare,
        kwargs={"comparison": operator.eq},
        fail=True,
    ),
    Configuration(
        x=jnp.arange(5),
        y=jnp.arange(5).astype(float),
        func=testax.assert_array_compare,
        kwargs={"comparison": operator.eq, "strict": True},
        fail=True,
    ),
]


@pytest.mark.parametrize("configuration", CONFIGURATIONS)
def test_assert_xyz(configuration: Configuration) -> None:
    @checkify.checkify
    def target(x, y):
        configuration.func(x=x, y=y, **configuration.kwargs)

    numpy_func = getattr(np.testing, configuration.func.__name__)

    if not configuration.fail:
        target(configuration.x, configuration.y)
        numpy_func(x=configuration.x, y=configuration.y, **configuration.kwargs)
        return

    with pytest.raises(testax.TestaxError) as testax_ex:
        target(configuration.x, configuration.y)

    with pytest.raises(AssertionError) as numpy_ex:
        numpy_func(x=configuration.x, y=configuration.y, **configuration.kwargs)

    assert testax_ex.exconly()
    assert numpy_ex.exconly()
