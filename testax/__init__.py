import enum
import functools
from jax import numpy as jnp
from jax._src import checkify
from jaxlib.xla_client import Traceback
import numpy as np
from typing import Callable, Iterable


class TestaxErrorReason(enum.Enum):
    """
    Reason for a *testax* assertion to fail. The enum values are arrays so a change in
    failure reason does not trigger a new jit compilation.
    """

    SHAPE_MISMATCH = jnp.asarray(1)
    DTYPE_MISMATCH = jnp.asarray(2)
    NAN_MISMATCH = jnp.asarray(3)
    POSINF_MISMATCH = jnp.asarray(4)
    NEGINF_MISMATCH = jnp.asarray(5)
    COMPARISON = jnp.asarray(6)


class TestaxError(checkify.JaxException):
    """
    Raised when a *testax* assertion fails.
    """

    def __init__(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        reason: TestaxErrorReason,
        *,
        traceback_info: Traceback,
        err_msg: str,
        verbose: bool,
        header: str,
        precision: int,
        debug: bool = False,
    ) -> None:
        super().__init__(traceback_info)
        self.x = x
        self.y = y
        self.reason = reason
        self.err_msg = err_msg
        self.verbose = verbose
        self.header = header
        self.precision = precision
        # We don't consider the debug field here, but it may arrive through unpacked
        # keyword arguments.

    def tree_flatten(self):
        children = (self.x, self.y, self.reason.value)
        aux_data = {
            "traceback_info": self.traceback_info,
            "err_msg": self.err_msg,
            "verbose": self.verbose,
            "header": self.header,
            "precision": self.precision,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        x, y, reason = children
        return cls(x, y, TestaxErrorReason(reason), **aux_data)

    def __str__(self):
        return str(self.reason)

    def get_effect_type(self):
        values: Iterable[jnp.ndarray] = checkify.jtu.tree_leaves(
            (self.x, self.y, self.reason.value)
        )
        return checkify.ErrorEffect(
            TestaxError,
            tuple(checkify.api.ShapeDtypeStruct(x.shape, x.dtype) for x in values),
        )

    @classmethod
    def check(cls, predicate: bool, *args, debug: bool = False, **kwargs) -> None:
        new_error = cls(*args, traceback_info=checkify.get_traceback(), **kwargs)
        error = checkify.assert_func(
            checkify.init_error, jnp.logical_not(predicate), new_error
        )
        checkify._check_error(error, debug=debug)


def _assert_predicate_same_pos(
    x: jnp.ndarray,
    y: jnp.ndarray,
    reason: jnp.ndarray,
    predicate: Callable[..., jnp.ndarray],
    **kwargs,
) -> jnp.ndarray:
    """
    Assert that x and y satisfy the predicate at the same locations and return said
    locations if successful.
    """
    cond_x = predicate(x)
    cond_y = predicate(y)
    TestaxError.check((cond_x == cond_y).all(), x, y, reason, **kwargs)
    return cond_x & cond_y


@functools.wraps(np.testing.assert_array_compare)
def assert_array_compare(
    comparison: Callable[..., jnp.ndarray],
    x: jnp.ndarray,
    y: jnp.ndarray,
    err_msg: str = "",
    verbose: bool = True,
    header: str = "",
    precision: int = 6,
    equal_nan: bool = True,
    equal_inf: bool = True,
    *,
    strict: bool = False,
    debug: bool = False,
) -> None:
    """"""

    # Construct metadata to pass to the exception if required.
    kwargs = {
        "err_msg": err_msg,
        "verbose": verbose,
        "header": header,
        "precision": precision,
        "debug": debug,
    }

    # Check shapes and data types match.
    if strict:
        pass_ = x.shape == y.shape and x.dtype == y.dtype
    else:
        pass_ = (x.shape == () or y.shape == ()) or x.shape == y.shape
    if not pass_:
        if x.shape != y.shape:
            raise TestaxError(
                x,
                y,
                TestaxErrorReason.SHAPE_MISMATCH,
                traceback_info=checkify.get_traceback(),
                **kwargs,
            )
        else:
            raise TestaxError(
                x,
                y,
                TestaxErrorReason.DTYPE_MISMATCH,
                traceback_info=checkify.get_traceback(),
                **kwargs,
            )

    pass_ = comparison(x, y)
    if equal_nan:
        pass_ = pass_ | _assert_predicate_same_pos(
            x, y, TestaxErrorReason.NAN_MISMATCH, jnp.isnan, **kwargs
        )
    if equal_inf:
        pass_ = pass_ | _assert_predicate_same_pos(
            x, y, TestaxErrorReason.POSINF_MISMATCH, lambda z: z == +jnp.inf, **kwargs
        )
        pass_ = pass_ | _assert_predicate_same_pos(
            x, y, TestaxErrorReason.NEGINF_MISMATCH, lambda z: z == -jnp.inf, **kwargs
        )
    TestaxError.check(pass_.all(), x, y, TestaxErrorReason.COMPARISON, **kwargs)


def assert_allclose(
    actual, desired, rtol=1e-7, atol=0, equal_nan=True, err_msg="", verbose=True
) -> None:
    """
    Raises an AssertionError if two objects are not equal up to desired tolerance.

    Given two arrays, check that their shapes and all elements are equal (but see the
    Notes for the special handling of a scalar). An exception is raised if the shapes
    mismatch or any values conflict. In contrast to the standard usage in numpy, NaNs
    are compared like numbers, no assertion is raised if both objects have NaNs in the
    same positions.

    Args:
        actual: Array obtained.
        desired: Array desired.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        equal_nan: If True, NaNs will compare equal.
        err_msg: The error message to be printed in case of failure.
        verbose: If True, the conflicting values are appended to the error message.

    Raises:
        TestaxError: If actual and desired are not equal up to specified precision.

    Notes:

        When one of :code:`actual` and :code:`desired` is a scalar and the other is an
        array, the function checks that each element of the array is equal to the
        scalar.

    Examples:

        >>> x = jnp.asarray([1e-5, 1e-3, 1e-1])
        >>> y = jnp.arccos(jnp.cos(x))
        >>> testax.assert_allclose(x, y, rtol=1e-5, atol=0)
    """
    header = f"Not equal to tolerance rtol={rtol:g}, atol={atol:g}"
    assert_array_compare(
        lambda x, y: jnp.isclose(x, y, rtol, atol, equal_nan=equal_nan),
        actual,
        desired,
        err_msg=err_msg,
        verbose=verbose,
        header=header,
        equal_nan=equal_nan,
    )
