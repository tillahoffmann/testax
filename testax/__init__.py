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
