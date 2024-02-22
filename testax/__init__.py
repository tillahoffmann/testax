from dataclasses import dataclass
import enum
import functools
import jax
from jax import numpy as jnp
from jax._src import checkify
import numpy as np
from typing import Callable, Dict, Iterable


class TestaxReason(enum.Enum):
    DTYPE_MISMATCH = 1
    SHAPE_MISMATCH = 2
    NAN_MISMATCH = 3
    POSINF_MISMATCH = 4
    NEGINF_MISMATCH = 5
    COMPARISON_FAILED = 6


@dataclass
class TestaxErrorMetadata:
    err_msg: str
    verbose: bool
    header: str
    precision: int
    reason: TestaxReason


class TestaxError(checkify.JaxException):
    """
    Exception raised when a *testax* assertion fails.
    """

    def __init__(self, traceback_info, reason, metadata, *args, **kwargs):
        super().__init__(traceback_info)
        self.reason = reason
        self.metadata = metadata
        self.args = args
        self.kwargs = kwargs

    def tree_flatten(self):
        return (
            (self.args, self.kwargs),
            (self.traceback_info, self.reason, self.metadata),
        )

    @classmethod
    def tree_unflatten(cls, metadata, payload):
        args, kwargs = payload
        return cls(*metadata, *args, **kwargs)

    def __str__(self):
        return "whoop"

    def get_effect_type(self):
        values: Iterable[jnp.ndarray] = checkify.jtu.tree_leaves(
            (self.args, self.kwargs)
        )
        return checkify.ErrorEffect(
            TestaxError,
            tuple(checkify.api.ShapeDtypeStruct(x.shape, x.dtype) for x in values),
        )


def check(pred, reason, metadata, debug=False, *args, **kwargs):
    for arg in jax.tree_util.tree_leaves((args, kwargs)):
        if not isinstance(arg, (jnp.ndarray, np.ndarray)):
            raise TypeError(
                "Arguments to testax.check need to be PyTrees of arrays, but got "
                f"{arg!r} of type {type(arg)}."
            )
    new_error = TestaxError(checkify.get_traceback(), reason, metadata, *args, **kwargs)
    error = checkify.assert_func(checkify.init_error, jnp.logical_not(pred), new_error)
    checkify._check_error(error, debug=debug)


def assert_predicate_same_pos(
    x: jnp.ndarray,
    y: jnp.ndarray,
    predicate: Callable[..., jnp.ndarray],
    reason: TestaxReason,
    metadata: Dict,
) -> jnp.ndarray:
    """
    Assert that x and y satisfy the predicate at the same locations and return said
    locations if successful.
    """
    cond_x = predicate(x)
    cond_y = predicate(y)
    check((cond_x == cond_y).all(), reason, metadata, x, y)
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
) -> None:
    """ """

    # Construct metadata to pass to the exception if required.
    metadata = {
        "precision": precision,
        "err_msg": err_msg,
        "verbose": verbose,
        "header": header,
    }

    # Check shapes and data types match.
    if strict:
        cond = x.shape == y.shape and x.dtype == y.dtype
    else:
        cond = (x.shape == () or y.shape == ()) or x.shape == y.shape
    if not cond:
        if x.shape != y.shape:
            reason = TestaxReason.SHAPE_MISMATCH
        else:
            reason = TestaxReason.DTYPE_MISMATCH
        raise AssertionError(
            "we can raise this here if we want because these are static"
        )

    # Check nans
    ok = jnp.zeros(jnp.broadcast_shapes(x.shape, y.shape), dtype=bool)
    if equal_nan:
        ok = ok | assert_predicate_same_pos(
            x, y, jnp.isnan, TestaxReason.NAN_MISMATCH, metadata
        )
    if equal_inf:
        ok = ok | assert_predicate_same_pos(
            x, y, lambda z: z == +jnp.inf, TestaxReason.POSINF_MISMATCH, metadata
        )
        ok = ok | assert_predicate_same_pos(
            x, y, lambda z: z == -jnp.inf, TestaxReason.NEGINF_MISMATCH, metadata
        )
    ok = ok | comparison(x, y)
    check(
        ok.all(),
        TestaxReason.COMPARISON_FAILED,
        metadata,
    )
