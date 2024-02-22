import enum
from numpy.testing import build_err_msg
from jax import numpy as jnp
from jax._src import checkify
from jaxlib.xla_client import Traceback
import operator
from typing import Callable, Iterable

__all__ = [
    "assert_allclose",
    "assert_array_almost_equal",
    "assert_array_equal",
    "assert_array_compare",
    "assert_array_less",
]


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
        pass_: jnp.ndarray,
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
        self.pass_ = pass_
        self.reason = reason
        self.err_msg = err_msg
        self.verbose = verbose
        self.header = header
        self.precision = precision
        # We don't consider the debug field here, but it may arrive through unpacked
        # keyword arguments.

    def tree_flatten(self):
        children = (self.x, self.y, self.pass_, self.reason.value)
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
        x, y, pass_, reason = children
        return cls(x, y, pass_, TestaxErrorReason(reason), **aux_data)

    def __str__(self):
        has_val_by_reason = {
            TestaxErrorReason.NAN_MISMATCH: "nan",
            TestaxErrorReason.POSINF_MISMATCH: "+inf",
            TestaxErrorReason.NEGINF_MISMATCH: "-inf",
        }
        has_val = has_val_by_reason.get(self.reason)
        if has_val:
            return build_err_msg(
                [self.x, self.y],
                self.err_msg + f"\nx and y {has_val} location mismatch:",
                verbose=self.verbose,
                header=self.header,
                names=("x", "y"),
                precision=self.precision,
            )

        msg_by_reason = {
            TestaxErrorReason.SHAPE_MISMATCH: f"\n(shapes {self.x.shape}, {self.y.shape} mismatch)",
            TestaxErrorReason.DTYPE_MISMATCH: f"\n(dtypes {self.x.dtype}, {self.y.dtype} mismatch)",
        }
        msg = msg_by_reason.get(self.reason)
        if msg:
            return build_err_msg(
                [self.x, self.y],
                self.err_msg + msg,
                verbose=self.verbose,
                header=self.header,
                names=("x", "y"),
                precision=self.precision,
            )

        n_mismatch = self.pass_.size - self.pass_.sum()
        n_elements = self.pass_.size
        percent_mismatch = 100 * n_mismatch / n_elements
        remarks = [
            "Mismatched elements: {} / {} ({:.3g}%)".format(
                n_mismatch, n_elements, percent_mismatch
            )
        ]

        error = jnp.abs(self.x - self.y)
        if jnp.issubdtype(self.x.dtype, jnp.unsignedinteger):
            error2 = abs(self.y - self.x)
            error = jnp.minimum(error, error2)
        max_abs_error = jnp.max(error)
        remarks.append(f"Max absolute difference: {max_abs_error}")

        # note: this definition of relative error matches that one
        # used by assert_allclose (found in np.isclose)
        # Filter values where the divisor would be zero
        nonzero = self.y != 0
        if jnp.all(~nonzero):
            max_rel_error = float("inf")
        else:
            if nonzero.ndim > 0:
                max_rel_error = jnp.max(error[nonzero] / jnp.abs(self.y[nonzero]))
            else:
                max_rel_error = error / jnp.abs(self.y)
        remarks.append(f"Max relative difference: {max_rel_error}")

        err_msg = self.err_msg + "\n" + "\n".join(remarks)
        return build_err_msg(
            [self.x, self.y],
            err_msg,
            verbose=self.verbose,
            header=self.header,
            names=("x", "y"),
            precision=self.precision,
        )

    def get_effect_type(self):
        values: Iterable[jnp.ndarray] = checkify.jtu.tree_leaves(
            (self.x, self.y, self.pass_, self.reason.value)
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
    TestaxError.check((cond_x == cond_y).all(), x, y, None, reason, **kwargs)
    return cond_x & cond_y


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

    x = jnp.asarray(x)
    y = jnp.asarray(y)

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
                None,
                TestaxErrorReason.SHAPE_MISMATCH,
                traceback_info=checkify.get_traceback(),
                **kwargs,
            )
        else:
            raise TestaxError(
                x,
                y,
                None,
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
    TestaxError.check(pass_.all(), x, y, pass_, TestaxErrorReason.COMPARISON, **kwargs)


def assert_allclose(
    actual: jnp.ndarray,
    desired: jnp.ndarray,
    rtol: float = 1e-7,
    atol: float = 0,
    equal_nan: bool = True,
    err_msg: str = "",
    verbose: bool = True,
    *,
    debug: bool = False,
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
    # We need to cast to an inexact dtype to get the same behavior as numpy due to
    # https://github.com/google/jax/issues/19935.
    from jax._src.numpy.util import promote_args_inexact

    def comparison(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        x, y = promote_args_inexact("assert_allclose", x, y)
        return jnp.isclose(x, y, rtol=rtol, atol=atol, equal_nan=equal_nan)

    header = f"Not equal to tolerance rtol={rtol:g}, atol={atol:g}"
    assert_array_compare(
        comparison,
        actual,
        desired,
        err_msg=err_msg,
        verbose=verbose,
        header=header,
        equal_nan=equal_nan,
        debug=debug,
    )


def assert_array_equal(
    x: jnp.ndarray,
    y: jnp.ndarray,
    err_msg: str = "",
    verbose: bool = True,
    *,
    strict: bool = False,
    debug: bool = False,
) -> None:
    """
    Raises an AssertionError if two array_like objects are not equal.

    Given two array_like objects, check that the shape is equal and all
    elements of these objects are equal (but see the Notes for the special
    handling of a scalar). An exception is raised at shape mismatch or
    conflicting values. In contrast to the standard usage in numpy, NaNs
    are compared like numbers, no assertion is raised if both objects have
    NaNs in the same positions.

    The usual caution for verifying equality with floating point numbers is
    advised.

    Parameters
    ----------
    x : array_like
        The actual object to check.
    y : array_like
        The desired, expected object.
    err_msg : str, optional
        The error message to be printed in case of failure.
    verbose : bool, optional
        If True, the conflicting values are appended to the error message.
    strict : bool, optional
        If True, raise an AssertionError when either the shape or the data
        type of the array_like objects does not match. The special
        handling for scalars mentioned in the Notes section is disabled.

        .. versionadded:: 1.24.0

    Raises
    ------
    AssertionError
        If actual and desired objects are not equal.

    See Also
    --------
    assert_allclose: Compare two array_like objects for equality with desired
                     relative and/or absolute precision.
    assert_array_almost_equal_nulp, assert_array_max_ulp, assert_equal

    Notes
    -----
    When one of `x` and `y` is a scalar and the other is array_like, the
    function checks that each element of the array_like object is equal to
    the scalar. This behaviour can be disabled with the `strict` parameter.

    Examples
    --------
    The first assert does not raise an exception:

    >>> np.testing.assert_array_equal([1.0,2.33333,np.nan],
    ...                               [np.exp(0),2.33333, np.nan])

    Assert fails with numerical imprecision with floats:

    >>> np.testing.assert_array_equal([1.0,np.pi,np.nan],
    ...                               [1, np.sqrt(np.pi)**2, np.nan])
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not equal
    <BLANKLINE>
    Mismatched elements: 1 / 3 (33.3%)
    Max absolute difference: 4.4408921e-16
    Max relative difference: 1.41357986e-16
     x: array([1.      , 3.141593,      nan])
     y: array([1.      , 3.141593,      nan])

    Use `assert_allclose` or one of the nulp (number of floating point values)
    functions for these cases instead:

    >>> np.testing.assert_allclose([1.0,np.pi,np.nan],
    ...                            [1, np.sqrt(np.pi)**2, np.nan],
    ...                            rtol=1e-10, atol=0)

    As mentioned in the Notes section, `assert_array_equal` has special
    handling for scalars. Here the test checks that each value in `x` is 3:

    >>> x = np.full((2, 5), fill_value=3)
    >>> np.testing.assert_array_equal(x, 3)

    Use `strict` to raise an AssertionError when comparing a scalar with an
    array:

    >>> np.testing.assert_array_equal(x, 3, strict=True)
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not equal
    <BLANKLINE>
    (shapes (2, 5), () mismatch)
     x: array([[3, 3, 3, 3, 3],
           [3, 3, 3, 3, 3]])
     y: array(3)

    The `strict` parameter also ensures that the array data types match:

    >>> x = np.array([2, 2, 2])
    >>> y = np.array([2., 2., 2.], dtype=np.float32)
    >>> np.testing.assert_array_equal(x, y, strict=True)
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not equal
    <BLANKLINE>
    (dtypes int64, float32 mismatch)
     x: array([2, 2, 2])
     y: array([2., 2., 2.], dtype=float32)
    """
    assert_array_compare(
        operator.__eq__,
        x,
        y,
        err_msg=err_msg,
        verbose=verbose,
        header="Arrays are not equal",
        strict=strict,
        debug=debug,
    )


def assert_array_almost_equal(
    x: jnp.ndarray,
    y: jnp.ndarray,
    decimal: int = 6,
    err_msg: str = "",
    verbose: bool = True,
    *,
    debug: bool = False,
) -> None:
    """
    Raises an AssertionError if two objects are not equal up to desired
    precision.

    .. note:: It is recommended to use one of `assert_allclose`,
              `assert_array_almost_equal_nulp` or `assert_array_max_ulp`
              instead of this function for more consistent floating point
              comparisons.

    The test verifies identical shapes and that the elements of ``actual`` and
    ``desired`` satisfy.

        ``abs(desired-actual) < 1.5 * 10**(-decimal)``

    That is a looser test than originally documented, but agrees with what the
    actual implementation did up to rounding vagaries. An exception is raised
    at shape mismatch or conflicting values. In contrast to the standard usage
    in numpy, NaNs are compared like numbers, no assertion is raised if both
    objects have NaNs in the same positions.

    Parameters
    ----------
    x : array_like
        The actual object to check.
    y : array_like
        The desired, expected object.
    decimal : int, optional
        Desired precision, default is 6.
    err_msg : str, optional
      The error message to be printed in case of failure.
    verbose : bool, optional
        If True, the conflicting values are appended to the error message.

    Raises
    ------
    AssertionError
        If actual and desired are not equal up to specified precision.

    See Also
    --------
    assert_allclose: Compare two array_like objects for equality with desired
                     relative and/or absolute precision.
    assert_array_almost_equal_nulp, assert_array_max_ulp, assert_equal

    Examples
    --------
    the first assert does not raise an exception

    >>> np.testing.assert_array_almost_equal([1.0,2.333,np.nan],
    ...                                      [1.0,2.333,np.nan])

    >>> np.testing.assert_array_almost_equal([1.0,2.33333,np.nan],
    ...                                      [1.0,2.33339,np.nan], decimal=5)
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not almost equal to 5 decimals
    <BLANKLINE>
    Mismatched elements: 1 / 3 (33.3%)
    Max absolute difference: 6.e-05
    Max relative difference: 2.57136612e-05
     x: array([1.     , 2.33333,     nan])
     y: array([1.     , 2.33339,     nan])

    >>> np.testing.assert_array_almost_equal([1.0,2.33333,np.nan],
    ...                                      [1.0,2.33333, 5], decimal=5)
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not almost equal to 5 decimals
    <BLANKLINE>
    x and y nan location mismatch:
     x: array([1.     , 2.33333,     nan])
     y: array([1.     , 2.33333, 5.     ])

    """
    from jax._src.numpy.util import promote_args_inexact

    x, y = promote_args_inexact("assert_array_almost_equal", x, y)

    def compare(x, y):
        return jnp.abs(x - y) < 1.5 * 10.0 ** (-decimal)

    assert_array_compare(
        compare,
        x,
        y,
        err_msg=err_msg,
        verbose=verbose,
        header=("Arrays are not almost equal to %d decimals" % decimal),
        precision=decimal,
    )


def assert_array_less(
    x: jnp.ndarray,
    y: jnp.ndarray,
    err_msg: str = "",
    verbose: bool = True,
    *,
    debug: bool = False,
) -> None:
    """
    Raises an AssertionError if two array_like objects are not ordered by less
    than.

    Given two array_like objects, check that the shape is equal and all
    elements of the first object are strictly smaller than those of the
    second object. An exception is raised at shape mismatch or incorrectly
    ordered values. Shape mismatch does not raise if an object has zero
    dimension. In contrast to the standard usage in numpy, NaNs are
    compared, no assertion is raised if both objects have NaNs in the same
    positions.

    Parameters
    ----------
    x : array_like
      The smaller object to check.
    y : array_like
      The larger object to compare.
    err_msg : string
      The error message to be printed in case of failure.
    verbose : bool
        If True, the conflicting values are appended to the error message.

    Raises
    ------
    AssertionError
      If x is not strictly smaller than y, element-wise.

    See Also
    --------
    assert_array_equal: tests objects for equality
    assert_array_almost_equal: test objects for equality up to precision

    Examples
    --------
    >>> np.testing.assert_array_less([1.0, 1.0, np.nan], [1.1, 2.0, np.nan])
    >>> np.testing.assert_array_less([1.0, 1.0, np.nan], [1, 2.0, np.nan])
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not less-ordered
    <BLANKLINE>
    Mismatched elements: 1 / 3 (33.3%)
    Max absolute difference: 1.
    Max relative difference: 0.5
     x: array([ 1.,  1., nan])
     y: array([ 1.,  2., nan])

    >>> np.testing.assert_array_less([1.0, 4.0], 3)
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not less-ordered
    <BLANKLINE>
    Mismatched elements: 1 / 2 (50%)
    Max absolute difference: 2.
    Max relative difference: 0.66666667
     x: array([1., 4.])
     y: array(3)

    >>> np.testing.assert_array_less([1.0, 2.0, 3.0], [4])
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not less-ordered
    <BLANKLINE>
    (shapes (3,), (1,) mismatch)
     x: array([1., 2., 3.])
     y: array([4])

    """
    assert_array_compare(
        operator.__lt__,
        x,
        y,
        err_msg=err_msg,
        verbose=verbose,
        header="Arrays are not less-ordered",
        equal_inf=False,
        debug=debug,
    )
