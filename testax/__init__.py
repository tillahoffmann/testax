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
        max_abs_error = jnp.nanmax(error)
        remarks.append(f"Max absolute difference: {max_abs_error:.{self.precision}g}")

        # note: this definition of relative error matches that one
        # used by assert_allclose (found in jnp.isclose)
        # Filter values where the divisor would be zero
        nonzero = self.y != 0
        if jnp.all(~nonzero):
            max_rel_error = float("inf")
        else:
            if nonzero.ndim > 0:
                max_rel_error = jnp.nanmax(error[nonzero] / jnp.abs(self.y[nonzero]))
            else:
                max_rel_error = jnp.nanmax(error / jnp.abs(self.y))
        remarks.append(f"Max relative difference: {max_rel_error:.{self.precision}g}")

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
    Raises an AssertionError if two objects are not equal up to desired
    tolerance.

    Given two :class:`~jax.Array`\ s, check that their shapes and all elements
    are equal (but see the Notes for the special handling of a scalar). An
    exception is raised if the shapes mismatch or any values conflict. In
    contrast to the standard usage in numpy, :code:`nan`\ s are compared like numbers,
    no assertion is raised if both objects have :code:`nan`\ s in the same positions.

    The test is equivalent to :code:`allclose(actual, desired, rtol, atol)` (note
    that :func:`~.jax.numpy.allclose` has different default values). It compares the
    difference between :code:`actual` and :code:`desired` to
    :code:`atol + rtol * abs(desired)`.

    Args:
        actual: Array obtained.
        desired: Array desired.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        equal_nan: If True, :code:`nan`\ s will compare equal.
        err_msg: The error message to be printed in case of failure.
        verbose: If True, the conflicting values are appended to the error message.

    Raises:
        AssertionError: If actual and desired are not equal up to specified precision.

    See Also:
        :func:`assert_array_almost_equal_nulp`, :func:`assert_array_max_ulp`

    Notes:
        When one of :code:`actual` and :code:`desired` is a scalar and the other is an
        :class:`~jax.Array`, the function checks that each element of the
        :class:`~jax.Array` is equal to the scalar.

    Examples:
        >>> x = jnp.asarray([1e-5, 1e-3, 1e-1])
        >>> y = jnp.arccos(jnp.cos(x))
        >>> testax.assert_allclose(x, y, atol=1e-4)
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
    Raises an :class:`AssertionError` if two :class:`~jax.Array`\ s are not equal.

    Given two :class:`~jax.Array`\ s, check that the shape is equal and all elements of
    these objects are equal (but see the Notes for the special handling of a scalar).
    An exception is raised at shape mismatch or conflicting values. In contrast to the
    standard usage in numpy, :code:`nan`\ s are compared like numbers, no assertion is
    raised if both objects have :code:`nan`\ s in the same positions.

    The usual caution for verifying equality with floating point numbers is advised.

    Args:
        x: The actual object to check.
        y: The desired, expected object.
        err_msg: The error message to be printed in case of failure.
        verbose: If True, the conflicting values are appended to the error message.
        strict: If True, raise an AssertionError when either the shape or the data type
            of the :class:`~jax.Array`\ s does not match. The special handling for
            scalars mentioned in the Notes section is disabled.

    Raises:
        AssertionError: If actual and desired objects are not equal.

    See Also:
        :func:`assert_allclose`, :func:`assert_array_almost_equal_nulp`,
        :func:`assert_array_max_ulp`, :func:`assert_equal`

    Notes:
        When one of :code:`x` and :code:`y` is a scalar and the other is an
        :class:`~jax.Array`, the function checks that each element of the
        :class:`~jax.Array` is equal to the scalar. This behaviour can be disabled with
        the :code:`strict` parameter.

    Examples:

        The first assert does not raise an exception:

        >>> testax.assert_array_equal([1.0, 2.33333, jnp.nan],
        ...                           [jnp.exp(0), 2.33333, jnp.nan])

        Assert fails with numerical imprecision with floats:

        >>> testax.assert_array_equal([1.0, 1e-5, jnp.nan],
        ...                           [1, jnp.arccos(jnp.cos(1e-5)), jnp.nan])
        Traceback (most recent call last):
            ...
        jax._src.checkify.JaxRuntimeError:
        Arrays are not equal
        <BLANKLINE>
        Mismatched elements: 1 / 3 (33.3%)
        Max absolute difference: 1e-05
        Max relative difference: 0
         x: Array([1.e+00, 1.e-05,    nan], dtype=float32)
         y: Array([ 1.,  0., nan], dtype=float32)

        Use :func:`assert_allclose` or one of the nulp (number of floating point values)
        functions for these cases instead:

        >>> testax.assert_allclose([1.0, 1e-5, jnp.nan],
        ...                        [1, jnp.arccos(jnp.cos(1e-5)), jnp.nan], atol=1e-5)

        As mentioned in the Notes section, :func:`assert_array_equal` has special
        handling for scalars. Here the test checks that each value in :code:`x` is 3:

        >>> x = jnp.full((2, 5), fill_value=3)
        >>> testax.assert_array_equal(x, 3)

        Use :code:`strict` to raise an AssertionError when comparing a scalar with an
        array:

        >>> testax.assert_array_equal(x, 3, strict=True)
        Traceback (most recent call last):
            ...
        testax.TestaxError:
        Arrays are not equal
        <BLANKLINE>
        (shapes (2, 5), () mismatch)
         x: Array([[3, 3, 3, 3, 3],
                   [3, 3, 3, 3, 3]], dtype=int32, weak_type=True)
         y: Array(3, dtype=int32, weak_type=True)

        The :code:`strict` parameter also ensures that the array data types match:

        >>> x = jnp.array([2, 2, 2])
        >>> y = jnp.array([2., 2., 2.], dtype=jnp.float32)
        >>> testax.assert_array_equal(x, y, strict=True)
        Traceback (most recent call last):
            ...
        testax.TestaxError:
        Arrays are not equal
        <BLANKLINE>
        (dtypes int32, float32 mismatch)
         x: Array([2, 2, 2], dtype=int32)
         y: Array([2., 2., 2.], dtype=float32)
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
    Raises an :class:`AssertionError` if two :class:`~jax.Array`\ s are not equal up to
    the desired precision.

    .. note::
        It is recommended to use one of :func:`assert_allclose`,
        :func:`assert_array_almost_equal_nulp` or :func:`assert_array_max_ulp`
        instead of this function for more consistent floating point comparisons.

    The test verifies identical shapes and that the elements of :code:`actual` and
    :code:`desired` satisfy :code:`abs(desired - actual) < 1.5 * 10 ** - decimal`.

    An exception is raised at shape mismatch or conflicting values. In contrast to the
    standard usage in numpy, :code:`nan`\ s are compared like numbers, no assertion is
    raised if both objects have :code:`nan`\ s in the same positions.

    Args:
        x: The actual object to check.
        y: The desired, expected object.
        decimal: Desired precision, default is 6.
        err_msg: The error message to be printed in case of failure.
        verbose: If True, the conflicting values are appended to the error message.

    Raises:
        AssertionError: If :code:`actual` and :code:`desired` are not equal up to
            specified :code:`precision`.

    See Also:
        :func:`assert_allclose`, :func:`assert_array_almost_equal_nulp`,
        :func:`assert_array_max_ulp`, :func:`assert_equal`

    Examples:

        The first assert does not raise an exception

        >>> testax.assert_array_almost_equal([1.0, 2.333, jnp.nan],
        ...                                  [1.0, 2.333, jnp.nan])

        >>> testax.assert_array_almost_equal([1.0, 2.33333, jnp.nan],
        ...                                  [1.0, 2.33339, jnp.nan], decimal=5)
        Traceback (most recent call last):
            ...
        jax._src.checkify.JaxRuntimeError:
        Arrays are not almost equal to 5 decimals
        <BLANKLINE>
        Mismatched elements: 1 / 3 (33.3%)
        Max absolute difference: 6.0081e-05
        Max relative difference: 2.5749e-05
         x: Array([1.     , 2.33333,     nan], dtype=float32)
         y: Array([1.     , 2.33339,     nan], dtype=float32)

        >>> testax.assert_array_almost_equal([1.0,2.33333,jnp.nan],
        ...                                      [1.0,2.33333, 5], decimal=5)
        Traceback (most recent call last):
            ...
        jax._src.checkify.JaxRuntimeError:
        Arrays are not almost equal to 5 decimals
        <BLANKLINE>
        x and y nan location mismatch:
         x: Array([1.     , 2.33333,     nan], dtype=float32)
         y: Array([1.     , 2.33333, 5.     ], dtype=float32)
    """
    from jax._src.numpy.util import promote_args_inexact

    def compare(x, y):
        x, y = promote_args_inexact("assert_array_almost_equal", x, y)
        return jnp.abs(x - y) < 1.5 * 10.0 ** (-decimal)

    assert_array_compare(
        compare,
        x,
        y,
        err_msg=err_msg,
        verbose=verbose,
        header=("Arrays are not almost equal to %d decimals" % decimal),
        precision=decimal,
        debug=debug,
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
    Raises an :class:`AssertionError` if two :class:`~jax.Array`\ s are not ordered by
    less than.

    Given two :class:`~jax.Array`\ s, check that the shape is equal and all elements of
    the first array are strictly smaller than those of the second. An exception is
    raised at shape mismatch or incorrectly ordered values. Shape mismatch does not
    raise if an object has zero dimension. In contrast to the standard usage in numpy,
    :code:`nan`\ s are compared, no assertion is raised if both objects have
    :code:`nan`\ s in the same positions.

    Args:
        x: The smaller object to check.
        y: The larger object to compare.
        err_msg: The error message to be printed in case of failure.
        verbose: If True, the conflicting values are appended to the error message.

    Raises:
        AssertionError: If :code:`x` is not strictly smaller than :code:`y`,
            element-wise.

    See Also:
        :func:`assert_array_equal`, :func:`assert_array_almost_equal`

    Examples:

        >>> testax.assert_array_less([1.0, 1.0, jnp.nan], [1.1, 2.0, jnp.nan])
        >>> testax.assert_array_less([1.0, 1.0, jnp.nan], [1, 2.0, jnp.nan])
        Traceback (most recent call last):
            ...
        jax._src.checkify.JaxRuntimeError:
        Arrays are not less-ordered
        <BLANKLINE>
        Mismatched elements: 1 / 3 (33.3%)
        Max absolute difference: 1
        Max relative difference: 0.5
         x: Array([ 1.,  1., nan], dtype=float32)
         y: Array([ 1.,  2., nan], dtype=float32)

        >>> testax.assert_array_less([1.0, 4.0], 3)
        Traceback (most recent call last):
            ...
        jax._src.checkify.JaxRuntimeError:
        Arrays are not less-ordered
        <BLANKLINE>
        Mismatched elements: 1 / 2 (50%)
        Max absolute difference: 2
        Max relative difference: 0.666667
         x: Array([1., 4.], dtype=float32)
         y: Array(3, dtype=int32, weak_type=True)

        >>> testax.assert_array_less([1.0, 2.0, 3.0], [4])
        Traceback (most recent call last):
            ...
        testax.TestaxError:
        Arrays are not less-ordered
        <BLANKLINE>
        (shapes (3,), (1,) mismatch)
         x: Array([1., 2., 3.], dtype=float32)
         y: Array([4], dtype=int32)
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
