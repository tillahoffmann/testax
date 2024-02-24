import contextlib
import inspect
from jax.experimental import checkify
import numpy as np
from numpy.testing.tests import test_utils
import testax
import pytest
import unittest


@contextlib.contextmanager
def _patched_assertRaises():
    original = unittest.TestCase.assertRaises

    def _patched(self, expected, *args, **kwargs):
        if expected is AssertionError:
            expected = (AssertionError, checkify.JaxRuntimeError, testax.TestaxError)
        return original(self, expected, *args, **kwargs)

    unittest.TestCase.assertRaises = _patched
    yield
    unittest.TestCase.assertRaises = original


@contextlib.contextmanager
def _patched_pytest_raises():
    original = pytest.raises

    def _patched(expected, *args, **kwargs):
        if expected is AssertionError:
            expected = (AssertionError, checkify.JaxRuntimeError, testax.TestaxError)
        return original(expected, *args, **kwargs)

    pytest.raises = _patched
    yield
    pytest.raises = original


def patch_numpy_test(cls, xfail=None, assert_func=None):
    xfail = xfail or {}
    # Get all the testax methods we can patch.
    testax_funcs = {
        name: getattr(testax, name)
        for name in testax.__all__
        if name.startswith("assert_")
    }

    # Patch the assert function if we have one.
    if assert_func:
        def _set_assert_func(self):
            self._assert_func = assert_func

        cls.setup_method = _set_assert_func

    # Iterate over the test methods and patch the globals.
    test_methods = inspect.getmembers(
        cls, lambda x: callable(x) and x.__name__.startswith("test_")
    )
    for name, method in test_methods:
        xfail_reason = xfail.get(name)
        if xfail_reason is not None:
            setattr(cls, name, pytest.mark.xfail(reason=xfail_reason)(method))
            continue

        # method=method argument takes care of the closure so method doesnt reference
        # the last element in the iteration.
        def _patched_method(*args, method=method, **kwargs):
            original_funcs = {
                key: method.__globals__.pop(key)
                for key in list(method.__globals__)
                if key.startswith("assert_")
                and not key in {"assert_raises", "assert_", "assert_equal"}
            }
            try:
                method.__globals__.update(testax_funcs)
                with _patched_assertRaises(), _patched_pytest_raises():
                    method(*args, **kwargs)
            finally:
                for key in testax_funcs:
                    method.__globals__.pop(key)
                method.__globals__.update(original_funcs)

        setattr(cls, name, _patched_method)
    return cls


TestAssertAllclose = patch_numpy_test(
    test_utils.TestAssertAllclose,
    xfail={
        "test_timedelta": "Only arrays of numeric types are supported by JAX.",
    },
)
TestArrayAssertLess = patch_numpy_test(
    test_utils.TestArrayAssertLess,
    assert_func=testax.assert_array_less,
)
TestArrayAlmostEqual = patch_numpy_test(
    test_utils.TestArrayAlmostEqual,
    xfail={
        "test_subclass": "numpy masked arrays are not supported as JAX inputs",
        "test_objarray": "arrays of objects are not supported by JAX",
    },
    assert_func=testax.assert_array_almost_equal,
)
TestArrayEqual = patch_numpy_test(
    test_utils.TestArrayEqual,
    xfail={
        "test_recarrays": "record arrays are not supported as JAX inputs",
        "test_string_arrays": "string arrays are not supported as JAX inputs",
        "test_masked_nan_inf": "numpy masked arrays are not supported as JAX inputs",
        "test_objarray": "arrays of objects are not supported as JAX inputs",
        "test_generic_rank1": "sub used as test on boolean dtype is not supported",
        "test_generic_rank3": "sub used as test on boolean dtype is not supported",
        "test_0_ndim_array": "integer overflow represented by object type",
    },
    assert_func=testax.assert_array_equal,
)


@pytest.mark.parametrize(
    "name", [name for name in np.testing.__all__ if name.startswith("assert_")]
)
def test_numpy_parity(name: str) -> None:
    # Get the functions.
    testax_func = getattr(testax, name, None)
    if testax_func is None:
        pytest.xfail(f"testax does not yet implement `{name}`")
    numpy_func = getattr(np.testing, name)

    # Compare the signature.
    testax_signature = inspect.signature(testax_func)
    numpy_signature = inspect.signature(numpy_func)

    # Check that the parameter names are the same and appear in the same order (except
    # the debug flag).
    testax_parameter_names = list(testax_signature.parameters)
    assert testax_parameter_names.pop() == "debug"
    assert testax_parameter_names == list(numpy_signature.parameters)

    # Check that position and keyword status match.
    for name, numpy_parameter in numpy_signature.parameters.items():
        testax_parameter = testax_signature.parameters[name]
        assert numpy_parameter.kind == testax_parameter.kind, name
        assert numpy_parameter.default == testax_parameter.default, name

    # Check that the testax implementation has type annotations.
    for name, parameter in testax_signature.parameters.items():
        assert parameter.annotation != inspect._empty, name
    assert testax_signature.return_annotation is None
