import inspect
from jax.experimental import checkify
import numpy as np
from numpy.testing.tests import test_utils
import testax
import pytest
from unittest import mock


def patch_numpy_test(cls, xfail=None):
    xfail = xfail or {}

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
            # Get all the testax methods we can patch.
            testax_funcs = {
                name: getattr(testax, name)
                for name in testax.__all__
                if name.startswith("assert_")
            }
            # We need to patch the globals referencing test function names and the base
            # classes of the JaxRuntimeError so it looks like an AssertionError to the
            # numpy tests. We do the __bases__ patching manually, because mock.patch
            # tries to delete the __bases__ attribute. Using is_local doesn't seem to
            # fix the problem (https://stackoverflow.com/a/12220965/1150961).
            with mock.patch.dict(method.__globals__, testax_funcs):
                original_bases = checkify.JaxRuntimeError.__bases__
                try:
                    checkify.JaxRuntimeError.__bases__ = (AssertionError,)
                    method(*args, **kwargs)
                finally:
                    checkify.JaxRuntimeError.__bases__ = original_bases

        setattr(cls, name, _patched_method)
    return cls


TestAssertAllclose = patch_numpy_test(
    test_utils.TestAssertAllclose,
    xfail={
        "test_timedelta": "Only arrays of numeric types are supported by JAX.",
    },
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
