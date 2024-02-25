# 🧪 testax

**tl;dr**: Ordinary Python assertions and `numpy.testing.*` checks cannot be used in
`jit`-compiled JAX code because it needs to be free of side-effects like raising
exceptions. `testax` provides a familiar interface to implement runtime assertions for
JAX. Simply replace all your `numpy.testing.*` calls with `testax.*` calls, apply
JAX's `checkify` transformation before `jit`-compiling, and you're good to go.

## Background

The following code snippet fails because it may raise an exception depending on the
*value* of `x`. When the code is `jit`-compiled, the value is not available because JAX
injects tracer values, and we end up with an error.

```python
>>> import jax
>>> from jax import numpy as jnp


>>> @jax.jit
... def safe_log(x):
...     assert (x > 0).all()
...     return jnp.log

>>> safe_log(jnp.arange(2))
Traceback (most recent call last):
    ...
jax.errors.TracerBoolConversionError: Attempted boolean conversion of traced array
with shape bool[] ...
```

The `jax.experimental.checkify` API facilitates runtime checks in `jit`-compiled code by
functionalizing the assertions. As a result, a function that previously returned `value`
will return a tuple `(error, value)` after being `checkify`d. This means execution can
proceed without side effects as JAX expects. The error can subsequently be inspected.

```python
>>> from jax.experimental import checkify

>>> @jax.jit
... @checkify.checkify
... def safe_log(x):
...     checkify.check((x > 0).all(), "non-positive values: {x}", x=x)
...     return jnp.log(x)

>>> error, value = safe_log(jnp.arange(2))
>>> error.throw()
Traceback (most recent call last):
    ...
jax._src.checkify.JaxRuntimeError: non-positive values: [0 1] (`check` failed)
```

Numpy's `testing` module provides a nice assertion interface with informative error
messages. `testax` provides the same functionality.

```python
>>> from functools import partial
>>> import testax

>>> @jax.jit
... def safe_log(x):
...     testax.assert_array_less(0, x)
...     return jnp.log(x)

>>> error, value = testax.checkify(safe_log)(jnp.arange(2))
>>> error.throw()
Traceback (most recent call last):
...
JaxRuntimeError:
Arrays are not less-ordered

Mismatched elements: 1 / 2 (50%)
Max absolute difference: 1
Max relative difference: 1
 x: Array(0, dtype=int32, weak_type=True)
 y: Array([0, 1], dtype=int32)
```
