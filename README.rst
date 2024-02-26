🧪 testax
=========

.. image:: https://img.shields.io/pypi/v/testax
    :target: https://pypi.org/project/testax
.. image:: https://github.com/tillahoffmann/testax/actions/workflows/build.yml/badge.svg
    :target: https://github.com/tillahoffmann/testax/actions/workflows/build.yml
.. image:: https://readthedocs.org/projects/testax/badge/?version=latest
    :target: https://testax.readthedocs.io/en/latest/?badge=latest

testax provides runtime assertions for JAX through the testing interface familiar to NumPy users.

>>> import jax
>>> from jax import numpy as jnp
>>> import testax
>>>
>>> def safe_log(x):
...     testax.assert_array_less(0, x)
...     return jnp.log(x)
>>>
>>> safe_log(jnp.arange(2))
Traceback (most recent call last):
    ...
jax._src.checkify.JaxRuntimeError:
Arrays are not less-ordered
<BLANKLINE>
Mismatched elements: 1 / 2 (50%)
Max absolute difference: 1
Max relative difference: 1
 x: Array(0, dtype=int32, weak_type=True)
 y: Array([0, 1], dtype=int32)

testax assertions are :code:`jit`-able, although errors need to be functionalized to conform to JAX's requirement that functions are pure and do not have side effects (see the :code:`checkify` `guide <https://jax.readthedocs.io/en/latest/debugging/checkify_guide.html>`__ for details). In short, a :code:`checkify`-d function returns a tuple :code:`(error, value)`. The first element is an error that *may* have occurred, and the second is the return value of the original function.

>>> jitted = jax.jit(safe_log)
>>> checkified = testax.checkify(jitted)
>>> error, y = checkified(jnp.arange(2))
>>> error.throw()
Traceback (most recent call last):
    ...
jax._src.checkify.JaxRuntimeError:
Arrays are not less-ordered
<BLANKLINE>
Mismatched elements: 1 / 2 (50%)
Max absolute difference: 1
Max relative difference: 1
 x: Array(0, dtype=int32, weak_type=True)
 y: Array([0, 1], dtype=int32)
>>> y
Array([-inf,   0.], dtype=float32)

Installation
------------

testax is pip-installable and can be installed by running

.. code-block:: bash

    pip install testax

Interface
---------

testax mirrors the `testing <https://numpy.org/doc/stable/reference/routines.testing.html>`__ interface familiar to NumPy users, such as :code:`assert_allclose`.
