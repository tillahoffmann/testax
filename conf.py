import doctest


project = "testax"
html_theme = "sphinx_book_theme"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
]
exclude_patterns = [
    "venv",
]
intersphinx_mapping = {
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3", None),
}
doctest_global_setup = """
from jax import numpy as jnp
import testax
"""
doctest_default_flags = (
    doctest.ELLIPSIS | doctest.DONT_ACCEPT_TRUE_FOR_1 | doctest.NORMALIZE_WHITESPACE
)
nitpick_ignore = [
    ("py:func", "assert_equal"),
    ("py:func", "assert_array_almost_equal_nulp"),
    ("py:func", "assert_array_max_ulp"),
    # https://github.com/sphinx-doc/sphinx/issues/10974.
    ("py:class", "testax.T"),
    # Not documented by jax.
    ("py:class", "jax._src.checkify.JaxException"),
]
