import codecs
import os.path

import setuptools


def read(rel_path):
    """Read the file located at the provided relative path."""
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    """Get the package's version number.

    We fetch the version  number from the `__version__` variable located in the
    package root's `__init__.py` file. This way there is only a single source
    of truth for the package's version number.

    """
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setuptools.setup(
    name="blackjax",
    version=get_version("blackjax/__init__.py"),
    description="Flexible and fast inference in Python",
    packages=setuptools.find_packages(),
    install_requires=[
        "jax==0.2.7",
        "jaxlib==0.1.57",
    ],
)
