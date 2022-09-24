import codecs
import os.path
import sys

import setuptools

# READ README.md for long description on PyPi.
try:
    long_description = open("README.md", encoding="utf-8").read()
except Exception as e:
    sys.stderr.write(f"Failed to read README.md:\n  {e}\n")
    sys.stderr.flush()
    long_description = ""


# Get the package's version number of the __init__.py file
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
    author="The BlackJAX team",
    version=get_version("blackjax/__init__.py"),
    description="Flexible and fast inference in Python",
    long_description=long_description,
    packages=setuptools.find_packages(),
    install_requires=[
        "fastprogress>=0.2.0",
        "jax>=0.3.13",
        "jaxlib>=0.3.10",
        "jaxopt>=0.4.2",
    ],
    long_description_content_type="text/markdown",
    keywords="probabilistic machine learning bayesian statistics sampling algorithms",
    license="Apache License 2.0",
    license_files=("LICENSE",),
)
