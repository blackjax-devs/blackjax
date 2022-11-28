import os
import sys

import setuptools

import versioneer

NAME = "blackjax"

# Handle builds of nightly release
if "BUILD_BLACKJAX_NIGHTLY" in os.environ:
    NAME += "-nightly"

    from versioneer import get_versions as original_get_versions

    def get_versions():
        from datetime import datetime, timezone

        suffix = datetime.now(timezone.utc).strftime(r".dev%Y%m%d")
        versions = original_get_versions()
        versions["version"] = versions["version"].split("+")[0] + suffix
        return versions

    versioneer.get_versions = get_versions


# READ README.md for long description on PyPi.
try:
    long_description = open("README.md", encoding="utf-8").read()
except Exception as e:
    sys.stderr.write(f"Failed to read README.md:\n  {e}\n")
    sys.stderr.flush()
    long_description = ""


setuptools.setup(
    name=NAME,
    author="The BlackJAX team",
    description="Flexible and fast inference in Python",
    long_description=long_description,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=setuptools.find_packages(),
    install_requires=[
        "fastprogress>=0.2.0",
        "jax>=0.3.13",
        "jaxlib>=0.3.10",
        "jaxopt>=0.5.5",
    ],
    long_description_content_type="text/markdown",
    keywords="probabilistic machine learning bayesian statistics sampling algorithms",
    license="Apache License 2.0",
    license_files=("LICENSE",),
)
