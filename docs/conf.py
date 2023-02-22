# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import pathlib
import sys
from datetime import date

import blackjax

PROJECT_DIR = pathlib.Path(__file__).absolute().parent.parent
sys.path.append(str(PROJECT_DIR))

# -- Project information -----------------------------------------------------

project = "blackjax"
copyright = f"{date.today().year}, The Blackjax developers"
author = "The Blackjax developers"
version = blackjax.__version__


# General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "IPython.sphinxext.ipython_console_highlighting",  # ipython3 Pygments highlighting
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_math_dollar",
    "sphinx.ext.mathjax",
    "myst_nb",
    "sphinx_design",
    "autoapi.extension",
    "sphinxcontrib.bibtex",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "jax": ("https://jax.readthedocs.io/en/latest", None),
}

# AutoAPI configuration
autoapi_dirs = ["../blackjax"]
autoapi_type = "python"
autoapi_add_toctree_entry = False
autoapi_options = ["show-module-summary", "undoc-members"]
autodoc_typehints = "signature"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/blackjax-devs/blackjax",
    "use_repository_button": True,
    "use_download_button": False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_title = ""
html_logo = "_static/blackjax.png"
html_css_files = ["custom.css"]

# We only display the typehints in the description, even though they would
# be better in the signature, because we cannot apply our CSS trick to add
# line breaks in the signature when the typehints are present.
autosummary_generate = True
add_module_names = False

source_suffix = {".rst": "restructuredtext", ".ipynb": "myst-nb", ".md": "myst-nb"}

nb_execution_mode = "auto"
nb_execution_timeout = 300
suppress_warnings = ["mystnb.unknown_mime_type"]

nb_custom_formats = {
    ".md": ["jupytext.reads", {"fmt": "mystnb"}],
}
myst_enable_extensions = ["colon_fence"]


# Skip files we do not want to be included in the documentation
def skip_util_classes(app, what, name, obj, skip, options):
    excluded_modules = [
        "blackjax._version",
        "blackjax.progress_bar",
        "blackjax.util",
        "blackjax.types",
        "blackjax.base",
    ]
    if what == "module" and name in excluded_modules:
        skip = True

    excluded_packages = ["blackjax.optimizers"]
    if what == "package" and name in excluded_packages:
        skip = True

    return skip


def setup(sphinx):
    sphinx.connect("autoapi-skip-member", skip_util_classes)


bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "alpha"  # alpha, plain, unsrt, unsrtalpha
