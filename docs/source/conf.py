# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# pylint: disable=invalid-name,redefined-builtin

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
from importlib import metadata


# -- Project information -----------------------------------------------------

project = 'Loki'
copyright = '2018- European Centre for Medium-Range Weather Forecasts (ECMWF)'
author = 'Michael Lange, Balthasar Reuter'

# The full version, including alpha/beta/rc tags.
release = metadata.version('loki')
# The short X.Y version.
version = release


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',  # create documentation from docstrings
    'sphinx.ext.napoleon',  # understand docstrings also in other formats
    'sphinx.ext.autosummary',  # automatically compile lists of classes/functions
    'sphinx.ext.intersphinx',  # link to docs of other projects
    'sphinx.ext.autosectionlabel',  # allows to refer to sections using their title
#    'recommonmark',  # read markdown
    'sphinx_rtd_theme',  # read the docs theme
    'myst_parser',  # parse markdown files
    'nbsphinx',  # parse Jupyter notebooks
    'sphinx_design',  # cards, panels and dropdown content
]

autosummary_generate = True  # Turn on sphinx.ext.autosummary
html_show_sourcelink = False  # Remove 'view source code' from top of page (for html, not python)
add_module_names = False # Remove namespaces from class/method signatures

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'pymbolic': ('https://documen.tician.de/pymbolic/', None),
    'fparser': ('https://fparser.readthedocs.io/en/latest/', None)
}

# The file extensions of source files. Sphinx considers the files with
# this suffix as sources. The value can be a dictionary mapping file
# extensions to file types.
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown'
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['**/tests/']

# Prefix each section label with the document it is in, followed by a colon
autosectionlabel_prefix_document = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': 'view',
    'style_nav_header_background': '',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# -- Options for autodoc extension -------------------------------------------

autodoc_default_options = {
    'members': True,  # include members in the documentation
    'member-order': 'bysource',  # members in the order they appear in source
    'show-inheritance': True,  # list base classes
    'undoc-members': True,  # show also undocumented members
}
