# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
sys.setrecursionlimit(1500)

master_doc = 'index'


# -- Project information -----------------------------------------------------

project = 'inpystem'
copyright = '2019, Etienne Monier'
author = 'Etienne Monier'

# The full version, including alpha/beta/rc tags
release = '0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinxcontrib.napoleon',
    'sphinxcontrib.katex',
    'sphinxcontrib.bibtex'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

napoleon_use_ivar = True
napoleon_include_init_with_doc = True

todo_include_todos = True
todo_include_todos = True

autodoc_member_order = 'bysource'

pygments_style = 'sphinx'

# latex_elements = {
#     'preamble': r'''

# '''}

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
# html_logo = '_static/hyperspy_logo.png'

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = 'hyperspy.ico'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# If given, this must be the name of an image file (path relative to the
# configuration directory) that is the favicon of the docs. Modern browsers
# use this as the icon for tabs, windows and bookmarks. It should be a
# Windows-style icon file (.ico), which is 16x16 or 32x32 pixels large.
# Default: None.
# html_favicon = ''

# If given, this must be the name of an image file (path relative to the
# configuration directory) that is the logo of the docs. It is placed at
# the top of the sidebar; its width should therefore not exceed 200 pixels.
# Default: None.
# html_logo = ''


# -- Extension configuration -------------------------------------------------


def setup(app):
    app.add_javascript('copybutton.js')
