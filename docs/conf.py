# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Backblaze MLOps Pipeline'
copyright = '2025, dvarfe'
author = 'dvarfe'
release = '0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinx.ext.viewcode']

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Add and use Pylons theme
# if 'sphinx-build' in ' '.join(sys.argv):  # protect against dumb importers
#     from subprocess import call, Popen, PIPE

#     p = Popen('which git', shell=True, stdout=PIPE)
#     git = p.stdout.read().strip()
#     cwd = os.getcwd()
#     _themes = os.path.join(cwd, '_themes')

#     if not os.path.isdir(_themes):
#         call([git, 'clone', 'git://github.com/Pylons/pylons_sphinx_theme.git',
#               '_themes'])
#     else:
#         os.chdir(_themes)
#         call([git, 'checkout', 'master'])
#         call([git, 'pull'])
#         os.chdir(cwd)

#     sys.path.append(os.path.abspath('_themes'))

#     parent = os.path.dirname(os.path.dirname(__file__))
#     sys.path.append(os.path.abspath(parent))
#     wd = os.getcwd()
#     os.chdir(parent)
#     os.chdir(wd)

#     for item in os.listdir(parent):
#         if item.endswith('.egg'):
#             sys.path.append(os.path.join(parent, item))

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

autodoc_mock_imports = [
    # Core ML/DL frameworks
    "torch",
    "sklearn",

    # Data processing
    "pandas",
    "numpy",

    # MLOps tools
    "mlflow",

    # Utils
    "matplotlib",
    "seaborn",
    "plotly",
    "os",
    "sys"
]
