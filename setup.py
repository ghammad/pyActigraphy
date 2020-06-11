from setuptools import setup, find_packages
from os import path
import re

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


# Get the version of the packages
def read(*parts):
    with open(path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    # Name of your project. Determine how users can install this project, e.g.:
    #
    # $ pip install actimetry
    #
    # And where it will live on PyPI: https://pypi.org/project/actimetry/
    #
    # There are some restrictions on what makes a valid project name
    # specification here:
    # https://packaging.python.org/specifications/core-metadata/#name
    name='pyActigraphy',

    # Versions should comply with PEP 440:
    # https://www.python.org/dev/peps/pep-0440/
    #
    # For a discussion on single-sourcing the version across setup.py and the
    # project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=find_version("pyActigraphy", "__init__.py"),
    # version='0.1',

    # One-line description
    description='Analysis package for actigraphy data',

    # Optional longer description of your project that represents
    # the body of text which users will see when they visit PyPI.
    #
    # This field corresponds to the "Description" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#description-optional
    long_description=long_description,
    long_description_content_type='text/x-rst',

    # Valid link to your project's main homepage.
    #
    # This field corresponds to the "Home-Page" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#home-page-optional
    url='http://github.com/ghammad/pyActigraphy',

    author='Grégory Hammad',
    author_email='gregory.hammad@hotmail.fr',

    # Classifiers help users find your project by categorizing it.
    #
    # For a list of valid classifiers, see
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 5 - Production/Stable',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',

        # Pick your license as you wish
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
    ],

    # This field adds keywords for your project which will appear on the
    # project page. What does your project relate to?
    #
    # Note that this is a string of words separated by whitespace, not a list.
    keywords='actigraphy actimetry analysis python open-source',  # Optional

    # Specify package directories manually or use find_packages().
    #
    # packages=['actimetry'],
    packages=find_packages(exclude=['docs', 'tests']),  # Required

    # This field lists other packages that your project depends on to run.
    # Any package you put here will be installed by pip when your project is
    # installed, so they must be valid existing projects.
    #
    # For an analysis of "install_requires" vs pip's requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        'joblib', 'lmfit', 'pandas', 'numba', 'numpy', 'pyexcel',
        'pyexcel-ods3', 'scipy', 'spm1d', 'statsmodels>=0.10',
        'stochastic>=0.4.0'
    ],  # Optional

    # Data files included in your packages that need to be installed.
    #
    # If using Python 2.6 or earlier, then these have to be included in
    # MANIFEST.in as well.
    package_data={  # Optional
        'pyActigraphy': ['tests/data/*']
    },

    license='GNU GPL-3.0',

    setup_requires=['pytest-runner'],
    tests_require=['pytest'],

    zip_safe=False)
