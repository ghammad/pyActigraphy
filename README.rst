.. image:: https://img.shields.io/badge/License-GPL%20v3-blue.svg
  :target: https://www.gnu.org/licenses/gpl-3.0
.. image:: https://gitlab.com/ghammad/pyActigraphy/badges/master/pipeline.svg
  :target: https://gitlab.com/ghammad/pyActigraphy/commits/master
.. image:: https://gitlab.com/ghammad/pyActigraphy/badges/master/coverage.svg
  :target: https://gitlab.com/ghammad/pyActigraphy/commits/master
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.2537921.svg
  :target: https://doi.org/10.5281/zenodo.2537921

**pyActigraphy**
================
Open-source python package for actigraphy data analysis.


This package is meant to provide a comprehensive set of tools to:

* read actigraphy raw data files with various formats
* calculate typical wake/sleep cycle-related variables (ex: IS, IV, ...)
* perform complex analyses (ex: FDA, SSA, HMM, ...)

Requirements
============
* python 3.X
* joblib
* pandas
* numba
* numpy
* pyexcel
* pyexcel-ods3
* scipy
* statsmodels

Installation
============
In a (bash) shell, simply type:

* For users:

.. code-block:: shell

  pip install pyActigraphy

To update the package:

.. code-block:: shell

  pip install -U pyActigraphy

It is strongly recommended to use the latest version of the pyActigraphy package.


* For developers:

.. code-block:: shell

  git clone git@github.com:ghammad/pyActigraphy.git
  cd pyActigraphy/
  git checkout develop
  pip install -e .

Quick start
===========

The following example illustrates how to calculate the interdaily stability
with the pyActigraphy package:

.. code-block:: python

  >>> import pyActigraphy
  >>> rawAWD = pyActigraphy.io.read_raw_awd(fpath + 'SUBJECT_01.AWD')
  >>> rawAWD.IS()
  0.6900175913031027
  >>> rawAWD.IS(freq='30min', binarize=True, threshold=4)
  0.6245582891144925
  >>> rawAWD.IS(freq='1H', binarize=False)
  0.5257020914453097


Contributing
============

There are plenty of ways to contribute to this package, including (but not limiting to):

* report bugs (and, ideally, how to reproduce the bug)
* suggest improvements
* improve the documentation
* hug or high-five the authors when you meet them!

Authors
=======

* **Grégory Hammad** `@ghammad <https://github.com/ghammad>`_ - *Initial and main developer*
* **Mathilde Reyt** `@ReytMathilde <https://github.com/ReytMathilde>`_

See also the list of `contributors <https://github.com/ghammad/pyActigraphy/contributors>`_ who participated in this project.

License
=======

This project is licensed under the GNU GPL-3.0 License - see the `LICENSE <LICENSE>`_ file for details

Acknowledgments
===============

* **Aubin Ardois** `@aardoi <https://github.com/aardoi>`_ developed the first version of the MTN class during his internship at the CRC, in May-August 2018.
* The CRC colleagues for their support, ideas, etc.
