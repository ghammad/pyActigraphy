.. image:: https://img.shields.io/badge/License-GPL%20v3-blue.svg
  :target: https://www.gnu.org/licenses/gpl-3.0
.. image:: https://gitlab.com/ghammad/pyActigraphy/badges/master/pipeline.svg
  :target: https://gitlab.com/ghammad/pyActigraphy/commits/master
.. image:: https://gitlab.com/ghammad/pyActigraphy/badges/master/coverage.svg
  :target: https://gitlab.com/ghammad/pyActigraphy/commits/master
.. image:: https://img.shields.io/pypi/v/pyActigraphy.svg
  :target: https://pypi.org/project/pyActigraphy
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.2537921.svg
  :target: https://doi.org/10.5281/zenodo.2537921

**pyActigraphy**
================
Open-source python package for actigraphy data visualization and analysis.


This package is meant to provide a comprehensive set of tools to:

* read native actigraphy data files with various formats:

  * Actigraph: wGT3X-BT
  * Condor Instrument: ActTrust 2
  * CamNtech: Actiwatch 4 and MotionWatch 8
  * Respironics: Actiwatch 2 and Actiwatch Spectrum (plus)
  * Daqtix: Daqtometer

..

* clean the raw data and mask spurious periods of inactivity

* produce activity profile plots

* visualize sleep agendas and compute summary statistics

* calculate typical wake/sleep cycle-related variables:

  * Non-parametric rest-activity variables: IS(m), IV(m), RA
  * Activity or Rest  fragmentation: kRA, kAR
  * Sleep regularity index (SRI)

..

* automatically detect rest periods using various algorithms (Cole-Kripke, Sadeh, ..., Crespo, Roenneberg)

* perform complex analyses:

  * Cosinor analysis
  * Detrended Fluctuation Analysis (DFA)
  * Functional Linear Modelling (FLM)
  * Locomotor Inactivity During Sleep (LIDS)
  * Singular Spectrum Analysis (SSA)
  * and much more...

Code and documentation
======================

The pyActigraphy package is open-source and its source code is accessible `online <https://github.com/ghammad/pyActigraphy>`_.


An online documentation of the package is also available `here <https://ghammad.github.io/pyActigraphy/index.html>`_.
It contains `notebooks <https://ghammad.github.io/pyActigraphy/documentation.html>`_ illustrating various functionalities of the package.

Installation
============
In a (bash) shell, simply type:

* For users:

.. code-block:: shell

  pip3 install pyActigraphy

To update the package:

.. code-block:: shell

  pip3 install -U pyActigraphy

It is strongly recommended to use the latest version of the pyActigraphy package.


* For developers:

.. code-block:: shell

  git clone git@github.com:ghammad/pyActigraphy.git
  cd pyActigraphy/
  git checkout develop
  pip3 install -e .

Quick start
===========

The following example illustrates how to calculate the interdaily stability
with the pyActigraphy package:

.. code-block:: python

  >>> import pyActigraphy
  >>> rawAWD = pyActigraphy.io.read_raw_awd('/path/to/your/favourite/file.AWD')
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
