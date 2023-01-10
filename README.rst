.. image:: https://img.shields.io/badge/License-GPL%20v3-blue.svg
  :target: https://www.gnu.org/licenses/gpl-3.0
.. image:: https://gitlab.com/ghammad/pyActigraphy/badges/master/pipeline.svg?key_text=CI+tests
  :target: https://gitlab.com/ghammad/pyActigraphy/commits/master
.. .. image:: https://gitlab.com/ghammad/pyActigraphy/badges/master/coverage.svg
..   :target: https://gitlab.com/ghammad/pyActigraphy/commits/master
.. image:: https://img.shields.io/pypi/v/pyActigraphy.svg
  :target: https://pypi.org/project/pyActigraphy
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.2537920.svg
  :target: https://doi.org/10.5281/zenodo.2537920

**pyActigraphy**
================
Open-source python package for actigraphy and light exposure data visualization and analysis.


This package is meant to provide a comprehensive set of tools to:

* read native actigraphy data files with various formats:

  * Actigraph: wGT3X-BT
  * CamNtech: Actiwatch 4 and MotionWatch 8
  * Condor Instrument: ActTrust 2
  * Daqtix: Daqtometer
  * Respironics: Actiwatch 2 and Actiwatch Spectrum (plus)
  * Tempatilumi (CE Brasil)

..

* **NEW** read light exposure data recorded by the aforementioned devices (when available)

* clean the raw data and mask spurious periods of inactivity

* produce activity profile plots

* visualize sleep agendas and compute summary statistics

* calculate typical wake/sleep cycle-related variables:

  * Non-parametric rest-activity variables: IS(m), IV(m), RA
  * Activity or Rest  fragmentation: kRA, kAR
  * Sleep regularity index (SRI)

..

* **NEW** compute light exposure metrics (TAT, :math:`MLit^{500}`, summary statistics, ...)

* automatically detect rest periods using various algorithms (Cole-Kripke, Sadeh, ..., Crespo, Roenneberg)

* perform complex analyses:

  * Cosinor analysis
  * Detrended Fluctuation Analysis (DFA)
  * Functional Linear Modelling (FLM)
  * Locomotor Inactivity During Sleep (LIDS)
  * Singular Spectrum Analysis (SSA)
  * and much more...

Citation
========

We are very pleased to announce that the `v1.0 <https://github.com/ghammad/pyActigraphy/releases/tag/v1.0>`_ version of the pyActigraphy package has been published. So, if you find this package useful in your research, please consider citing:

  Hammad G, Reyt M, Beliy N, Baillet M, Deantoni M, Lesoinne A, et al. (2021) pyActigraphy: Open-source python package for actigraphy data visualization and    analysis. PLoS Comput Biol 17(10): e1009514. https://doi.org/10.1371/journal.pcbi.1009514

pyLight
=======

In the context of the Daylight Academy Project, `The role of daylight for humans <https://daylight.academy/projects/state-of-light-in-humans>`_ and
thanks to the support of its members, Profs Mirjam Münch and `Manuel Spitschan <https://github.com/spitschan>`_,
a pyActigraphy module for analysing light exposure data has been developed, **pyLight**.
This module is part of the Human Light Exposure Database and is included in pyActigraphy version `v1.1 <https://github.com/ghammad/pyActigraphy/releases/tag/v1.1>`_ and higher.

Code and documentation
======================

The pyActigraphy package is open-source and its source code is accessible `online <https://github.com/ghammad/pyActigraphy>`_.


An online documentation of the package is also available `here <https://ghammad.github.io/pyActigraphy/index.html>`_.
It contains `notebooks <https://ghammad.github.io/pyActigraphy/tutorials.html>`_ illustrating various functionalities of the package.

Specific tutorials for the processing and the analysis of light exposure data with pyLight are also available.

Installation
============
In a (bash) shell, simply type:

* For users:

.. code-block:: shell

  pip3 install pyActigraphy

To update the package:

.. code-block:: shell

  pip3 install -U pyActigraphy


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
