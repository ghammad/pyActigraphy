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
.. image:: https://bestpractices.coreinfrastructure.org/projects/6933/badge
  :target: https://bestpractices.coreinfrastructure.org/projects/6933
.. image:: https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg
  :target: CODE_OF_CONDUCT.md


**pyActigraphy**
================
Open-source python package for actigraphy and light exposure data visualization and analysis.


This package is meant to provide a comprehensive set of tools to:

* read native actigraphy data files with various formats:

  * Actigraph: wGT3X-BT
  * CamNtech: Actiwatch 4, 7, L(-Plus) and MotionWatch 8
  * Condor Instrument: ActTrust 2
  * Daqtix: Daqtometer
  * Respironics: Actiwatch 2 and Actiwatch Spectrum (plus)
  * Tempatilumi (CE Brasil)

..

* **NEW** read actigraphy data format from the `MESA dataset <https://sleepdata.org/datasets/mesa>`_, hosted by the `National Sleep Research Resource <https://sleepdata.org>`_.

* **NEW** read actigraphy data files produced by the `accelerometer <https://biobankaccanalysis.readthedocs.io/en/latest/index.html>`_ package that can be used to calibrate and convert raw accelerometer data recorded with:

  * Axivity: AX3, device used by UK Biobank,
  * Activinsights: GENEActiv, used by the Whitehall II study.

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
thanks to the support of its members, Dr. Mirjam Münch and Prof. `Manuel Spitschan <https://github.com/spitschan>`_,
a pyActigraphy module for analysing light exposure data has been developed, **pyLight**.
This module is part of the Human Light Exposure Database and is included in pyActigraphy version `v1.1 <https://github.com/ghammad/pyActigraphy/releases/tag/v1.1>`_ and higher.

When using this module, please consider citing:

  Hammad, G., Wulff, K., Skene, D. J., Münch, M., & Spitschan, M. (2024). Open-Source Python Module for the Analysis of Personalized Light Exposure Data from   Wearable Light Loggers and Dosimeters. LEUKOS, 20(4), 380–389. https://doi.org/10.1080/15502724.2023.2296863

Code and documentation
======================

The pyActigraphy package is open-source and its source code is accessible `online <https://github.com/ghammad/pyActigraphy>`_.


An online documentation of the package is also available `here <https://ghammad.github.io/pyActigraphy/index.html>`_.
It contains `notebooks <https://ghammad.github.io/pyActigraphy/tutorials.html>`_ illustrating various functionalities of the package. Specific tutorials for the processing and the analysis of light exposure data with pyLight are also available.

Installation
============

For the time being, :code:`pyActigraphy` has been tested for :code:`python>=3.7 & python<=3.9`. Dependencies will be installed automatically.

Before installing python packages, it is often advised to create a virtual environment:

#. Via `venv <https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment>`_ (Linux/Mac OS)
#. Via `miniconda <https://www.anaconda.com/docs/getting-started/miniconda/main>_` (Linux/Mac OS/Windows)

Installing pyActigraphy (alone)
-----------------------
  
Within a virtual env, in a Terminal (Linux/Mac OS) or in an Anaconda Prompt (if you installed miniconda/anaconda), simply type:
  
* For users:
  
.. code-block:: shell
  
  python -m pip install numba==0.57.1
  python -m pip install pyActigraphy
  
To update the package:
  
.. code-block:: shell
  
  python -m pip install -U pyActigraphy
  
  
* For developers:
  
.. code-block:: shell

  python -m pip install numba==0.57.1
  git clone git@github.com:ghammad/pyActigraphy.git
  cd pyActigraphy/
  git checkout develop
  python -m pip install -e .


Installing pyActigraphy+Jupyter (tutorials)
--------------------------------------------------

The `pyActigraphy` package provides a series of tutorial `notebooks <https://ghammad.github.io/pyActigraphy/tutorials.html>`_. These `Jupyter notebooks <https://jupyter.org/>`_ (file extension: .ipynb) are part of the package but can also be downloaded from the `Github repository <https://github.com/ghammad/pyActigraphy/tree/master/docs/source/>`_.
In order to interactively run these tutorials, one needs to install the Jupyter Notebook application.

While users are encouraged to install and tailor these tools to their needs, a simpler one-stop-shop solution consists in using `Anaconda <https://www.anaconda.com/docs/main>`_.

Instructions:

#. Download and install `Anaconda Distribution <https://www.anaconda.com/docs/getting-started/anaconda/install>`_
#. Via the **Anaconda Prompt** (Windows) or a **Terminal** (Mac OS, Linux):

   #. Create a virtual environment::

       conda create -n pyActi39 python=3.9

   #. Activate the newly created environment::

       conda activate pyActi39

   #. Install the `Numba <https://numba.readthedocs.io/en/stable/index.html>`_ package which is a dependency of :code:`pyActigraphy`::

       pip install numba==0.57.1

   #. Install :code:`pyActigraphy`::

       pip install pyActigraphy

#. Launch the Jupyter Notebook via the Anaconda Navigator:

   #. Via the application menu (On Windows)
   #. Via a **Terminal** (On Mac OS/Linux only)::

      anaconda-navigator

   .. warning::
      
      Once the navigator is running, **before** launching the Jupyter Notebook app, select the **pyActi39** environment (instead of :code:`base (root)`)

   .. image:: docs/source/img/anaconda-navigator-instructions.png
      :width: 600

#. Download the tutorial `notebooks <https://github.com/ghammad/pyActigraphy/tree/master/docs/source/>`_:

   * `pyActigraphy-Intro.ipynb <https://github.com/ghammad/pyActigraphy/blob/cce641bb09bd1ac1912aa5eb09894ed152844475/docs/source/pyActigraphy-Intro.ipynb>`_
   * `pyActigraphy-Batch.ipynb <https://github.com/ghammad/pyActigraphy/blob/cce641bb09bd1ac1912aa5eb09894ed152844475/docs/source/pyActigraphy-Batch.ipynb>`_
   * `pyActigraphy-Masking.ipynb <https://github.com/ghammad/pyActigraphy/blob/cce641bb09bd1ac1912aa5eb09894ed152844475/docs/source/pyActigraphy-Masking.ipynb>`_
   * `pyActigraphy-SSt-log.ipynb <https://github.com/ghammad/pyActigraphy/blob/cce641bb09bd1ac1912aa5eb09894ed152844475/docs/source/pyActigraphy-SSt-log.ipynb>`_
   * `pyActigraphy-Sleep-Algorithms.ipynb <https://github.com/ghammad/pyActigraphy/blob/cce641bb09bd1ac1912aa5eb09894ed152844475/docs/source/pyActigraphy-Sleep-Algorithms.ipynb>`_
   * `pyActigraphy-Sleep-Diary.ipynb <https://github.com/ghammad/pyActigraphy/blob/cce641bb09bd1ac1912aa5eb09894ed152844475/docs/source/pyActigraphy-Sleep-Diary.ipynb>`_
   * `pyActigraphy-StateTransitionProb.ipynb <https://github.com/ghammad/pyActigraphy/blob/cce641bb09bd1ac1912aa5eb09894ed152844475/docs/source/pyActigraphy-StateTransitionProb.ipynb>`_

#. Via the Jupyter interface, navigate to the tutorial notebooks you previously downloaded and simply launch them.

#. Voilà. Good luck.


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
