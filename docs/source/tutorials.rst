======================
pyActigraphy tutorials
======================

This series of notebooks are meant to illustrate the different features of the
*pyActigraphy* package.

Introduction
============

* A gentle introduction to the basic functionalities: `intro`_
* How to read files by batch: `batch`_

.. _intro: pyActigraphy-Intro.ipynb
.. _batch: pyActigraphy-Batch.ipynb


Feature examples
================

* How to discard invalid sequences in actigraphy recordings before analysis:

  * `Invalid sequences during the recordings`_
  * `Invalid sequences at the beginning and/or the end of the recordings`_

..

* `How to calculate the usual rest-activity rhythm variables`_
* `How to visualise sleep diaries and compute summary statistics`_
* `How to detect rest periods automatically`_
* `How to quantify sleep fragmentation using state transition probabilities`_


.. _Invalid sequences during the recordings: pyActigraphy-Masking.ipynb
.. _Invalid sequences at the beginning and/or the end of the recordings: pyActigraphy-SSt-log.ipynb
.. _How to calculate the usual rest-activity rhythm variables: pyActigraphy-Non-parametric-variables.ipynb
.. _How to visualise sleep diaries and compute summary statistics: pyActigraphy-Sleep-Diary.ipynb
.. _How to detect rest periods automatically: pyActigraphy-Sleep-Algorithms.ipynb
.. _How to quantify sleep fragmentation using state transition probabilities: pyActigraphy-StateTransitionProb.ipynb


Analysis examples
=================

* How to perform a Cosinor analysis with pyActigraphy: `cosinor`_
* How to perform a Functional linear modelling with pyActigraphy: `flm`_
* How to perform a (Multifractal) Detrended fluctuation analysis (MF-DFA) with pyActigraphy: `mfdfa`_
* How to perform a Singular spectrum analysis (SSA) with pyActigraphy: `ssa`_

.. _cosinor: pyActigraphy-Cosinor.ipynb
.. _flm: pyActigraphy-FLM.ipynb
.. _mfdfa: pyActigraphy-MFDFA.ipynb
.. _ssa: pyActigraphy-SSA.ipynb


pyLight examples
================

The *pyActigraphy* package contains a module dedicated to the analysis of light
exposure data, named *pyLight*. The following tutorials specifically illustrate
its functionalities:

* A gentle introduction to the basics of *pyLight*: `pylight_intro`_
* How to process light exposure data with *pyLight*: `pylight_manip`_
* How to calculate various light exposure metrics with *pyLight*: `pylight_metrics`_

.. _pylight_intro: pyLight-Intro.ipynb
.. _pylight_manip: pyLight-DataManip.ipynb
.. _pylight_metrics: pyLight-Metrics.ipynb


Work-in-progress
================

The features illustrated in this section have to be considered as work-in-progress;
most likely, they will be turned into full-fledged features in a next release.
However, for impatient and brave users, it provides a starting point for testing:

* How to read actigraphy data stored in a pandas.DataFrame with pyActigraphy: `pandas`_

.. _pandas: pyActigraphy-Pandas.ipynb


Suggestions & Requests
======================

If a feature of the *pyActigraphy* package is not illustrated here, do not
hesitate to suggest it by fill an issue.

Or, even better, contribute to this section by providing us with your favourite
notebook where you illustrate how this feature is relevant for your analysis.
