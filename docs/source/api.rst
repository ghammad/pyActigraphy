:orphan:

.. _api_reference:

=============
API Reference
=============

.. contents:: Table of Contents
    :local:
    :depth: 2

:mod:`pyActigraphy`:

.. automodule:: pyActigraphy
   :no-members:
   :no-inherited-members:

I/O
===

Uniform API to read multiple actigraphy data formats.
Currently, the supported formats are:

* ActiWatch (CamNtech): .awd
* MotionWatch8 (CamNtech): .mtn
* Respironics (Philips): .rpx

Reading raw data
----------------
:mod:`pyActigraphy.io`:

.. currentmodule:: pyActigraphy.io

.. automodule:: pyActigraphy.io
   :no-members:
   :no-inherited-members:

.. autosummary::
  :toctree: _autosummary/
  :template: function.rst

    read_raw
    read_raw_awd
    read_raw_mtn
    read_raw_rpx


Reader classes
--------------
.. currentmodule:: pyActigraphy.io
.. autosummary::
  :toctree: _autosummary/
  :template: class.rst

    base.BaseRaw
    awd.awd.RawAWD
    mtn.mtn.RawMTN
    rpx.rpx.RawRPX
    reader.reader.RawReader


Metrics
=======

API to calculate various activity/rest cycle-related variables.

:mod:`pyActigraphy.metrics`:

.. currentmodule:: pyActigraphy.metrics

.. automodule:: pyActigraphy.metrics
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: _autosummary/
   :template: class.rst

   metrics.MetricsMixin
   metrics.ForwardMetricsMixin


Distributions
-------------
.. autosummary::
   :toctree: _autosummary
   :template: function.rst

   metrics._average_daily_activity
   metrics.MetricsMixin.average_daily_activity
   metrics.MetricsMixin.average_daily_light


Activity variables
------------------
.. autosummary::
    :toctree: _autosummary
    :template: function.rst

    metrics.MetricsMixin.ADAT
    metrics.MetricsMixin.ADATp

Non-parametric variables
------------------------
.. autosummary::
    :toctree: _autosummary
    :template: function.rst

    metrics.MetricsMixin.IS
    metrics.MetricsMixin.ISm
    metrics.MetricsMixin.ISp
    metrics.MetricsMixin.IV
    metrics.MetricsMixin.IVm
    metrics.MetricsMixin.IVp
    metrics.MetricsMixin.L5
    metrics.MetricsMixin.L5p
    metrics.MetricsMixin.M10
    metrics.MetricsMixin.M10p
    metrics.MetricsMixin.RA
    metrics.MetricsMixin.RAp

Transition probability variables
--------------------------------
.. autosummary::
    :toctree: _autosummary
    :template: function.rst

    metrics.MetricsMixin.pAR
    metrics.MetricsMixin.pRA
    metrics.MetricsMixin.kAR
    metrics.MetricsMixin.kRA
