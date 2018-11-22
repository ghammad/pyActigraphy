API Reference
*************

.. contents:: Table of Contents
   :depth: 2

I/O
===

Uniform API to read multiple actigraphy data formats.
Currently, the supported formats are:

* ActiWatch (CamNtech): .awd
* Respironics (Philips): .rpx
* MotionWatch8 (CamNtech): .mtn

Generator functions
-----------------------
.. currentmodule:: pyActigraphy.io
.. autosummary::
    :toctree: _autosummary

    read_raw
    read_raw_awd
    read_raw_mtn
    read_raw_rpx

AWD
---
.. currentmodule:: pyActigraphy.io.awd.awd
.. autosummary::
    :toctree: _autosummary

    RawAWD

MTN
---
.. currentmodule:: pyActigraphy.io.mtn.mtn
.. autosummary::
    :toctree: _autosummary

    RawMTN

RPX
---
.. currentmodule:: pyActigraphy.io.rpx.rpx
.. autosummary::
    :toctree: _autosummary

    RawRPX

Metrics
=======

API to calculate various wake/sleep cycle-related variables.

Non-parametric variables
------------------------
.. currentmodule:: pyActigraphy.metrics.metrics.MetricsMixin
.. autosummary::
    .. :template: module.rst
    :toctree: _autosummary

    IS
    ISm
    ISp
    IV
    IVm
    IVp
    L5
    L5p
    M10
    M10p
    RA
    RAp
