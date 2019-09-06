====================
Python API Reference
====================

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

.. autosummary::
   :toctree: _autosummary/
   :template: function.rst

    read_raw
    read_raw_awd
    read_raw_mtn
    read_raw_rpx


Reader classes
--------------
.. currentmodule:: pyActigraphy.io.BaseRaw
.. autoclass:: pyActigraphy.io.BaseRaw

    .. rubric:: Attributes
    .. autoautosummary:: pyActigraphy.io.BaseRaw
        :attributes:

    .. rubric:: Methods
    .. autosummary::
        :toctree: _autosummary/

        duration
        length
        time_range
        mask_fraction
        binarized_data
        resampled_data
        resampled_light
        read_sleep_diary

    .. rubric:: Daily profiles
    .. autosummary::
        :toctree: _autosummary/

        average_daily_activity
        average_daily_light

    .. rubric:: Total activity
    .. autosummary::
        :toctree: _autosummary/

        ADAT
        ADATp

    .. rubric:: Non-parametric methods
    .. autosummary::
        :toctree: _autosummary/

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

    .. rubric:: Transition probabilities
    .. autosummary::
        :toctree: _autosummary/

        pRA
        pAR
        kRA
        kAR

    .. rubric:: Activity/Rest identification
    .. autosummary::
        :toctree: _autosummary/

        Chronosapiens
        Crespo
        CK
        Oakley
        Sadeh
        Scripps
        SoD
        fSoD

    .. rubric:: Activity onset/offset
    .. autosummary::
        :toctree: _autosummary/

        AoffT
        AonT
        Chronosapiens_AoT
        Crespo_AoT


    .. .. autoautosummary:: pyActigraphy.io.BaseRaw
    ..     :methods:

.. currentmodule:: pyActigraphy.io.RawReader
.. autoclass:: pyActigraphy.io.RawReader
