:orphan:

.. _api:

====
APIs
====

------
AutoMS
------

.. _api_automs_cmdline:

Command line interface
----------------------

.. argparse::
    :module: automs.main
    :func: get_parser
    :prog: automs
    :nodefault:
 
.. _api_automs_py:

Python interface
----------------

.. autofunction:: automs.automs.automs

.. note::
    The default values for the keyword arguments ``oneshot``, ``num_processes`` and ``warehouse_path`` for function :func:`automs.automs.automs` will be overriden with default values configured using *AutoMS configuration wizard*.

.. _api_dataset_config:

---------------------
Dataset Configuration
---------------------

CSV Data Format
---------------

.. autoclass:: automs.config.CsvConfig

LIBSVM Data Format
------------------

.. autoclass:: automs.config.LibsvmConfig

ARFF Data Format
----------------

.. autoclass:: automs.config.ArffConfig
