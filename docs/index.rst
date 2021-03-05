.. AutoMS documentation master file, created by
   sphinx-quickstart on Wed Feb 19 13:43:19 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

======
AutoMS
======

*AutoMS (Automatic Model Selection Using Cluster Indices)* is a **machine learning model recommendation** and **dataset classifiability assessment** toolkit.

AutoMS estimates the **maximum achievable f1 scores** corresponding to various classifier models for a given **binary classification dataset**. These estimated scores help make informed choices about the classifier models to experiment on the dataset, and also speculate what to expect from each of them. AutoMS also predicts the **classification complexity** of the dataset which characterizes the ease with which the dataset can be classified.

AutoMS extracts **clustering-based metafeatures** from the dataset and uses fitted classification and regression models to predict the classification complexity and estimate the maximum achievable f1-scores corresponding to various classifier models for the dataset.

.. note::
    *f1-score* in all discussions pertaining to AutoMS refers to a variant of **weighted average f1-score** for binary datasets from **class imbalance learning** literature that weights the f1-scores of classes inversely proportional to their proportions in the dataset.

    .. math:: f1 = \frac{f1_{majority\ class} + R * f1_{minority\ class}}{1 + R}

    where, :math:`R` is the class imbalance ratio, which is the fraction of number of samples in the majority class to the number of samples in the minority class.
        

Installing AutoMS
=================

We recommend installing *automs* into a `virtual environment <https://docs.python-guide.org/dev/virtualenvs/#lower-level-virtualenv>`_.

.. code-block:: console

    $ sudo pip install virtualenv


.. code-block:: console

    $ virtualenv --python=python3.6 automs-venv
    $ source automs-venv/bin/activate
    $ pip install automs

.. hint::
    If you **encounter errors** in installing AutoMS, **install** ``python3.6-dev`` system package (which contains the header files and static library for Python) and, then attempt installing ``automs`` again.

    .. code-block:: console

        $ sudo apt-get install python3.6-dev
        $ pip install automs

Configuring AutoMS
==================

The **default configurations** with which to run :program:`automs` can be configured using the **AutoMS Configuration Wizard** with:

.. code-block:: console

    $ automs-config 

The configured defaults can be overriden for each invocation of :program:`automs` by suppling appropriate arguments to the command-line or python interface.

Running AutoMS on a dataset
===========================

-------------------------------
Step 1: Downloading the dataset
-------------------------------

Download a **binary classification dataset** of choice (in csv, libsvm or arff format) from the web. In this illustration, we will be using the `Connectionist Bench (Sonar, Mines vs. Rocks) Data Set <https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks)>`_. Download the dataset in csv format from :download:`here <https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data>` with:

.. code-block:: console

    $ wget https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data 

Change the current working directory to the directory into which the dataset was downloaded. **Rename the dataset file to have a '.csv' extension**.

.. code-block:: console

    $ mv sonar.all-data sonar.csv

.. note::
    AutoMS infers the data format of a dataset file from its filename extension. Therefore, you must rename the dataset file to have a filename extension that corresponds to its data format. Supported filename extensions (and data formats) are '**.csv**', '**.libsvm**' and '**.arff**'.

-----------------------------------------------
Step 2: Creating the dataset configuration file
-----------------------------------------------

The configuration file for the dataset encodes information about the structure of the dataset file.

**Create a dataset configuration file** for the dataset **in the same directory as the dataset file**, with **filename same as the dataset filename suffixed with a '.config.py' extension** (i.e., in this case :file:`sonar.csv.config.py`).

.. code-block:: console

    $ echo -e "from automs.config import CsvConfig\nconfig = CsvConfig()" > sonar.csv.config.py
    $ cat sonar.csv.config.py

For examples of the **configuration file content** corresponding to variety of dataset files, refer to the :ref:`examples` section.

.. note::
    For the dataset file :file:`sonar.csv`, the contents of the dataset configuration file :file:`sonar.csv.config.py` is:

    .. code-block:: python
       :caption: sonar.csv.config.py

        from automs.config import CsvConfig
        config = CsvConfig()

    Since, the dataset file in this case is aligned with the default values of the arguments to :class:`CsvConfig` class, no arguments have been **explicitly** passed to :class:`CsvConfig` class in the creation of the :obj:`config` object. However, you may need to override some of the default values of the arguments to your data format specific dataset configuration class in the creation of the :obj:`config` object, to suit to your dataset file.

For information about the dataset configuration classes corresponding to the various data formats and the arguments they accept, refer to :ref:`API documentation of Dataset Configuration Classes<api_dataset_config>`.

-------------------------------------------------------------------------------------
Step 3: Predicting Classification Complexity and Estimating F1 scores for the dataset
-------------------------------------------------------------------------------------

Command-line Interface
----------------------

.. code-block:: console

    $ automs sonar.csv --oneshot --truef1 --result sonar_results

For the more information about the **oneshot** and **subsampling** approaches, refers to :ref:`what_are_oneshot_and_subsampling_approaches` and :ref:`when_should_i_use_the_oneshot_and_subsampling_approaches` in the :ref:`faq` section.

The predicted classification complexity, estimated f1-score and true f1-score results for the dataset should be available in the :file:`sonar_results` file after the completion of execution of the program.

.. code-block:: console

    $ cat sonar_results

.. note::

    The predicted **classification complexity** **boolean** value indicates if the dataset can be classified with a **f1-score > 0.6** using any of the classification methods. ``True`` indicates that the dataset is **hard to to classify** and ``False`` indicates that the dataset is **easy to classify**.

    The **estimated f1-scores** corresponding to various classifier models should help identify the **candidate top performing classification methods** for the dataset, and help reduce the search space of classification algorithms to be experimented on the dataset.

For more information about the AutoMS command line interface and the arguments it accepts, refer to :ref:`API Documentation for AutoMS command line interface<api_automs_cmdline>`.

.. code-block:: console

    $ automs --help

Python Interface
----------------

.. code-block:: python

    >>> from automs.automs import automs
    >>> is_hard_to_classify, estimated_f1_scores, true_f1_scores = automs('sonar.csv', oneshot=True, return_true_f1s=True)
    >>> print(f"IS HARD TO CLASSIFY = {is_hard_to_classify}")
    >>> print(f"Estimated F1-scores = {estimated_f1_scores}")
    >>> print(f"True F1-scores = {true_f1_scores}")


For more information about the AutoMS python interface and the arguments it accepts, refer to :ref:`API Documentation for AutoMS python interface<api_automs_py>`.

.. code-block:: python

    >>> from automs.automs import automs
    >>> help(automs)

.. tip::
    Inspect the configured (or specified) **warehouse sub-directory** corresponding to the **last run of AutoMS** for result files :file:`results.xlsx`, :file:`predicted_classification_complexity`, :file:`estimated_f1_scores` and :file:`true_f1_scores`, and the intermediate data subsample files in its :file:`bags/` sub-directory.

    .. code-block:: console
    
        $ ls <Path to configured AutoMS warehouse>
        $ cd <Path to configured AutoMS warehouse>/sonar.csv/
        $ tail -n +1 predicted_classification_complexity estimated_f1_scores true_f1_scores
        $ xdg-open results.xlsx


Manual
======

* :ref:`api`
* :ref:`examples`
* :ref:`faq`

