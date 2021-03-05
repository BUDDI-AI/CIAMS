:orphan:

.. _examples:

========
Examples
========

-------------------------------------------------------
Example 1: Statlog (Australian Credit Approval) DataSet
-------------------------------------------------------

Download dataset and generate configuration file
------------------------------------------------

**Download the dataset** and **generate the configuration file** for the `Australian dataset <http://archive.ics.uci.edu/ml/datasets/statlog+(australian+credit+approval)>`_ with:

.. code-block:: bash

   wget http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/australian.dat
   mv australian.dat australian.csv

   echo -e "from automs.config import CsvConfig\nconfig = CsvConfig(sep=' ', categorical_cols=[0,3,4,5,7,8,10,11])" > australian.csv.config.py
   cat australian.csv.config.py

Command-line interface
----------------------

Run AutoMS on the Australian dataset using the **command-line interface** by running the following command in your **terminal**:

.. code-block:: bash

   automs australian.csv --subsampling --truef1 --result results_australian
   echo "AUTOMS RESULTS FOR AUSTRALIAN DATASET"
   cat results_australian

Python interface
----------------

Alternatively, run AutoMS on the Australian dataset using the **python interface** by running the following command in your **python interpreter**:

.. code-block:: python

    >>> from automs.automs import automs
    >>> is_hard_to_classify, estimated_f1_scores, true_f1_scores = automs('australian.csv', oneshot=False, return_true_f1s=True)
    >>> print(f"IS HARD TO CLASSIFY = {is_hard_to_classify}")
    >>> print(f"Estimated F1-scores = {estimated_f1_scores}")
    >>> print(f"True F1-scores = {true_f1_scores}")


--------------------------
Example 2: Titanic Dataset
--------------------------

Download dataset and generate configuration file
------------------------------------------------

**Download the dataset** and **generate the configuration file** for the `Titanic dataset <https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/problem12.html>`_ with:

.. code-block:: bash

   wget https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv

   echo -e "from automs.config import CsvConfig\nconfig = CsvConfig(header_row=0, usecols=['Survived', 'Pclass', 'Sex', 'Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Fare'], target_col=0, categorical_cols=['Pclass', 'Sex', 'Siblings/Spouses Aboard', 'Parents/Children Aboard'])" > titanic.csv.config.py
   cat titanic.csv.config.py

Command-line interface
----------------------

Run AutoMS on the Titanic dataset using the **command-line interface** by running the following command in your **terminal**:

.. code-block:: bash

   automs titanic.csv --subsampling --truef1 --result results_titanic
   echo "AUTOMS RESULTS FOR TITANIC DATASET"
   cat results_titanic

Python interface
----------------

Alternatively, run AutoMS on the Titanic dataset using the **python interface** by running the following command in your **python interpreter**:

.. code-block:: python

    >>> from automs.automs import automs
    >>> is_hard_to_classify, estimated_f1_scores, true_f1_scores = automs('titanic.csv', oneshot=False, return_true_f1s=True)
    >>> print(f"IS HARD TO CLASSIFY = {is_hard_to_classify}")
    >>> print(f"Estimated F1-scores = {estimated_f1_scores}")
    >>> print(f"True F1-scores = {true_f1_scores}")


----------------------------------------
Example 3: Pima Indians Diabetes Dataset
----------------------------------------

**Download the dataset** and **generate the configuration file** for the `Diabetes dataset <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#diabetes>`_ with:

.. code-block:: bash

   wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/diabetes
   mv diabetes diabetes.libsvm

   echo -e "from automs.config import LibsvmConfig\nconfig = LibsvmConfig()" > diabetes.libsvm.config.py
   cat diabetes.libsvm.config.py

Command-line interface
----------------------

Run AutoMS on the Diabetes dataset using the **command-line interface** by running the following command in your **terminal**:

.. code-block:: bash

   automs diabetes.libsvm --subsampling --truef1 --result results_diabetes
   echo "AUTOMS RESULTS FOR DIABETES DATASET"
   cat results_diabetes

Python interface
----------------

Alternatively, run AutoMS on the Diabetes dataset using the **python interface** by running the following command in your **python interpreter**:

.. code-block:: python

    >>> from automs.automs import automs
    >>> is_hard_to_classify, estimated_f1_scores, true_f1_scores = automs('diabetes.libsvm', oneshot=False, return_true_f1s=True)
    >>> print(f"IS HARD TO CLASSIFY = {is_hard_to_classify}")
    >>> print(f"Estimated F1-scores = {estimated_f1_scores}")
    >>> print(f"True F1-scores = {true_f1_scores}")

-----------------------------
Example 4: Ionosphere Dataset
-----------------------------

**Download the dataset** and **generate the configuration file** for the `Ionosphere dataset <https://storm.cis.fordham.edu/~gweiss/data-mining/datasets.html>`_ with:

.. code-block:: bash

   wget https://storm.cis.fordham.edu/~gweiss/data-mining/weka-data/ionosphere.arff

   echo -e "from automs.config import ArffConfig\nconfig = ArffConfig()" > ionosphere.arff.config.py
   cat ionosphere.arff.config.py

Command-line interface
----------------------

Run AutoMS on the Ionosphere dataset using the **command-line interface** by running the following command in your **terminal**:

.. code-block:: bash

   automs ionosphere.arff --oneshot --truef1 --result results_ionosphere
   echo "AUTOMS RESULTS FOR IONOSPHERE DATASET"
   cat results_ionosphere

Python interface
----------------

Alternatively, run AutoMS on the Ionosphere dataset using the **python interface** by running the following command in your **python interpreter**:

.. code-block:: python

    >>> from automs.automs import automs
    >>> is_hard_to_classify, estimated_f1_scores, true_f1_scores = automs('ionosphere.arff', oneshot=True, return_true_f1s=True)
    >>> print(f"IS HARD TO CLASSIFY = {is_hard_to_classify}")
    >>> print(f"Estimated F1-scores = {estimated_f1_scores}")
    >>> print(f"True F1-scores = {true_f1_scores}")

