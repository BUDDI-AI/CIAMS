# CIAMS - Clustering Indices based Automatic classification Model Selection

## The code for CIAMS is packaged under the name **AutoMS**.

*AutoMS (Automatic Model Selection Using Cluster Indices)* is a **machine learning model recommendation** and **dataset classifiability assessment** toolkit.

Find the documentation [**here**](https://automs.readthedocs.io/en/latest/).

## Table of Contents

- [Overview](#overview)
- [Installing AutoMS](#installing-automs)
- [Configuring AutoMS](#configuring-automs)
- [Running AutoMS on a dataset](#running-automs-on-a-dataset)
	- [Step 1: Downloading the dataset](#step-1-downloading-the-dataset)
	- [Step 2: Creating the dataset configuration file](#step-2-creating-the-dataset-configuration-file)
	- [Step 3: Predicting Classification Complexity and Estimating F1 scores for the dataset](#step-3-predicting-classification-complexity-and-estimating-f1-scores-for-the-dataset)
		- [Command-line Interface](#command-line-interface)
		- [Python Interface](#python-interface)
- [Documentation](#documentation)		
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)


## Overview

AutoMS estimates the **maximum achievable f1 scores** corresponding to various classifier models for a given **binary classification dataset**. These estimated scores help make informed choices about the classifier models to experiment on the dataset, and also speculate what to expect from each of them. AutoMS also predicts the **classification complexity** of the dataset which characterizes the ease with which the dataset can be classified.

AutoMS extracts **clustering-based metafeatures** from the dataset and uses fitted classification and regression models to predict the classification complexity and estimate the maximum achievable f1-scores corresponding to various classifier models for the dataset.

> **Note:**
> *f1-score* in all discussions pertaining to AutoMS refers to a variant of **weighted average f1-score** for binary datasets from **class imbalance learning** literature that weights the f1-scores of classes inversely proportional to their proportions in the dataset.
>
> <img src="https://render.githubusercontent.com/render/math?math=f1%20%3D%20%5Cfrac%7Bf1_%7Bmajority%5C%20class%7D%20%2B%20R%20%2A%20f1_%7Bminority%5C%20class%7D%7D%7B1%20%2B%20R%7D">
> 
> where, `R` is the class imbalance ratio, which is the fraction of number of samples in the majority class to the number of samples in the minority class.

## Installing AutoMS

We recommend installing *automs* into a [virtual environment](https://docs.python-guide.org/dev/virtualenvs/#lower-level-virtualenv).

```console
$ sudo pip install virtualenv
```
```console
$ virtualenv --python=python3.6 automs-venv
$ source automs-venv/bin/activate
$ pip install automs
```

> **Tip:** If you **encounter errors** in installing AutoMS, **install** ``python3.6-dev`` system package (which contains the header files and static library for Python) and, then attempt installing ``automs`` again.
> ```console
> $ sudo apt-get install python3.6-dev
> $ pip install automs
> ```

## Configuring AutoMS

The **default configurations** with which to run `automs` can be configured using the **AutoMS Configuration Wizard** with:

```console
$ automs-config
```

The configured defaults can be overriden for each invocation of `automs` by suppling appropriate arguments to the command-line or python interface.

## Running AutoMS on a dataset

### Step 1: Downloading the dataset

Download a **binary classification dataset** of choice (in csv, libsvm or arff format) from the web. In this illustration, we will be using the [Connectionist Bench (Sonar, Mines vs. Rocks) Data Set](https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks)). Download the dataset in csv format from [here](https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data) with:

```console
$ wget https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data 
```

Change the current working directory to the directory into which the dataset was downloaded. **Rename the dataset file to have a '.csv' extension**.

```console
$ mv sonar.all-data sonar.csv
```

> **Note:**
> AutoMS infers the data format of a dataset file from its filename extension. Therefore, you must rename the dataset file to have a filename extension that corresponds to its data format. Supported filename extensions (and data formats) are '**.csv**', '**.libsvm**' and '**.arff**'.

### Step 2: Creating the dataset configuration file

The configuration file for the dataset encodes information about the structure of the dataset file.

**Create a dataset configuration file** for the dataset **in the same directory as the dataset file**, with **filename same as the dataset filename suffixed with a '.config.py' extension** (i.e., in this case `sonar.csv.config.py`).

```console
$ echo -e "from automs.config import CsvConfig\nconfig = CsvConfig()" > sonar.csv.config.py
$ cat sonar.csv.config.py
```

For examples of the **configuration file content** corresponding to variety of dataset files, refer to the [examples](https://automs.readthedocs.io/en/latest/examples.html) section in documentation.

> **Note:**
> For the dataset file `sonar.csv`, the contents of the dataset configuration file `sonar.csv.config.py` is:
> ```python
> from automs.config import CsvConfig
> config = CsvConfig()
> ```
> Since, the dataset file in this case is aligned with the default values of the arguments to `CsvConfig` class, no arguments have been **explicitly** passed to `CsvConfig` class in the creation of the `config` object. However, you may need to override some of the default values of the arguments to your data format specific dataset configuration class in the creation of the `config` object, to suit to your dataset file.

For information about the dataset configuration classes corresponding to the various data formats and the arguments they accept, refer to [API documentation of Dataset Configuration Classes](https://automs.readthedocs.io/en/latest/api.html#dataset-configuration).

### Step 3: Predicting Classification Complexity and Estimating F1 scores for the dataset

#### Command-line Interface

```console
$ automs sonar.csv --oneshot --truef1 --result sonar_results
```

For the more information about the **oneshot** and **subsampling** approaches, refers to [What are the oneshot and sub-sampling appeoaches ?](https://automs.readthedocs.io/en/latest/faq.html#what-are-the-oneshot-and-sub-sampling-approaches) and [When should I use the oneshot and sub-sampling approaches ?](https://automs.readthedocs.io/en/latest/faq.html#when-should-i-use-the-oneshot-and-sub-sampling-approaches) in the [FAQ](https://automs.readthedocs.io/en/latest/faq.html) section in documentation.

The predicted classification complexity, estimated f1-score and true f1-score results for the dataset should be available in the `sonar_results` file after the completion of execution of the program.

```console
$ cat sonar_results
```

> **Note:**
> The predicted **classification complexity** **boolean** value indicates if the dataset can be classified with a **f1-score > 0.6** using any of the classification methods. ``True`` indicates that the dataset is **hard to to classify** and ``False`` indicates that the dataset is **easy to classify**.
> 
> The **estimated f1-scores** corresponding to various classifier models should help identify the **candidate top performing classification methods** for the dataset, and help reduce the search space of classification algorithms to be experimented on the dataset.

For more information about the AutoMS command line interface and the arguments it accepts, refer to [API Documentation for AutoMS command line interface](https://automs.readthedocs.io/en/latest/api.html#command-line-interface).

```console
$ automs --help
```

#### Python Interface

```pycon
>>> from automs.automs import automs
>>> is_hard_to_classify, estimated_f1_scores, true_f1_scores = automs('sonar.csv', oneshot=True, return_true_f1s=True)
>>> print(f"IS HARD TO CLASSIFY = {is_hard_to_classify}")
>>> print(f"Estimated F1-scores = {estimated_f1_scores}")
>>> print(f"True F1-scores = {true_f1_scores}")
```

For more information about the AutoMS python interface and the arguments it accepts, refer to [API Documentation for AutoMS python interface](https://automs.readthedocs.io/en/latest/api.html#python-interface).

```pycon
>>> from automs.automs import automs
>>> help(automs)
```

> **Tip:**
> Inspect the configured (or specified) **warehouse sub-directory** corresponding to the **last run of AutoMS** for result files `results.xlsx`, `predicted_classification_complexity`, `estimated_f1_scores` and `true_f1_scores`, and the intermediate data subsample files in its `bags/` sub-directory.
>
> ```console
> $ ls <Path to configured AutoMS warehouse>
> $ cd <Path to configured AutoMS warehouse>/sonar.csv/
> $ tail -n +1 predicted_classification_complexity estimated_f1_scores true_f1_scores
> $ xdg-open results.xlsx
> ```

## Documentation

The AutoMS documentation is hosted at [https://automs.readthedocs.io/](https://automs.readthedocs.io/en/latest/).

## Authors

* [Sudarsun Santhiappan](https://www.linkedin.com/in/sudarsun/), IIT Madras & BUDDI.AI
* [Nitin Shravan](https://www.linkedin.com/in/nitin-shravan-b56bb134/), BUDDI.AI

## Acknowledgments

* [Mukesh Reghu](https://github.com/elixir-code), BUDDI.AI
* [Jeshuren Chelladurai](https://jeshuren.github.io/), IIT Madras & BUDDI.AI
