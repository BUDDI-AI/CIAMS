:orphan:

.. _faq:

FAQs
====

Can I use AutoMS to predict the classification complexity and estimate the F1-scores for a multi-class problem ?
----------------------------------------------------------------------------------------------------------------

**No**. AutoMS supports predicting classification complexity and estimating f1-scores for **binary classification datasets** only. However, the idea can be extended to multi-class problems using the one-vs-rest statergy.


How does AutoMS predict the classification complexity and estimate the f1 scores for a datasets ?
--------------------------------------------------------------------------------------------------

AutoMS extracts **clustering-based metafeatures** from the dataset and uses fitted classification and regression models to predict the **classification complexity** and estimate the **maximum achievable f1-scores** corresponding to various classifier models for the dataset.


.. _what_are_oneshot_and_subsampling_approaches:

What are the oneshot and sub-sampling approaches ?
--------------------------------------------------

**Oneshot** approach processes the dataset as a whole to predict the classification complexity and estimate the f1-scores for the dataset.

**Sub-sampling** approach subsamples the datasets into stratified overlapping subsamples and processes these subsamples to predict the classification complexity and estimate the f1-scores for the dataset.

.. _when_should_i_use_the_oneshot_and_subsampling_approaches:

When should I use the oneshot and sub-sampling approaches ?
-----------------------------------------------------------

Oneshot is preferable while dealing with datasets having considerably small number of data points. Typically, datasets with less than 2000 data points is considered to be small. Sub-sampling approach is preferable in case of large datasets.


How are the f1-scores of the classes averaged in the estimated f1-scores and computed true f1-scores ?
------------------------------------------------------------------------------------------------------

AutoMS uses a variant of weighted average f1-score for binary datasets from **class imbalance learning** literature that weights the f1-scores of classes inversely proportional to their proportions in the dataset.

.. math:: f1 = \frac{f1_{majority\ class} + R * f1_{minority\ class}}{1 + R}

where, :math:`R` is the class imbalance ratio, which is the fraction of number of samples in the majority class to the number of samples in the    minority class.


Why do you estimate the weighted average f1-score from class imbalance learning literature as opposed to the regular f1-score ?
-------------------------------------------------------------------------------------------------------------------------------

The weighted f1-score will account for the class imbalance in the dataset. This is more preferable when dealing with high class imbalance datasets.


What does your predicted classification complexity signify?
-----------------------------------------------------------

The predicted classification complexity suggests if the given dataset is hard to classify or not. Hard to classify implies that the none of the chosen classification methods will be able to score f1 > 0.6. A dataset is considered easy to classify even if one of the chosen classification method is able to score f1 > 0.6.


How accurate are your classification complexity predictions ?
-------------------------------------------------------------

Our classification complexity predictions have an accuracy of approximately 0.9 on our test set.


How accurate are your f1-score estimations ?
--------------------------------------------

In most cases, the estimated f1-scores might be off by +/- 0.10 to the true f1-scores. But, it is emperically proven that the actual top performing classification method is one among the top three classification methods as inferred from the estimated f1-scores. 


