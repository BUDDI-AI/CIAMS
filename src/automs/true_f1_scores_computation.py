""" Module containing functions for computing the true f1 scores corresponding to various classification models """
import logging

from .exceptions import UnableToLearnBothClassesError

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
			datefmt = '%m/%d/%Y %H:%M:%S',
			level = logging.INFO)
logger = logging.getLogger(__name__)


def get_majority_minority_class_labels(y):
    """ Find the majority and minority class labels in a targat numpy array `y` """

    unique_labels, label_counts = np.unique(y, return_counts=True)
    
    # check if the label is binary
    assert len(unique_labels) == 2

    if label_counts[0] >= label_counts[1]:
        majority_class_labels = unique_labels[0]
        minority_class_labels = unique_labels[1]

    else:
        majority_class_labels = unique_labels[1]
        minority_class_labels = unique_labels[0]

    return majority_class_labels, minority_class_labels


def compute_class_imbalance_ratio(y):
    """ Find the class imbalance ratio for a targat numpy array `y` """

    # :func:`get_majority_minority_class_labels` checks if the label is binary
    majority_class_label, minority_class_label = get_majority_minority_class_labels(y)

    n_majority_class_samples = np.count_nonzero(y == majority_class_label)
    n_minority_class_samples = np.count_nonzero(y == minority_class_label)

    return n_majority_class_samples / n_minority_class_samples


def compute_binary_weighted_f1_score(y_true, y_pred):
    
    class_imbalance_ratio = compute_class_imbalance_ratio(y_true)

    majority_class_label, minority_class_label = get_majority_minority_class_labels(y_true)

    majority_class_f1_score = f1_score(y_true, y_pred, pos_label=majority_class_label)
    minority_class_f1_score = f1_score(y_true, y_pred, pos_label=minority_class_label)

    return (majority_class_f1_score + class_imbalance_ratio*minority_class_f1_score)/(1 + class_imbalance_ratio)


scorer = make_scorer(compute_binary_weighted_f1_score, greater_is_better=True)


def get_class_balanced_sample_weights(y):
    """ Generate a sample weights array which accounts for the imbalance in classes in a target numpy array `y` """ 

    n_samples = len(y)
    sample_weights = np.ones(n_samples)

    majority_class_label, minority_class_label = get_majority_minority_class_labels(y)
    # Assign a sample weight of rounded 'class imbalance ratio' to minority class and '1' to majority class
    ## Note: **round**(compute_class_imbalance_ratio(y))
    sample_weights[y == minority_class_label] = compute_class_imbalance_ratio(y)
    
    return sample_weights

    
def compute_decision_tree_true_f1_score(dataset, n_jobs=None):

    # Split the dataset into train (0.7) and test (0.3) splits
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, stratify=dataset.target)
    # Since, performing stratified sampling, majority and minority classes do not change in splits
    majority_class_label, minority_class_label = get_majority_minority_class_labels(dataset.target)

    while True:

        # class_weight='balanced' : uses y values to automatically adjust weights inversely proportional to class frequencies in input data
        clf = DecisionTreeClassifier(class_weight='balanced')

        # fit_parameters = {'sample_weight' : get_class_balanced_sample_weights(y_train)}
        fit_parameters = {}

        # Fit a decision tree on the training set and get the depth of the fitted tree
        max_depth = clone(clf).fit(X_train, y_train, **fit_parameters).get_depth()
        max_depth_search_points = [ max_depth, int(0.75*max_depth), int(0.5*max_depth), int(0.25*max_depth) ]
        # Remove all 0s from `max_depth_search_points`. (Atleast `max_depth` will be > 0)
        max_depth_search_points = list(filter(lambda x : x!=0, max_depth_search_points))
        parameters_search_grid = {'max_depth' : max_depth_search_points}

        # cv = 5 [number of splits in a (stratified)KFold] for cross-validation
        # refit = True : refit estimator using best found parameters
        # scorer : Tp evaluate predictions on the test set
        # GridSearchCV : Parameters of the optimizer used are optimized by cross-validated grid-search over a parameter grid
        search = GridSearchCV(clf, parameters_search_grid, cv=5, refit=True, scoring=scorer, n_jobs=n_jobs)
        search.fit(X_train, y_train, **fit_parameters)
        
        y_test_pred = search.predict(X_test)
        minority_class_f1_score = f1_score(y_test, y_test_pred, pos_label=minority_class_label)

        if minority_class_f1_score == 0:
            logger.warning("Failed to learn minority class. Resplitting train and test sets. Refitting decision tree classifier.")
            # Resplit the train and test datasets and fit
            X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, stratify=dataset.target)

        else:
            break
    
    binary_weighted_f1_score = compute_binary_weighted_f1_score(y_test, y_test_pred)
    return binary_weighted_f1_score


def compute_random_forest_true_f1_score(dataset, n_jobs=None):

    # Split the dataset into train (0.7) and test (0.3) splits
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, stratify=dataset.target)
    # Since, performing stratified sampling, majority and minority classes do not change in splits
    majority_class_label, minority_class_label = get_majority_minority_class_labels(dataset.target)

    while True:

        # class_weight='balanced_subsample' : weights are computed for each bootstrap sample for every tree grown, inversely proportional to class frequencies
        clf = RandomForestClassifier(class_weight='balanced_subsample')

        fit_parameters = {}

        # Fit a decision tree on the training set and get the depth of the fitted tree
        max_depth = max([tree.get_depth() for tree in clone(clf).fit(X_train, y_train, **fit_parameters).estimators_])
        max_depth_search_points = [ max_depth, int(0.75*max_depth), int(0.5*max_depth), int(0.25*max_depth) ]
        # Remove all 0s from `max_depth_search_points`. (Atleast `max_depth` will be > 0)
        max_depth_search_points = list(filter(lambda x : x!=0, max_depth_search_points))

        parameters_search_grid = {'n_estimators' : [1, 3, 7, 10], 'max_depth' : max_depth_search_points}

        # cv = 5 [number of splits in a (stratified)KFold] for cross-validation
        # refit = True : refit estimator using best found parameters
        # scorer : Tp evaluate predictions on the test set
        # GridSearchCV : Parameters of the optimizer used are optimized by cross-validated grid-search over a parameter grid
        search = GridSearchCV(clf, parameters_search_grid, cv=5, refit=True, scoring=scorer, n_jobs=n_jobs)
        search.fit(X_train, y_train, **fit_parameters)
        
        y_test_pred = search.predict(X_test)
        minority_class_f1_score = f1_score(y_test, y_test_pred, pos_label=minority_class_label)

        if minority_class_f1_score == 0:
            logger.warning("Failed to learn minority class. Resplitting train and test sets. Refitting random forest classifier.")
            # Resplit the train and test datasets and fit
            X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, stratify=dataset.target)

        else:
            break
    
    binary_weighted_f1_score = compute_binary_weighted_f1_score(y_test, y_test_pred)
    return binary_weighted_f1_score


def compute_logistic_regression_true_f1_score(dataset, n_jobs=None):

    # Split the dataset into train (0.7) and test (0.3) splits
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, stratify=dataset.target)
    # Since, performing stratified sampling, majority and minority classes do not change in splits
    majority_class_label, minority_class_label = get_majority_minority_class_labels(dataset.target)

    while True:

        # class_weight='balanced' : class associated with classses
        # solver='saga' : possible faster convergence
        # max_iter=500 : increased from 100, parameter used in training model
        clf = LogisticRegression(penalty='l1', class_weight='balanced', tol=0.01, max_iter=500, solver='saga')

        fit_parameters = {}

        parameters_search_grid = {'C' : [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]}

        # cv = 5 [number of splits in a (stratified)KFold] for cross-validation
        # refit = True : refit estimator using best found parameters
        # scorer : Tp evaluate predictions on the test set
        # GridSearchCV : Parameters of the optimizer used are optimized by cross-validated grid-search over a parameter grid
        search = GridSearchCV(clf, parameters_search_grid, cv=5, refit=True, scoring=scorer, n_jobs=n_jobs)
        search.fit(X_train, y_train, **fit_parameters)
        
        y_test_pred = search.predict(X_test)
        minority_class_f1_score = f1_score(y_test, y_test_pred, pos_label=minority_class_label)

        if minority_class_f1_score == 0:
            logger.warning("Failed to learn minority class. Resplitting train and test sets. Refitting logistic regression classifier.")
            # Resplit the train and test datasets and fit
            X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, stratify=dataset.target)

        else:
            break
    
    binary_weighted_f1_score = compute_binary_weighted_f1_score(y_test, y_test_pred)
    return binary_weighted_f1_score


def compute_k_nearest_neighbor_true_f1_score(dataset, n_jobs=None):

    # Split the dataset into train (0.7) and test (0.3) splits
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, stratify=dataset.target)
    # Since, performing stratified sampling, majority and minority classes do not change in splits
    majority_class_label, minority_class_label = get_majority_minority_class_labels(dataset.target)

    test_class_imbalance_ratio = compute_class_imbalance_ratio(y_test)

    while True:

        if test_class_imbalance_ratio < 10:

            clf = KNeighborsClassifier()
            fit_parameters = {}
            parameters_search_grid = {'n_neighbors' : [3, 5, 7, 9], 'weights' : ['uniform', 'distance'], 'leaf_size' : [2, 5, 10, 15, 20, 25, 30]}
           
        else:

            clf = KNeighborsClassifier(n_neighbors=test_class_imbalance_ratio)
            fit_parameters = {}
            parameters_search_grid = {'weights' : ['uniform', 'distance'], 'leaf_size' : [2, 5, 10, 15, 20, 25, 30]}

        # cv = 5 [number of splits in a (stratified)KFold] for cross-validation
        # refit = True : refit estimator using best found parameters
        # scorer : Tp evaluate predictions on the test set
        # GridSearchCV : Parameters of the optimizer used are optimized by cross-validated grid-search over a parameter grid
        search = GridSearchCV(clf, parameters_search_grid, cv=5, refit=True, scoring=scorer, n_jobs=n_jobs)
        search.fit(X_train, y_train, **fit_parameters)
        
        if test_class_imbalance_ratio < 10:
            y_test_pred = search.predict(X_test)

        else:
            y_pred_prob = search.predict_proba(X_test)

            n_test_samples = len(X_test)
            y_test_pred = np.full(n_test_samples, majority_class_label)

            # Classes are ordered by lexicographic order
            minority_class_lex_index = 0 if minority_class_label < majority_class_label else 1
            y_test_pred[ y_pred_prob[:,minority_class_lex_index]>(1/test_class_imbalance_ratio) ] = minority_class_label

        minority_class_f1_score = f1_score(y_test, y_test_pred, pos_label=minority_class_label)

        if minority_class_f1_score == 0:
            logger.warning("Failed to learn minority class. Resplitting train and test sets. Refitting k-nearest neighbor classifier.")
            # Resplit the train and test datasets and fit
            X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, stratify=dataset.target)

        else:
            break
    
    binary_weighted_f1_score = compute_binary_weighted_f1_score(y_test, y_test_pred)
    return binary_weighted_f1_score
    

def compute_xgboost_true_f1_score(dataset, n_jobs=None):

    # Split the dataset into train (0.7) and test (0.3) splits
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, stratify=dataset.target)
    # Since, performing stratified sampling, majority and minority classes do not change in splits
    majority_class_label, minority_class_label = get_majority_minority_class_labels(dataset.target)

    while True:

        clf = XGBClassifier()
        fit_parameters = {'sample_weight' : get_class_balanced_sample_weights(y_train)}

        parameters_search_grid = {'learning_rate' : np.linspace(0.05, 0.3, 3), 'max_depth': [1, 2, 4, 8, 16, 16, 32, 64]}

        # cv = 5 [number of splits in a (stratified)KFold] for cross-validation
        # refit = True : refit estimator using best found parameters
        # scorer : Tp evaluate predictions on the test set
        # GridSearchCV : Parameters of the optimizer used are optimized by cross-validated grid-search over a parameter grid
        search = GridSearchCV(clf, parameters_search_grid, cv=5, refit=True, scoring=scorer, n_jobs=n_jobs)
        ## TODO: Check if the GridSearchCV appropriately splits the `sample_weight` for each internal cv split
        # https://github.com/scikit-learn/scikit-learn/issues/2879
        search.fit(X_train, y_train, **fit_parameters)
        
        y_test_pred = search.predict(X_test)
        minority_class_f1_score = f1_score(y_test, y_test_pred, pos_label=minority_class_label)

        if minority_class_f1_score == 0:
            logger.warning("Failed to learn minority class. Resplitting train and test sets. Refitting xgboost classifier.")
            # Resplit the train and test datasets and fit
            X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, stratify=dataset.target)

        else:
            break
    
    binary_weighted_f1_score = compute_binary_weighted_f1_score(y_test, y_test_pred)
    return binary_weighted_f1_score


def compute_support_vector_machine_true_f1_score(dataset, n_jobs=None):

    # Split the dataset into train (0.7) and test (0.3) splits
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, stratify=dataset.target)
    # Since, performing stratified sampling, majority and minority classes do not change in splits
    majority_class_label, minority_class_label = get_majority_minority_class_labels(dataset.target)

    train_class_imbalance_ratio = compute_class_imbalance_ratio(y_train)

    i_retry = 0
    while True:

        if i_retry in range(0, 2): # i_retry : 0, 1
            # `cache_size=3000` or (~3GB) not feasible for computers with limited RAM
            clf = SVC(gamma='scale', cache_size=3000)
            fit_parameters = {}
            parameters_search_grid = {'C' : [1e-2, 1e-1, 1, 1e1, 1e2]}

        if i_retry in range(2, 5): # i_retry : 2, 3, 4
            # `cache_size=3000` or (~3GB) not feasible for computers with limited RAM
            clf = SVC(class_weight='balanced', gamma='scale', cache_size=3000, shrinking=True, max_iter=15000)
            fit_parameters = {}
            parameters_search_grid = {'C' : [1e-2, 1e-1, 1, 1e1, 1e2]}

        if i_retry in range(5, 8): # i_retry : 5, 6, 7
            class_weights_dict = {
                        majority_class_label : train_class_imbalance_ratio / i_retry,
                        minority_class_label : train_class_imbalance_ratio + (i_retry*2)
                    }
            # `cache_size=3000` or (~3GB) not feasible for computers with limited RAM
            clf = SVC(class_weight=class_weights_dict, gamma='scale', cache_size=3000, shrinking=True, max_iter=15000)
            fit_parameters = {}
            parameters_search_grid = {'C' : [1e-2, 1e-1, 1, 1e1, 1e2]}

        if i_retry in range(8, 12): # i_retry : 8, 9, 10, 11
            # `cache_size=3000` or (~3GB) not feasible for computers with limited RAM
            clf = SVC(gamma='scale', cache_size=3000, shrinking=True, max_iter=15000)

            # assign sample weights of 'class_imbalance_ratio + (i_retry*2)' to minority class and '1 - class_imbalance_ratio' to majority class
            n_train_samples = len(X_train)
            sample_weights = np.empty(n_train_samples)

            sample_weights[y_train == minority_class_label] = train_class_imbalance_ratio + (i_retry*2)
            sample_weights[y_train == majority_class_label] = 1 - train_class_imbalance_ratio

            fit_parameters = {'sample_weight' : sample_weights}
            parameters_search_grid = {'C' : [1e-2, 1e-1, 1, 1e1, 1e2]}

        else: # i_retry : 12, 13, ...
            # `cache_size=15000` or (~15B) not feasible for computers with limited RAM [EXTREME SETTINGS]
            clf = SVC(gamma='scale', cache_size=15000, tol=1e-1, shrinking=True, max_iter=150000)
            fit_parameters = {}
            parameters_search_grid = {'C' : [1e-2, 1e-1, 1, 1e1, 1e2], 'kernel' : ['rbf', 'poly'], 'degree' : [2, 3, 4, 5]}

        # cv = 5 [number of splits in a (stratified)KFold] for cross-validation
        # refit = True : refit estimator using best found parameters
        # scorer : Tp evaluate predictions on the test set
        # GridSearchCV : Parameters of the optimizer used are optimized by cross-validated grid-search over a parameter grid
        search = GridSearchCV(clf, parameters_search_grid, cv=5, refit=True, scoring=scorer, n_jobs=n_jobs)
        ## TODO: Check if the GridSearchCV appropriately splits the `sample_weight` for each internal cv split
        # https://github.com/scikit-learn/scikit-learn/issues/2879
        search.fit(X_train, y_train, **fit_parameters)

        y_test_pred = search.predict(X_test)

        minority_class_f1_score = f1_score(y_test, y_test_pred, pos_label=minority_class_label)
        majority_class_f1_score = f1_score(y_test, y_test_pred, pos_label=majority_class_label)

        i_retry += 1

        if majority_class_f1_score == 0 or minority_class_f1_score == 0:

            if i_retry == 15:
                logger.error("Unable to learn both majority and minority class patterns in dataset using support vector machine classifier after 15 attempts.")
                raise UnableToLearnBothClassesError("unable to learn both classes in dataset using SVC")

            logger.warning("Failed to learn majority or minority class. Resplitting train and test sets. Refitting svm classifier.")
            # Resplit the train and test datasets and fit
            X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, stratify=dataset.target)

        else:
            break

    binary_weighted_f1_score = compute_binary_weighted_f1_score(y_test, y_test_pred)
    return binary_weighted_f1_score


def compute_true_f1_scores(dataset, n_jobs=1):
    """ Compute the true f1-scores corresponding to various classifier models for the dataset """

    clf_models_true_f1_scores = dict()

    logger.info("Computing true f1 score for decision tree classifier")
    decision_tree_true_f1_score = compute_decision_tree_true_f1_score(dataset, n_jobs=n_jobs)
    clf_models_true_f1_scores['decision tree'] = decision_tree_true_f1_score

    logger.info("Computing true f1 score for random forest classifier")
    random_forest_true_f1_score = compute_random_forest_true_f1_score(dataset, n_jobs=n_jobs)
    clf_models_true_f1_scores['random forest'] = random_forest_true_f1_score

    logger.info("Computing true f1 score for logistic regression classifier")
    logistic_regression_true_f1_score = compute_logistic_regression_true_f1_score(dataset, n_jobs=n_jobs)
    clf_models_true_f1_scores['logistic regression'] = logistic_regression_true_f1_score

    logger.info("Computing true f1 score for k-nearest neighbor classifier")
    k_nearest_neighbor_true_f1_score = compute_k_nearest_neighbor_true_f1_score(dataset, n_jobs=n_jobs)
    clf_models_true_f1_scores['k-nearest neighbor'] = k_nearest_neighbor_true_f1_score

    logger.info("Computing true f1 score for xgboost classifier")
    xgboost_true_f1_score = compute_xgboost_true_f1_score(dataset, n_jobs=n_jobs)
    clf_models_true_f1_scores['xgboost'] = xgboost_true_f1_score

    logger.info("Computing true f1 score for support vector machine classifier")
    try:
        support_vector_machine_true_f1_score = compute_support_vector_machine_true_f1_score(dataset, n_jobs=n_jobs)
        clf_models_true_f1_scores['support vector machine'] = support_vector_machine_true_f1_score

    except UnableToLearnBothClassesError:
        logger.warning("Failed to compute true f1 score for support vector machine classifier. Skipping computation of true f1s for SVC.")

    except Exception:
        logger.warning("Some internal occurred in computing true f1 score for support vector machine classifier. Skipping computation of true f1s for SVC.")

    return clf_models_true_f1_scores
