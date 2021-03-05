""" Module containing functions for estimating the f1 scores corresponding to various classification models """
import logging
import os
import pickle

import numpy as np
import onnxruntime as rt

from .cluster_indices_generation import FEATURE_VECTOR_CLUSTER_INDICES_ORDER_TRIPLES


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -     %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


MODELS_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')


def estimate_decision_tree_f1_scores(feature_vectors):

    # Load the decision tree f1-score estimation regressor model
    model_onnx_filename = os.path.join(MODELS_PATH, 'decision_tree_f1_estimator.onnx')
    sess = rt.InferenceSession(model_onnx_filename)

    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    estimated_f1_scores = sess.run([label_name], {input_name: feature_vectors})[0].squeeze(-1)
    return estimated_f1_scores


def estimate_random_forest_f1_scores(feature_vectors):

    # Load the random forest f1-score estimation regressor model
    model_onnx_filename = os.path.join(MODELS_PATH, 'random_forest_f1_estimator.onnx')
    sess = rt.InferenceSession(model_onnx_filename)

    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    estimated_f1_scores = sess.run([label_name], {input_name: feature_vectors})[0].squeeze(-1)
    return estimated_f1_scores


def estimate_logistic_regression_f1_scores(feature_vectors):

    # Load the logistic regression f1-score estimation regressor model
    model_onnx_filename = os.path.join(MODELS_PATH, 'logistic_regression_f1_estimator.onnx')
    sess = rt.InferenceSession(model_onnx_filename)

    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    estimated_f1_scores = sess.run([label_name], {input_name: feature_vectors})[0].squeeze(-1)
    return estimated_f1_scores


def estimate_k_nearest_neighbors_f1_scores(feature_vectors):

    # Load the k-nearest neighbors f1-score estimation regressor model
    model_onnx_filename = os.path.join(MODELS_PATH, 'k_nearest_neighbor_f1_estimator.onnx')
    sess = rt.InferenceSession(model_onnx_filename)

    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    estimated_f1_scores = sess.run([label_name], {input_name: feature_vectors})[0].squeeze(-1)
    return estimated_f1_scores


def estimate_xgboost_f1_scores(feature_vectors):

    # Load the xgboost f1-score estimation regressor model
    model_onnx_filename = os.path.join(MODELS_PATH, 'xgboost_f1_estimator.onnx')
    sess = rt.InferenceSession(model_onnx_filename)

    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    estimated_f1_scores = sess.run([label_name], {input_name: feature_vectors})[0].squeeze(-1)
    return estimated_f1_scores


def estimate_support_vector_machine_f1_scores(feature_vectors):

    # Load the support vector classifier f1-score estimation regressor model
    model_onnx_filename = os.path.join(MODELS_PATH, 'support_vector_machine_f1_estimator.onnx')
    sess = rt.InferenceSession(model_onnx_filename)

    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    estimated_f1_scores = sess.run([label_name], {input_name: feature_vectors})[0].squeeze(-1)
    return estimated_f1_scores


def estimate_f1_scores(feature_vectors):
    """ Estimate the f1-scores corresponding to various classifier models from bag's metafeatures """

    feature_vectors = np.array(feature_vectors, dtype=np.float32)

    ## Verify that features in feature vector generated is same as features expected by the f1 estimation models
    models_features_filename = os.path.join(MODELS_PATH, 'features.pkl')
    with open(models_features_filename, 'rb') as f_features:
        f1_estimators_feature_input_cluster_indices_triple = pickle.load(f_features)

    if not f1_estimators_feature_input_cluster_indices_triple == FEATURE_VECTOR_CLUSTER_INDICES_ORDER_TRIPLES:
        logger.error("Mismatch between features in feature vector generated using cluster indices and features in feature vector expected by f1-score estimation models.")
        raise ValueError("mismatch between features generated, expected by f1 estimation models")

    decision_tree_estimated_f1_scores = estimate_decision_tree_f1_scores(feature_vectors)
    random_forest_estimated_f1_scores = estimate_random_forest_f1_scores(feature_vectors)
    logistic_regression_estimated_f1_scores = estimate_logistic_regression_f1_scores(feature_vectors)
    k_nearest_neighbor_estimated_f1_scores = estimate_k_nearest_neighbors_f1_scores(feature_vectors)
    xgboost_estimated_f1_scores = estimate_xgboost_f1_scores(feature_vectors)
    support_vector_machine_estimated_f1_scores = estimate_support_vector_machine_f1_scores(feature_vectors)

    clf_models_estimated_f1_scores = {
        'decision tree': decision_tree_estimated_f1_scores,
        'random forest': random_forest_estimated_f1_scores,
        'logistic regression': logistic_regression_estimated_f1_scores,
        'k-nearest neighbor': k_nearest_neighbor_estimated_f1_scores,
        'xgboost': xgboost_estimated_f1_scores,
        'support vector machine': support_vector_machine_estimated_f1_scores
    }

    return clf_models_estimated_f1_scores

