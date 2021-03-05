""" Module containing functions for predicting the classification complexity """
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

def predict_classification_complexity(feature_vectors):
    """ Predict the classification complexity (a.k.a. classification hardness) from bag's metafeatures """

    feature_vectors = np.array(feature_vectors, dtype=np.float32)

    ## Verify that features in feature vector generated is same as features expected by the hardness classifier
    models_features_filename = os.path.join(MODELS_PATH, 'features.pkl')
    with open(models_features_filename, 'rb') as f_features:
        hardness_classifier_feature_input_cluster_indices_triple = pickle.load(f_features)

    if not hardness_classifier_feature_input_cluster_indices_triple == FEATURE_VECTOR_CLUSTER_INDICES_ORDER_TRIPLES:
        logger.error("Mismatch between features in feature vector generated using cluster indices and features in feature vector expected by hardness classifier model.")
        raise ValueError("mismatch between features generated, expected by hardness classifier model")

    # Load the classification hardness classification model
    model_onnx_filename = os.path.join(MODELS_PATH, 'hardness_classifier.onnx')
    sess = rt.InferenceSession(model_onnx_filename)

    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    # model returns 1 for easy to classify, 0 for hard to classify
    classification_complexity = sess.run([label_name], {input_name: feature_vectors})[0]
    # convert the classification complexities to boolean value with 'False' for easy and 'True' for hard
    classification_complexity = (classification_complexity == 0)
    return classification_complexity
