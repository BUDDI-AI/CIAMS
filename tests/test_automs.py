import os
import random

import numpy as np
from pytest import approx

from automs.automs import automs

DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def test_automs(tmpdir):

    set_seed(42)

    dataset_filename = os.path.join(DATA_PATH, 'sonar.csv')
    is_hard_to_classify, estimated_f1_scores, true_f1_scores = automs(dataset_filename, oneshot=True, num_processes=1, warehouse_path=tmpdir.strpath, return_true_f1s=True)

    assert is_hard_to_classify == False
    assert all(estimated_f1_scores[clf_model] == approx(f1_score) for clf_model, f1_score in {'decision tree': 0.66704917, 'random forest': 0.7747609, 'logistic regression': 0.8416766, 'k-nearest neighbor': 0.86353064, 'xgboost': 0.7316445, 'support vector machine': 0.5480081}.items())
    assert all(true_f1_scores[clf_model] == approx(f1_score) for clf_model, f1_score in {'decision tree': 0.7929221962008847, 'random forest': 0.816084656084656, 'logistic regression': 0.7826455026455025, 'k-nearest neighbor': 0.8186275113129597, 'xgboost': 0.7705761316872427, 'support vector machine': 0.8186275113129597}.items())

