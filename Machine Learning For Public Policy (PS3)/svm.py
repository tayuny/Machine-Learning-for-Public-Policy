# SVM
import pandas as pd
import numpy as np
from sklearn.svm import SVC

# personal
import train_test_split

KERNEL = ["linear", "poly", "rbf", "sigmoid"]
CLASS_WEIGHT = [None, "balanced"]
DECISION_FUNCTION_SHAPE = ["ovr", "ovo"]


def classifier_settings_svm(C=1.0, kernel="rbf", degree=3, \
                gamma="auto", coef0=0.0, shrinking=True,\
                probability=True, tol=0.001, cache_size=200,\
                class_weight=None, verbose=False, max_iter=1000,\
                decision_function_shape="ovr", random_state=None):
    '''
    Classifier for the svm model
    '''
    model = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma,\
                coef0=coef0, shrinking=shrinking, probability=probability,\
                tol=tol, cache_size=cache_size, class_weight=class_weight,\
                verbose=verbose, max_iter=max_iter,\
                decision_function_shape=decision_function_shape, \
                random_state=random_state)

    return model


def train_svm(X_train, y_train, X_test, model):
    '''
    This function is used to train svm model
    Inputs:
        X_train, y_train, X_test: sub datasets for training purpose
        model: model of classifier used in training
    Returns: model trained, list of predicted probability
    '''
    svm_m = model.fit(X_train, y_train)
    # y_predp = svm_m.predict_proba(X_test)
    confidence_score = model.decision_function(X_test)
    y_predp = np.zeros((len(confidence_score), 2))
    y_predp[:, 1] = confidence_score
    return svm_m, y_predp

