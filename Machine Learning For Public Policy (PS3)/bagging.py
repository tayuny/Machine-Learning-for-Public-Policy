# Bagging
import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingClassifier

# personal
import train_test_split


def classifier_settings_bagging(base_estimator=None, n_estimators=10, max_samples=1.0,\
                 max_features=1.0, bootstrap=True, bootstrap_features=False,\
                 oob_score=False, warm_start=False, n_jobs=2, \
                 random_state=None, verbose=0):
    '''
    Classifier for the bagging model
    '''
    model = BaggingClassifier(base_estimator=base_estimator, 
                n_estimators=n_estimators, max_samples=max_samples, \
                max_features=max_features, bootstrap=bootstrap, \
                bootstrap_features=bootstrap_features, oob_score=oob_score,\
                warm_start=warm_start, n_jobs=n_jobs, \
                random_state=random_state, verbose=verbose)

    return model


def train_bagging(X_train, y_train, X_test, model):
    '''
    This function is used to train bagging model
    Inputs:
        X_train, y_train, X_test: sub datasets for training purpose
        model: model of classifier used in training
    Returns: model trained, list of predicted probability
    '''
    bagging = model.fit(X_train, y_train)
    y_predp = bagging.predict_proba(X_test)
    return bagging, y_predp