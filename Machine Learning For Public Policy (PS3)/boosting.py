# Boosting
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier

# personal
import train_test_split


def classifier_settings_boosting(base_estimator=None, n_estimators=50):
    '''
    Classifier for the boosting model
    '''
    model = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=n_estimators,\
                               learning_rate=1.0, algorithm="SAMME.R", random_state=None)

    return model


def train_boosting(X_train, y_train, X_test, model):
    '''
    This function is used to train boosting model
    Inputs:
        X_train, y_train, X_test: sub datasets for training purpose
        model: model of classifier used in training
    Returns: model trained, list of predicted probability
    '''
    grad_boost = model.fit(X_train, y_train)
    y_predp = grad_boost.predict_proba(X_test)
    return grad_boost, y_predp