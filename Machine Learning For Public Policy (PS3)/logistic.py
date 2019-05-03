# Logistic Regression
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# personal
import train_test_split

PANELTY = ["l1", "l2"]
CLASS_WEIGHT = [None, "balanced"]
SOLVER = ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
MULTI_CLASS = ["ovr", "multinomial", "auto"]

def classifier_settings_log(penalty="l2", dual=False, tol=0.0001, C=1.0,\
                               fit_intercept=True, intercept_scaling=1, \
                               class_weight=None, random_state=None, \
                               solver="liblinear", max_iter=100, multi_class="ovr",\
                               verbose=0, warm_start=False, n_jobs=2):
    '''
    Classifier for the logistics model
    '''
    model = LogisticRegression(penalty=penalty, dual=dual, tol=tol, C=C,\
                        fit_intercept=fit_intercept,\
                        intercept_scaling=intercept_scaling,\
                        class_weight=class_weight, random_state=random_state, \
                        solver=solver, max_iter=max_iter, \
                        multi_class=multi_class, verbose=verbose, \
                        warm_start=warm_start, n_jobs=n_jobs)

    return model

def train_logistics(X_train, y_train, X_test, model):
    '''
    This function is used to train logistics model
    Inputs:
        X_train, y_train, X_test: sub datasets for training purpose
        model: model of classifier used in training
    Returns: model trained, list of predicted probability
    '''
    logistics = model.fit(X_train, y_train)
    y_predp = logistics.predict_proba(X_test)
    return logistics, y_predp
