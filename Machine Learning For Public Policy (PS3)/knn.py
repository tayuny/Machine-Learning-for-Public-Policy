# K-Nearest Neighbor
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# personal
import train_test_split

WEIGHTS = ["uniform", "distance"]
METRIC = ["euclidean", "manhattan", "chebyshev", "minkowski"]

def classifier_settings_knn(n_neighbors=5, weights="uniform", \
                    algorithm="auto", leaf_size=30, p=2, metric="minkowski",\
                    metric_params=None, n_jobs=2, **kwargs):
    '''
    Classifier for the knn model
    '''
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, \
                    algorithm=algorithm, leaf_size=leaf_size, p=p,\
                    metric=metric, metric_params=metric_params,\
                    n_jobs=n_jobs, **kwargs)
    return model


def train_knn(X_train, y_train, X_test, model):
    '''
    This function is used to train knn model
    Inputs:
        X_train, y_train, X_test: sub datasets for training purpose
        model: model of classifier used in training
    Returns: model trained, list of predicted probability
    '''
    knn_m = model.fit(X_train, y_train)
    y_predp = knn_m.predict_proba(X_test)
    return knn_m, y_predp