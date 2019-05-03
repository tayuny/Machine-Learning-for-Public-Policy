# model for decision tree and random forest
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import graphviz

# personal
import train_test_split

CRITERION = ["gini", "entropy"]
SPLITTER = ["random", "best"]
CLASS_WEIGHT = [None, "balanced", "balanced_subsample"]
################################################################################################

# Decision Tree

def classifier_settings_dt(criterion='gini', splitter="best",\
                    max_depth=None, min_samples_split=2, min_samples_leaf=1,\
                    min_weight_fraction_leaf=0.0, max_features=None,\
                    random_state=None, max_leaf_nodes=None, \
                    min_impurity_decrease=0.0, min_impurity_split=None,\
                    class_weight=None, presort=False):
    '''
    Classifier for the decision tree model
    '''
    model = tree.DecisionTreeClassifier(criterion=criterion, splitter=splitter,\
                    max_depth=max_depth, min_samples_split=min_samples_split, \
                    min_samples_leaf=min_samples_leaf, \
                    min_weight_fraction_leaf=min_weight_fraction_leaf,\
                    max_features=max_features, random_state=random_state,\
                    max_leaf_nodes=max_leaf_nodes, \
                    min_impurity_decrease=min_impurity_decrease,\
                    min_impurity_split=min_impurity_split,\
                    class_weight=class_weight, presort=presort)
    return model


def train_decision_tree(X_train, y_train, X_test, model):
    '''
    This function is used to train boosting model
    Inputs:
        X_train, y_train, X_test: sub datasets for training purpose
        model: model of classifier used in training
    Returns: model trained, list of predicted probability
    '''
    decision_tree = model.fit(X_train, y_train)
    y_predp = decision_tree.predict_proba(X_test)
    return decision_tree, y_predp


####################################################################################################
# Random Forest

def classifier_settings_rf(n_estimators, criterion="gini", \
                    max_depth=None, min_samples_split=2, min_samples_leaf=1,\
                    min_weight_fraction_leaf=0.0, max_features="auto", \
                    max_leaf_nodes=None, min_impurity_decrease=0.0, \
                    min_impurity_split=None, bootstrap=True, oob_score=False,\
                    n_jobs=2, random_state=None, verbose=0, \
                    warm_start=False, class_weight=None):
    '''
    Classifier for the random forest model
    '''
    model = RandomForestClassifier(n_estimators, criterion=criterion, \
                    max_depth=max_depth, min_samples_split=min_samples_split,\
                    min_samples_leaf=min_samples_leaf,\
                    min_weight_fraction_leaf=min_weight_fraction_leaf,\
                    max_features=max_features, max_leaf_nodes=max_leaf_nodes,\
                    min_impurity_decrease=min_impurity_decrease, \
                    min_impurity_split=min_impurity_split, \
                    bootstrap=bootstrap, oob_score=oob_score,\
                    n_jobs=n_jobs, random_state=random_state, verbose=verbose,\
                    warm_start=warm_start, class_weight=class_weight)
    return model


def train_random_forest(X_train, y_train, X_test, model):
    '''
    This function is used to train boosting model
    Inputs:
        X_train, y_train, X_test: sub datasets for training purpose
        model: model of classifier used in training
    Returns: model trained, list of predicted probability
    '''
    random_f = model.fit(X_train, y_train)
    y_predp = random_f.predict_proba(X_test)
    return random_f, y_predp


#####################################################################################################
def map_feature_importances(model, features):
    '''
    This function is used to provide feature importances of the
    festures used in the decision tree
    Inputs:
        tree: decision tree object
        features: features used in the decision tree
    Returns: feature importances dictionary
    '''
    feature_dict = {}
    importances = model.feature_importances_
    for i, val in enumerate(features):
        feature_dict[val] = importances[i]

    return feature_dict


#############################################################
# Source: Plotting Decision Trees via Graphviz, scikit-learn
# Author: scikit-learn developers
# Date: 2007 - 2018
#############################################################
def depict_decision_tree(decision_tree, features, classifier, output_filename):
    '''
    This function is used to generate the decision tree graph which
    contains the tree node and related information. The output will be
    saved as pdf file
    Inputs:
        dataframe
        features: list of feature names used in decision tree
        classifier: name of the classifier used in decision tree
        output_filename: the name of the pdf file
    '''
    dot_data = tree.export_graphviz(decision_tree, out_file="tree.dot", \
                                                   feature_names=features,\
                                                   class_names=classifier,\
                                                   filled=True, rounded=True,\
                                                   special_characters=True)
    file = open('tree.dot', 'r')
    text = file.read()
    graph = graphviz.Source(text)  
    graph.render(output_filename) 
