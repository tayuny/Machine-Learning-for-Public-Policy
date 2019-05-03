import pandas as pd
import numpy as np

import train_test_split as tts
import evaluation as eva
import data_util as util
import feature_selection_missing_data as femd

import decision_tree
import logistic
import svm
import knn
import boosting
import bagging

#######################################################################################################
# Decision Tree and Random Forest
DECISION_TREE_HP = {"criterion" : ["gini", "entropy"],
                    "splitter" : ["random", "best"],
                    "max_depth" : range(10,20,1)}

# Decision Tree
'''
dt_hp = [criterion, splitter, max_depth, min_samples_split, min_samples_leaf,\
         min_weight_fraction_leaf, max_features, random_state, max_leaf_nodes, \
         min_impurity_decrease, min_impurity_split, class_weight, presort]
'''

# Random Forest
RANDOM_FOREST_HP = {"criterion" : ["gini", "entropy"],
                    "n_estimators" : [5, 10, 15],
                    "max_depth" : range(10,20,1)}
'''
rf_hp = [n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf,\
         min_weight_fraction_leaf, max_features, max_leaf_nodes,\
         min_impurity_decrease, min_impurity_split, bootstrap, oob_score,\
         n_jobs, random_state, verbose, warm_start, class_weight]
'''

# SVM
SVM_HP = {"kernel" : ["linear", "poly", "rbf", "sigmoid"],
          "C": [0.01, 0.1, 1, 10, 100]}

'''
svm_hp = [C, kernel, degree, gamma, coef0, shrinking, probability,\
          tol, cache_size, class_weight, verbose, max_iter,\
                decision_function_shape, random_state]
'''

# Logistics
LOG_HP = {"penalty" : ["l1", "l2"],
          "C": [0.01, 0.1, 1, 10, 100]}
'''
log_hp = [penalty, dual, tol, C, fit_intercept, intercept_scaling, \
          class_weight, random_state, solver, max_iter, multi_class,\
          verbose, warm_start, n_jobs]
'''

# KNN
KNN_HP = {"weights" : ["uniform", "distance"],
          "metric" : ["euclidean", "manhattan", "chebyshev", "minkowski"],
          "n_neighbors" : np.random.random_integers(0, 100, 5)}
'''
knn_hp = [n_neighbors, weights=, algorithm, leaf_size, p, metric,\
          metric_params, n_jobs]
'''

BOOSTING_HP = {"n_estimators": [50, 100, 150]}
'''
boosting_hp = [base_estimator, n_estimators, learning_rate, algorithm, random_state]
'''

BAGGING_HP = {"n_estimators" : [5, 10, 15]}
'''
bagging_hp = [base_estimator, n_estimators, max_samples, max_features,\
              bootstrap, bootstrap_features, oob_score, warm_start, n_jobs, \
              random_state, verbose]
'''
#################################################################################################

SPLIT = ["general", "KFold", "Temperal"]
NA = ["univariate", "multiple"]
FEATURE_SELECT = [None, "all_possiple", "KBest"]
METHOD = ["decision_tree", "random_forest", "svm", "logistics", "knn", "boosting", "bagging"]
EVAL_METHOD = ["accuracy", "f1", "recall", "precision", "AUC_ROC"]

#################################################################################################

def decision_tree_performance(X_train, y_train, X_test, y_test, eval_method, threshold_list):
    '''
    This function generate the performance matrix of decision tree model
    Inputs:
        X_train, y_train, X_test, y_test: Traing and testing dataframes
        eval_method: evaluation method used in the function
        threshold_list: list of the percentiles of the population for
                        extracting thresholds from the predicted probability
    Returns: performance matrix
    '''
    columns=["criterion", "splitter", "max_depth", "best_thres", "best_eval"]
    m_performance = pd.DataFrame(columns=columns)

    for crit in DECISION_TREE_HP["criterion"]:
        for split in DECISION_TREE_HP["splitter"]:
            for m_depth in DECISION_TREE_HP["max_depth"]:
                dt = decision_tree.classifier_settings_dt(\
                     criterion = crit, splitter=split, max_depth=m_depth)
                y_predp = decision_tree.train_decision_tree(X_train, \
                                               y_train, X_test, dt)[1]
                best_thres, best_eval = eva.best_threshold(y_test, \
                                        y_predp, eval_method, threshold_list)
                output = {"criterion":[crit], "splitter":[split], "max_depth":[m_depth], \
                          "best_thres":[best_thres], "best_eval":[best_eval]}
                sub_p = pd.DataFrame(data=output)
                m_performance = pd.concat([m_performance, sub_p], join="inner")
                m_performance["method"] = "decision_tree"

    return m_performance


def random_forest_performance(X_train, y_train, X_test, y_test, eval_method, threshold_list):
    '''
    This function generate the performance matrix of random forest model
    Inputs:
        X_train, y_train, X_test, y_test: Traing and testing dataframes
        eval_method: evaluation method used in the function
        threshold_list: list of the percentiles of the population for
                        extracting thresholds from the predicted probability
    Returns: performance matrix
    '''
    columns=["criterion", "n_estimators", "max_depth", "best_thres", "best_eval"]
    m_performance = pd.DataFrame(columns=columns)

    for crit in RANDOM_FOREST_HP["criterion"]:
        for n in RANDOM_FOREST_HP["n_estimators"]:
            for m_depth in RANDOM_FOREST_HP["max_depth"]:
                rf = decision_tree.classifier_settings_rf(\
                     criterion = crit, n_estimators=n, max_depth=m_depth)
                y_predp = decision_tree.train_random_forest(X_train, \
                                               y_train, X_test, rf)[1]
                best_thres, best_eval = eva.best_threshold(y_test, \
                                        y_predp, eval_method, threshold_list)
                output = {"criterion":[crit], "n_estimators":[n], "max_depth":[m_depth], \
                          "best_thres":[best_thres], "best_eval":[best_eval]}
                sub_p = pd.DataFrame(data=output)
                m_performance = pd.concat([m_performance, sub_p], join="inner")
                m_performance["method"] = "random_forset"

    return m_performance


def knn_performance(X_train, y_train, X_test, y_test, eval_method, threshold_list):
    '''
    This function generate the performance matrix of knn model
    Inputs:
        X_train, y_train, X_test, y_test: Traing and testing dataframes
        eval_method: evaluation method used in the function
        threshold_list: list of the percentiles of the population for
                        extracting thresholds from the predicted probability
    Returns: performance matrix
    '''
    columns=["weights", "metric", "n_neighbors", "best_thres", "best_eval"]
    m_performance = pd.DataFrame(columns=columns)

    for wgt in KNN_HP["weights"]:
        for met in KNN_HP["metric"]:
            for n in KNN_HP["n_neighbors"]:
                knn_m = knn.classifier_settings_knn(\
                        weights=wgt, metric=met, n_neighbors=n)
                y_predp = knn.train_knn(X_train, y_train, X_test, knn_m)[1]
                best_thres, best_eval = eva.best_threshold(y_test, \
                                        y_predp, eval_method, threshold_list)
                output = {"weight":[wgt], "metric":[met], "n_neighbors":[n], \
                          "best_thres":[best_thres], "best_eval":[best_eval]}
                sub_p = pd.DataFrame(data=output)
                m_performance = pd.concat([m_performance, sub_p], join="inner")
                m_performance["method"] = "knn"

    return m_performance


def svm_performance(X_train, y_train, X_test, y_test, eval_method, threshold_list):
    '''
    This function generate the performance matrix of svm model
    Inputs:
        X_train, y_train, X_test, y_test: Traing and testing dataframes
        eval_method: evaluation method used in the function
        threshold_list: list of the percentiles of the population for
                        extracting thresholds from the predicted probability
    Returns: performance matrix
    '''
    columns=["kernel", "C", "best_thres", "best_eval"]
    m_performance = pd.DataFrame(columns=columns)

    for ker in SVM_HP["kernel"]:
        for c in SVM_HP["C"]:
            svm_m = svm.classifier_settings_svm(kernel=ker, C=c)
            y_predp = svm.train_svm(X_train, y_train, X_test, svm_m)[1]
            best_thres, best_eval = eva.best_threshold(y_test, \
                                    y_predp, eval_method, threshold_list)
            output = {"kernel":[ker], "C":[c], "best_thres":[best_thres], \
                                                  "best_eval":[best_eval]}
            sub_p = pd.DataFrame(data=output)
            m_performance = pd.concat([m_performance, sub_p], join="inner")
            m_performance["method"] = "svm"

    return m_performance


def logistics_performance(X_train, y_train, X_test, y_test, eval_method, threshold_list):
    '''
    This function generate the performance matrix of logistics model
    Inputs:
        X_train, y_train, X_test, y_test: Traing and testing dataframes
        eval_method: evaluation method used in the function
        threshold_list: list of the percentiles of the population for
                        extracting thresholds from the predicted probability
    Returns: performance matrix
    '''
    columns=["penalty", "C", "best_thres", "best_eval"]
    m_performance = pd.DataFrame(columns=columns)

    for pan in LOG_HP["penalty"]:
        for c in LOG_HP["C"]:
            log_m = logistic.classifier_settings_log(penalty=pan, C=c)
            y_predp = logistic.train_logistics(X_train, y_train, X_test, log_m)[1]
            best_thres, best_eval = eva.best_threshold(y_test, \
                                    y_predp, eval_method, threshold_list)
            output = {"penalty":[pan], "C":[c], "best_thres":[best_thres], \
                                                  "best_eval":[best_eval]}
            sub_p = pd.DataFrame(data=output)
            m_performance = pd.concat([m_performance, sub_p], join="inner")
            m_performance["method"] = "logistics"

    return m_performance


def boosting_performance(X_train, y_train, X_test, y_test, base_method, eval_method, threshold_list):
    '''
    This function generate the performance matrix of boosting model
    Inputs:
        X_train, y_train, X_test, y_test: Traing and testing dataframes
        base_method: the method to be passed in for boosting
        eval_method: evaluation method used in the function
        threshold_list: list of the percentiles of the population for
                        extracting thresholds from the predicted probability
    Returns: performance matrix
    '''
    columns=["n_estimators", "best_thres", "best_eval"]
    m_performance = pd.DataFrame(columns=columns)

    for n in BOOSTING_HP["n_estimators"]:
        boost = boosting.classifier_settings_boosting(n_estimators=n, base_estimator=base_method)
        y_predp = boosting.train_boosting(X_train, y_train, X_test, boost)[1]
        best_thres, best_eval = eva.best_threshold(y_test, \
                                y_predp, eval_method, threshold_list)
        output = {"n_estimators":[n], "best_thres":[best_thres], "best_eval":[best_eval]}
        sub_p = pd.DataFrame(data=output)
        m_performance = pd.concat([m_performance, sub_p], join="inner")
        m_performance["method"] = "boosting"

    return m_performance


def bagging_performance(X_train, y_train, X_test, y_test, base_method, eval_method, threshold_list):
    '''
    This function generate the performance matrix of bagging model
    Inputs:
        X_train, y_train, X_test, y_test: Traing and testing dataframes
        base_method: the method to be passed in for boosting
        eval_method: evaluation method used in the function
        threshold_list: list of the percentiles of the population for
                        extracting thresholds from the predicted probability
    Returns: performance matrix
    '''
    columns=["n_estimators", "best_thres", "best_eval"]
    m_performance = pd.DataFrame(columns=columns)

    for n in BAGGING_HP["n_estimators"]:
        bag = bagging.classifier_settings_bagging(n_estimators=n, base_estimator=base_method)
        y_predp = bagging.train_bagging(X_train, y_train, X_test, bag)[1]
        best_thres, best_eval = eva.best_threshold(y_test, \
                                y_predp, eval_method, threshold_list)
        output = {"n_estimators":[n], "best_thres":[best_thres], "best_eval":[best_eval]}
        sub_p = pd.DataFrame(data=output)
        m_performance = pd.concat([m_performance, sub_p], join="inner")
        m_performance["method"] = "bagging"

    return m_performance

METHOD_DICT = {"decision_tree": decision_tree_performance, "random_forest":random_forest_performance,\
               "svm": svm_performance, "logistics": logistics_performance, "knn": knn_performance,\
               "boosting": boosting_performance, "bagging": bagging_performance}


def cross_validate_performance(df_dict, method, base_method, eval_method, threshold_list):
    '''
    This model is used to perform cross validation analysis for specific models
    Inputs:
        df_dict: dictionary of list of trainng and testing subsets
        method: model used
        base_method: models used in boosting and bagging models
        eval_method: evaluation method used in the model
        threshold_list: list of the percentiles of the population for
                        extracting thresholds from the predicted probability
    Returns: performance matrix
    '''
    perform_list= []
    for df_name, dfs in df_dict.items():
        X_train, y_train, X_test, y_test = dfs[0], dfs[1], dfs[2], dfs[3]

        if (method == "boosting") or (method == "bagging"):
            m_performance = METHOD_DICT[method](X_train, y_train, X_test, y_test, base_method, eval_method, threshold_list)
        else:
            m_performance = METHOD_DICT[method](X_train, y_train, X_test, y_test, eval_method, threshold_list)

        print("cross_validation is operated, method:{}, eval_method:{}, df_name:{}".format(method, eval_method, df_name))
        perform_list.append(m_performance["best_eval"])
    
    avg_performance = np.array([0.0]*len(perform_list[0]))
    for perform in perform_list:
        perform_a = np.array(perform)
        avg_performance += perform_a
    
    avg_performance = pd.Series(avg_performance) / len(df_dict)
    m_index = m_performance[m_performance.columns[:-3]].reset_index(drop=True)
    m_index.insert(len(m_index.columns), "avg_performance", avg_performance)

    for i, perform in enumerate(perform_list):
        perform_n = perform.reset_index(drop=True)
        m_index.insert(len(m_index.columns), "performance_temp_model" + str(i),  pd.Series(perform_n))

    return m_index
    






	
    

