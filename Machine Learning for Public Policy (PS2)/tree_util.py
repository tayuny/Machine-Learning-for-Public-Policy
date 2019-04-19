import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
import graphviz
import data_util

# Build classifier
def train_decision_tree(df, keep_columns, classifier, test_size, seed=1):
    '''
    This function is used to build the classifier via training with decision
    tree method.
    Inputs: 
        dataframe
        keep_columns: the columns used as features in the model
        classifier: column name of the column use as classifier
        test_size: proportion of the whole dataset used for test
        seed: seed for random state
    Returns: 
        decision_tree object
        X_test: dataframe of features used for testing
        y_test: series of classifiers used for testing
    '''
    train_columns = list(df[keep_columns].columns)
    X_train, X_test, y_train, y_test = train_test_split(df[train_columns], \
                                                        df[classifier], \
                                                        test_size=test_size,\
                                                        random_state=seed)
    model = tree.DecisionTreeClassifier()
    decision_tree = model.fit(X_train, y_train)
    
    return decision_tree, X_test, y_test


def map_feature_importances(tree, features):
    '''
    This function is used to provide feature importances of the
    festures used in the decision tree
    Inputs:
        tree: decision tree object
        features: features used in the decision tree
    Returns: feature importances dictionary
    '''
    feature_dict = {}
    importances = tree.feature_importances_
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


# Evaluate classifier
def accuracy_calculation(gn, g0n, g1n, evaluated):
    '''
    This function is used to calculate the accuracy of the prediction
    Inputs: 
        gn: number of instances in the testing data
        g0n: number of 0 in the testing data
        g1n: number of 1 in the testing data
        evaluated: numpy array with testing data and predicted 
                   classifications 
    Returns: accuracy for all cases, accuracy for case 0, accuracy for case 1
    '''
    count, count_g0, count_g1 = 0, 0, 0
    for i,j in list(evaluated):
        if i == 0:
            if i == j:
                count += 1
                count_g0 += 1
        else:
            if i == j:
                count += 1
                count_g1 += 1
    
    return count/gn, count_g0/g0n, count_g1/g1n

def best_prediction_threshold(tree, X_test, y_test, lower_bound, gap):
    '''
    This function is used to find the best threshold to classified the
    predicted probability, using best accuracy or f1_score.
    Inputs:
        tree: decision tree
        X_test: dataframe of features used for testing
        y_test: series of classifiers used for testing
    Returns:
        y_pred: the predictions of classifier using training data
        best_accuracy: accuracy of all cases
        current_acc0: accuracy of classifier which is labelled 0 in 
                      testing data
        current_acc1: accuracy of classifier which is labelled 1 in 
                      testing data
    '''
    gn, g0n, g1n = len(y_test), list(y_test).count(0), list(y_test).count(1)
    y_predp = tree.predict_proba(X_test)[:,1]
    best_accuracy = 0

    for threshold in np.arange(lower_bound, 1, gap):
        y_predp = pd.Series(y_predp > threshold)
        y_pred = pd.Series([0] * len(y_predp))
        y_pred.loc[y_predp] = 1
        evaluated = list(zip(y_pred, y_test))

        acc_all, acc_0, acc_1 = accuracy_calculation(gn, g0n, g1n, evaluated)
        if acc_all > best_accuracy:
            best_accuracy = acc_all
            current_acc0, current_acc1 = acc_0, acc_1
            best_threshold = threshold

    return best_threshold, best_accuracy, current_acc0, current_acc1


def tree_prediction_evaluation(tree, X_test, y_test, lower_bound, gap):
    '''
    This function is used to evaluate the prediction made by the decision tree
    Inputs:
        tree: decision tree
        X_test: dataframe of features used for testing
        y_test: series of classifiers used for testing
    Returns:
        y_pred: the predictions of classifier using training data
        best_accuracy: accuracy of all cases
        current_acc0: accuracy of classifier which is labelled 0 in 
                      testing data
        current_acc1: accuracy of classifier which is labelled 1 in 
                      testing data
    '''
    best_threshold, best_accuracy, current_acc0, current_acc1 =\
    best_prediction_threshold(tree, X_test, y_test, lower_bound, gap)
    
    return best_threshold, best_accuracy, current_acc0, current_acc1


def select_best_variables(df, features, classifier, test_size, lower_bound,\
                          gap, seed=1, target_accuracy_index=1):
    '''
    This function is used to extract the best combination of the features 
    which return the highest accuracy of predictions
    Inputs: dataframe, features, classifier, test_size, seed
            target_accuracy_index: 
                1: accuracy of all cases
                2: accuracy of classifier which is labelled 0 in 
                   testing data
                3. accuracy of classifier which is labelled 1 in 
                   testing data
    Returns: best combination of features, best accuracy
    '''
    var_combinations = data_util.get_var_combinations(features)
    var_combinations.remove([])
    best_accuracy = 0

    for combination in var_combinations:
        tree, X_test, y_test = train_decision_tree(df, combination, 
                                                   classifier, test_size,\
                                                   seed=seed)
        accuracy = tree_prediction_evaluation(tree, X_test, y_test,\
                           lower_bound, gap)[target_accuracy_index]
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_features = combination

    return best_features, best_accuracy


def reset_balance(df, classifier, sample_times, cls0_size=1, cls1_size=1):
    '''
    This function is used to sample the majority in the unbalance classification
    Inputs:
        df: dataframe
        classifier: name of classifier in the decision tree
        sample_times: time used for resampling
        cls0_size: relative size of classification 0
        cls1_size: relative size of classification 1
    Returns: list of balanced dataframe
    '''
    df_list = []
    
    if cls0_size >= cls1_size:
        base_size = df[df[classifier] == 1].shape[0]
        sample_size = base_size * cls0_size
        base_df = df[df[classifier] == 1]
        sample_df = df[df[classifier] == 0]

    else:
        base_size = df[df[classifier] == 0].shape[0]
        sample_size = base_size * cls1_size
        base_df = df[df[classifier] == 0]
        sample_df = df[df[classifier] == 1]

    for i in range(sample_times):
        sub_sample_df = sample_df.sample(sample_size, axis=0)
        whole_sub_df = pd.concat([sub_sample_df, base_df], join="inner")
        df_list.append(whole_sub_df)

    return df_list

