import pandas as pd
import numpy as np
from scipy import stats
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image,ImageSequence
from sklearn.manifold import MDS
from sklearn.metrics import euclidean_distances
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn import tree
import graphviz


def dimension_reduction(df, sample_limit, category_feature=None, n_components=3, n_jobs=2):
    '''
    This function is used to conduct multidimentional scaling
    Inputs:
        df: dataframe
        sample_limit: given restriction for the size of rows
        category_features: feature used as predefined label
        n_components: the final dimensions
    Returns: numpy array with n dinmensions and labels, index for the label
    '''
    if df.shape[0] > sample_limit:
        sub_df = df.sample(n=sample_limit).reset_index()
    else:
        sub_df = df.reset_index()

    used_columns = list(sub_df.columns)
    used_columns.remove(category_feature)

    sub_dfm = np.matrix(sub_df[used_columns])
    similarities = euclidean_distances(sub_dfm)
    mds = manifold.MDS(n_components=n_components, max_iter=3000, eps=1e-9,
                       dissimilarity='precomputed', n_jobs=1)
    pos = mds.fit(similarities).embedding_

    if category_feature:
        category_index = {}
        sub_df["label"] = 0
        for i, category in enumerate(sorted(list(sub_df[category_feature].unique()))):
            sub_df.loc[sub_df[category_feature] == category, "label"] = i
            category_index[category] = i

        new_pos = np.zeros((pos.shape[0], n_components+1))
        new_pos[:,:-1] = pos
        new_pos[:,-1] = sub_df["label"]

    return new_pos, category_index


def depict_mds_plot(pos, angle):
    '''
    This function is used to depict 3D gif plot for the MDS result with
    three dimensions
    Inputs:
        pos: the datframe with n dimensions
        angle: the angle variation of each png
    '''
    set_dict = {}
    label = pd.DataFrame(pos[:,-1]).rename(columns = {0:"label"})
    for cat_index in list(label["label"].unique()):
        set_dict[cat_index] = set(label[label["label"] == cat_index].index)

    color = ["b", "g", "r", "c", "m", "y", "k"]
    fig=plt.figure()
    ax=fig.add_subplot(111, projection='3d')
    for i in range(pos.shape[0]):
        x3=pos[i,0]
        y3=pos[i,1]
        z3=pos[i,2]
        for category_index, sets in set_dict.items():
            if i in sets:
                ax.scatter(x3,y3,z3,c=color[int(category_index)])

    for angle in range(0,360, angle):
        ax.view_init(30, angle)
        plt.savefig(r'D:\Project Data\ML project\mds_plot%d'%angle)

    seq=[]
    for i in range(0,360, angle):
        im=Image.open(r'D:\Project Data\ML project\mds_plot%d.png'%i)
        seq.append(im)
        seq[0].save(r'D:\Project Data\ML project\mds_plot.gif',save_all=True,append_images=seq[1:])


def k_mean_analysis(df, clusters, label_col, plot=False):
    '''
    This function implement k-mean clustering for given number of clusters
    Inputs:
        df: dataframe
        clusters: number of clusters
        label_col: name of the label column
    Returns: dataframe with clustered label
    '''
    df_col = list(df.columns)
    col1, col2 = df_col[0], df_col[1]

    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(df)
    df[label_col] = pd.Series(kmeans.labels_)
    groups = df.groupby(label_col)

    if len(df_col) > 2:
        print("not allow to plot in 2D")
        return  df

    if plot:
        fig, ax = plt.subplots()
        for pred_class, group in groups:
            ax.scatter(group[col1], group[col2], label=pred_class)
        ax.legend()
        plt.show()

    return df


def summarize_clusters(df, label_col):
    '''
    This function generate the summary statistics for the clusters
    Inputs:
        df: dataframe
        label_col: the columns denotes the information of the clusters
    Returns: dictionary of dataframes of descriptive statistics
    '''
    label_dict = {}
    for label in list(df[label_col].unique()):
        label_dict["label = " + str(label)] = df[df[label_col] == label].describe()

    return label_dict


def simple_tree(df, features, label_col, max_depth=1, min_samples_split=2):
    '''
    This model is used to provide simple tree analysis to find 
    distinctive features
    Inputs:
        df: dataframe
        features: features list
        label_col: column of the label
        max_depth: max_depth of decision tree
        min_samles_split: the minimun number of sample required to split a node
    Returns: a trained tree model
    '''
    DT = tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
    DT.fit(df[features], df[label_col])

    return DT


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


def cluster_combiner(df, label_col, old_labels, new_label_name):
    '''
    This function allow merging clusters to a new cluster
    Inputs:
        df: dataframe
        label_col: column denoted the label
        old_labels: a list of old labels considered as a single cluster
        new_label_name: the name for the new label created
    Return: a dataframe with new labels assigned
    '''
    for label in old_labels:
    	df.loc[df[label_col] == label, label_col] = new_label_name

    return df


def recluser(df, label_col, clusters):
    '''
    This function is used to recluster the dataframe with new clusters
    Inputs:
        df: dataframe
        label_col: column of the labels
        clusters: the number of the clusters
    '''
    if len(list(df[label_col].unique())) == clusters:
        return df
    
    df = df.drop([label_col], axis=1)
    df = k_mean_analysis(df, clusters, label_col)

    return df


def cluster_splitter(df, label_col, old_label, clusters):
    '''
    This function is used to split the cluster to several clusters
    Inputs:
        df: dataframe
        label: column denoted the label
        old_label: label used to split
        clusters: number of clusters for the new label
    Return: dataframe with new label assigned
    '''
    sub_df = df[df[label_col] == old_label]
    sub_df = sub_df.drop([label_col], axis=1)
    sub_df = k_mean_analysis(sub_df, clusters, "tmp_" + label_col)
    sub_df["tmp_" + label_col] += 10

    preserve_df = df[df[label_col] != old_label]
    preserve_df["tmp_" + label_col] = preserve_df[label_col]
    preserve_df = preserve_df.drop([label_col], axis=1)

    new_df = pd.concat([sub_df, preserve_df], join="inner")
    new_df["new_" + label_col] = new_df["tmp_" + label_col]
    new_df = new_df.drop(["tmp_" + label_col], axis=1)

    return new_df