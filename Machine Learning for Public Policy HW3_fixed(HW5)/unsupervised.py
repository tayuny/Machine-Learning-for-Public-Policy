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

    sub_dfm = np.matrix(sub_df)
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