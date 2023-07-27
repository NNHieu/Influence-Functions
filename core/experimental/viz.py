import matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

def plot_results(sel_ranges, precs, error_label_ranking, right_traces):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].plot(sel_ranges, precs)
    axes[1].bar(np.arange(len(error_label_ranking)), error_label_ranking)
    axes[2].bar(np.arange(len(right_traces)), right_traces)
    return fig, axes

def tsne_dim_reduction(features, pca_n_components, tsne_n_components):
    pca = decomposition.PCA(n_components=pca_n_components)
    all_X_reduced = pca.fit_transform(features)
    tsne = TSNE(n_components=tsne_n_components, init='random', learning_rate='auto')
    all_X_reduced_tsne = tsne.fit_transform(all_X_reduced)
    df = pd.DataFrame(all_X_reduced_tsne[:, :2], columns=['comp_1', 'comp_2'])
    return df