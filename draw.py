import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import copy

plt.switch_backend('agg')

def zplot(location, predict_label):
    k = len(set(predict_label["cluster"].values))
    data = pd.DataFrame(location, columns=["x", "y"])
    data["cluster"] = predict_label["cluster"]
    color = sns.color_palette("hls", k)
    ax = sns.scatterplot(data=data, x="x", y="y", hue="cluster", palette=color, zorder=2)

    plt.legend(title="cluster", fancybox=False, loc=4)

def drawprofile(data, refer, predict_label):
    classify = copy.deepcopy(predict_label)
    classify.index = classify["location"]
    classify = classify["cluster"]
    k = len(set(classify.values))
    refer.index = refer["name"]
    refer = refer[~refer.index.duplicated()]
    data = data[~data.index.duplicated()]
    refer = refer.join(data)
    refer = refer["chr"]
    data = data.T

    color = sns.color_palette("hls", k)
    lut = dict(zip(classify.unique(), color))
    row_colors = classify.map(lut)

    color = plt.get_cmap('tab20')
    lut = dict(zip(refer.unique(), color.colors))
    col_colors = refer.map(lut)
    ax = sns.clustermap(data, row_colors=row_colors, col_colors=col_colors, row_cluster=False, col_cluster=False,
                   cmap=ListedColormap(['steelblue', 'lightgrey', 'darkorange']),
                   cbar_kws={'label': 'CNV',
                           'orientation': 'vertical',
                           "ticks":np.arange(-1, 1.2, 1),
                           "format":"%d",
                           "pad":0.05,
                             },
                   cbar_pos=(0.09, 0.04, 0.05, 0.1))

def mutAndOcc(occ, mut):
    occ = occ.astype(float)
    mut = mut.astype(float)
    occ = -np.log2(occ + 1)
    mut = -np.log2(mut + 1)
    n = occ.index.values
    data = copy.deepcopy(occ)
    for i in n:
        for j in n:
            if occ.loc[i, j] > mut.loc[i, j]:
                data.loc[i, j] = 1 + occ.loc[i, j]
            else:
                data.loc[i, j] = -1 - mut.loc[i, j]

    print(data)
    mask0 = data == 0
    mask1 = data == 1
    mask2 = data == -1
    mask3 = np.triu(np.ones_like(data, dtype=bool))
    mask = mask0 | mask1
    mask = mask | mask2
    mask = mask | mask3

    cmap = sns.diverging_palette(200, 20, sep=20, as_cmap=True)
    ax = sns.heatmap(data, vmin=None, vmax=None, mask=mask, cmap=cmap, linewidths=0.01, cbar_kws={
        "ticks": [], "pad": 0.0})
    plt.subplots_adjust(left=0.15, bottom=0.2, right=0.95, top=0.92, wspace=0.1, hspace=1)
