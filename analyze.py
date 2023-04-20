import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import rankdata
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
import copy
import ripleyk
plt.switch_backend('agg')
matplotlib.use('TkAgg')


def ptoq(data, name=0):
    '''
    :param p_values: the p value
    :return: the p-value and FDR*
    '''
    p_values = copy.deepcopy(data)
    p_values["rank"] = rankdata(p_values[name], method='max')
    n = p_values.shape[0]
    p_values["q"] = p_values[name]*n / p_values["rank"]
    tmp = p_values["q"].values
    for i in range(n):
        tmp[i] = min(tmp[i], 1)
    p_values["q"] = tmp

    p_values = p_values.sort_values(by="rank")
    tmp = p_values["q"].values

    for i in range(2, n+1):
        tmp[n-i] = min(tmp[n-i+1], tmp[n-i])
    p_values["q"] = tmp

    p_values.columns.values[0] = "p"
    del p_values["rank"]
    return p_values

def diffCategory(classify, data, j, k):
    '''
    :param classify: first column: (spot name)      second column: (category)
    :param cnv: the cnv profile, shape: (gene x spot)
    :param j: category j
    :param k: category k
    :return: the Q-value profile about the significant difference of every gene
    '''
    cnv = copy.deepcopy(data) - 1
    col_classify = [c for c in classify]

    jk_matrix = []
    tmp = classify.loc[classify[col_classify[1]] == j, col_classify[0]]  # choose spots belong to j
    cnvj = cnv[tmp]
    tmp = classify.loc[classify[col_classify[1]] == k, col_classify[0]]  # choose spots belong to j
    cnvk = cnv[tmp]

    for i in cnv.index.values:
        fenbu_matrix = np.zeros((2, 3))
        # jth tmp fenbu
        tmp_cnv = cnvj.loc[i]
        fenbu_matrix[0][0] = np.sum(tmp_cnv == -1)
        fenbu_matrix[0][1] = np.sum(tmp_cnv == 0)
        fenbu_matrix[0][2] = np.sum(tmp_cnv == 1)
        # kth tmp fenbu
        tmp_cnv = cnvk.loc[i]
        fenbu_matrix[1][0] = np.sum(tmp_cnv == -1)
        fenbu_matrix[1][1] = np.sum(tmp_cnv == 0)
        fenbu_matrix[1][2] = np.sum(tmp_cnv == 1)

        flag = 1
        if fenbu_matrix[:, 0].sum() == 0:
            fenbu_matrix = fenbu_matrix[:, 1:]
            flag = 0
        else:
            if fenbu_matrix[:, 1].sum() == 0:
                fenbu_matrix = fenbu_matrix[:, [0, 2]]
                flag = 0
            else:
                if fenbu_matrix[:, 2].sum() == 0:
                    fenbu_matrix = fenbu_matrix[:, :2]
                    flag = 0

        if flag == 0:
            res = stats.fisher_exact(fenbu_matrix)
        else:
            res = stats.chi2_contingency(observed=fenbu_matrix)
        jk_matrix.append(res[1])

    jk_matrix = pd.DataFrame(jk_matrix, index=cnv.index.values)
    jk_matrix = ptoq(jk_matrix)
    jk_matrix = jk_matrix.loc[cnv.index.values, :]
    return jk_matrix

def diffGene(all_mixMatrix, category_number, gene_id):
    num = 0
    matrix = np.zeros(category_number, category_number)
    for j in range(category_number):
        for k in range(j, category_number):
            matrix[j][k] = all_mixMatrix[num].loc[gene_id, "q"]
            matrix[k][j] = matrix[j][k]
            num = num + 1
    return matrix

def all_diffCate(classify, cnv, number):
    all_res = []
    for i in range(number):
        res = []
        for j in range(i, number):
            if len(res) == 0:
                res = diffCategory(classify, cnv, i, j)
                res.columns = ["p" + str(i) + str(j), "q" + str(i) + str(j)]
            else:
                tmp = diffCategory(classify, cnv, i, j)
                tmp.columns = ["p" + str(i) + str(j), "q" + str(i) + str(j)]
                res = res.join(tmp, how='inner')
        all_res.append(res)

    return all_res

def drawQ(data, k):

    data = data.T
    data = -np.log10(data)
    print(data)

    sns.heatmap(data, yticklabels=True, xticklabels=400, cmap=plt.get_cmap('YlGnBu'), cbar_kws={'label': '$-log_{10}Q.value$',
                          "pad": 0.02,
                          })
    plt.title("Q-value Profile compared the category " + str(k) + " with other categories")
    plt.subplots_adjust(left=0.15, bottom=0.4, right=0.95, top=0.9,
                        wspace=0.1, hspace=0.1)


def DG(res, control, p=1e-5, q=1e-5):
    '''
    :param res: the Q-value profile about the significant difference of every gene compared one category to others
    :param control: control group id
    :return:
    '''
    index = np.array(["p"+str(control), "q"+str(control)])
    tmp = res[index]
    tmp = tmp.loc[res[index[0]] < p]
    tmp = tmp.loc[res[index[1]] < q]
    return tmp

def idtoName(data, refer):
    '''
    :param data: gene_id x spot
    :param refer: ["id", "chr", "start", "end", "name"]
    :return:
    '''
    refer.index = refer["id"]
    data = data.join(refer, how='inner')
    data.index = data["name"].values
    return data

def mutualgene(data, alpha=0.65):
    res = pd.DataFrame(index=["gene1", "gene2", "00", "10", "01", "11", "p-value"])

    x = data.eq(0).sum(axis=1)
    data = data.loc[x < data.shape[1] * alpha, :]

    n = data.shape[0]
    data = data.T
    col = data.columns.values
    k = 0

    profile = np.ones((n, n)) - 2
    for i in range(n):
        for j in range(i+1, n):
            tmp = [col[i], col[j]]
            tmp_data = data[[col[i], col[j]]]

            oo = tmp_data.loc[(tmp_data[col[i]] == -1) & (tmp_data[col[j]] == -1), :]
            tmp.append(oo.shape[0])
            oo = tmp_data.loc[(tmp_data[col[i]] == -1) & (tmp_data[col[j]] == 1), :]
            tmp.append(oo.shape[0])
            oo = tmp_data.loc[(tmp_data[col[i]] == 1) & (tmp_data[col[j]] == -1), :]
            tmp.append(oo.shape[0])
            oo = tmp_data.loc[(tmp_data[col[i]] == 1) & (tmp_data[col[j]] == 1), :]
            tmp.append(oo.shape[0])
            if (tmp[2] == 0 and tmp[3] == 0) or (tmp[2] == 0 and tmp[4] == 0) or (tmp[2] == 0 and tmp[5] == 0) or \
                    (tmp[3] == 0 and tmp[4] == 0) or (tmp[3] == 0 and tmp[5] == 0) or (tmp[4] == 0 and tmp[5] == 0):
                continue

            tmp_data = np.array(tmp[2:]).reshape(2, 2)

            fisher = stats.fisher_exact(tmp_data, 'less')
            tmp.append(fisher[1])

            res[k] = tmp
            k = k + 1
            profile[i][j] = fisher[1]
            profile[j][i] = fisher[1]
    profile = pd.DataFrame(profile, index=col, columns=col)
    profile = profile.loc[profile.sum(axis=1) != -n, :]
    col2 = profile.index.values
    profile = profile[col2]

    return res, profile

def mutualstate(data, alpha=0.5):
    data = data - 1
    res = pd.DataFrame(index=["s1", "s2", "11", "10", "01", "00", "occ_p-value", "mut_p-value"])

    x = data.eq(0).sum(axis=1)
    data = data.loc[x < data.shape[1] * alpha, :]
    print(data)
    n = data.shape[0]
    m = data.shape[1]
    data = data.T
    col = data.columns.values
    k = 0

    for i in range(n):
        print(i)
        for j in range(i+1, n):
            flagi = [-1, 1]
            flagj = [-1, 1]
            tmp_data = data[[col[i], col[j]]]
            for fi in flagi:
                for fj in flagj:
                    if fi == -1:
                        tmp = [col[i] + '-del']
                    else:
                        tmp = [col[i] + '-amp']
                    if fj == -1:
                        tmp.append(col[j] + '-del')
                    else:
                        tmp.append(col[j] + '-amp')

                    oo = tmp_data.loc[(tmp_data[col[i]] == fi) & (tmp_data[col[j]] == fj), :]
                    tmp.append(oo.shape[0])
                    oo = tmp_data.loc[(tmp_data[col[i]] == fi) & (tmp_data[col[j]] != fj), :]
                    tmp.append(oo.shape[0])
                    oo = tmp_data.loc[(tmp_data[col[i]] != fi) & (tmp_data[col[j]] == fj), :]
                    tmp.append(oo.shape[0])
                    oo = tmp_data.loc[(tmp_data[col[i]] != fi) & (tmp_data[col[j]] != fj), :]
                    tmp.append(oo.shape[0])

                    tmp_data2 = np.array(tmp[2:]).reshape(2, 2)
                    sum1 = tmp_data2.sum(axis=1)
                    sum0 = tmp_data2.sum(axis=0)
                    if 0 in sum0 or 0 in sum1:
                        continue

                    fisher = stats.fisher_exact(tmp_data2, 'greater')
                    tmp.append(fisher[1])

                    fisher = stats.fisher_exact(tmp_data2, 'less')
                    tmp.append(fisher[1])

                    res[k] = tmp
                    k = k + 1
    res = res.T
    occ_q = ptoq(res, "occ_p-value")
    res["occ_q-value"] = occ_q.loc[res.index, "q"]
    mut_q = ptoq(res, "mut_p-value")
    res["mut_q-value"] = mut_q.loc[res.index, "q"]

    gene = list(set(list(set(res["s1"].values)) + list(set(res["s2"].values))))
    n = len(gene)
    profile1 = np.zeros((n, n))
    profile1 = pd.DataFrame(profile1, index=gene, columns=gene)
    profile2 = np.zeros((n, n))
    profile2 = pd.DataFrame(profile2, index=gene, columns=gene)

    for i in res.index:
        s1 = res.loc[i, ["s1"]]
        s2 = res.loc[i, ["s2"]]
        profile1.loc[s1, s2] = res.loc[i, ["occ_p-value"]].values
        profile1.loc[s2, s1] = res.loc[i, ["occ_p-value"]].values
        profile2.loc[s1, s2] = res.loc[i, ["mut_p-value"]].values
        profile2.loc[s2, s1] = res.loc[i, ["mut_p-value"]].values

    return res, profile1, profile2


def moniK(location_number, number):
    N = range(location_number)
    a = np.random.choice(N, size=number, replace=False)
    return a

def pattern(data, k):
    n = 1
    total_mean = np.mean(data)
    total_std = np.std(data)
    z = (k - total_mean) / (total_std / np.sqrt(n))
    P = stats.norm.sf(abs(z))
    if k > total_mean:
        flag = "aggregation"
    else:
        flag = "dispersed"
    return P, flag, total_mean, total_std

def inferPattern(CNV, rrange, path):
    '''
    :param CNV: gene x spot
    :param r:
    :return:
    '''
    number = np.shape(CNV)[1]
    x = []
    y = []
    for i in CNV:
        tmp = i.split('x')
        x.append(tmp[0])
        y.append(tmp[1])
    CNV = CNV.T
    x = np.array(x)
    y = np.array(y)
    for gene in CNV:
        res = pd.DataFrame(index=["p", "flag", "mean", "std", "k"])
        tmp = CNV[gene].values
        xs = x[tmp != 1]
        ys = y[tmp != 1]
        point_number = len(xs)
        caiyang = []
        for j in range(1000):
            index = moniK(number, point_number)
            caiyang.append(index)
        for r in rrange:
            k = ripleyk.calculate_ripley(r, [number, number], d1=xs, d2=ys, sample_shape='rectangle', boundary_correct=False, CSR_Normalise=False)
            gene_k = k / number
            moni_k = []
            for j in range(1000):
                index = caiyang[j]
                xs = x[index]
                ys = y[index]
                k = ripleyk.calculate_ripley(r, [number, number], d1=xs, d2=ys, sample_shape='rectangle',
                                             boundary_correct=False, CSR_Normalise=False)
                k = k / number
                moni_k.append(k)
            P, flag, mean, std = pattern(moni_k, gene_k)
            res[r] = [P, flag, mean, std, gene_k]
        res.to_csv(path + "/" + "gene.csv", sep=',')

