from hmmlearn import hmm
import scipy.signal as signal
import copy
import community as community_louvain
import networkx as nx
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import novosparc
import random

np.set_printoptions(threshold=np.inf)

'''data process'''
def getLocation(location):
    """extract the location from the string

    :param location: the location of every spot, eg. 5x6
    :return: a array including tow columns: x  y
    """
    x = []
    y = []
    for i in range(len(location)):
        res = location[i].split('x')
        x.append(int(res[0]))
        y.append(int(res[1]))
    x = np.array(x)
    y = np.array(y)
    location = np.c_[x, y]

    return location

def loadData(path):
    """load the gene expression data

    the data: the first row is the spot_id, it should be the coordinate of the spot  (eg. 5x6)
              the first column is the gene_name
              others are expression value

    :param sample: the name of the sample
    :param path: the direction of the gene expression file
    :return: feature: spot x gene
    """

    file = open(path)
    i = 0
    name = []
    location = []
    feature = []
    while 1:
        line = file.readline()
        line = line.replace('\n', '')
        if not line:
            break
        res = line.split('\t')
        if i == 0:
            name = res[1:]
            i = i + 1
            continue
        else:
            location.append(res[0])
            feature.append(res[1:])

        i = i + 1
    feature = np.array(feature).astype(float)
    print("the gene expression info:  spots: ", np.shape(feature)[0], ", genes: ", np.shape(feature)[1])

    name = np.array(name)
    location = np.array(location)

    location = getLocation(location)
    file.close()

    return location, name, feature

def OT(data, k):
    '''
    :param data: gene x spot
    :return: data: gene x spot (with location name)
    '''
    random.seed(1)
    dataset = ad.AnnData(data.T)

    num_cells, num_genes = dataset.shape

    sc.pp.normalize_per_cell(dataset)
    sc.pp.log1p(dataset)
    sc.pp.pca(dataset)

    dge_rep = pd.DataFrame(dataset.obsm['X_pca'])

    num_locations = num_cells
    locations_circle = novosparc.gm.construct_target_grid(num_locations=num_locations)

    tissue = novosparc.cm.Tissue(dataset=dataset, locations=locations_circle)

    num_neighbors_s = num_neighbors_t = k
    tissue.setup_smooth_costs(dge_rep=dge_rep, num_neighbors_s=num_neighbors_s, num_neighbors_t=num_neighbors_t)

    tissue.reconstruct(alpha_linear=0, epsilon=5e-3)

    gw = tissue.gw
    ngw = (gw.T / gw.sum(1)).T

    m = np.shape(ngw)[1]
    flag = np.zeros(m)
    path = []
    for i in range(np.shape(ngw)[0]):
        index = ngw[i].argmax()
        while flag[index] == 1:
            ngw[i, index] = -1
            index = ngw[i].argmax()
        flag[index] = 1
        path.append(index)

    result = ngw.argmax(axis=1)
    location = pd.DataFrame(locations_circle.astype(int))
    location.columns = ['x', 'y']
    location = location.T

    new_name = []
    for i in range(len(result)):
        new_name.append(str(location.loc['x', result[i]]) + 'x' + str(location.loc['y', result[i]]))

    data.columns = new_name

    return data

def normalise(data, alpha=0.05, sigma=10):
    '''
    :param data: gene x spot
    :param cutoff: average cuttoff
    :param alpha: the percent of the cell whose number of gene express
    :return:
    '''
    col = [c for c in data]
    number = len(col)
    data["sum"] = data[col[:number]].eq(0.0).sum(axis=1) / number
    data = data[data["sum"] < 1 - alpha]
    data = data[col]
    data = np.log2(data + 1)
    data["std"] = data[col].std(axis=1)
    data = data[data["std"] > sigma]
    data = data[col]

    return data

def Hmmgene(data, refer_data, sigma=0.05):
    '''
    :param data: gene x spot
    :param refer_data: gene x spot
    :return:
    '''

    col = [c for c in data]
    number = len(col)
    data["sum"] = data[col[:number]].eq(0.0).sum(axis=1) / number

    data = data[data["sum"] < 1 - sigma]
    data = data[col]

    c_data = [c for c in data]
    c_refer = [c for c in refer_data]

    data = data.join(refer_data, how='inner')
    refer_data = data[c_refer]
    data = data[c_data]

    for i in data:
        data[i] = data[i]/data[i].sum() * 1e6
    for i in refer_data:
        refer_data[i] = refer_data[i]/refer_data[i].sum() * 1e6


    data = np.log2(data + 1)
    refer_data = np.log2(refer_data + 1)

    data["med"] = data.median(axis=1)
    for i in data:
        data[i] = data[i] - data["med"]
    data = data[c_data]

    return data

def afterK_G(model_data, location, alpha=0.5, distance=1, k=3):
    n = np.shape(location)[0]
    A = np.zeros((n, n))
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i][j] = np.linalg.norm(model_data[i] - model_data[j])
        xxx = copy.deepcopy(dist[i])
        xxx = np.sort(xxx)
        threshold = xxx[k]
        for j in range(n):
            if dist[i][j] <= threshold:
                A[i][j] = 1 - alpha
                A[j][i] = 1 - alpha

    for i in range(n):
        for j in range(i+1, n):
            if abs(location[i][0] - location[j][0]) + abs(location[i][1] - location[j][1]) <= distance:
                A[i][j] += alpha
                A[j][i] += alpha

    return A

'''reference process'''

def getpq(path):
    """to get the p and q high-level point information of every chromosome

    the data includes 5 columns: chromosome_id, start_location, end_location, p/q, reference

    :param path: the path of the data file
    :return: p/q information data: chromosome_id, start_location, end_location, p/q
    """
    pqdata = np.loadtxt(path, dtype=str, delimiter='\t')
    res = []
    start = pqdata[0]
    i = 1
    j = np.shape(pqdata)[0]
    while i < j:
        if len(start) != 0 and (start[0] != pqdata[i][0] or start[3][0] != pqdata[i][3][0]):
            start[2] = pqdata[i-1][2]
            start[3] = start[3][0]
            res.append(start)
            start = []
            while i < j and (len(pqdata[i][0]) > 6 or pqdata[i][0] == 'chrM'):
                i = i + 1
            if i < j:
                start = pqdata[i]
        i = i + 1
    res = np.array(res)[:, :4]

    return res

def mixrefer(pqdata, referdata):
    """to mix the p/q information with every genes information together

    :param pqdata: p/q information file
    :param referdata: the file after-processed based on the hg38.gtf, which includes 5 columns:
                      chromosome_id, gene_start, gene_end, +/-, gene_name
    :return:the mixed reference file including chromosome_id, gene_start, gene_end, +/-, gene_name, p/q
    """
    i = 0
    n = np.shape(referdata)[0]
    res = []
    while i < n:
        loc = np.argwhere(pqdata[:, 0] == referdata[i][1])
        loc = loc[0][0]

        if referdata[i][2].astype(int) <= pqdata[loc][2].astype(int):
            res.append('p')
        else:
            res.append('q')
        i = i + 1
    res = np.array(res)
    res = np.c_[referdata, res]

    return res

def cutpd(gene_exp):
    '''
    :param gene_exp: 0: chr_id, 1: start, 2: end, 3: gene_name, 4: p/q
    :return:
    '''
    n = gene_exp.shape[0]
    index = gene_exp.index.values
    i = 0
    k = 0
    pqpoint = []
    pqinfo = []
    while i < n-1:
        k += 1
        if (gene_exp.loc[index[i], 0] != gene_exp.loc[index[i+1], 0]) or (gene_exp.loc[index[i], 4] != gene_exp.loc[index[i+1], 4]):
            pqpoint.append(k)
            k = 0
            pqinfo.append([gene_exp.loc[index[i], 0], gene_exp.loc[index[i], 4]])
        i += 1
    pqpoint.append(k + 1)
    pqinfo.append([gene_exp.loc[index[i], 0], gene_exp.loc[index[i], 4]])

    return np.array(pqpoint), np.array(pqinfo)

'''VGAE'''
#GraphConvolution
class GraphConv(nn.Module):
    def __init__(self, a_matrix, dim_in, dim_out):
        super(GraphConv, self).__init__()
        self.fgcn = nn.Linear(dim_in, dim_out)
        self.Agcn = a_matrix

    def forward(self, data):
        x = data
        x = self.fgcn(x)
        x = torch.mm(self.Agcn, x)
        return x

#Variational Graph Auto-encoder
class VGAE(nn.Module):
    def __init__(self, a_matrix, dim_in, middle_out, output):
        super(VGAE, self).__init__()
        self.gcn = GraphConv(a_matrix, dim_in, middle_out)
        # self.fmean = nn.Linear(middle_out, output)
        # self.fstd = nn.Linear(middle_out, output)
        #reconstruct A
        self.delayerA1 = nn.Linear(output, output // 2)
        self.delayerA2 = nn.Linear(output // 2, a_matrix.size(0))

        #reconstruct X
        self.delayerX1 = nn.Linear(output, output // 2)
        self.delayerX2 = nn.Linear(output // 2, dim_in)

        self.Avgae = a_matrix
        self.out = output

        self.f_mean = GraphConv(a_matrix, middle_out, output)
        self.f_logstddev = GraphConv(a_matrix, middle_out, output)

    def encode(self, data):
        x = data
        layer1 = self.gcn(x)
        layer1 = torch.relu(layer1)
        self.mean = self.f_mean(layer1)
        self.logstddev = self.f_logstddev(layer1)
        noise = torch.randn(data.size(0), self.out)
        z = noise*torch.exp(self.logstddev) + self.mean

        return z


    def forward(self, data):
        z = self.encode(data)
        # print("z ", z[:5])
        '''reconstruct A'''
        dec_A = torch.relu(self.delayerA1(z))
        dec_A = torch.sigmoid(self.delayerA2(dec_A))
        '''reconstruct X'''
        dec_X = torch.relu(self.delayerX1(z))
        dec_X = torch.relu(self.delayerX2(dec_X))
        # cla_result = F.softmax(self.classify2(F.softmax(self.classify(z))))
        # cla_result = F.softmax(self.classify(z), dim=1)
        return dec_A, dec_X, z

#Normalize A
def norMatrix(a):
    """to normalise the adjacency matrix
    :param a:
    :return:
    """
    a = a + torch.eye(a.size(0))
    d = a.sum(1)
    D = torch.diag(torch.pow(d, -0.5))

    return D.mm(a).mm(D)

def loss_function(dec_a, a,dec_x, x, mu=0, logstddev=0, w=[0.5, 0.5, 1e-5]):
    """

    :param dec_a: generated adjacency matrix
    :param a: original adjacency matrix
    :param mu: latent mean of z
    :param logstddev: latent log variance of z
    :param classify: the semi-classify softmax result of the spots
    :param label: the noraml label
    :return:
    """
    BCE_loss = nn.BCELoss(reduction='mean')
    l_A = BCE_loss(dec_a.view(-1), a.view(-1))
    MSE_loss = nn.MSELoss(reduction='mean')
    l_X = MSE_loss(dec_x, x)
    KL = -0.5 * torch.sum(1+2*logstddev-torch.exp(2*logstddev)-mu**2)

    loss = w[0]*l_A + w[1]*l_X + w[2]*KL

    return loss, l_A, l_X, KL

def rightresult(mean, predict):
    new_predict = copy.deepcopy(predict)
    value0 = mean[0][0]
    value1 = mean[1][0]
    value2 = mean[2][0]
    value = np.array([value0, value1, value2])

    value = value.argsort()
    new_predict[predict == value[0]] = new_predict[predict == value[0]] - value[0] - 1
    new_predict[predict == value[1]] = new_predict[predict == value[1]] - value[1] + 0
    new_predict[predict == value[2]] = new_predict[predict == value[2]] - value[2] + 1
    return new_predict

def HMMinferCNV(geneexp, pqpoint, k1, k2):
    '''
    :param geneexp: spot x gene
    :param pqpoint: record the number of genes of the segment
    :return:
    '''
    sampleNumber = np.shape(geneexp)[0]
    states = ["deletion", "neutral", "amplification"]  ##隐藏状态
    n_states = len(states)  # 隐藏状态长度
    # t = 1e-5

    model = hmm.GaussianHMM(n_components=n_states, n_iter=30, tol=0.05, covariance_type="diag", verbose=False,
                             params='stc', init_params='st')

    model.means_ = np.array([[k1], [0.0], [k2]])
    model.covars_ = np.array([[0.1], [0.1], [0.1]])

    train_data = copy.deepcopy(geneexp.reshape((-1, 1), order='C'))
    train_len = np.tile(pqpoint, (sampleNumber, 1))
    train_len = train_len.flatten()

    model.fit(train_data, train_len)

    predict_result = model.decode(train_data, train_len)[1]
    predict_result = predict_result.reshape((sampleNumber, -1), order='C')

    return predict_result

def med_filter(data, windows=21):
    '''
    :param data: n samples x m features
    :param windows:
    :return:
    '''
    n = np.shape(data)[0]
    m = np.shape(data)[1]
    if windows > m:
        if m % 2 == 1:
            windows = m
        else:
            windows = m - 1
    result = []
    for i in range(n):
        result.append(signal.medfilt(data[i], windows))
    return np.array(result)

def MultiHMM(geneexp, pqpoint, k1, k2, scale=3, windows=11, delta=20):
    '''
    :param geneexp: spot x gene
    :param pqpoint: record the number of genes of the chains
    :param scale:
    :param windows:
    :return:
    '''

    n = np.shape(geneexp)[0]
    m = np.shape(geneexp)[1]
    num = len(pqpoint)# number of chains
    newdata = []
    for i in range(scale):
        data = np.empty(shape=(n, 0))
        last = 0
        for j in range(num):
            data = np.append(data, med_filter(geneexp[:, last:last + pqpoint[j]], windows), axis=1)
            last += pqpoint[j]
        newdata.append(data)
        geneexp = data
        windows += delta
    newdata = np.array(newdata)
    newdata = np.median(newdata, axis=0)
    result = HMMinferCNV(newdata, pqpoint, k1, k2)
    result = np.array(result)

    return result

def createG(Adj, data):
    n = np.shape(Adj)[0]
    nodes = [str(i) for i in range(n)]
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if Adj[i][j] == 0:
                continue
            weight = np.linalg.norm(data[i] - data[j])
            edges.append((str(i), str(j), weight))
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from(edges)
    return G

def classifyLouvin(G, resolu=1.0):
    partition = community_louvain.best_partition(G, resolution=resolu, randomize=True)
    predict = list(partition.values())

    return np.array(predict)
