import model as m
import numpy as np
import pandas as pd
import torch
import copy
import os
import matplotlib.pyplot as plt
import matplotlib
plt.switch_backend('agg')
matplotlib.use('TkAgg')

class CVAM():
    def __init__(self, EXPRESSION_PATH, PQ_PATH, REFERENCE_PATH, RES_PATH, ST_flag=True, alpha=0.05, sigma=4,
          OT_number=5, percent=0.5):
        self.RES_PATH = RES_PATH
        self.percent = percent
        gene_exp = pd.read_csv(EXPRESSION_PATH, delimiter='\t', index_col=0)  # gene_id x sample
        self.raw_col = gene_exp.columns.values
        gene_id = np.array(gene_exp.index.values)
        print("raw exp shape: ", np.shape(gene_exp))

        isExists = os.path.exists(self.RES_PATH)
        if not isExists:
            os.makedirs(self.RES_PATH)

        if ST_flag == False:
            VGAE_exp = m.OT(gene_exp, OT_number)
            gene_exp.columns = VGAE_exp.columns.values
        else:
            VGAE_exp = copy.deepcopy(gene_exp)

        self.col = VGAE_exp.columns.values
        self.location = m.getLocation(self.col)

        # mix annotation files
        PQ = m.getpq(PQ_PATH)  # ['chr id', 'start', 'end', 'p'/'q']
        GENE_REFERENCE = np.loadtxt(REFERENCE_PATH, dtype=str, delimiter='\t')
        mixrefer = m.mixrefer(PQ, GENE_REFERENCE)
        mixrefer = pd.DataFrame(mixrefer[:, 1:], index=mixrefer[:, 0])

        gene_exp = gene_exp.join(mixrefer, how='inner')
        gene_exp[1] = pd.to_numeric(gene_exp[1])
        gene_exp.sort_values(by=[0, 1], inplace=True)  # gene_id x [sample + chr_id + start + end + gene_name + p/q]
        self.mixrefer = mixrefer
        self.gene_exp = gene_exp
        self.VGAE_exp = m.normalise(VGAE_exp, alpha, sigma)

    def cluster(self, Model_number=10, m_out=512, dim_output=128, LEARNING_RAT=5e-5, NUM_EPOCHS=50, solution=2, distance=1, weight=[1, 1e-1, 1e-3]):
        model_data = copy.deepcopy(self.VGAE_exp.values.transpose())  # samples x gene
        A_matrix = m.afterK_G(model_data, self.location, self.percent, distance)
        G = copy.deepcopy(A_matrix)
        A_matrix = torch.tensor(A_matrix)
        A_matrix2 = A_matrix.float()
        A_matrix = m.norMatrix(A_matrix2)

        i = 0
        print("model_data shape:", np.shape(model_data))
        m_in = model_data.shape[1]

        model = m.VGAE(A_matrix, m_in, m_out, dim_output)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RAT)

        print("running VGAE")

        while i < Model_number:
            re_loss = []
            kl_loss = []
            total_loss = []
            print("Time: ", i+1)

            data = torch.tensor(model_data)
            data = data.float()

            # train VGAE model
            for epoch in range(NUM_EPOCHS):
                # data = data.to(device)
                decode_A, decode_X, z = model(data)
                # print(decode_A)

                loss, l_A, l_X, KL = m.loss_function(decode_A, A_matrix2, decode_X, data, model.mean, model.logstddev,
                                                     weight)
                re_loss = re_loss + l_X.detach().numpy().reshape(-1).tolist()
                kl_loss = kl_loss + KL.detach().numpy().reshape(-1).tolist()
                total_loss = total_loss + loss.detach().numpy().reshape(-1).tolist()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if epoch % 10 == 0:
                    print(
                        'Loss: {:.3f} AL: {:.3f} XL: {:.3f} KL: {:.3f}'.format(loss.data, l_A.data, l_X.data, KL.data))

            i = i + 1

        z = z.detach().numpy()

        newG = m.createG(G, z)
        predict_label = m.classifyLouvin(newG, solution)

        col = pd.DataFrame(self.col)
        predict_label = pd.DataFrame(predict_label)
        cluster_res = pd.concat([col, predict_label], axis=1)
        cluster_res.columns = ["location", "cluster"]
        return cluster_res

    def inferCNV(self, REFERGENEEXP, predict_label, scale=3, windows=31, delta=20, low_per=0.15, high_per=0.85):
        category_number = len(set(predict_label["cluster"]))
        predict = copy.deepcopy(predict_label)
        predict = predict["cluster"].values
        refer_geneexp = pd.read_csv(REFERGENEEXP, delimiter='\t', index_col=0)
        hmmgene = copy.deepcopy(self.gene_exp[self.col])
        hmmgene = m.Hmmgene(hmmgene, refer_geneexp)
        hmmgene = hmmgene.join(self.mixrefer, how='inner')
        hmmgene[1] = pd.to_numeric(hmmgene[1])
        hmmgene.sort_values(by=[0, 1], inplace=True)
        self.hmmgene_index = hmmgene.index
        pqpoint, pqinfo = m.cutpd(hmmgene)
        np.savetxt(self.RES_PATH + "/pq.txt", pqpoint, fmt='%d', delimiter='\t')
        featureHMM = hmmgene[self.col]
        featureHMM = copy.deepcopy(featureHMM.values.transpose())
        print("HMM_data shape:", np.shape(featureHMM))

        model_data = copy.deepcopy(featureHMM)

        k1 = np.median(np.quantile(model_data, low_per, method='lower', axis=1))
        k2 = np.median(np.quantile(model_data, high_per, method='higher', axis=1))

        for j in range(category_number):
            sample = featureHMM[predict == j]

            print("sample " + str(j) + " shape: ", np.shape(sample))
            predict_result = m.MultiHMM(sample, pqpoint, k1, k2, scale, windows, delta)
            model_data[predict == j] = predict_result
        res = pd.DataFrame(np.transpose(model_data), index=self.hmmgene_index, columns=self.col, dtype=int)
        return res



