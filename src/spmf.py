import numpy as np
import random as rd
from tqdm import tqdm
import time
from utils import read_edge_list, save_edge_list, save_null_edge_list, sign_prediction, link_prediction
from sklearn.model_selection import train_test_split
import networkx as nx
from scipy import sparse
import math
from scipy.sparse.linalg import svds


class SPMFModel(object):
    def __init__(self, args):
        self.args = args
        self.setup()

    def setup(self):
        self.edges, self.numNodes = read_edge_list(self.args)
        self.trainEdges, self.testEdges, = train_test_split(self.edges,
                                                            test_size=self.args.testSize,
                                                            random_state=self.args.splitSeed)
        self.G = nx.DiGraph()
        self.G.add_nodes_from(range(self.numNodes))
        for edge in self.trainEdges:
            self.G.add_edge(edge[0], edge[1], weight=edge[2])

        # Generate null links set for link prediction task.
        if self.args.linkPrediction:
            G = nx.DiGraph()
            G.add_nodes_from(range(self.numNodes))
            G.add_edges_from([[e[0], e[1]] for e in self.edges])
            self.trainNullEdges, self.testNullEdges = [], []
            for _ in range(3 * len(self.testEdges)):
                u = rd.choice(range(self.numNodes))
                v = rd.choice(range(self.numNodes))
                while v in list(G.successors(u)):
                    v = rd.choice(range(self.numNodes))
                self.testNullEdges.append([u, v, 'n'])
            for _ in range(3 * len(self.trainEdges)):
                u = rd.choice(range(self.numNodes))
                v = rd.choice(range(self.numNodes))
                while v in list(G.successors(u)):
                    v = rd.choice(range(self.numNodes))
                self.trainNullEdges.append([u, v, 'n'])

    def calculate(self):
        np.seterr(divide='ignore', invalid='ignore')
        dataType = np.float32
        A = (nx.to_scipy_sparse_matrix(self.G, dtype=dataType, format='csc')).astype(dataType)
        D_list = (np.abs(A) @ np.ones((self.numNodes, 1)))[:, 0]
        D_invList = D_list.copy()
        for i in range(D_invList.shape[0]):
            if D_invList[i] != 0:
                D_invList[i] = 1 / D_list[i]
        D_inv = sparse.diags(D_invList, format='csc', dtype=dataType)
        P = D_inv @ A
        P_abs = D_inv @ np.abs(A)
        P_sum = P @ np.ones((self.numNodes, 1))
        tmp1 = P_sum
        for i in range(1, self.args.h):
            tmp1 = P @ tmp1
            P_sum += tmp1

        term1 = np.ones((self.numNodes, 1)) + 1 / self.args.h * P_sum
        term2 = term1 * term1
        term3 = np.ones((self.numNodes, 1)) - 1 / self.args.h * P_sum
        term4 = term3 * term3
        tmp = term1 * term3 + self.args.k * term4
        term5 = sparse.diags(tmp[:, 0], format='csc')
        tmp = term1 * term3 + self.args.k * term2
        term6 = sparse.diags(tmp[:, 0], format='csc')

        M = np.zeros((self.numNodes, self.numNodes), dtype=dataType)
        splitNum = math.ceil(self.numNodes / self.args.sliceSize)
        pbar = tqdm(total=splitNum, desc='Calculating summary matrix', ncols=100)
        for i in range(splitNum):
            pbar.update(1)
            startIndex = i * self.args.sliceSize
            if i != (splitNum - 1):
                endIndex = (i + 1) * self.args.sliceSize
            else:
                endIndex = self.numNodes

            P_sum = P[:, startIndex: endIndex]
            P_absSum = P_abs[:, startIndex: endIndex]
            tmp1 = P[:, startIndex: endIndex]
            tmp2 = P_abs[:, startIndex: endIndex]
            for j in range(self.args.h):
                if j == 0:
                    continue
                tmp1 = P @ tmp1
                tmp2 = P_abs @ tmp2
                P_sum = P_sum + tmp1
                P_absSum = P_absSum + tmp2
            numerator = term5 * (P_absSum + P_sum)
            denominator = term6 * (P_absSum - P_sum)
            partM = numerator / denominator
            partM = np.log(partM)
            partM[np.isnan(partM)] = 0
            partM[np.isinf(partM)] = 0
            if self.args.filter:
                partM[np.where((partM < np.log(self.args.r)) & (partM > np.log(1 / self.args.r)))] = 0
            M[:, startIndex: endIndex] = partM
        pbar.close()
        M = sparse.csc_matrix(M, dtype=dataType)
        u, s, vt = svds(M, self.args.dim)
        sqrtS = np.diag(np.sqrt(s))
        outEmb = u @ sqrtS
        inEmb = (sqrtS @ vt).T
        outEmb = outEmb[:, ::-1]
        inEmb = inEmb[:, ::-1]

        # Evaluation.
        print('Evaluating...')
        if self.args.signPrediction:
            auc, f1 = sign_prediction(outEmb, inEmb, self.trainEdges, self.testEdges)
            print('Sign prediction: AUC %.3f, F1 %.3f' % (auc, f1))
        if self.args.linkPrediction:
            auc, f1 = link_prediction(outEmb, inEmb, self.trainEdges, self.testEdges,
                                      self.trainNullEdges, self.testNullEdges)
            print('Link prediction: AUC %.3f, F1 %.3f' % (auc, f1))

        # Save representations.
        np.save(self.args.sourceRepPath, outEmb)
        np.save(self.args.targetRepPath, inEmb)
