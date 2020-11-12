import numpy as np
import math
from scipy.sparse.linalg import svds
import time
from scipy import sparse
import networkx as nx

DATA_DIR = 'E:/xph/SignedNetwork/'
EMB_DIR = 'E:/xph/Emb/SPMF/'
DATASET_NAME = 'Epinions'
REMAIN = 80
SEED = 'Matlab'
H = 5  # Highest order
K = 5  # Number of negative samples
DIM = 32  # Dimension of emb
FILTER = True  # Use filter or not
FILTER_PARAM = 5
DTYPE = np.float32

G = nx.DiGraph()
with open(DATA_DIR + 'RawNetwork/' + DATASET_NAME + '.txt') as f:
    for line in f:
        line = line.strip().split('\t')
        line = list(map(int, line))
        G.add_edge(line[0], line[1], weight=line[2])
numNodes = max(G.nodes) + 1

G = nx.DiGraph()
G.add_nodes_from(list(range(numNodes)))
with open(DATA_DIR + DATASET_NAME + '/remain=' + str(REMAIN) + '/seed=' + str(SEED) + '/trainSet.txt') as f:
# with open('../trainSet.txt') as f:
    for line in f:
        line = line.strip().split('\t')
        line = list(map(int, line))
        G.add_edge(line[0], line[1], weight=line[2])
# There maybe a bug in function nx.to_scipy_sparse_matrix(). Although I specify the dtype to np.float32, the dtype of
# the output is int. Thus, I use astype() to convert the dtype.
A = (nx.to_scipy_sparse_matrix(G, dtype=DTYPE, format='csc')).astype(np.float32)
D_list = (np.abs(A) @ np.ones((numNodes, 1)))[:, 0]
D = sparse.diags(D_list, format='csc', dtype=DTYPE)
D_invList = D_list.copy()
for i in range(D_invList.shape[0]):
    if D_invList[i] != 0:
        D_invList[i] = 1 / D_list[i]
D_inv = sparse.diags(D_invList, format='csc', dtype=DTYPE)
P = D_inv @ A
P_abs = D_inv @ np.abs(A)
P_sum = P @ np.ones((numNodes, 1))
tmp1 = P_sum
for i in range(1, H):
    tmp1 = P @ tmp1
    P_sum += tmp1

term1 = np.ones((numNodes, 1)) + 1 / H * P_sum
term2 = term1 * term1
term3 = np.ones((numNodes, 1)) - 1 / H * P_sum
term4 = term3 * term3
tmp = term1 * term3 + K * term4
term5 = sparse.diags(tmp[:, 0], format='csc')
tmp = term1 * term3 + K * term2
term6 = sparse.diags(tmp[:, 0], format='csc')

start = time.time()
M = np.zeros((numNodes, numNodes), dtype=DTYPE)
sliceSize = 1000
splitNum = math.ceil(numNodes / sliceSize)

for i in range(splitNum):
    tmp_start = time.time()
    print(i)
    startIndex = i * sliceSize
    if i != (splitNum - 1):
        endIndex = (i + 1) * sliceSize
    else:
        endIndex = numNodes

    P_sum = P[:, startIndex: endIndex]
    P_absSum = P_abs[:, startIndex: endIndex]
    tmp1 = P[:, startIndex: endIndex]
    tmp2 = P_abs[:, startIndex: endIndex]
    for j in range(H):
        if j == 0:
            continue
        tmp1 = P @ tmp1
        tmp2 = P_abs @ tmp2
        P_sum = P_sum + tmp1
        P_absSum = P_absSum + tmp2
    # print('stage 0')
    numerator = term5 * (P_absSum + P_sum)
    denominator = term6 * (P_absSum - P_sum)
    # print('stage 1')
    partM = numerator / denominator
    partM = np.log(partM)
    partM[np.isnan(partM)] = 0
    partM[np.isinf(partM)] = 0
    if FILTER:
        partM[np.where((partM < np.log(FILTER_PARAM)) & (partM > np.log(1 / FILTER_PARAM)))] = 0
    # print('stage 2')
    M[:, startIndex: endIndex] = partM
    tmp_end = time.time()
    print('Iteration time: ', tmp_end - tmp_start)
end = time.time()
print("Execution Time: ", end - start)

start = time.time()
M = sparse.csc_matrix(M, dtype=np.float32)
end = time.time()
print("Execution Time: ", end - start)

start = time.time()
u, s, vt = svds(M, DIM)
end = time.time()
print("Execution Time: ", end - start)

sqrtS = np.diag(np.sqrt(s))
outEmb = u @ sqrtS
inEmb = (sqrtS @ vt).T
outEmb = outEmb[:, ::-1]
inEmb = inEmb[:, ::-1]
np.save(EMB_DIR + DATASET_NAME + '/remain=' + str(REMAIN) + '/seed=' + str(SEED) + '/outEmb', outEmb)
np.save(EMB_DIR + DATASET_NAME + '/remain=' + str(REMAIN) + '/seed=' + str(SEED) + '/inEmb', inEmb)
