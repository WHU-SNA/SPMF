import argparse
import numpy as np
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from texttable import Texttable


def parameter_parser():
    """
    Parse up command line parameters.
    """
    parser = argparse.ArgumentParser(description="Run SLF.")
    parser.add_argument("--edgePath",
                        nargs="?",
                        default="./input/WikiElec.txt",
                        help="Edge list in txt format.")
    parser.add_argument("--sourceRepPath",
                        nargs="?",
                        default="./output/WikiElec_source",
                        help="Source representation path.")
    parser.add_argument("--targetRepPath",
                        nargs="?",
                        default="./output/WikiElec_target",
                        help="Target representation path.")
    parser.add_argument("--dim",
                        type=int,
                        default=32,
                        help="Dimension of the representation. Default is 32.")
    parser.add_argument("--k",
                        type=int,
                        default=5,
                        help="Number of noise samples. Default is 5.")
    parser.add_argument("--h",
                        type=int,
                        default=5,
                        help="Highest order. Default is 5.")
    parser.add_argument("--sliceSize",
                        type=int,
                        default=1000,
                        help="Slice size. Default is 1000.")
    parser.add_argument("--filter",
                        type=bool,
                        default=False,
                        help="Use filter trick or not. Default is False.")
    parser.add_argument("--r",
                        type=int,
                        default=5,
                        help="Parameter of filter trick. Default is 5.")
    parser.add_argument("--testSize",
                        type=float,
                        default=0.2,
                        help="Test ratio. Default is 0.2.")
    parser.add_argument("--splitSeed",
                        type=int,
                        default=1,
                        help="Random seed for splitting dataset. Default is 1.")
    parser.add_argument("--signPrediction",
                        type=bool,
                        default=True,
                        help="Perform sign prediction or not. Default is True.")
    parser.add_argument("--linkPrediction",
                        type=bool,
                        default=False,
                        help="perform link prediction or not. Default is False.")

    return parser.parse_args()


def read_edge_list(args):
    """
    Load edges from a txt file.
    """
    G = nx.DiGraph()
    edges = np.loadtxt(args.edgePath)
    for i in range(edges.shape[0]):
        G.add_edge(int(edges[i][0]), int(edges[i][1]), weight=edges[i][2])
    edges = [[e[0], e[1], e[2]['weight']] for e in G.edges.data()]
    return edges, max(G.nodes) + 1  # index can start from 0.


def save_edge_list(file_path, edges):
    with open(file_path, 'w') as f:
        for edge in edges:
            edge = list(map(str, edge))
            f.write('\t'.join(edge) + '\n')

def save_null_edge_list(file_path, edges):
    with open(file_path, 'w') as f:
        for edge in edges:
            edge[2] = 0
            edge = list(map(str, edge))
            f.write('\t'.join(edge) + '\n')

@ignore_warnings(category=ConvergenceWarning)
def sign_prediction(out_emb, in_emb, train_edges, test_edges):
    """
    Evaluate the performance on the sign prediction task.
    :param out_emb: Outward embeddings.
    :param in_emb: Inward embeddings.
    :param train_edges: Edges for training the model.
    :param test_edges: Edges for test.
    """
    out_dim = out_emb.shape[1]
    in_dim = in_emb.shape[1]
    train_edges = train_edges
    train_x = np.zeros((len(train_edges), (out_dim + in_dim) * 2))
    train_y = np.zeros((len(train_edges), 1))
    for i, edge in enumerate(train_edges):
        u = edge[0]
        v = edge[1]
        if edge[2] > 0:
            train_y[i] = 1
        else:
            train_y[i] = 0
        train_x[i, : out_dim] = out_emb[u]
        train_x[i, out_dim: out_dim + in_dim] = in_emb[u]
        train_x[i, out_dim + in_dim: out_dim * 2 + in_dim] = out_emb[v]
        train_x[i, out_dim * 2 + in_dim:] = in_emb[v]

    test_edges = test_edges
    test_x = np.zeros((len(test_edges), (out_dim + in_dim) * 2))
    test_y = np.zeros((len(test_edges), 1))
    for i, edge in enumerate(test_edges):
        u = edge[0]
        v = edge[1]
        if edge[2] > 0:
            test_y[i] = 1
        else:
            test_y[i] = 0
        test_x[i, : out_dim] = out_emb[u]
        test_x[i, out_dim: out_dim + in_dim] = in_emb[u]
        test_x[i, out_dim + in_dim: out_dim * 2 + in_dim] = out_emb[v]
        test_x[i, out_dim * 2 + in_dim:] = in_emb[v]

    ss = StandardScaler()
    train_x = ss.fit_transform(train_x)
    test_x = ss.fit_transform(test_x)
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(train_x, train_y.ravel())
    test_y_score = lr.predict_proba(test_x)[:, 1]
    test_y_pred = lr.predict(test_x)
    auc_score = roc_auc_score(test_y, test_y_score, average='macro')
    macro_f1_score = f1_score(test_y, test_y_pred, average='macro')

    return auc_score, macro_f1_score


@ignore_warnings(category=ConvergenceWarning)
def link_prediction(out_emb, in_emb, train_edges, test_edges, train_edges_null, test_edges_null):
    """
    Evaluate the performance on the link prediction task.
    :param out_emb: Outward embeddings.
    :param in_emb: Inward embeddings.
    :param train_edges: Edges for training the model.
    :param test_edges: Edges for test.
    """
    dim = out_emb.shape[1]
    train_x = np.zeros((len(train_edges) + len(train_edges_null), dim * 4))
    train_y = np.zeros((len(train_edges) + len(train_edges_null), 1))
    for i, edge in enumerate(train_edges):
        u = edge[0]
        v = edge[1]
        train_x[i, : dim] = out_emb[u]
        train_x[i, dim: dim * 2] = in_emb[u]
        train_x[i, dim * 2: dim * 3] = out_emb[v]
        train_x[i, dim * 3:] = in_emb[v]
        if edge[2] > 0:
            train_y[i] = 1
        else:
            train_y[i] = -1

    for i, edge in enumerate(train_edges_null):
        i += len(train_edges)
        u = edge[0]
        v = edge[1]
        train_x[i, : dim] = out_emb[u]
        train_x[i, dim: dim * 2] = in_emb[u]
        train_x[i, dim * 2: dim * 3] = out_emb[v]
        train_x[i, dim * 3:] = in_emb[v]
        train_y[i] = 0

    test_x = np.zeros((len(test_edges) + len(test_edges_null), dim * 4))
    test_y = np.zeros((len(test_edges) + len(test_edges_null), 1))
    for i, edge in enumerate(test_edges):
        u = edge[0]
        v = edge[1]
        test_x[i, : dim] = out_emb[u]
        test_x[i, dim: dim * 2] = in_emb[u]
        test_x[i, dim * 2: dim * 3] = out_emb[v]
        test_x[i, dim * 3:] = in_emb[v]
        if edge[2] > 0:
            test_y[i] = 1
        else:
            test_y[i] = -1

    for i, edge in enumerate(test_edges_null):
        i += len(test_edges)
        u = edge[0]
        v = edge[1]
        test_x[i, : dim] = out_emb[u]
        test_x[i, dim: dim * 2] = in_emb[u]
        test_x[i, dim * 2: dim * 3] = out_emb[v]
        test_x[i, dim * 3:] = in_emb[v]
        test_y[i] = 0

    ss = StandardScaler()
    train_x = ss.fit_transform(train_x)
    test_x = ss.fit_transform(test_x)
    lr = LogisticRegression()
    lr.fit(train_x, train_y.ravel())
    pred_prob = lr.predict_proba(test_x)
    pred_label = lr.predict(test_x)
    test_y = test_y[:, 0]
    AucScore = roc_auc_score(test_y, pred_prob, average='macro', multi_class='ovo')
    MacroF1Score = f1_score(test_y, pred_label, average='macro')

    return AucScore, MacroF1Score


def args_printer(args):
    """
    Print the parameters in tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    t = Texttable()
    l = [[k, args[k]] for k in args.keys()]
    l.insert(0, ["Parameter", "Value"])
    t.add_rows(l)
    print(t.draw())
