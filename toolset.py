import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def class_transform(G, class_=nx.Graph):
    G_ = class_()
    for u, attr in G.nodes.items():
        G_.add_node(u, **attr)
    for u, v, attr in G.edges:
        G_.add_edge(u, v, **attr)
    return G_

def gen_test_embedding(graph, length=5):
    data = [np.concatenate((np.array([node]), np.random.random(length))) for node in graph.nodes]
    data = pd.DataFrame(data, columns = ["id"] + ["emb_" + str(i) for i in range(length)])
    data.set_index(["id"], inplace=True)
    return data

def metrics(y, y_pred, verbose=True):
    accuracy, precision, recall, f1 = accuracy_score(y, y_pred), precision_score(y, y_pred), recall_score(y, y_pred), f1_score(y, y_pred)
    if verbose:
        print("Accuracy: %.2f%%"%(accuracy * 100.0))
        print('Precision: %.2f%%'%(precision * 100.0))
        print('Recall: %.2f%%'%(recall * 100.0))
        print('F1-score: %.2f%%'%(f1 * 100.0))
    return accuracy, precision, recall, f1