import networkx as nx
import numpy as np
import pandas as pd


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