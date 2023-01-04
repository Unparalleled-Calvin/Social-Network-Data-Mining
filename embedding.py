import pickle

import easygraph as eg
import networkx as nx
import pandas as pd
from node2vec import Node2Vec

from toolset import class_transform

emb_list = ['bet_cen', 'cons', 'hier', 'node']

def betweenness_centrality(G, filename="between.pkl", n_workers=None):
    try:
        with open(filename, "rb") as f:
            bet_cen = pickle.load(f)
    except:
        bet_cen = eg.betweenness_centrality(G, n_workers=n_workers)
    data = pd.DataFrame(bet_cen.items(), columns=["id", "bet_cen"])
    data.set_index(["id"], inplace=True)
    return data

def constraint(G, filename="constraint.pkl", n_workers=None):
    try:
        with open(filename, "rb") as f:
            cons = pickle.load(f)
    except:
        cons = eg.constraint(G, n_workers=n_workers)
    data = pd.DataFrame(cons.items(), columns=["id", "cons"])
    data.set_index(["id"], inplace=True)
    return data

def hierarchy(G, filename="hierarchy.pkl", n_workers=None):
    try:
        with open(filename, "rb") as f:
            hier = pickle.load(f)
    except:
        hier = eg.hierarchy(G, n_workers=n_workers)
    data = pd.DataFrame(hier.items(), columns=["id", "hier"])
    data.set_index(["id"], inplace=True)
    return data

def node2vec(G, dimensions=16, workers=8, filename="node2vec.csv", model_filename="node2vec.mode"):
    try:
        data = pd.read_csv("node2vec.csv", skiprows=1, sep=" ", index_col=0, names=["id"] + list(range(dimensions)))
    except:
        if not isinstance(G, nx.Graph):
            G = class_transform(G, nx.Graph)
        node2vec = Node2Vec(G, dimensions=dimensions, walk_length=30, num_walks=200, workers=workers)
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        model.wv.save_word2vec_format(filename)
        model.save(model_filename)
        data = pd.read_csv("node2vec.csv", skiprows=1, sep=" ", index_col=0, names=["id"] + list(range(dimensions)))
    finally:
        data = data.sort_index()
        return data

def node_attr(G):
    data = []
    for node in G.nodes:
        try:
            comment_num = G.nodes[node]["comment_num"]
        except:
            comment_num = 0
        data.append((node, comment_num))
    data = pd.DataFrame(data, columns=["id", "attr"])
    data.set_index(["id"], inplace=True)
    return data

def gen_embedding(G, drop_index=None):
    bet_cen = betweenness_centrality(G)
    cons = constraint(G)
    hier = hierarchy(G)
    node = node2vec(G)
    # attr = node_attr(G)
    embeds = [bet_cen, cons, hier, node]
    if drop_index != None:
        print(f"drop {emb_list[drop_index]}")
        embeds.pop(drop_index)
    data = pd.concat(embeds, axis=1)
    return data