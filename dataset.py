import numpy as np
import pandas as pd
import easygraph as eg
import random
from sklearn.model_selection import train_test_split

class DataSet:
    def __init__(self, graph: eg.Graph):
        self.graph, self.index_of_node, self.node_of_index = graph.to_index_node_graph()
    
    def gen_postive_label(self, num, music_comments: pd.DataFrame): # 目标：拿到u,v,label
        num = min(num, music_comments.shape[0])
        music_comments = music_comments.sort_values(by=["timestamp"], ascending=True) #根据时间排序
        sample_indice = sorted(random.sample(range(music_comments.shape[0]), num)) #抽取子集的index
        sample = music_comments.iloc[sample_indice]
        music_indice = sample.loc[:, ["music_id"]].applymap(lambda x: self.index_of_node["music_" + str(x)])
        user_indice = sample.loc[:, ["user_id"]].applymap(lambda x: self.index_of_node["user_" + str(x)])
        data = pd.concat([music_indice, user_indice], axis=1)
        data.loc[:, ["label"]] = 1
        return data

    def gen_negative_label(self, num):
        music, user = [], []
        pairs = []
        for node, attr in self.graph.nodes.items():
            music.append(node) if attr["type"] == "music" else user.append(node)
        for i in range(num):
            music_node = music[random.randint(0, len(music) - 1)]
            user_node = user[random.randint(0, len(user) - 1)]
            if not music_node in self.graph[user_node]:
                pairs.append([music_node, user_node])
        data = pd.DataFrame(pairs, columns=["music_id", "user_id"])
        data.loc[:, ["label"]] = 0
        return data

    def gen_training_data(self, embedding, music_comments, num=10000, ratio=0.5): #ratio是正负例比重，embedding格式为Dataframe，索引为id
        pos = self.gen_postive_label(num, music_comments)
        neg = self.gen_negative_label(num)
        labels = pd.concat((pos, neg), axis=0)
        data = [pd.concat((embedding.loc[row["music_id"]], embedding.loc[row["user_id"]]), axis=1) for _, row in labels.iterrows()]
        return data