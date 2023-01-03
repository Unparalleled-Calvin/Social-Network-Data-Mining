import os
import random

import easygraph as eg
import networkx as nx
import numpy as np
import pandas as pd

file_names = {
    "user_id_list": "user_id_list.txt",
    "userInfo": "userInfo.txt",
    "user_follow": "user_follow.csv",
    "user_followed": "user_followed.csv",
    "music_id_list": "music_id_list.txt",
    "music_data": "music_data.csv",
    "music_comments": "music_comments.csv",
}

class DataLoader:
    def __init__(self, file_names=file_names, root_path="./Netease_music_social"):
        self.file_paths = {
            file: os.path.join(root_path, file_name)
            for file, file_name in file_names.items()
        }

    def load_user_id_list(self):
        file_path = self.file_paths["user_id_list"]
        return np.loadtxt(file_path, dtype=np.int64)

    def load_userInfo(self):
        file_path = self.file_paths["userInfo"]
        names = ["id", "name"] + [str(id) for id in range(15)] + ["description"]
        dtypes = {"id": np.int64, "name": object, "description": object}
        dtypes.update({str(id): object for id in range(15)})
        return pd.read_csv(file_path, sep="/", names=names, dtype=dtypes).fillna(value={"description": ""}) # 这里未知意义列没有对NaN做处理

    def load_user_follow(self):
        file_path = self.file_paths["user_follow"]
        dtypes = {"user_id": np.int64, "followed_id": np.int64, "followed_type": np.int64, "followed_gender": np.int64, "timestamp": np.int64}
        return pd.read_csv(file_path, dtype=dtypes)

    def load_user_followed(self):
        file_path = self.file_paths["user_followed"]
        dtypes = {"user_id": np.int64, "followed_id": np.int64, "followed_type": np.int64, "followed_gender": np.int64, "timestamp": np.int64}
        return pd.read_csv(file_path, dtype=dtypes)

    def load_music_id_list(self):
        file_path = self.file_paths["music_id_list"]
        return np.loadtxt(file_path, dtype=object)
    
    def load_music_data(self):
        file_path = self.file_paths["music_data"]
        dtypes = {"id": object}
        return pd.read_csv(file_path, index_col="num", dtype=dtypes)

    def load_music_comments(self):
        file_path = self.file_paths["music_comments"]
        dtypes = {"music_id": object, "user_id": np.int64, "timestamp": np.int64, "comment_id": np.int64, "liked_count": np.int64}
        return pd.read_csv(file_path, dtype=dtypes)

    def load_graph(self, use_file=True, save_file=True, file_name="graph.pkl", class_=eg.Graph):
        def user_node(user_id):
            return f"user_{user_id}"

        def music_node(music_id):
            return f"music_{music_id}"

        if use_file:
            try:
                print(f"try to load graph from file {file_name} ...")
                G = eg.read_pickle(file_name)
                assert(isinstance(G, class_))
                print(f"ok")
                return G
            except:
                print(f"fail to load {file_name}")
        
        print("build now ...")

        graph = class_()

        #user_id_list
        user_id_list = self.load_user_id_list()
        for user_id in user_id_list:
            graph.add_node(user_node(user_id), type="user")
        del user_id_list
        
        #userInfo
        # userInfo = self.load_userInfo()
        # for index, row in userInfo.iterrows():
        #     node = user_node(row["id"])
        #     graph.nodes[node]["name"] = row["name"]
        #     graph.nodes[node]["description"] = row["description"]
        # del userInfo

        #user_follow
        user_follow = self.load_user_follow()
        for index, row in user_follow.iterrows():
            graph.add_edge(user_node(row["user_id"]), user_node(row["follow_id"]), type="follow")
        del user_follow

        #user_followed
        user_followed = self.load_user_followed()
        for index, row in user_followed.iterrows():
            graph.add_edge(user_node(row["followed_id"]), user_node(row["user_id"]), type="follow")
        del user_followed

        #music_id_list
        music_id_list = self.load_music_id_list()
        for music_id in music_id_list:
            graph.add_node(music_node(music_id), type="music")
        del music_id_list

        #music_data
        music_data = self.load_music_data()
        for index, row in music_data.iterrows():
            node = music_node(row["id"])
            graph.nodes[node]["singer"] = row["singer"]
            graph.nodes[node]["album"] = row["album"]
            graph.nodes[node]["comment_num"] = row["comment_num"]
            graph.nodes[node]["play_list"] = row["playlist"]
        del music_data

        music_comments = self.load_music_comments()
        for index, row in music_comments.iterrows():
            graph.add_edge(user_node(row["user_id"]), music_node(row["music_id"]), timestamp=row["timestamp"], liked_count=row["liked_count"], type="like")
        del music_comments

        if save_file:
            eg.write_pickle(file_name, graph)

        print(f"ok")
        return graph

    def load_social_subgraph(self, use_file=True, save_file=True, file_name="social_subgraph.pkl", class_=eg.Graph):
        if use_file:
            try:
                print(f"try to load graph from file {file_name} ...")
                G = eg.read_pickle(file_name)
                assert(isinstance(G, class_))
                print(f"ok")
                return G
            except:
                print(f"fail to load {file_name}")
        
        print("build now ...")

        graph = self.load_graph(use_file=use_file, save_file=save_file, class_=class_)
        nodes = []
        for u, attr in graph.nodes.items():
            if attr["type"] == "user":
                nodes.append(u)
        subgraph = graph.nodes_subgraph(nodes)

        if save_file:
            eg.write_pickle(file_name, subgraph)
        
        print("ok")
        return subgraph


class DataSet:
    def __init__(self, loader: DataLoader):
        self.loader = loader
        graph = loader.load_graph()
        self.graph, self.index_of_node, self.node_of_index = graph.to_index_node_graph()
    
    def gen_postive_label(self, num): # 目标：拿到u,v,label
        music_comments = self.loader.load_music_comments()
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

    def gen_training_data(self, embedding, num=10000, ratio=0.5): #ratio是正负例比重，embedding格式为Dataframe，索引为id
        pos = self.gen_postive_label(int(num * ratio))
        neg = self.gen_negative_label(num - int(num * ratio))
        labels = pd.concat((pos, neg), axis=0)
        labels.reset_index(drop=True, inplace=True)
        data = [embedding.loc[row["music_id"]].to_list() +  embedding.loc[row["user_id"]].to_list() for _, row in labels.iterrows()]
        data = pd.DataFrame(data)
        return pd.concat((data, labels), axis=1) #这里还保留着node号