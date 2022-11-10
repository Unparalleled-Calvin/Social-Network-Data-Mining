import numpy as np
import pandas as pd
import easygraph as eg
import networkx as nx
import random
import os

file_names = {
    "user_id_list": "user_id_list.txt",
    "userInfo": "userInfo.txt",
    "user_follow": "user_follow.csv",
    "user_followed": "user_followed.csv",
    "music_id_list": "music_id_list.txt",
    "music_data": "music_data.csv",
    "music_comments": "music_comments.csv",
}

def eg2nx(G, class_=nx.Graph):
    G_ = class_()
    for u, attr in G.nodes.items():
        G_.add_node(u, **attr)
    for u, v, attr in G.edges:
        G_.add_edge(u, v, **attr)
    return G_
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

    def load_graph(self, use_file=True, save_file=True, file_name="graph.pkl", class_=eg.DiGraph):
        def user_node(user_id):
            return f"user_{user_id}"

        def music_node(music_id):
            return f"music_{music_id}"

        if use_file:
            try:
                print(f"try to load graph from file {file_name} ...")
                G = eg.read_pickle(file_name)
                assert(isinstance(G, class_))
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
            graph.add_edge(user_node(row["user_id"]), music_node(row["music_id"]), timestamp=row["timestamp"], liked_count=row["liked_count"])
        del music_comments

        if save_file:
            eg.write_pickle(file_name, graph)

        return graph

class Sampler:
    def __init__(self, graph: eg.Graph):
        self.graph = graph
    
    def random_node(self, ratio = 0.01, max_num = 1500, min_num = 0, save_file = True, file_name = "subgraph.gexf"): # 随机抽样
        nodes = list(self.graph.nodes)
        num = max(min(int(ratio * len(nodes)), max_num), min_num)
        sample = nodes[:num]
        random.shuffle(sample)
        subgraph = self.graph.nodes_subgraph(from_nodes=sample)
        if save_file:
            eg.write_gexf(subgraph, file_name)
        return subgraph
        
    def diffusion(self, ratio = 0.01, max_num = 1500, min_num = 0, save_file = True, file_name = "subgraph.gexf"): # diffusion
        num = max(min(int(ratio * len(self.graph.nodes)), max_num), min_num)
        from littleballoffur import DiffusionSampler
        sampler = DiffusionSampler(number_of_nodes=num)
        G_, index_of_node, node_of_index = self.graph.to_index_node_graph()
        subgraph_ = sampler.sample(eg2nx(G_, nx.Graph))
        nodes = [node_of_index[i] for i in subgraph_.nodes]
        subgraph = self.graph.nodes_subgraph(nodes)
        if save_file:
            eg.write_gexf(subgraph, file_name)
        return subgraph

if __name__ == "__main__":
    loader = DataLoader()
    graph = loader.load_graph(class_=eg.DiGraph)
    sampler = Sampler(graph)
    sampler.diffusion()
