import numpy as np
import pandas as pd
import pickle as pkl
import easygraph as eg
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

    def load_graph(self, use_file=True, save_file=True, file_name="graph.pkl"):
        if use_file:
            try:
                print(f"try to load graph from file {file_name} ...")
                with open(file_name, "rb") as f:
                    graph = pkl.load(f)
                return graph
            except:
                print(f"fail to load {file_name}\nbuild now ...")
        
        graph = eg.DiGraph()

        #user_id_list
        user_id_list = self.load_user_id_list()
        for user_id in user_id_list:
            graph.add_node(user_id, type="user")
        del user_id_list
        
        #userInfo
        # userInfo = self.load_userInfo()
        # for index, row in userInfo.iterrows():
        #     id = row["id"]
        #     graph.nodes[id]["name"] = row["name"]
        #     graph.nodes[id]["description"] = row["description"]
        # del userInfo

        #user_follow
        user_follow = self.load_user_follow()
        for index, row in user_follow.iterrows():
            user_id, follow_id = row["user_id"], row["follow_id"]
            assert(user_id in graph and follow_id in graph)
            graph.add_edge(user_id, follow_id, type="follow")
        del user_follow

        #user_followed
        user_followed = self.load_user_followed()
        for index, row in user_followed.iterrows():
            user_id, followed_id = row["user_id"], row["followed_id"]
            assert(user_id in graph and followed_id in graph)
            graph.add_edge(followed_id, user_id, type="follow")
        del user_followed

        #music_id_list
        music_id_list = self.load_music_id_list()
        for music_id in music_id_list:
            graph.add_node(music_id, type="music")
        del music_id_list

        #music_data
        music_data = self.load_music_data()
        for index, row in music_data.iterrows():
            id = row["id"]
            graph.nodes[id]["singer"] = row["singer"]
            graph.nodes[id]["album"] = row["album"]
            graph.nodes[id]["comment_num"] = row["comment_num"]
            graph.nodes[id]["play_list"] = row["playlist"]
        del music_data

        music_comments = self.load_music_comments()
        for index, row in music_comments.iterrows():
            graph.add_edge(row["user_id"], row["music_id"], timestamp=row["timestamp"], liked_count=row["liked_count"])
        del music_comments

        if save_file:
            with open(file_name, "wb") as f:
                pkl.dump(graph, f)

        return graph
        

if __name__ == "__main__":
    loader = DataLoader()
    graph = loader.load_graph()
