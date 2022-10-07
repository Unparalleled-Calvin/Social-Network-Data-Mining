import numpy as np
import pandas as pd
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
        return np.loadtxt(file_path, dtype=np.int64)
    
    def load_music_data(self):
        file_path = self.file_paths["music_data"]
        dtypes = {"id": np.int64}
        return pd.read_csv(file_path, index_col="num", dtype=dtypes)

    def load_music_comments(self):
        file_path = self.file_paths["music_comments"]
        dtypes = {"music_id": np.int64, "user_id": np.int64, "timestamp": np.int64, "comment_id": np.int64, "liked_count": np.int64}
        return pd.read_csv(file_path, dtype=dtypes)
