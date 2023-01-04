from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from dataset import DataLoader, DataSet
from embedding import gen_embedding
from toolset import metrics


def train(model, X_train, X_test, y_train, y_test):
    print(f"{model.__class__}:")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics(y_test, y_pred)

if __name__ == "__main__":
    loader = DataLoader()
    dataset = DataSet(loader)
    embedding = gen_embedding(dataset.graph)
    data = dataset.gen_training_data(embedding, 10000)
    X, y = data.iloc[:, :-3], data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    models = [
        DecisionTreeClassifier(),
        XGBClassifier(),
        LGBMClassifier()
    ]

    for model in models:
        train(model, X_train, X_test, y_train, y_test)
        print()
    