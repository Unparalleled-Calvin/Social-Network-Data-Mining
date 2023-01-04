from sklearn.model_selection import RandomizedSearchCV, train_test_split
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

    xgb_grid = {
        'n_estimators': [100,200,400,800,1000],
        'max_depth': [2,3,4,6,7],
        'min_child_weight':[1,2,3,4],
        'eta':[0.1, 0.2,0.3],
        'gamma': [0],
        }
    xgb_base = XGBClassifier()

    xgb_random = RandomizedSearchCV(estimator = xgb_base, param_distributions=xgb_grid, n_iter=200, cv=3, verbose=2)
    xgb_random.fit(X_train, y_train)
    print(xgb_random.best_params_)