
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


RANDOM_STATE = 42


def train_svm(X_train, y_train):
    param_grid = {
        "C": [1, 3, 5, 7, 9],
        "kernel": ["linear", "rbf"]
    }

    svm = SVC(
        probability=True,
        random_state=RANDOM_STATE
    )

    grid = GridSearchCV(
        svm,
        param_grid,
        cv=5,
        scoring="roc_auc_ovr",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    return grid.best_estimator_


def train_rf(X_train, y_train):
    param_grid = {
        "n_estimators": [10, 50, 80],
        "max_depth": [2, 4, 8]
    }

    rf = RandomForestClassifier(
        random_state=RANDOM_STATE
    )

    grid = GridSearchCV(
        rf,
        param_grid,
        cv=5,
        scoring="roc_auc_ovr",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    return grid.best_estimator_


def train_lr(X_train, y_train):
    param_grid = {
        "C": [0.1, 1, 10],
        "solver": ["lbfgs", "saga", "newton-cg"],
        "penalty": ["l2"]
    }

    lr = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    )

    grid = GridSearchCV(
        lr,
        param_grid,
        cv=5,
        scoring="roc_auc_ovr",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    return grid.best_estimator_


def train_xgb(X_train, y_train):
    param_grid = {
        "learning_rate": [0.2, 0.5, 0.7],
        "n_estimators": [5, 10],
        "max_depth": [3, 5, 7]
    }

    xgb = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE
    )

    grid = GridSearchCV(
        xgb,
        param_grid,
        cv=5,
        scoring="roc_auc_ovr",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    return grid.best_estimator_


def train_nn(X_train, y_train):
    param_grid = {
        "hidden_layer_sizes": [(10,), (20,)],
        "activation": ["tanh", "relu"],
        "batch_size": [32, 64],
        "max_iter": [32, 64]
    }

    nn = MLPClassifier(
        random_state=RANDOM_STATE
    )

    grid = GridSearchCV(
        nn,
        param_grid,
        cv=5,
        scoring="roc_auc_ovr",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    return grid.best_estimator_


def train_all_models(X_train, y_train):
    models = {}

    models["SVM"] = train_svm(X_train, y_train)
    models["RF"] = train_rf(X_train, y_train)
    models["LR"] = train_lr(X_train, y_train)
    models["XGBoost"] = train_xgb(X_train, y_train)
    models["NN"] = train_nn(X_train, y_train)

    return models





































