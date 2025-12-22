

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense, Input


def build_svm(C=1.0, kernel="rbf", probability=True, random_state=0):
    """
    Build an SVM classifier (sklearn)

    Parameters
    ----------
    C : float
        Regularization parameter
    kernel : str
        Kernel type ("rbf", "linear", etc.)
    probability : bool
        Whether to enable probability estimates
    random_state : int
        Random seed

    Returns
    -------
    sklearn.svm.SVC
        Untrained SVM model
    """
    return SVC(
        C=C,
        kernel=kernel,
        probability=probability,
        random_state=random_state
    )


def build_random_forest(
    n_estimators=80,
    max_depth=8,
    random_state=0
):
    """
    Build a Random Forest classifier (sklearn)

    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest
    max_depth : int or None
        Maximum depth of each tree
    random_state : int
        Random seed

    Returns
    -------
    sklearn.ensemble.RandomForestClassifier
        Untrained Random Forest model
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )



def build_xgboost(
    learning_rate=0.5,
    n_estimators=10,
    max_depth=5,
    random_state=0
):
    """
    Build an XGBoost classifier

    Parameters
    ----------
    learning_rate : float
        Learning rate
    n_estimators : int
        Number of trees
    max_depth : int
        Maximum depth of trees
    random_state : int
        Random seed

    Returns
    -------
    xgboost.XGBClassifier
        Untrained XGBoost model
    """
    return XGBClassifier(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        eval_metric="error"
    )


def build_logistic_regression(
    C=10.0,
    penalty="l2",
    solver="saga",
    max_iter=1000,
    random_state=0
):
    """
    Build a Logistic Regression classifier

    Parameters
    ----------
    C : float
        Regularization strength
    penalty : str
        Regularization type ("l2")
    solver : str
        Optimization solver
    max_iter : int
        Maximum number of iterations
    random_state : int
        Random seed

    Returns
    -------
    sklearn.linear_model.LogisticRegression
        Untrained Logistic Regression model
    """
    return LogisticRegression(
        C=C,
        penalty=penalty,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state
    )


def build_nn(input_dim, hidden_units=10, activation="tanh", num_classes=3):
    """
    Build a simple fully-connected neural network (Keras)
    """
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(hidden_units, activation=activation))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
































