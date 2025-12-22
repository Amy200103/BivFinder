
#####machine learning
import numpy as np
from sklearn.model_selection import train_test_split
def build_dataset(X, y):
    """
    Build dataset for classification

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray or list
        Labels (0/1 or 0/1/2)

    Returns
    -------
    X : np.ndarray
    y : np.ndarray
    """
    X = np.asarray(X)
    y = np.asarray(y)

    assert len(X) == len(y), "X and y must have the same length"
    return X, y


def split_train_test(
    X,
    y,
    test_size=0.3,
    random_state=0,
    stratify=True
):
    """
    Randomly split dataset into train and test sets

    Parameters
    ----------
    X : np.ndarray
    y : np.ndarray
    test_size : float
    random_state : int
    stratify : bool

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    stratify_y = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_y
    )

    return X_train, X_test, y_train, y_test


def save_dataset(
    out_file,
    X_train,
    X_test,
    y_train,
    y_test
):
    """
    Save dataset to npz file
    """
    np.savez(
        out_file,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )


def load_dataset(npz_file):
    """
    Load dataset from npz file
    """
    data = np.load(npz_file)
    return (
        data["X_train"],
        data["X_test"],
        data["y_train"],
        data["y_test"]
    )


#####deep learning

def load_data(val_ratio=0.2, random_state=42):
    reader = SampleReader()
    seqs, labels = reader.get_seq(Test=False)

    train_seqs, val_seqs, train_labels, val_labels = train_test_split(
        seqs, labels, test_size=val_ratio, random_state=random_state
    )

    train_X = torch.tensor(train_seqs).permute(0, 2, 1)
    train_y = torch.tensor(train_labels).float().unsqueeze(1)

    val_X = torch.tensor(val_seqs).permute(0, 2, 1)
    val_y = torch.tensor(val_labels).float().unsqueeze(1)

    return train_X, train_y, val_X, val_y




















