

import numpy as np
from collections import Counter
import itertools

# -------------------------
# One-hot encoding
# -------------------------

ONEHOT_NUC = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'T': [0, 0, 0, 1],
    'N': [0, 0, 0, 0]
}


def one_hot(seq):
    """
    Convert a DNA sequence to one-hot encoding.

    Parameters
    ----------
    seq : str

    Returns
    -------
    np.ndarray
        Shape (L, 4)
    """
    code = np.zeros((len(seq), 4), dtype=np.float32)
    for i, base in enumerate(str(seq).upper()):
        code[i] = ONEHOT_NUC.get(base, ONEHOT_NUC['N'])
    return code


# -------------------------
# k-mer encoding
# -------------------------

def generate_all_kmers(k, alphabet="ACGT"):
    """Generate all possible k-mers."""
    return [''.join(x) for x in itertools.product(alphabet, repeat=k)]


def kmer_frequency(seq, k, all_kmers=None):
    """
    Count k-mer occurrences in a sequence.

    Returns
    -------
    dict
        {kmer: count}
    """
    if all_kmers is None:
        all_kmers = generate_all_kmers(k)

    counts = Counter(seq[i:i + k] for i in range(len(seq) - k + 1))
    return {kmer: counts.get(kmer, 0) for kmer in all_kmers}


def normalize_kmer_counts(kmer_df):
    """
    Normalize k-mer counts to frequencies.
    """
    return kmer_df.div(kmer_df.sum(axis=1), axis=0)




#####feature selection

from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

def lasso_feature_selection(X, y, cv=5, random_state=0, max_iter=1_000_000):
    """
    LASSO-based feature selection.
    Fit on training data only.
    """
    lasso = LassoCV(
        cv=cv,
        random_state=random_state,
        max_iter=max_iter
    )
    lasso.fit(X, y)

    selector = SelectFromModel(lasso, prefit=True)
    X_selected = selector.transform(X)

    return X_selected, selector


def rf_feature_selection(X, y, n_estimators=100, threshold="mean", random_state=0):
    """
    Random Forest feature selection using feature importance.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X, y)

    selector = SelectFromModel(rf, prefit=True, threshold=threshold)
    X_selected = selector.transform(X)

    return X_selected, selector


####confusion_matrix
def plot_confusion_matrix(
    cm,
    class_labels,
    save_filename="confusion_matrix.pdf",
    show=False
):
    """
    Plot and save confusion matrix.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix (num_classes x num_classes)
    class_labels : list
        Class names
    save_filename : str
        Output file name
    show : bool
        Whether to display the figure
    """
    num_classes = cm.shape[0]

    plt.figure(figsize=(5, 5))
    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap",
        ["#FFFFFF", "#2486B9", "#005E91"],
        N=256
    )

    im = plt.imshow(cm, interpolation="nearest", cmap=cmap)

    cbar = plt.colorbar(im, shrink=0.8)
    cbar.ax.tick_params(labelsize=11)

    plt.xlabel("Predicted Labels", fontsize=13)
    plt.ylabel("True Labels", fontsize=13)

    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_labels, fontsize=12, rotation=30, ha="right")
    plt.yticks(tick_marks, class_labels, fontsize=12)

    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(
                j, i, cm[i, j],
                ha="center", va="center",
                color="black", fontsize=12
            )

    plt.grid(False)
    plt.savefig(save_filename, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()


















