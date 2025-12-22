import shap
import numpy as np
import pandas as pd


def shap_feature_importance_rf(
    model,
    X,
    feature_names,
    class_names=None
):
    """
    Compute SHAP feature importance for RandomForest / Tree-based models.

    Parameters
    ----------
    model : trained tree-based model
        RandomForest / XGBoost
    X : pd.DataFrame or np.ndarray
        Input features
    feature_names : list
        Feature names
    class_names : list, optional
        Class labels

    Returns
    -------
    pd.DataFrame
        Mean absolute SHAP values per class
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # shap_values: (n_samples, n_features, n_classes)
    importance = {}

    for i in range(shap_values.shape[-1]):
        importance[i] = np.abs(shap_values[:, :, i]).mean(axis=0)

    importance_df = pd.DataFrame(
        importance,
        index=feature_names
    )

    if class_names is not None:
        importance_df.columns = class_names

    return importance_df


def get_activate_W_from_fmap(
    fmap,
    X,
    padding,
    pool=1,
    threshold=0.99,
    motif_width=10
):
    """
    Extract motif PWMs from CNN feature maps.

    Parameters
    ----------
    fmap : np.ndarray
        Feature maps of shape (N, filters, positions)
    X : np.ndarray
        One-hot encoded sequences (N, 4, seq_len)
    padding : int
        Padding size used in convolution
    pool : int
        Pooling size
    threshold : float
        Activation threshold (relative to max)
    motif_width : int
        Width of motif

    Returns
    -------
    W : np.ndarray
        PWM matrix for each filter
    seq_ls : list
        List of aligned sequences per filter
    """
    motif_nb = fmap.shape[1]
    X_dim, seq_len = X.shape[1], X.shape[-1]

    W = []
    seq_ls = []

    for filter_index in range(motif_nb):
        data_index, pos_index = np.where(
            fmap[:, filter_index, :]
            > np.max(fmap[:, filter_index, :]) * threshold
        )

        seq_align = []
        count_matrix = []

        for i in range(len(pos_index)):
            start = pos_index[i] - padding
            end = start + motif_width + 2

            if end > seq_len:
                end = seq_len
                start = end - motif_width - 2
            if start < 0:
                start = 0
                end = start + motif_width + 2

            seq = X[data_index[i], :, start * pool : end * pool]
            seq_align.append(seq)
            count_matrix.append(np.sum(seq, axis=0, keepdims=True))

        seq_align = np.array(seq_align)
        seq_ls.append(seq_align)

        count_matrix = np.array(count_matrix)

        pwm = (
            np.sum(seq_align, axis=0) /
            np.sum(count_matrix, axis=0)
        ) * np.ones((X_dim, (motif_width + 2) * pool))

        pwm[np.isnan(pwm)] = 0
        W.append(pwm)

    return np.array(W), seq_ls


def meme_generate_top(W, tf_list, output_file="meme.txt", prefix="Motif_"):
    """
    Write motifs to MEME format.
    """
    nt_freqs = [0.25, 0.25, 0.25, 0.25]

    with open(output_file, "w") as f:
        f.write("MEME version 4\n\n")
        f.write("ALPHABET= ACGT\n\n")
        f.write("strands: + -\n\n")
        f.write("Background letter frequencies:\n")
        f.write("A %.4f C %.4f G %.4f T %.4f\n\n" % tuple(nt_freqs))

        for j in tf_list:
            pwm = W[j]
            if np.sum(pwm) > 0:
                f.write(f"MOTIF {prefix}{j} {j}\n\n")
                f.write(
                    f"letter-probability matrix: alength= 4 w= {pwm.shape[1]} "
                    f"nsites= {pwm.shape[1]} E= 0\n"
                )
                for i in range(pwm.shape[1]):
                    f.write(
                        "  %.4f\t%.4f\t%.4f\t%.4f\n" % tuple(pwm[:, i])
                    )
                f.write("\n")


def calc_motif_IC(motif, background=0.25):
    """
    Calculate information content (IC) of a motif.
    """
    return (motif * np.log2(motif / background + 1e-6)).sum()

























