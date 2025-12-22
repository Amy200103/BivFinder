

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report
)
from sklearn.preprocessing import label_binarize


def evaluate_model(model, X_test, y_test, model_name=None):
    """
    Evaluate a trained model on the held-out test set

    Parameters
    ----------
    model : fitted classifier
        Trained machine learning model
    X_test : array-like
        Test features
    y_test : array-like
        True labels
    model_name : str, optional
        Name of the model (for printing)

    Returns
    -------
    dict
        Evaluation metrics
    """

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)

    # Multi-class ROC-AUC (OVR)
    classes = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=classes)
    auc = roc_auc_score(
        y_test_bin,
        y_prob,
        multi_class="ovr",
        average="macro"
    )

    results = {
        "accuracy": acc,
        "auc": auc
    }

    if model_name is not None:
        print("=" * 50)
        print(f"Model: {model_name}")
        print(f"Accuracy: {acc:.4f}")
        print(f"ROC-AUC (OVR): {auc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("=" * 50)

    return results


def evaluate_all_models(models, X_test, y_test):
    """
    Evaluate multiple trained models

    Parameters
    ----------
    models : dict
        {model_name: trained_model}
    X_test : array-like
        Test features
    y_test : array-like
        True labels
    """

    all_results = {}

    for name, model in models.items():
        all_results[name] = evaluate_model(
            model,
            X_test,
            y_test,
            model_name=name
        )

    return all_results













