# BivFinder
A sequence-based predictive framework for bivalent histone modifications

## Dependencies
- Python 3.10.14
- PyTorch 2.6.0
- NumPy 1.26.4
- Pandas 2.2.2
- scikit-learn 1.5.1

## Data Preparation
Peak regions were extracted and categorized into bivalent and monovalent
classes based on the co-occurrence of histone marks. For each genomic region,
a fixed-length DNA sequence (e.g., 1000 bp centered on the peak summit)
was retrieved from the mouse reference genome.

## Model Training

We trained both traditional machine learning models and deep learning models
to predict bivalent chromatin regions from DNA sequence features.

### Machine Learning Models

The following classifiers were evaluated:

- Logistic Regression (LR)
- Support Vector Machine (SVM)
- Random Forest (RF)
- XGBoost

Hyperparameters were optimized using grid search on the training set.
The final models were trained using the optimal hyperparameters and
evaluated on a held-out test set.

### Deep Learning Models

Deep learning models, including CNN-based architectures, were implemented
using PyTorch. Models were trained with binary cross-entropy loss and
optimized using the Adam optimizer. Early stopping and learning rate
scheduling were applied to prevent overfitting.

Model training scripts are provided in `train.py`.
