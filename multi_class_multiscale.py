import numpy as np
import sys

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    recall_score,
    precision_score,
)

# ============================================================
# Configuration
# ============================================================

SEED = 1001
K = 10

np.random.seed(SEED)

# ============================================================
# Select MULTIPLE scales
# ============================================================
#
# Usage examples:
#
#   python multi_class_multiscale_corrected.py 20
#   python multi_class_multiscale_corrected.py 10 20
#   python multi_class_multiscale_corrected.py 1 10 20
#   python multi_class_multiscale_corrected.py 5 10 20 30
#
# The script accepts one or more scales. The corresponding
# feature matrices are loaded and concatenated along axis=1.
#
# Example:
#
#   scale 10 -> (N, 24)
#   scale 20 -> (N, 24)
#   combined -> (N, 48)
#
# ============================================================

if len(sys.argv) < 2:
    raise ValueError(
        "At least one scale must be provided.\n"
        "Examples:\n"
        "  python multi_class_multiscale_corrected.py 10 20\n"
        "  python multi_class_multiscale_corrected.py 1 10 20"
    )

# All command-line arguments except the final classifier
# are interpreted as scales. The classifier is specified
# with --classifier or -c.
#
# Examples:
#   python multi_class_multiscale_corrected.py 10 20
#       -> scales 10,20; default SVM
#
#   python multi_class_multiscale_corrected.py 10 20 --classifier 2
#       -> scales 10,20; Random Forest
#
#   python multi_class_multiscale_corrected.py 1 10 20 -c 3
#       -> scales 1,10,20; SVM-RBF
#
#   python multi_class_multiscale_corrected.py 20 -c 4
#       -> scale 20; ANN

args = sys.argv[1:]

if "--classifier" in args:
    classifier_pos = args.index("--classifier")
    if classifier_pos + 1 >= len(args):
        raise ValueError("--classifier requires an integer from 0 to 4.")
    classifier = int(args[classifier_pos + 1])
    scale_args = args[:classifier_pos]
elif "-c" in args:
    classifier_pos = args.index("-c")
    if classifier_pos + 1 >= len(args):
        raise ValueError("-c requires an integer from 0 to 4.")
    classifier = int(args[classifier_pos + 1])
    scale_args = args[:classifier_pos]
else:
    classifier = 1  # Default: SVM-RBF
    scale_args = args

try:
    scales = [int(s) for s in scale_args]
except ValueError:
    raise ValueError(
        "All scale arguments must be positive integers."
    )

if len(scales) < 1:
    raise ValueError(
        "At least one scale must be provided."
    )

if any(s <= 0 for s in scales):
    raise ValueError(
        "All scales must be positive integers."
    )

if len(set(scales)) != len(scales):
    raise ValueError(
        f"Duplicate scales are not allowed: {scales}"
    )

# Preserve the order supplied by the user.
scales = list(scales)

print("=" * 70)
print("Multiscale RMEn classification")
print(f"Selected scales: {scales}")
print(f"Number of scales: {len(scales)}")
print("=" * 70)

# ============================================================
# Load feature matrices
# ============================================================

feature_matrices = []

for scale in scales:

    feature_file = f"Data_S_msrec_scale_{scale}.npy"

    try:
        data_scale = np.load(feature_file)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Could not find '{feature_file}'. "
            f"Make sure the RMEn feature file for scale {scale} exists."
        )

    if data_scale.ndim != 2:
        raise ValueError(
            f"Feature array for scale {scale} must be two-dimensional, "
            f"but got shape {data_scale.shape}."
        )

    print(
        f"Scale {scale}: {feature_file} -> "
        f"{data_scale.shape}"
    )

    feature_matrices.append(data_scale)

# ============================================================
# Load labels
# ============================================================

data_in_L = np.load("Data_L_msrec_scale_10.npy")

# ============================================================
# Sanity checks
# ============================================================

n_samples = feature_matrices[0].shape[0]

for scale, data_scale in zip(scales, feature_matrices):

    if data_scale.shape[0] != n_samples:
        raise ValueError(
            f"Samples mismatch at scale {scale}: "
            f"expected {n_samples} rows but found "
            f"{data_scale.shape[0]}."
        )

    if not np.isfinite(data_scale).all():
        raise ValueError(
            f"Scale {scale} contains NaN or infinite values."
        )

if data_in_L.shape[0] != n_samples:
    raise ValueError(
        f"Samples mismatch: feature data has {n_samples} rows "
        f"but labels have {data_in_L.shape[0]} rows."
    )

# ============================================================
# Concatenate multiscale features
# ============================================================
#
# The feature blocks remain in the order supplied by the user.
#
# Example:
#   scales = [10, 20]
#   (N,24) + (N,24) -> (N,48)
#
#   scales = [1,10,20]
#   (N,24) + (N,24) + (N,24) -> (N,72)
# ============================================================

data_in_S = np.concatenate(
    feature_matrices,
    axis=1,
)

expected_features = sum(
    data.shape[1] for data in feature_matrices
)

if data_in_S.shape[1] != expected_features:
    raise RuntimeError(
        "Unexpected number of concatenated features."
    )

print(
    f"Combined feature matrix shape: "
    f"{data_in_S.shape}"
)

# ============================================================
# ============================================================
# Select classes of interest
# ============================================================

group_list = np.array([0, 1, 2, 3])

X = []
Y = []

for i in range(len(data_in_L)):
    if data_in_L[i] in group_list:
        Y.append(data_in_L[i])
        X.append(data_in_S[i])

X = np.asarray(X, dtype=np.float64)
Y = np.asarray(Y)

if len(X) == 0:
    raise ValueError("No samples found for the selected classes.")

# ------------------------------------------------------------
# Check for invalid feature values
# ------------------------------------------------------------

if not np.isfinite(X).all():
    raise ValueError("X contains NaN or infinite values.")

# ------------------------------------------------------------
# Shuffle once before cross-validation
# ------------------------------------------------------------
# This is not required by StratifiedKFold when shuffle=True,
# but preserves the reproducible behavior of the original code.

rng = np.random.default_rng(SEED)
idx = rng.permutation(len(X))

X = X[idx]
Y = Y[idx]

print(f"Number of patients: {len(X)}")
print(f"Number of features: {X.shape[1]}")
print(f"Classes: {np.unique(Y)}")

# ============================================================
# Class distribution
# ============================================================

print("\nClass distribution:")
for cls in np.unique(Y):
    n = np.sum(Y == cls)
    print(f"  Class {cls}: {n} patients ({100*n/len(Y):.2f}%)")

# ============================================================
# Choose classifier
# ============================================================
#
# 0 = KNN
# 1 = Decision Tree
# 2 = Random Forest
# 3 = SVM-RBF
# 4 = ANN
#
# The classifier can be supplied with:
#   --classifier 0
# or:
#   -c 0
#
# Default: SVM-RBF
# ============================================================

classifier_list = ["KNN", "Decision Tree", "Random Forest", "SVM-RBF", "ANN"]

if classifier < 0 or classifier >= len(classifier_list):
    raise ValueError(
        f"Invalid classifier index {classifier}. "
        f"Use 0=KNN, 1=Decision Tree, 2=Random Forest, 3=SVM-RBF, or 4=ANN."
    )

print(
    f"\nMulticlass Classification "
    f"({len(group_list)} classes) using {classifier_list[classifier]}."
)

# ============================================================
# Cross-validation
# ============================================================
#
# IMPORTANT:
# Each record corresponds to one independent patient.
# Therefore, patient-level grouping is not necessary.
#
# StratifiedKFold is used to preserve class proportions in
# each fold.
#
# Feature normalization is performed INSIDE each training fold.
# Therefore, the test fold never contributes to the mean/std
# used for feature scaling.
# ============================================================

cv = StratifiedKFold(
    n_splits=K,
    shuffle=True,
    random_state=SEED,
)

accuracy_scores = []
balanced_accuracy_scores = []
macro_f1_scores = []
macro_sensitivity_scores = []
macro_specificity_scores = []
macro_precision_scores = []

# Optional: store predictions for a global confusion-matrix
# analysis if desired later.
all_test_labels = []
all_predictions = []

# Confusion matrix accumulated across all held-out folds.
confusion_matrix_total = np.zeros(
    (len(group_list), len(group_list)), dtype=int
)

for fold, (train_idx, test_idx) in enumerate(cv.split(X, Y), start=1):

    print(f"\nProcessing fold #{fold}")

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = Y[train_idx]
    y_test = Y[test_idx]

    # --------------------------------------------------------
    # Build classifier
    # --------------------------------------------------------

    if classifier == 0:
        # KNN is distance-based, so standardization is required.
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=10)),
        ])

    elif classifier == 1:
        # Decision trees do not require feature scaling.
        model = DecisionTreeClassifier(
            random_state=SEED
        )

    elif classifier == 2:
        # Random forests do not require feature scaling.
        model = RandomForestClassifier(
            random_state=SEED
        )

    elif classifier == 3:
        # RBF-SVM is sensitive to feature scale.
        # StandardScaler is fitted ONLY on X_train.
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf",
                class_weight="balanced",
            )),
        ])

    elif classifier == 4:
        # ANN/MLP implemented using sklearn so that the same
        # cross-validation protocol can be used consistently.
        #
        # StandardScaler is fitted only on the training fold.
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", MLPClassifier(
                hidden_layer_sizes=(64, 64),
                activation="relu",
                solver="adam",
                batch_size=256,
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=SEED,
            )),
        ])

    # --------------------------------------------------------
    # Train using ONLY the training fold
    # --------------------------------------------------------

    model.fit(X_train, y_train)

    # --------------------------------------------------------
    # Evaluate ONLY on the held-out test fold
    # --------------------------------------------------------

    prediction = model.predict(X_test)

    acc = accuracy_score(y_test, prediction)

    balanced_acc = balanced_accuracy_score(
        y_test,
        prediction
    )

    macro_f1 = f1_score(
        y_test,
        prediction,
        average="macro",
        zero_division=0,
    )

    # Sensitivity (recall) per class and macro-average.
    # For multiclass classification, each class is evaluated
    # against all remaining classes (one-vs-rest).
    sensitivity_per_class = recall_score(
        y_test,
        prediction,
        labels=group_list,
        average=None,
        zero_division=0,
    )

    macro_sensitivity = recall_score(
        y_test,
        prediction,
        labels=group_list,
        average="macro",
        zero_division=0,
    )

    # Specificity per class:
    # TN / (TN + FP), computed from the multiclass
    # confusion matrix using a one-vs-rest interpretation.
    cm_fold = confusion_matrix(
        y_test,
        prediction,
        labels=group_list,
    )

    specificity_per_class = []

    for c in range(len(group_list)):
        tp = cm_fold[c, c]
        fn = np.sum(cm_fold[c, :]) - tp
        fp = np.sum(cm_fold[:, c]) - tp
        tn = np.sum(cm_fold) - tp - fn - fp

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else 0.0
        )

        specificity_per_class.append(specificity)

    specificity_per_class = np.asarray(specificity_per_class)
    macro_specificity = np.mean(specificity_per_class)

    # Macro precision is also retained as an optional diagnostic.
    macro_precision = precision_score(
        y_test,
        prediction,
        labels=group_list,
        average="macro",
        zero_division=0,
    )

    accuracy_scores.append(acc)
    balanced_accuracy_scores.append(balanced_acc)
    macro_f1_scores.append(macro_f1)
    macro_sensitivity_scores.append(macro_sensitivity)
    macro_specificity_scores.append(macro_specificity)
    macro_precision_scores.append(macro_precision)

    all_test_labels.extend(y_test)
    all_predictions.extend(prediction)

    # Accumulate the confusion matrices from the held-out folds.
    confusion_matrix_total += cm_fold

    print(f"  ACC                = {acc:.4f}")
    print(f"  Balanced Accuracy  = {balanced_acc:.4f}")
    print(f"  Macro-F1           = {macro_f1:.4f}")
    print(f"  Sensitivity (macro)= {macro_sensitivity:.4f}")
    print(f"  Specificity (macro)= {macro_specificity:.4f}")

# ============================================================
# Final cross-validation results
# ============================================================

accuracy_scores = np.asarray(accuracy_scores)
balanced_accuracy_scores = np.asarray(balanced_accuracy_scores)
macro_f1_scores = np.asarray(macro_f1_scores)
macro_sensitivity_scores = np.asarray(macro_sensitivity_scores)
macro_specificity_scores = np.asarray(macro_specificity_scores)
macro_precision_scores = np.asarray(macro_precision_scores)

print("\n" + "=" * 70)
print(f"Classifier: {classifier_list[classifier]}")
print(f"Scales: {scales}")
print("Validation: Stratified 10-fold cross-validation")
print("=" * 70)

print(
    f"ACC                 = "
    f"{np.mean(accuracy_scores) * 100:.2f} "
    f"+/- {np.std(accuracy_scores) * 100:.2f}%"
)

print(
    f"Balanced Accuracy   = "
    f"{np.mean(balanced_accuracy_scores) * 100:.2f} "
    f"+/- {np.std(balanced_accuracy_scores) * 100:.2f}%"
)

print(
    f"Macro-F1            = "
    f"{np.mean(macro_f1_scores) * 100:.2f} "
    f"+/- {np.std(macro_f1_scores) * 100:.2f}%"
)

print(
    f"Sensitivity (macro) = "
    f"{np.mean(macro_sensitivity_scores) * 100:.2f} "
    f"+/- {np.std(macro_sensitivity_scores) * 100:.2f}%"
)

print(
    f"Specificity (macro) = "
    f"{np.mean(macro_specificity_scores) * 100:.2f} "
    f"+/- {np.std(macro_specificity_scores) * 100:.2f}%"
)

print(
    f"Precision (macro)   = "
    f"{np.mean(macro_precision_scores) * 100:.2f} "
    f"+/- {np.std(macro_precision_scores) * 100:.2f}%"
)

# ------------------------------------------------------------
# Per-class metrics from the aggregated held-out predictions
# ------------------------------------------------------------
#
# These predictions come exclusively from the held-out fold
# in which each patient was not used for model fitting.
# ------------------------------------------------------------

all_test_labels = np.asarray(all_test_labels)
all_predictions = np.asarray(all_predictions)

cm_total = confusion_matrix(
    all_test_labels,
    all_predictions,
    labels=group_list,
)

sensitivity_total = recall_score(
    all_test_labels,
    all_predictions,
    labels=group_list,
    average=None,
    zero_division=0,
)

precision_total = precision_score(
    all_test_labels,
    all_predictions,
    labels=group_list,
    average=None,
    zero_division=0,
)

specificity_total = []

for c in range(len(group_list)):
    tp = cm_total[c, c]
    fn = np.sum(cm_total[c, :]) - tp
    fp = np.sum(cm_total[:, c]) - tp
    tn = np.sum(cm_total) - tp - fn - fp

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    specificity_total.append(specificity)

specificity_total = np.asarray(specificity_total)

print("\nPer-class performance (aggregated held-out predictions):")
print(
    f"{'Class':>8} "
    f"{'Sensitivity':>15} "
    f"{'Specificity':>15} "
    f"{'Precision':>15}"
)

for i, cls in enumerate(group_list):
    print(
        f"{str(cls):>8} "
        f"{sensitivity_total[i] * 100:>14.2f}% "
        f"{specificity_total[i] * 100:>14.2f}% "
        f"{precision_total[i] * 100:>14.2f}%"
    )

print("\nAggregated confusion matrix:")
print("Rows = true class; columns = predicted class")
print(cm_total)

print("=" * 70)
