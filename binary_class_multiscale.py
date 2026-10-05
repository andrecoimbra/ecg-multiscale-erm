import numpy as np
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SEED = 1001
N_SPLITS = 10

CLASSIFIER_NAMES = {
    0: "Random Forest",
    1: "SVM-RBF",
    2: "ANN (MLP)",
}


def parse_arguments():
    """
    Usage:
        python binary_class_multiscale_corrected.py <group> <scale1> [scale2 ...] [-c 0|1|2]

    Examples:
        python binary_class_multiscale_corrected.py 1 10
        python binary_class_multiscale_corrected.py 1 20
        python binary_class_multiscale_corrected.py 1 10 20
        python binary_class_multiscale_corrected.py 1 1 10 20
        python binary_class_multiscale_corrected.py 1 10 20 -c 0
        python binary_class_multiscale_corrected.py 1 10 20 --classifier 1
    """

    if len(sys.argv) < 3:
        print(
            "Usage: python binary_class_multiscale_corrected.py "
            "<group> <scale1> [scale2 ...] [-c 0|1|2]"
        )
        print("\nExamples:")
        print("  python binary_class_multiscale_corrected.py 1 10")
        print("  python binary_class_multiscale_corrected.py 1 20")
        print("  python binary_class_multiscale_corrected.py 1 10 20")
        print("  python binary_class_multiscale_corrected.py 1 1 10 20")
        print("  python binary_class_multiscale_corrected.py 1 10 20 -c 0")
        print("  python binary_class_multiscale_corrected.py 1 10 20 -c 1")
        print("\nClassifier: 0 = Random Forest, 1 = SVM-RBF, 2 = ANN (MLP)")
        sys.exit(1)

    group = int(sys.argv[1])
    if group < 1 or group > 10:
        raise ValueError("Group must be an integer from 1 to 10.")

    classifier = 1  # default: SVM-RBF
    scales = []

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] in ("-c", "--classifier"):
            if i + 1 >= len(sys.argv):
                raise ValueError("Missing classifier code.")
            classifier = int(sys.argv[i + 1])
            i += 2
        else:
            scales.append(int(sys.argv[i]))
            i += 1

    if len(scales) < 1:
        raise ValueError("At least one scale must be provided.")

    if any(scale < 1 for scale in scales):
        raise ValueError("All scales must be positive integers.")

    if len(set(scales)) != len(scales):
        raise ValueError("Scales must be unique.")

    if classifier not in CLASSIFIER_NAMES:
        raise ValueError("Classifier must be 0 (Random Forest), 1 (SVM-RBF), or 2 (ANN).")

    return group, scales, classifier


def load_features(scales):
    """Load and concatenate the feature matrices for the requested scales."""

    feature_blocks = []

    for scale in scales:
        filename = f"Data_S_msrec_scale_{scale}.npy"

        try:
            data_scale = np.load(filename)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find '{filename}'. "
                f"Make sure the feature file for scale {scale} exists."
            )

        if data_scale.ndim != 2:
            raise ValueError(
                f"{filename} must be a 2-D array; "
                f"found shape {data_scale.shape}."
            )

        if not np.all(np.isfinite(data_scale)):
            raise ValueError(f"{filename} contains NaN or infinite values.")

        feature_blocks.append(data_scale)

        print(
            f"Scale {scale}: shape={data_scale.shape}, "
            f"features={data_scale.shape[1]}"
        )

    n_samples = feature_blocks[0].shape[0]

    for scale, data_scale in zip(scales, feature_blocks):
        if data_scale.shape[0] != n_samples:
            raise ValueError(
                f"Number of samples mismatch for scale {scale}: "
                f"expected {n_samples}, found {data_scale.shape[0]}."
            )

    X = np.concatenate(feature_blocks, axis=1)

    print(f"Selected scales: {scales}")
    print(f"Data_in_S shape: {X.shape}")

    return X


def load_labels(n_samples):
    """
    Load diagnostic labels.

    Labels are shared by all scales. The original script used the
    scale-20 label file; scale 10 is used here consistently with the
    corrected multiclass pipeline.
    """

    label_file = "Data_L_msrec_scale_10.npy"

    try:
        y = np.load(label_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find '{label_file}'.")

    y = np.asarray(y).reshape(-1)

    if len(y) != n_samples:
        raise ValueError(
            f"Samples mismatch: feature matrix has {n_samples} rows, "
            f"but labels have {len(y)} samples."
        )

    return y


def build_classifier(classifier):
    """
    Build the classifier.

    SVM:
        StandardScaler is inside the Pipeline. Therefore, mean and
        standard deviation are fitted only on the training partition
        of each fold.

    Random Forest:
        No scaling is applied because tree-based models do not require it.

    ANN:
        Multilayer perceptron with feature standardization. Scaling is
        fitted exclusively on the training partition of each fold.
    """

    if classifier == 0:
        return RandomForestClassifier(
            random_state=SEED,
            class_weight="balanced",
        )

    if classifier == 1:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        class_weight="balanced",
                    ),
                ),
            ]
        )

    if classifier == 2:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    MLPClassifier(
                        hidden_layer_sizes=(100,),
                        activation="relu",
                        solver="adam",
                        alpha=0.0001,
                        batch_size="auto",
                        learning_rate="constant",
                        learning_rate_init=0.001,
                        max_iter=1000,
                        random_state=SEED,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=20,
                    ),
                ),
            ]
        )

    raise ValueError("Unknown classifier.")


def binary_specificity(y_true, y_pred, positive_label):
    """
    Compute specificity for a binary class in one-vs-rest form.
    """

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    if positive_label == 0:
        tp = cm[0, 0]
        fn = cm[0, 1]
        fp = cm[1, 0]
        tn = cm[1, 1]
    else:
        tp = cm[1, 1]
        fn = cm[1, 0]
        fp = cm[0, 1]
        tn = cm[0, 0]

    denominator = tn + fp

    if denominator == 0:
        return np.nan

    return tn / denominator


def main():

    # --------------------------------------------------------
    # Arguments
    # --------------------------------------------------------

    G, scales, classifier = parse_arguments()

    diag_list = np.array(
        [
            "SR",
            "SB",
            "AFIB",
            "ST",
            "SVT",
            "AF",
            "SI",
            "AT",
            "AVNRT",
            "AVRT",
            "SAAWR",
        ]
    )

    # --------------------------------------------------------
    # Load multiscale features
    # --------------------------------------------------------

    data_in_S = load_features(scales)

    # --------------------------------------------------------
    # Load labels
    # --------------------------------------------------------

    data_in_L = load_labels(data_in_S.shape[0])

    # --------------------------------------------------------
    # Select SR (0) vs target group G
    # --------------------------------------------------------

    group_list = np.array([0, G])
    mask = np.isin(data_in_L, group_list)

    X = data_in_S[mask]
    y_original = data_in_L[mask]

    # Binary labels:
    #   0 = SR
    #   1 = target group
    Y = (y_original == G).astype(int)

    if len(X) == 0:
        raise ValueError(
            f"No samples found for SR (0) and group {G}."
        )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    class_counts = np.bincount(Y, minlength=2)

    print(f"\nBinary classification: {diag_list[0]} vs {diag_list[G]}")
    print(f"Class 0 ({diag_list[0]}): {class_counts[0]} samples")
    print(f"Class 1 ({diag_list[G]}): {class_counts[1]} samples")
    print(f"Total samples: {len(Y)}")
    print(f"Number of features: {X.shape[1]}")

    if np.min(class_counts) < N_SPLITS:
        raise ValueError(
            f"Each class must contain at least {N_SPLITS} samples for "
            f"stratified {N_SPLITS}-fold cross-validation. "
            f"Current counts: {class_counts.tolist()}"
        )

    # --------------------------------------------------------
    # Shuffle while preserving X/Y alignment
    # --------------------------------------------------------

    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(X))

    X = X[indices]
    Y = Y[indices]

    # --------------------------------------------------------
    # Stratified 10-fold cross-validation
    # --------------------------------------------------------

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED,
    )

    fold_metrics = []

    # Predictions are collected only from held-out folds.
    all_test_labels = []
    all_test_predictions = []

    for fold, (train_idx, test_idx) in enumerate(
        skf.split(X, Y), start=1
    ):

        print(f"\nProcessing fold {fold}/{N_SPLITS}")

        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = Y[train_idx]
        y_test = Y[test_idx]

        model = build_classifier(classifier)

        # SVM scaling is fitted exclusively on X_train because the
        # StandardScaler is part of the Pipeline.
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # ----------------------------------------------------
        # Metrics for this fold
        # ----------------------------------------------------

        accuracy = accuracy_score(y_test, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

        macro_f1 = f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )

        sensitivity = recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )

        specificity_0 = binary_specificity(
            y_test, y_pred, positive_label=0
        )
        specificity_1 = binary_specificity(
            y_test, y_pred, positive_label=1
        )
        specificity = np.nanmean([specificity_0, specificity_1])

        precision = precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )

        fold_result = {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "macro_f1": macro_f1,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
        }

        fold_metrics.append(fold_result)

        all_test_labels.extend(y_test)
        all_test_predictions.extend(y_pred)

        print(
            f"Fold {fold}: "
            f"ACC={accuracy:.4f} | "
            f"BA={balanced_accuracy:.4f} | "
            f"Macro-F1={macro_f1:.4f} | "
            f"Sens={sensitivity:.4f} | "
            f"Spec={specificity:.4f} | "
            f"Prec={precision:.4f}"
        )

    # --------------------------------------------------------
    # Mean +/- standard deviation across folds
    # --------------------------------------------------------

    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "sensitivity",
        "specificity",
        "precision",
    ]

    mean_metrics = {}
    std_metrics = {}

    for metric in metric_names:
        values = np.array(
            [result[metric] for result in fold_metrics],
            dtype=float,
        )

        mean_metrics[metric] = np.nanmean(values)
        std_metrics[metric] = np.nanstd(values)

    # --------------------------------------------------------
    # Aggregated confusion matrix from held-out predictions
    # --------------------------------------------------------

    all_test_labels = np.asarray(all_test_labels)
    all_test_predictions = np.asarray(all_test_predictions)

    aggregated_cm = confusion_matrix(
        all_test_labels,
        all_test_predictions,
        labels=[0, 1],
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    scale_label = ", ".join(str(scale) for scale in scales)

    print("\n" + "=" * 72)
    print(
        f"SR - {diag_list[G]} | "
        f"Scales: [{scale_label}] | "
        f"Classifier: {CLASSIFIER_NAMES[classifier]}"
    )
    print("=" * 72)

    print(
        f"Accuracy:          "
        f"{mean_metrics['accuracy'] * 100:.2f}% "
        f"+/- {std_metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Balanced Accuracy: "
        f"{mean_metrics['balanced_accuracy'] * 100:.2f}% "
        f"+/- {std_metrics['balanced_accuracy'] * 100:.2f}%"
    )

    print(
        f"Macro-F1:          "
        f"{mean_metrics['macro_f1'] * 100:.2f}% "
        f"+/- {std_metrics['macro_f1'] * 100:.2f}%"
    )

    print(
        f"Sensitivity:       "
        f"{mean_metrics['sensitivity'] * 100:.2f}% "
        f"+/- {std_metrics['sensitivity'] * 100:.2f}%"
    )

    print(
        f"Specificity:       "
        f"{mean_metrics['specificity'] * 100:.2f}% "
        f"+/- {std_metrics['specificity'] * 100:.2f}%"
    )

    print(
        f"Precision:         "
        f"{mean_metrics['precision'] * 100:.2f}% "
        f"+/- {std_metrics['precision'] * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Per-class metrics from aggregated held-out predictions
    # --------------------------------------------------------

    aggregated_sensitivity = recall_score(
        all_test_labels,
        all_test_predictions,
        labels=[0, 1],
        average=None,
        zero_division=0,
    )

    aggregated_precision = precision_score(
        all_test_labels,
        all_test_predictions,
        labels=[0, 1],
        average=None,
        zero_division=0,
    )

    aggregated_specificity = [
        binary_specificity(
            all_test_labels,
            all_test_predictions,
            positive_label=0,
        ),
        binary_specificity(
            all_test_labels,
            all_test_predictions,
            positive_label=1,
        ),
    ]

    print("\nPer-class metrics from aggregated held-out predictions:")
    print("-" * 72)

    for c, class_name in enumerate(
        [diag_list[0], diag_list[G]]
    ):
        print(
            f"{class_name:8s} | "
            f"Sens={aggregated_sensitivity[c] * 100:.2f}% | "
            f"Spec={aggregated_specificity[c] * 100:.2f}% | "
            f"Prec={aggregated_precision[c] * 100:.2f}%"
        )

    # --------------------------------------------------------
    # Aggregated confusion matrix
    # --------------------------------------------------------

    print("\nAggregated confusion matrix:")
    print("Rows = true labels, columns = predicted labels")

    print(
        f"             Pred {diag_list[0]:>6s}  "
        f"Pred {diag_list[G]:>6s}"
    )

    print(
        f"True {diag_list[0]:>6s}   "
        f"{aggregated_cm[0, 0]:>8d}  "
        f"{aggregated_cm[0, 1]:>8d}"
    )

    print(
        f"True {diag_list[G]:>6s}   "
        f"{aggregated_cm[1, 0]:>8d}  "
        f"{aggregated_cm[1, 1]:>8d}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
