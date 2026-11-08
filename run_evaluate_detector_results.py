import numpy as np
import pandas as pd
import argparse
import os


METHODS = ['MDRS', 'TRAKR', 'SPE', 'HST', 'RZS', 'MTAD','TranAD']

# NAB-like window settings
ANOMALY_WINDOW_MODE = 'NAB'
NORMALIZE_METRICS = True



def adjust_predicts(pred, label):
    """
    Point-adjusted evaluation.

    If an anomaly segment contains at least one detected point,
    the whole ground-truth anomaly segment is considered detected.
    """
    pred = pred.copy()
    anomaly_state = False

    for i in range(len(label)):
        if label[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True

            for j in range(i, 0, -1):
                if label[j] == 0:
                    break
                pred[j] = 1

        elif label[i] == 0:
            anomaly_state = False

        if anomaly_state:
            pred[i] = 1

    return pred


def compute_confusion_matrix(pred, labels):
    tp = np.sum((pred == 1) & (labels == 1))
    tn = np.sum((pred == 0) & (labels == 0))
    fp = np.sum((pred == 1) & (labels == 0))
    fn = np.sum((pred == 0) & (labels == 1))

    return tp, tn, fp, fn


def calculate_metrics(tp, tn, fp, fn):

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    accuracy = (
        (tp + tn) / (tp + tn + fp + fn)
        if (tp + tn + fp + fn) > 0
        else 0.0
    )

    denom = np.sqrt(
        (tp + fp) *
        (tp + fn) *
        (tn + fp) *
        (tn + fn)
    )

    mcc = (
        (tp * tn - fp * fn) / denom
        if denom > 0
        else 0.0
    )

    return precision, recall, f1, accuracy, mcc


def compute_metrics_at_threshold(
    scores,
    labels,
    threshold,
    point_adjust=False
):

    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)

    pred = (scores >= threshold).astype(int)

    if point_adjust:
        pred = adjust_predicts(pred, labels)

    tp, tn, fp, fn = compute_confusion_matrix(pred, labels)

    return calculate_metrics(tp, tn, fp, fn)


def compute_best_f1(
    scores,
    labels,
    point_adjust=False,
    n_thresholds=200
):

    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)

    percentiles = np.linspace(0, 100, n_thresholds)
    thresholds = np.unique(
        np.percentile(scores, percentiles)
    )

    best_f1 = -1.0
    best_threshold = (
        thresholds[0]
        if len(thresholds) > 0
        else 0.0
    )

    best_precision = 0.0
    best_recall = 0.0

    for th in thresholds:

        precision, recall, f1, _, _ = (
            compute_metrics_at_threshold(
                scores,
                labels,
                th,
                point_adjust=point_adjust
            )
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = th
            best_precision = precision
            best_recall = recall

    return (
        best_threshold,
        best_f1,
        best_precision,
        best_recall
    )


# ============================================================
# WINDOW-BASED EVALUATION
# ============================================================

def calculate_anomaly_window_size(
    dataset_length,
    num_anomalies
):
    """
    NAB-style anomaly window size:

        ceil(0.1 * dataset_length / num_anomalies)
    """

    if num_anomalies > 0:
        return int(
            np.ceil(
                0.02 * dataset_length / num_anomalies
            )
        )

    return int(
        np.ceil(0.1 * dataset_length)
    )


def create_anomaly_windows(
    labels,
    anomaly_window_size
):
    """
    Creates anomaly windows around every ground-truth
    anomaly point.

    Each separate anomaly event receives a unique ID.
    """

    labels = np.asarray(labels).astype(int)

    length = len(labels)

    anomaly_windows = np.zeros(
        length,
        dtype=int
    )

    # Find contiguous anomaly segments
    anomaly_segments = []

    in_anomaly = False
    start = None

    for i, label in enumerate(labels):

        if label == 1 and not in_anomaly:
            start = i
            in_anomaly = True

        elif label == 0 and in_anomaly:
            anomaly_segments.append(
                (start, i - 1)
            )
            in_anomaly = False

    if in_anomaly:
        anomaly_segments.append(
            (start, length - 1)
        )

    # Create one window around each anomaly segment
    for anomaly_id, (start, end) in enumerate(
        anomaly_segments,
        start=1
    ):

        window_start = max(
            0,
            start - anomaly_window_size
        )

        window_end = min(
            length,
            end + anomaly_window_size + 1
        )

        anomaly_windows[
            window_start:window_end
        ] = anomaly_id

    return anomaly_windows


def adjust_anomaly_signals_posedge(
    detections
):
    """
    Convert consecutive anomaly detections into
    a single detection event.

    Example:

        0 0 1 1 1 1 0

    becomes:

        0 0 1 0 0 0 0
    """

    detections = np.asarray(
        detections,
        dtype=bool
    )

    adjusted = np.zeros(
        len(detections),
        dtype=bool
    )

    for i in range(1, len(detections)):
        adjusted[i] = (
            detections[i]
            and not detections[i - 1]
        )

    return adjusted


def measure_window_confusion_matrix(
    detections,
    anomaly_windows,
    anomaly_window_size,
    normalize=True
):
    """
    NAB-like event/window evaluation.

    A ground-truth anomaly window counts as TP if
    at least one detection occurs inside that window.

    Multiple detections inside the same window count
    as only one TP.
    """

    detections = np.asarray(
        detections,
        dtype=bool
    )

    anomaly_windows = np.asarray(
        anomaly_windows,
        dtype=int
    )

    tp = 0
    fp = 0
    fn = 0

    detected_windows = set()

    # Detect TP / FP
    for i in range(len(detections)):

        if not detections[i]:
            continue

        window_id = anomaly_windows[i]

        if window_id > 0:

            # First detection in this anomaly window
            if window_id not in detected_windows:
                tp += 1
                detected_windows.add(window_id)

        else:
            fp += 1

    # Any anomaly window without detection = FN
    anomaly_ids = set(
        anomaly_windows[
            anomaly_windows > 0
        ]
    )

    for anomaly_id in anomaly_ids:

        if anomaly_id not in detected_windows:
            fn += 1

    if normalize:

        fp = fp / (
            2 * anomaly_window_size + 1
        )

        tn = (
            len(detections)
            / (2 * anomaly_window_size + 1)
            - tp
            - fp
            - fn
        )

    else:

        tn = (
            len(detections)
            - tp
            - fp
            - fn
        )

    return tp, tn, fp, fn


def compute_window_metrics(
    scores,
    labels,
    threshold,
    normalize=True
):

    scores = np.asarray(scores).astype(float)
    labels = np.asarray(labels).astype(int)

    detections = (
        scores >= threshold
    )

    # Convert consecutive detections into events
    detections = adjust_anomaly_signals_posedge(
        detections
    )

    # Number of actual anomaly events
    num_anomalies = 0
    in_anomaly = False

    for label in labels:

        if label == 1 and not in_anomaly:
            num_anomalies += 1
            in_anomaly = True

        elif label == 0:
            in_anomaly = False

    anomaly_window_size = (
        calculate_anomaly_window_size(
            len(labels),
            num_anomalies
        )
    )

    anomaly_windows = create_anomaly_windows(
        labels,
        anomaly_window_size
    )

    tp, tn, fp, fn = (
        measure_window_confusion_matrix(
            detections,
            anomaly_windows,
            anomaly_window_size,
            normalize=normalize
        )
    )

    return calculate_metrics(
        tp,
        tn,
        fp,
        fn
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Evaluate detector results"
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        required=True
    )

    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=METHODS
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "pointwise",
            "point_adjust",
            "window"
        ],
        default="pointwise"
    )

    args = parser.parse_args()

    benchmark = args.benchmark
    methods = args.methods
    mode = args.mode

    benchmark_path = (
        f"benchmarks/{benchmark}"
    )

    folder_path = (
        f"{benchmark_path}/data"
    )

    result_path = (
        f"{benchmark_path}/results"
    )

    analysis_path = (
        f"{benchmark_path}/analysis"
    )

    os.makedirs(
        analysis_path,
        exist_ok=True
    )

    filename_list = os.listdir(
        f"{folder_path}/test_label"
    )

    results_list = []

    for filename in filename_list:

        anomaly_label = np.loadtxt(
            f"{folder_path}/test_label/{filename}"
        )

        filename_stem = os.path.splitext(
            filename
        )[0]

        for method in methods:

            score_path = (
                f"{result_path}/"
                f"{filename_stem}_{method}.txt"
            )

            if not os.path.exists(score_path):

                print(
                    f"{filename_stem} "
                    f"[{method}]: SKIPPED"
                )

                continue

            test_score = np.loadtxt(
                score_path
            )

            if len(test_score) != len(anomaly_label):

                print(
                    f"{filename_stem} "
                    f"[{method}]: ERROR!"
                )

                continue

            # ==================================================
            # POINT-WISE
            # ==================================================

            if mode == "pointwise":

                (
                    best_threshold,
                    best_f1,
                    best_precision,
                    best_recall
                ) = compute_best_f1(
                    test_score,
                    anomaly_label,
                    point_adjust=False
                )

                (
                    precision,
                    recall,
                    f1,
                    accuracy,
                    mcc
                ) = compute_metrics_at_threshold(
                    test_score,
                    anomaly_label,
                    best_threshold,
                    point_adjust=False
                )

            # ==================================================
            # POINT-ADJUST
            # ==================================================

            elif mode == "point_adjust":

                (
                    best_threshold,
                    best_f1,
                    best_precision,
                    best_recall
                ) = compute_best_f1(
                    test_score,
                    anomaly_label,
                    point_adjust=True
                )

                (
                    precision,
                    recall,
                    f1,
                    accuracy,
                    mcc
                ) = compute_metrics_at_threshold(
                    test_score,
                    anomaly_label,
                    best_threshold,
                    point_adjust=True
                )

            # ==================================================
            # WINDOW
            # ==================================================

            elif mode == "window":

                (
                    best_threshold,
                    _,
                    _,
                    _
                ) = compute_best_f1(
                    test_score,
                    anomaly_label,
                    point_adjust=False
                )

                (
                    precision,
                    recall,
                    f1,
                    accuracy,
                    mcc
                ) = compute_window_metrics(
                    test_score,
                    anomaly_label,
                    best_threshold,
                    normalize=NORMALIZE_METRICS
                )

            print(
                f"{filename_stem} [{method}] "
                f"mode={mode} "
                f"threshold={best_threshold:.6f} "
                f"F1={f1:.4f} "
                f"precision={precision:.4f} "
                f"recall={recall:.4f} "
                f"accuracy={accuracy:.4f} "
                f"mcc={mcc:.4f}"
            )

            results_list.append({

                "Dataset": filename_stem,

                "Algorithm": method,

                "Evaluation": mode,

                "Threshold": best_threshold,

                "Precision": precision,

                "Recall": recall,

                "F-score": f1,

                "Accuracy": accuracy,

                "MCC": mcc
            })

    if len(results_list) > 0:

        results_df = pd.DataFrame(
            results_list
        )

        results_df = results_df.sort_values(
            by=[
                "Dataset",
                "Algorithm"
            ]
        )

        out_path = (
            f"{analysis_path}/"
            f"benchmark_results_{mode}.csv"
        )

        results_df.to_csv(
            out_path,
            index=False
        )

        print(
            f"\nResults saved here: "
            f"{out_path}"
        )

    else:

        print(
            f"\nCan't save results for "
            f"'{benchmark}' benchmark."
        )