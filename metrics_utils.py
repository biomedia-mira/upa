import numpy as np
from sklearn import metrics, utils
import pandas as pd

COLS_FOR_RESULTS_W_BOOTSTRAP = [
    "ROCAUC",
    "ROCAUC-boot",
    "vendOP",
    "vendOP-boot",
    "vendSens/Spec",
    "GlobalOP",
    "globalSens/Spec",
    "globalSens/Spec - boot",
    "Global Spec/Sens Diff",
]

COLS_FOR_RESULTS_SIMPLE = [
    "ROCAUC",
    "vendOP",
    "vendSens/Spec",
    "GlobalOP",
    "globalSens/Spec",
    "Global Spec/Sens Diff",
    "Global Youden",
]


def get_op_threshold(true, pred, operating_point="diag"):
    fpr, tpr, threshold = metrics.roc_curve(true, pred)
    if operating_point == "diag":
        op = np.argmin(np.abs(tpr - (1 - fpr)))
    elif operating_point == "spec90":
        op = np.argmin(np.abs(fpr - 0.10))
    else:
        raise ValueError("Operating point has to be diag, spec90")
    return threshold[op], 1 - fpr[op], tpr[op]


def all_metrics(true, pred, global_op, return_bootstrap=True, operating_point="diag"):
    roc = metrics.roc_auc_score(true, pred)
    thres, spec, sens = get_op_threshold(true, pred, operating_point)
    global_sens, global_spec = get_sens_spec_at_threshold(true, pred, global_op)
    global_youden = global_sens + global_spec - 1
    if return_bootstrap:
        roc_b, sens_b, spec_b, thres_b, spec_sens_abs_diff_b = (
            np.zeros(500),
            np.zeros(500),
            np.zeros(500),
            np.zeros(500),
            np.zeros(500),
        )
        for b in range(500):
            true_b, pred_b = utils.resample(true, pred, stratify=true)
            roc_b[b] = metrics.roc_auc_score(true_b, pred_b)
            sens_b[b], spec_b[b] = get_sens_spec_at_threshold(true_b, pred_b, global_op)
            spec_sens_abs_diff_b[b] = spec_b[b] - sens_b[b]
            thres_b[b] = get_op_threshold(
                true_b, pred_b, operating_point=operating_point
            )[0]
        roc_i, sens_i, spec_i, thres_i, spec_sens_diff_i = (
            np.percentile(roc_b, [2.5, 97.5]),
            np.percentile(sens_b, [2.5, 97.5]),
            np.percentile(spec_b, [2.5, 97.5]),
            np.percentile(thres_b, [2.5, 97.5]),
            np.percentile(spec_sens_abs_diff_b, [2.5, 97.5]),
        )
        return {
            "roc_ic": roc_i,
            "roc": roc,
            "vendor_threshold": thres,
            "vendor_threshold_ic": thres_i,
            "vendor_sens": sens,
            "vendor_spec": spec,
            "global_op": global_op,
            "global_sens": global_sens,
            "global_spec": global_spec,
            "sens_ic": sens_i,
            "spec_ic": spec_i,
            "spec_sens_diff_ic": spec_sens_diff_i,
        }
    return {
        "roc": roc,
        "vendor_threshold": thres,
        "vendor_sens": sens,
        "vendor_spec": spec,
        "global_op": global_op,
        "global_sens": global_sens,
        "global_spec": global_spec,
        "spec_sens_diff": global_spec - global_sens,
        "global_youden": global_youden,
    }


def all_metrics_as_str(true, pred, global_op, bootstrap=True, operating_point="diag"):
    metrics_dict = all_metrics(true, pred, global_op, bootstrap, operating_point)
    if bootstrap:
        return [
            metrics_dict["roc"],
            f"[{metrics_dict['roc_ic'][0]:.3f};{metrics_dict['roc_ic'][1]:.3f}]",
            metrics_dict["vendor_threshold"],
            f"[{metrics_dict['vendor_threshold_ic'][0]:.3f};{metrics_dict['vendor_threshold_ic'][1]:.3f}]",
            f"{metrics_dict['vendor_sens']:.3f}/{metrics_dict['vendor_spec']:.3f}",
            metrics_dict["global_op"],
            f"{metrics_dict['global_sens']:.3f}/{metrics_dict['global_spec']:.3f}",
            f"[{metrics_dict['sens_ic'][0]:.3f};{metrics_dict['sens_ic'][1]:.3f}] / [{metrics_dict['spec_ic'][0]:.3f};{metrics_dict['spec_ic'][1]:.3f}]",
            f"[{metrics_dict['spec_sens_diff_ic'][0]:.3f};{metrics_dict['spec_sens_diff_ic'][1]:.3f}]",
        ]
    return [
        metrics_dict["roc"],
        metrics_dict["vendor_threshold"],
        f"{metrics_dict['vendor_sens']:.3f}/{metrics_dict['vendor_spec']:.3f}",
        metrics_dict["global_op"],
        f"{metrics_dict['global_sens']:.3f}/{metrics_dict['global_spec']:.3f}",
        f"{metrics_dict['spec_sens_diff']:.3f}",
        f"{metrics_dict['global_youden']:.3f}",
    ]


def get_summary_df_from_preds_df(
    list_dfs,
    global_threshold,
    bootstrap=True,
    label_column="malignant",
    operating_point="diag",
):
    results = {}
    for df in list_dfs:
        results[str(df.vendor.unique())] = all_metrics_as_str(
            df[label_column].values,
            df.pred1.values,
            global_threshold,
            bootstrap,
            operating_point,
        )
    if bootstrap:
        return pd.DataFrame.from_dict(
            results, orient="index", columns=COLS_FOR_RESULTS_W_BOOTSTRAP
        )
    return pd.DataFrame.from_dict(
        results, orient="index", columns=COLS_FOR_RESULTS_SIMPLE
    )


def get_all_dfs_results(model_dir, datasets_to_fetch):
    all_dfs = {}
    for ds in datasets_to_fetch:
        df = pd.read_csv(model_dir / f"{ds}.csv")
        df["vendor"] = ds
        all_dfs[ds] = df
        print(len(df))
    return all_dfs


def get_all_camelyon_results(model_dir):
    all_dfs = {}
    for split in ["id_val", "val", "test"]:
        df = pd.read_csv(model_dir / f"{split}_outputs.csv")
        df["vendor"] = split
        all_dfs[split] = df
    return all_dfs


def get_sens_spec_at_threshold(true, pred, op):
    sens = metrics.recall_score(true, pred >= op, pos_label=1, zero_division=1)
    spec = metrics.recall_score(true, pred >= op, pos_label=0, zero_division=1)
    return sens, spec


def get_mcc_at_threshold(true, pred, op):
    pred_pos = (pred >= op).astype(bool)
    pos = true.astype(bool)
    TP = (pred_pos & pos).sum()
    TN = (~pred_pos & ~pos).sum()
    FP = (pred_pos & ~pos).sum()
    FN = (~pred_pos & pos).sum()
    assert (TP + TN + FP + FN) == true.shape[0]
    return (TN * TP - FN * FP) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
