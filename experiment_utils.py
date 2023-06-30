from sklearn.model_selection import train_test_split
from upa import align_predictions
import numpy as np
import pandas as pd
from metrics_utils import get_mcc_at_threshold, get_sens_spec_at_threshold
from sklearn.metrics import roc_auc_score
import seaborn as sns
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm


def fit_and_evaluate_shift_detection(
    reference_df, df_for_alignment, df_for_testing, ax=None
):
    """
    Align binary predictions using alignment set, and return prediction difference as shift metric.
    """

    transformed_pred1 = align_predictions(
        reference_df, df_for_alignment, df_for_testing
    )[:, 1]

    original_pred1 = df_for_testing.pred1

    hist_pred1 = np.histogram(transformed_pred1, bins=np.linspace(0, 1, 20))[0]
    hist_pred2 = np.histogram(original_pred1, bins=np.linspace(0, 1, 20))[0]

    if ax is not None:
        sns.kdeplot(
            original_pred1, ax=ax, c="blue", alpha=0.2, clip=[0, 1], label="Before"
        )
        sns.kdeplot(
            transformed_pred1, ax=ax, c="red", alpha=0.2, clip=[0, 1], label="After"
        )
        ax.legend()
    metrics_dict = {
        "N_images_in_alignment_set": len(df_for_alignment),
        "N_images_in_evaluation_set": len(df_for_testing),
        "N_images_in_reference_set": len(reference_df),
        "Predictions": ["Aligned"],
        "MAD": [np.abs(transformed_pred1 - original_pred1).mean()],
        "HistMAD": np.abs(hist_pred1 - hist_pred2).mean(),
    }

    return pd.DataFrame(metrics_dict)


def fit_and_evaluate_alignment_binary_classification(
    reference_df,
    reference_threshold,
    df_for_alignment,
    df_for_testing,
    label_column="malignant",
    return_preds=False,
):
    """
    Align binary predictions using alignment set and evaluate on testing set.
    """

    transformed_pred1 = align_predictions(
        reference_df, df_for_alignment, df_for_testing
    )[:, 1]

    labels = df_for_testing[label_column].astype(int)
    original_pred1 = df_for_testing.pred1
    before_sens, before_spec = get_sens_spec_at_threshold(
        labels,
        original_pred1,
        reference_threshold,
    )

    youden_before = before_sens + before_spec - 1

    after_sens, after_spec = get_sens_spec_at_threshold(
        labels,
        transformed_pred1,
        reference_threshold,
    )

    youden_after = after_sens + after_spec - 1

    metrics_dict = {
        "N_images_in_alignment_set": len(df_for_alignment),
        "N_images_in_evaluation_set": len(df_for_testing),
        "N_images_in_reference_set": len(reference_df),
        "Predictions": ["Original", "Aligned"],
        "roc_auc": [
            roc_auc_score(labels, original_pred1),
            roc_auc_score(labels, transformed_pred1),
        ],
        "sens": [before_sens, after_sens],
        "spec": [before_spec, after_spec],
        "youden": [youden_before, youden_after],
    }

    # Don't compute MCC for breast application, under imbalanced settings it is unreliable.
    if label_column != "malignant":
        metrics_dict["mcc"] = [
            get_mcc_at_threshold(labels, original_pred1, reference_threshold),
            get_mcc_at_threshold(labels, transformed_pred1, reference_threshold),
        ]

    if return_preds:
        return pd.DataFrame(metrics_dict), original_pred1, transformed_pred1

    return pd.DataFrame(metrics_dict)


def evaluate_alignement_with_repeated_sampling(
    ood_df,
    reference_df,
    reference_threshold,
    n_repeat,
    alignment_size=1000,
    reference_size=None,
    test_size=2500,
    label_column="malignant",
):
    all_results_df = []
    for _ in range(n_repeat):
        if reference_size is not None:
            no_positive_in_subset = True
            # Reject subsets that have no positive
            while no_positive_in_subset:
                patients_for_reference = np.random.choice(
                    reference_df.id.unique(),
                    size=reference_size,
                    replace=False,
                )
                subset_ref_df = reference_df.loc[
                    reference_df.id.isin(patients_for_reference)
                ]
                no_positive_in_subset = subset_ref_df[label_column].values.sum() == 0
        else:
            subset_ref_df = reference_df
        # Split patients in testing and alignment splits
        patients_for_alignment, patients_for_testing = train_test_split(
            ood_df.id.unique(), test_size=test_size, train_size=alignment_size
        )
        df_for_alignment = ood_df.loc[ood_df.id.isin(patients_for_alignment)]
        df_for_testing = ood_df.loc[ood_df.id.isin(patients_for_testing)]

        current_result = fit_and_evaluate_alignment_binary_classification(
            subset_ref_df,
            reference_threshold,
            df_for_alignment,
            df_for_testing,
            label_column=label_column,
        )

        all_results_df.append(current_result)

    all_results = pd.concat(all_results_df, ignore_index=True)
    all_results["Difference Sensitivity - Specifity"] = (
        all_results["sens"] - all_results["spec"]
    )
    all_results["N_patients_alignment"] = alignment_size
    all_results["N_patients_reference"] = len(subset_ref_df.id.unique())
    all_results.loc[
        all_results.Predictions == "Original", "N_patients_alignment"
    ] = "Before"
    all_results.loc[
        all_results.Predictions == "Original", "N_patients_reference"
    ] = "Before"
    return all_results


####################### Code for experiment with continuous shifts ###########################


def n_per_scanner(scenario, time):
    if scenario == "A":
        scanner_1 = np.interp(
            time, [0, 29, 30, 99, 100, 120], [250, 250, 225, 49, 0, 0]
        ).astype(int)
        return scanner_1, 250 - scanner_1
    elif scenario == "B":
        return np.ones_like(time).astype(int) * 250, np.interp(
            time, [0, 29, 30, 100, 120], [0, 0, 25, 250, 250]
        ).astype(int)
    elif scenario == "C":
        return np.ones_like(time).astype(int) * 250, np.zeros_like(time).astype(int)
    elif scenario == "D":
        scanner_1 = np.interp(time, [0, 49, 50], [250, 250, 0]).astype(int)
        return scanner_1, 250 - scanner_1


def sample_patients_df(scenario, time, scanner1_df, scanner2_df):
    n_samples_scanner_1, n_samples_scanner_2 = n_per_scanner(scenario, time)
    patients_scanner_1 = np.random.choice(
        scanner1_df.id.unique(), n_samples_scanner_1, replace=False
    )
    patients_scanner_2 = np.random.choice(
        scanner2_df.id.unique(), n_samples_scanner_2, replace=False
    )
    return pd.concat(
        [
            scanner1_df.loc[scanner1_df.id.isin(patients_scanner_1)],
            scanner2_df.loc[scanner2_df.id.isin(patients_scanner_2)],
        ]
    )


def get_experiment_result_as_plot_update(
    scanner1_df,
    scanner2_df,
    reference_df,
    reference_threshold,
    n_repeat=250,
):
    start_time_evaluation = 3
    stop_time_evaluation = 131
    combined_results_df = pd.DataFrame(
        columns=["time", "roc_auc", "sens", "spec", "Predictions"]
    )
    combined_results_df = []
    for _ in tqdm(range(n_repeat)):
        patients_sampled_2_past_weeks = []
        patients_already_in_window = []
        # Init the data buffer with data before the evaluation period starts
        for t in range(start_time_evaluation - 2, start_time_evaluation):
            patients_sampled_2_past_weeks.append(
                sample_patients_df(
                    "D",
                    t,
                    scanner1_df.loc[
                        ~scanner1_df.id.isin(patients_already_in_window)
                    ],
                    scanner2_df.loc[
                        ~scanner2_df.id.isin(patients_already_in_window)
                    ],
                )
            )
            patients_already_in_window = pd.concat(
                patients_sampled_2_past_weeks
            ).id.unique()

        # Start evaluation while continuing the generate the data flow
        for t in range(start_time_evaluation, stop_time_evaluation):
            patients_already_in_window = pd.concat(
                patients_sampled_2_past_weeks
            ).id.unique()

            assert patients_already_in_window.size == 500

            patient_current_time = sample_patients_df(
                "D",
                t,
                scanner1_df.loc[
                    ~scanner1_df.id.isin(patients_already_in_window)
                ],
                scanner2_df.loc[
                    ~scanner2_df.id.isin(patients_already_in_window)
                ],
            )

            df_for_alignment = pd.concat(patients_sampled_2_past_weeks)

            result = fit_and_evaluate_alignment_binary_classification(
                reference_df,
                reference_threshold,
                df_for_alignment,
                patient_current_time,
            )
            result["time"] = t
            combined_results_df.append(result)
            # Update patient buffer
            patients_sampled_2_past_weeks[0] = patients_sampled_2_past_weeks[1]
            patients_sampled_2_past_weeks[1] = patient_current_time

    combined_results_df = pd.concat(combined_results_df, ignore_index=True)
    combined_results_df["Difference Sens - Spec"] = (
        combined_results_df["sens"] - combined_results_df["spec"]
    )

    x = np.arange(0, 130)
    scanner_1, scanner_2 = n_per_scanner("D", x)

    f, ax = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [3, 4]})
    ax[0].plot(x, scanner_1, label="Scanner A", c="black")
    ax[0].plot(
        x, scanner_2, label="Scanner A after software update", c="black", ls="--"
    )
    ax[0].axvline(50, label="T1 simulated software update", ls=":", color="grey")
    ax[1].axvline(50, label="T1", ls=":", color="grey")
    ax[1].set_xlim(0, 130)
    ax[0].set_xlabel("Time in weeks")
    ax[0].set_ylabel("N patients scanned at site")
    ax[0].legend()
    ax[0].set_title("Scanner distribution over time")
    ax[0].set_xlim(0, 130)
    sns.lineplot(
        data=combined_results_df,
        x="time",
        hue="Predictions",
        y="Difference Sens - Spec",
        ax=ax[1],
        errorbar=("pi", 90),
    )
    ax[1].axhline(y=0, label="Target", color="black", ls=":")
    ax[1].set_title("Sensitivity/Specifity difference over time")
    plt.suptitle(
        r"$\bf{Scenario}$ " + r"$\bf{4:}$ " + r"$\bf{Image}$ "
        r"$\bf{processing}$ "
        r"$\bf{update}$"
    )
    plt.savefig(
        "plots/scenario_update.pdf",
        bbox_inches="tight",
    )
    plt.savefig("plots/scenario_update.jpg", bbox_inches="tight", dpi=600)


def get_experiment_result_as_plot_several_scanners(
    scenario,
    scanner1_df,
    scanner2_df,
    reference_df,
    reference_threshold,
    n_repeat=250,
):
    start_time_evaluation = 3
    stop_time_evaluation = 131
    combined_results_df = []
    for _ in tqdm(range(n_repeat)):
        patients_sampled_2_past_weeks = defaultdict(list)
        patients_already_in_window = defaultdict(list)

        # Init the data buffer with data before the evaluation period starts
        for t in range(start_time_evaluation - 2, start_time_evaluation):
            n_samples_scanner_1, n_samples_scanner_2 = n_per_scanner(scenario, t)
            patients_scanner_1 = np.random.choice(
                scanner1_df.id.unique(), n_samples_scanner_1, replace=False
            )
            patients_scanner_2 = np.random.choice(
                scanner2_df.id.unique(), n_samples_scanner_2, replace=False
            )
            patients_sampled_2_past_weeks[1].append(
                scanner1_df.loc[
                    (scanner1_df.id.isin(patients_scanner_1))
                    & (~scanner1_df.id.isin(patients_already_in_window[1]))
                ]
            )
            patients_sampled_2_past_weeks[2].append(
                scanner2_df.loc[
                    (scanner2_df.id.isin(patients_scanner_2))
                    & (~scanner2_df.id.isin(patients_already_in_window[2]))
                ]
            )
            patients_already_in_window[1] = pd.concat(
                patients_sampled_2_past_weeks[1]
            ).id.unique()
            patients_already_in_window[2] = pd.concat(
                patients_sampled_2_past_weeks[2]
            ).id.unique()

        # Start evaluation while continuing the generate the data flow
        for t in range(start_time_evaluation, stop_time_evaluation):
            current_dfs = {}
            for i in [1, 2]:
                patients_already_in_window[i] = pd.concat(
                    patients_sampled_2_past_weeks[i]
                ).id.unique()
            n_samples_scanner_1, n_samples_scanner_2 = n_per_scanner(scenario, t)
            patients_scanner_1 = np.random.choice(
                scanner1_df.loc[
                    (~scanner1_df.id.isin(patients_already_in_window[1]))
                ].id.unique(),
                n_samples_scanner_1,
                replace=False,
            )
            patients_scanner_2 = np.random.choice(
                scanner2_df.loc[
                    (~scanner2_df.id.isin(patients_already_in_window[2]))
                ].id.unique(),
                n_samples_scanner_2,
                replace=False,
            )

            current_dfs[1] = scanner1_df.loc[
                scanner1_df.id.isin(patients_scanner_1)
            ]
            current_dfs[2] = scanner2_df.loc[
                scanner2_df.id.isin(patients_scanner_2)
            ]
            transformed_pred1 = defaultdict(lambda: np.ndarray(0))
            original_pred1 = defaultdict(lambda: np.ndarray(0))
            labels = defaultdict(lambda: np.ndarray(0))
            for i in [1, 2]:
                df_for_alignment = pd.concat(patients_sampled_2_past_weeks[i])
                # Update patient buffer
                patients_sampled_2_past_weeks[i][0] = patients_sampled_2_past_weeks[i][
                    1
                ]
                patients_sampled_2_past_weeks[i][1] = current_dfs[i]
                if len(current_dfs[i]) < 1:
                    continue
                original_pred1[i] = current_dfs[i].pred1
                labels[i] = current_dfs[i]["malignant"].astype(int)
                before_sens, before_spec = get_sens_spec_at_threshold(
                    labels[i],
                    original_pred1[i],
                    reference_threshold,
                )
                if len(df_for_alignment.id.unique()) < 75:
                    result = pd.DataFrame(
                        {
                            "Predictions": ["Original"],
                            "sens": [before_sens],
                            "spec": [before_spec],
                            "scanner": "A" if i == 1 else "B",
                            "time": t,
                        }
                    )
                else:
                    transformed_pred1[i] = align_predictions(
                        reference_df, df_for_alignment, current_dfs[i]
                    )[:, 1]

                    after_sens, after_spec = get_sens_spec_at_threshold(
                        labels[i],
                        transformed_pred1[i],
                        reference_threshold,
                    )
                    result = pd.DataFrame(
                        {
                            "Predictions": ["Original", "Aligned"],
                            "sens": [before_sens, after_sens],
                            "spec": [before_spec, after_spec],
                            "scanner": "A" if i == 1 else "B",
                            "time": t,
                        }
                    )
                combined_results_df.append(result)
                before_sens, before_spec = get_sens_spec_at_threshold(
                    np.concatenate([labels[1], labels[2]]),
                    np.concatenate([original_pred1[1], original_pred1[2]]),
                    reference_threshold,
                )
                if transformed_pred1[2].size > 0:
                    after_sens, after_spec = get_sens_spec_at_threshold(
                        np.concatenate([labels[1], labels[2]]),
                        np.concatenate([transformed_pred1[1], transformed_pred1[2]]),
                        reference_threshold,
                    )
                else:
                    after_sens, after_spec = get_sens_spec_at_threshold(
                        np.concatenate([labels[1], labels[2]]),
                        np.concatenate([transformed_pred1[1], original_pred1[2]]),
                        reference_threshold,
                    )
                combined_results_df.append(
                    pd.DataFrame(
                        {
                            "Predictions": ["Original", "Aligned"],
                            "sens": [before_sens, after_sens],
                            "spec": [before_spec, after_spec],
                            "scanner": "All",
                            "time": t,
                        }
                    )
                )

    combined_results_df = pd.concat(combined_results_df, ignore_index=True)
    combined_results_df["Difference Sens - Spec"] = (
        combined_results_df["sens"] - combined_results_df["spec"]
    )

    x = np.arange(0, 130)
    scanner_1, scanner_2 = n_per_scanner(scenario, x)

    f, ax = plt.subplots(
        1, 4, figsize=(25, 5), gridspec_kw={"width_ratios": [3, 4, 4, 4]}
    )
    ax[0].plot(x, scanner_1, label="Scanner A", c="black")
    ax[0].plot(x, scanner_2, label="Scanner B", c="black", ls="--")
    ax[0].set_xlabel("Time in weeks")
    ax[0].set_ylabel("N patients scanned at site")
    ax[0].legend()
    ax[0].set_title("Scanner distribution over time")
    ax[0].set_xlim(0, 130)

    sns.lineplot(
        data=combined_results_df[combined_results_df.scanner == "A"],
        x="time",
        hue="Predictions",
        y="Difference Sens - Spec",
        ax=ax[2],
        errorbar=("pi", 90),
    )
    ax[2].axhline(y=0, label="Target", color="black", ls=":")
    ax[2].set_title("SEN / SPC difference over time for scanner A")
    ax[2].set_xlim(0, 130)
    sns.lineplot(
        data=combined_results_df[combined_results_df.scanner == "B"],
        x="time",
        hue="Predictions",
        y="Difference Sens - Spec",
        ax=ax[3],
        errorbar=("pi", 90),
    )
    ax[3].set_xlim(0, 130)
    ax[3].axhline(y=0, label="Target", color="black", ls=":")
    ax[3].set_title("SEN / SPC difference over time for Scanner B")

    sns.lineplot(
        data=combined_results_df[combined_results_df.scanner == "All"],
        x="time",
        hue="Predictions",
        y="Difference Sens - Spec",
        ax=ax[1],
        errorbar=("pi", 90),
    )
    ax[1].set_xlim(0, 130)
    ax[1].axhline(y=0, label="Target", color="black", ls=":")
    ax[1].set_title("Overall SEN / SPC difference over time")

    scenario_name_paper_mapping = {
        "A": r"$\bf{Scenario}$ " + r"$\bf{2:}$ " + r"$\bf{Transition}$ "
        r"$\bf{to}$ "
        r"$\bf{a}$ "
        r"$\bf{new}$ " + r"$\bf{scanner}$",
        "B": r"$\bf{Scenario}$ " + r"$\bf{3:}$ " + r"$\bf{Addition}$ "
        r"$\bf{of}$ "
        r"$\bf{a}$ "
        r"$\bf{new}$ " + r"$\bf{scanner}$",
    }
    plt.suptitle(scenario_name_paper_mapping[scenario.upper()])
    plt.savefig(
        f"plots/scenario_{scenario}.pdf",
        bbox_inches="tight",
    )
    plt.savefig(f"plots/scenario_{scenario}.jpg", bbox_inches="tight", dpi=600)
