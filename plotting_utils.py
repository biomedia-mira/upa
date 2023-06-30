import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_results_ablation_studies(
    all_results,
    x,
    dataset_name,
    file_name_prefix="exp1",
    site_prefix="Scanner",
    operating_point="diag",
):
    sns.set_style("whitegrid")

    if operating_point == "diag":
        f, ax = plt.subplots(
            1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [5, 5]}
        )
        ax[1].axhline(y=0, c="black", ls=":", label="Target", linewidth=2)
        sns.boxplot(
            data=all_results,
            x=x,
            y="Difference Sensitivity - Specifity",
            ax=ax[1],
            whis=[5, 95],
        )
        ax[1].legend()

    else:
        f, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax = [ax]

    for s in all_results[x].unique():
        df = all_results.loc[all_results[x] == s]
        ax[0].errorbar(
            df["sens"].values.mean(),
            df["spec"].values.mean(),
            np.abs(
                np.percentile(df["sens"].values, [2.5, 97.5]).reshape(2, 1)
                - df["sens"].values.mean()
            ),
            np.abs(
                np.percentile(df["spec"].values, [2.5, 97.5]).reshape(2, 1)
                - df["spec"].values.mean()
            ),
            ls="None",
            elinewidth=2,
            marker="o",
            label=s,
            ms=10,
        )
    if operating_point == "diag":
        ax[0].plot(
            np.linspace(*ax[0].get_xlim(), 50),
            np.linspace(*ax[0].get_xlim(), 50),
            c="black",
            ls=":",
            label="Target",
            linewidth=2,
        )
    elif operating_point == "sens90":
        ax[0].axvline(
            x=0.90,
            c="black",
            ls=":",
            label="Target",
            linewidth=2,
        )
    elif operating_point == "spec90":
        ax[0].axhline(
            y=0.90,
            c="black",
            ls=":",
            label="Target",
            linewidth=2,
        )
    else:
        raise ValueError
    ax[0].set_xlabel("Sensitivity")
    ax[0].set_ylabel("Specifity")
    ax[0].legend()
    dataset_name = dataset_name.upper()
    if file_name_prefix == "exp1":
        roc_auc_ic = np.percentile(
            all_results.loc[all_results["Predictions"] == "Original", "roc_auc"].values,
            [2.5, 97.5],
        )
        plt.suptitle(
            r"$\bf{Evaluating}$ $\bf{on}$"
            + r" $\bf{"
            + site_prefix
            + r"}$"
            + r" $\bf{"
            + dataset_name
            + r"}$"
            + f"\nROC-AUC (95%-IC): [{roc_auc_ic[0]:.2f} - {roc_auc_ic[1]:.2f}]"
        )
    elif file_name_prefix == "exp2":
        plt.suptitle(
            f"Evaluating the effect of alignment set size on {site_prefix} {dataset_name}"
        )
    elif file_name_prefix == "exp2b":
        plt.suptitle(
            f"Evaluating the effect of reference set size on {site_prefix} {dataset_name}"
        )
    elif file_name_prefix == "exp2c":
        plt.suptitle(
            f"Evaluating the effect of prevalence in alignment set on {site_prefix} {dataset_name}\n(with p=0.05 in reference set)"
        )
        ax[1].set_xlabel("Prevalence patients with malignancy in evaluation set")
    else:
        plt.suptitle(
            r"$\bf{Evaluating}$ $\bf{on}$"
            + r" $\bf{"
            + site_prefix
            + r"}$"
            + r" $\bf{"
            + dataset_name
            + r"}$"
        )
    plt.savefig(
        f"plots/{file_name_prefix}_{dataset_name.lower()}.pdf", bbox_inches="tight"
    )
    plt.savefig(
        f"plots/{file_name_prefix}_{dataset_name.lower()}.jpg",
        bbox_inches="tight",
        dpi=600,
    )
    plt.show()
