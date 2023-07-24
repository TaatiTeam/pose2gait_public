import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


def plot_results(
    df,
    target_features,
    output_dir,
    name,
    incl_sources=True,
    datasets=("ds1", "ds2"),
    sources=("openpose", "alphapose", "detectron"),
):
    """Plot the true vs predicted values to show correlation

    Args:
        df (pandas.DataFrame): _description_
        target_features (list[str]): A list of gait features to plot
        output_dir (str): location to save plots
        name (str): name to use when saving plots. The output file will be
            saved as <name>_correlation_plots.png
        incl_sources (bool, optional): If true, use different shape markers f
            or each source (openpose, alphapose, detectron). Defaults to True.
        datasets (tuple, optional): List of datasets to set different colors
            for and include in legend. Datasets not present in df columns will
            be filtered out. Defaults to ['tri', 'mdc','belmont', 'lakeside'].
        sources (tuple, optional): List of pose sequence sources to set
            different shape markers for and include in legend. Sources
            not present in df columns will be fitered out. Defaults to
            ['openpose', 'alphapose', 'detectron'].
    """
    os.makedirs(output_dir, exist_ok=True)
    colors = ["red", "green", "blue", "orange", "black", "purple", "grey"]
    shapes = [".", "v", "s", "*", "1", "p"]

    fig = plt.figure(figsize=(10, 10))
    axes = fig.subplots((len(target_features) + 1) // 2, 2)
    if len(target_features) == 1:
        axes = [axes]
    for i, feat in enumerate(target_features):
        filtered = df.loc[df[f"gt_{feat}"].notnull()]
        ax = axes[i // 2][i % 2]
        x = filtered[f"pred_{feat}"].to_list()
        y = filtered[f"gt_{feat}"].to_list()
        if len(x) == 0 or len(y) == 0:
            continue
        ax.set_xlim(0, max(x) + 0.1)
        ax.set_ylim(0, max(y) + 0.1)
        correlation, pvalue = spearmanr(x, y)
        for ds, color in zip(datasets, colors):
            if ds not in filtered.columns:
                continue
            ds_filtered = filtered.loc[filtered[ds] == 1]
            if ds_filtered.empty:
                continue
            if incl_sources:
                for s, shape in zip(sources, shapes):
                    if s not in ds_filtered.columns:
                        continue
                    s_filtered = ds_filtered.loc[ds_filtered[s] == 1]
                    if s_filtered.empty:
                        continue
                    x = s_filtered[f"pred_{feat}"].to_list()
                    y = s_filtered[f"gt_{feat}"].to_list()
                    ax.scatter(
                        x,
                        y,
                        c=color,
                        marker=shape,
                        label=f"{ds}_{s}",
                        alpha=0.5,
                    )
            else:
                x = ds_filtered[f"pred_{feat}"].to_list()
                y = ds_filtered[f"gt_{feat}"].to_list()
                ax.scatter(x, y, c=color, label=f"{ds}", alpha=0.5)
        ax.text(
            0.6,
            0.1,
            f"Spearman {correlation:.2f}",
            transform=ax.transAxes,
        )
        ax.text(0.6, 0.05, f"pvalue {pvalue:.2f}", transform=ax.transAxes)
        ax.axline([0, 0], [1, 1])
        if i == 0:
            legend = ax.get_legend_handles_labels()
        # ax.text(0.1, 0.9, f"{not_computed}/{len(pairs)} not computed",
        #         transform=ax.transAxes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{feat}")

    fig.legend(*legend, loc="lower center", ncol=3)
    fig.savefig(os.path.join(output_dir, f"{name}_correlation_plots.png"))
    plt.close(fig=fig)


def compute_metrics(df, target_features):
    maes = []
    mapes = []
    mses = []
    rs = []
    pvals = []
    r2s = []
    for feat in target_features:
        filtered = df.loc[df[f"gt_{feat}"].notnull()]

        predictions = filtered[f"pred_{feat}"].to_list()
        gt = filtered[f"gt_{feat}"].to_list()
        if len(gt) == 0 or len(predictions) == 0:
            maes.append(None)
            mapes.append(None)
            mses.append(None)
            rs.append(None)
            pvals.append(None)
            r2s.append(None)
        else:
            maes.append(mean_absolute_error(gt, predictions))
            mapes.append(mean_absolute_percentage_error(gt, predictions))
            mses.append(mean_squared_error(gt, predictions))
            correlation, pvalue = spearmanr(gt, predictions)
            rs.append(correlation)
            pvals.append(pvalue)
            r2s.append(r2_score(gt, predictions))
    return maes, mapes, mses, r2s, rs, pvals


def load_and_evaluate_results(results_file, outdir, target_features):
    df = pd.read_csv(results_file, na_values=["None"])
    evaluate_results(df, outdir, target_features)


def cap_values(df, max_vals):
    for feature, value in max_vals.items():
        df[f"pred_{feature}"] = df[f"pred_{feature}"].clip(upper=value)
    return df


def evaluate_results(
    df,
    outdir,
    target_features,
    max_vals=None,
    dataset_names=["ds1", "ds2"],
    source_names=["openpose", "alphapose", "detectron"],
):
    os.makedirs(outdir, exist_ok=True)
    datasets = [d for d in dataset_names if d in df.columns]
    sources = [s for s in source_names if s in df.columns]
    combinations = [(d, s) for d in datasets for s in sources]
    splits = [[d] for d in datasets] + [[s] for s in sources] + combinations
    default_max_vals = {
        "avg_step_time": 2,
        "avg_step_width": 0.8,
        "avg_step_length": 1.2,
        "velocity": 1.8,
    }
    if max_vals is None:
        max_vals = {f: v for f, v in default_max_vals.items() if f in target_features}
    df = cap_values(df, max_vals)
    results = {}
    results["overall"] = compute_metrics(df, target_features)
    plot_results(df, target_features, outdir, "overall", incl_sources=sources)
    for s in splits:
        split_name = "_".join(s)
        filtered = df
        for element in s:
            filtered = filtered.loc[filtered[element] == 1]
        if not filtered.empty:
            results[split_name] = compute_metrics(filtered, target_features)
            incl_sources = True if s in sources else False
            plot_results(
                filtered, target_features, outdir, split_name, incl_sources=incl_sources,
            )
    write_results(results, outdir, target_features)


def write_results(results, outdir, target_features):
    metrics = ("mae", "mape", "mse", "r2", "spearmans_r", "pvalues")
    for i, metric in enumerate(metrics):
        with open(os.path.join(outdir, f"{metric}.csv"), "w") as f:
            header = ["name"]
            header.extend(target_features)
            f.write(",".join(header))
            f.write("\n")
            for name, result in results.items():
                error = result[i]
                f.write(f"{name},")
                f.write(",".join([f"{d:.4f}" if d is not None else "" for d in error]))
                f.write("\n")


def combine_folds(base_path, num_folds, target_features, individual=True):
    # assumes only one csv in each fold dir
    dfs = []
    for i in range(num_folds):
        fold_dir = os.path.join(base_path, f"fold{i}")
        files = list(os.listdir(fold_dir))
        csvs = [f for f in files if f.endswith("predictions.csv")]
        assert len(csvs) == 1,\
            f"{len(csvs)} predictions csvs in dir {fold_dir}"
        csv = csvs[0]
        result_file = os.path.join(fold_dir, csv)
        dfs.append(pd.read_csv(result_file, na_values=["None"]))
        if individual:
            load_and_evaluate_results(result_file, fold_dir, target_features)
    master_df = pd.concat(dfs)
    evaluate_results(master_df, base_path, target_features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument(
        "-f",
        "--features",
        nargs="*",
        default=[
            "avg_step_time",
            "avg_step_width",
            "avg_step_length",
            "velocity",
            "avg_MOS",
        ],
    )
    args = parser.parse_args()
    path = args.path
    num_folds = args.folds
    target_features = args.features
    combine_folds(path, num_folds, target_features, individual=False)
