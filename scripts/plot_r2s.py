import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from collections import defaultdict
from synonymous_rates import SynonymousRates
from basel_model import plot_map


def plot_r2s(out_path):
    """
    Plots the per mutation R^2 values over the full default data set, over 10 simulated
    data sets assuming a Poisson model (trained on the full dataset), and over training
    and test data sets for 10 replicates (trained on the training dataset).
    """

    # Choose new dataset
    path = "../results/syn_mut_train_test_splits.csv"

    # number of train/test splits
    reps = 10

    # Train models on full dataset and get R^2 per mutation type
    full_model = SynonymousRates(test_holdout=0.0, mut_count_path=path)
    r2_full = {'Basel': full_model.calc_R2_on_full(model='Basel'),
               'Seattle': full_model.calc_R2_on_full(model='Seattle')}

    # Prepare dictionaries for other metrics
    r2_simulated = {'Basel': defaultdict(list),
                    'Seattle': defaultdict(list)}
    r2_train = {'Basel': defaultdict(list),
                'Seattle': defaultdict(list)}
    r2_test = {'Basel': defaultdict(list),
               'Seattle': defaultdict(list)}

    # Loop over all train/test splits
    for rep in range(reps):
        print(f'repetition: {rep}')

        # Calculate R^2 on Poisson simulated data
        for model_type in ['Basel', 'Seattle']:
            for mut_type, r2 in full_model.calc_R2_on_poisson_simulated(model=model_type).items():
                r2_simulated[model_type][mut_type].append(r2)

        # Train models on the current training set
        model = SynonymousRates(split_index=rep, mut_count_path=path)

        for model_type in ['Basel', 'Seattle']:

            # Calculate R^2 on the current training set
            for mut_type, r2 in model.calc_R2_on_train(model=model_type).items():
                r2_train[model_type][mut_type].append(r2)

            # Calculate R^2 on the current test set
            for mut_type, r2 in model.calc_R2_on_test(model=model_type).items():
                r2_test[model_type][mut_type].append(r2)

    # Create plot for both models
    for model_type in ['Basel', 'Seattle']:

        columns = ["mut_type", "R2", "dataset"]

        # Add R^2 over full dataset
        rows = [[mut_type, r2, "full"] for mut_type, r2 in r2_full[model_type].items()]

        # Add R^2 on Poisson simulated data set
        rows.extend(
            [
                [mut_type, r2, "simulated"]
                for mut_type in r2_simulated[model_type]
                for r2 in r2_simulated[model_type][mut_type]
            ]
        )

        # Add R^2 on train and test data sets
        rows.extend(
            [[mut_type, r2, "train"] for mut_type in r2_train[model_type] for r2 in r2_train[model_type][mut_type]]
        )
        rows.extend(
            [[mut_type, r2, "test"] for mut_type in r2_test[model_type] for r2 in r2_test[model_type][mut_type]]
        )

        # ?
        r2_df = pd.DataFrame(data=rows, columns=columns)
        view_full = r2_df[r2_df.dataset == "full"]
        view_others = r2_df[r2_df.dataset != "full"]
        view_others = view_others.sort_values(
            by=["mut_type", "dataset"], ascending=[True, False]
        )

        # Create boxplots
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
        sns.boxplot(
            data=view_others,
            x="mut_type",
            y="R2",
            hue="dataset",
            ax=ax,
            # legend=True,
            fliersize=3,
        )

        # ?
        for tick in ax.get_xticklabels():
            x = tick.get_position()[0]
            mut_type = tick.get_text()
            x0, x1 = x - 0.4, x + 0.4
            y = view_full[view_full.mut_type == mut_type].R2
            ax.plot([x0, x1], [y, y], c="red")

        # ?
        h, l = ax.get_legend_handles_labels()
        extra = Line2D([0], [0], c="red")
        h.append(extra)
        l.append("full model")
        ax.legend(h, l, loc="lower left")

        ax.set_ylim(-0.25, 1.1)
        ax.set_title(model_type)
        ax.grid(True, which='both', axis='both')

        fig.tight_layout()
        file_path = os.path.join(out_path, f"r2s_{model_type}.png")
        plt.savefig(file_path)
        plt.show()


def compare_basel_vs_seattle_predictions(show_log_counts=True):
    # Choose new dataset
    path = "../results/syn_mut_train_test_splits.csv"

    # Train both models on the full dataset
    model = SynonymousRates(mut_count_path=path, split_index=0)

    # Make a plot of Basel vs. Seattle predictions for every mutation type
    fig, axes = plt.subplots(3, 4, figsize=(16, 9), dpi=150)

    # Loop over all mutation types
    for mut_type in model.mut_types:

        # Select correct subplot for this mutation type
        ax = axes[plot_map[mut_type]]

        # Get Seattle predictions
        rates_df = model.all_rates_df[model.all_rates_df.mut_type == mut_type]
        seattle_predictions = rates_df.prefered_rate.values

        # Get Basel predictions
        rates_df = model.basel_model_full.add_predictions(rates_df.copy())
        basel_predictions = rates_df.predicted_count_basel.values

        # Plot log counts if desired
        if show_log_counts:
            seattle_predictions = np.log(seattle_predictions)
            basel_predictions = np.log(basel_predictions)

        # Make scatter plot
        ax.scatter(seattle_predictions, basel_predictions, alpha=0.6, edgecolor='black')
        ax.set_title(rf"{mut_type[0]}$\rightarrow${mut_type[1]}")
        if plot_map[mut_type][0] == 2:
            ax.set_xlabel("Seattle prediction")
        if plot_map[mut_type][1] == 0:
            ax.set_ylabel("Basel prediction")
        ax.grid(True)
        min_val = min(min(seattle_predictions), min(basel_predictions))
        max_val = max(max(seattle_predictions), max(basel_predictions))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

    plt.tight_layout()
    plt.savefig("../results/basel_vs_seattle.png")
    plt.show()


if __name__ == "__main__":
    # Make R^2 plots
    plot_r2s(out_path="../results")

    # Compare predictions of Basel and Seattle model
    compare_basel_vs_seattle_predictions()
