import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from collections import defaultdict
from synonymous_rates import SynonymousRates


def plot_r2s(out_path):
    """
    Plots the per mutation R^2 values over the full default data set, over 10 simulated
    data sets assuming a Poisson model (trained on the full dataset), and over training
    and test data sets for 10 replicates (trained on the training dataset).
    """
    reps = 10
    full_model = SynonymousRates(test_holdout=0.0)
    r2_full = full_model.calc_R2_on_full()
    r2_simulated = defaultdict(list)
    r2_train = defaultdict(list)
    r2_test = defaultdict(list)

    for _ in range(reps):
        for mut_type, r2 in full_model.calc_R2_on_poisson_simulated().items():
            r2_simulated[mut_type].append(r2)
        model = SynonymousRates(test_holdout=0.2)
        for mut_type, r2 in model.calc_R2_on_train().items():
            r2_train[mut_type].append(r2)
        for mut_type, r2 in model.calc_R2_on_test().items():
            r2_test[mut_type].append(r2)
    columns = ["mut_type", "R2", "dataset"]
    rows = [[mut_type, r2, "full"] for mut_type, r2 in r2_full.items()]
    rows.extend(
        [
            [mut_type, r2, "simulated"]
            for mut_type in r2_simulated
            for r2 in r2_simulated[mut_type]
        ]
    )
    rows.extend(
        [[mut_type, r2, "train"] for mut_type in r2_train for r2 in r2_train[mut_type]]
    )
    rows.extend(
        [[mut_type, r2, "test"] for mut_type in r2_test for r2 in r2_test[mut_type]]
    )
    r2_df = pd.DataFrame(data=rows, columns=columns)
    view_full = r2_df[r2_df.dataset == "full"]
    view_others = r2_df[r2_df.dataset != "full"]
    view_others = view_others.sort_values(
        by=["mut_type", "dataset"], ascending=[True, False]
    )

    fig, ax = plt.subplots()
    sns.boxplot(
        data=view_others,
        x="mut_type",
        y="R2",
        hue="dataset",
        ax=ax,
        legend=True,
        fliersize=3,
    )
    for tick in ax.get_xticklabels():
        x = tick.get_position()[0]
        mut_type = tick.get_text()
        x0, x1 = x - 0.4, x + 0.4
        y = view_full[view_full.mut_type == mut_type].R2
        ax.plot([x0, x1], [y, y], c="red")

    h, l = ax.get_legend_handles_labels()
    extra = Line2D([0], [0], c="red")
    h.append(extra)
    l.append("full model")
    ax.legend(h, l, loc="lower left")
    fig.tight_layout()
    plt.savefig(out_path)
    plt.show()


if __name__ == "__main__":
    plot_r2s("../results/r2s.png")
