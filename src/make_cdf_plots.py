import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from math import exp
from scipy import stats
from itertools import permutations

# Read in mutation counts.
site_muts_df = pd.read_csv("../results/curated_mut_counts.csv").sort_values(
    by="nt_site"
)
site_muts_df = site_muts_df[site_muts_df.synonymous]
site_muts_df.reset_index(drop=True, inplace=True)
site_muts_df.rename(
    columns={"nt_site": "site", "actual_count": "mut_count"}, inplace=True
)
site_muts_df = site_muts_df[["site", "mut_count", "mut_type"]]

q = 0.98
upper = site_muts_df.groupby(by=["mut_type"]).mut_count.quantile(q=q).to_dict()
clipper = lambda row: min(row.mut_count, upper[row.mut_type])
site_muts_df["mut_count"] = site_muts_df.apply(clipper, axis=1)

mut_by_site_dfs = {
    mut_type: site_muts_df[site_muts_df.mut_type == mut_type].set_index("site")
    for mut_type in site_muts_df.mut_type.unique()
}


def make_cdf_plot(ax, mut_type):
    """
    Plot on ax the observed counts CDF and theoretical log-normal CDF for the given mutation type.
    """
    the_df = mut_by_site_dfs[mut_type]
    mut_counts = the_df.mut_count
    if len(mut_counts) < 10:
        return -1

    ax.set_xlabel("mutation count")
    title = f"CDFs for {mut_type} ({len(mut_counts)} samples)"
    ax.set_title(title)

    # Oberserved counts.
    sns.ecdfplot(the_df.mut_count, ax=ax, stat="proportion", label="observed")

    # Log-normal expected counts
    eps = 1 / 2
    log_mut_counts = np.log(mut_counts + eps)
    mean_log = log_mut_counts.mean()
    var_log = ((log_mut_counts - mean_log) ** 2).mean()
    std_log = var_log**0.5
    _, x1 = ax.get_xlim()
    xs = np.linspace(eps, x1 + eps, 500)

    lognormal_cdf = stats.lognorm.cdf(xs, s=std_log, scale=exp(mean_log))
    ax.plot(xs - eps, lognormal_cdf, lw=2, c="red", label="lognormal")

    ax.legend()

    return None


mut_types = list(map("".join, permutations("AGCT", 2)))

fig, axes = plt.subplots(nrows=4, ncols=3, sharey=True)
axes = [axes[i][j] for i in range(4) for j in range(3)]
fig.set_figwidth(2 * fig.get_figwidth())
fig.set_figheight(2.5 * fig.get_figheight())


for mut_type, ax in zip(mut_types, axes):
    make_cdf_plot(ax, mut_type)
fig.tight_layout()

plt.savefig("../results/log_normal.pdf")
