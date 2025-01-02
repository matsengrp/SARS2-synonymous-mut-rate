import numpy as np
import matplotlib.pyplot as plt

from helpers import load_synonymous_muts, GeneralLinearModel


def plot_sequential_r_squared(mean_sq_err_dic):

    model_types = list(mean_sq_err_dic.keys())
    mut_types = sorted(list(mean_sq_err_dic[model_types[0]].keys()))

    r2s = {mut_type: [] for mut_type in mut_types}
    for mut_type in mut_types:
        for i in range(len(model_types)):
            r2s[mut_type].append(1 - (mean_sq_err_dic[model_types[i]][mut_type] / mean_sq_err_dic[model_types[0]][mut_type]))

    plt.figure(figsize=(8, 4.5), dpi=150)

    bar_positions = np.arange(len(mut_types))
    bar_width = 0.6

    base_bar = np.zeros(len(mut_types))

    for i in range(1, len(model_types)):
        increment_values = np.array([r2s[mut_type][i] - r2s[mut_type][i-1] for mut_type in mut_types])
        plt.bar(bar_positions, increment_values, bottom=base_bar, zorder=3,
                label=model_types[i], width=bar_width, color=['#0072B2', '#E69F00', '#009E73'][i-1])
        base_bar += increment_values

    plt.ylabel('$R^{2}$')
    plt.xticks(ticks=bar_positions, labels=[rf"{mut_type[0]}$\rightarrow${mut_type[1]}" for mut_type in mut_types], rotation=45, ha="center")
    plt.grid(True, axis='y', zorder=1)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3)

    plt.tight_layout()
    plt.savefig('../results/R2_sequentially.pdf')
    plt.show()


if __name__ == '__main__':

    df = load_synonymous_muts()

    factors = ['global_context', 'rna_structure', 'local_context']
    names = ['base', 'genomic position', 'RNA secondary structure', 'local context']

    mean_squared_errs = {}

    for i in range(4):

        model = GeneralLinearModel(included_factors=factors[:i])

        model.train(df_train=df)

        mean_squared_errs[names[i]] = model.test(df_test=df)

    plot_sequential_r_squared(mean_squared_errs)
