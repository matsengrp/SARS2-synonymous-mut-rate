import numpy as np
import matplotlib.pyplot as plt

from helpers import load_synonymous_muts, GeneralLinearModel


def plot_separate_r_squared(mean_sq_err_dic):

    r_squareds = {}
    for mut_type in ['AC', 'AG', 'AT', 'CA', 'CG', 'CT', 'GA', 'GC', 'GT', 'TA', 'TC', 'TG']:
        r_squareds[mut_type] = []
        for model in mean_sq_err_dic.keys():
            r_squareds[mut_type].append(1 - mean_sq_err_dic[model][mut_type]/mean_sq_err_dic['base'][mut_type])

    n_keys = len(r_squareds)
    x_pos = np.arange(n_keys)
    bar_width = 0.25
    offset = 0.25

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)

    colors = ['#0072B2', '#E69F00', '#009E73']
    legend_names = ['genomic position', 'RNA secondary structure', 'local context']

    for i, (key, values) in enumerate(r_squareds.items()):

        diff_2_4_minus_1 = [values[1] - values[0], values[2] - values[0], values[3] - values[0]]
        diff_5_7_minus_8 = [values[4] - values[7], values[5] - values[7], values[6] - values[7]]

        positive_x = x_pos[i] + np.array([-offset, 0, offset])
        negative_x = x_pos[i] + np.array([-offset, 0, offset])

        for j in range(3):
            ax.bar(positive_x[j], diff_2_4_minus_1[j], bar_width, zorder=3, color=colors[j], label=legend_names[j])

        for j in range(3):
            ax.bar(negative_x[j], diff_5_7_minus_8[j], bar_width, zorder=3, color=colors[j], label=legend_names[j])

        for j in range(3):
            top_edge = - diff_2_4_minus_1[j]
            ax.plot([negative_x[j] - bar_width / 2.2, negative_x[j] + bar_width / 2.2],
                    [top_edge, top_edge],
                    zorder=3, color='black', alpha=0.8, linewidth=1, label=legend_names[j])

        for j in range(3):
            top_edge = - diff_5_7_minus_8[j]
            ax.plot([negative_x[j] - bar_width / 2.2, negative_x[j] + bar_width / 2.2],
                    [top_edge, top_edge],
                    zorder=3, color='black', alpha=0.8, linewidth=1, label=legend_names[j])

    ax.set_ylabel('Difference in $R^{2}$')
    ax.set_xticks(x_pos)
    modified_keys = [rf"{k[0]}$\rightarrow${k[1]}" for k in r_squareds.keys()]

    ax.set_xticklabels(modified_keys, rotation=45)
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.grid(True, which='minor', axis='y', linestyle='-', linewidth=0.5, zorder=1)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.grid(True, which='major', axis='y', linestyle='-', linewidth=0.5, zorder=1)
    ax.axhline(y=0, color='black', linewidth=0.75, zorder=3)

    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3)

    plt.tight_layout()
    plt.savefig('../results/R2_separately.pdf')
    plt.show()


if __name__ == '__main__':

    df = load_synonymous_muts()

    factors = [[], ['global_context'], ['rna_structure'], ['local_context'],
               ['rna_structure', 'local_context'], ['global_context', 'local_context'],
               ['global_context', 'rna_structure'], ['global_context', 'rna_structure', 'local_context']]
    names = ['base', '+ global context', '+ RNA secondary structure', '+ local context',
             '- global context', '- RNA secondary structure', '- local context', 'full']

    mean_squared_errs = {}

    for i in range(8):

        model = GeneralLinearModel(included_factors=factors[i])

        model.train(df_train=df)

        mean_squared_errs[names[i]] = model.test(df_test=df)

    plot_separate_r_squared(mean_squared_errs)
