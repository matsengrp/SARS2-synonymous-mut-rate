import numpy as np
import matplotlib.pyplot as plt
from helpers import load_synonymous_muts, GeneralLinearModel

mut_types = ['AC', 'AG', 'AT', 'CA', 'CG', 'CT', 'GA', 'GC', 'GT', 'TA', 'TC', 'TG']

if __name__ == '__main__':

    # Load mutation counts across full tree
    df_syn = load_synonymous_muts()

    # Split pre/post Omicron
    df_pre_omicron = df_syn.copy()
    df_pre_omicron['actual_count'] = df_pre_omicron['actual_count_pre_omicron']
    df_post_omicron = df_syn.copy()
    df_post_omicron['actual_count'] = df_post_omicron['actual_count_omicron']

    # Get total number of mutation counts
    log_N_total_pre_omicron = np.log(df_pre_omicron['actual_count'].sum())
    log_N_total_post_omicron = np.log(df_post_omicron['actual_count'].sum())

    # Train separate models
    model_pre_omicron = GeneralLinearModel(included_factors=['global_context', 'rna_structure', 'local_context'])
    model_pre_omicron.train(df_train=df_pre_omicron)
    model_post_omicron = GeneralLinearModel(included_factors=['global_context', 'rna_structure', 'local_context'])
    model_post_omicron.train(df_train=df_post_omicron)

    fig, axes = plt.subplots(3, 4, figsize=(16, 9), dpi=200)
    axes = axes.flatten(order='F')

    min_bar, max_bar = 0, 0
    w_base_pre = []
    w_base_post = []
    mut_labels = []

    for i, mut_type in enumerate(mut_types):
        w_pre_omicron = model_pre_omicron.W[mut_type].flatten()
        w_post_omicron = model_post_omicron.W[mut_type].flatten()

        w_base_pre.append(w_pre_omicron[0])
        w_base_post.append(w_post_omicron[0])
        mut_labels.append(mut_type)

        min_bar = min(np.min(w_pre_omicron[1:]), np.min(w_post_omicron[1:]), min_bar)
        max_bar = max(np.max(w_pre_omicron[1:]), np.max(w_post_omicron[1:]), max_bar)

        indices = np.arange(len(w_pre_omicron))
        bar_width = 0.35
        ax = axes[i]

        ax.bar(indices[1:], w_pre_omicron[1:], width=bar_width, label='pre-Omicron', color='blue')
        ax.bar(indices[1:] + bar_width, w_post_omicron[1:], width=bar_width, label='post-Omicron', color='orange')
        ax.grid(True)
        if i < 3:
            ax.set_ylabel('coefficient')
        ax.set_title(f'{mut_type}')
        ax.legend()

        if len(w_pre_omicron) == 9:
            x_labels = [r"$\beta^{switch}$", r"$\beta^{paired}$", r"$\beta^{C,5'}$", r"$\beta^{G,5'}$", r"$\beta^{T,5'}$",
                        r"$\beta^{C,3'}$", r"$\beta^{G,3'}$", r"$\beta^{T,3'}$"]
        else:
            x_labels = [r"$\beta^{paired}$", r"$\beta^{C,5'}$", r"$\beta^{G,5'}$", r"$\beta^{T,5'}$",
                        r"$\beta^{C,3'}$", r"$\beta^{G,3'}$", r"$\beta^{T,3'}$"]

        ax.set_xticks(indices[1:])
        ax.set_xticklabels(x_labels, rotation=0, ha="center")

    for i, mut_type in enumerate(mut_types):
        axes[i].set_ylim((min_bar - 0.2, max_bar + 0.2))

    plt.tight_layout()
    plt.savefig("../results/pre_vs_post_Omicron_w.pdf")
    plt.show()

    plt.figure(figsize=(8, 6), dpi=200)
    plt.scatter(w_base_pre, w_base_post, color='red')

    for i, mut_type in enumerate(mut_types):
        plt.text(w_base_pre[i], w_base_post[i], mut_type)

    plt.xlabel('$w_{base}$ pre-Omicron')
    plt.ylabel('$w_{base}$ post-Omicron')
    plt.grid(True)

    x_vals = np.array(w_base_pre)
    y_vals = 1 * x_vals + (log_N_total_post_omicron - log_N_total_pre_omicron)
    plt.plot(x_vals, y_vals, color='black')

    plt.tight_layout()
    plt.savefig("../results/pre_vs_post_Omicron_w_base.pdf")
    plt.show()
