import numpy as np
import matplotlib.pyplot as plt

from helpers import load_synonymous_muts, load_nonsynonymous_muts, GeneralLinearModel


if __name__ == '__main__':

    # Load data
    df_syn = load_synonymous_muts()
    df_nonsyn = load_nonsynonymous_muts()

    # Train model
    model = GeneralLinearModel(included_factors=['global_context', 'rna_structure', 'local_context'])
    model.train(df_train=df_syn)

    # pseudo-count
    pc = 0.5
    count_threshold = 10
    # Plot previous vs. refined estimates
    plt.figure(figsize=(6, 6.25), dpi=150)

    for i, df in enumerate([df_syn, df_nonsyn]):

        model.add_predicted_counts(df)

        previous_estimate = (np.log(df.actual_count + pc) - np.log(df.expected_count + pc))[df.expected_count>count_threshold]
        refined_estimate = (np.log(df.actual_count + pc) - np.log(df.predicted_count + pc))[df.predicted_count>count_threshold]

        max_fit = max(np.max(refined_estimate), np.max(previous_estimate))
        min_fit = min(np.min(refined_estimate), np.min(previous_estimate))
        hist_range = (min_fit, max_fit)
        n_bins = 70

        mean_uncorr = np.mean(previous_estimate)
        var_uncorr = np.var(previous_estimate)
        mean_corr = np.mean(refined_estimate)
        var_corr = np.var(refined_estimate)

        plt.subplot(2, 1, i + 1)

        n_uncorr, bins_uncorr, patches_uncorr = plt.hist(previous_estimate, bins=n_bins, range=hist_range, alpha=0.6,
                                                         label='context ignored \n(Bloom and Neher, 2023)', color='C0')
        n_corr, bins_corr, patches_corr = plt.hist(refined_estimate, bins=n_bins, range=hist_range, alpha=0.6,
                                                   label='context aware', color='C1')

        plt.axvline(0, color='black', linestyle='dashed', linewidth=2, alpha=0.5)
        plt.annotate(f'mean: {mean_uncorr:.2f}\nvariance: {var_uncorr:.2f}',
                     xy=(0.95, 0.98), xycoords='axes fraction',
                     ha='right', va='top', color='C0', alpha=0.8)
        plt.annotate(f'mean: {mean_corr:.2f}\nvariance: {var_corr:.2f}',
                     xy=(0.95, 0.85), xycoords='axes fraction',
                     ha='right', va='top', color='C1', alpha=0.8)

        plt.xlim(left=-8.5, right=5.5)

        if i == 0:
            plt.gca().axes.get_xaxis().set_visible(False)
            plt.legend(loc='upper left')

            plt.title("Synonymous mutations")
            # add panel label
            plt.text(-0.1, 1.05, 'A', transform=plt.gca().transAxes,
                     fontsize=16, va='top', ha='right')

        if i == 1:
            plt.title("Non-synonymous mutations")
            plt.text(-0.1, 1.05, 'B', transform=plt.gca().transAxes,
                     fontsize=16, va='top', ha='right')

        plt.xlabel('Fitness estimate')
        plt.ylabel('Number of mutations')

    plt.tight_layout()
    plt.savefig('../results/previous_vs_refined_estimates.pdf')
    plt.show()

    alpha = 'ACGT'
    print("Mutation & $\\tau^2$ & $\\tau$\\\\")
    for anc in alpha:
        for der in alpha:
            if anc != der:
                tau_sq = df_syn.loc[(df_syn.clade_founder_nt==anc) & (df_syn.mut_nt==der), "tau_squared"].mean()
                print(f'{anc}->{der} & {tau_sq} & {tau_sq**0.5}\\\\')
