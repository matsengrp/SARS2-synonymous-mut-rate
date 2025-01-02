import numpy as np
import matplotlib.pyplot as plt

from helpers import load_synonymous_muts


if __name__ == '__main__':

    # Load data
    df_all = load_synonymous_muts()

    # 12 mutation types
    mutation_types = ['AC', 'AG', 'AT', 'CA', 'CG', 'CT', 'GA', 'GC', 'GT', 'TA', 'TC', 'TG']

    # 3 x 4 subplots
    fig, axs = plt.subplots(3, 4, figsize=(16, 10), dpi=150)
    axes = axs.flatten(order='F')

    # Define number of bins
    n_bins = 40

    # Loop over all mutation types
    for i, mut_type in enumerate(mutation_types):
        # Filter mutation type
        df = df_all[df_all['mut_type'] == mut_type]

        # Remove outliers
        lower_bound = df['actual_count'].quantile(0.005)
        upper_bound = df['actual_count'].quantile(0.999)
        df = df[(df['actual_count'] > lower_bound) & (df['actual_count'] < upper_bound)]

        # Extract counts from dataframe
        counts = df['actual_count'].values
        counts_paired = counts[df['unpaired'].values == 0]
        counts_unpaired = counts[df['unpaired'].values == 1]

        # Select the correct subplot
        ax = axes[i]

        # Create histogram with a log-scaled x-axis
        ax.hist(counts, bins=np.logspace(np.log10(np.min(counts)), np.log10(np.max(counts)), n_bins),
                edgecolor='black', facecolor='none', label='all sites')
        ax.hist(counts_paired, bins=np.logspace(np.log10(np.min(counts)), np.log10(np.max(counts)), n_bins),
                color='blue', alpha=0.7, label='paired sites')
        ax.hist(counts_unpaired, bins=np.logspace(np.log10(np.min(counts)), np.log10(np.max(counts)), n_bins),
                color='red', alpha=0.7, label='unpaired sites')

        # Set limits, labels, title
        ax.set_xscale('log')
        ax.set_title(rf'Synonymous {mut_type[0]}$\rightarrow${mut_type[1]} mutations', fontsize=10)
        ax.set_ylabel('Number of sites', fontsize=8)
        ax.set_xlabel(r'Number of counts per site', fontsize=8)

        # Add legend
        ax.legend(loc='upper left', fontsize=6)

    # Save and show figure
    plt.tight_layout()
    plt.savefig('../results/paired_vs_unpaired.pdf')
    plt.show()
