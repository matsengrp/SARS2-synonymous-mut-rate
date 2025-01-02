import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from helpers import load_synonymous_muts


if __name__ == '__main__':

    # Load data for synonymous mutations
    syn_mut_counts_df = load_synonymous_muts()

    # List of all 12 mutation types, sorted alphabetically
    mutation_types = sorted(syn_mut_counts_df['mut_type'].unique())

    # Create a 3x4 grid of subplots
    fig, axes = plt.subplots(3, 4, figsize=(15, 10), dpi=150, sharex=True, sharey=True)
    axes = axes.flatten(order='F')

    # Loop through the mutation types and plot each on a separate subplot
    for i, mut_type in enumerate(mutation_types):

        # Filter current mutation type
        syn_mut_counts_XY_df = syn_mut_counts_df[syn_mut_counts_df['mut_type'] == mut_type]

        # Get log counts on terminal and inner branches separately
        terminal = np.log(syn_mut_counts_XY_df['count_terminal'].values + 0.5)
        non_terminal = np.log(syn_mut_counts_XY_df['count_non_terminal'].values + 0.5)

        # Calculate ratio of medians
        terminal_median = np.median(syn_mut_counts_XY_df['count_terminal'].values)
        non_terminal_median = np.median(syn_mut_counts_XY_df['count_non_terminal'].values)
        median_ratio = terminal_median / non_terminal_median

        # Calculate Pearson correlation
        corr, _ = pearsonr(terminal, non_terminal)

        # Plot on the respective subplot
        axes[i].scatter(non_terminal, terminal, alpha=0.05, edgecolors='none', s=20, color='blue')
        axes[i].set_title(f'{mut_type} (r = {corr:.2f}, ratio of medians = {median_ratio:.2f})')
        if (i+1) % 3 == 0:
            axes[i].set_xlabel('Log-count on inner branches')
        if i < 3:
            axes[i].set_ylabel('Log-count on terminal branches')

        # Add guides for the eye
        x = np.linspace(0, 8, 200)
        y_diag = x + np.log(1.25)
        # y_upper = y_diag + np.log(1 + 1 / np.sqrt(np.exp(y_diag)))
        # y_lower = y_diag + np.log(1 - 1 / np.sqrt(np.exp(y_diag)))
        y_upper = np.log(np.exp(y_diag) + np.sqrt(np.exp(y_diag) - 0.5))
        y_lower = np.log(np.exp(y_diag) - np.sqrt(np.exp(y_diag) - 0.5))
        axes[i].plot(x, y_diag, color='black', linewidth=1)
        axes[i].plot(x, y_upper, color='black', linewidth=1, linestyle='--')
        axes[i].plot(x, y_lower, color='black', linewidth=1, linestyle='--')

        # Add grid lines
        axes[i].grid(True, alpha=0.5)

    # Save plot
    plt.tight_layout()
    plt.savefig('../results/terminal_vs_non-terminal.pdf')
    plt.show()
