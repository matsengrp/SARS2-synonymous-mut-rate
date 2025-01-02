import numpy as np
import matplotlib.pyplot as plt

from helpers import load_synonymous_muts


if __name__ == '__main__':

    # Load data for synonymous mutations
    syn_mut_counts_df = load_synonymous_muts(remove_orf9b=False, include_stop_tolerant_orfs=False, include_noncoding_muts=False)

    # Select AT mutations
    syn_mut_counts_AT_df = syn_mut_counts_df[syn_mut_counts_df['mut_type'] == 'AT']

    # Get coordinates and counts
    coordinates = syn_mut_counts_AT_df['nt_site'].values
    terminal_counts = syn_mut_counts_AT_df['count_terminal'].values
    non_terminal_counts = syn_mut_counts_AT_df['count_non_terminal'].values

    # Parameters for sliding window
    window_size = 1000

    # Initialize lists to store the medians and window positions
    medians_terminal = []
    medians_non_terminal = []
    window_positions = []
    eps=0.5

    # Loop through genome with sliding window
    for start in range(min(coordinates), max(coordinates) - window_size):
        end = start + window_size
        # Get indices of coordinates that fall within the current window
        indices_in_window = np.where((coordinates >= start) & (coordinates < end))[0]
        if len(indices_in_window) > 0:
            # Get the counts for those indices
            window_terminal_counts = terminal_counts[indices_in_window]
            window_non_terminal_counts = non_terminal_counts[indices_in_window]

            # Calculate the medians and store them
            # medians_terminal.append(np.mean(np.log(window_terminal_counts+eps)))
            # medians_non_terminal.append(np.mean(np.log(window_non_terminal_counts+eps)))
            medians_terminal.append(np.median(window_terminal_counts))
            medians_non_terminal.append(np.median(window_non_terminal_counts))
            window_positions.append((start + end) / 2)

    # Plot the normalized medians
    median_terminal = np.median(terminal_counts)
    median_non_terminal = np.median(non_terminal_counts)
    plt.figure(figsize=(8, 4.5), dpi=150)
    plt.plot(window_positions, medians_terminal / median_terminal, label='terminal branches', color='blue')
    plt.plot(window_positions, medians_non_terminal / median_non_terminal, label='inner branches', color='green')
    # spike protein
    plt.axvline(21563, linestyle='--', color='black', linewidth=2, alpha=0.7)
    plt.axvline(25383, linestyle='--', color='black', linewidth=2, alpha=0.7)
    # N protein
    plt.axvline(28274, linestyle='--', color='black', linewidth=2, alpha=0.7)
    plt.axvline(29532, linestyle='--', color='black', linewidth=2, alpha=0.7)
    # orf9b
    #plt.axvline(28284, linestyle='--', color='black', linewidth=2)
    #plt.axvline(28577, linestyle='--', color='black', linewidth=2)
    plt.axhline(1, linestyle='--', color='black', linewidth=2)

    plt.xlabel('Genome position')
    plt.ylabel('Median in sliding window / overall median')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('../results/AT-step_terminal_vs_non-terminal.pdf')
    plt.show()
