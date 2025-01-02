import numpy as np
from Bio import SeqIO
import matplotlib.cm as cm
from matplotlib import pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from helpers import load_synonymous_muts, GeneralLinearModel

gene_boundaries = [(0, 265, r"5' UTR"),
                   (266, 13480, 'ORF1a'),
                   (13468, 21552, 'ORF1b'),
                   (21563, 25381, 'S'),
                   (25393, 26217, 'ORF3a'),
                   (26245, 26469, 'E'),
                   (26523, 27188, 'M'),
                   (27202, 27384, 'ORF6'),
                   (27394, 27756, 'ORF7a'),
                   (27756, 27884, 'ORF7b'),
                   (27894, 28256, 'ORF8'),
                   (28274, 29530, 'N'),
                   (29558, 29671, 'ORF10'),
                   (29672, 29903, r"3' UTR")]


def get_conserved_regions():

    TRS_motif = "ACGAAC"
    margin = 0
    ref = SeqIO.read('../data/reference.gb', 'genbank')

    motifs = []
    pos = 1
    while pos > 0:
        pos = ref.seq.find(TRS_motif, pos)
        if pos > 1:
            motifs.append(pos)
            pos += 1

    conserved_regions = {}
    for mi, mpos in enumerate(motifs):
        conserved_regions[f'TRS {mi + 1}'] = (mpos - margin, mpos + len(TRS_motif) + margin)

    conserved_regions['frameshift element'] = (ref.seq.find("TTTAAACG"), ref.seq.find("TTTAAACG") + 7)

    conserved_vector = np.zeros(len(ref.seq), dtype=bool)
    for r in conserved_regions.values():
        conserved_vector[r[0]:r[1]] = True

    return conserved_vector, conserved_regions


def solve_tridiagonal_lse(d, e, f, y):

    # Create the tridiagonal matrix in sparse format
    A = sp.diags([f, d, e], offsets=[-1, 0, 1], format='csc')

    # Solve the system Ax = y
    x = spla.spsolve(A, y)

    return x


def solve_lse(sigma, rho, tau, g):

    # Precompute 1/tau**2
    inv_tau_sq = 1 / tau ** 2

    # Prepare tridiagonal coefficient matrix
    diagonal = 1 / sigma ** 2 + 1 / rho ** 2 + np.append(0, inv_tau_sq[:-1]) + np.append(inv_tau_sq[:-1], 0)
    superdiagonal = - inv_tau_sq[:-1]
    subdiagonal = - inv_tau_sq[:-1]

    # Prepare y in the linear system of equations Af = y
    y = g / rho ** 2

    # Solve linear system of equations to get probabilistic fitness estimates
    f = solve_tridiagonal_lse(diagonal, superdiagonal, subdiagonal, y)

    return f


def plot_gene_boundaries(ax, first_site, last_site, min_fit, color_list, y_offset=1.1):
    for i, (start, end, name) in enumerate(gene_boundaries):
        if ((start <= first_site <= end) or (start <= last_site <= end)) or (
                (first_site <= start <= last_site) and (first_site <= end <= last_site)):
            if last_site - first_site > 100:
                ax.barh(y=min_fit + y_offset + (i % 2) * 0.4, width=end - start, left=start, height=0.4, color=color_list(i),
                         alpha=0.6)
            else:
                ax.barh(y=min_fit + y_offset - 0.35 + (i % 2) * 0.7, width=end - start, left=start, height=0.7,
                        color=color_list(i), alpha=0.6)
        if first_site <= start <= last_site:
            if first_site > 15000 or i < 3:
                if last_site - first_site > 100:
                    ax.text(start, min_fit + y_offset + (i % 2) * 0.4, name, ha='left', va='center')
                else:
                    ax.text(start, min_fit + y_offset - 0.35 + (i % 2) * 0.7, name, ha='left', va='center')
        elif first_site <= end <= last_site or (start < first_site and end > last_site):
            if first_site > 15000 or i < 3:
                if last_site - first_site > 100:
                    ax.text(first_site, min_fit + y_offset + (i % 2) * 0.4, name, ha='left', va='center')
                else:
                    ax.text(first_site, min_fit + y_offset - 0.35 + (i % 2) * 0.7, name, ha='left', va='center')


def plot_conserved_regions(ax, first_site, last_site, min_fit, conserved_regions, y_offset=2.2):
    for i, (start, end, name) in enumerate(conserved_regions):
        if ((start <= first_site <= end) or (start <= last_site <= end)) or (
                (first_site <= start <= last_site) and (first_site <= end <= last_site)):
            ax.barh(y=min_fit + y_offset + (i % 2) * 0.4, width=end - start + 50, left=start-25, height=0.4,
                     color='red', alpha=0.8)
            if first_site > 15000 or i < 3:
                ax.text(max(start, first_site), min_fit + y_offset + (i % 2) * 0.4, name, ha='left', va='center')


def add_xticks(probabilistic_pos, first_site, last_site):
    sites = probabilistic_pos[(probabilistic_pos >= first_site) & (probabilistic_pos <= last_site)]
    xticks = range(min(sites) + 10 - min(sites) % 10, max(sites), 10)
    plt.xticks(xticks)


def zoom_into_trs4(sites, estimates, ax=None):

    ax.axhline(0, linestyle='-', color='gray', lw=2, alpha=0.5)
    ax.plot(sites, estimates, linestyle='-', marker='o',
            markersize=10, linewidth=1.5, color='green', alpha=0.5)

    ax.set_ylim((-8, 2))
    ax.set_xlim((sites.min(), sites.max()))
    ax.set_xlabel('Nucleotide position')
    ax.set_ylabel('Fitness estimate')

    # Add gene boundaries
    colors = cm.get_cmap('tab20', len(gene_boundaries))
    plot_gene_boundaries(ax, sites[0], sites[-1], -8, colors)

    # Add x-ticks and grid
    add_xticks(sites, sites[0], sites[-1])
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.grid(which='major', axis='x', linestyle='-', linewidth=0.5, color='black')
    ax.grid(which='minor', axis='x', linestyle=':', linewidth=0.5, color='gray')

    # Add TRS motif
    ax.text(26237, -7, 'A', color='red', ha='center', weight='bold')
    ax.text(26238, -7, 'C', color='red', ha='center', weight='bold')
    ax.text(26239, -7, 'G', color='red', ha='center', weight='bold')
    ax.text(26240, -7, 'A', color='red', ha='center', weight='bold')
    ax.text(26241, -7, 'A', color='red', ha='center', weight='bold')
    ax.text(26242, -7, 'C', color='red', ha='center', weight='bold')

    # Add subplot label
    ax.text(
        -0.06, 1.05,
        "C",
        transform=ax.transAxes,
        fontsize=20,
        verticalalignment='top',
        horizontalalignment='left'
    )


def plot_fitness_along_genome(sites, estimate, nt_lims, ax=None, shuffled_trace=False, label=""):

    if shuffled_trace:

        ax.plot(sites, estimate, color='gray', alpha=0.5, linewidth=0.8, zorder=1)

    else:

        ax.plot(sites, estimate, color='green', alpha=1, linewidth=0.8)

        ax.axhline(y=0, color='green', linestyle='-')

        ax.set_ylabel('Fitness estimate')

        ax.set_ylim((-4, 2))
        ax.set_xlim(nt_lims)

        # Add gene boundaries
        colors = cm.get_cmap('tab20', len(gene_boundaries))
        plot_gene_boundaries(ax, nt_lims[0], nt_lims[1], -4, colors, y_offset=0.5)

        # Add conserved regions
        _, conserved_regions = get_conserved_regions()
        conserved_regions = [(value[0], value[1], key) for key, value in conserved_regions.items()]
        conserved_regions[1], conserved_regions[2:] = conserved_regions[9], conserved_regions[1:9]
        plot_conserved_regions(ax, nt_lims[0], nt_lims[1], -4, conserved_regions, y_offset=1.5)

        # Add subplot label
        ax.text(
            -0.06, 1.05,
            label,
            transform=ax.transAxes,
            fontsize=20,
            verticalalignment='top',
            horizontalalignment='left'
        )


def get_smoothened_fitness_estimates(hyperparams, shuffle=False):

    # Train model on synonymous mutations
    df_train = load_synonymous_muts()
    model = GeneralLinearModel(included_factors=['global_context', 'rna_structure', 'local_context'])
    model.train(df_train=df_train)

    # Load mutation counts and add predicted counts
    df = load_synonymous_muts(include_noncoding_muts=True, include_stop_tolerant_orfs=True, remove_orf9b=True)
    model.add_predicted_counts(df)

    # Make an indicator list for sites which appear in the dataframe
    all_nt_sites = np.arange(df.nt_site.min(), df.nt_site.max() + 1)
    L = len(all_nt_sites)
    mask_rho = np.isin(all_nt_sites, df.nt_site)

    # Helper function to aggregate counts at a site
    def get_aggregated_fitness(group):
        actual_counts_sum = group['actual_count'].sum()
        predicted_counts_sum = group['predicted_count'].sum()
        return np.log(actual_counts_sum + 0.5) - np.log(predicted_counts_sum + 0.5)

    # Get observed fitness effect at all sites present in the dataframe
    g_i = np.zeros(L)
    g_i[mask_rho] = df.groupby('nt_site').apply(get_aggregated_fitness).values

    if shuffle:
        permutation = np.random.permutation(len(g_i))
        g_i = g_i[permutation]
        mask_rho = mask_rho[permutation]

    # Set sigma_i
    sigma_i = np.full((L,), hyperparams['sigma_s'])

    # Set rho_i
    rho_i = np.full((L,), hyperparams['sigma_f'][0])
    rho_i[~mask_rho] = hyperparams['sigma_f'][1]

    # Set tau_i
    tau_i = np.full((L,), hyperparams['sigma_n'])

    # Get smoothened fitness estimates
    f_i = solve_lse(sigma_i, rho_i, tau_i, g_i)

    return all_nt_sites, f_i, all_nt_sites[mask_rho], g_i[mask_rho]


if __name__ == '__main__':

    # Smoothen fitness estimates along genome
    hyperparams = {'sigma_n': 0.05, 'sigma_s': 1000, 'sigma_f': [0.2, 10000]}
    all_sites, all_estimates, noncoding_sites, noncoding_estimates = get_smoothened_fitness_estimates(hyperparams)

    # Smoothen shuffled fitness estimates along genome
    _, shuffled_estimates, _, _ = get_smoothened_fitness_estimates(hyperparams, shuffle=True)

    # Prepare figure with all three subplots
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 8), dpi=250)

    # Plot smoothened fitness estimates along the full genome (1st subplot)
    plot_fitness_along_genome(all_sites, all_estimates, nt_lims=(1, 29903), ax=axes[0], label="A")
    plot_fitness_along_genome(all_sites, shuffled_estimates, nt_lims=(1, 29903), ax=axes[0], shuffled_trace=True)

    # Plot smoothened fitness estimates (including shuffled ones) along the last third of the genome (2nd subplot)
    plot_fitness_along_genome(all_sites, all_estimates, nt_lims=(19500, 29903), ax=axes[1], label="B")
    plot_fitness_along_genome(all_sites, shuffled_estimates, nt_lims=(19500, 29903), ax=axes[1], shuffled_trace=True)

    # Plot aggregated fitness effects around TRS4 (3rd subplot)
    sites = noncoding_sites[(noncoding_sites > 26200) & (noncoding_sites < 26265)]
    estimates = noncoding_estimates[(noncoding_sites > 26200) & (noncoding_sites < 26265)]
    zoom_into_trs4(sites, estimates, ax=axes[2])

    # Display the combined plots
    plt.tight_layout()
    plt.savefig('../results/noncoding_constraint.pdf')
    plt.show()
