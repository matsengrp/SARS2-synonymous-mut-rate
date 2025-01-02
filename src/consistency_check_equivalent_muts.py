import os
import math
import random
import itertools
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize

from helpers import load_nonexcluded_muts, load_synonymous_muts, GeneralLinearModel


def get_probabilistic_estimate(n_obs, n_pred, tau_squared, show_p_f_given_n_obs=False):

    # Define offset in the logarithm
    OFFSET = 0.000001

    # P(f) (normalized)
    def p_f(f):
        # Gaussian centered around 0 with standard deviation 2.5
        return norm.pdf(f, loc=0.0, scale=2.5)

    # P(n_exp) (normalized)
    c2 = 1 / math.sqrt(tau_squared * 2 * math.pi)
    c3 = 2 * tau_squared

    def p_n_exp(n_exp):
        # Log-normal distribution with mean log(n_pred) and variance tau^2
        return c2 * np.exp(-np.log((n_exp + OFFSET)/n_pred)**2 / c3) / (n_exp + OFFSET)

    # P(n_obs | f, n_exp) (normalized)
    n_obs_factorial = math.log(math.factorial(n_obs))

    def p_n_obs_given_f_n_exp(n_exp, f):
        # Poisson distribution with mean n_exp e^f
        log_p = n_obs * (np.log(n_exp + OFFSET) + f) - n_exp * np.exp(f) - n_obs_factorial
        return np.exp(log_p)

    # P(n_obs | f) (normalized)
    mean_log_normal = np.exp(np.log(n_pred) + tau_squared/2)
    std_log_normal = np.sqrt((np.exp(tau_squared) - 1) * np.exp(2*np.log(n_pred) + tau_squared))
    upper_bound_log_normal = mean_log_normal + 5 * std_log_normal
    lower_bound_log_normal = mean_log_normal - 5 * std_log_normal
    n_obs_plus_1 = (1 + n_obs)
    n_obs_plus_1_sqrt = np.sqrt(1 + n_obs)

    def p_n_obs_given_f(f):
        # integrand: P(n_exp) * P(n_obs | f, n_exp)
        integrand = lambda n_exp: p_n_exp(n_exp) * p_n_obs_given_f_n_exp(n_exp, f)

        # Set integral boundaries according to shape of involved distributions
        e_to_the_minus_f = np.exp(-f)
        mean_poisson = e_to_the_minus_f * n_obs_plus_1
        std_poisson = e_to_the_minus_f * n_obs_plus_1_sqrt
        lower_lim = max(0, max(lower_bound_log_normal, mean_poisson - 5 * std_poisson))
        upper_lim = min(upper_bound_log_normal, mean_poisson + 5 * std_poisson)

        # plot_integrand(f, lower_lim, upper_lim)

        # Perform integral
        n_points = 100
        xgrid = np.linspace(lower_lim, upper_lim, n_points)
        dx = (upper_lim-lower_lim)/(n_points-1)
        integral = np.sum(integrand(xgrid))*dx

        # integral = quad(func=integrand, a=lower_lim, b=upper_lim)[0]

        return integral

    # Plot P(n_exp) * P(n_obs | f, n_exp) as a function of n_exp
    def plot_integrand(f, lower_lim, upper_lim):
        # Define values of n_exp
        n_exp_values = np.linspace(lower_lim, upper_lim, 500)
        integrand_values = [p_n_exp(n_exp) * p_n_obs_given_f_n_exp(n_exp, f) for n_exp in n_exp_values]

        # Plot integrand
        plt.figure(figsize=(8, 4.5))
        plt.plot(n_exp_values, integrand_values, label=r'Integrand $p(n_{exp}) \cdot p(n_{obs}|n_{exp}, f)$')
        plt.xlabel('$n_{exp}$')
        plt.ylabel('Integrand')
        plt.title(f'f = {f}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # P(f | n_obs) (normalization 1/P(n_obs) omitted)
    def p_f_given_n_obs(f):
        return p_n_obs_given_f(f) * p_f(f)

    # f_hat = argmax_f p(f | n_obs)
    def get_f_hat():
        # Determine f_hat by finding the argmax of p_f_given_n_obs. Use the naive estimate as an initial guess.
        initial_guess = np.log(n_obs + 0.5) - np.log(n_pred + 0.5)
        # start = time.time()
        result = minimize(lambda f: -np.log(p_f_given_n_obs(f) + OFFSET), x0=initial_guess, method='BFGS')
        # end = time.time()
        # print(f"find f_max: {end-start}")
        return result.x[0]

    def characterize_p_f_given_nobs(f_max):

        # start = time.time()
        # Check where the distribution goes to zero
        f_values = np.linspace(f_max - 6, f_max + 5, 30)
        p_values = np.array([p_f_given_n_obs(f) for f in f_values])

        max_p_value = np.max(p_values)
        threshold = 0.0001 * max_p_value

        indices_above_threshold = np.where(p_values > threshold)[0]
        smallest_f_value = f_values[max(0, indices_above_threshold[0] - 1)]
        largest_f_value = f_values[min(indices_above_threshold[-1] + 1, len(p_values) - 1)]

        f_values = np.linspace(smallest_f_value, largest_f_value, 200)
        p_values = np.array([p_f_given_n_obs(f) for f in f_values])

        total_area = np.trapz(y=p_values, x=f_values)
        normalized_p_values = p_values / total_area

        f_mean = np.trapz(f_values * normalized_p_values, f_values)
        f_std = np.sqrt(np.trapz(normalized_p_values * (f_values - f_mean)**2, f_values))

        left_interval, right_interval = get_hdi(f_values, normalized_p_values)

        # Plot p_f_given_n_obs(f) to validate result
        if show_p_f_given_n_obs:
            plt.figure(figsize=(8, 4.5))
            plt.plot(f_values, p_values, label=r'$p(f|n_{obs})$')
            plt.axvline(x=f_hat, color='green', label=r'new $\hat{f}$')
            # plt.axvline(x=f_mean, color='blue', label=r'mean')
            orig_estim = np.log((n_obs + 0.5) / (n_pred + 0.5))
            plt.axvline(x=orig_estim, color='red', label=r'old $\hat{f}$')
            plt.fill_between(f_values, p_values, where=((f_values >= left_interval) & (f_values <= right_interval)),
                             color='gray', alpha=0.5)
            plt.xlabel('$f$')
            plt.legend()
            plt.grid(True)
            plt.title(rf"n_pred = {round(n_pred, 2)}, n_obs = {round(n_obs, 2)}, $\tau^{2}$ = {round(tau_squared, 2)}")
            plt.tight_layout()
            plt.show()

        return f_mean, f_max - left_interval, right_interval - f_max, f_std

    f_hat = get_f_hat()

    f_mean, std_minus, std_plus, f_std = characterize_p_f_given_nobs(f_hat)

    return f_hat, std_minus, std_plus, f_std


def get_hdi(x, p, coverage=0.682):

    # Calculate the contribution of all intervals
    dx = x[1] - x[0]
    dp = dx * (p[:-1] + (p[1:] - p[:-1]) / 2)

    # Sort x and corresponding dp in descending order of dp
    sorted_indices = np.argsort(-dp)
    x_sorted = x[sorted_indices]
    dp_sorted = dp[sorted_indices]

    # Compute cumulative sum of probabilities
    cumulative_prob = np.cumsum(dp_sorted)

    # Find the smallest set of points that cover the desired probability
    cutoff_idx = np.argmax(cumulative_prob >= coverage)
    hdi_x_values = x_sorted[:cutoff_idx + 1]

    # Get the smallest interval by taking the min and max of the selected x values
    interval = (min(hdi_x_values), max(hdi_x_values))

    return interval


def add_probabilistic_estimates(dataframe):

    # Prepare arrays to store probabilistic fitness estimates
    L = len(dataframe)
    f_max, left_conf_int, right_conf_int, f_st_dev = np.zeros(L), np.zeros(L), np.zeros(L), np.zeros(L)

    # Calculate probabilistic fitness estimate for every mutation in the dataframe
    for j, (i, row) in enumerate(dataframe.iterrows()):

        observed_count = row.actual_count
        predicted_count = row.predicted_count
        tau_squared = row.tau_squared

        f_max[j], left_conf_int[j], right_conf_int[j], f_st_dev[j] = (
            get_probabilistic_estimate(observed_count, predicted_count, tau_squared, show_p_f_given_n_obs=False))

        if j % int(L/10) == 0:
            print(j)

    # Add estimates to the dataframe
    (dataframe['f_max'], dataframe['left_conf_int'],
     dataframe['right_conf_int'], dataframe['f_st_dev']) = f_max, left_conf_int, right_conf_int, f_st_dev


def z_score_of_equivalent_muts():

    file_path = '../results/equivalent_aa_muts.csv'

    if os.path.exists(file_path):

        # Load dataframe from a CSV file
        df = pd.read_csv('../results/equivalent_aa_muts.csv')

    else:

        # Load non-excluded mutations and remove synonymous/non-coding ones for this analysis
        df_1 = load_nonexcluded_muts()
        df_1 = df_1[(~df_1.synonymous) & (~df_1.noncoding)]

        # Count occurrences of each 'mutant_aa' within each ('gene', 'codon_position') combination
        count_df = df_1.groupby(['gene', 'codon_site', 'mutant_aa']).size().reset_index(name='count')

        # Filter out 'mutant_aa' that appear only once within each ('gene', 'codon_position') combination
        valid_mutant_aa = count_df[count_df['count'] > 1].drop(columns='count')

        # Merge back to get rows of non-unique amino acid mutations
        df = pd.merge(df_1, valid_mutant_aa, on=['gene', 'codon_site', 'mutant_aa'])

        # Train a general linear model on synonymous mutations
        df_syn = load_synonymous_muts()
        model = GeneralLinearModel(included_factors=['global_context', 'rna_structure', 'local_context'])
        model.train(df_syn)

        # Add predicted counts, remaining variances, and naive fitness estimates
        model.add_predicted_counts(df)
        df['naive_f'] = np.log(df.actual_count + 0.5) - np.log(df.predicted_count + 0.5)

        # Add argmax and standard deviation of posterior
        add_probabilistic_estimates(df)

        # Save dataframe as a CSV file
        df.to_csv('../results/equivalent_aa_muts.csv', index=False)

    # Group by 'gene', 'codon_site', and 'mutant_aa'
    grouped = df.groupby(['gene', 'codon_site', 'mutant_aa'])

    # Initialize an empty list to store all computed z-scores
    z_scores = []
    differences = []
    std_devs =[]
    # Function to calculate the z-score for two mutations
    def calculate_z_score(row1, row2):
        sign_choice = random.choice([-1, 1])
        return sign_choice * (row1.f_max - row2.f_max) / (np.sqrt(row1.f_st_dev**2 + row2.f_st_dev**2))

    # Iterate through each combination of gene, codon_site, and mutant_aa
    for name, group in grouped:
        # Get all possible pairs of rows within the group
        for row1, row2 in itertools.combinations(group.itertuples(), 2):
            # if row1.nt_site != row2.nt_site:
            if 1 == 1:
                # Calculate the z_score for each pair and append to the list
                z_score = calculate_z_score(row1, row2)
                z_scores.append(z_score)
                differences.append(row1.f_max - row2.f_max)
                std_devs.append(np.sqrt(row1.f_st_dev**2 + row2.f_st_dev**2))

    # Calculate mean and standard deviation of calculated z-scores
    mean_zscore = np.mean(z_scores)
    std_zscore = np.std(z_scores)

    # Plot z-scores
    plt.figure(figsize=(8, 4), dpi=150)
    plt.text(0.95, 0.95, f'Mean: {mean_zscore:.3f}\nStd. Dev.: {std_zscore:.3f}',
             horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.5), fontsize=12)
    # plt.yscale('log')
    plt.hist(z_scores, bins=60, range=(-5, 5), edgecolor='black')
    plt.xlim(left=-5, right=+5)
    # plt.title('Equivalent non-synonymous mutations at different sites')
    # plt.title('All pairs of equivalent non-synonymous mutations')
    plt.xlabel(r'$(\hat{f}_{1} - \hat{f}_{2}) / \sqrt{\sigma_{1}^{2} + \sigma_{2}^{2}}$', fontsize=16)
    plt.ylabel('number of mutations', fontsize=16)
    plt.tight_layout()
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.savefig('../results/consistency_check_equivalent_muts.pdf')
    plt.show()

    plt.figure()
    plt.scatter(np.abs(differences), std_devs, s=1)
    plt.xlabel('Difference in fitness |f_1 - f_2|')
    plt.ylabel('Combined standard deviation of f_1 - f_2')

def show_example_posteriors(n):

    df = load_nonexcluded_muts()

    model = GeneralLinearModel(included_factors=['global_context', 'rna_structure', 'local_context'])
    model.train(df[df['synonymous']])

    df = df.sample(n)
    model.add_predicted_counts(df)

    for j, (i, row) in enumerate(df.iterrows()):

        observed_count = row.actual_count
        predicted_count = row.predicted_count
        tau_squared = row.tau_squared

        _, _, _, _ = get_probabilistic_estimate(observed_count, predicted_count, tau_squared, show_p_f_given_n_obs=True)


if __name__ == '__main__':

    show_example_posteriors(0)

    z_score_of_equivalent_muts()
