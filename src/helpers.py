import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

letters = ['A', 'C', 'G', 'T']


def load_synonymous_muts(include_noncoding_muts=False, include_stop_tolerant_orfs=False, remove_orf9b=False):

    # Read in curated dataset
    df = pd.read_csv('../results/curated_mut_counts.csv')

    # Transform paired/unpaired information into 0/1 format
    df['unpaired'] = (df['ss_prediction'] == 'unpaired').values.astype(int)

    # Create masks for synonymous mutations, non-coding mutations, stop-codon tolerant ORFs, and ORF9b
    mask_synonymous = (df['synonymous'] == True).values
    mask_noncoding = (df['noncoding'] == True).values
    stop_tolerant_orfs = ['ORF6', 'ORF7a', 'ORF7b', 'ORF8', 'ORF10']
    pattern = '|'.join(stop_tolerant_orfs)
    mask_tolerant = (df['gene'].str.contains(pattern)).values
    mask_orf9b = ((df['gene'] == 'N;ORF9b') & (df['clade_founder_aa'].apply(lambda x: x[0]) == df['mutant_aa'].apply(lambda x: x[0]))).values

    # Apply masks
    if include_noncoding_muts:
        mask_synonymous = mask_synonymous + mask_noncoding
    if include_stop_tolerant_orfs:
        mask_synonymous = mask_synonymous + mask_tolerant
    if remove_orf9b:
        mask_synonymous = mask_synonymous + mask_orf9b

    return df[mask_synonymous]


def load_nonsynonymous_muts():

    # Read in curated dataset
    df = pd.read_csv('../results/curated_mut_counts.csv')

    # Transform paired/unpaired information into 0/1 format
    df['unpaired'] = (df['ss_prediction'] == 'unpaired').values.astype(int)

    # Create masks for synonymous mutations, non-coding mutations, stop-codon tolerant ORFs, and ORF9b
    mask_nonsynonymous = (df.mut_class == 'nonsynonymous').values

    return df[mask_nonsynonymous]


def load_nonexcluded_muts():

    # Read in curated dataset
    df = pd.read_csv('../results/curated_mut_counts.csv')

    # Transform paired/unpaired information into 0/1 format
    df['unpaired'] = (df['ss_prediction'] == 'unpaired').values.astype(int)

    return df


class GeneralLinearModel:

    def __init__(self, included_factors, interacting=False, W=None, tau_squared=None):

        self.included_factors = included_factors
        self.interacting = interacting
        self.W = W
        self.tau_squared = tau_squared

    def train(self, df_train):

        # Initialise dictionaries for model parameters and remaining variances
        self.W = {}
        self.tau_squared = {}

        # Check which mutation types are present in the training data
        present_mut_types = df_train['nt_mutation'].apply(lambda x: x[0] + x[-1]).unique()

        # Make a separate fit for every mutation type
        for mut_type in present_mut_types:

            # Define ancestral and mutated nucleotide
            nt1, nt2 = mut_type[0], mut_type[1]

            # Select the current mutation type
            df_mut_type = df_train[df_train['nt_mutation'].str.match('^' + nt1 + '.*' + nt2 + '$')]

            # Prepare log-counts, dimensions: (# of sites, 1)
            log_counts = np.log(df_mut_type['actual_count'].values + 0.5).reshape(-1, 1)

            # Create data matrix X, dimensions: (# of sites, # of parameters in model)
            X = self.create_data_matrix(df_mut_type.copy(), mut_type)

            # Fit a linear model as log_counts = w @ X using a mean squared loss with a l2-regularization
            regularization_strength = 0.1
            regularization_matrix = regularization_strength * np.identity(X.shape[1])
            regularization_matrix[0, 0] = 0  # (don't regularize the offset term)
            w = np.linalg.inv(X.T @ X + regularization_matrix) @ X.T @ log_counts

            # Store model parameters
            self.W[mut_type] = w

            # Store remaining variance on log counts
            self.tau_squared[mut_type] = mean_squared_error(log_counts, X @ w)

    def create_data_matrix(self, mut_counts_df, mut_type):

        factor_cols = {}

        # Get global context
        if mut_type in ['AT', 'CG', 'GC']:
            factor_cols['global_context'] = [(mut_counts_df['nt_site'] > 21562).values.astype(int)]
        elif mut_type in ['CT']:
            factor_cols['global_context'] = [(mut_counts_df['nt_site'] > 13467).values.astype(int)]
        else:
            factor_cols['global_context'] = None

        # Get RNA structure
        factor_cols['rna_structure'] = [mut_counts_df['unpaired'].values]

        # Get left and right context
        left_context = mut_counts_df['motif'].apply(lambda x: x[0]).values
        right_context = mut_counts_df['motif'].apply(lambda x: x[2]).values
        factor_cols['local_context'] = self.one_hot_l_r(left_context, right_context)

        # Offset term
        base = np.full(len(mut_counts_df), 1)

        # Add columns depending on which model is fitted and which factors are included
        columns = [base]
        if self.interacting:  # TODO: make this work
            if mut_type in ['AT', 'CG', 'GC', 'CT']:
                columns += self.one_hot_lrp(left_context, right_context, 1)
            else:
                columns += self.one_hot_lrp(left_context, right_context, 1)
        else:
            for factor in self.included_factors:
                if factor_cols[factor] is not None:
                    columns += factor_cols[factor]

        # Compile data matrix
        X = np.column_stack(columns)

        return X

    def test(self, df_test):

        mean_sq_errs = {}

        # Check which mutation types are present in the training data
        present_mut_types = df_test['nt_mutation'].apply(lambda x: x[0] + x[-1]).unique()

        # Compute mean squared error separately for every mutation type
        for mut_type in present_mut_types:

            # Define ancestral and mutated nucleotide
            nt1, nt2 = mut_type[0], mut_type[1]

            # Select the current mutation type
            df_mut_type = df_test[df_test['nt_mutation'].str.match('^' + nt1 + '.*' + nt2 + '$')]

            # Prepare log-counts, dimensions: (# of sites, 1)
            log_counts = np.log(df_mut_type['actual_count'].values + 0.5).reshape(-1, 1)

            # Create data matrix X, dimensions: (# of sites, # of parameters in model)
            X = self.create_data_matrix(df_mut_type.copy(), mut_type)

            # Compute the mean squared error of the fitted model on the training data
            mean_sq_errs[mut_type] = mean_squared_error(log_counts, (X @ self.W[mut_type]).flatten())

        return mean_sq_errs

    def add_predicted_counts(self, df):

        # Check which mutation types are present in the data
        present_mut_types = df['nt_mutation'].apply(lambda x: x[0] + x[-1]).unique()

        # Prepare array for predicted counts and remaining variance
        predicted_counts = np.empty(len(df))
        remaining_variance = np.empty(len(df))

        # Compute mean squared error separately for every mutation type
        for mut_type in present_mut_types:

            # Define ancestral and mutated nucleotide
            nt1, nt2 = mut_type[0], mut_type[1]

            # Select the current mutation type
            mask_mut_type = df['nt_mutation'].str.match('^' + nt1 + '.*' + nt2 + '$').values
            df_mut_type = df[mask_mut_type]

            # Create data matrix X, dimensions: (# of sites, # of parameters in model)
            X = self.create_data_matrix(df_mut_type.copy(), mut_type)

            # Compute the mean squared error of the fitted model on the training data
            predicted_counts[mask_mut_type] = (X @ self.W[mut_type]).flatten()
            remaining_variance[mask_mut_type] = self.tau_squared[mut_type]

        df['predicted_count'] = np.exp(predicted_counts) - 0.5
        df['tau_squared'] = remaining_variance

    @staticmethod
    def one_hot_l_r(context_l, context_r):
        sigma = {}
        for nt in ['C', 'G', 'T']:
            sigma[nt + '_l'] = (context_l == nt).astype(int)
            sigma[nt + '_r'] = (context_r == nt).astype(int)

        return [sigma['C_l'], sigma['G_l'], sigma['T_l'],
                sigma['C_r'], sigma['G_r'], sigma['T_r']]

    @staticmethod
    def one_hot_lrp(context_l, context_r, unpaired):
        sigma = {}
        for nt1 in letters:
            for nt2 in letters:
                for state in [0, 1]:
                    st = 'p' if state == 0 else 'up'
                    sigma[nt1 + nt2 + '_' + st] = ((context_l + context_r == nt1 + nt2) * (unpaired == state)).astype(
                        int)

        return [sigma['AA_up'], sigma['AC_p'], sigma['AC_up'], sigma['AG_p'], sigma['AG_up'],
                sigma['AT_p'], sigma['AT_up'], sigma['CA_p'], sigma['CA_up'], sigma['CC_p'], sigma['CC_up'],
                sigma['CG_p'], sigma['CG_up'], sigma['CT_p'], sigma['CT_up'], sigma['GA_p'], sigma['GA_up'],
                sigma['GC_p'], sigma['GC_up'], sigma['GG_p'], sigma['GG_up'], sigma['GT_p'], sigma['GT_up'],
                sigma['TA_p'], sigma['TA_up'], sigma['TC_p'], sigma['TC_up'], sigma['TG_p'], sigma['TG_up'],
                sigma['TT_p'], sigma['TT_up']]


if __name__ == '__main__':

    print('test')
