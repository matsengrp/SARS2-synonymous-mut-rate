import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

plt.rcParams.update({'font.size': 12})

letters = ['A', 'C', 'G', 'T']

mut_types = ['AC', 'AG', 'AT', 'CA', 'CG', 'CT', 'GA', 'GC', 'GT', 'TA', 'TC', 'TG']

contexts = ['AA', 'AC', 'AG', 'AT', 'CA', 'CC', 'CG', 'CT', 'GA', 'GC', 'GG', 'GT', 'TA', 'TC', 'TG', 'TT']

mut_type_cols = ['blue', 'orange', 'green', 'red', 'purple', 'brown',
                 'pink', 'gray', 'cyan', 'magenta', 'lime', 'teal']

plot_map = {'AC': (0, 0), 'CA': (0, 1), 'GA': (0, 2), 'TA': (0, 3),
            'AG': (1, 0), 'CG': (1, 1), 'GC': (1, 2), 'TC': (1, 3),
            'AT': (2, 0), 'CT': (2, 1), 'GT': (2, 2), 'TG': (2, 3)
            }


def add_predictions(dataframe):

    gen_lin_model = GeneralLinearModel(type='l_r', test_data=dataframe)

    # Get learned parameters
    gen_lin_model.W = pd.read_csv(f"../results/Basel/learned_params.csv").to_dict(orient='list')

    # Predict counts
    pred_counts = np.exp(gen_lin_model.predict_log_counts()) - 0.5

    # Save predicted counts
    dataframe['predicted_count_basel'] = pred_counts

    return dataframe


class GeneralLinearModel:

    def __init__(self, type, training_data=None, regularization=None, W=None, test_data=None):
        self.type = type  # type of one-hot encoding
        self.df_train = training_data
        self.reg_type = regularization[0] if regularization is not None else None
        self.reg_strength = regularization[1] if regularization is not None else None
        self.W = W  # learned parameters
        self.df_test = test_data

    def train(self):

        self.W = {}

        # Check which mutation types are present in the data
        present_mut_types = self.df_train['nt_mutation'].apply(lambda x: x[0] + x[-1])

        # Loop over all present mutation types
        for mut_type in present_mut_types.unique():

            # Get ancestral and mutated nucleotide
            nt1, nt2 = mut_type[0], mut_type[1]

            # Filter current mutation type
            df_local = self.df_train[self.df_train['nt_mutation'].str.match('^' + nt1 + '.*' + nt2 + '$')]

            # Get log counts
            log_counts = np.log(df_local['mut_count'].values + 0.5).reshape(-1, 1)  # dimensions (# of sites, 1)

            # Create data matrix
            X, _ = self.create_data_matrix(df_local.copy())  # (# of sites, # of parameters in model), _

            # Perform regression
            if self.reg_type == 'l1':
                lasso = Lasso(alpha=self.reg_strength, fit_intercept=False)
                lasso.fit(X, log_counts)
                self.W[mut_type] = lasso.coef_
            elif self.reg_type == 'l2':
                w = np.linalg.inv(X.T @ X + self.reg_strength * np.identity(X.shape[1])) @ X.T @ log_counts
                self.W[mut_type] = w

        # Save learned parameters
        params = {key: value.flatten() for key, value in self.W.items()}
        params_df = pd.DataFrame(params)
        directory = f"../results/Basel"

        if not os.path.exists(directory):
            os.makedirs(directory)

        filepath = os.path.join(directory, 'learned_params.csv')
        params_df.to_csv(filepath, index=False, header=True)

        # Plot observed vs. predicted mean log(counts + 0.5) for every context and pairing state
        self.plot_fit_vs_data()

    def create_data_matrix(self, mut_counts_df):

        # Get pairing state and context
        unpaired = 1 - mut_counts_df['basepair'].values
        context_l = mut_counts_df['motif'].apply(lambda x: x[0]).values
        context_r = mut_counts_df['motif'].apply(lambda x: x[2]).values

        # First column is equal to 1
        base = np.full(len(unpaired), 1)

        # Construct remaining columns according to selected type of one-hot encoding
        columns = [base]
        if self.type == 'p_up':
            columns += [unpaired]
        elif self.type == 'l_r':
            columns += [unpaired] + self.one_hot_l_r(context_l, context_r)
        elif self.type == 'l_r_st':
            columns += [unpaired] + self.one_hot_l_r_st(context_l, context_r, unpaired)
        elif self.type == 'lr':
            columns += [unpaired] + self.one_hot_lr(context_l, context_r)
        elif self.type == 'lr_pup':
            columns += self.one_hot_lr_pup(context_l, context_r, unpaired)

        # Compile data matrix
        X = np.column_stack(columns)

        indices = {}
        present_mut_types = mut_counts_df['nt_mutation'].apply(lambda x: x[0] + x[-1])
        for mut_type in present_mut_types.unique():
            indices[mut_type] = np.where(present_mut_types == mut_type)

        return X, indices

    @staticmethod
    def one_hot_l_r(context_l, context_r):
        sigma = {}
        for nt in ['C', 'G', 'T']:
            sigma[nt + '_l'] = (context_l == nt).astype(int)
            sigma[nt + '_r'] = (context_r == nt).astype(int)

        return [sigma['C_l'], sigma['G_l'], sigma['T_l'],
                sigma['C_r'], sigma['G_r'], sigma['T_r']]

    @staticmethod
    def one_hot_l_r_st(context_l, context_r, unpaired):
        sigma = {}
        for nt in letters:
            for state in [0, 1]:
                st = 'p' if state == 0 else 'up'
                sigma[nt + '_l_' + st] = ((context_l == nt) * (unpaired == state)).astype(int)
                sigma[nt + '_r_' + st] = ((context_r == nt) * (unpaired == state)).astype(int)

        return [sigma['A_l_up'], sigma['C_l_p'], sigma['C_l_up'], sigma['G_l_p'], sigma['G_l_up'],
                sigma['T_l_p'], sigma['T_l_up'], sigma['A_r_up'], sigma['C_r_p'], sigma['C_r_up'],
                sigma['G_r_p'], sigma['G_r_up'], sigma['T_r_p'], sigma['T_r_up']]

    @staticmethod
    def one_hot_lr(context_l, context_r):
        sigma = {}
        for nt1 in letters:
            for nt2 in letters:
                sigma[nt1 + nt2] = (context_l + context_r == nt1 + nt2).astype(int)

        return [sigma['AC'], sigma['AG'], sigma['AT'], sigma['CA'], sigma['CC'], sigma['CG'],
                sigma['CT'], sigma['GA'], sigma['GC'], sigma['GG'], sigma['GT'], sigma['TA'],
                sigma['TC'], sigma['TG'], sigma['TT']]

    @staticmethod
    def one_hot_lr_pup(context_l, context_r, unpaired):
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

    def predict_log_counts(self):

        # Create data matrix
        X, indices = self.create_data_matrix(self.df_test.copy())

        predicted_log_counts = np.zeros(X.shape[0])

        # Predict log counts
        present_mut_types = self.df_test['nt_mutation'].apply(lambda x: x[0] + x[-1])
        for mut_type in present_mut_types.unique():
            indices = np.where(present_mut_types == mut_type)
            predicted_log_counts[indices] = X[indices] @ self.W[mut_type]

        return predicted_log_counts

    def plot_fit_vs_data(self):

        # Prepare figure with the same aspect ratio as Keynote slides
        fig, axes = plt.subplots(3, 4, figsize=(16, 9))

        df_local = self.df_train.copy()

        # Loop over all 12 mutation types
        for mut_type in mut_types:

            df = df_local[df_local['nt_mutation'].str.match('^' + mut_type[0] + '.*' + mut_type[1] + '$')]

            # Select correct figure for this mutation type
            ax = axes[plot_map[mut_type]]

            # Prepare dictionaries with contexts as keys
            observed_mean_paired, observed_mean_unpaired = {}, {}
            predicted_mean_paired, predicted_mean_unpaired = {}, {}

            # Loop over all 16 contexts
            for ctxt in contexts:

                # Set context
                context = ctxt[0] + mut_type[0] + ctxt[1]

                # Choose sites
                df_paired = df[(df['motif'] == context) & (df['basepair'] == 1)]
                df_unpaired = df[(df['motif'] == context) & (df['basepair'] == 0)]

                # Get mean log counts for every context/pairing state
                observed_mean_paired[context] = np.mean(np.log(df_paired['mut_count'].values + 0.5)) if len(
                    df_paired) > 0 else 0
                observed_mean_unpaired[context] = np.mean(np.log(df_unpaired['mut_count'].values + 0.5)) if len(
                    df_unpaired) > 0 else 0

                nt_l, nt_r = context[0], context[2]

                x_p = np.array([1])
                x_up = np.array([1])
                if self.type == 'p_up':
                    x_p = np.array([1, 0]).astype(int)
                    x_up = np.array([1, 1]).astype(int)
                if self.type == 'l_r':
                    x_p = np.array([1, 0, nt_l == 'C', nt_l == 'G', nt_l == 'T',
                                    nt_r == 'C', nt_r == 'G', nt_r == 'T']).astype(int)
                    x_up = np.array([1, 1, nt_l == 'C', nt_l == 'G', nt_l == 'T',
                                     nt_r == 'C', nt_r == 'G', nt_r == 'T']).astype(int)
                if self.type == 'l_r_st':
                    x_p = np.array([1, 0, 0, nt_l == 'C', 0, nt_l == 'G', 0, nt_l == 'T', 0,
                                    0, nt_r == 'C', 0, nt_r == 'G', 0, nt_r == 'T', 0]).astype(int)
                    x_up = np.array([1, 1, nt_l == 'A', 0, nt_l == 'C', 0, nt_l == 'G', 0, nt_l == 'T',
                                     nt_r == 'A', 0, nt_r == 'C', 0, nt_r == 'G', 0, nt_r == 'T']).astype(int)
                if self.type == 'lr':
                    k = nt_l + nt_r
                    x_p = np.array([1, 0, k == 'AC', k == 'AG', k == 'AT', k == 'CA', k == 'CC',
                                    k == 'CG', k == 'CT', k == 'GA', k == 'GC', k == 'GG', k == 'GT',
                                    k == 'TA', k == 'TC', k == 'TG', k == 'TT']).astype(int)
                    x_up = np.array([1, 1, k == 'AC', k == 'AG', k == 'AT', k == 'CA', k == 'CC',
                                     k == 'CG', k == 'CT', k == 'GA', k == 'GC', k == 'GG', k == 'GT',
                                     k == 'TA', k == 'TC', k == 'TG', k == 'TT']).astype(int)
                if self.type == 'lr_pup':
                    k = nt_l + nt_r
                    x_p = np.array([1, 0, k == 'AC', 0, k == 'AG', 0, k == 'AT', 0, k == 'CA', 0, k == 'CC', 0,
                                    k == 'CG', 0, k == 'CT', 0, k == 'GA', 0, k == 'GC', 0, k == 'GG', 0, k == 'GT',
                                    0, k == 'TA', 0, k == 'TC', 0, k == 'TG', 0, k == 'TT', 0]).astype(int)
                    x_up = np.array([1, k == 'AA', 0, k == 'AC', 0, k == 'AG', 0, k == 'AT', 0,
                                     k == 'CA', 0, k == 'CC', 0, k == 'CG', 0, k == 'CT', 0, k == 'GA',
                                     0, k == 'GC', 0, k == 'GG', 0, k == 'GT', 0, k == 'TA', 0, k == 'TC',
                                     0, k == 'TG', 0, k == 'TT']).astype(int)

                predicted_mean_paired[context] = (self.W[mut_type].T @ x_p)[0]
                predicted_mean_unpaired[context] = (self.W[mut_type].T @ x_up)[0]

            keys = [s[0] + '_' + s[2] for s in predicted_mean_paired.keys()]
            obs_mu_p = observed_mean_paired.values()
            obs_mu_up = observed_mean_unpaired.values()
            pred_mu_p = predicted_mean_paired.values()
            pred_mu_up = predicted_mean_unpaired.values()
            ax.bar(np.arange(16) - 0.2, obs_mu_p, tick_label=keys, width=0.4, color='blue', alpha=0.7,
                   label='paired')
            ax.bar(np.arange(16) + 0.2, obs_mu_up, tick_label=keys, width=0.4, color='red', alpha=0.7,
                   label='unpaired')
            ax.bar(np.arange(16) - 0.2, pred_mu_p, tick_label=keys, width=0.4, color='none', edgecolor='black',
                   alpha=0.7, label='predicted', hatch="//")
            ax.bar(np.arange(16) + 0.2, pred_mu_up, tick_label=keys, width=0.4, color='none', edgecolor='black',
                   alpha=0.7, hatch="//")

            ax.set_xticks(np.arange(16), keys, rotation='vertical', fontsize=10)
            ax.set_title(rf"{mut_type[0]}$\rightarrow${mut_type[1]}")
            if plot_map[mut_type] == (0, 0):
                ax.legend()
            if plot_map[mut_type][1] == 0:
                ax.set_ylabel('mean log(n + 0.5)')

        plt.suptitle(f'model type: {self.type}, reg. type: {self.reg_type}, reg. strength: {self.reg_strength}')
        plt.tight_layout()
        plt.savefig(f'../results/Basel/predicted_vs_observed_means.png')
        plt.close()
