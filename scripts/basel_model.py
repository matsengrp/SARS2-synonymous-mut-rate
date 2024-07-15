import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})

mut_types = ['AC', 'AG', 'AT', 'CA', 'CG', 'CT', 'GA', 'GC', 'GT', 'TA', 'TC', 'TG']

contexts = ['AA', 'AC', 'AG', 'AT', 'CA', 'CC', 'CG', 'CT', 'GA', 'GC', 'GG', 'GT', 'TA', 'TC', 'TG', 'TT']

plot_map = {'AC': (0, 0), 'CA': (0, 1), 'GA': (0, 2), 'TA': (0, 3),
            'AG': (1, 0), 'CG': (1, 1), 'GC': (1, 2), 'TC': (1, 3),
            'AT': (2, 0), 'CT': (2, 1), 'GT': (2, 2), 'TG': (2, 3)
            }

lightswitch_mutations = ['AT', 'CG', 'GC']


class BaselModel:

    def __init__(self, W=None):
        self.W = W  # learned parameters

    def train(self, df_train):

        # Initialize dictionary which stores the learned parameters for every mutation type
        self.W = {}

        # Check which mutation types are present in the data
        present_mut_types = (df_train['nt_mutation'].apply(lambda x: x[0] + x[-1])).unique()

        # Loop over all present mutation types
        for nt1, nt2 in present_mut_types:

            # Filter current mutation type
            df_local = df_train[df_train['nt_mutation'].str.match('^' + nt1 + '.*' + nt2 + '$')]

            # Get log counts (y)
            y = np.log(df_local['mut_count'].values + 0.5).reshape(-1, 1)  # dimensions (# of sites, 1)

            # Create data matrix (X)
            X = self.create_data_matrix(df_local)  # (# of sites, # of parameters in model), _

            # Perform least squares regression with l2 regularization
            reg_strength = 0.1
            w = np.linalg.inv(X.T @ X + reg_strength * np.identity(X.shape[1])) @ X.T @ y
            self.W[nt1+nt2] = w

        # Plot observed vs. predicted mean log(counts + 0.5) for every context and pairing state
        # self.plot_fit_vs_data(df_train)

    def create_data_matrix(self, mut_counts_df):

        # Get pairing state (0: paired, 1: unpaired) and context
        unpaired = 1 - mut_counts_df['basepair'].values
        context_l = mut_counts_df['motif'].apply(lambda x: x[0]).values
        context_r = mut_counts_df['motif'].apply(lambda x: x[2]).values

        # Construct columns of data matrix
        base = np.full(len(unpaired), 1)
        columns = [base] + [unpaired] + self.one_hot_l_r(context_l, context_r)

        # Add information for position in the genome for AT, CG, and GC
        present_mut_type = mut_counts_df.mut_type.unique()[0]
        if present_mut_type in lightswitch_mutations:
            if 'site' in mut_counts_df.columns:
                before_lightswitch = (mut_counts_df.site < 21555).values
            elif 'nt_mutation' in mut_counts_df.columns:
                before_lightswitch = (mut_counts_df['nt_mutation'].apply(lambda x: int(x[1:-1])).values < 21555).astype(int)
            elif 'partition' in mut_counts_df.columns:
                before_lightswitch = mut_counts_df['partition'].values
            columns += [before_lightswitch]

        # Compile data matrix
        X = np.column_stack(columns)

        return X

    @staticmethod
    def one_hot_l_r(context_l, context_r):
        sigma = {}
        for nt in ['C', 'G', 'T']:
            sigma[nt + '_l'] = (context_l == nt).astype(int)
            sigma[nt + '_r'] = (context_r == nt).astype(int)

        return [sigma['C_l'], sigma['G_l'], sigma['T_l'],
                sigma['C_r'], sigma['G_r'], sigma['T_r']]

    def add_predictions(self, df):

        # Predict counts
        pred_counts = np.exp(self.predict_log_counts(df)) - 0.5

        # Save predicted counts in the dataframe
        df['predicted_count_basel'] = pred_counts

        return df

    def predict_log_counts(self, dataframe):

        # Prepare array to store predicted log counts
        predicted_log_counts = np.zeros(len(dataframe))

        # Get present mutation types
        if 'nt_mutation' in dataframe.columns:
            present_mut_types = dataframe['nt_mutation'].apply(lambda x: x[0] + x[-1])
        else:
            present_mut_types = dataframe['mut_type']

        # Predict log counts for all mutation types
        for mut_type in present_mut_types.unique():
            indices = np.where(present_mut_types == mut_type)
            X = self.create_data_matrix(dataframe.iloc[indices])
            predicted_log_counts[indices] = (X @ self.W[mut_type]).flatten()

        return predicted_log_counts

    def plot_fit_vs_data(self, df_train):

        # Prepare figure with the same aspect ratio as Keynote slides
        fig, axes = plt.subplots(3, 4, figsize=(16, 9))

        df_local = df_train.copy()

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
                x_p = np.array([1, 0, nt_l == 'C', nt_l == 'G', nt_l == 'T',
                                nt_r == 'C', nt_r == 'G', nt_r == 'T']).astype(int)
                x_up = np.array([1, 1, nt_l == 'C', nt_l == 'G', nt_l == 'T',
                                 nt_r == 'C', nt_r == 'G', nt_r == 'T']).astype(int)

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

        plt.tight_layout()
        plt.show()
        plt.close()
