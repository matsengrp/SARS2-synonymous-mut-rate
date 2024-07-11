import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar, root_scalar
import statsmodels.api as sm
from scipy.special import betainc
from itertools import permutations, product
import pickle
from Bio import SeqIO
from general_linear_models import GeneralLinearModel, add_predictions


# As long as you have the data files
# ../results/curated_mut_counts.csv and
# ../data/lan_2022/reference_seq.fasta,
# you can simply instantiate the class with SynonymousRates() then call the writeout
# method.


class SynonymousRates:
    """
    A class to organize and fit various models to estimate synonymous mutation rates
    and counts.

    Attributes:
        all_fitted_models (dict): A dictionary mapping a mutation type and model name
            to the model.
        all_rates_df (pd.DataFrame): A dataframe with all model rates for all
            combinations of mutation type, motif, lightswitch, and basepairing.
        bases (list): The nucleotide bases.
        lightswitch_boundary (int): The last site before the lightswitch.
        lightswitch_mut_types (list): The mutation types prefering a model with
            a term for sites past the lightswitch.
        min_motif_count (int): The minimum number of times a motif must appear at a site
            to include the motif when fitting models.
        model_columns (dict): A dictionary mapping a model name to the names of columns,
            of mut-by_site_df, used by the model.
        model_count (int): The number of different models.
        model_names (list): The names of the different models.
        model_row_condition (dict): A dictionary mapping a model name to the row
            conditions used to select rows, of mut_by_site_df, before fitting the model.
        rate_names (list): The names of the model rates, which are based on model names.
        rna_independent_mut_types (list): The mutation types prefering a model with
            separate motif rates for paired and unpaired sites.
        rna_structure_mut_types (list): The mutation types prefering a model with
            a term for paired sites.
        mut_by_site_dfs (dict): A dictionary mapping a mutation type to the
            mut_by_site_df restricted to that mutation type.
        mut_by_site_df (pd.DataFrame): The dataframe with mutation counts per site with
            motif data.
        motifs_to_fit (dict): A dictionary mapping a mutation type to the motifs
            included when fitting models.
        mut_test_dfs (dict): The entries of mut_by_site_dfs, split into testing data.
        mut_train_dfs (dict): The entries of mut_by_site_dfs, split into training data.
        mut_types (list): The mutation types, e.g. "AC".
        rate_preference (dict): A dictionary mapping a combination of mutation type,
            motif, lightswitch, and basepair to name of the prefered rate.
        site_motif_path (str): The path to the csv file with site, mutation type, motif,
            and mutation count data.
    """

    def __init__(
        self,
        mut_count_path="../results/curated_mut_counts.csv",
        sequence_path="../data/lan_2022/reference_seq.fasta",
        clip_data=True,
        clip_bottom=False,
        clip_percent=0.02,
        test_holdout=0.0,
    ):
        self.set_mut_count_path(mut_count_path)
        self.set_constants()
        self.set_model_parameters()
        self.set_models_for_mutations()

        self.load_sequence(sequence_path)
        self.load_dataframes(clip_data, clip_bottom, clip_percent)
        self.train_test_split(test_holdout)
        self.set_motifs_to_fit(on_full=True)
        self.set_model_preferences()

        # Add column 'predicted_count_basel' to the dataframe
        self.get_basel_predictions()

        self.fit_models()
        self.make_rates()
        self.make_scaling_factor()

    def get_basel_predictions(self):

        # Train general linear model
        gen_lin_model = GeneralLinearModel(type='l_r', training_data=self.mut_by_site_df.copy(), regularization=('l2', 0.1))
        gen_lin_model.train()
        self.basel_model = gen_lin_model

        # Add predicted counts to dataframe
        self.mut_by_site_df = add_predictions(self.mut_by_site_df.copy())

        # Add prediction to all subsets
        for mut_type in self.mut_types:
            self.mut_by_site_dfs[mut_type] = add_predictions(self.mut_by_site_dfs[mut_type].copy())
            self.mut_train_dfs[mut_type] = add_predictions(self.mut_train_dfs[mut_type].copy())
            self.mut_test_dfs[mut_type] = add_predictions(self.mut_test_dfs[mut_type].copy())

    def set_constants(self):
        """
        Set various instance attributes to default values.
        """
        self.bases = ["A", "C", "G", "T"]
        self.set_min_motif_count()

    def set_mut_count_path(self, file_path):
        """
        Set the file path for the csv with per site mutation data.
        """
        self.site_motif_path = file_path

    def set_min_motif_count(self, count=10):
        """
        Sets the miniminum number of sites required to include a motif when fitting
        a model.
        """
        self.min_motif_count = count

    def set_model_parameters(self):
        """
        Sets to default instance variables recording model names, which columns are used in
        a model, row conditions defining a subset of data when fitting a model, and
        names.
        """
        self.model_columns = {
            "factored": ["first_base", "last_base"],
            "factored_basepair": ["first_base", "last_base", "basepair"],
            "factored_lightswitch": ["first_base", "last_base", "partition"],
            "factored_basepair_lightswitch": [
                "first_base",
                "last_base",
                "basepair",
                "partition",
            ],
            "motif": ["motif"],
            "motif_basepair": ["motif", "basepair"],
            "motif_lightswitch": ["motif", "partition"],
            "motif_basepair_lightswitch": ["motif", "basepair", "partition"],
            "motif_basepair0": ["motif"],
            "motif_basepair1": ["motif"],
        }
        self.model_row_condition = {
            "factored": None,
            "factored_basepair": None,
            "factored_lightswitch": None,
            "factored_basepair_lightswitch": None,
            "motif": None,
            "motif_basepair": None,
            "motif_lightswitch": None,
            "motif_basepair_lightswitch": None,
            "motif_basepair0": ("basepair", 0),
            "motif_basepair1": ("basepair", 1),
        }
        self.model_names = list(self.model_columns.keys())
        self.model_count = len(self.model_names)
        self.rate_names = [f"{name}_rate" for name in self.model_names]
        return None

    def set_models_for_mutations(self):
        """
        Sets to default which mutation types prefer models accounting for lightswitch
        and basepair effects.
        """
        self.set_lightswith_mutations()
        self.set_basepair_mutations()
        self.set_basepair_independent_motif_mutations()
        return None

    def set_lightswith_mutations(self, mut_types=["AT", "CG", "GC"]):
        """
        Sets which mutation types prefer models with a binary variable for site position
        relative to the lightswitch.
        """
        self.lightswitch_mut_types = mut_types
        return None

    def set_basepair_mutations(
        self,
        mut_types=["AC", "AG", "AT", "CA", "CG", "GA", "GC", "GT", "TA", "TC", "TG"],
    ):
        """
        Set which mutations types prefer models with a binary variable for a site being
        paired or unpaired.
        """
        self.rna_structure_mut_types = mut_types
        return None

    def set_basepair_independent_motif_mutations(self, mut_types=["CT"]):
        """
        Set which mutation types prefer a model with separate motif rates for sites that
        are paired and sites that are unpaired.
        """
        self.rna_independent_mut_types = mut_types
        return None

    def load_sequence(self, file_path):
        """
        Load the sars2 genome sequence from a fasta file.
        """
        with open(file_path) as the_file:
            fasta_parser = SeqIO.parse(the_file, "fasta")
            self.s2genome = str(next(fasta_parser).seq)

    def motif_centered_at(self, site, base_count=1):
        """
        Return the motif, in the original sequence, centered at the given site and of
        length 2*base_count+1. Sites start at 1.
        """
        return self.s2genome[site - base_count - 1 : site + base_count]

    def load_dataframes(self, clip_data, clip_bottom, clip_percent):
        """
        Load the site mutation motif data and optionally clips mutation counts.
        """
        site_muts_df = pd.read_csv(self.site_motif_path).sort_values(by="nt_site")
        site_muts_df.reset_index(drop=True, inplace=True)
        site_muts_df.rename(
            columns={
                "nt_site": "site",
                "bte_counts": "mut_count",
                "counts": "mut_count",
                "actual_count": "mut_count",
                "nt_site_before_21555": "partition",
                "ss_prediction": "basepair",
            },
            inplace=True,
        )
        site_muts_df = site_muts_df[site_muts_df.synonymous]
        site_muts_df["motif"] = site_muts_df.site.apply(self.motif_centered_at)
        site_muts_df["partition"] = site_muts_df.partition.astype(int)
        site_muts_df["basepair"] = (site_muts_df.basepair == "paired").astype(int)
        site_muts_df["first_base"] = site_muts_df.motif.str[0]
        site_muts_df["last_base"] = site_muts_df.motif.str[-1]
        if clip_data:
            self.data_clip(site_muts_df, clip_bottom, clip_percent)

        self.mut_types = site_muts_df.mut_type.unique()
        self.mut_by_site_dfs = {
            mut_type: site_muts_df[site_muts_df.mut_type == mut_type].set_index("site")
            for mut_type in self.mut_types
        }
        self.mut_by_site_df = site_muts_df

        return None

    def set_motifs_to_fit(self, on_full=False):
        """
        Sets the motifs to use when fitting models, based on the the minimum required
        site count. This site count is taken over the full data set when on_full is
        True and overwise is over the training data.
        """
        the_dfs = self.mut_by_site_dfs if on_full else self.mut_train_dfs
        site_counts = {
            mut_type: the_dfs[mut_type].groupby(by="motif").size().to_dict()
            for mut_type in self.mut_types
        }
        min_count = self.min_motif_count
        self.motifs_to_fit = {
            mut_type: [
                motif for motif, c in site_counts[mut_type].items() if c >= min_count
            ]
            for mut_type in self.mut_types
        }

    def train_test_split(self, test_holdout):
        """
        Splits the mutation data into training and testing data. The data is split after
        accounting for mutation type. Specify test_holdout=0 to train on all of the
        data.
        """
        self.mut_test_dfs = {
            mut_type: the_df.sample(frac=test_holdout)
            for mut_type, the_df in self.mut_by_site_dfs.items()
        }
        self.mut_train_dfs = {
            mut_type: the_df.drop(self.mut_test_dfs[mut_type].index)
            for mut_type, the_df in self.mut_by_site_dfs.items()
        }
        return None

    def set_model_preferences(self):
        """
        Sets which rate is prefered for each combination of mutation type, motif,
        site before/after the lightswitch, and site paired/unpaired.
        """
        self.rate_preference = {
            (mut_type, f"{left}{mut_type[0]}{right}", partition, basepair): None
            for mut_type in map("".join, permutations(self.bases, 2))
            for left in self.bases
            for right in self.bases
            for partition in (0, 1)
            for basepair in (0, 1)
        }
        for mut_type, motif, partition, basepair in self.rate_preference:
            rate = "motif_" if motif in self.motifs_to_fit[mut_type] else "factored_"
            if mut_type in self.rna_independent_mut_types:
                rate += f"basepair{basepair}_"
            if mut_type in self.rna_structure_mut_types:
                rate += "basepair_"
            if mut_type in self.lightswitch_mut_types:
                rate += "lightswitch_"
            rate += "rate"
            self.rate_preference[(mut_type, motif, partition, basepair)] = rate

    def yX_of_prepped_df(self, prepped_df):
        """
        Make y and X from a dataframe of dummies.
        """
        y = prepped_df["mut_count"]
        X = prepped_df.drop(columns=["mut_count"])
        X = sm.add_constant(X)
        X = X.astype("float64")
        return y, X

    def make_yX(self, mut_type, motifs_to_use, columns_to_use, row_condition):
        """
        Makes the target vector y and data matrix X that are used to fit a model.
        """
        the_df = self.mut_train_dfs[mut_type].reset_index(drop=True)
        if motifs_to_use is not None:
            the_df = the_df[the_df.motif.isin(motifs_to_use)]
        if row_condition is not None:
            column, value = row_condition
            the_df = the_df[the_df[column] == value]
        the_df = the_df[["mut_count", *columns_to_use]]
        the_df = pd.get_dummies(the_df, columns=columns_to_use, drop_first=True)
        return self.yX_of_prepped_df(the_df)

    def full_nb_fit(self, y, X, max_alpha=5):
        "Optimize the Negative Binomial alpha parameter and fit the model."

        def nb_loglike_given_alpha(alpha):
            model_nb = sm.GLM(
                y, X, family=sm.families.NegativeBinomial(alpha=alpha)
            ).fit()
            return model_nb.llf

        # find the alpha value that maximizes the log-likelihood using derivative-free optimization
        res = minimize_scalar(
            lambda alpha: -nb_loglike_given_alpha(alpha),
            bounds=(0, max_alpha),
            method="bounded",
        )
        alpha = res.x
        model_nb = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha)).fit()
        return model_nb

    def make_and_fit_model(self, mut_type, model_name):
        """Creates and fits the specified model for the given mutation type."""
        columns = self.model_columns[model_name]
        row_condition = self.model_row_condition[model_name]
        if model_name.startswith("factor"):
            motifs = None
        else:
            motifs = self.motifs_to_fit[mut_type]
        y, X = self.make_yX(mut_type, motifs, columns, row_condition)
        return self.full_nb_fit(y, X)

    def fit_models(self):
        """Fits all models with the current data."""
        self.all_fitted_models = {
            (mut_type, model_name): self.make_and_fit_model(mut_type, model_name)
            for mut_type in self.mut_types
            for model_name in self.model_names
        }
        return None

    def estimate_nb_median(self, the_model, input_df):
        """
        Returns an estimate of the median of the negative binomial exponential
        distribution of the given model at the given inputs.
        """
        mean = the_model.predict(input_df, which="mean").squeeze()
        var = the_model.predict(input_df, which="var_unscaled").squeeze()
        p, r = mean / var, mean**2 / (var - mean)
        cdf_offset = lambda k: betainc(r, k + 1, p) - 0.5
        if cdf_offset(0) >= 0:
            return 0
        else:
            x0, x1 = 0, mean
            while cdf_offset(x1) < 0:
                x0 = x1
                x1 *= 2
            return root_scalar(cdf_offset, bracket=(x0, x1)).root

    def predict_median(self, row, model_name):
        """
        Returns the median of the distribution for the specified model at the DataFrame
        row (which specifies mutation type, motif, lightswitch, and basepairing).
        """
        model = self.all_fitted_models[(row.mut_type, model_name)]
        columns = tuple((f"{col}_{row[col]}" for col in self.model_columns[model_name]))
        predictor_names = model.model.exog_names
        input_dict = {col: 0.0 for col in predictor_names}
        input_dict["const"] = 1.0  # Set constant term for the intercept
        for col in columns:
            if col in input_dict:
                input_dict[col] = 1.0
        input_df = pd.DataFrame([input_dict])
        return self.estimate_nb_median(model, input_df)

    def make_rates(self):
        """
        Make the dataframe with every model rate for every combination of mutation type,
        motif, partition, and basepairing. Additionally, sets a column with the rate
        of the model prefered for that combination.
        """
        data = [
            (
                mut_type,
                f"{left}{mut_type[0]}{right}",
                left,
                right,
                partition,
                basepair,
                *[0] * self.model_count,
            )
            for mut_type in self.mut_types
            for left in self.bases
            for right in self.bases
            for partition in [0, 1]
            for basepair in [0, 1]
        ]
        columns = [
            "mut_type",
            "motif",
            "first_base",
            "last_base",
            "partition",
            "basepair",
            *self.rate_names,
        ]
        all_rates_df = pd.DataFrame(data, columns=columns)

        for model_name, rate_name in zip(self.model_names, self.rate_names):
            all_rates_df[rate_name] = all_rates_df.apply(
                self.predict_median, args=(model_name,), axis=1
            )

        row_to_rate = lambda row: row[
            self.rate_preference[(row.mut_type, row.motif, row.partition, row.basepair)]
        ]
        all_rates_df["prefered_rate"] = all_rates_df.apply(row_to_rate, axis=1)
        self.all_rates_df = all_rates_df
        return None

    def make_scaling_factor(self):
        """
        Add the scaling factor to the rates dataframe, which is the prefered model rate
        for a combination of mutation type, motif, lightswitch, and basepairing divided
        by the median mutation count for the mutation type.
        """
        medians = {
            mut_type: the_df.mut_count.median()
            for mut_type, the_df in self.mut_train_dfs.items()
        }

        medians = pd.DataFrame.from_dict(
            medians, orient="index", columns=["median_mut_count"]
        )
        medians = medians.reset_index().rename(columns={"index": "mut_type"})

        self.all_rates_df = self.all_rates_df.merge(medians, on="mut_type")
        self.all_rates_df["scaling_factor"] = (
            self.all_rates_df.prefered_rate / self.all_rates_df.median_mut_count
        )
        return None

    def write_out(self, base_path):
        """
        Writes to file the dataframe of site motif mutations (after optional clipping),
        the dataframe of all rates, and the dictionary of prefered rates.
        """
        self.mut_by_site_df.to_csv(base_path + "input_data.csv")
        self.all_rates_df.to_csv(base_path + "rates.csv")
        with open(base_path + "prefered_rates.pkl", "wb") as the_file:
            pickle.dump(self.rate_preference, the_file)

    def data_clip(self, the_df, clip_bottom=False, clip_percent=0.02):
        """
        Clip the mutation counts of the site motif mutation dataframe. The boundaries
        for clipping are determined per mutation type. By default we clip with only an
        upper bound, but allow for clipping with an upper and lower bound.
        """
        if clip_bottom:
            q1 = clip_percent / 2
            q2 = 1 - q1
            lower = the_df.groupby(by=["mut_type"]).mut_count.quantile(q=q1).to_dict()
            upper = the_df.groupby(by=["mut_type"]).mut_count.quantile(q=q2).to_dict()
            clipper = lambda row: min(
                max((lower[row.mut_type], row.mut_count)), upper[row.mut_type]
            )
        else:
            q = 1 - clip_percent
            upper = the_df.groupby(by=["mut_type"]).mut_count.quantile(q=q).to_dict()
            clipper = lambda row: min(row.mut_count, upper[row.mut_type])
        the_df["mut_count"] = the_df.apply(clipper, axis=1)
        return None

    def calc_R2(self, mut_df):
        """
        Returns the R^2 values, per mutation type, for the dataframe mut_df. This
        dataframe must have columns for mutation type, motif, partition, basepair, and
        mutation count. The mutation counts of mut_df are taken as the true values and the
        predicted values are given by multiplying the appropriate prefered rate by the
        median mutation count in mut_df.
        """
        join_columns = ["mut_type", "motif", "partition", "basepair"]
        rate_columns = [*join_columns, "scaling_factor"]
        mutation_columns = [*join_columns, "mut_count"]

        rates_df = self.all_rates_df[rate_columns]
        mutations_df = mut_df[mutation_columns].merge(
            rates_df, on=join_columns, how="left"
        )

        medians = mutations_df.groupby("mut_type").mut_count.median()
        mutations_df = mutations_df.merge(
            medians.rename("median_mut_count"), on="mut_type"
        )
        mutations_df["predicted_mut_count"] = (
            mutations_df.scaling_factor * mutations_df.median_mut_count
        )

        R2s = {}
        for mut_type in mutations_df.mut_type.unique():
            mut_view = mutations_df[mutations_df.mut_type == mut_type]
            ybar = mut_view.mut_count.mean()
            SSres = ((mut_view.mut_count - mut_view.predicted_mut_count) ** 2).sum()
            SStot = ((mut_view.mut_count - ybar) ** 2).sum()
            R2 = 1 - SSres / SStot
            R2s[mut_type] = R2
        return R2s

    def calc_R2_Basel(self, mut_df):
        """
        Returns the R^2 values, per mutation type, for the dataframe mut_df. This
        dataframe must have columns for mutation type, motif, partition, basepair, and
        mutation count. The mutation counts of mut_df are taken as the true values and the
        predicted values are given by multiplying the appropriate prefered rate by the
        median mutation count in mut_df.
        """

        R2s = {}
        for mut_type in mut_df.mut_type.unique():
            mut_view = mut_df[mut_df.mut_type == mut_type]
            ybar = mut_view.mut_count.mean()
            SSres = ((mut_view.mut_count - mut_view.predicted_count_basel) ** 2).sum()
            SStot = ((mut_view.mut_count - ybar) ** 2).sum()
            R2 = 1 - SSres / SStot
            R2s[mut_type] = R2
        return R2s

    def calc_R2_on_full(self, model):
        """
        Returns the per mutation type R^2 values on the full data (training + test).
        """
        mut_df = self.mut_by_site_df
        if model == 'Seattle':
            return self.calc_R2(mut_df)
        elif model == 'Basel':
            return self.calc_R2_Basel(mut_df)

    def calc_R2_on_train(self, model):
        """Returns the per mutation type R^2 values on the training data."""
        mut_df = pd.concat(
            [
                the_df.reset_index(names="site")
                for the_df in self.mut_train_dfs.values()
            ],
            ignore_index=True,
        )
        if model == 'Seattle':
            return self.calc_R2(mut_df)
        elif model == 'Basel':
            return self.calc_R2_Basel(mut_df)

    def calc_R2_on_test(self, model):
        """Returns the per mutation type R^2 values on the test data."""
        mut_df = pd.concat(
            [the_df.reset_index(names="site") for the_df in self.mut_test_dfs.values()],
            ignore_index=True,
        )
        if model == 'Seattle':
            return self.calc_R2(mut_df) if not mut_df.empty else {}
        elif model == 'Basel':
            return self.calc_R2_Basel(mut_df) if not mut_df.empty else {}

    def calc_R2_on_poisson_simulated(self, model, seed=None):
        """
        Returns the per mutation type R^2 values on a simulated data set. The simulated
        data is given by taking the prefered rate for a combination of mutation type,
        motif, lightswitch, and basepair and sampling from a Poisson distribution, whose
        mean is that prefered rate, the number of times the combination appears in the
        training data.
        """
        rng = np.random.default_rng(seed)
        simulated_data = []
        prod = product(self.mut_types, self.bases, self.bases, [0, 1], [0, 1])
        for mut_type, left, right, partition, basepair in prod:
            motif = f"{left}{mut_type[0]}{right}"

            if model == 'Seattle':
                rate_row = self.all_rates_df[
                    (self.all_rates_df.mut_type == mut_type)
                    & (self.all_rates_df.motif == motif)
                    & (self.all_rates_df.partition == partition)
                    & (self.all_rates_df.basepair == basepair)
                ]
                rate = rate_row.prefered_rate
            elif model == 'Basel':
                # Get one-hot encoding
                one_hot = [1, 1-basepair] + self.basel_model.one_hot_l_r(np.array(left), np.array(right))
                rate = np.exp((self.basel_model.W[mut_type].T@one_hot)[0]) - 0.5

            data_df = self.mut_train_dfs[mut_type]
            data_row = data_df[
                (data_df.motif == motif)
                & (data_df.partition == partition)
                & (data_df.basepair == basepair)
            ]
            sample_count = len(data_row)

            # For Poisson, the mean and median are close.
            samples = rng.poisson(rate, sample_count)
            for sample in samples:
                simulated_data.append((mut_type, motif, partition, basepair, sample))

        mut_df = pd.DataFrame(
            simulated_data,
            columns=["mut_type", "motif", "partition", "basepair", "mut_count"],
        )
        if model == 'Seattle':
            return self.calc_R2(mut_df)
        elif model == 'Basel':
            return self.calc_R2_Basel(mut_df)
