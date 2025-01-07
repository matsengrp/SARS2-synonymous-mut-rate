from synonymous_rates_nn import *
from neutral_nn_utils import KMerData, pick_device, train_model, r2_for_model
import numpy as np
import pandas as pd
import torch
import torch.nn
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from Bio import SeqIO
from math import ceil
from itertools import permutations


class NNComparison:
    """
    A helper class to compare NN models.

    Attributes:
        base_names (list): The names of models (without extra features) that will be
            trained and tested.
        counts_df_with_splits (pd.DataFrame): A pandas dataframe of the curated mutation
            counts marked with train-test splits.
        ls_names (list): The names of models (with the extra feature for global genomic
            position) that will be trained and tested.
        ls_rna_names (list): The names of models (with the extra features for global
            genomic position and secondary rna stucture) that will be trained and
            tested.
        log_counts (bool): When true, the models work with the logarithm of the mutation
            counts plus as pseudocount. Otherwise, the models use the mutation counts.
        make_ds_args (dict): A dictionary mapping an NN model name to method arguments
            for constructing an S2IndicesDataset.
        make_ds_method (dict): A dictionary that maps a known model name (i.e, entries
            of base_names, ls_names, ls_rna_names, and rna_names) to the method for
            generating a k-mer dataset, for the appropriate k.
        max_k (int): The maximum motif length. Models are included for all motif lengths
            up to and including this value.
        model_maker (dict): A dictionary mapping an NN model name to an init method with
            filled in parameters.
        motif_data (dict): A dictionary storing KMerData objects for each motif length.
        n_splits (int): The number of train-test splits.
        rna_names (list): The names of models (with the extra feature for secondary rna
            stucture) that will be trained and tested.
        seq (str): The reference genome sequence.
    """

    def __init__(
        self,
        seq_path="../data/lan_2022/reference_seq.fasta",
        mut_count_path="../results/split_syn_mut_counts.csv",
        log_counts=True,
        max_k=7,
    ):
        self.max_k = max_k
        self.log_counts = log_counts
        with open(seq_path) as the_file:
            fasta_parser = SeqIO.parse(the_file, "fasta")
            fasta_sequences = [fasta for fasta in fasta_parser]
        self.seq = str(fasta_sequences[0].seq)

        self.counts_df_with_splits = pd.read_csv(mut_count_path)
        self.n_splits = 1 + max(
            (
                int(col[11:])
                for col in self.counts_df_with_splits.columns
                if col.startswith("train_split")
            )
        )
        self.motif_data = {i: KMerData(i) for i in range(1, max_k + 1, 2)}
        self.set_dicts()

    def default_model_preferences(self):
        """
        Return a dictionary that maps a mutation type to the string description of
        additional features ("rna" for pairing of sites, "ls" for global genomic
        position) used when modeling counts of that type.
        """
        base_types = ()
        rna_types = ("AC", "AG", "CA", "GA", "GT", "TA", "TC", "TG")
        ls_types = ()
        ls_rna_types = ("AT", "CG", "CT", "GC")
        features_for_mut_type = {mut_type: "" for mut_type in base_types}
        features_for_mut_type |= {mut_type: "rna" for mut_type in rna_types}
        features_for_mut_type |= {mut_type: "ls" for mut_type in ls_types}
        features_for_mut_type |= {mut_type: "ls+rna" for mut_type in ls_rna_types}
        return features_for_mut_type

    def set_dicts(self):
        """
        Initialize the dictionary attributes make_ds_args, model_maker,
        mut_type_to_stats_rate_name, and stats_rate_name_to_nn_model_names.
        """
        r = " + rna"
        l = " + lightswitch"
        seq = self.seq
        lc = self.log_counts
        max_k = self.max_k

        model_maker_args = {}
        for k in range(1, 9, 2):
            for ell in range(1, k + 1, 2):
                name = f"{k}_sites_by_{ell}"
                model_maker_args[name] = (4**ell, 4, k - ell + 1, 0.0, lc)
                model_maker_args[f"{name}{r}"] = (4**ell, 4, k - ell + 1, 0, 1, lc)
                model_maker_args[f"{name}{l}"] = (4**ell, 4, k - ell + 1, 0.0, 1, lc)
                model_maker_args[f"{name}{l}{r}"] = (4**ell, 4, k - ell + 1, 0.0, 2, lc)
        model_maker_method = {
            key: S2FlatNNModel if len(value) == 5 else S2FlatNNPlusFeaturesModel
            for key, value in model_maker_args.items()
        }

        def function_trick(name):
            def fn():
                return model_maker_method[name](*model_maker_args[name])

            return fn

        self.model_maker = {key: function_trick(key) for key in model_maker_args}

        get_ell = lambda x: int(x[x.index("by_") + 3 :].split()[0])
        self.make_ds_method = {
            name: self.motif_data[get_ell(name)].kmer_dataset_for_mut_df
            for name in self.model_maker
        }

        name = "{}_sites_by_{}".format
        self.make_ds_args = (
            {
                name(k, ell): ("+", seq, None, k - ell + 1, False, False, lc)
                for k in range(1, max_k + 1, 2)
                for ell in range(1, k + 1, 2)
            }
            | {
                f"{name(k, ell)}{r}": ("+", seq, None, k - ell + 1, False, True, lc)
                for k in range(1, max_k + 1, 2)
                for ell in range(1, k + 1, 2)
            }
            | {
                f"{name(k, ell)}{l}": ("+", seq, None, k - ell + 1, True, False, lc)
                for k in range(1, max_k + 1, 2)
                for ell in range(1, k + 1, 2)
            }
            | {
                f"{name(k, ell)}{l}{r}": ("+", seq, None, k - ell + 1, True, True, lc)
                for k in range(1, max_k + 1, 2)
                for ell in range(1, k + 1, 2)
            }
        )

        self.base_names = [
            f"{k}_sites_by_{ell}"
            for k in range(1, max_k + 1, 2)
            for ell in range(1, k + 1, 2)
        ]
        self.rna_names = [f"{name}{r}" for name in self.base_names]
        self.ls_names = [f"{name}{l}" for name in self.base_names]
        self.ls_rna_names = [f"{name}{l}{r}" for name in self.base_names]
        return None

    def make_comparison_df(
        self,
        mut_types=None,
        model_types=None,
        n_epochs=500,
        n_batches=1,
        device=None,
        features_for_mut_type=None,
    ):
        """
        Constructs a pandas DataFrame with columns "mut_type", "R2", "MAE", and
        "source", where R2 and MAE  are for the model and dataset described by source
        for the given mut_type.

        Args:
            mut_types (list): Optional list of mutation types (e.g., "AC") to include.
                When None, all 12 mutation types are included.
            model_types (list): Optional list of model types to use. Currently supported
                entries of this list are "stats", "kmer", "flatw3", and "flatw5". When
                None, all are used.
            reps (int): The number of times to indepedently compute the R^2 values.
            n_epochs (int): The number of epochs to train the neural network models.
            n_batches (int): The number of batches to split up the training data.
            device (torch.device): The pytorch device.
            features_for_mut_type (dict): ...mapping of mutation type to string that's
                "" for base model, "rna" for rna, "ls" for ligthswitch, and "rna+ls" or
                "ls+rna" for ls+rna.
        """
        if device is None:
            device = pick_device()
        if mut_types is None:
            mut_types = tuple(map("".join, permutations("AGCT", 2)))
        if model_types is None:
            model_types = self.base_names
        if features_for_mut_type is None:
            features_for_mut_type = self.default_model_preferences()
        which_one = {
            "": self.base_names,
            "rna": self.rna_names,
            "ls": self.ls_names,
            "rna+ls": self.ls_rna_names,
            "ls+rna": self.ls_rna_names,
        }
        models_for_mut_type = {
            mut_type: which_one[features]
            for mut_type, features in features_for_mut_type.items()
        }

        results_dict = {
            mut_type: {
                f"{model_type}_{ds_name}_r2": []
                for model_type in model_types
                for ds_name in ("train", "test")
            }
            for mut_type in mut_types
        }
        train_test_sizes = {mut_type: [] for mut_type in mut_types}
        for split_index in range(self.n_splits):
            self.make_comparison(
                results_dict,
                train_test_sizes,
                mut_types,
                model_types,
                n_epochs,
                n_batches,
                split_index,
                device,
                models_for_mut_type,
            )

        results_df = self.make_comparison_df_from_dict(model_types, results_dict)
        return results_df, train_test_sizes

    def make_comparison(
        self,
        results_dict,
        train_test_sizes,
        mut_types,
        model_types,
        n_epochs,
        n_batches,
        split_index,
        device,
        models_for_mut_type,
    ):
        """
        Train various NN models, add entries to results_dict for R^2, and add entries to
        train_test_sizes with sizes of train-test splits.
        """
        col = f"train_split{split_index}"
        all_muts_train_df = self.counts_df_with_splits.query(col)
        all_muts_test_df = self.counts_df_with_splits.drop(all_muts_train_df.index)

        print(f"Comparing models on split {split_index}")
        for mt in mut_types:
            print(f"\t on mut_type {mt}")

            train_df = all_muts_train_df.query("mut_type==@mt").reset_index()
            test_df = all_muts_test_df.query("mut_type==@mt").reset_index()
            train_test_sizes[mt].append((len(train_df), len(test_df)))

            nn_model_names = models_for_mut_type[mt]
            for nn_model_name in nn_model_names:
                # Strip rna and lightswitch from the name.
                stop = nn_model_name.find(" ")
                name = nn_model_name[:stop] if stop != -1 else nn_model_name
                if name not in model_types:
                    continue

                stats = self.calc_stats(
                    train_df, test_df, nn_model_name, n_epochs, n_batches, device
                )
                results_dict[mt][f"{name}_train_r2"].append(stats["train_r2"])
                results_dict[mt][f"{name}_test_r2"].append(stats["test_r2"])

    def calc_stats(self, train_df, test_df, model_name, n_epochs, n_batches, device):
        """
        For a single model type, train and return a dictionary with entries for R^2 on
        the train and test datasets for the model trained on the train dataset.
        """
        make_ds_method = self.make_ds_method[model_name]
        make_ds = lambda mut_df, *args: make_ds_method(*args[:2], mut_df, *args[2:])

        ds_args = self.make_ds_args[model_name]
        train_ds = make_ds(train_df, *ds_args)
        train_batch_size = ceil(len(train_ds) / n_batches)
        train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)

        test_ds = make_ds(test_df, *ds_args)
        test_batch_size = ceil(len(test_df) / n_batches)
        test_loader = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False)

        nn_model = self.model_maker[model_name]()
        train_model(nn_model, train_loader, test_loader, n_epochs, None, device, False)

        stats_dict = {
            "train_r2": r2_for_model(nn_model, train_loader, device),
            "test_r2": r2_for_model(nn_model, test_loader, device),
        }

        return stats_dict

    def make_comparison_df_from_dict(self, model_types, results_dict):
        """
        Create a pandas Datafame from a results_dict with columns for "mut_type", "R2",
        "source", and "replicate".
        """
        columns = ["mut_type", "R2", "source", "replicate"]
        sources = [
            f"{model_type}_{ds_name}"
            for ds_name in ("train", "test")
            for model_type in model_types
        ]
        rows = [
            [mut_type, r2, s, i]
            for mut_type, inner in results_dict.items()
            for s in sources
            for i, r2 in enumerate(inner[f"{s}_r2"])
        ]

        results_df = pd.DataFrame(data=rows, columns=columns)
        results_df.sort_values(by=["source", "mut_type"], inplace=True)
        return results_df

    def format_tick_label(self, mut_type, test_size):
        """Create a convenient tick label for the mutation type."""
        return f"{mut_type}\n{test_size}"

    def plot_test_r2_as_boxes(self, r2_df, split_sizes, outpath):
        r2_df = r2_df.copy()
        r2_df["R2"] = r2_df.R2.clip(lower=0.0)
        test_sizes = {
            mut_type: int(np.mean(next(zip(*map(reversed, sizes)))))
            for mut_type, sizes in split_sizes.items()
        }

        format_kl = lambda x: f"(k,$\ell$)=({int(x[0])//2},{int(x[11])//2})"
        rename = lambda x: "linear" if x.startswith("linear") else format_kl(x)
        r2_df["model"] = r2_df.source.apply(rename)
        r2_df["mut_type"] = r2_df.mut_type.apply(
            lambda m: self.format_tick_label(m, test_sizes[m])
        )
        r2_df = r2_df[r2_df.source.str.endswith("test")]

        fig, ax = plt.subplots()
        fig.set_figwidth(1.25 * fig.get_figwidth())
        fig.set_figheight(1.1 * fig.get_figheight())
        sns.boxplot(
            r2_df, x="mut_type", y="R2", hue="model", ax=ax, legend=True, fliersize=3
        )
        ax.set_ylabel("$R^2$")
        ax.set_xlabel("mutation type and withheld site count")
        ax.set_title("neural network model performance by $R^2$ on withheld data")
        ax.legend(ncol=2)
        fig.tight_layout()

        plt.savefig(outpath)

        return None
