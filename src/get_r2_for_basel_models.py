from itertools import permutations
import pandas as pd
import numpy as np
from helpers import GeneralLinearModel, load_synonymous_muts


def prepare_train_test_splits(n_splits, test_holdout=0.2, seed=8675309):
    """
    Return a dataframe of the synonymous mutation counts in curated_mut_counts.csv. The
    dataframe records train-test splits with boolean valued columns named
    f"train_split{j}" for j=0,1,...,(n_splits-1). For example, the rows where
    train_split0 is True form the training data of the first train-test split.

    Args:
        n_splits (int): The number of train-test splits.
        test_holdout (float): The ratio of data assigned to test. This ratio holds true
            for each mutation type.
        seed (int): Seed of the random number generator for train-test splits.
            Reproducibility is guaranteed only with identical versions of numpy.
    """
    pseudo_count = 1 / 2
    rng = np.random.default_rng(seed)

    the_df = load_synonymous_muts()
    the_df["site"] = the_df.nt_site
    the_df["mut_count"] = the_df.actual_count
    the_df["log_mut_count"] = np.log(the_df.mut_count + pseudo_count)
    the_df["basepair"] = (the_df.ss_prediction == "paired").astype(int)
    # Updated site boundaries for partitions
    boundaries = {"".join(mut_type): 0 for mut_type in permutations("AGCT", 2)}
    for mut_type in ("AT", "CG", "GC"):
        boundaries[mut_type] = 21562
    boundaries["CT"] = 13467
    pre_boundary = lambda row: row.site <= boundaries[row.mut_type]
    the_df["partition"] = the_df.apply(pre_boundary, axis=1).astype(int).values

    # Create train-test splits at the level of mutation types.
    mut_types = the_df.mut_type.unique()
    for j in range(n_splits):
        the_df[f"train_split{j}"] = True
        for mut_type in mut_types:
            mut_view = the_df.query("mut_type==@mut_type")
            test_split = mut_view.sample(frac=test_holdout, random_state=rng)
            the_df.loc[test_split.index, f"train_split{j}"] = False

    return the_df


def r2_on_splits(dataset_path, r2_path):
    """
    Compute R^2 for each version of the basel model on each mutation type for a total
    of 10 times.

    Args:
        dataset_path (str): Write path for the data with train-test splits as a csv file.
        r2_path (str): Write path for the R^2 values as a csv file.
    """
    n_splits = 10
    the_df = prepare_train_test_splits(n_splits)
    the_df.to_csv(dataset_path)

    model_parameters = {
        "linear": ["local_context"],
        "linear+ls": ["local_context", "global_context"],
        "linear+rna": ["local_context", "rna_structure"],
        "linear+ls+rna": ["local_context", "global_context", "rna_structure"],
    }

    header_row = "model_name,mut_type,R2,source,replicate\n"
    results_rows = []
    for name, factors in model_parameters.items():
        for j in range(n_splits):
            train_split = the_df.query(f"train_split{j}")
            test_split = the_df.drop(train_split.index)
            model = GeneralLinearModel(included_factors=factors)
            model.train(df_train=train_split)

            mse_on_train = model.test(df_test=train_split)
            mse_on_test = model.test(df_test=test_split)
            for mut_type, mse in mse_on_train.items():
                var = np.var(train_split.query("mut_type==@mut_type").log_mut_count)
                r2 = 1 - mse / var
                results_rows.append(f"{name},{mut_type},{r2},linear_train,{j}\n")
            for mut_type, mse in mse_on_test.items():
                var = np.var(test_split.query("mut_type==@mut_type").log_mut_count)
                r2 = 1 - mse / var
                results_rows.append(f"{name},{mut_type},{r2},linear_test,{j}\n")

    with open(r2_path, "w") as the_file:
        the_file.write(header_row)
        for row in results_rows:
            the_file.write(row)

    return None


if __name__ == "__main__":
    data_set_path = "../results/split_syn_mut_counts.csv"
    r2_path = "../results/basel_r2s.csv"
    r2_on_splits(data_set_path, r2_path)
