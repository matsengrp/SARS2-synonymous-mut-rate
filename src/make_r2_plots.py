import numpy as np
import pandas as pd
from nn_comparison import NNComparison

compare = NNComparison()
model_preferences = compare.default_model_preferences()
curated_counts_df = pd.read_csv("../results/split_syn_mut_counts.csv")

basel_r2_df = pd.read_csv("../results/basel_r2s.csv")
name_mapping = {
    "": "linear",
    "rna": "linear+rna",
    "ls": "linear+ls",
    "ls+rna": "linear+ls+rna",
}
prefered_linear_model = lambda mut_type: name_mapping[model_preferences[mut_type]]
basel_r2_df = basel_r2_df[
    basel_r2_df.model_name == basel_r2_df.mut_type.apply(prefered_linear_model)
]
basel_r2_df.drop(columns=["model_name"], inplace=True)
basel_r2_df.sort_values(by="mut_type", inplace=True)

model_types = [
    "3_sites_by_1",
    "3_sites_by_3",
    "5_sites_by_1",
    "5_sites_by_3",
    "7_sites_by_1",
    "7_sites_by_3",
]
n_epochs = 500
n_batches = 2
results_df, train_test_sizes = compare.make_comparison_df(
    None, model_types, n_epochs, n_batches, None, model_preferences
)
results_df.sort_values(by=["source", "mut_type"], inplace=True)


combined_df = pd.concat([basel_r2_df, results_df])
combined_df.to_csv("../results/combined_results.csv")
compare.plot_test_r2_as_boxes(
    combined_df, train_test_sizes, "../results/r2_comparison.pdf"
)
