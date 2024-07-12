# SARS2-synonymous-mut-rate

## Organization of repo

* We include the `SARS2-mut-fitness` repository as a submodule

* `notebooks\curate_counts.ipynb`: reads in and curates counts from specific files in the `SARS2-mut-fitness` submodule. Specifically, we curate the counts data as follows:
    * We read in a dataframe of actual and expected counts from the `SARS2-mut-fitness` repo, using data from the 2024-04-24 GISAID tree.
    * We identified sites where the site, the site's codon, and the site's 3mer sequence context are conserved in all clade founders (ignoring the codon requirement for noncoding sites).
    * We filtered out sites that: i) did not meet the above conservation criteria, ii) were masked in the UShER tree in any clade, iii) where the `exclude` column `== True`.
    * We summed counts across all clades, using subset == 'all' (as opposed to England or USA), such that there is one row per mutation and the `actual_count` and `expected_count` columns give total counts for that mutation
    * We also added a few columns giving actual counts from different subsets of the data (e.g., England vs. USA; pre-Omicron clades vs. post-Omicron clades)
    * We wrote the curated counts to the file: `results/curated_mut_counts.csv`