# SARS2-synonymous-mut-rate

This repository has the code for carying out the analyses in [SARS-CoV-2's mutation rate is highly variable between sites and is influenced by sequence context, genomic region, and RNA structure]().

## Organization of repository

* `data/` contains input data
* `notebooks/` contains Jupyter notebooks used to analyze the data
* `results/` contains outputs from Jupyter notebooks
* `environment.yml` encodes the environment used to run the Jupyter notebooks

### Key results files
* `results/curated_mut_counts.csv`: is a file with curated site-specific mutational counts. See below for how counts were curated. Key columns include:
    * `nt_mutation`: the wildtype nucleotide, site, and mutant nucleotide for a given mutation
    * `synonymous`: a bool indicating whether a mutation is synonymous
    * `ss_prediction`: indicates whether a site is paired or unpaired in the RNA secondary structure of the SARS-CoV-2 genome, as predicted by [Lan et al.](https://www.nature.com/articles/s41467-022-28603-2); see `data/lan_2022/41467_2022_28603_MOESM11_ESM.txt`.
    * `motif`: the 3mer sequence motif centered on a site
    * `actual_count`: the counts of the mutation along the branches of the UShER tree
    * `count_terminal` and `count_non_terminal`: counts on terminal or non-terminal branches, respectively
    * `actual_count_pre_omicron` and `actual_count_omicron`: counts in pre-Omicron clades or Omicron clades, respectively

### Summary of Jupyter notebooks
* `notebooks/curate_counts.ipynb`
    * This notebook generates the file with curated site-specific mutation counts.
    * As input, the notebook takes the results of running the [Bloom and Neher pipeline](https://github.com/jbloomlab/SARS2-mut-fitness) on an UShER tree with all sequences in GISAID as of 2024-04-24. The pipeline generates a file that reports the counts of each possible nucleotide mutation across the genome along the branches of the tree. In doing so, the pipeline divides the tree into several different clades and separately reports mutational counts for each clade, only reporting counts for mutations away from a given clade's founder sequence.
    * The notebook curates these raw counts data as follows:
        * First, we identified all sites in the genome where the nucleotide identities at that site, the site's codon, and the site's 5' and 3' nucleotides are conserved in all clade founder sequences, including the Wuhan-Hu-1 sequence (note: we ignore the codon requirement for noncoding sites).
        * Next, we filtered out mutations at sites that: i) did not meet the above conservation criteria, ii) were masked in the UShER tree in any clade, iii) were identified as being error-prone (we also filtered out the set of error-prone sites identified by [De Maio et al.](https://virological.org/t/issues-with-sars-cov-2-sequencing-data/473)).
        * Next, for the remaining mutations, we summed the counts of each mutation across all clades (using the counts in the `actual_counts` column, and only summing rows where the `subset` column equals `all`, as opposed to `England` or `USA`), resulting in the site-specific mutation counts that we use in our analyses.
        * To compute counts for terminal or non-terminal branches, we simply summed counts in the columns `count_terminal` or `count_non_terminal`, and to compute counts for pre-Omicron vs. Omicron clades, we simply summed counts for the relevant set of clades.
        * We wrote the curated counts to the file: `results/curated_mut_counts.csv`

* `notebooks/analyze_counts.ipynb`
    * This notebook reads in the curated counts from above and generates many of the plots that explore patterns in synonymous mutation counts between sites.