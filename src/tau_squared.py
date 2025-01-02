import numpy as np
import matplotlib.pyplot as plt

from helpers import load_synonymous_muts, GeneralLinearModel


if __name__ == '__main__':

    df = load_synonymous_muts()

    model = GeneralLinearModel(included_factors=['global_context', 'rna_structure', 'local_context'])

    model.train(df_train=df)

    tau_squared_dict = model.tau_squared

    mut_types = sorted(tau_squared_dict.keys())
    tau_squareds = [tau_squared_dict[mut_type] for mut_type in mut_types]

    plt.figure(figsize=(8, 4))
    
    bar_positions = np.arange(len(mut_types))
    plt.bar(bar_positions, tau_squareds, width=0.7)

    plt.xticks(ticks=bar_positions, labels=[rf"{mut_type[0]}$\rightarrow${mut_type[1]}" for mut_type in mut_types], rotation=45, ha="center")
    plt.xlabel('Mutation type')
    plt.ylabel(r'$\tau^{2}$')
    # plt.title('Remaining variance on log-counts')

    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig("../results/tau_squared.pdf")
    plt.show()
