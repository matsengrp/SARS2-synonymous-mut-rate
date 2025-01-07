import numpy as np
from itertools import product
from math import log
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from tqdm import tqdm
from scipy.special import gammaln

# Helper classes


class S2IndicesDataset(Dataset):
    def __init__(self, indices, mut_counts):
        self.indices = indices
        self.mut_counts = mut_counts

    def __len__(self):
        return len(self.mut_counts)

    def __getitem__(self, idx):
        return self.indices[idx], self.mut_counts[idx]

    def extend(self, other, inplace=False):
        with ThreadFix():
            the_indices = torch.cat((self.indices, other.indices))
            the_mut_counts = torch.cat((self.mut_counts, other.mut_counts))
        if inplace:
            self.indices = the_indices
            self.mut_counts = the_mut_counts
            return None
        else:
            return S2IndicesDataset(the_indices, the_mut_counts)


class KMerData:
    """
    A utitility class to make k-mer datasets. Call kmer_dataset_for_mut_df for motif
    style kmers.

    Attributes:
        all_kmers (list): The k-mers.
        all_pairs (list): All pairs of a kmer with 0 or 1.
        all_triples (list):  All triples of a kmer, 0 or 1, and 0 or 1.
        BASES (str): The nucleotide bases "ACGT".
        k (int): The k in k-mer.
        kmer_count (int): The number of k-mers.
        kmer_index (dict): The map of a k-mer to its index in all_kmers.
        left_base_index (dict): The map of a base to its index in BASES.
        pair_index (dict): The map of a pair to its index in all_pairs.
        right_base_index (dict): The map of a base to its index in BASES.
        triple_index (dict): The map of a triple to its index in all_triples.
    """

    BASES = "ACGT"

    def __init__(self, k):
        if k >= log(torch.iinfo(torch.int32).max) / log(4):
            raise ValueError("There are too many motifs to embed.")
        if k % 2 == 0:
            raise ValueError("The motif length must be odd.")
        self.k = k
        self.kmer_count = 4**k

        self.all_kmers = list(map("".join, product(self.BASES, repeat=k)))
        self.all_pairs = product(self.all_kmers, [0, 1])
        self.all_triples = product(self.all_kmers, [0, 1], [0, 1])

        self.kmer_index = dict(map(reversed, enumerate(self.all_kmers)))
        self.pair_index = dict(map(reversed, enumerate(self.all_pairs)))
        self.triple_index = dict(map(reversed, enumerate(self.all_triples)))

    def kmer_dataset_for_mut_df(
        self,
        kind,
        seq,
        mut_df,
        mut_type=None,
        filter_width=1,
        partition=False,
        rna=False,
        log_counts=False,
    ):
        """
        Return an S2IndicesDataset with per site k-mer indices and mutation counts.

        Args:
            kind (str): How the partition and rna values are combined with the
                kmer-indices. When kind is "x" or "cross", unique integer indices are
                assigned to each tuple of motif and feature values, and
                S2IndicesDataset.indices is a tensor of these integer indices. Use
                "cross" for models that include partition or rna structure in the
                initial layer (e.g., the embedding layer). When kind is "+" or "plus",
                unique integer indices are assigned to each motif, and
                S2IndicesDataset.indices is a tensor tensors, where each inner tensor
                stores a motif index (or a tensor of motif indices when filter width is
                larger than 1) and additional feature values for the motif. When filter
                width is larger the 1, the additional feature values are only for the
                center motif. Use "plus" for models that incorporate the partition or
                rna structure at a later layer. For both kinds, partition values are
                listed before rna structure values, when both partition and rna are
                True. When partition and rna are both False, this argument has no
                effect.
            seq (str): The nucleotide sequence before mutations.
            mut_df (pandas.DataFrame): A DataFrame with columns "site", "mut_type",
                "partition", "basepair", and "mut_count".
            mut_type (str): The mutation type to restrict the dataset to. When None, no
                resriction is imposed.
            filter_width (int): How many k-mers to associate with each site. Use 1 for a
                a simple kmer model, use a larger width for more complicated models.
                When filter_width is 1, the kmer_indices of the dataset is a flat
                tensor, otherwise it is a tensor of tensors.
            partition (bool): When True, include the partition (lightswitch) values in
                the returned S2IndicesDataset.
            rna (bool): When True, include the additional rna-structure (basepairing)
                values in the returned S2IndicesDataset.
            log_counts (bool): When True, use the logarithm of mutation counts instead
                of mutation counts.

        Returns:
            S2IndicesDataset: The dataset.
        """
        if kind not in ("+", "plus", "x", "cross"):
            raise ValueError("kind must be '+', 'plus', 'x', or 'cross'")
        if filter_width % 2 == 0:
            raise ValueError(f"The filter width must be odd.")
        if kind in ("x", "cross") and filter_width > 1:
            if partition or rna:
                # This version requires we know the partition and basepair
                # classification of sites outside the df. We can do that, if we want.
                raise NotImplementedError()
            else:
                return self.kmer_dataset_for_mut_df(
                    "+", seq, mut_df, mut_type, filter_width, False, False
                )
        if mut_type is not None:
            mut_df = mut_df[mut_df.mut_type == mut_type]
        side_len = self.k // 2
        motif_around = lambda site: seq[site - side_len - 1 : site + side_len]

        sites = mut_df.site
        partitions = mut_df.partition.values
        basepairs = mut_df.basepair.values
        mut_counts = torch.tensor(mut_df.mut_count.values, dtype=torch.float32)
        if log_counts:
            with ThreadFix():
                mut_counts = (mut_counts + 1 / 2).log()

        motifs = map(motif_around, mut_df.site)
        if kind in ("x", "cross"):
            # Entering this block means the filter width is 1.
            index_dict, feature_values = {
                (False, False): (self.kmer_index, motifs),
                (False, True): (self.pair_index, zip(motifs, basepairs)),
                (True, False): (self.pair_index, zip(motifs, partitions)),
                (True, True): (self.triple_index, zip(motifs, partitions, basepairs)),
            }[partition, rna]
            indices = torch.tensor([index_dict[key] for key in feature_values])
        elif kind in ("+", "plus"):
            filter_offset = filter_width // 2
            motif_index_lists = [
                [
                    self.kmer_index[motif_around(site + offset)]
                    for offset in range(-filter_offset, filter_offset + 1)
                ]
                for site in sites
            ]
            indices = [torch.tensor(motif_index_lists)]
            reshape = lambda w: w.view(w.shape[0], 1).expand(-1, filter_width)
            if partition:
                partitions = reshape(torch.tensor(partitions))
                indices.append(partitions)
            if rna:
                basepairs = reshape(torch.tensor(basepairs))
                indices.append(basepairs)
            indices = torch.stack(indices).swapaxes(1, 0)
            if filter_width > 1:
                indices = indices.squeeze()

        return S2IndicesDataset(indices, mut_counts)


class ThreadFix:
    """
    Pytorch is sometimes buggy when creating tensors with multiple threads. For example,
    this can happen when constructing datasets for TC. This context manager temporary
    sets the number of threads to 1.
    """

    def __enter__(self):
        self.thread_count = torch.get_num_threads()
        torch.set_num_threads(1)
        return None

    def __exit__(self, *args, **kwargs):
        torch.set_num_threads(self.thread_count)
        return None


# Helper functions


def pick_device():
    """
    Return an available pytorch device.
    """

    def check_CUDA():
        try:
            torch._C._cuda_init()
            return True
        except:
            return False

    if torch.backends.cudnn.is_available() and check_CUDA():
        print("Using CUDA")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Using Metal Performance Shaders")
        return torch.device("mps")
    else:
        print("Using CPU")
        return torch.device("cpu")


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs,
    loss=None,
    device="cpu",
    progress_bar=True,
):
    """
    Train the given model using the provided train_loader and validate using val_loader
    for a specified number of epochs.

    Args:
        model (nn.Module): The PyTorch model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        num_epochs (int): Number of epochs to train the model.
        loss (nn.Module): The loss function for training. When None, the loss defaults
            to nn.MSELoss.
        device (torch.device): Device to train the model on (CPU or GPU).
        progress_bar (bool): Optionally print a progress bar to screen.

    Returns:
        tuple: A tuple containing two lists, the average training loss and the average
            validation loss for each epoch.
    """
    # Define loss function and optimizer
    criterion = nn.MSELoss() if loss is None else loss
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # Move model to the device
    model.to(device)

    # Lists to store the loss values
    train_losses = []
    val_losses = []

    if progress_bar:
        epoch_range = tqdm(range(num_epochs), desc="Training Epochs")
    else:
        epoch_range = range(num_epochs)

    for epoch in epoch_range:
        # Training
        model.train()
        epoch_train_loss = 0
        for batch in train_loader:
            kmer_indices_batch, mut_counts_batch = batch

            # Move data to the device
            kmer_indices_batch = kmer_indices_batch.to(device, dtype=torch.long)
            mut_counts_batch = mut_counts_batch.to(device, dtype=torch.float)

            # Forward pass
            outputs = model(kmer_indices_batch)
            loss = criterion(outputs, mut_counts_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            epoch_train_loss += loss.item()

        # Average training loss for the epoch
        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)

        # Validation
        epoch_val_loss = 0
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    kmer_indices_batch, mut_counts_batch = batch

                    # Move data to the device
                    kmer_indices_batch = kmer_indices_batch.to(device, dtype=torch.long)
                    mut_counts_batch = mut_counts_batch.to(device, dtype=torch.float)

                    # Forward pass
                    outputs = model(kmer_indices_batch)
                    loss = criterion(outputs, mut_counts_batch)

                    # Accumulate loss
                    epoch_val_loss += loss.item()

            # Average validation loss for the epoch
            epoch_val_loss /= len(val_loader)
        val_losses.append(epoch_val_loss)

    return train_losses, val_losses


def summary_stat_for_model(stat_fn, model, loader, device):
    """
    Return stat_fn applied to the actual and predicted mutation counts for the given
    model and data loader.
    """
    model.eval()
    predicted_mut_counts = []
    actual_mut_counts = []
    with torch.no_grad():
        for batch in loader:
            kmer_indices_batch, mut_counts_batch = batch
            kmer_indices_batch = kmer_indices_batch.to(device)
            predicted_mut_counts.append(model(kmer_indices_batch))
            actual_mut_counts.append(mut_counts_batch)

        predicted_mut_counts = torch.cat(predicted_mut_counts).numpy(force=True)
        actual_mut_counts = torch.cat(actual_mut_counts).numpy(force=True)
    return stat_fn(actual_mut_counts, predicted_mut_counts)


def mean_nb_log_likelihood(alpha, actual, predicted):
    return (
        actual * log(alpha)
        + actual * np.log(predicted)
        - (actual + 1 / alpha) * np.log(1 + alpha * predicted)
        + gammaln(actual + 1 / alpha)
        - gammaln(actual + 1)
        - gammaln(1 / alpha)
    ).mean()


def mean_nb_logl_for_model(alpha, model, loader, device):
    logl = lambda actual, predicted: mean_nb_log_likelihood(alpha, actual, predicted)
    return summary_stat_for_model(logl, model, loader, device)


def mse_for_model(model, loader, device):
    mse = lambda x, y: ((x - y) ** 2).mean()
    return summary_stat_for_model(mse, model, loader, device)


def mae_for_model(model, loader, device):
    mse = lambda x, y: np.abs(x - y).mean()
    return summary_stat_for_model(mse, model, loader, device)


def r2_for_model(model, loader, device):
    return summary_stat_for_model(r2_for_arrays, model, loader, device)


def r2_for_arrays(actual, predicted):
    ssr = ((actual - predicted) ** 2).sum()
    mean = actual.mean()
    sst = ((actual - mean) ** 2).sum()
    return 1 - ssr / sst
