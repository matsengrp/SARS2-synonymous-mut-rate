from functools import reduce, partial
import torch
import torch.nn as nn
import torch.nn.functional as F


class S2EmbeddedModel(nn.Module):
    """
    A generic class for NN models of SARS2 mutation rates that begin with a simple
    embedding layer. This class may be directly instantiated for models whose forward
    method is easily written as f_N∘f_{N-1}∘...∘f_1∘embedding by supplying
    layers = [f_1∘...∘f_{N-1}∘f_N]. Otherwise, inherit and override as needed.
    """

    def __init__(self, num_embeddings, embedding_dim, layers):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.layers = nn.ParameterList((self.embedding, *layers))

    def forward(self, x):
        return reduce(lambda z, f: f(z), self.layers, x)


class MeanNBLoss(nn.Module):
    """
    Negative binomial loss function. The loss is the average negative log-likelihood for
    the data using the negative binomial distribution with specified alpha, up to an
    additive constant. The additive constant is
    log(Gamma(actual+1/alpha)/Gamma(actual+1)/Gamma(1/alpha)).
    """

    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, predicted, actual):
        return (
            (actual + 1 / self.alpha) * (1 + self.alpha * predicted).log()
            - actual * predicted.log()
        ).mean()


class S2KmerModel(S2EmbeddedModel):
    """
    This model parameterizes a simple k-mer model using an Embedding for each k-mer,
    followed by exp.
    """

    def __init__(self, motif_count, log_counts=False, *args, **kwargs):
        num_embeddings = motif_count
        embedding_dim = 1
        if log_counts:
            layers = [partial(torch.squeeze, dim=-1)]
        else:
            layers = [partial(torch.squeeze, dim=-1), torch.exp]
        super().__init__(num_embeddings, embedding_dim, layers)


class S2KmerCrossFeaturesModel(S2KmerModel):
    """
    This model parameterizes a k-mer with additional binary classification features
    model using an Embedding for each combination of k-mer and features, followed by
    exp.
    """

    def __init__(self, motif_count, num_features, log_counts=False, *args, **kwargs):
        super().__init__(2**num_features * motif_count, log_counts=log_counts)


class S2FlatNNModel(S2EmbeddedModel):
    """
    This model has an (multidimensional) Embedding for the motifs, followed by a linear
    layer across the embedding, and finally an exp.
    """

    def __init__(
        self,
        motif_count,
        embedding_dim,
        filter_width,
        dropout_prob=0.05,
        log_counts=False,
    ):
        num_embeddings = motif_count
        layers = [
            lambda x: x.view(x.shape[0], -1),
            nn.Dropout(dropout_prob),
            nn.Linear(in_features=embedding_dim * filter_width, out_features=1),
            lambda x: x.squeeze(-1),
        ]
        if not log_counts:
            layers.append(torch.exp)
        super().__init__(num_embeddings, embedding_dim, layers)


class S2FlatNNPlusFeaturesModel(nn.Module):
    """
    This model has an (multidimensional) Embedding for the motifs, followed by a linear
    layer across the embedding and additional features, and finally an exp.
    """

    def __init__(
        self,
        motif_count,
        embedding_dim,
        filter_width,
        dropout_prob,
        num_features,
        log_counts=False,
    ):
        super().__init__()
        self.log_counts = log_counts
        self.motif_embedding = nn.Embedding(motif_count, embedding_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(
            in_features=embedding_dim * filter_width + num_features, out_features=1
        )

    def forward(self, x):
        kmer_indices = x[:, 0, :]
        extra_feature_values = x[:, 1:, 0]

        embedding = self.motif_embedding(kmer_indices).squeeze(dim=-1)
        embedding = embedding.view(embedding.shape[0], -1)
        embedding = self.dropout(embedding)

        input_to_linear = torch.cat((embedding, extra_feature_values), dim=1)
        output = self.linear(input_to_linear).squeeze(dim=-1)

        return output if self.log_counts else torch.exp(output)
