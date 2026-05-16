"""
Neural Tucker Tensor Factorization with normalization
"""

import torch
from torch import nn
from TFEngine import TFEngine


class NeuTucF(torch.nn.Module):
    """
    Neural Tucker Factorization
    """

    def __init__(self, config) -> None:
        super(NeuTucF, self).__init__()
        self.config = config
        self.max_aID = config["dimensions"][0]
        self.max_bID = config["dimensions"][1]
        self.max_cID = config["dimensions"][2]
        ranks = config["embedding_dims"]
        self.flag_bias = config["flag_bias"]

        if isinstance(ranks, int):
            self.rankA = ranks
            self.rankB = ranks
            self.rankC = ranks
        else:
            self.rankA = ranks[0]
            self.rankB = ranks[1]
            self.rankC = ranks[2]

        self.embedding_layer_A = torch.nn.Embedding(
            num_embeddings=self.max_aID, embedding_dim=self.rankA
        )
        self.embedding_layer_B = torch.nn.Embedding(
            num_embeddings=self.max_bID, embedding_dim=self.rankB
        )
        self.embedding_layer_C = torch.nn.Embedding(
            num_embeddings=self.max_cID, embedding_dim=self.rankC
        )

        if self.flag_bias:
            self.tucker_linear_layer = torch.nn.Linear(
                in_features=self.rankA * self.rankB * self.rankC,
                out_features=1,
            )
        else:
            self.tucker_linear_layer = torch.nn.Linear(
                in_features=self.rankA * self.rankB * self.rankC,
                out_features=1,
                bias=False,
            )

        self.output_mapping = torch.nn.Sigmoid()

        nn.init.uniform_(self.embedding_layer_A.weight, 0, 1)
        nn.init.uniform_(self.embedding_layer_B.weight, 0, 1)
        nn.init.uniform_(self.embedding_layer_C.weight, 0, 1)
        nn.init.uniform_(self.tucker_linear_layer.weight, 0, 1)
        if self.flag_bias:
            nn.init.constant_(self.tucker_linear_layer.bias, 0)

    def forward(self, aIDs, bIDs, cIDs):
        a_embeddings = self.embedding_layer_A(aIDs)
        b_embeddings = self.embedding_layer_B(bIDs)
        c_embeddings = self.embedding_layer_C(cIDs)

        outer_product = torch.einsum(
            "bi,bj,bk->bijk", a_embeddings, b_embeddings, c_embeddings
        )
        interactions = outer_product.view(
            -1, self.rankA * self.rankB * self.rankC
        )
        logits = self.tucker_linear_layer(interactions)
        rating = self.output_mapping(logits)
        return rating

    def init_weight(self):
        pass

    def load_model(self, model_dir: str):
        """Loading weights from trained NCPF model"""
        config = self.config
        neutucf_model = NeuTucF(config)
        if config["use_mps"] is True:
            neutucf_model.to(torch.device("mps"))
            device = "mps"
        else:
            device = "cpu"

        state_dict = torch.load(
            model_dir,
            map_location=torch.device(device),
        )  # ensure all storage are on same device with map_location setting
        neutucf_model.load_state_dict(state_dict)

        self.embedding_layer_A.weight.data = (
            neutucf_model.embedding_layer_A.weight.data
        )
        self.embedding_layer_B.weight.data = (
            neutucf_model.embedding_layer_B.weight.data
        )
        self.embedding_layer_C.weight.data = (
            neutucf_model.embedding_layer_C.weight.data
        )
        self.tucker_linear_layer.weight.data = (
            neutucf_model.tucker_linear_layer.weight.data
        )
        if self.flag_bias:
            self.tucker_linear_layer.bias.data = (
                neutucf_model.tucker_linear_layer.bias.data
            )


class NeuTucFEngine(TFEngine):
    """Training & evaluating NeuTucF model"""

    def __init__(self, config, bounds=[0, 1], flag_norm=False, flag_log=False):
        self.model = NeuTucF(config)
        if config["use_mps"] is True:
            mps_device = torch.device("mps")
            self.model.to(mps_device)
        super(NeuTucFEngine, self).__init__(
            config, bounds, flag_norm, flag_log
        )
