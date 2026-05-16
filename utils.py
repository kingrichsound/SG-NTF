"""
Utility functions for training and testing
"""

import torch
from torch import nn


def use_optimizer(module, params):
    """
    create optimizer based on the choice of hyper-parameters"""
    if params["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(
            module.parameters(),
            lr=params["learning_rate"],
            momentum=params["momentum"],
            weight_decay=params["l2_regularization"],
        )
    elif params["optimizer"] == "adam":
        optimizer = torch.optim.Adam(
            module.parameters(),
            lr=params["learning_rate"],
            weight_decay=params["l2_regularization"],
        )
    elif params["optimizer"] == "rmsprop":
        optimizer = torch.optim.RMSprop(
            module.parameters(),
            lr=params["learning_rate"],
            alpha=params["alpha"],
            momentum=params["momentum"],
        )
    return optimizer


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.uniform_(m.weight, 0, 1)
        if m.bias:
            torch.nn.init.zeros_(m.bias)


def use_activation(activation):
    """
    create activation function based on the choice of hyper-parameters
    """
    if activation == "relu":
        return torch.nn.ReLU()
    elif activation == "tanh":
        return torch.nn.Tanh()
    elif activation == "sigmoid":
        return torch.nn.Sigmoid()
    elif activation == "softmax":
        return torch.nn.Softmax(dim=-1)
    elif activation == "leaky_relu":
        return torch.nn.LeakyReLU()
    elif activation == "elu":
        return torch.nn.ELU()
    elif activation == "selu":
        return torch.nn.SELU()
    elif activation == "gelu":
        return torch.nn.GELU()
    elif activation == "swish":
        return torch.nn.SiLU()
    elif activation == "mish":
        return torch.nn.Mish()
    elif activation == "hardshrink":
        return torch.nn.Hardshrink()
    elif activation == "hardtanh":
        return torch.nn.Hardtanh()
    elif activation == "softplus":
        return torch.nn.Softplus()
    elif activation == "softshrink":
        return torch.nn.Softshrink()
    elif activation == "softsign":
        return torch.nn.Softsign()
    elif activation == "tanhshrink":
        return torch.nn.Tanhshrink()
    # elif activation == "threshold":
    #     return torch.nn.Threshold()
    elif activation == "log_sigmoid":
        return torch.nn.LogSigmoid()
    elif activation == "hardsigmoid":
        return torch.nn.Hardsigmoid()
    elif activation == "hardswish":
        return torch.nn.Hardswish()
    elif activation == "log_softmax":
        return torch.nn.LogSoftmax(dim=-1)
    # elif activation == "adaptive_log_softmax":
    #     return torch.nn.AdaptiveLogSoftmaxWithLoss()
    elif activation == "glu":
        return torch.nn.GLU()
