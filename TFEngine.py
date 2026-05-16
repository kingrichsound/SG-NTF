"""
A base class for neural tensor factorization models with no box-cox transformation or log transformation.
"""

import numpy as np
import torch
from torch.nn import MSELoss, L1Loss
from torch.utils.tensorboard import SummaryWriter

from utils import use_optimizer
from data import TensorDataset


class TFEngine(object):
    """
    Meta engine class for wrapping training and evaluation process
    of tensor factorization.
    """

    model: torch.nn.Module

    def __init__(
        self,
        config,
        bounds=[0, 1],
        flag_norm=False,
        flag_log=False,
        flag_BCE=False,
    ):
        """
        Initialize the engine based on the configuration.
        """
        self.config = config  # model configuration
        self.flag_norm = flag_norm
        self.flag_log = flag_log

        if self.flag_log:
            self.min_value = np.log(bounds[0])
            self.max_value = np.log(bounds[1])
        else:
            self.min_value = bounds[0]
            self.max_value = bounds[1]

        self._writer = SummaryWriter(
            log_dir="runs/{}_{}".format(
                self.config["model"], self.config["embedding_dims"]
            )
        )  # tensorboard writer
        self._writer.add_text("config", str(self.config), 0)
        self.opt = use_optimizer(self.model, self.config)
        if flag_BCE:
            self.crit = torch.nn.BCELoss()
        self.crit = torch.nn.MSELoss()

    def train_single_batch(self, users, services, timeslices, ratings):
        """
        Train a single batch of data.
        """
        assert hasattr(self, "model"), "Please specify the exact model !"

        # reset all grad
        self.opt.zero_grad()

        # forward
        ratings_pred = self.model(users, services, timeslices)

        # calculate loss
        loss = self.crit(ratings_pred.view(-1), ratings)

        # backward
        loss.backward()

        # Update parameters
        self.opt.step()

        loss = loss.item()
        return loss

    def train_an_epoch(self, train_loader, epoch_id):
        """
        Train a single epoch.
        """
        assert hasattr(self, "model"), "Please specify the exact model !"

        self.model.train()
        total_loss = 0

        for _, batch in enumerate(train_loader):
            assert isinstance(batch[0], torch.Tensor)
            aIDs, bIDs, cIDs, values = (
                batch[0],
                batch[1],
                batch[2],
                batch[3],
            )

            if self.flag_log:
                values = torch.log(values)

            if self.flag_norm:
                values = (values - self.min_value) / (
                    self.max_value - self.min_value
                )

            if self.config["use_mps"] is True:
                aIDs = aIDs.to("mps")
                bIDs = bIDs.to("mps")
                cIDs = cIDs.to("mps")
                values = values.to("mps")

            loss = self.train_single_batch(aIDs, bIDs, cIDs, values)
            total_loss += loss

        self._writer.add_scalar("model/loss", total_loss, epoch_id)

    def evaluate(self, eval_loader, epoch_id=None): # 1. 这里加上 epoch_id=None
        """
        Evaluate the results.
        """
        assert hasattr(self, "model"), "Please specify the exact model !"

        self.model.eval()
        rmse = 0
        mre = 0
        mae = 0
        l1_fn = L1Loss(reduction="sum")
        mse_fn = MSELoss(reduction="sum")
        for _, batch in enumerate(eval_loader):
            assert isinstance(batch[0], torch.Tensor)
            with torch.no_grad():
                aIDs, bIDs, cIDs, batch_values = (
                    batch[0],
                    batch[1],
                    batch[2],
                    batch[3],
                )
                if self.config["use_mps"] is True:
                    aIDs = aIDs.to("mps")
                    bIDs = bIDs.to("mps")
                    cIDs = cIDs.to("mps")
                batch_preds = self.model(aIDs, bIDs, cIDs)
                if self.config["use_mps"] is True:
                    aIDs = aIDs.to("cpu")
                    bIDs = bIDs.to("cpu")
                    cIDs = cIDs.to("cpu")
                    batch_preds = batch_preds.to("cpu")

                if self.flag_norm:
                    batch_preds = (
                        batch_preds * (self.max_value - self.min_value)
                        + self.min_value
                    )
                if self.flag_log:
                    batch_preds = torch.exp(batch_preds)

                mae += l1_fn(batch_preds.view(-1), batch_values)
                mre += torch.sum(
                    torch.abs(batch_preds.view(-1) - batch_values)
                    / batch_values
                )
                rmse += mse_fn(batch_preds.view(-1), batch_values)

        mae /= eval_loader.dataset.values.size(0)
        mre /= eval_loader.dataset.values.size(0)
        rmse = np.sqrt(rmse / eval_loader.dataset.values.size(0))

        # 2. 修改这里：将 epoch_id 传给 TensorBoard
        if epoch_id is not None:
            self._writer.add_scalar("eval/MAE", mae, epoch_id)
            self._writer.add_scalar("eval/MRE", mre, epoch_id)
            self._writer.add_scalar("eval/RMSE", rmse, epoch_id)

        return mae, mre, rmse

    def pred(self, test_loader):
        """
        Predict the results.
        """
        assert hasattr(self, "model"), "Please specify the exact model !"

        self.model.eval()
        batch_count = 0
        preds = torch.Tensor()
        for _, batch in enumerate(test_loader):
            assert isinstance(batch[0], torch.Tensor)
            batch_count += 1
            with torch.no_grad():
                aIDs, bIDs, cIDs = (
                    batch[0],
                    batch[1],
                    batch[2],
                )
                if self.config["use_mps"] is True:
                    aIDs = aIDs.to("mps")
                    bIDs = bIDs.to("mps")
                    cIDs = cIDs.to("mps")
                batch_preds = self.model(aIDs, bIDs, cIDs)
                if self.config["use_mps"] is True:
                    aIDs = aIDs.to("cpu")
                    bIDs = bIDs.to("cpu")
                    cIDs = cIDs.to("cpu")
                    batch_preds = batch_preds.to("cpu")
            preds = torch.cat((preds, batch_preds.view(-1)))

        if self.flag_norm:
            preds = preds * (self.max_value - self.min_value) + self.min_value
        if self.flag_log:
            preds = torch.exp(preds)

        values = test_loader.dataset.values

        l1_fn = L1Loss(reduction="mean")
        mae = l1_fn(preds, values)

        mre = torch.mean(torch.abs(preds - values) / values)

        mse_fn = MSELoss(reduction="mean")
        rmse = np.sqrt(mse_fn(preds, values))

        return mae, mre, rmse

    def save(self, model_dir):
        """
        Save the model.
        """
        assert hasattr(self, "model"), "Please specify the exact model !"

        torch.save(
            self.model.state_dict(),
            model_dir,
        )
