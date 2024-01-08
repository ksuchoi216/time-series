import math
import inspect
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random
import os
import optuna
from optuna.trial import TrialState


class TSFRunner:
    def __init__(self, run_equipment, epochs):
        print(f"\n>>>>>>>> func: {inspect.stack()[0][3]} <<<<<<<<")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device: {self.device}")
        # * run_equipment
        self.model = run_equipment["model"]
        self.optimizer = run_equipment["optimizer"]
        self.loss_fn = run_equipment["loss_fn"]
        self.seed = run_equipment["seed"]

        # * data related
        self.dataloaders = run_equipment["dataloaders"]
        self.train_loader = self.dataloaders["train"]
        self.val_loader = self.dataloaders["val"]
        self.test_loader = self.dataloaders["test"]

        # * seed
        random.seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        # * run
        self.interval = 100
        self.seq_len = run_equipment["seq_len"]
        self.epochs = epochs

        # * save
        self.ckpt_path = run_equipment["ckpt_path"]
        self.model_path = run_equipment["model_path"]
        # self.output_path = run_equipment["output_path"]

    def save_ckpt(self, epoch, loss):
        torch.save(
            {
                "epoch": epoch,
                "loss": loss,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            self.ckpt_path,
        )

    def load_ckpt(self):
        checkpoint = torch.load(PATH)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]

    def train_step(
        self,
    ):
        train_loss_epoch = 0.0
        self.model.train()
        for i, (x_batch, y_batch) in enumerate(self.train_loader):
            # print(x_batch.shape, y_batch.shape)
            # * input
            x_batch = x_batch.to(self.device)
            # y_batch = y_batch.unsqueeze(1).to(self.device)
            y_batch = y_batch.to(self.device)

            # * forward
            y_pred = self.model(x_batch)
            # print(y_pred.shape, y_batch.shape)
            loss = self.loss_fn(y_pred, y_batch)

            # * backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # * result
            train_loss_epoch += loss

        train_loss_epoch = train_loss_epoch / len(self.train_loader)
        train_loss_epoch = np.round(train_loss_epoch.detach().numpy(), 4)  # type: ignore
        return train_loss_epoch

    def val_step(self):
        val_loss_epoch = 0.0
        self.model.eval()
        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(self.val_loader):
                # * input
                x_batch = x_batch.to(self.device)
                # y_batch = y_batch.unsqueeze(1).to(self.device)
                y_batch = y_batch.to(self.device)

                # * forward
                y_pred = self.model(x_batch)
                loss = self.loss_fn(y_pred, y_batch)

                # * result
                val_loss_epoch += loss

        val_loss_epoch = val_loss_epoch / len(self.val_loader)
        val_loss_epoch = np.round(val_loss_epoch.detach().numpy(), 4)  # type: ignore
        return val_loss_epoch

    def test_step(self):
        res_true = []
        res_pred = []

        # print(len(self.test_loader.dataset))
        test_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(self.test_loader):
                # * input
                x_batch = x_batch.to(self.device)
                # y_batch = y_batch.unsqueeze(1).to(self.device)
                y_batch = y_batch.to(self.device)

                # * forward
                y_pred = self.model(x_batch)
                loss = self.loss_fn(y_batch, y_pred)
                test_loss += loss
                print(y_batch.shape, y_pred.shape)

                # * result
                y_batch = np.squeeze(y_batch.detach().numpy())
                y_pred = np.squeeze(y_pred.detach().numpy())
                # print(y_batch.shape, y_pred.shape)
                res_true.extend(y_batch)
                res_pred.extend(y_pred)
                # print(len(self.res_true))

        test_loss = test_loss / len(self.test_loader)
        print(f"test loss: {test_loss}")

        res_df = pd.DataFrame(
            {
                "y_true": res_true,
                "y_pred": res_pred,
            }
        )
        return res_df

    def run(self, trial=None):
        print(f"epochs: {self.epochs}")
        losses = []
        for epoch in range(self.epochs):
            # ! train
            train_loss_epoch = self.train_step()

            # ! val
            val_loss_epoch = self.val_step()
            losses.append([train_loss_epoch, val_loss_epoch])

            if trial:
                trial.report(val_loss_epoch, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            print(
                f"[epoch {epoch+1:03d}] loss train {train_loss_epoch:.4f} val {val_loss_epoch:.4f}"
            )

        self.res_train_df = pd.DataFrame(losses, columns=["train_loss", "val_loss"])

        # ! test
        self.res_test_df = self.test_step()

        # # ! save res
        # self.res_train_df.to_csv(f"{self.output_path}/res_train.csv")
        # self.res_test_df.to_csv(f"{self.output_path}/res_test.csv")

        return self.res_train_df, self.res_test_df


def build_runner(run_equipment: dict, epochs: int):
    return TSFRunner(run_equipment, epochs)
