import os
from abc import *

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import data_provider.feat_time


class TSDataLoader(metaclass=ABCMeta):
    @abstractmethod
    def _read_data(self):  # including scaler
        pass

    @abstractmethod
    def _make_dataloader(self, data, shuffle=True):
        pass

    @abstractmethod
    def inverse_transform(self, data):
        pass


DATA_DIR = "./data/base"


class TSFDataLoader(TSDataLoader):
    def __init__(
        self,
        filename,
        batch_size,
        seq_len,
        pred_len,
        feature_type,
        target_column,
        limit=[None, None],
        print_option=False,
    ):
        self.filename = filename
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.feature_type = feature_type
        self.target_column = target_column
        self.target_idx = None

        self._read_data(limit, print_option)

    def _read_data(self, limit, print_option):  # including scaler
        filepath = f"{DATA_DIR}/{self.filename}.csv"
        print("dir:", filepath)
        df_raw = pd.read_csv(filepath)
        df_raw["dt"] = pd.to_datetime(df_raw["dt"])
        df = df_raw.set_index("dt")

        def slice_data(arr, start, end):
            _slice = slice(start, end, None)
            arr = arr[_slice]
            return arr

        df = slice_data(df, limit[0], limit[1])

        self.df = df.copy()

        if self.feature_type == "S":
            df = df[[self.target_column]]
        elif self.feature_type == "MS":
            self.target_idx = df.columns.get_loc(self.target_column) - 1
        print(f"target_column idx: {self.target_idx}")

        # split train/valid/test
        n = len(df)
        if self.filename.startswith("ETTm"):
            train_end = 12 * 30 * 24 * 4
            val_end = train_end + 4 * 30 * 24 * 4
            test_end = val_end + 4 * 30 * 24 * 4
        elif self.filename.startswith("ETTh"):
            train_end = 12 * 30 * 24
            val_end = train_end + 4 * 30 * 24
            test_end = val_end + 4 * 30 * 24
        else:
            train_end = int(n * 0.7)
            val_end = n - int(n * 0.2)
            test_end = n

        train_df = df[:train_end]
        val_df = df[train_end - self.seq_len : val_end]
        test_df = df[val_end - self.seq_len : test_end]

        # standardize by training set
        self.scaler = StandardScaler()
        self.scaler.fit(train_df.values)

        def scale_df(df, scaler):
            data = scaler.transform(df.values)
            return pd.DataFrame(data, index=df.index, columns=df.columns)

        self.train_df = scale_df(train_df, self.scaler)
        self.val_df = scale_df(val_df, self.scaler)
        self.test_df = scale_df(test_df, self.scaler)
        self.n_feature = self.train_df.shape[-1]

        if print_option:
            display(self.df.head(2))
            print(
                f"data split index: (0~{train_end})/({train_end}~{val_end})/({val_end}~{test_end})"
            )
            print(f"data length: {train_df.shape}/{val_df.shape}/{test_df.shape}")
            print(f"scaled data e.g.:")
            display(self.train_df[: self.seq_len])
            display(self.train_df[self.seq_len : self.seq_len + self.pred_len])
            print(f"n_feature: {self.n_feature}")
            print(f"target_column idx: {self.target_idx}")
            print(f"feature type: {self.feature_type}")

    def _make_dataloader(self, data_df, shuffle=True):
        dataset = TimeSeriesDataset(
            data_df, self.seq_len, self.pred_len, self.target_idx
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        return dataloader

    def get_dataloaders(self, shuffle=True):
        train_dataloader = self._make_dataloader(self.train_df, shuffle=shuffle)
        val_dataloader = self._make_dataloader(self.val_df, shuffle=False)
        test_dataloader = self._make_dataloader(self.test_df, shuffle=False)

        self.dataloaders = dict(
            train=train_dataloader, val=val_dataloader, test=test_dataloader
        )
        return self.dataloaders

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def get_data_info(self, phase="train"):
        x_batch, y_batch = next(iter(self.dataloaders[phase]))
        n_feature = x_batch.shape[-1]
        print(f"x_batch: {x_batch.shape}")
        print(f"y_batch: {y_batch.shape}")
        print(f"n_feature:{n_feature}")
        return ((x_batch, y_batch), n_feature, self.target_idx)

    def get_df(self):
        return self.df


class TimeSeriesDataset(Dataset):
    def __init__(self, df, seq_len, pred_len, target_idx):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_idx = target_idx
        self.data = np.array(df, dtype=np.float32)

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len, :]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len, :]
        if self.target_idx:
            y = y[:, self.target_idx]

        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)


# if self.timeenc == 0:
#     df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
#     df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
#     df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
#     df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
#     df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
#     df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
