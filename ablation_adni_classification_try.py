# This file is part of Dynamic Affine Feature Map Transform (DAFT).
#
# DAFT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DAFT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DAFT. If not, see <https://www.gnu.org/licenses/>.
import os
import socket
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import torch

from daft.evaluate import load_model_and_evaluate
from train import main as main_train


def get_experiment(data_dir: Path,
                   experiment_name: str,
                   net: str,
                   num_classes: int,
                   train_path: Path,
                   val_path: Path,
                   test_path: Path,
                   batch_size: str,
                   optimizer: str,
                   activation: str,
                   lr: str,
                   decay_rate: str,
                   input_channels: str) -> Dict[str, Any]:
    cfg = {
        "epoch": "100",
        "batchsize": batch_size,
        "optimizer": optimizer,#SGD balanced/06 03 - 20 12
        "workers": "2",
        "train_data": os.path.join(data_dir, train_path),
        "val_data": os.path.join(data_dir, val_path),
        "test_data": os.path.join(data_dir, test_path),
        "discriminator_net": net,
        "activation": activation,
        "learning_rate": lr,
        "decay_rate": decay_rate,
        "experiment_name": experiment_name,
        "num_classes": str(num_classes),
        "input_channels": input_channels,
        "n_basefilters": "4",
        "bottleneck_factor": "7",
        "normalize_image": "minmax",
        "dataset": "longitudinal",
        "task": "clf"

    }
    cmd = []
    for k, v in cfg.items():
        cmd.append(f"--{k}")
        cmd.append(v)

    cmd.append("--contrastive_loss")

    return cmd


def main():
    experiment_name = "SiameseCL_config_daft_best_secondtry"
    net = "siameseCL"
    for fold in [1, 2, 3]:
        print("siamese fold ", fold)
        args = get_experiment("/home/ecarvajal /Desktop/DAFT_branch/DAFT/data_dir",
                              experiment_name,
                              net, # name of the network
                              2, # number of classes
                              "mix_balanced_2g_train_fold"+str(fold)+".h5",
                              "mix_balanced_2g_val_fold"+str(fold)+".h5",
                              "mix_balanced_2g_test_fold"+str(fold)+".h5",
                            "2", # batch size
                            "AdamW", # optimizer
                            "tanh", # activation function
                            "0.00013", # lr
                            "0.001", # decay rate
                            "3" # input channels
                              )
        main_train(args)
        print("end siamese fold ", fold)


if __name__ == "__main__":
    main()
