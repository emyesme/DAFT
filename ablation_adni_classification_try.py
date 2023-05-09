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
                   num_classes: int,
                   train_path: Path,
                   val_path: Path,
                   test_path: Path) -> Dict[str, Any]:
    cfg = {
        "epoch": "150",
        "batchsize": "16",
        "optimizer": "AdamW",
        "workers": "2",
        "train_data": os.path.join(data_dir, train_path),
        "val_data": os.path.join(data_dir, val_path),
        "test_data": os.path.join(data_dir, test_path),
        "discriminator_net": "daft_v2",
        "activation": "tanh",
        "learning_rate": "0.00013",
        "decay_rate": "0.001",
        "experiment_name": experiment_name,
        "num_classes": str(num_classes),
        "input_channels": "3",
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

    return cmd


def main():
    experiment_name = "mask_flair_t1_folds_manual_good_16batch_2g_aug"
    print(" ## experiment 2 groups cisrr sppp")
    args = get_experiment("/home/ecarvajal /Desktop/MyCloneDAFT/DAFT/data_dir",
                          experiment_name,
                          2,
                          "manual_2g_train_fold1.h5",
                          "manual_2g_val_fold1.h5",
                          "manual_2g_test_fold1.h5"
                          )
    main_train(args)

    print(" end fold 1")
    args = get_experiment("/home/ecarvajal /Desktop/MyCloneDAFT/DAFT/data_dir",
                          experiment_name,
                          2,
                          "manual_2g_train_fold2.h5",
                          "manual_2g_val_fold2.h5",
                          "manual_2g_test_fold2.h5"
                          )
    main_train(args)

    print(" end fold 2")
    args = get_experiment("/home/ecarvajal /Desktop/MyCloneDAFT/DAFT/data_dir",
                          experiment_name,
                          2,
                          "manual2_2g_train_fold3.h5",
                          "manual2_2g_val_fold3.h5",
                          "manual2_2g_test_fold3.h5"
                          )
    main_train(args)

    print("end fold 3")

    print(" ## experiment 4 groups cis rr sp pp")

    experiment_name = "mask_flair_t1_folds_manual_good_16batch_4g_aug"
    args = get_experiment("/home/ecarvajal /Desktop/MyCloneDAFT/DAFT/data_dir",
                          experiment_name,
                          4,
                          "manual_4g_train_fold1.h5",
                          "manual_4g_val_fold1.h5",
                          "manual_4g_test_fold1.h5"
                          )
    main_train(args)

    print(" end fold 1")
    args = get_experiment("/home/ecarvajal /Desktop/MyCloneDAFT/DAFT/data_dir",
                          experiment_name,
                          4,
                          "manual_4g_train_fold2.h5",
                          "manual_4g_val_fold2.h5",
                          "manual_4g_test_fold2.h5"
                          )
    main_train(args)

    print(" end fold 2")
    args = get_experiment("/home/ecarvajal /Desktop/MyCloneDAFT/DAFT/data_dir",
                          experiment_name,
                          4,
                          "manual_4g_train_fold3.h5",
                          "manual_4g_val_fold3.h5",
                          "manual_4g_test_fold3.h5"
                          )
    main_train(args)

    print("end fold 3")


if __name__ == "__main__":
    main()
