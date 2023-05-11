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
                   test_path: Path) -> Dict[str, Any]:
    cfg = {
        "epoch": "150",
        "batchsize": "16",
        "optimizer": "AdamW",
        "workers": "2",
        "train_data": os.path.join(data_dir, train_path),
        "val_data": os.path.join(data_dir, val_path),
        "test_data": os.path.join(data_dir, test_path),
        "discriminator_net": net,
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
    '''
    print(" ## experiment 3 groups edss")

    experiment_name = "11may_3edss_NOaugmentation_manualfile"
    net = "daft_v2"
    args = get_experiment("/home/ecarvajal /Desktop/MyCloneDAFT/DAFT/data_dir",
                          experiment_name,
                          net,
                          3,
                          "manualfile_3edss_train_fold1.h5",
                          "manualfile_3edss_val_fold1.h5",
                          "manualfile_3edss_test_fold1.h5"
                          )
    main_train(args)

    print(" end fold 1")
    args = get_experiment("/home/ecarvajal /Desktop/MyCloneDAFT/DAFT/data_dir",
                          experiment_name,
                          net,
                          3,
                          "manualfile_3edss_train_fold2.h5",
                          "manualfile_3edss_val_fold2.h5",
                          "manualfile_3edss_test_fold2.h5"
                          )
    main_train(args)

    print(" end fold 2")
    args = get_experiment("/home/ecarvajal /Desktop/MyCloneDAFT/DAFT/data_dir",
                          experiment_name,
                          net,
                          3,
                          "manualfile_3edss_train_fold3.h5",
                          "manualfile_3edss_val_fold3.h5",
                          "manualfile_3edss_test_fold3.h5"
                          )
    main_train(args)

    print("end fold 3")

    experiment_name = "11may_2group_NOaugmentation_manualfile"
    print(" ## experiment 2 groups cisrr sppp")
    args = get_experiment("/home/ecarvajal /Desktop/MyCloneDAFT/DAFT/data_dir",
                          experiment_name,
                          net,
                          2,
                          "manualfile_2g_train_fold1.h5",
                          "manualfile_2g_val_fold1.h5",
                          "manualfile_2g_test_fold1.h5"
                          )
    main_train(args)

    print(" end fold 1")
    args = get_experiment("/home/ecarvajal /Desktop/MyCloneDAFT/DAFT/data_dir",
                          experiment_name,
                          net,
                          2,
                          "manualfile_2g_train_fold2.h5",
                          "manualfile_2g_val_fold2.h5",
                          "manualfile_2g_test_fold2.h5"
                          )
    main_train(args)

    print(" end fold 2")
    args = get_experiment("/home/ecarvajal /Desktop/MyCloneDAFT/DAFT/data_dir",
                          experiment_name,
                          net,
                          2,
                          "manualfile_2g_train_fold3.h5",
                          "manualfile_2g_val_fold3.h5",
                          "manualfile_2g_test_fold3.h5"
                          )
    main_train(args)

    print("end fold 3")

    print(" ## experiment 4 groups cis rr sp pp")

    experiment_name = "11may_4group_NOaugmentation_manualfile"
    args = get_experiment("/home/ecarvajal /Desktop/MyCloneDAFT/DAFT/data_dir",
                          experiment_name,
                          net,
                          4,
                          "manualfile_4g_train_fold1.h5",
                          "manualfile_4g_val_fold1.h5",
                          "manualfile_4g_test_fold1.h5"
                          )
    main_train(args)

    print(" end fold 1")
    args = get_experiment("/home/ecarvajal /Desktop/MyCloneDAFT/DAFT/data_dir",
                          experiment_name,
                          net,
                          4,
                          "manualfile_4g_train_fold2.h5",
                          "manualfile_4g_val_fold2.h5",
                          "manualfile_4g_test_fold2.h5"
                          )
    main_train(args)

    print(" end fold 2")
    args = get_experiment("/home/ecarvajal /Desktop/MyCloneDAFT/DAFT/data_dir",
                          experiment_name,
                          net,
                          4,
                          "manualfile_4g_train_fold3.h5",
                          "manualfile_4g_val_fold3.h5",
                          "manualfile_4g_test_fold3.h5"
                          )
    main_train(args)

    print("end fold 3")

    print(" ## experiment 3 groups edss no dropout")

    experiment_name = "11may_3edss_NOdropout_NOaugmentation_manualfile"
    net = "daft"
    args = get_experiment("/home/ecarvajal /Desktop/MyCloneDAFT/DAFT/data_dir",
                          experiment_name,
                          net,
                          3,
                          "manualfile_3edss_train_fold1.h5",
                          "manualfile_3edss_val_fold1.h5",
                          "manualfile_3edss_test_fold1.h5"
                          )
    main_train(args)

    print(" end fold 1")
    args = get_experiment("/home/ecarvajal /Desktop/MyCloneDAFT/DAFT/data_dir",
                          experiment_name,
                          net,
                          3,
                          "manualfile_3edss_train_fold2.h5",
                          "manualfile_3edss_val_fold2.h5",
                          "manualfile_3edss_test_fold2.h5"
                          )
    main_train(args)

    print(" end fold 2")
    args = get_experiment("/home/ecarvajal /Desktop/MyCloneDAFT/DAFT/data_dir",
                          experiment_name,
                          net,
                          3,
                          "manualfile_3edss_train_fold3.h5",
                          "manualfile_3edss_val_fold3.h5",
                          "manualfile_3edss_test_fold3.h5"
                          )
    main_train(args)

    print("end fold 3")

    experiment_name = "11may_2group_NOdropout_NOaugmentation_manualfile"
    print(" ## experiment 2 groups cisrr sppp")
    args = get_experiment("/home/ecarvajal /Desktop/MyCloneDAFT/DAFT/data_dir",
                          experiment_name,
                          net,
                          2,
                          "manualfile_2g_train_fold1.h5",
                          "manualfile_2g_val_fold1.h5",
                          "manualfile_2g_test_fold1.h5"
                          )
    main_train(args)

    print(" end fold 1")
    args = get_experiment("/home/ecarvajal /Desktop/MyCloneDAFT/DAFT/data_dir",
                          experiment_name,
                          net,
                          2,
                          "manualfile_2g_train_fold2.h5",
                          "manualfile_2g_val_fold2.h5",
                          "manualfile_2g_test_fold2.h5"
                          )
    main_train(args)

    print(" end fold 2")
    args = get_experiment("/home/ecarvajal /Desktop/MyCloneDAFT/DAFT/data_dir",
                          experiment_name,
                          net,
                          2,
                          "manualfile_2g_train_fold3.h5",
                          "manualfile_2g_val_fold3.h5",
                          "manualfile_2g_test_fold3.h5"
                          )
    main_train(args)

    print("end fold 3")

    print(" ## experiment 4 groups cis rr sp pp")

    experiment_name = "11may_4group_NOdropout_NOaugmentation_manualfile"
    args = get_experiment("/home/ecarvajal /Desktop/MyCloneDAFT/DAFT/data_dir",
                          experiment_name,
                          net,
                          4,
                          "manualfile_4g_train_fold1.h5",
                          "manualfile_4g_val_fold1.h5",
                          "manualfile_4g_test_fold1.h5"
                          )
    main_train(args)

    print(" end fold 1")
    args = get_experiment("/home/ecarvajal /Desktop/MyCloneDAFT/DAFT/data_dir",
                          experiment_name,
                          net,
                          4,
                          "manualfile_4g_train_fold2.h5",
                          "manualfile_4g_val_fold2.h5",
                          "manualfile_4g_test_fold2.h5"
                          )
    main_train(args)

    print(" end fold 2")
    args = get_experiment("/home/ecarvajal /Desktop/MyCloneDAFT/DAFT/data_dir",
                          experiment_name,
                          net,
                          4,
                          "manualfile_4g_train_fold3.h5",
                          "manualfile_4g_val_fold3.h5",
                          "manualfile_4g_test_fold3.h5"
                          )
    main_train(args)

    print("end fold 3")

    '''
    experiment_name = "10may_4group_augmentation_manualfile"
    # missing this with augmentation adni_hdf uncomment img_transform line
    print(" end fold 2")
    args = get_experiment("/home/ecarvajal /Desktop/MyCloneDAFT/DAFT/data_dir",
                          experiment_name,
                          "daft_v2",
                          4,
                          "manualfile_4g_train_fold3.h5",
                          "manualfile_4g_val_fold3.h5",
                          "manualfile_4g_test_fold3.h5"
                          )
    main_train(args)

    print("end fold 3")
    


if __name__ == "__main__":
    main()
