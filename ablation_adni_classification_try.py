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
from pathlib import Path
from typing import Any, Dict
from train import main as main_train


def get_experiment(data_dir: Path,
                   experiment_name: str,
                   net: str,
                   num_classes: int,
                   train_path: Path,
                   val_path: Path,
                   test_path: Path,
                   batch_size: int,
                   input_channels: int) -> Dict[str, Any]:
    cfg = {
        "epoch": "150",
        "batchsize": str(batch_size),
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
        "input_channels": str(input_channels),
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


def files_names_groups(classes, num):
    if classes != 3:
        train_file = "manualfile_" + str(classes) + "g_train_fold" + str(num) + ".h5"
        val_file = "manualfile_" + str(classes) + "g_val_fold" + str(num) + ".h5"
        test_file = "manualfile_" + str(classes) + "g_test_fold" + str(num) + ".h5"

    else:
        train_file = "manualfile_" + str(classes) + "edss_train_fold" + str(num) + ".h5"
        val_file = "manualfile_" + str(classes) + "edss_val_fold" + str(num) + ".h5"
        test_file = "manualfile_" + str(classes) + "edss_test_fold" + str(num) + ".h5"

    return train_file, val_file, test_file


def main():

    for net in ["daft_v2_p6_replu"]:
        for classes in [2, 3, 4]:
            experiment_name = str(classes) + "classes_" + net
            print(" start experiment ", experiment_name)
            for fold in [1, 2, 3]:
                print(" fold ", fold)
                train_file, val_file, test_file = files_names_groups(classes, fold)
                args = get_experiment("/home/ecarvajal /Desktop/MyCloneDAFT/DAFT/data_dir",
                                      experiment_name,
                                      net,
                                      classes,
                                      train_file,
                                      val_file,
                                      test_file,
                                      16, # batch_size
                                      3)  # channels
                main_train(args)
            print(" end experiment ", experiment_name)

if __name__ == "__main__":
    main()
