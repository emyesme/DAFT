import importlib



import os
import torch
import pandas as pd
import numpy as np
import seaborn as sn
from pathlib import Path
from torch import Tensor
from matplotlib import pyplot as plt
from daft.training.metrics import Metric
from daft.data_utils.adni_hdf import Constants
from daft.testing.test_and_save import ModelTester
from daft.cli import HeterogeneousModelFactory, create_parser
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple




def load_model(
    checkpoints_dir: Path,
    data_dir: str,
    experiment_name: str,
    net: str,
    classes: int,
    train_data: str,
    val_data: str,
    test_data: str,
    batch_size: str,
    input_channels: str
) -> torch.nn.Module:

    from ablation_adni_classification_try import get_experiment

    args = get_experiment(data_dir,
                          experiment_name,
                          net,
                          classes,
                          train_data,
                          val_data,
                          test_data,
                          batch_size,
                          input_channels)

    args = create_parser().parse_args(args=args)

    factory = HeterogeneousModelFactory(args)

    best_net_path = os.path.join(checkpoints_dir, "best_discriminator_balanced_accuracy.pth")

    _, _, test_loader = factory.get_data()
    best_discriminator = factory.get_and_init_model()
    best_discriminator.load_state_dict(torch.load(best_net_path))

    return factory, best_discriminator, test_loader


def evaluate_model(*, metrics: Sequence[Metric], **kwargs) -> Tuple[Dict[str, float], Dict[str, Tensor]]:
    """Obtain predictions from model and evaluate its performance.

    Args:
      metrics (list):
        List of metrics to compute on test data
      model (BaseModel):
        Instance of model to call.
      data (DataLoaderWrapper):
        Instance of DataLoader to obtain batches from.
        Keys of `data.output_names` must be contained in keys of `model.input_names`
        and `loss.input_names`.
      device (torch.device):
        Optional; Which device to run on.
      progressbar (bool):
        Optional; Whether to display a progess bar.

    Returns:
        Two dicts. The first dict contains the computed metrics
        on the entire data. The second dict contains the model's raw
        output for each data point.
    """
    tester = ModelTester(**kwargs)


    predictions, unconsumed_inputs = tester.predict_all()

    metrics_dict = {}
    for i, m in enumerate(metrics):
        if i == (len(metrics)-1):
            print("last metric")
            m.reset()
            m.update(inputs=unconsumed_inputs, outputs=predictions)
            metrics_dict.update(m.values_matrix())
        else:
            m.reset()
            m.update(inputs=unconsumed_inputs, outputs=predictions)
            metrics_dict.update(m.values())

    predictions.update(unconsumed_inputs)

    return metrics_dict, predictions


def main():
    # fold 1 no augmentation
    experiment_name_nodaft = ["2group_NOaugmentation_nodaft_manualfile",
                       "4group_NOaugmentation_nodaft_manualfile",
                       "3edss_NOaugmentation_nodaft_manualfile"]

    experiment_name_daft = ["10may_2group_augmentation_manualfile",
                            "11may_2group_NOaugmentation_manualfile",
                            "11may_2group_NOdropout_NOaugmentation_manualfile",
                            "10may_4group_augmentation_manualfile",
                            "11may_4group_NOaugmentation_manualfile",
                            "11may_4group_NOdropout_NOaugmentation_manualfile",
                            "10may_3edss_augmentation_manualfile",
                            "11may_3edss_NOaugmentation_manualfile",
                            "11may_3edss_NOdropout_NOaugmentation_manualfile"]

    experiment_name_daft = ["11may_2group_NOdropout_NOaugmentation_manualfile",
                            "11may_4group_NOdropout_NOaugmentation_manualfile",
                            "11may_3edss_NOdropout_NOaugmentation_manualfile"]

    experiment_name_t1 = [#"baseline_T1_4groups",
                          "baseline_T1_2groups"]
                          #"baseline_T1_3edss"]

    experiment_name_daft = ["4group_Nodropout_augmentation_manualfile",
                            "3edss_Nodropout_augmentation_manualfile"]

    net = "daft_v2_nodropout"

    experiments_dir = "/home/ecarvajal /Desktop/MyCloneDAFT/DAFT/experiments_clf"

    for exp_name in experiment_name_daft:
        print("folders ", exp_name)
        _, folders, _ = next(os.walk(os.path.join(experiments_dir, exp_name)))

        folders.sort()

        if "4" in exp_name:
            num_classes = 4
        elif "2" in exp_name:
            num_classes = 2
        else:
            num_classes = 3

        for num, folder in enumerate(folders):
            print("folder ", os.path.join(experiments_dir, exp_name, folder))

            if num_classes != 3:
                train_file = "manualfile_" + str(num_classes) + "g_train_fold" + str(num+1) + ".h5"
                val_file = "manualfile_" + str(num_classes) + "g_val_fold" + str(num+1) + ".h5"
                test_file = "manualfile_" + str(num_classes) + "g_test_fold" + str(num+1) + ".h5"
            elif num_classes == 3:
                train_file = "manualfile_" + str(num_classes) + "edss_train_fold" + str(num+1) + ".h5"
                val_file = "manualfile_" + str(num_classes) + "edss_val_fold" + str(num+1) + ".h5"
                test_file = "manualfile_" + str(num_classes) + "edss_test_fold" + str(num+1) + ".h5"
            #elif num_classes == 4:
            #    train_file = "manualfile_t1_train_fold" + str(num + 1) + ".h5"
            #    val_file = "manualfile_t1_val_fold" + str(num + 1) + ".h5"
            #    test_file = "manualfile_t1_test_fold" + str(num + 1) + ".h5"

            factory, model, test_loader = load_model(
                os.path.join(experiments_dir, exp_name, folder, "checkpoints"),
                "/home/ecarvajal /Desktop/MyCloneDAFT/DAFT/data_dir",
                "it doesnt affect",
                net,
                num_classes,
                train_file,
                val_file,
                test_file,
                str(16),
                str(3)
            )
            metrics, preds = evaluate_model(
                metrics=factory.get_test_metrics(), model=model, data=test_loader, progressbar=True,
            )

            #print("preds ", preds)

            # draw confusion matrix of test set
            matrix = metrics["conf_matrix"]

            if matrix.shape[0] == 2:
                classes = Constants.DIAGNOSIS_CODES_BINARY.keys()
            elif matrix.shape[0] == 3:
                classes = Constants.DIAGNOSIS_CODES_MULTICLASS3.keys()
            elif matrix.shape[0] == 4:
                classes = Constants.DIAGNOSIS_CODES_MULTICLASS.keys()

            # when checking the create.hd5 just count and compare
            df_cm = pd.DataFrame(matrix,
                                 index=[i for i in classes],
                                 columns=[i for i in classes])
            plt.figure(figsize=(12, 7))
            fig_conf_matrix = sn.heatmap(df_cm, annot=True).get_figure()
            fig_conf_matrix.savefig(os.path.join(experiments_dir,
                                                 exp_name,
                                                 "confmatrix_" + exp_name + "_test_fold" + str(num+1) + ".png"))
            print("metrics experiment: ", exp_name, " fold: ", str(num+1), " metrics: ", metrics)



if __name__ == "__main__":
    main()
