import importlib

import os
import torch
import nibabel
import nibabel as nib
import seaborn as sn
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from torch import Tensor
from captum.attr import Saliency
from captum.attr import LayerGradCam
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

    ######## type of method for explainability
    best_discriminator.eval()
    model = Saliency(best_discriminator)

    return factory, model, best_discriminator, test_loader


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
        if i == (len(metrics) - 1):
            #print("last metric")
            m.reset()
            m.update(inputs=unconsumed_inputs, outputs=predictions)
            metrics_dict.update(m.values_matrix())
        else:
            m.reset()
            m.update(inputs=unconsumed_inputs, outputs=predictions)
            metrics_dict.update(m.values())

    predictions.update(unconsumed_inputs)
    return metrics_dict, predictions

def get_model_inputs_from_batch(batch: Dict[int, Tensor]) -> Sequence[Tensor]:
    # to get rid of unnecessary dimensions when adding multiple channels
    batch[0] = batch[0].squeeze(0)
    in_batch = tuple([batch[num] for num in [0, 1]])
    return in_batch


def main():
    # fold 1 no augmentation
    experiment_name_daft_2g = [#"10may_2group_augmentation_manualfile",
                               #"11may_2group_NOdropout_NOaugmentation_manualfile",
                            "11may_2group_NOaugmentation_manuallfile",
                            "2group_Nodropout_augmentation_manualfile"
                            ]

    experiment_name_daft_ed = ["10may_3edss_augmentation_manualfile",
                               "11may_3edss_NOdropout_NOaugmentation_manualfile",
                            "11may_3edss_NOaugmentation_manualfile",
                            "3edss_Nodropout_augmentation_manualfile"
                            ]

    experiment_name_daft_4g = ["11may_4group_NOaugmentation_manualfile",
                            "10may_4group_augmentation_manualfile",
                            "4group_Nodropout_augmentation_manualfile",
                            "4group_NOaugmentation_nodaft_manualfile"]

    experiments_dropout = [#"11may_2group_NOaugmentation_manualfile",
                            #"2group_Nodropout_augmentation_manualfile",
                           #"10may_3edss_augmentation_manualfile",
                            #"11may_3edss_NOdropout_NOaugmentation_manualfile",
                            #"11may_3edss_NOaugmentation_manualfile",
                            #"3edss_Nodropout_augmentation_manualfile",
                            #"11may_4group_NOaugmentation_manualfile",
                            #"10may_4group_augmentation_manualfile",
                            #"4group_Nodropout_augmentation_manualfile"
                            '11may_4group_NOdropout_NOaugmentation_manualfile']

    experiments_d = ["2classes_nodaft",
                     "3classes_nodaft",
                     "4classes_nodaft"]

    corresponding_net = ["nodaft",
                         "nodaft",
                         "nodaft"]

    experiments_dir = "/home/ecarvajal /Desktop/MyCloneDAFT/DAFT/experiments_clf"

    predictions_dict = {}

    for exp_name, net in zip(experiments_d, corresponding_net):

        if not os.path.exists(os.path.join(experiments_dir, exp_name)):
            print(" the path ", exp_name, " doesnt exist!")
            return 1
        else:
            print("experiment ", exp_name)

        _, folders, _ = next(os.walk(os.path.join(experiments_dir, exp_name)))
        folders.sort()

        if "4classes" in exp_name:
            num_classes = 4
        elif "2classes" in exp_name:
            num_classes = 2
        else:
            num_classes = 3

        for num, folder in enumerate(folders):

            if num_classes != 3:
                train_file = "manualfile_" + str(num_classes) + "g_train_fold" + str(num + 1) + ".h5"
                val_file = "manualfile_" + str(num_classes) + "g_val_fold" + str(num + 1) + ".h5"
                test_file = "manualfile_" + str(num_classes) + "g_test_fold" + str(num + 1) + ".h5"
            else:
                train_file = "manualfile_" + str(num_classes) + "edss_train_fold" + str(num + 1) + ".h5"
                val_file = "manualfile_" + str(num_classes) + "edss_val_fold" + str(num + 1) + ".h5"
                test_file = "manualfile_" + str(num_classes) + "edss_test_fold" + str(num + 1) + ".h5"

            factory, saliency, model, test_loader = load_model(
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

            # cumulate models or cumulate predictions?

            metrics_fold, preds = evaluate_model(
                metrics=factory.get_test_metrics(), model=model, data=test_loader, progressbar=True,
            )

            targets = preds['target']
            predictions_dict["fold"+str(num+1)] = preds['logits']

            targets = preds['target']
            predictions_dict["fold" + str(num + 1)] = preds['logits']

            print("fold ", num)
            print("metrics ", metrics_fold)

        probs_dict = {}

        for k, v in predictions_dict.items():
            probs_dict[k] = torch.sigmoid(v).detach().numpy()

        new_probs = np.mean([probs_dict["fold1"], probs_dict["fold2"], probs_dict["fold3"]], axis=0)
        if num_classes == 2:

            pred2 = torch.zeros([new_probs.shape[0], 2])
            pred2[new_probs[:, 0] > 0.5, 1] = 1
            pred2[new_probs[:, 0] <= 0.5, 0] = 1
            new_probs = pred2

        new_pred = new_probs.argmax(axis=1)

        # confusion matrices
        from sklearn.metrics import confusion_matrix

        matrix = confusion_matrix(targets, new_pred)

        print("confusion matrix ", matrix)

        # balanced accuracy
        correct = np.zeros(num_classes)
        total = np.zeros(num_classes)

        if num_classes == 2:
            is_correct = new_pred == targets
        else:
            is_correct = new_pred == targets.numpy()

        classes, counts = np.unique(targets, return_counts=True)
        for i, c in zip(classes, counts):
            total[i] += c
            correct[i] += is_correct[targets == i].sum()

        print("correct ", correct)
        print("total ", total)
        balanced_accuracy = np.mean(correct / total)

        print("metrics experiment: ", exp_name, " balanced_accuracy: ", balanced_accuracy)


if __name__ == "__main__":
    main()
