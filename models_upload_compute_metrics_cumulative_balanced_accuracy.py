import importlib

import os
import torch
import numpy as np
import pandas as pd
import seaborn as sn
from pathlib import Path
from torch import Tensor
from captum.attr import LayerGradCam
from matplotlib import pyplot as plt
from daft.training.metrics import Metric
from typing import Dict, Sequence, Tuple
from sklearn.metrics import confusion_matrix
from daft.data_utils.adni_hdf import Constants
from daft.testing.test_and_save import ModelTester
from daft.cli import HeterogeneousModelFactory, create_parser


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
        if i == (len(metrics) - 1):
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


def get_folders(experiments_dir, exp_name):
    _, folders, _ = next(os.walk(os.path.join(experiments_dir, exp_name)))
    folders.sort()

    if ("4classes" in exp_name) or ("4group" in exp_name):
        num_classes = 4
    elif ("2classes" in exp_name) or ("2group" in exp_name):
        num_classes = 2
    else:
        num_classes = 3

    return folders, num_classes

def get_datafiles_names(num_classes, num):
    if num_classes != 3:
        train_file = "manualfile_" + str(num_classes) + "g_train_fold" + str(num + 1) + ".h5"
        val_file = "manualfile_" + str(num_classes) + "g_val_fold" + str(num + 1) + ".h5"
        test_file = "manualfile_" + str(num_classes) + "g_test_fold" + str(num + 1) + ".h5"
    else:
        train_file = "manualfile_" + str(num_classes) + "edss_train_fold" + str(num + 1) + ".h5"
        val_file = "manualfile_" + str(num_classes) + "edss_val_fold" + str(num + 1) + ".h5"
        test_file = "manualfile_" + str(num_classes) + "edss_test_fold" + str(num + 1) + ".h5"

    return train_file, val_file, test_file

def main():

    experiment_name = ['2classes_daft_v2_d6_prelu_augmentation',
             '3classes_daft_v2_d6_prelu_augmentation',
             '4classes_daft_v2_d6_prelu_augmentation',
             "11may_2group_NOdropout_NOaugmentation_manualfile",
             "11may_3edss_NOdropout_NOaugmentation_manualfile",
             '11may_4group_NOdropout_NOaugmentation_manualfile',
             '2classes_daft_v2_d6',
             '3classes_daft_v2_d6',
             '4classes_daft_v2_d6',
             "4group_Nodropout_augmentation_manualfile",
             "2group_Nodropout_augmentation_manualfile",
             "3edss_Nodropout_augmentation_manualfile",
             '2classes_daft_v2_d6_augmentation',
             '3classes_daft_v2_d6_augmentation',
             '4classes_daft_v2_d6_augmentation',
             ]

    corresponding_net = ["daft_v2_d6_prelu",
                         "daft_v2_d6_prelu",
                         "daft_v2_d6_prelu",
                         "daft_v2",
                         "daft_v2",
                         "daft_v2",
                         "daft_v2_d6",
                         "daft_v2_d6",
                         "daft_v2_d6",
                         "daft_v2",
                         "daft_v2",
                         "daft_v2",
                         "daft_v2_d6",
                         "daft_v2_d6",
                         "daft_v2_d6"
                         ]

    experiments_dir = "/home/ecarvajal /Desktop/MyCloneDAFT/DAFT/experiments_clf"

    predictions_dict = {}

    for exp_name, net in zip(experiment_name, corresponding_net):

        if not os.path.exists(os.path.join(experiments_dir, exp_name)):
            print(" the path ", exp_name, " doesnt exist!")
            return 1
        else:
            print("experiment ", exp_name)

        print("start experiment metrics: ", exp_name)
        folders, num_classes = get_folders(experiments_dir, exp_name)

        for num, folder in enumerate(folders):

            train_file, val_file, test_file = get_datafiles_names(num_classes, num)

            # load the checkpoint model with the given folder, network, number of classes, data, channels and batch size
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

            # to print model input and output sizes in every layer
            # It will not work using the daft block structure incompatible
            #from torchinfo import summary
            #summary(model, ((16, 3, 181, 217, 181), (16, 1)), device='cpu')


            # upload all the predictions and metrics outputs given the given model and test set dataloader
            metrics_fold, preds = evaluate_model(
                metrics=factory.get_test_metrics(), model=model, data=test_loader, progressbar=True,
            )

            targets = preds['target']
            predictions_dict["fold" + str(num + 1)] = preds['logits']

            print("fold ", num)
            print("metrics ", metrics_fold)

        probs_dict = {}

        # the predictions are a tensor with one of several values. This depends on the task
        # to compute the probability is necessary to apply a sigmoid function
        for k, v in predictions_dict.items():
            probs_dict[k] = torch.sigmoid(v).detach().numpy()

        # we concatenate the probabilities of each fold
        new_probs = np.mean([probs_dict["fold1"], probs_dict["fold2"], probs_dict["fold3"]], axis=0)
        if num_classes == 2:
            pred2 = torch.zeros([new_probs.shape[0], 2])
            pred2[new_probs[:, 0] > 0.5, 1] = 1
            pred2[new_probs[:, 0] <= 0.5, 0] = 1
            new_probs = pred2

        # get the prediction with the maximum probability from the probabilities of the folds
        new_pred = new_probs.argmax(axis=1)

        # confusion matrix computation
        matrix = confusion_matrix(targets, new_pred)

        print("confusion matrix ", matrix)

        if matrix.shape[0] == 2:
            classes = Constants.DIAGNOSIS_CODES_BINARY.keys()
        elif matrix.shape[0] == 3:
            classes = Constants.DIAGNOSIS_CODES_MULTICLASS3.keys()
        elif matrix.shape[0] == 4:
            classes = Constants.DIAGNOSIS_CODES_MULTICLASS.keys()

        # ensemble confusion matrix drawing in the corresponding folder
        df_cm = pd.DataFrame(matrix,
                             index=[i for i in classes],
                             columns=[i for i in classes])
        plt.figure(figsize=(12, 7))
        fig_conf_matrix = sn.heatmap(df_cm, annot=True).get_figure()
        fig_conf_matrix.savefig(os.path.join(experiments_dir,
                                             exp_name,
                                             "acum_confmatrix_" + exp_name + "_test_fold" + str(num + 1) + ".png"))

        ########################################

        # balanced accuracy computation
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
        print("end experiment metrics: ", exp_name)

if __name__ == "__main__":
    main()
