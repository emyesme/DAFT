import os
import torch
import numpy as np
import pandas as pd
import seaborn as sn
from pathlib import Path
from torch import Tensor
from matplotlib import pyplot as plt
from daft.training.metrics import Metric
from typing import Dict, Sequence, Tuple
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
                          "AdamW",  # optimizer
                          "tanh",  # activation function
                          "0.00013",  # lr
                          "0.001",  # decay rate
                          input_channels
                          )

    args = create_parser().parse_args(args=args)

    factory = HeterogeneousModelFactory(args)

    best_net_path = os.path.join(checkpoints_dir, "best_discriminator_balanced_accuracy.pth")

    _, _, test_loader = factory.get_data()
    best_discriminator = factory.get_and_init_model()
    best_discriminator.load_state_dict(torch.load(best_net_path))

    return factory, best_discriminator, best_discriminator, test_loader


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
            # print("last metric")
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
    exp_d = ['SiameseCL_config_daft_best_secondtry']

    corresponding_net = ["siameseCL"]

    experiments_dir = "/home/ecarvajal /Desktop/DAFT_branch_distance/DAFT/experiments_clf"

    predictions_dict = {}

    for exp_name, net in zip(exp_d, corresponding_net):

        if not os.path.exists(os.path.join(experiments_dir, exp_name)):
            print(" the path ", exp_name, " doesnt exist!")
            return 1
        else:
            print("experiment ", exp_name)
        print("start experiment metrics: ", exp_name)
        _, folders, _ = next(os.walk(os.path.join(experiments_dir, exp_name)))
        folders.sort()

        num_classes = 2

        for num, folder in enumerate(folders):
            train_file = "mix_balanced_2g_train_fold" + str(num + 1) + ".h5"
            val_file = "mix_balanced_2g_val_fold" + str(num + 1) + ".h5"
            test_file = "mix_balanced_2g_test_fold" + str(num + 1) + ".h5"

            factory, saliency, model, test_loader = load_model(
                os.path.join(experiments_dir, exp_name, folder, "checkpoints"),
                "/home/ecarvajal /Desktop/DAFT_branch_distance/DAFT/data_dir",
                "it doesnt affect",
                net,
                num_classes,
                train_file,
                val_file,
                test_file,
                str(2),
                str(3)
            )

            metrics_fold, preds = evaluate_model(
                metrics=factory.get_test_metrics(), model=model, data=test_loader, progressbar=True,
            )

            targets = preds['target']
            predictions_dict["fold" + str(num + 1)] = preds['logits']

            print("fold ", num)
            print("metrics ", metrics_fold)

        probs_dict = {}

        for k, v in predictions_dict.items():
            probs_dict[k] = torch.sigmoid(v).detach().numpy()

        new_probs = np.expand_dims(np.mean([probs_dict["fold1"], probs_dict["fold2"], probs_dict["fold3"]],
                                           axis=0),
                                   axis=1)

        # new balanced accuracy embedded
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

        classes = Constants.DIAGNOSIS_CODES_BINARY.keys()

        # when checking the create.hd5 just count and compare
        df_cm = pd.DataFrame(matrix,
                             index=[i for i in classes],
                             columns=[i for i in classes])
        plt.figure(figsize=(12, 7))
        fig_conf_matrix = sn.heatmap(df_cm, annot=True).get_figure()
        fig_conf_matrix.savefig(os.path.join(experiments_dir,
                                             exp_name,
                                             "acum_confmatrix_" + exp_name + ".png"))

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

        ########################################

        # ensemble accuracy
        correct_acu = np.zeros(num_classes)
        total_acu = np.zeros(num_classes)

        if new_probs.shape[1] < 2:
            pred2 = torch.zeros([new_probs.shape[0], 2])
            pred2[new_probs[:, 0] > 0, 1] = 1
            pred2[new_probs[:, 0] <= 0, 0] = 1
            new_probs = pred2
        class_id = new_probs.argmax(dim=1)
        correct_acu += (class_id == targets).sum().item()
        total_acu += new_probs.size()[0]

        accuracy = correct_acu / total_acu

        print("metrics experiment: ", exp_name, " balanced_accuracy: ", balanced_accuracy, "accuracy: ", accuracy)
        print("end experiment metrics: ", exp_name)


if __name__ == "__main__":
    main()
