import importlib



import os
import torch
from pathlib import Path
from torch import Tensor
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple
from daft.cli import HeterogeneousModelFactory, create_parser
from daft.testing.test_and_save import ModelTester
from daft.training.metrics import Metric

def get_experiment(data_dir: Path) -> Dict[str, Any]:
    cfg = {
        "epoch": "200",
        "batchsize": "8",
        "optimizer": "AdamW",
        "workers": "2",
        "train_data": os.path.join(data_dir, "manual2_2g_train_fold1.h5"),
        "val_data":  os.path.join(data_dir, "manual2_2g_test_fold1.h5"),
        "test_data":  os.path.join(data_dir, "manual2_2g_val_fold1.h5"),
        "discriminator_net": "daft",
        "activation": "tanh",
        "learning_rate": "0.00013",
        "decay_rate": "0.001",
        "experiment_name": "mask_flair_t1_folds_manual_good_nodropout",
        "num_classes": "4",
        "input_channels": "3",
        "n_basefilters": "4",
        "bottleneck_factor": "7",
        "normalize_image": "minmax",
        "dataset": "longitudinal",
        "task": "clf",

    }
    cmd = []
    for k, v in cfg.items():
        cmd.append(f"--{k}")
        cmd.append(v)

    return cmd

def load_model(
    checkpoints_dir: Path
) -> torch.nn.Module:

    args = get_experiment("/home/ecarvajal /Desktop/MyCloneDAFT/DAFT/data_dir")
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
    for m in metrics:
        m.reset()
        m.update(inputs=unconsumed_inputs, outputs=predictions)
        metrics_dict.update(m.values())

    predictions.update(unconsumed_inputs)
    return metrics_dict, predictions


def main():
    # fold 1 no augmentation
    '''
    factory, model, test_loader = load_model(
        "/home/ecarvajal /Desktop/MyCloneDAFT/DAFT/experiments_clf/mask_flair_t1_folds_manual2/2023-04-28_15-24/checkpoints")
    metrics, preds = evaluate_model(
        metrics=factory.get_test_metrics(), model=model, data=test_loader, progressbar=True,
    )
    '''
    # fold 2 no augmentation
    factory, model, test_loader = load_model(
        "/home/ecarvajal /Desktop/MyCloneDAFT/DAFT/experiments_clf/mask_flair_t1_folds_manual_good_nodropout/2023-05-04_10-41/checkpoints")
    metrics, preds = evaluate_model(
        metrics=factory.get_test_metrics(), model=model, data=test_loader, progressbar=True,
    )
    print(metrics)



if __name__ == "__main__":
    main()
