import importlib



import os
import torch
from pathlib import Path
from torch import Tensor
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple
from daft.cli import HeterogeneousModelFactory, create_parser
from daft.testing.test_and_save import ModelTester
from daft.training.metrics import Metric

def load_model(
    checkpoints_dir: Path,
    data_dir: str,
    experiment_name: str,
    net: str,
    classes: int,
    train_data: str,
    val_data: str,
    test_data: str
) -> torch.nn.Module:

    from ablation_adni_classification_try import get_experiment

    args = get_experiment(data_dir,
                          experiment_name,
                          net,
                          classes,
                          train_data,
                          val_data,
                          test_data)

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
    experiments_dir = "/home/ecarvajal /Desktop/MyCloneDAFT/DAFT/experiments_clf"
    factory, model, test_loader = load_model(
        os.path.join(experiments_dir, "10may_4group_augmentation_manualfile/2023-05-10_21-05/checkpoints"),
        "/home/ecarvajal /Desktop/MyCloneDAFT/DAFT/data_dir",
        "it doesnt affect",
        "daft_v2",
        4,
        "manualfile_4g_train_fold2.h5",
        "manualfile_4g_val_fold2.h5",
        "manualfile_4g_test_fold2.h5"
    )
    metrics, preds = evaluate_model(
        metrics=factory.get_test_metrics(), model=model, data=test_loader, progressbar=True,
    )
    print("metrics fold1", metrics )




if __name__ == "__main__":
    main()
