import importlib

import os
import torch
import nibabel
import numpy as np
import nibabel as nib
from pathlib import Path
from torch import Tensor
import torch.nn.functional as F
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
    #model = Saliency(best_discriminator)

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

def get_model_inputs_from_batch(batch: Dict[int, Tensor]) -> Sequence[Tensor]:
    # to get rid of unnecessary dimensions when adding multiple channels
    batch[0] = batch[0].squeeze(0)
    in_batch = tuple([batch[num] for num in [0, 1]])
    return in_batch


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

    experiment_name_daft = [#"11may_2group_NOdropout_NOaugmentation_manualfile",
                            "11may_4group_NOdropout_NOaugmentation_manualfile"]
                            #"11may_3edss_NOdropout_NOaugmentation_manualfile"]

    net = "daft_v2_captum"
    experiments_dir = "/home/ecarvajal /Desktop/MyCloneDAFT/DAFT/experiments_clf"

    for exp_name in experiment_name_daft:

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
                train_file = "manualfile_" + str(num_classes) + "g_train_fold" + str(num + 1) + ".h5"
                val_file = "manualfile_" + str(num_classes) + "g_val_fold" + str(num + 1) + ".h5"
                test_file = "manualfile_" + str(num_classes) + "g_test_fold" + str(num + 1) + ".h5"
            else:
                train_file = "manualfile_" + str(num_classes) + "edss_train_fold" + str(num + 1) + ".h5"
                val_file = "manualfile_" + str(num_classes) + "edss_val_fold" + str(num + 1) + ".h5"
                test_file = "manualfile_" + str(num_classes) + "edss_test_fold" + str(num + 1) + ".h5"

            factory, model, test_loader = load_model(
                os.path.join(experiments_dir, exp_name, folder, "checkpoints"),
                "/home/ecarvajal /Desktop/MyCloneDAFT/DAFT/data_dir",
                "it doesnt affect",
                net,
                num_classes,
                train_file,
                val_file,
                test_file,
                str(1),
                str(3)
            )

            _, model_preds, _ = load_model(
                os.path.join(experiments_dir, exp_name, folder, "checkpoints"),
                "/home/ecarvajal /Desktop/MyCloneDAFT/DAFT/data_dir",
                "it doesnt affect",
                "daft_v2",
                num_classes,
                train_file,
                val_file,
                test_file,
                str(16),
                str(3)
            )

            # to get input image and target check test_loader
            # test_loader has image, tabular and target in every element

            test_loader.required_grads = True
            #print("test_loader visits ", test_loader.dataset.visits)
            #print("test_loader targets ", test_loader.dataset.targets)

            # prediction
            _, preds = evaluate_model(
                metrics=factory.get_test_metrics(), model=model_preds, data=test_loader, progressbar=True,
            )

            targets = preds["target"]
            preds = torch.sigmoid(preds["logits"])


            print(preds)
            for position, batch in enumerate(test_loader):
                element = batch
                in_element = get_model_inputs_from_batch(element)

                image = in_element[0]
                tabular = in_element[1]
                name = test_loader.dataset.visits[position][0]
                visit = test_loader.dataset.visits[position][1]
                label = test_loader.dataset.targets['DX'][position]

                #print("argmax ", preds[position].item())
                print(" name : ", name, " label : ", label, ' pred : ', str(preds[position].detach().numpy().argmax()))

                image.requires_grad = True
                tabular.requires_grad = True

                layer_gradcam = LayerGradCam(model, model.block3)
                print(" check")
                attributes = layer_gradcam.attribute(image, target=1, additional_forward_args=tabular)


                # thresholding the attributions
                th_attributes = np.zeros(attributes.shape)

                fquantile = np.quantile(attributes.detach().numpy().ravel(), 0.96)
                th_attributes[attributes.detach().numpy() >= fquantile] = 1
                th_attributes[attributes.detach().numpy() < fquantile] = 0

                # masking ?

                # normalising 0,1
                #attributes -= torch.min(attributes)
                #attributes /= torch.max(attributes)

                resized_attributes = F.interpolate(torch.Tensor(th_attributes), image.shape[2:], mode='trilinear', align_corners=False)

                print("resized_attributes.shape ", resized_attributes.shape)

                if len(label) == 4:
                    group1, group2 = label[:2], label[2:]
                else:
                    group1, group2 = label[:3], label[3:]

                if os.path.exists(os.path.join("/secure-data",
                                                 "scientific",
                                                 "ms_lesions_data",
                                                 "rochelyon_icobrain_v510",
                                                 "RocheLyon_"+group1,
                                                 name)):
                    group = group1
                else:
                    group = group2

                original = nib.load(os.path.join("/secure-data",
                                                 "scientific",
                                                 "ms_lesions_data",
                                                 "rochelyon_icobrain_v510",
                                                 'RocheLyon_' + group,
                                                 name,
                                                 visit,
                                                 't1_mni.nii.gz'))

                print(" name attributes ", "attribute_" + group + "_" + name + "_pred_" + str(preds[position].detach().numpy().argmax()) + "_" + exp_name + "_fold" + str(num + 1) + ".nii.gz")

                nib.save(nibabel.Nifti1Image(resized_attributes.squeeze().detach().numpy(), original.affine),
                         os.path.join("/home",
                                      "ecarvajal ",
                                      "Desktop",
                                      "MyCloneDAFT",
                                      "DAFT",
                                      'gradcam',
                                      "attribute_" + group + "_" + name + "_pred_" + str(preds[position].detach().numpy().argmax()) + "_" + exp_name + "_fold" + str(num + 1) + ".nii.gz"))

                print("saved!")

if __name__ == "__main__":
    main()
