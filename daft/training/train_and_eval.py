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
from itertools import chain
from operator import itemgetter
from typing import Dict, Optional, Sequence, Union

import torch
from torch import Tensor
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from tqdm import tqdm

from ..models.base import BaseModel, check_is_unique
from .hooks import Hook
from .wrappers import DataLoaderWrapper


def train_and_evaluate(
        model: BaseModel,
        loss: BaseModel,
        train_data: DataLoaderWrapper,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        num_epochs: int = 1,
        eval_data: Optional[DataLoaderWrapper] = None,
        train_hooks: Optional[Sequence[Hook]] = None,
        eval_hooks: Optional[Sequence[Hook]] = None,
        device: Optional[torch.device] = None,
        progressbar: bool = True,
) -> None:
    """Train and evaluate a model.

    Evaluation is run after every epoch.

    Args:
      model (BaseModel):
        Instance of model to call.
      loss (BaseModel):
        Instance of loss to compute.
      train_data (DataLoaderWrapper):
        Instance of DataLoader to obtain batches from for training.
        Keys of `data.output_names` must be contained in keys of `model.input_names`
        and `loss.input_names`.
      optimizer (Optimizer):
        Instance of Optimizer to use.
      scheduler (_LRScheduler):
        Optional; Scheduler to adjust the learning rate.
      num_epochs (int):
        Optional; For how many epochs to train.
      eval_data (DataLoaderWrapper):
        Optional; Instance of DataLoader to obtain batches from for evaluation.
        Keys of `data.output_names` must be contained in keys of `model.input_names`
        and `loss.input_names`.
      train_hooks (list of Hook):
        Optional; List of hooks to call during training.
      eval_hooks (list of Hook):
        Optional; List of hooks to call during evaluation.
      device (torch.device):
        Optional; Which device to run on.
      progressbar (bool):
        Optional; Whether to display a progess bar.
    """
    train_hooks = train_hooks or []
    train_hooks = list(train_hooks)

    evaluator = None
    if eval_data is not None:
        evaluator = ModelEvaluator(
            model=model, loss=loss, data=eval_data, device=device, hooks=eval_hooks, progressbar=progressbar,
        )

    trainer = ModelTrainer(
        model=model,
        loss=loss,
        data=train_data,
        optimizer=optimizer,
        device=device,
        hooks=train_hooks,
        progressbar=progressbar,
    )
    get_lr = itemgetter("lr")

    early_stopping = EarlyStopping(tolerance=10, min_delta=10)

    for i in range(num_epochs):
        lr = [get_lr(pg) for pg in optimizer.param_groups]
        print("EPOCH: {:>3d};    Learning rate: {}".format(i, lr))
        trainer.run()
        if evaluator is not None:
            evaluator.run()
        if scheduler is not None:
            #scheduler.step(evaluator.outputs['cross_entropy'])
            scheduler.step()

        # early stopping
        early_stopping(trainer.outputs["cross_entropy"], evaluator.outputs["cross_entropy"])
        if early_stopping.early_stop:
            print("we apply early stopping at epoch : ", i)
            break

class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True


class ModelRunner:
    """Base class for calling a model on every batch of data.

    Args:
      model (BaseModel):
        Instance of model to call.
      data (DataLoaderWrapper):
        Instance of DataLoader to obtain batches from.
        Keys of `data.output_names` must be contained in keys of `model.input_names`.
      device (torch.device):
        Optional; Which device to run on.
      hooks (list of Hook):
        Optional; List of hooks to call during execution.
      progressbar (bool):
        Optional; Whether to display a progess bar.
    """

    def __init__(
            self,
            model: BaseModel,
            data: DataLoaderWrapper,
            device: Optional[torch.device] = None,
            hooks: Optional[Sequence[Hook]] = None,
            progressbar: bool = True,
    ) -> None:
        if device is not None:
            model = model.to(device)

        self.model = model
        self.data = data
        self.device = device
        self.hooks = hooks or []
        self.progressbar = progressbar
        self.outputs = None

    def _dispatch(self, func: str, *args) -> None:
        with torch.no_grad():
            for h in self.hooks:
                fn = getattr(h, func)
                fn(*args)

    def _batch_to_device(self, batch: Union[Tensor, Sequence[Tensor]]) -> Dict[str, Tensor]:
        if not isinstance(batch, (list, tuple)):
            batch = tuple(batch, )

        assert len(self.data.output_names) == len(batch), "output_names suggests {:d} tensors, but only found {:d} " \
                                                          "outputs".format(len(self.data.output_names), len(batch))
        import numpy as np
        batch = dict(zip(self.data.output_names, batch))
        #print(" _batch_to_device batch keys ", batch.keys())
        if self.device is not None:
            for k, v in batch.items():
                #print(" _batch_to_device key ", k)
                #print(" _batch_to_device v 0", np.array(v, dtype=object)[0].shape)  # (2,) 8 images ( when batch 8)
                #print(" _batch_to_device v 1", np.array(v, dtype=object)[1].shape)  # 8 tabular

                if k in ["image", "tabular"]:
                    batch[k] = [v[0].to(self.device), v[1].to(self.device)]
                else:
                    batch[k] = v.to(self.device)

        return batch

    def _set_model_state(self) -> None:
        pass

    def run(self) -> None:
        """Execute model for every batch."""
        self._set_model_state()
        self._dispatch("on_begin_epoch")
        #print(" train and eval data len ", len(self.data))
        pbar = tqdm(self.data, total=len(self.data), disable=not self.progressbar)
        import numpy as np
        #print("train and eval data ", np.array(self.data, dtype=object).shape)
        for batch in pbar:
            #print("train and eval run batch shape ", np.array(batch, dtype=object).shape)
            batch = self._batch_to_device(batch)
            self._dispatch("before_step", batch)
            self.outputs = self._step(batch)  # to track outputs for schedulers that follow the metric
            self._dispatch("after_step", self.outputs)

        self._dispatch("on_end_epoch")

    def _get_model_inputs_from_batch(self, batch: Dict[str, Tensor]) -> Sequence[Tensor]:
        # to get rid of unnecessary dimensions when adding multiple channels
        import numpy as np
        #print("train and eval _get_model_inputs_from_batch ", np.array(batch["image"], dtype=object).shape)
        ####################################################con
        #print("train and eval _get_model_inputs_from_batch ", batch["image"][0].shape)
        batch["image"][0] = batch["image"][0].squeeze(1)
        batch["image"][1] = batch["image"][1].squeeze(1)
        assert len(batch) >= len(self.model.input_names), "model expects {:d} inputs, but batch has only {:d}".format(
            len(self.model.input_names), len(batch)
        )
        in_batch = tuple([batch[k] for k in self.model.input_names])
        return in_batch

    def _step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        in_batch = self._get_model_inputs_from_batch(batch)
        out_tensors = self.model(*in_batch)
        assert len(out_tensors) == len(
            self.model.output_names
        ), "output_names suggests {:d} tensors, but only found {:d} outputs".format(
            len(self.model.output_names), len(out_tensors),
        )

        return out_tensors


class ModelEvaluator(ModelRunner):
    """Execute a model on every batch of data in evaluation mode.

    Args:
      model (BaseModel):
        Instance of model to call.
      loss (BaseModel):
        Instance of loss to compute.
      data (DataLoaderWrapper):
        Instance of DataLoader to obtain batches from.
        Keys of `data.output_names` must be contained in keys of `model.input_names`
        and `loss.input_names`.
      device (torch.device):
        Optional; Which device to run on.
      hooks (list of Hook):
        Optional; List of hooks to call during execution.
      progressbar (bool):
        Optional; Whether to display a progess bar.
    """

    def __init__(
            self,
            model: BaseModel,
            loss: BaseModel,
            data: DataLoaderWrapper,
            device: Optional[torch.device] = None,
            hooks: Optional[Sequence[Hook]] = None,
            progressbar: bool = True,
    ) -> None:
        super().__init__(
            model=model, data=data, device=device, hooks=hooks, progressbar=progressbar,
        )
        all_names = list(chain(model.input_names, model.output_names, loss.output_names))
        check_is_unique(all_names)

        if "total_loss" in all_names:
            raise ValueError("total_loss cannot be used as input or output name")

        model_loss_intersect = set(model.output_names).intersection(set(loss.input_names))
        if len(model_loss_intersect) == 0:
            raise ValueError("model outputs and loss inputs do not agree")

        model_data_intersect = set(model.input_names).intersection(set(data.output_names))
        if len(model_data_intersect) == 0:
            raise ValueError("model inputs and data loader outputs do not agree")

        self.loss = loss

    def _set_model_state(self):
        self.model = self.model.eval()

    def _step_with_loss(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        outputs = super()._step(batch)

        batch.update(outputs)
        loss_inputs = [batch[key] for key in self.loss.input_names]

        losses = self.loss(*loss_inputs)
        outputs["total_loss"] = sum(losses.values())
        outputs.update(losses)

        return outputs

    def _step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        with torch.no_grad():
            return self._step_with_loss(batch)


class ModelTrainer(ModelEvaluator):
    """Execute a model on every batch of data in train mode, compute the gradients, and update the weights.

    Args:
      model (BaseModel):
        Instance of model to call.
      loss (BaseModel):
        Instance of loss to compute.
      data (DataLoaderWrapper):
        Instance of DataLoader to obtain batches from.
        Keys of `data.output_names` must be contained in keys of `model.input_names`
        and `loss.input_names`.
      optimizer (Optimizer):
        Instance of Optimizer to use.
      device (torch.device):
        Optional; Which device to run on.
      hooks (list of Hook):
        Optional; List of hooks to call during execution.
      progressbar (bool):
        Optional; Whether to display a progess bar.
    """

    def __init__(
            self,
            model: BaseModel,
            loss: BaseModel,
            data: DataLoaderWrapper,
            optimizer: Optimizer,
            device: Optional[torch.device] = None,
            hooks: Optional[Sequence[Hook]] = None,
            progressbar: bool = True,
    ) -> None:
        super().__init__(
            model=model, loss=loss, data=data, device=device, hooks=hooks, progressbar=progressbar,
        )

        self.optimizer = optimizer

    def _set_model_state(self):
        self.model = self.model.train()

    def _step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        self.optimizer.zero_grad()
        outputs = super()._step_with_loss(batch)

        total_loss = outputs["total_loss"]

        total_loss.backward()
        self.optimizer.step()

        return outputs
