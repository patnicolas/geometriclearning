__author__ = "Patrick R. Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC

import torch.nn as nn
from deeplearning.training.neural_training import NeuralTraining
from deeplearning.training.hyper_params import HyperParams
from deeplearning.training.early_stopping import EarlyStopping
from plots.plotter import PlotterParameters
from metric.built_in_metric import BuiltInMetric
from metric.metric_type import MetricType
from deeplearning.training.exec_config import ExecConfig
from deeplearning import ConvException, VAEException
from deeplearning.loss.vae_kl_loss import VAEKLLoss
from typing import AnyStr, List, Optional, Dict, Self, Tuple
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
__all__ = ['VAETraining']

EvaluatedImages = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class VAETraining(NeuralTraining, ABC):
    """
    Lightweight implementation of the variational auto-encoder using PyTorch and reusable neural block
    The key components are
    - Model (VAEModel) composed of an encoder, decoder as inverted encoder and variational neural block
    - Hyper parameters for training and tuning
    - Early stop logger for early stop and monitoring training and evaluation
    - Dictionary of metrics data
    - Optional set of plotting parameters
    """
    max_debug_images = 3

    def __init__(self,
                 hyper_params: HyperParams,
                 metrics_attributes: Dict[AnyStr, BuiltInMetric],
                 early_stopping: Optional[EarlyStopping] = None,
                 exec_config: ExecConfig = ExecConfig.default(),
                 plot_parameters: Optional[List[PlotterParameters]] = None):
        """
        Default constructor for this variational auto-encoder

        @param hyper_params:  Hyper-parameters for training and optimizatoin
        @type hyper_params: HyperParams
        @param metrics_attributes: Dictionary of metrics and values
        @type metrics_attributes: Dictionary
        @param exec_config: Configuration for optimization of execution of training
        @type exec_config: ExecConfig
        @param plot_parameters: Optional plotting parameters
        @type plot_parameters: List[PlotterParameters]
        """
        super(VAETraining, self).__init__(hyper_params,
                                          metrics_attributes,
                                          early_stopping,
                                          exec_config,
                                          plot_parameters)

    @classmethod
    def build(cls,
              hyper_params: HyperParams,
              metrics_attributes: Dict[AnyStr, BuiltInMetric]) -> Self:
        """
        Simplified constructor for the training and execution of any neural network.
        @param hyper_params: Hyperparameters associated with the training of th emodel
        @type hyper_params: HyperParams
        @param metrics_attributes: Dictionary of metrics
        @type metrics_attributes: Dict[AnyStr, BuiltInMetric]
        @return Instance of VAETraining
        @rtype VAETraining
        """
        # Create metrics
        # Initialize the plotting parameters
        plot_parameters = [PlotterParameters(0, x_label='x', y_label='y', title=label, fig_size=(11, 7))
                           for label, _ in metrics_attributes.items()]
        return cls(hyper_params=hyper_params,
                   metrics_attributes=metrics_attributes,
                   exec_config=ExecConfig.default(),
                   plot_parameters=plot_parameters)

    def train(self,
              model_id: AnyStr,
              neural_model: nn.Module,
              train_loader: DataLoader,
              eval_loader: DataLoader) -> None:
        """
        Train and evaluation of a neural network given a data loader for a training set, a
        data loader for the evaluation/test1 set and a encoder_model. The weights of the various linear modules
        (neural_blocks) will be initialized if self.hyper_params using a Normal distribution

        @param model_id: Identifier for the model
        @type model_id: str
        @param neural_model: Neural model as torch module
        @type neural_model: nn_Module
        @param train_loader: Data loader for the training set
        @type train_loader: DataLoader
        @param eval_loader:  Data loader for the valuation set
        @type eval_loader: DataLoader
        """
        from deeplearning.model.generative.vae_model import VAEModel

        if not isinstance(neural_model, VAEModel):
            raise VAEException(f'Neural model {type(neural_model)} cannot not be trained as VAE')

        # Initialization of the weights
        torch.manual_seed(42)
        self.hyper_params.initialize_weight(neural_model.get_modules())

        for epoch in tqdm(range(self.hyper_params.epochs)):
            # Set training mode and execute training
            self.__train_epoch(neural_model, epoch, train_loader)

            # Set mode and execute evaluation
            self.__val_epoch(neural_model, epoch, eval_loader)
        # Generate summary and save into file
        self.performance_metrics.summary(f'{model_id}_metrics')

    @staticmethod
    def _reshape_output_variation(shapes: list, z: torch.Tensor) -> torch.Tensor:
        assert 2 < len(shapes) < 5, f'Shape {str(shapes)} for variational auto encoder should have at least 3 dimension'
        return z.view(shapes[0], shapes[1], shapes[2], shapes[3]) if len(shapes) == 4 \
            else z.view(shapes[0], shapes[1], shapes[2])

    """ -----------------------  Private class and object methods -------------------- """

    def __train_epoch(self, neural_model: nn.Module, epoch: int, train_loader: DataLoader) -> None:
        neural_model.train()
        total_loss = 0

        encoder_optimizer = self.hyper_params.optimizer(neural_model)
        decoder_optimizer = self.hyper_params.optimizer(neural_model)
        num_records = len(train_loader)
        mu, log_var = neural_model.get_mu_log_var()
        model = neural_model.to(device=self.target_device, non_blocking=True)
        vae_kl_loss = VAEKLLoss(mu=mu,
                                log_var=log_var,
                                num_records=num_records,
                                loss_func=self.hyper_params.loss_function)

        for data in train_loader:
            try:
                # Add noise if requested
                # data = self.model.add_noise(data)

                # Forward pass
                input_data = data[0].to(device=self.target_device, non_blocking=True)
                reconstructed = model(input_data)

                _input = input_data.view(input_data.size(0), input_data.size(1), -1)
                _reconstructed = reconstructed.view(input_data.size(0), input_data.size(1), -1)
                loss = vae_kl_loss(_reconstructed, model.z, _input)

                if loss is torch.nan:
                    raise VAEException(f'Train loss: {_reconstructed}, z: {model.z} output {_input} is NAN')
                loss.backward(retain_graph=True)
                total_loss += loss.item()

                encoder_optimizer.step()
                decoder_optimizer.step()
            except ConvException as e:
                raise VAEException(str(e))
            except RuntimeError as e:
                raise VAEException(str(e))
            except ValueError as e:
                raise VAEException(str(e))
            except KeyError as e:
                raise VAEException(str(e))
            except AttributeError as e:
                raise VAEException(str(e))

        return total_loss / num_records

    def __val_epoch(self, neural_model: nn.Module, epoch: int, eval_loader: DataLoader) -> None:
        neural_model.eval()
        total_loss = 0
        mu, log_var = neural_model.get_mu_log_var()
        model = neural_model.to(self.target_device)
        num_records = len(eval_loader)
        vae_kl_loss = VAEKLLoss(mu=mu,
                                log_var=log_var,
                                num_records=num_records,
                                loss_func=self.hyper_params.loss_function)

        eval_images: List[EvaluatedImages] = []

        with torch.no_grad():
            images_cnt = 0
            try:
                for data in eval_loader:
                    # noisy_data = neural_model.add_noise(data)

                    input_data = data[0].to(self.target_device, non_blocking=True)
                    reconstructed = model(input_data)

                    _input = input_data.view(input_data.size(0), input_data.size(1), -1)
                    _reconstructed = reconstructed.view(input_data.size(0), input_data.size(1), -1)
                    loss = vae_kl_loss(_reconstructed, model.z, _input)
                    if loss is torch.nan:
                        raise VAEException(f'Eval Loss: {_reconstructed}, z: {model.z}, output {_input} is NAN')
                    total_loss += loss.item()

                    if self.__are_images(images_cnt):
                        eval_images.append((data, data, reconstructed))
            except ConvException as e:
                raise VAEException(str(e))
            except RuntimeError as e:
                raise VAEException(str(e))
            except ValueError as e:
                raise VAEException(str(e))
            except KeyError as e:
                raise VAEException(str(e))
            except AttributeError as e:
                raise VAEException(str(e))

        eval_loss = total_loss / num_records
        self.performance_metrics.update_metric(MetricType.EvalLoss, eval_loss)
        self.__visualize(eval_images)

    """ ------------------------------  Private helper method for visual debugging --------------- """

    def __are_images(self, images_cnt: int) -> bool:
        return self.plot_parameters is not None \
            and self.plot_parameters[0].is_image \
            and images_cnt < VAETraining.max_debug_images

    def __visualize(self, eval_images: List[EvaluatedImages]) -> None:
        if self.plot_parameters is not None and self.plot_parameters[0].is_image:
            import matplotlib.pyplot as plt

            fig_x = len(eval_images)
            fig_y: int = 3 if eval_images[0][1] is not None else 2
            plt.figure(figure_size=(fig_x, fig_y))

            for i, eval_image in enumerate(eval_images):
                idy = i+1+fig_x
                plt.subplot(fig_y, fig_x, idy)
                plt.imshow(eval_image[0].view(28, 28), cmap='gray')
                plt.axis('off')

                if fig_y == 3:
                    idy += fig_x
                    plt.subplot(fig_y, fig_x, idy)
                    plt.imshow(eval_image[0].view(28, 28), cmap='gray')
                    plt.axis('off')

                idy += fig_x
                plt.subplot(fig_y, fig_x, idy)
                plt.imshow(eval_image[0].view(28, 28), cmap='gray')
                plt.axis('off')



