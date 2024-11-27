__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from abc import ABC

from dl.training.neural_net_training import NeuralNetTraining
from dl.training.hyper_params import HyperParams
from dl.training.early_stop_logger import EarlyStopLogger
from plots.plotter import PlotterParameters
from metric.metric import Metric
from dl.model.vae_model import VAEModel
from dl.training.exec_config import ExecConfig
from dl import DLException, VAEException
from dl.loss.vae_kl_loss import VAEKLLoss
from typing import AnyStr, List, Optional, Dict, NoReturn, Self, Tuple
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import logging
logger = logging.getLogger('dl.VAE')

EvaluatedImages = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

"""
Light weight implementation of the variational auto-encoder using PyTorch and reusable neural block
The key components are
- Model (VAEModel) composed of an encoder, decoder as inverted encoder and variational neural block
- Hyper parameters for training and tuning
- Early stop logger for early stop and monitoring training and evaluation
- Dictionary of metrics data
- Optional set of plotting parameters
"""


class VAETraining(NeuralNetTraining, ABC):
    max_debug_images = 3
    def __init__(self,
                 vae_model: VAEModel,
                 hyper_params: HyperParams,
                 early_stop_logger: EarlyStopLogger,
                 metrics: Dict[AnyStr, Metric],
                 exec_config: ExecConfig,
                 plot_parameters: Optional[List[PlotterParameters]] = None):
        """
        Default constructor for this variational auto-encoder
        @param vae_model: Model for the variational auto-encoder
        @type vae_model: VAEModel
        @param hyper_params:  Hyper-parameters for training and optimizatoin
        @type hyper_params: HyperParams
        @param early_stop_logger: Training monitoring
        @type early_stop_logger: EarlyStopLogger
        @param metrics: Dictionary of metrics and values
        @type metrics: Dictionary
        @param exec_config: Configuration for optimization of execution of training
        @type exec_config: ExecConfig
        @param plot_parameters: Optional plotting parameters
        @type plot_parameters: List[PlotterParameters]
        """
        super(VAETraining, self).__init__(vae_model,
                                          hyper_params,
                                          early_stop_logger,
                                          metrics,
                                          exec_config,
                                          plot_parameters)

    @classmethod
    def build(cls,
              vae_model: VAEModel,
              hyper_params: HyperParams,
              early_stop_logger: EarlyStopLogger,
              exec_config: ExecConfig) -> Self:
        """
        Alternative, simplified constructor for this variational auto-encoder for which only the training
        and evaluation losses are created
        @param vae_model: Model for the variational auto-encoder
        @type vae_model: VAEModel
        @param hyper_params:  Hyper-parameters for training and optimizatoin
        @type hyper_params: HyperParams
        @param early_stop_logger: Training monitoring
        @type early_stop_logger: EarlyStopLogger
        @param exec_config: Configuration for optimization of execution of training
        @type exec_config: ExecConfig
        @return Instance of VAE
        """
        return cls(vae_model, hyper_params, early_stop_logger, metrics={}, exec_config=exec_config, plot_parameters=[])

    def __call__(self, plot_title: AnyStr, loaders: (DataLoader, DataLoader)) -> NoReturn:
        """
        Polymorphic execution of the cycle of training and evaluation for the variational auto-encoder
        @param plot_title: Labeling metric for output to file and plots
        @type plot_title: str
        @param loaders: Pair of loader for training data and test data
        @type loaders: Tuple[DataLoader]
        """
        # Initialization of the weights
        torch.manual_seed(42)
        self.hyper_params.initialize_weight(list(self.model.modules()))

        train_loader, eval_loader = loaders
        for epoch in tqdm(range(self.hyper_params.epochs)):
            # Set training mode and execute training
            train_loss = self.__train(epoch, train_loader)

            # Set mode and execute evaluation
            eval_metrics = self.__eval(epoch, eval_loader)
            self.early_stop_logger(epoch, train_loss, eval_metrics)

        # Generate summary
        self.early_stop_logger.summary()

    def _enc_dec_params(self) -> (dict, dict):
        """
        Extract the model parameters for the encoder and decoder
        @returns: pair of encoder and decoder parameters (dictionaries)
        """
        return self.model.encoder_model.parameters(), self.model.decoder_model.parameters()

    """ ---------------------------   Private helper methods  --------------------------------  """
    """
    def __compute_loss(
            self,
            predicted: torch.Tensor,
            x: torch.Tensor,
            mu: torch.Tensor,
            log_var: torch.Tensor,
            num_records: int) -> torch.Tensor:
        
            Aggregate the loss of reconstruction and KL divergence between proposed and current Normal distribution
            @param predicted: Predicted values
            @type predicted: Torch tensor
            @param x: target values
            @type x: Torch tensor
            @param mu: Mean of the proposed Normal distribution
            @type mu:Torch tensor
            @param log_var: Log of variance of the proposed Gaussian distribution
            @type log_var: Torch tensor
            @param num_records: Number of records used to compute the reconstruction loss and KL divergence
            @type int
            @return: Aggregate auto-encoder loss
            @rtype: torch.Tensor
        
        reconstruction_loss = self.__reconstruction_loss(predicted, x)
        kl_divergence = (-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp())) / num_records
        logger.info(f"Reconstruction loss {reconstruction_loss} KL divergence {kl_divergence}")
        return reconstruction_loss + kl_divergence
    """

    def __reconstruction_loss(self, predicted: torch.Tensor, x: torch.Tensor) -> float:
        from dl import DLException

        try:
            # Cross-entropy for reconstruction loss for binary values
            # and MSE for continuous (TF-IDF) variable
            print(f'Input loss {x.shape}, Prediction shape {predicted.shape}')
            return self.hyper_params.loss_function(predicted, x)
        except RuntimeError as e:
            logger.error(f'Runtime error {str(e)}')
            raise VAEException(f'Runtime error {str(e)}')
        except ValueError as e:
            logger.error(f'Value error {str(e)}')
            raise VAEException(f'Value error {str(e)}')
        except KeyError as e:
            logger.error(f'Key error {str(e)}')
            raise VAEException(f'Key error {str(e)}')

    @staticmethod
    def _reshape_output_variation(shapes: list, z: torch.Tensor) -> torch.Tensor:
        assert 2 < len(shapes) < 5, f'Shape {str(shapes)} for variational auto encoder should have at least 3 dimension'
        return z.view(shapes[0], shapes[1], shapes[2], shapes[3]) if len(shapes) == 4 \
            else z.view(shapes[0], shapes[1], shapes[2])

    def __train(self, epoch: int, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0

        encoder_optimizer = self.hyper_params.optimizer(self.model)
        decoder_optimizer = self.hyper_params.optimizer(self.model)
        vae_kl_loss = VAEKLLoss(self.model.mu, self.model.log_var, len(train_loader))

        for data in tqdm(train_loader):
            try:
                for params in self.model.parameters():
                    params.grad = None
                # Add noise if requested
                data = self.model.add_noise(data)

                # Forward pass
                z = self.model(data)
                loss = vae_kl_loss(z, data)

                loss.backward(retain_graph=True)
                total_loss += loss.data
                encoder_optimizer.step()
                decoder_optimizer.step()
            except RuntimeError as e:
                raise VAEException(str(e))
            except AttributeError as e:
                raise VAEException(str(e))
            except Exception as e:
                raise VAEException(str(e))
        return total_loss / len(train_loader)

    def __eval(self, epoch: int, eval_loader: DataLoader) -> Dict[AnyStr, float]:
        self.model.eval()
        total_loss = 0
        vae_kl_loss = VAEKLLoss(self.model.mu, self.model.log_var, len(eval_loader))
        metric_collector = {}
        eval_images: List[EvaluatedImages] = []

        with torch.no_grad():
            images_cnt = 0
            try:
                for data in eval_loader:
                    noisy_data = self.model.add_noise(data)
                    reconstructed = self.model(noisy_data)
                    loss = vae_kl_loss(reconstructed, noisy_data)
                    total_loss += loss.data

                    if self.__are_images(images_cnt):
                        eval_images.append((data, noisy_data, reconstructed))
            except RuntimeError as e:
                raise VAEException(str(e))
            except AttributeError as e:
                raise VAEException(str(e))

        eval_loss = total_loss / len(eval_loader)
        metric_collector[Metric.eval_loss_label] = eval_loss
        self.__visualize(eval_images)
        return metric_collector

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



