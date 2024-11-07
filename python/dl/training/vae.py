__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from abc import ABC

from dl.training.neural_net import NeuralNet
from dl.training.hyper_params import HyperParams
from dl.training.early_stop_logger import EarlyStopLogger
from plots.plotter import PlotterParameters
from metric.metric import Metric
from dl.model.vae_model import VAEModel
from dl.dl_exception import DLException
from typing import AnyStr, List, Optional, Dict, NoReturn, Self
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import logging
logger = logging.getLogger('dl.VAE')

"""
Light weight implementation of the variational auto-encoder using PyTorch and reusable neural block
The key components are
- Model (VAEModel) composed of an encoder, decoder as inverted encoder and variational neural block
- Hyper parameters for training and tuning
- Early stop logger for early stop and monitoring training and evaluation
- Dictionary of metrics data
- Optional set of plotting parameters
"""


class VAE(NeuralNet, ABC):
    def __init__(self,
                 vae_model: VAEModel,
                 hyper_params: HyperParams,
                 early_stop_logger: EarlyStopLogger,
                 metrics: Dict[AnyStr, Metric],
                 plot_parameters: Optional[List[PlotterParameters]]):
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
        @param plot_parameters: Optional plotting parameters
        @type plot_parameters: List[PlotterParameters]
        """
        super(VAE, self).__init__(vae_model, hyper_params, early_stop_logger, metrics, plot_parameters)

    @classmethod
    def build(cls, vae_model: VAEModel, hyper_params: HyperParams, early_stop_logger: EarlyStopLogger) -> Self:
        """
        Alternative, simplified constructor for this variational auto-encoder for which only the training
        and evaluation losses are created
        @param vae_model: Model for the variational auto-encoder
        @type vae_model: VAEModel
        @param hyper_params:  Hyper-parameters for training and optimizatoin
        @type hyper_params: HyperParams
        @param early_stop_logger: Training monitoring
        @type early_stop_logger: EarlyStopLogger
        @return Instance of VAE
        """
        return cls(vae_model, hyper_params, early_stop_logger, metrics={}, plot_parameters=[])

    def __call__(self,
                 train_loader: DataLoader,
                 eval_loader: DataLoader,
                 output_file_name: Optional[AnyStr] = None) -> NoReturn:
        """
E       Execute the cycle of training and evaluation for the
        @param train_loader: Loader for the training data
        @type train_loader: DataLoader
        @param eval_loader: Loader for the evaluation data
        @type eval_loader: DataLoader
        @param output_file_name Optional file name for the output of metrics
        @type output_file_name: AnyStr
        """
        # Initialization of the weights
        torch.manual_seed(42)
        self.hyper_params.initialize_weight(list(self.model.modules()))

        for epoch in tqdm(range(self.hyper_params.epochs)):
            # Set training mode and execute training
            train_loss = self.__train(epoch, train_loader)
            # constants.log_info(f'Epoch # {epoch} training loss {train_loss}')
            # Set mode and execute evaluation
            eval_metrics = self.__eval(epoch, eval_loader)
            # constants.log_info(f'Epoch # {epoch} eval loss {eval_loss}')
            self.early_stop_logger(epoch, train_loss, eval_metrics)

        # Generate summary
        if self.plot_parameters is not None:
            self.early_stop_logger.summary()

    def _enc_dec_params(self) -> (dict, dict):
        """
            Extract the model parameters for the encoder and decoder
            :returns: pair of encoder and decoder parameters (dictionaries)
        """
        return self.model.encoder_model.parameters(), self.model.decoder_model.parameters()

    """ ---------------------------   Private helper methods  --------------------------------  """

    def __compute_loss(
            self,
            predicted: torch.Tensor,
            x: torch.Tensor,
            mu: torch.Tensor,
            log_var: torch.Tensor,
            num_records: int) -> torch.Tensor:
        """
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
        """
        reconstruction_loss = self.__reconstruction_loss(predicted, x)
        kl_divergence = (-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp())) / num_records
        logger.info(f"Reconstruction loss {reconstruction_loss} KL divergence {kl_divergence}")
        return reconstruction_loss + kl_divergence

    def __reconstruction_loss(self,
                              predicted: torch.Tensor,
                              x: torch.Tensor) -> float:
        from python.dl.dl_exception import DLException

        try:
            # Cross-entropy for reconstruction loss for binary values
            # and MSE for continuous (TF-IDF) variable
            print(f'Input loss {x.shape}, Prediction shape {predicted.shape}')
            return self.hyper_params.loss_function(predicted, x)
        except RuntimeError as e:
            logging.error(f'Runtime error {str(e)}')
            raise DLException(f'Runtime error {str(e)}')
        except ValueError as e:
            logging.error(f'Value error {str(e)}')
            raise DLException(f'Value error {str(e)}')
        except KeyError as e:
            logging.error(f'Key error {str(e)}')
            raise DLException(f'Key error {str(e)}')

    @staticmethod
    def _reshape_output_variation(shapes: list, z: torch.Tensor) -> torch.Tensor:
        assert 2 < len(shapes) < 5, f'Shape {str(shapes)} for variational auto encoder should have at least 3 dimension'
        return z.view(shapes[0], shapes[1], shapes[2], shapes[3]) if len(shapes) == 4 \
            else z.view(shapes[0], shapes[1], shapes[2])

    def __train(self, epoch: int, data_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0

        encoder_optimizer = self.hyper_params.optimizer(self.model)
        decoder_optimizer = self.hyper_params.optimizer(self.model)
        vae_kl_loss = VAEKLLoss(self.model.mu, self.model.log_var, len(data_loader))

        for data in tqdm(data_loader):
            try:
                for params in self.model.parameters():
                    params.grad = None
                z = self.model(data)
                loss = vae_kl_loss(z, data)
                # loss = self.__compute_loss(z, data, self.model.mu, self.model.log_var, len(data_loader))

                loss.backward(retain_graph=True)
                total_loss += loss.data
                encoder_optimizer.step()
                decoder_optimizer.step()
            except RuntimeError as e:
                raise DLException(str(e))
            except AttributeError as e:
                raise DLException(str(e))
            except Exception as e:
                raise DLException(str(e))
        return total_loss / len(data_loader)

    def __eval(self, epoch: int, eval_loader: DataLoader) -> Dict[AnyStr, float]:
        self.model.eval()
        total_loss = 0
        metric_collector = {}

        with torch.no_grad():
            for data in tqdm(eval_loader):
                z = self.model(data)
                loss = self.__compute_loss(z, data, self.model.mu, self.model.log_var, len(eval_loader))
                total_loss += loss.data

        eval_loss = total_loss / len(eval_loader)
        metric_collector[Metric.eval_loss_label] = eval_loss
        return metric_collector


from torch.nn.modules.loss import _Loss

class VAEKLLoss(_Loss):
    def __init__(self, mu: torch.Tensor, log_var: torch.Tensor, num_records: int):
        super(VAEKLLoss, self).__init__(size_average=None, reduce=None, reduction='mean')
        self.mu = mu
        self.log_var = log_var
        self.num_records = num_records

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        reconstruction_loss = self.hyper_params.loss_function(input, target)
        kl_loss = (-0.5 * torch.sum(1 + self.log_var - self.mu ** 2 - self.log_var.exp())) / self.num_records
        return reconstruction_loss + kl_loss
