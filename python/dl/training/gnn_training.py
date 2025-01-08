
from dl.training.neural_training import NeuralTraining
from dl.training.hyper_params import HyperParams
from dl.training.early_stop_logger import EarlyStopLogger
from dl.model.gnn_base_model import GNNBaseModel
from dl import GNNException
from plots.plotter import PlotterParameters
from metric.metric import Metric
from dl.training.exec_config import ExecConfig
from typing import Dict, AnyStr, Optional, List
import torch.nn as nn
from torch.utils.data import DataLoader


class GNNTraining(NeuralTraining):
    def __init__(self,
                 hyper_params: HyperParams,
                 early_stop_logger: EarlyStopLogger,
                 metrics: Dict[AnyStr, Metric],
                 exec_config: ExecConfig,
                 plot_parameters: Optional[List[PlotterParameters]] = None):
        """
        Default constructor for this variational auto-encoder
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
        super(GNNTraining, self).__init__(hyper_params,
                                          early_stop_logger,
                                          metrics,
                                          exec_config,
                                          plot_parameters)

    def train(self,
              model_id: AnyStr,
              neural_model: nn.Module,
              data_loader: DataLoader) -> None:

        if not isinstance(neural_model, GNNBaseModel):
            raise GNNException(f'Neural model {type(neural_model)} cannot not be trained as VAE')


    def __train(self, neural_model: nn.Module, epoch: int, train_loader: DataLoader) -> float:
        neural_model.train()
        total_loss = 0

        optimizer = self.hyper_params.optimizer(neural_model)
        num_records = len(train_loader)
        mu, log_var = neural_model.get_mu_log_var()
        model = neural_model.to(self.target_device)

        for data in train_loader:
            try:
                # Add noise if requested
                # data = self.model.add_noise(data)

                # Forward pass
                input_data = data[0].to(self.target_device)
                reconstructed = model(input_data)

                _input = input_data.view(input_data.size(0), input_data.size(1), -1)
                _reconstructed = reconstructed.view(input_data.size(0), input_data.size(1), -1)


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
