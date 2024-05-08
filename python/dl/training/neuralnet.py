__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

import torch
from torch.utils.data import Dataset, DataLoader
from abc import abstractmethod
from typing import AnyStr, Dict
from dl.model.neuralmodel import NeuralModel
from dl.dlexception import DLException
from dl.training.hyperparams import HyperParams
from dl.training.earlystoplogger import EarlyStopLogger
from util.plotter import PlotterParameters
from metric.metric import Metric
from tqdm import tqdm
from typing import List, Optional, NoReturn
import logging
logger = logging.getLogger('dl.NeuralNet')

"""
    Generic Neural Network abstract class. There are 2 version of train and evaluation
    - _train_and_evaluate Training and evaluation from a pre-configure train loader
    -  train_and_evaluate Training and evaluation from a raw data set
    The method transform_label has to be overwritten in the inheriting classes to support
    transformation/conversion of labels if needed.
    The following methods have to be overwritten in derived classes
    - transform_label Transform the label input_tensor if necessary
    - model_label Model identification
"""


class NeuralNet(object):
    def __init__(self,
                 model: NeuralModel,
                 hyper_params: HyperParams,
                 early_stop_logger: EarlyStopLogger,
                 metrics: Dict[AnyStr, Metric],
                 plot_parameters: Optional[List[PlotterParameters]]):
        """
        Constructor for the training and execution of any neural network.
        @param model: Neural network model (CNN, FeedForward,...)
        @type model: NeuralModel or derived types
        @param hyper_params: Hyper parameters associated with the training of th emodel
        @type hyper_params: HyperParams
        @param early_stop_logger: Dynamic condition for early stop in training
        @type early_stop_logger: EarlyStopLogger
        """
        self.hyper_params = hyper_params
        self.model = model
        self.early_stop_logger = early_stop_logger
        self.plot_parameters = plot_parameters
        self.metrics: Dict[AnyStr, Metric] = metrics


    @abstractmethod
    def model_label(self) -> AnyStr:
        raise NotImplementedError('NeuralNet.model_label is an abstract method')

    def __call__(self, train_loader: DataLoader, test_loader: DataLoader) -> NoReturn:
        """
        Train and evaluation of a neural network given a data loader for a training set, a
        data loader for the evaluation/test1 set and a encoder_model. The weights of the various linear modules
        (neural_blocks) will be initialized if self.hyper_params using a Normal distribution

        @param train_loader: Data loader for the training set
        @type train_loader: DataLoader
        @param test_loader:  Data loader for the valuation set
        @type test_loader: DataLoader
        """
        torch.manual_seed(42)
        self.hyper_params.initialize_weight(list(self.model.modules()))

        # Train and evaluation process
        for epoch in range(self.hyper_params.epochs):
            # Set training mode and execute training
            train_loss = self.__train(epoch, train_loader)
            # constants.log_info(f'Epoch # {epoch} training loss {train_loss}')
            # Set mode and execute evaluation
            eval_metrics = self.__eval(epoch, test_loader)
            # constants.log_info(f'Epoch # {epoch} eval loss {eval_loss}')
            self.early_stop_logger(epoch, train_loss, eval_metrics)
        # Generate summary
        if self.plot_parameters is not None:
            self.early_stop_logger.summary()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            try:
                return self.model(features)
            except RuntimeError as e:
                raise DLException(str(e))
            except AttributeError as e:
                raise DLException(str(e))
            except Exception as e:
                raise DLException(str(e))

    def init_data_loader(self, batch_size: int, dataset: Dataset) -> (DataLoader, DataLoader):
        torch.manual_seed(42)

        _len = len(dataset)
        train_len = int(_len * self.hyper_params.train_eval_ratio)
        test_len = _len - train_len
        train_set, test_set = torch.utils.data.random_split(dataset, [train_len, test_len])
        logger.info(f'Extract {len(train_set)} training and {len(test_set)} test data')

        # Finally initialize the training and test1 loader
        train_loader = DataLoader(dataset=train_set,
                                  batch_size=batch_size,
                                  shuffle=True)
        test_loader = DataLoader(dataset=test_set,
                                 batch_size=batch_size,
                                 shuffle=True)
        return train_loader, test_loader

    """ ------------------------------------   Private methods --------------------------------- """

    def __train(self,epoch: int,train_loader: DataLoader) -> float:
        total_loss = 0.0
        # Initialize the gradient for the optimizer
        loss_function = self.hyper_params.loss_function
        optimizer = self.hyper_params.optimizer(self.model)

        for features, labels in tqdm(train_loader):
            try:
                self.model.train()
                # Reset the gradient to zero
                for params in self.model.parameters():
                    params.grad = None

                predicted = self.model(features)  # Call forward - prediction
                labels = labels.unsqueeze(dim=-1)
                raw_loss = loss_function(predicted, labels)
                # Set back propagation
                raw_loss.backward(retain_graph=True)
                total_loss += raw_loss.data
                optimizer.step()
            except RuntimeError as e:
                raise DLException(str(e))
            except AttributeError as e:
                raise DLException(str(e))
            except ValueError as e:
                raise DLException(f'{str(e)}, features: {str(features)}')
            except Exception as e:
                raise DLException(str(e))
        return total_loss / len(train_loader)

    def __eval(self, epoch: int, test_loader: DataLoader) -> Dict[AnyStr, float]:
        total_loss = 0
        loss_func = self.hyper_params.loss_function
        metric_collector = {}

        # No need for computing gradient for evaluation (NO back-propagation)
        with torch.no_grad():
            for features, labels in tqdm(test_loader):
                try:
                    self.model.eval()
                    predicted = self.model(features)
                    is_last_epoch = epoch == self.hyper_params.epochs-1
                    for key, metric in self.metrics.items():
                        value = metric(predicted, labels)
                        metric_collector[key] = value

                    labels = labels.unsqueeze(dim=-1)
                    loss = loss_func(predicted, labels)
                    total_loss += loss.data
                except RuntimeError as e:
                    raise DLException(str(e))
                except AttributeError as e:
                    raise DLException(str(e))
                except ValueError as e:
                    raise DLException(str(e))
                except Exception as e:
                    raise DLException(str(e))

        eval_loss = total_loss / len(test_loader)
        metric_collector[Metric.eval_loss_label] = eval_loss
        return metric_collector

    def __repr__(self) -> str:
        return repr(self.hyper_params)