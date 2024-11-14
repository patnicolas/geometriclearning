__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from typing import AnyStr, NoReturn
from dl.model.vision.base_model import BaseModel
from dl.model.vae_model import VAEModel
from dl.block import ConvException
from dl.training.early_stop_logger import EarlyStopLogger
from dl.training.vae import VAE
from metric.metric import Metric
from plots.plotter import PlotterParameters
from dl.training.hyper_params import HyperParams
from dl.dl_exception import DLException
from metric.built_in_metric import BuiltInMetric, MetricType


class VAEMNIST(object):

    def __init__(self, base_mnist: BaseModel, latent_size: int) -> None:
        self.base_mnist = base_mnist
        self.vae_model = VAEModel('VAE MNIST', base_mnist.model, latent_size)

    def __repr__(self) -> AnyStr:
        return repr(self.vae_model)

    def do_train(self, root_path: AnyStr, hyper_parameters: HyperParams) -> NoReturn:
        """
        Execute the training, evaluation and metrics for any model for MNIST data set
        @param root_path: Path for the root of the MNIST data
        @type root_path: str
        @param hyper_parameters: Hyper-parameters for the execution of the
        @type hyper_parameters: HyperParams
        """
        try:
            # Set up the early stopping
            patience = 2
            min_diff_loss = -0.001
            early_stopping_enabled = True
            early_stop_logger = EarlyStopLogger(patience, min_diff_loss, early_stopping_enabled)

            # Define the neural network as model, hyperparameters, early stopping criteria and metrics
            vae = VAE(vae_model=self.vae_model,
                      hyper_params=hyper_parameters,
                      early_stop_logger=early_stop_logger,
                      metrics={},
                      plot_parameters=[])

            # No need for labels as it is unsupervised
            train_data_loader, test_data_loader = self.base_mnist.load_dataset(root_path, use_labels=False)
            vae(train_data_loader, test_data_loader)
        except ConvException as e:
            print(str(e))
        except DLException as e:
            print(str(e))

