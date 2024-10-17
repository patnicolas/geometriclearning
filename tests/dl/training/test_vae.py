import unittest
from dl.block.ffnnblock import FFNNBlock
from dl.model.ffnnmodel import FFNNModel
from dl.model.vaemodel import VAEModel
from dataset.unlabeledloader import UnlabeledLoader
from dataset.unlabeleddataset import UnlabeledDataset
from dl.training.vae import VAE
from dl.training.hyperparams import HyperParams
from dl.training.earlystoplogger import EarlyStopLogger
from plots.plotter import PlotterParameters
import torch.nn as nn


class VAETest(unittest.TestCase):

    def test_init(self):
        features = ['age', 'sex', 'chest pain type', 'cholesterol', 'fasting blood sugar', 'max heart rate',
                    'exercise angina', 'ST slope']
        hidden_block = FFNNBlock.build('hidden', len(features), 4, nn.ReLU())
        output_block = FFNNBlock.build('latent', 4, 4, nn.ReLU())
        encoder = FFNNModel('encoder', [hidden_block, output_block])
        latent_size = 6
        vae_model = VAEModel('Autoencoder', encoder, latent_size)
        print(vae_model)

    def test_train(self):
        from python.metric.metric import Metric
        from python.metric.builtinmetric import BuiltInMetric, MetricType

        features = ['age', 'sex', 'chest pain type', 'cholesterol', 'fasting blood sugar', 'max heart rate',
                    'exercise angina', 'ST slope']
        hidden_block = FFNNBlock.build('hidden', len(features), 4, nn.ReLU())
        output_block = FFNNBlock.build('latent', 4, 4, nn.ReLU())
        encoder = FFNNModel('encoder', [hidden_block, output_block])
        latent_size = 4
        vae_model = VAEModel('Autoencoder', encoder, latent_size)
        print(vae_model)

        filename = '/users/patricknicolas/dev/geometriclearning/data/heart_diseases.csv'
        df = UnlabeledDataset.data_frame(filename)
        features_df = df[features]
        batch_size = 4
        train_eval_split_ratio = 0.85
        dataset_loader = UnlabeledLoader(batch_size, train_eval_split_ratio)
        train_loader, eval_loader = dataset_loader.from_dataframe(features_df)

        hyper_parameters = HyperParams(
            lr=0.001,
            momentum=0.95,
            epochs=8,
            optim_label='adam',
            batch_size=8,
            loss_function=nn.MSELoss(),
            drop_out=0.0,
            train_eval_ratio=0.9)

        patience = 2
        min_diff_loss = -0.001
        early_stopping_enabled = True
        early_stop_logger = EarlyStopLogger(patience, min_diff_loss, early_stopping_enabled)
        metric_labels = {
            Metric.accuracy_label: BuiltInMetric(MetricType.Accuracy, True)
        }
        plot_parameters = [PlotterParameters(0, x_label='x', y_label='y', title=label, fig_size=(11, 7))
                           for label, _ in metric_labels.items()]

        network = VAE(
            vae_model,
            hyper_parameters,
            early_stop_logger,
            metric_labels,
            plot_parameters)
        network(train_loader, eval_loader)


if __name__ == '__main__':
    unittest.main()
