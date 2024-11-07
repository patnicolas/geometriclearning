import unittest

from dl.model.ffnn_model import FFNNModel
from dl.block.ffnn_block import FFNNBlock
from dl.training.hyper_params import HyperParams
from dl.training.early_stop_logger import EarlyStopLogger
from plots.plotter import PlotterParameters
from dl.training.neural_net import NeuralNet
from metric.metric import Metric
from dataset.labeled_dataset import LabeledDataset
from dataset.unlabeled_dataset import UnlabeledDataset
from dataset.labeled_loader import LabeledLoader
from dataset.tdataset import min_max_scaler
from torch import nn
import numpy as np


class NeuralNetTest(unittest.TestCase):

    @unittest.skip('Ignored')
    def test_init(self):
        input_block = FFNNBlock.build('../../../python/input', 32, 16, nn.ReLU())
        hidden_block = FFNNBlock.build('hidden', 16, 4, nn.ReLU())
        output_block = FFNNBlock.build('output', 4, 1)
        binary_classifier = FFNNModel('test1', [input_block, hidden_block, output_block])
        print(repr(binary_classifier))
        hyper_parameters = HyperParams(
            lr=0.001,
            momentum=0.95,
            epochs=12,
            optim_label='adam',
            batch_size=8,
            loss_function=nn.CrossEntropyLoss(),
            drop_out=0.2,
            train_eval_ratio=0.9)
        patience = 2
        min_diff_loss = -0.001
        early_stopping_enabled = True
        early_stop_logger = EarlyStopLogger(patience, min_diff_loss, early_stopping_enabled)
        labels = [Metric.train_loss_label, EarlyStopLogger.eval_loss_label, EarlyStopLogger.accuracy_label]
        parameters = [PlotterParameters(0, x_label='x', y_label='y', title=label, fig_size=(12, 8)) for label in labels]
        network = NeuralNet(binary_classifier, hyper_parameters, early_stop_logger, parameters)
        filename = '/users/patricknicolas/dev/geometriclearning/data/wages_cleaned.csv'
        tensor_dataset = UnlabeledDataset.from_file(filename, ['Reputation', 'Age', 'Caps', 'Apps', 'Salary'])
        network.init_data_loader(batch_size=8, dataset=tensor_dataset)

    @unittest.skip('Ignored')
    def test_train_wages(self):
        from python.metric.metric import Metric
        from python.metric.built_in_metric import BuiltInMetric, MetricType

        hidden_block = FFNNBlock.build('hidden', 5, 4, nn.ReLU())
        output_block = FFNNBlock.build('output', 4, 1, nn.Sigmoid())
        binary_classifier = FFNNModel('test1', [hidden_block, output_block])
        print(repr(binary_classifier))
        hyper_parameters = HyperParams(
            lr=0.001,
            momentum=0.95,
            epochs=8,
            optim_label='adam',
            batch_size=8,
            loss_function=nn.BCELoss(),
            drop_out=0.0,
            train_eval_ratio=0.9)
        patience = 2
        min_diff_loss = -0.002
        early_stopping_enabled = True
        early_stop_logger = EarlyStopLogger(patience, min_diff_loss, early_stopping_enabled)
        metric_labels = {
            Metric.accuracy_label: BuiltInMetric(MetricType.Accuracy, True),
            Metric.precision_label: BuiltInMetric(MetricType.Precision, True)
        }
        parameters = [PlotterParameters(0, x_label='x', y_label='y', title=label, fig_size=(11, 7))
                      for label, _ in metric_labels.items()]
        network = NeuralNet(
            binary_classifier,
            hyper_parameters,
            early_stop_logger,
            metric_labels,
            parameters)
        filename = '/users/patricknicolas/dev/geometriclearning/data/wages_cleaned.csv'
        df = LabeledDataset.data_frame(filename)
        df = df[['Reputation', 'Age', 'Caps', 'Apps', 'Salary']]
        print(df)
        average_salary = df['Salary'].mean()
        df['Top_player'] = np.where(df['Salary'] > average_salary, 1.0, 0.0)
        print(f'Update data frame: {df.columns}')
        batch_size = 2
        train_eval_split_ratio = 0.85
        dataset_loader = LabeledLoader(batch_size, train_eval_split_ratio)
        train_loader, eval_loader = dataset_loader.from_dataframes(
            df[['Reputation', 'Age', 'Caps', 'Apps', 'Salary']],
            df['Top_player'],
            min_max_scaler)
        network(train_loader, eval_loader)

    @unittest.skip('Ignored')
    def test_train_eval_heart_diseases(self):
        from python.metric.metric import Metric
        from python.metric.built_in_metric import BuiltInMetric, MetricType

        features = ['age', 'sex', 'chest pain type', 'cholesterol', 'fasting blood sugar','max heart rate',
                    'exercise angina', 'ST slope']
        hidden_block = FFNNBlock.build('hidden', len(features), 4, nn.ReLU())
        output_block = FFNNBlock.build('output', 4, 1, nn.Sigmoid())
        binary_classifier = FFNNModel('test1', [hidden_block, output_block])
        print(repr(binary_classifier))
        hyper_parameters = HyperParams(
            lr=0.001,
            momentum=0.95,
            epochs=8,
            optim_label='adam',
            batch_size=8,
            loss_function=nn.BCELoss(),
            drop_out=0.0,
            train_eval_ratio=0.9)
        patience = 2
        min_diff_loss = -0.001
        early_stopping_enabled = True
        early_stop_logger = EarlyStopLogger(patience, min_diff_loss, early_stopping_enabled)
        metric_labels = {
            Metric.accuracy_label: BuiltInMetric(MetricType.Accuracy, True),
            Metric.precision_label: BuiltInMetric(MetricType.Precision, True),
            Metric.recall_label: BuiltInMetric(MetricType.Recall, True)
        }
        parameters = [PlotterParameters(0, x_label='x', y_label='y', title=label, fig_size=(11, 7))
                      for label, _ in metric_labels.items()]
        network = NeuralNet(
            binary_classifier,
            hyper_parameters,
            early_stop_logger,
            metric_labels,
            parameters)
        filename = '/users/patricknicolas/dev/geometriclearning/data/heart_diseases.csv'
        df = LabeledDataset.data_frame(filename)
        print(f'Heart Diseases data frame---\nColumns: {df.columns}\n{str(df)}')

        features_df = df[features]
        labels_df = df['target']
        batch_size = 4
        train_eval_split_ratio = 0.85
        dataset_loader = LabeledLoader(batch_size, train_eval_split_ratio)
        train_loader, eval_loader = dataset_loader.from_dataframes(features_df,labels_df,min_max_scaler,'float32')
        network(train_loader, eval_loader)


if __name__ == '__main__':
    unittest.main()
