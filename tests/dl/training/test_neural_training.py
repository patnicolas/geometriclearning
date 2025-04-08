import unittest

from dl.model.mlp_model import MLPModel
from dl.block.mlp_block import MLPBlock
from dl.training.hyper_params import HyperParams
from dl.training.training_summary import TrainingSummary
from plots.plotter import PlotterParameters
from dl.training.neural_training import NeuralTraining
from metric.metric import Metric
from dataset.tensor.labeled_loader import LabeledLoader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch import nn
import numpy as np


class NeuralTrainingTest(unittest.TestCase):

    @unittest.skip('Ignored')
    def test_init(self):
        input_block = MLPBlock(block_id='../../../python/input',
                               layer_module=nn.Linear(in_features=32, out_features=16),
                               activation_module=nn.ReLU())
        hidden_block = MLPBlock(block_id='hidden',
                                layer_module=nn.Linear(in_features=16, out_features=4),
                                activation_module=nn.ReLU())
        output_block = MLPBlock(block_id='output',
                                layer_module=nn.Linear(in_features=4, out_features=1),
                                activation_module=None)
        binary_classifier = MLPModel(model_id='test1', neural_blocks=[input_block, hidden_block, output_block])
        print(repr(binary_classifier))
        hyper_params = HyperParams(
            lr=0.001,
            momentum=0.95,
            epochs=12,
            optim_label='adam',
            batch_size=8,
            loss_function=nn.CrossEntropyLoss(),
            drop_out=0.2,
            train_eval_ratio=0.9)
        metric_labels = [Metric.train_loss_label, Metric.eval_loss_label, Metric.accuracy_label]
        network_training = NeuralTraining.build(hyper_params, metric_labels)


    def test_train_wages(self):
        from metric.metric import Metric
        from metric.built_in_metric import BuiltInMetric, MetricType

        hidden_block = MLPBlock.build(block_id='hidden',
                                      in_features=5,
                                      out_features=4,
                                      activation_module=nn.ReLU())
        output_block = MLPBlock.build(block_id='output',
                                      in_features=4,
                                      out_features=41,
                                      activation_module=nn.Sigmoid())
        binary_classifier = MLPModel(model_id='test1', neural_blocks=[hidden_block, output_block])
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
        training_summary = TrainingSummary(patience, min_diff_loss, early_stopping_enabled)
        metric_labels = {
            Metric.accuracy_label: BuiltInMetric(MetricType.Accuracy, encoding_len=-1, is_weighted=True),
            Metric.precision_label: BuiltInMetric(MetricType.Precision, encoding_len=-1, is_weighted=True)
        }
        parameters = [PlotterParameters(0, x_label='x', y_label='y', title=label, fig_size=(11, 7))
                      for label, _ in metric_labels.items()]

        network = NeuralTraining(
            hyper_parameters,
            training_summary,
            metric_labels,
            None,
            parameters)
        filename = '../../../data/misc/wages_cleaned.csv'
        df = pd.DataFrame(filename)
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
            MinMaxScaler())
        network(train_loader, eval_loader)

    @unittest.skip('Ignored')
    def test_train_eval_heart_diseases(self):
        from metric.metric import Metric
        from metric.built_in_metric import BuiltInMetric, MetricType

        features = ['age', 'sex', 'chest pain type', 'cholesterol', 'fasting blood sugar','max heart rate',
                    'exercise angina', 'ST slope']
        hidden_block = MLPBlock.build(block_id='hidden',
                                      in_features=len(features),
                                      out_features=4,
                                      activation_module=nn.ReLU())
        output_block = MLPBlock.build(block_id='output', in_features=4, out_features=1, activation_module=nn.Sigmoid())
        binary_classifier = MLPModel(model_id='test1', neural_blocks=[hidden_block, output_block])
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
        training_summary = TrainingSummary(patience, min_diff_loss, early_stopping_enabled)
        metric_labels = {
            Metric.accuracy_label: BuiltInMetric(MetricType.Accuracy, encoding_len=-1, is_weighted=True),
            Metric.precision_label: BuiltInMetric(MetricType.Precision, encoding_len=-1, is_weighted=True),
            Metric.recall_label: BuiltInMetric(MetricType.Recall, encoding_len=-1, is_weighted=True)
        }
        parameters = [PlotterParameters(0, x_label='x', y_label='y', title=label, fig_size=(11, 7))
                      for label, _ in metric_labels.items()]
        network = NeuralTraining(
            hyper_parameters,
            training_summary,
            metric_labels,
            None,
            parameters)
        filename = '/users/patricknicolas/dev/geometric_learning/data/heart_diseases.csv'
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
