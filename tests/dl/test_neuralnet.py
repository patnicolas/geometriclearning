
import unittest
from python.dl.model.ffnnmodel import FFNNModel
from python.dl.block.ffnnblock import FFNNBlock
from python.dl.hyperparams import HyperParams
from python.dl.earlystoplogger import EarlyStopLogger
from python.util.plotter import PlotterParameters
from python.dl.neuralnet import NeuralNet
from python.dataset.unlabeleddataset import UnlabeledDataset
from python.dataset.unlabeledloader import UnlabeledLoader
from torch import nn
import numpy as np


class NeuralNetTest(unittest.TestCase):

    @unittest.skip('Ignored')
    def test_init(self):
        input_block = FFNNBlock.build('input', 32, 16, nn.ReLU())
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
        labels = [EarlyStopLogger.train_loss_label, EarlyStopLogger.eval_loss_label, EarlyStopLogger.accuracy_label]
        parameters = [PlotterParameters(0, x_label='x', y_label='y', title=label, fig_size=(12, 8)) for label in labels]
        network = NeuralNet(binary_classifier, hyper_parameters, early_stop_logger, parameters)
        filename = '/users/patricknicolas/dev/geometriclearning/data/wages_cleaned.csv'
        tensor_dataset = UnlabeledDataset.from_file(filename, ['Reputation', 'Age', 'Caps', 'Apps', 'Salary'])
        network.init_data_loader(batch_size=8, dataset=tensor_dataset)

    def test_train(self):
        input_block = FFNNBlock.build('input', 32, 16, nn.ReLU())
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
        labels = [EarlyStopLogger.train_loss_label, EarlyStopLogger.eval_loss_label, EarlyStopLogger.accuracy_label]
        parameters = [PlotterParameters(0, x_label='x', y_label='y', title=label, fig_size=(12, 8)) for label in labels]
        network = NeuralNet(binary_classifier, hyper_parameters, early_stop_logger, parameters)
        filename = '/users/patricknicolas/dev/geometriclearning/data/wages_cleaned.csv'
        df = UnlabeledDataset.data_frame(filename)
        df = df[['Reputation', 'Age', 'Caps', 'Apps', 'Salary']]
        average_salary = df['Salary'].mean()
        labeled_data_frame = df['Top_player'] = np.where(df['Salary'] > average_salary, 1.0, 0.0)

        print(f'Labeled data frame: {labeled_data_frame}')
        print(f'Update data frame: {df.columns}')
        batch_size = 4
        train_eval_split_ratio = 0.85
        dataset_loader = UnlabeledLoader(batch_size, train_eval_split_ratio)
        train_loader, eval_loader = dataset_loader.from_dataframe(df)
        network(train_loader, eval_loader)


if __name__ == '__main__':
    unittest.main()
