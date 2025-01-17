import unittest
import random
import torch
from metric.metric import Metric
from dl.training.training_summary import TrainingSummary
from typing import Dict, AnyStr, List


class TrainingSummaryTest(unittest.TestCase):

    @unittest.skip('Ignore')
    def test_metrics_dict(self):
        metrics: Dict[AnyStr, List[float]] = {}
        label = 'Train loss'
        train_losses = [0.6, 0.5, 0.4, 0.2]
        for loss in train_losses:
            if label in metrics:
                lst = metrics[label]
                lst.append(loss)
            else:
                metrics[label] = [loss]
        print(str(metrics))

    @unittest.skip('Ignore')
    def test_update_metrics(self):
        patience = 2
        min_diff_loss = -0.001
        early_stopping_enabled = True
        training_summary = TrainingSummary(patience, min_diff_loss, early_stopping_enabled)

        new_metrics1 = {'Accuracy': 0.5, 'F1': 0.6}
        training_summary.update_metrics(new_metrics1)
        new_metrics2 = {'Accuracy': 0.67, 'F1': 0.62}
        training_summary.update_metrics(new_metrics2)
        new_metrics3 = {'Accuracy': 0.69, 'F1': 0.67}
        training_summary.update_metrics(new_metrics3)
        training_summary.summary(None)

    @unittest.skip('Ignore')
    def test_summary(self):
        import math
        patience = 2
        min_diff_loss = -0.001
        early_stopping_enabled = True
        training_summary = TrainingSummary(patience, min_diff_loss, early_stopping_enabled)

        train_loss = [10*math.exp(-n) + random.random() for n in range(1, 100)]
        eval_loss = [12 * math.exp(-n) + 1.5*random.random() for n in range(1, 100)]
        accuracy = [math.log(2.0 + n + random.random()) for n in range(1, 100)]
        f1 = [0.5 + 0.1*random.random() for _ in range(1, 100)]
        max_index = min(len(train_loss), len(eval_loss), len(accuracy))
        print(f'Recorded {max_index} values')
        for i in range(max_index):
            training_summary(
                epoch=i,
                train_loss=torch.Tensor(train_loss[i]),
                eval_metrics={
                    Metric.accuracy_label: torch.Tensor(accuracy[i]),
                    Metric.f1_label: torch.Tensor(f1[i])
                })
        training_summary.summary(None)



if __name__ == '__main__':
    unittest.main()
