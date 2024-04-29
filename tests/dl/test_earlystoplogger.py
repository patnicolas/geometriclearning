
import unittest
import random
from python.dl.earlystoplogger import EarlyStopLogger
from typing import Dict, AnyStr, List


class EarlyStopLoggerTest(unittest.TestCase):

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

    def test_update_metrics(self):
        patience = 2
        min_diff_loss = -0.001
        early_stopping_enabled = True
        early_stop_logger = EarlyStopLogger(patience, min_diff_loss, early_stopping_enabled)

        new_metrics = {'Accuracy': 0.5, 'F1': 0.6}
        early_stop_logger.update_metrics(new_metrics)
        new_metrics = {'Accuracy': 0.67, 'F1': 0.62}
        early_stop_logger.update_metrics(new_metrics)
        new_metrics = {'Accuracy': 0.69, 'F1': 0.67}
        early_stop_logger.update_metrics(new_metrics)
        early_stop_logger.summary()

    def test_summary(self):
        import math
        patience = 2
        min_diff_loss = -0.001
        early_stopping_enabled = True
        early_stop_logger = EarlyStopLogger(patience, min_diff_loss, early_stopping_enabled)

        train_loss = [10*math.exp(-n) + random.random() for n in range(1, 100)]
        eval_loss = [12 * math.exp(-n) + 1.5*random.random() for n in range(1, 100)]
        accuracy = [math.log(2.0 + n + random.random()) for n in range(1, 100)]
        f1 = [0.5 + 0.1*random.random() for _ in range(1, 100)]
        max_index = min(len(train_loss), len(eval_loss), len(accuracy))
        print(f'Recorded {max_index} values')
        for i in range(max_index):
            early_stop_logger(
                i,
                train_loss[i],
                eval_loss[i],
                {EarlyStopLogger.accuracy_label: accuracy[i], EarlyStopLogger.f1_label: f1[i]})
        early_stop_logger.summary()


if __name__ == '__main__':
    unittest.main()
