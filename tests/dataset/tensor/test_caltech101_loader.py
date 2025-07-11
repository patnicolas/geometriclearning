import unittest
import logging
from dataset.tensor.caltech101_loader import Caltech101Loader
from dl.training.exec_config import ExecConfig
import os
import python
from python import SKIP_REASON



class Caltech101LoaderTest(unittest.TestCase):

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_loader(self):
        try:
            caltech101_loader = Caltech101Loader(batch_size=8, split_ratio=0.9, resize_image=-1)
            train_loader, _ = caltech101_loader.loaders_from_path(
                root_path='../../data/caltech-101',
                exec_config=ExecConfig.default()
            )
            train_iter = iter(train_loader)
            first_data = next(train_iter)
            logging.info(str(first_data))
            self.assertTrue(len(train_loader) > 0)
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_show_samples(self):
        try:
            data_path = '../../data/caltech-101'
            Caltech101Loader.show_samples(data_path)
            self.assertTrue(True)
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)

    def test_extract_features_labels(self):
        try:
            caltech101_loader = Caltech101Loader(batch_size=8, split_ratio=0.9, resize_image=128)
            train_loader, _ = caltech101_loader.loaders_from_path(
                root_path='../../data/caltech-101',
                exec_config=ExecConfig.default()
            )
            for batch in train_loader:
                images, labels = batch
                logging.info(labels)
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)
