import unittest
import logging
from dataset.tensor.mnist_loader import MNISTLoader
from dl.training.exec_config import ExecConfig
import os
import python
from python import SKIP_REASON

class MNISTLoaderTest(unittest.TestCase):

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_loader(self):
        try:
            mnist_loader = MNISTLoader(batch_size=8, resize_image=-1)
            train_loader, _ = mnist_loader.loaders_from_path(
                root_path ='../../../data/MNIST',
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
            mnist_loader = MNISTLoader(batch_size=8, resize_image=-1)
            train_loader, _ = mnist_loader.loaders_from_path(
                root_path='../../../data/MNIST',
                exec_config=ExecConfig.default()
            )
            ds = train_loader.dataset
            MNISTLoader.show_samples(ds.dataset, is_random=True)
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)

    def test_extract_features(self):
        try:
            mnist_loader = MNISTLoader(batch_size=8, resize_image=-1)
            mnist_loader.extract_features(root_path='../../../data/MNIST')
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)


