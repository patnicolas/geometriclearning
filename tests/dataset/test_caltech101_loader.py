import unittest

from dataset.caltech101_loader import Caltech101Loader
from dl.training.exec_config import ExecConfig

class Caltech101LoaderTest(unittest.TestCase):

    def test_loader(self):
        caltech101_loader = Caltech101Loader(split_ratio=0.9, resize_image=-1)
        train_loader, _ = caltech101_loader.loaders_from_path(
            root_path='../../data/caltech-101',
            exec_config=ExecConfig.default()
        )
        train_iter = iter(train_loader)
        first_data = next(train_iter)
        print(str(first_data))
        self.assertTrue(len(train_loader) > 0)

    def test_show_samples(self):
        data_path = '../../data/caltech-101'
        Caltech101Loader.show_samples(data_path, is_random=True)
