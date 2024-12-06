import unittest

from dataset.caltech101_loader import Caltech101Loader
from dl.training.exec_config import ExecConfig


class Caltech101LoaderTest(unittest.TestCase):

    @unittest.skip("Ignore")
    def test_loader(self):
        caltech101_loader = Caltech101Loader(batch_size=8, split_ratio=0.9, resize_image=-1)
        train_loader, _ = caltech101_loader.loaders_from_path(
            root_path='../../data/caltech-101',
            exec_config=ExecConfig.default()
        )
        train_iter = iter(train_loader)
        first_data = next(train_iter)
        print(str(first_data))
        self.assertTrue(len(train_loader) > 0)

    @unittest.skip("Ignore")
    def test_show_samples(self):
        data_path = '../../data/caltech-101'
        Caltech101Loader.show_samples(data_path, is_random=True)
        self.assertTrue(True)

    def test_extract_features_labels(self):
        caltech101_loader = Caltech101Loader(batch_size=8, split_ratio=0.9, resize_image=128)
        train_loader, _ = caltech101_loader.loaders_from_path(
            root_path='../../data/caltech-101',
            exec_config=ExecConfig.default()
        )
        for batch in train_loader:
            images, labels = batch
            print(labels)
