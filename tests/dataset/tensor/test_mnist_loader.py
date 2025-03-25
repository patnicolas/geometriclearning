import unittest
from dataset.tensor.mnist_loader import MNISTLoader
from dl.training.exec_config import ExecConfig


class MNISTLoaderTest(unittest.TestCase):

    @unittest.skip('Ignore')
    def test_loader(self):
        mnist_loader = MNISTLoader(batch_size=8, resize_image=-1)
        train_loader, _ = mnist_loader.loaders_from_path(
            root_path ='../../../data/MNIST',
            exec_config=ExecConfig.default()
        )
        train_iter = iter(train_loader)
        first_data = next(train_iter)
        print(str(first_data))
        self.assertTrue(len(train_loader) > 0)

    @unittest.skip('Ignore')
    def test_show_samples(self):
        mnist_loader = MNISTLoader(batch_size=8, resize_image=-1)
        train_loader, _ = mnist_loader.loaders_from_path(
            root_path='../../../data/MNIST',
            exec_config=ExecConfig.default()
        )
        ds = train_loader.dataset
        MNISTLoader.show_samples(ds.dataset, is_random=True)

    def test_extract_features(self):
        mnist_loader = MNISTLoader(batch_size=8, resize_image=-1)
        mnist_loader.extract_features(root_path='../../../data/MNIST')


