import unittest
import torch
from python.dataset.labeleddataset import LabeledDataset
import numpy as np


class LabeledDatasetTest(unittest.TestCase):

    def test_init(self):
        nn_features = torch.Tensor([[0.3, 0.4, 0.9],[0.1, 0.1, 0.0],[0.8, 0.2, 0.0],[0.7, 0.4, 0.3]])
        nn_labels = torch.Tensor([[1.0], [0.0], [0.0], [1.0]])
        labeled_dataset = LabeledDataset(nn_features, nn_labels)
        print(repr(labeled_dataset))
        train_data, eval_data = labeled_dataset[0]
        print(f'Train data: {str(train_data.numpy())}\nEval data: {str(eval_data.numpy())}')



if __name__ == '__main__':
    unittest.main()



