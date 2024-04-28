from unittest import TestCase

import torch
from util.raytune import RayTuneModel
from torch.utils.data import DataLoader
from models.cnn.convmodel import ConvModel
from models.cnn import ConvNeuralBlock
from torchvision import transforms
from torchvision.datasets import MNIST
from models.nnet.hyperparams import HyperParams
from models.cnn.convnet import ConvNet


class TestRayTuneModel(TestCase):
    def test_setup(self):
        try:
            in_channels = 1
            hidden_dim = 16
            conv_model = TestRayTuneModel.__init_2d_model(in_channels, hidden_dim)
            # Step 2: Define the hyper parameters
            lr = 0.0002
            momentum = 0.99
            epochs = 20
            optim_label = 'adam'
            batch_size = 32
            early_stop_ratio = 10.0
            loss_func = torch.nn.BCEWithLogitsLoss()
            hyper_params = HyperParams(lr, momentum, epochs, optim_label, batch_size, early_stop_ratio, loss_func)

            # Step 3: Load data set
            data_loader = TestRayTuneModel.load_data(batch_size)

            # Step 4: Train the encoder_model
            conv_net = ConvNet(conv_model, hyper_params, ConvNet.transform_image_label_func)
            ray_tune_model = RayTuneModel(data_loader,  data_loader, conv_net)
        except Exception as e:
            self.fail(str(e))


    def test_reset_config(self):
        self.fail()


    def test_save_checkpoint(self):
        self.fail()


    def test_load_checkpoint(self):
        self.fail()


    def test_step(self):
        self.fail()


    @staticmethod
    def __init_2d_model(model_id: str, in_channels: int, hidden_dim: int) -> ConvModel:
        conv_neural_block_1 = TestRayTuneModel.__init_2d_block(2, in_channels, hidden_dim, torch.nn.LeakyReLU(0.2), True)
        conv_neural_block_2 = TestRayTuneModel.__init_2d_block(2, hidden_dim, hidden_dim*2, torch.nn.LeakyReLU(0.2), True)
        conv_neural_block_3 = TestRayTuneModel.__init_2d_block(2, hidden_dim*2, 1, None, False)
        return ConvModel(model_id, [conv_neural_block_1, conv_neural_block_2, conv_neural_block_3], -1, None, -1)

    @staticmethod
    def __init_2d_block(
            dim: int,
            in_channels: int,
            out_channels: int,
            activation: torch.nn.Module,
            batch_norm: bool) -> ConvNeuralBlock:
        kernel_size = 1
        max_pooling = False
        bias = False
        flatten = False
        stride = 1
        return ConvNeuralBlock(
            dim,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            batch_norm,
            max_pooling,
            activation,
            bias,
            flatten)

    @staticmethod
    def load_data(batch_size: int) -> DataLoader:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        return DataLoader(
            MNIST('../../data/', download=True, transform=transform),
            batch_size=batch_size,
            shuffle=True)
