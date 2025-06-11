import unittest

from Lie.SOn_group import SOnGroup
import logging
import util



class SOnGroupTest(unittest.TestCase):

    @unittest.skip('Ignore')
    def test_init_2(self):
        son_group = SOnGroup(dim=2, point_type='matrix', equip=False)
        logging.info(son_group)
        logging.info(son_group.sample_points(2))

    @unittest.skip('Ignore')
    def test_init_3(self):
        son_group = SOnGroup(dim=3, point_type='matrix', equip=False)
        logging.info(son_group)
        sampled_points = son_group.sample_points(6)
        self.assertTrue(son_group.belongs(sampled_points))
        logging.info(son_group.sample_points(6))

    @unittest.skip('Ignore')
    def test_init_4(self):
        son_group = SOnGroup(dim=4, point_type='matrix', equip=False)
        logging.info(son_group)
        logging.info(f'\n{son_group.sample_points(3)}')

    @unittest.skip('Ignore')
    def test_exp_3(self):
        import torch

        son_group = SOnGroup(dim=3, point_type='matrix', equip=False)
        logging.info(son_group)
        sampled_point = son_group.sample_points(1)
        logging.info(f'Sampled point:\n{sampled_point}')
        tgt_vector = torch.Tensor(
            [[0.0, 0.8, -1.0],
             [-0.8, 0.0, 0.5],
             [1.0, -0.5, 0.0]
             ]
        )
        self.assertTrue(son_group.belongs(sampled_point))
        end_point = son_group.exp(tgt_vector, sampled_point)
        logging.info(f'End point:\n{end_point}')


    # @unittest.skip('Ignore')
    def test_log_3(self):
        import torch

        son_group = SOnGroup(dim=3, point_type='matrix', equip=False)
        base_point = son_group.sample_points(1)
        tgt_vector = torch.Tensor(
            [[0.0, 0.8, -1.0],
             [-0.8, 0.0, 0.5],
             [1.0, -0.5, 0.0]
             ]
        )
        end_point = son_group.exp(tgt_vector, base_point)
        logging.info(f'End point:\n{end_point}')
        computed_tgt_vector = son_group.log(end_point, base_point)
        logging.info(f'Original vector:\n{tgt_vector}\nComputed tangent vector:\n{computed_tgt_vector}')
        self.assertTrue(son_group.equal(tgt_vector, computed_tgt_vector))

    @unittest.skip('Ignore')
    def test_exp_4(self):
        import torch

        son_group = SOnGroup(dim=4, point_type='matrix', equip=False)
        logging.info(son_group)
        sampled_point = son_group.sample_points(1)
        logging.info(f'Sampled point:\n{sampled_point}')
        tgt_vector = torch.Tensor(
            [[0.0, 0.8, -1.0, 0.3],
             [-0.8, 0.0, 0.5, -0.4],
             [1.0, -0.5, 0.0, 0.1],
             [-0.3, 0.4, -0.1, 0.0]
             ]
        )
        self.assertTrue(son_group.belongs(sampled_point))
        end_point = son_group.exp(tgt_vector, sampled_point)
        logging.info(f'End point:\n{end_point}')


    @unittest.skip('Ignore')
    def test_lie_algebra_3(self):
        son_group = SOnGroup(dim=3, point_type='matrix', equip=False)
        logging.info(son_group)
        sampled_point = son_group.sample_points(1)
        self.assertTrue(son_group.belongs(sampled_point))
        algebra = son_group.lie_algebra(sampled_point)
        logging.info(f'Algebra SO(3): {algebra}')



