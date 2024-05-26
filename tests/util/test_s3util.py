
from collections import OrderedDict
from unittest import TestCase
import constants
import unittest
from util.s3util import S3Util
import torch


class TestS3Util(TestCase):
    @unittest.skip("Not needed")
    def test_yield(self):
        lst = [0, 4, 5, 11, 8, 3, 5, 19, 22, 30, 2, 7]
        value = sum([x*x for x in lst])
        constants.log_info(value)

    @unittest.skip("Not needed")
    def test_s3_to_dataframe(self):
        s3_bucket = constants.s3_config['bucket_name']
        s3_folder = "dl/termsAndContext/utrad-all"
        is_nested = False
        num_files = 2
        s3_util = S3Util(s3_bucket, s3_folder, is_nested, num_files)
        df = s3_util.s3_to_dataframe('.json', ['termsText', 'age', 'gender', 'emrCode'])
        constants.log_info(str(df))

    @unittest.skip("Not needed")
    def test_write_ordered_dict(self):
        s3_folder = 'temp/models/test1-1'
        ordered_dict = OrderedDict([('l0.weight', torch.tensor([[0.1400, 0.4563, -0.0271, -0.4406],
                                           [-0.3289, 0.2827, 0.4588, 0.2031]])),
                     ('l0.bias', torch.tensor([0.0300, -0.1316])),
                     ('l1.weight', torch.tensor([[0.6533, 0.3413]])),
                     ('l1.bias', torch.tensor([-0.1112]))])

        s3_util = S3Util(constants.default_s3_bucket_name, s3_folder, False)
        s3_util.write_ordered_dict(ordered_dict)

    @unittest.skip("Not needed")
    def test_read_ordered_dict(self):
        s3_folder = 'temp/models/test1-1'
        s3_util = S3Util(constants.default_s3_bucket_name, s3_folder, False)
        loaded_ordered_dict = s3_util.read_ordered_dict()
        print(str(loaded_ordered_dict))

    @unittest.skip("Not needed")
    def test_write_value(self):
        value = 98
        s3_folder = 'temp/models/test1-2'
        s3_util = S3Util(constants.default_s3_bucket_name, s3_folder, False)
        s3_util.write_value(str(value), 'val')

    @unittest.skip("Not needed")
    def test_read_value(self):
        s3_folder = 'temp/models/test1-2'
        s3_util = S3Util(constants.default_s3_bucket_name, s3_folder, False)
        new_value = s3_util.read_value('val')
        print(f'New value {new_value}')

    @unittest.skip("Not needed")
    def test_dataframe(self):
        default_s3_bucket_name = constants.s3_config['bucket_name']
        s3_folder_name = "reports/latency/test1"
        s3_util = S3Util(default_s3_bucket_name, s3_folder_name, False, 200)
        df = s3_util.s3_to_dataframe('.json', ['dataSource', 'averageLatency'])
        print(df)

    def test_load_config_file(self):
        default_s3_bucket_name = constants.s3_config['bucket_name']
        s3_folder_name = "architecture/"
        s3_util = S3Util(default_s3_bucket_name, s3_folder_name, True, 200)
        config_dict = s3_util.s3_to_list('json')
        print(str(config_dict))


