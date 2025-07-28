__author__ = "Patrick R. Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import json
from typing import Optional
import boto3
import pandas as pd
from pandas import json_normalize
import pickle
from collections import OrderedDict
AWS_SHARED_CREDENTIALS_FILE = "~/.aws/credentials2"


def get_session() -> boto3.Session:
    """
        Instantiate a S3 session using credentials defined in the configuration file.
        @return: S3 client session
    """
    return boto3.Session(profile_name='default',
                         region_name='us-east-2',
                         aws_access_key_id='access_key',
                         aws_secret_access_key='secret_key')


session = get_session()
s3 = session.resource('s3')


class S3Util(object):
    """
        Utility class for manipulating S3 data
    """
    def __init__(self, s3_bucket_name: str, s3_folder_name: str, is_nested: bool, num_files: int, is_json: bool = True):
        """
        Constructor for AWS S3 utilitiies
        @param s3_bucket_name: name of bucket
        @param s3_folder_name: path to the S3 folder containing the file to load
        @param is_nested: Boolean flag to specify whether the json structure is nested
        @param num_files: Number of files to be extracted from s3_folder, the entire folder for -1
        @param is_json: Boolean flag to specify if the format of the file is JSON
        """
        self.s3_bucket_name = s3_bucket_name
        self.s3_folder_name = s3_folder_name
        self.is_nested = is_nested
        self.num_files = num_files
        self.is_json = is_json

    def s3_file_to_iter(self) -> iter:
        my_bucket = s3.Bucket(self.s3_bucket_name)
        obj_summaries = list(my_bucket.objects.filter(Prefix=self.s3_folder_name))
        object_summary = obj_summaries[0] if self.num_files > 0 else obj_summaries

        data_in_bytes = object_summary.get()['Body'].read()  # data in the form of bytes array.
        decoded_data = data_in_bytes.decode('utf-8')  # Decode it in 'utf-8' format
        stringio_data = io.StringIO(decoded_data)  # IO module for creating a StringIO object.
        data_list = stringio_data.readlines()

        def gen_fields(data_list: str) -> list:
            for data in data_list:
                key_values = json.loads(data)
                yield {k: v for k, v in key_values.items()}

        if self.is_json:
            records = (record for record in gen_fields(data_list))
        else:
            records = iter(data_list)
        del object_summary, data_in_bytes, decoded_data, stringio_data
        return records


    def s3_to_list(self, file_extension: str = None, col_names: list = None) -> list:
        """
             Load the content of S3 folder into a data frame. Optional feature_names can be used to extract
             only the relevant fields.
             @param file_extension: File extension to match if defined.
             @param feature_names: Optional names of column_names for the encoder (and label) to extract
             @param col_names: List of column names
             @returns: Pandas dataframe
         """
        my_bucket = s3.Bucket(self.s3_bucket_name)
        obj_summaries = list(my_bucket.objects.filter(Prefix=self.s3_folder_name))
        object_summaries = obj_summaries[0: self.num_files] if self.num_files > 0 else obj_summaries
        sampled_object_summaries = [obj for obj in object_summaries if len(obj.key) > len(self.s3_folder_name)]
        constants.log_info(f'Retrieve {len(sampled_object_summaries)} object summaries')
        assert len(sampled_object_summaries) > 0, 'S3 to DataFrame is empty'

        def gen_fields(data_list: str, cols: list) -> list:
            for data in data_list:
                key_values = json.loads(data)
                yield {k: v for k, v in key_values.items() if k in cols} if cols is not None else key_values

        def proceed(obj_summary, cols: list) -> tuple:
            data_in_bytes = obj_summary.get()['Body'].read()  # data in the form of bytes array.
            decoded_data = data_in_bytes.decode('utf-8')  # Decode it in 'utf-8' format
            stringio_data = io.StringIO(decoded_data)  # IO module for creating a StringIO object.
            data_list = stringio_data.readlines()
            if self.is_nested:
                data_list[0] = ' '.join(data_list)
                data_list = data_list[0:1]
            if self.is_json:
                records = (record for record in gen_fields(data_list, cols))
            else:
                records = iter(data_list)
            del obj_summary, data_in_bytes, decoded_data, stringio_data
            return records

        accu = []
        [[accu.append(record) for record in proceed(object_summary, col_names)]
         for object_summary in sampled_object_summaries
         if file_extension is not None and object_summary.key.endswith(file_extension)]
        return accu


    def s3_to_iter(self, file_extension: str = None, col_names: list = None):
        """
             Load the content of S3 folder into a data frame. Optional feature_names can be used to extract
             only the relevant fields.
             @param file_extension: File extension to match if defined.
             @param feature_names: Optional names of column_names for the encoder (and label) to extract
             @param col_names: List of column names
             @returns: Pandas dataframe
         """
        my_bucket = s3.Bucket(self.s3_bucket_name)
        obj_summaries = list(my_bucket.objects.filter(Prefix=self.s3_folder_name))
        object_summaries = obj_summaries[0: self.num_files] if self.num_files > 0 else obj_summaries
        sampled_object_summaries = [obj for obj in object_summaries if len(obj.key) > len(self.s3_folder_name)]
        constants.log_info(f'Retrieve {len(sampled_object_summaries)} object summaries')
        assert len(sampled_object_summaries) > 0, 'S3 to DataFrame is empty'

        def gen_fields(data_list: str, cols: list) -> list:
            for data in data_list:
                key_values = json.loads(data)
                yield {k: v for k, v in key_values.items() if k in cols} if cols is not None else key_values

        def proceed(obj_summary, cols: list) -> iter:
            data_in_bytes = obj_summary.get()['Body'].read()  # data in the form of bytes array.
            decoded_data = data_in_bytes.decode('utf-8')  # Decode it in 'utf-8' format
            stringio_data = io.StringIO(decoded_data)  # IO module for creating a StringIO object.
            data_list = stringio_data.readlines()
            if self.is_json:
                records = [record for record in gen_fields(data_list, cols)]
            else:
                records = data_list
            del obj_summary, data_in_bytes, decoded_data, stringio_data
            return records

        return (proceed(object_summary, col_names)
                for object_summary in sampled_object_summaries
                if file_extension is not None and object_summary.key.endswith(file_extension))



    def s3_to_dataframe(self, file_extension: str = None, col_names: list = None) -> pd.DataFrame:
        accu = self.s3_to_list(file_extension, col_names)
        # Finally generate the data frame
        df = pd.DataFrame(accu)
        del accu
        return df

    def write_ordered_dict(self, ordered_dict: OrderedDict, ext: str = ''):
        obj = pickle.dumps(ordered_dict)
        s3.Object(self.s3_bucket_name, self.__s3_folder(ext)).put(Body=obj)

    def read_ordered_dict(self, ext: str = '') -> OrderedDict:
        obj = s3.Object(self.s3_bucket_name, self.__s3_folder(ext)).get()['Body'].read()
        return pickle.loads(obj)

    def write_value(self, value: str, ext: str = ''):
        s3.Object(self.s3_bucket_name, self.__s3_folder(ext)).put(Body=value)

    def read_value(self, ext: str = ''):
        return s3.Object(self.s3_bucket_name, self.__s3_folder(ext)).get()['Body'].read()

    def to_dataframe(self, file_extension: str = '') -> pd.DataFrame:
        prediction_data = self.read_json(file_extension)
        return json_normalize(prediction_data) if self.is_nested else pd.DataFrame.from_records(prediction_data)

    @staticmethod
    def to_csv(output_csv: str, data_frame: pd.DataFrame) -> Optional[str]:
        return data_frame.to_csv(output_csv)

    def __s3_folder(self, ext: str):
        return f'{self.s3_folder_name}.{ext}'
