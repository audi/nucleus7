# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import random
from typing import Generator
from typing import Iterator
from typing import Union
from unittest.mock import MagicMock

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from nucleus7.data.data_feeder import DataFeeder
from nucleus7.data.data_feeder import DataFeederFileList
from nucleus7.data.data_filter import DataFilter
from nucleus7.data.data_pipe import DataPipe
from nucleus7.data.file_list import FileList
from nucleus7.data.processor import DataProcessor
from nucleus7.test_utils.model_dummies import DataReaderDummyNP
from nucleus7.test_utils.model_dummies import FileListDummy


class DataFeederWithData(DataFeederFileList):
    file_list_keys = ['data']


class _DataFeederDummy(DataFeeder):
    generated_keys = ["data"]

    def build_generator(self) -> Union[Generator, Iterator]:
        for i in range(10):
            yield {"data": i}


class _DummyDataFilterOddData(DataFilter):
    def predicate(self, data) -> bool:
        return bool(data % 2)


class TestDataFeeder(parameterized.TestCase, tf.test.TestCase):

    def setUp(self):
        self.num_samples = 100
        file_names = {
            'data': ['f_{}.ext'.format(i) for i in range(self.num_samples)]}
        self.file_list = FileListDummy(file_names=file_names).build()

    def test_build_generator(self):
        file_list = self.file_list
        data_feeder = DataFeederWithData(file_list)
        data_feeder.build()
        generator = data_feeder.build_generator()
        for i in range(len(file_list)):
            next(generator)

        with self.assertRaises(StopIteration):
            next(generator)

    @parameterized.parameters({'allow_smaller_final_batch': True},
                              {'allow_smaller_final_batch': False})
    def test_feed_batch(self, allow_smaller_final_batch):
        file_list = self.file_list
        batch_size = 6
        number_of_complete_iterations = len(file_list) // batch_size
        number_of_complete_iterations += (1 if allow_smaller_final_batch else 0)
        data_feeder = DataFeederWithData(
            file_list, allow_smaller_final_batch=allow_smaller_final_batch)
        data_feeder.build()
        generated_keys = ['inp1', 'inp2', 'inp3']
        data_feeder.generated_keys = generated_keys

        data_feeder.create_data_from_generator = MagicMock(
            side_effect=lambda x: {k: np.random.rand(10)
                                   for k in generated_keys})

        batch = data_feeder.get_batch(batch_size)
        keys_must = [k for k in data_feeder.generated_keys]
        self.assertSetEqual(set(batch), set(keys_must))

        for i in range(number_of_complete_iterations - 1):
            _ = data_feeder.get_batch(batch_size)

        with self.assertRaises(StopIteration):
            data_feeder.get_batch(batch_size)

    def test_feed_batch_with_filter(self):
        batch_size = 2
        data_feeder = _DataFeederDummy(allow_smaller_final_batch=True)
        data_feeder.add_data_filter(_DummyDataFilterOddData())
        data_feeder.build()
        data = data_feeder.get_batch(batch_size)
        batches = [data]

        while True:
            try:
                batches.append(data_feeder.get_batch(batch_size))
            except StopIteration:
                break

        batches_must = [{"data": np.array([1, 3])},
                        {"data": np.array([5, 7])},
                        {"data": np.array([9])}]
        self.assertAllClose(batches_must,
                            batches)


class _DataGenerator1(DataProcessor):
    exclude_from_register = True
    generated_keys = ["data11", "data12"]
    is_tensorflow = False

    def process(self, **inputs):
        return {"data11": np.random.normal(size=[10, 5]),
                "data12": np.random.normal(size=[1])}


class _DataGenerator2(DataProcessor):
    exclude_from_register = True
    generated_keys = ["data21", "data22"]
    is_tensorflow = False

    def process(self, **inputs):
        return {"data21": np.random.normal(size=[1, 3]),
                "data22": np.random.normal(size=[5])}


class _DataProcessorNP(DataProcessor):
    exclude_from_register = True
    incoming_keys = ["data1"]
    generated_keys = ["data1_p"]
    is_tensorflow = False

    def __init__(self, add_num=None, **kwargs):
        super(_DataProcessorNP, self).__init__(**kwargs)
        self.add_num = add_num

    def process(self, data1):
        if self.add_num is None:
            add_num = np.random.normal(size=data1.shape)
        else:
            add_num = self.add_num
        return {"data1_p": data1 + add_num}


class TestDataFeederFromPipe(tf.test.TestCase, parameterized.TestCase):

    @parameterized.parameters({"number_of_samples": -1},
                              {"number_of_samples": 5})
    def test_call(self, number_of_samples):
        batch_size = 4
        output_keys_mapping = {"processor1": {"data11": "data11_rm"}}
        processor1 = _DataGenerator1(name="processor1").build()
        processor2 = _DataGenerator2(name="processor2").build()
        data_pipe = DataPipe(processors=[processor1, processor2]
                             ).build()
        data_feeder = DataFeeder.from_data_pipe(
            data_pipe=data_pipe,
            output_keys_mapping=output_keys_mapping,
            number_of_samples=number_of_samples,
            allow_smaller_final_batch=True,
        ).build()
        result = data_feeder(batch_size=batch_size)

        shapes = {k: v.shape for k, v in result.items()}
        shapes_must = {
            "data11_rm": [batch_size, 10, 5],
            "data12": [batch_size, 1],
            "data21": [batch_size, 1, 3],
            "data22": [batch_size, 5]
        }
        self.assertAllClose(shapes_must,
                            shapes)
        self.assertSetEqual(set(shapes_must),
                            set(data_feeder.generated_keys_all))
        self.assertIsInstance(data_feeder,
                              DataFeeder)

        result = data_feeder(batch_size=batch_size)
        shapes = {k: v.shape for k, v in result.items()}
        if number_of_samples == -1:
            self.assertAllClose(shapes_must,
                                shapes)
        else:
            shapes_must = {
                "data11_rm": [1, 10, 5],
                "data12": [1, 1],
                "data21": [1, 1, 3],
                "data22": [1, 5]
            }
            self.assertAllClose(shapes_must,
                                shapes)
            with self.assertRaises(StopIteration):
                _ = data_feeder(batch_size=batch_size)


class TestDataFeederFileListFromPipe(tf.test.TestCase):
    def setUp(self):
        self.num_samples = 50
        self.file_names_with_floats = {
            "data1": ["%05d" % random.randint(0, 100)
                      for _ in range(self.num_samples)],
            "data2": ["%d" % random.randint(10, 40)
                      for _ in range(self.num_samples)],
            "data3": ["%08d" % random.randint(100, 1000)
                      for _ in range(self.num_samples)]}

    def test_call(self):
        batch_size = 4
        output_keys_mapping = {"processor2": {"data1_p": "data2_p"}}
        reader_tf = DataReaderDummyNP(name="reader").build()
        processor1 = _DataProcessorNP(name="processor1",
                                      inbound_nodes=["reader"]).build()
        processor2 = _DataProcessorNP(
            name="processor2",
            inbound_nodes=["reader"],
            incoming_keys_mapping={"reader": {"data2": "data1"}}
        ).build()
        data_pipe = DataPipe(processors=[processor1, processor2],
                             readers=reader_tf
                             ).build()
        file_list = FileList.from_matched_file_names(
            self.file_names_with_floats)
        data_feeder = DataFeeder.from_data_pipe(
            data_pipe=data_pipe, file_list=file_list,
            output_keys_mapping=output_keys_mapping
        ).build()

        result = data_feeder(batch_size=batch_size)

        shapes = {k: v.shape for k, v in result.items()}
        shapes_must = {
            "data1": [batch_size],
            "data2": [batch_size],
            "data1_p": [batch_size],
            "data2_p": [batch_size]
        }

        self.assertAllClose(shapes_must,
                            shapes)
        self.assertSetEqual(set(shapes_must),
                            set(data_feeder.generated_keys_all))
        self.assertIsInstance(data_feeder,
                              DataFeederFileList)
