# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import os

import numpy as np
import tensorflow as tf

from nucleus7.test_utils.model_dummies import DataReaderDummyNP
from nucleus7.test_utils.model_dummies import DataReaderDummyTF
from nucleus7.test_utils.model_dummies import TfRecordsDataReaderDummy
from nucleus7.test_utils.test_utils import write_tf_records


class TestDataReader(tf.test.TestCase):
    def setUp(self):
        self.file_names = {"data1": "0001",
                           "data2": "20",
                           "data3": "300"}
        self.data_must = {"data1": 1.0,
                          "data2": 20.}

    def test_call_numpy(self):
        data_reader = DataReaderDummyNP()
        result = data_reader(**self.file_names)
        self.assertAllClose(self.data_must,
                            result)

    def test_tf_dataset(self):
        data_reader_tf = DataReaderDummyTF()
        data = tf.data.Dataset.from_tensor_slices(
            {k: [v] for k, v in self.file_names.items()})

        data_result = data.map(lambda x: data_reader_tf(**x))
        self.assertSetEqual({"data1", "data2"},
                            set(data_result.output_types.keys()))
        data_result_iter = data_result.make_one_shot_iterator()
        self.assertAllClose(self.data_must,
                            self.evaluate(data_result_iter.get_next()))


class TestTfRecordsDataReader(tf.test.TestCase):

    def setUp(self):
        self.data = {"data1": np.float32([10.]),
                     "data2": np.arange(200).reshape(10, 20).astype(np.float32),
                     "data3": np.arange(10).reshape(10, 1).astype(np.float32)}
        self.file_name = os.path.join(self.get_temp_dir(), "data.tfrecords")

    def test_call(self):
        write_tf_records(self.data, self.file_name)
        data = tf.data.TFRecordDataset([self.file_name])
        data_reader = TfRecordsDataReaderDummy(
            file_list_keys_mapping={"data": "tfrecords"}).build()
        data_read = data.map(lambda x: data_reader(data=x))
        data_iter = data_read.make_one_shot_iterator()
        data_eval = self.evaluate(data_iter.get_next())
        data_must = {k: v for k, v in self.data.items()}
        data_must["data_default"] = np.zeros([1], np.float32)
        self.assertAllClose(data_must,
                            data_eval)
