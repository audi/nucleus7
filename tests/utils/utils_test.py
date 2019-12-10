# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from nucleus7.utils import utils


class TestUtils(tf.test.TestCase, parameterized.TestCase):

    def test_split_batch_inputs(self):
        batch_size = 3
        batch_data = {"data1": {"sub1": [np.random.rand(batch_size, 3),
                                         np.random.rand(batch_size),
                                         np.random.rand(batch_size, 2, 2)],
                                "sub2": np.arange(batch_size)},
                      "data2": np.random.rand(batch_size, 1),
                      "data3": [np.random.rand(batch_size),
                                np.random.rand(batch_size, 3)],
                      "not_batch": 10}
        data_split, data_not_batch = utils.split_batch_inputs(
            batch_data, ["not_batch"])

        data_not_batch_must = {"not_batch": 10}

        data_split_must = [
            {"data1": {"sub1": [each_item[i]
                                for each_item in batch_data["data1"]["sub1"]],
                       "sub2": batch_data["data1"]["sub2"][i]},
             "data2": batch_data["data2"][i],
             "data3": [each_item[i] for each_item in batch_data["data3"]]}
            for i in range(batch_size)
        ]

        self.assertAllClose(data_split_must,
                            data_split)
        self.assertDictEqual(data_not_batch_must,
                             data_not_batch)

    @parameterized.parameters(
        {"sample_mask": None, "is_last_iteration": False,
         "is_last_sample_must": [0, 0, 0, 0]},
        {"sample_mask": None, "is_last_iteration": True,
         "is_last_sample_must": [0, 0, 0, 1]},
        {"sample_mask": [1, 1, 1, 1], "is_last_iteration": False,
         "is_last_sample_must": [0, 0, 0, 0]},
        {"sample_mask": [1, 1, 1, 1], "is_last_iteration": True,
         "is_last_sample_must": [0, 0, 0, 1]},
        {"sample_mask": [0, 1, 1, 0], "is_last_iteration": False,
         "is_last_sample_must": [0, 0, 0, 0]},
        {"sample_mask": [0, 1, 1, 0], "is_last_iteration": True,
         "is_last_sample_must": [0, 0, 1, 1]})
    def test_get_is_last_sample_batchwise(self, sample_mask, is_last_iteration,
                                          is_last_sample_must):
        batch_size = 4
        is_last_sample = utils.get_is_last_sample_batchwise(
            batch_size, sample_mask=sample_mask,
            is_last_iteration=is_last_iteration)
        self.assertAllClose(is_last_sample_must,
                            is_last_sample)
