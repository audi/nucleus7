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

from nucleus7.test_utils.model_dummies import FileListDummy
from nucleus7.test_utils.test_utils import TestCaseWithReset
from nucleus7.utils import io_utils
from nucleus7.utils import nest_utils
from nucleus7.utils.kpi_utils import filter_kpi_values


class _FileListWithInt(FileListDummy):
    def sort_fn(self, path: int, key: str):
        return path


class TestKPIUtils(TestCaseWithReset, parameterized.TestCase,
                   tf.test.TestCase):

    def test_filter_kpi_values(self):
        kpi = {'kpi1': {'name1': 1, 'name2': 2,
                        'name3': [3, np.ones([2]), np.zeros([1])]},
               'kpi2': 10,
               'kpi3': [np.zeros([2, 2]), 1, 'temp1', np.zeros([2, 1])],
               'kpi4': [np.zeros([2, 2])],
               'kpi5': np.zeros([2, 2])}
        kpi_filtered, kpi_filtered_out = filter_kpi_values(kpi)
        kpi_filtered_must = {'kpi1': {'name1': 1, 'name2': 2,
                                      'name3': [3, 0.0]},
                             'kpi2': 10,
                             'kpi3': [1, 'temp1']}
        kpi_filtered_out_must = {'kpi1': {'name3': [np.ones([2])]},
                                 'kpi3': [np.zeros([2, 2]), np.zeros([2, 1])],
                                 'kpi4': [np.zeros([2, 2])],
                                 'kpi5': np.zeros([2, 2])}
        self.assertDictEqual(kpi_filtered_must, kpi_filtered)
        self.assertAllClose(kpi_filtered_out_must, kpi_filtered_out)
        kpi_filtered_flatten = nest_utils.flatten_nested_struct(kpi_filtered)
        for v in kpi_filtered_flatten.values():
            self.assertTrue(io_utils.is_jsonserializable(v))
