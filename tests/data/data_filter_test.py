# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

from unittest.mock import MagicMock

from absl.testing import parameterized
import tensorflow as tf

from nucleus7.data.data_filter import DataFilter
from nucleus7.data.data_filter import DataFilterMixin


class _DummyWithDataFilter(DataFilterMixin):
    pass


class _DummyFilter(DataFilter):
    pass


class TestDatFilterMixin(parameterized.TestCase):

    def test_data_filters_and_add_data_filter(self):
        obj = _DummyWithDataFilter()
        self.assertIsNone(obj.data_filters)
        filter1 = _DummyFilter()
        filter2 = _DummyFilter()

        obj.add_data_filter(filter1)
        obj.add_data_filter(filter2)
        self.assertListEqual([filter1, filter2],
                             obj.data_filters)

        with self.assertRaises(ValueError):
            obj.add_data_filter(object())

    @parameterized.parameters(
        {"use_tensorflow": False, "filter_values": [True, False, False],
         "result_must": False},
        {"use_tensorflow": False, "filter_values": [True, True, True],
         "result_must": True},
        {"use_tensorflow": False, "filter_values": [True],
         "result_must": True},
        {"use_tensorflow": False, "filter_values": [False],
         "result_must": False},
        {"use_tensorflow": False, "filter_values": [True, False, False],
         "result_must": False},
        {"use_tensorflow": False, "filter_values": [True, True, True],
         "result_must": True},
        {"use_tensorflow": False, "filter_values": [True],
         "result_must": True},
        {"use_tensorflow": False, "filter_values": [False],
         "result_must": False},
    )
    def test_data_filter_true(self, use_tensorflow, filter_values, result_must):
        tf.reset_default_graph()
        obj = _DummyWithDataFilter()
        filters = []

        for i, each_filter_value in enumerate(filter_values):
            filter_ = _DummyFilter(
                predicate_keys_mapping={"input1": "input1remapped"}).build()
            if use_tensorflow:
                return_value = tf.constant(each_filter_value)
            else:
                return_value = each_filter_value
            filter_.predicate = MagicMock(return_value=return_value)
            filters.append(filter_)

        for each_filter in filters:
            obj.add_data_filter(each_filter)

        inputs = {"input1": "value1", "input2": "value2"}
        result = obj.data_filter_true(**inputs)
        inputs_to_predicate_must = {
            "input1remapped": "value1", "input2": "value2"}

        for each_filter in filters:
            each_filter.predicate.assert_called_once_with(
                **inputs_to_predicate_must)

        if use_tensorflow:
            with tf.Session() as sess:
                result = sess.run(result)

        self.assertEqual(result_must,
                         result)
