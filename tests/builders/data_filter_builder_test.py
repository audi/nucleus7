# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

from unittest.mock import MagicMock

from absl.testing import parameterized

from nucleus7.builders import data_filter_builder
from nucleus7.data.data_filter import DataFilter
from nucleus7.test_utils.test_utils import register_new_class
from nucleus7.test_utils.test_utils import reset_register_and_logger


class _DummyDataFilter1(DataFilter):
    exclude_from_register = True

    def __init__(self, p1=10, p2=20, p3=30):
        super(_DummyDataFilter1, self).__init__()
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3


class _DummyDataFilter2(DataFilter):
    exclude_from_register = True

    def __init__(self, p1=10):
        super(_DummyDataFilter2, self).__init__()
        self.p1 = p1


class TestDataFilterBuilder(parameterized.TestCase):

    def setUp(self) -> None:
        reset_register_and_logger()
        register_new_class("data_filter1", _DummyDataFilter1)
        register_new_class("data_filter2", _DummyDataFilter2)
        self.config1 = {"class_name": "data_filter1",
                        "p1": 20,
                        "p2": 500}
        self.config2 = {"class_name": "data_filter2",
                        "p1": 100}

    def test_build_single(self):
        data_filter = data_filter_builder.build(self.config1)
        self.assertIsInstance(data_filter, _DummyDataFilter1)
        self.assertTrue(data_filter.built)
        self.assertEqual(20,
                         data_filter.p1)
        self.assertEqual(500,
                         data_filter.p2)
        self.assertEqual(30,
                         data_filter.p3)

    def test_build_chain(self):
        config = [self.config1, self.config2]
        data_filters = data_filter_builder.build(config)
        self.assertIsInstance(data_filters, list)
        self.assertEqual(2,
                         len(data_filters))
        data_filter1, data_filter2 = data_filters
        self.assertIsInstance(data_filter1, _DummyDataFilter1)
        self.assertIsInstance(data_filter2, _DummyDataFilter2)
        self.assertTrue(data_filter1.built)
        self.assertTrue(data_filter2.built)
        self.assertEqual(20,
                         data_filter1.p1)
        self.assertEqual(500,
                         data_filter1.p2)
        self.assertEqual(30,
                         data_filter1.p3)
        self.assertEqual(100,
                         data_filter2.p1)

    @parameterized.parameters({"number_of_filters": 0},
                              {"number_of_filters": 1},
                              {"number_of_filters": 3})
    def test_data_filter_build_callback(self, number_of_filters):
        built_object = MagicMock()
        built_object.add_data_filter = MagicMock(return_value=None)
        built_object.add_data_filters = MagicMock(return_value=None)

        if number_of_filters == 0:
            data_filter = None
        elif number_of_filters == 1:
            data_filter = DataFilter()
        else:
            data_filter = [DataFilter()] * number_of_filters

        result = data_filter_builder.data_filter_build_callback(
            built_object, data_filter, arg1=10, arg2=20)

        self.assertIs(result, built_object)
        if number_of_filters == 0:
            built_object.add_data_filter.assert_not_called()
            built_object.add_data_filters.assert_not_called()
        elif number_of_filters == 1:
            built_object.add_data_filter.assert_called_once_with(data_filter)
            built_object.add_data_filters.assert_not_called()
        else:
            built_object.add_data_filter.assert_not_called()
            built_object.add_data_filters.assert_called_once_with(data_filter)
