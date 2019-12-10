# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

from absl.testing import parameterized

from nucleus7.utils import deprecated


class TestDeprecated(parameterized.TestCase):

    def test_replace_deprecated_parameter_in_config(self):
        config = {'deprecated_name': 'value1', 'key1': 'value2'}
        with self.assertWarns(DeprecationWarning):
            config_replaced = deprecated.replace_deprecated_parameter_in_config(
                deprecated_name="deprecated_name", real_name="real_name",
                config=config)
        self.assertDictEqual(config_replaced,
                             {'real_name': "value1", "key1": "value2"})

        config = {'deprecated_name': 'value1', 'key1': 'value2'}
        with self.assertRaises(AssertionError):
            _ = deprecated.replace_deprecated_parameter_in_config(
                deprecated_name="deprecated_name_wrong", real_name="real_name",
                config=config)

        config = {'deprecated_name': 'value1', 'key1': 'value2'}
        config_replaced1 = deprecated.replace_deprecated_parameter_in_config(
            deprecated_name="deprecated_name_wrong", real_name="real_name",
            config=config, required=False)
        self.assertDictEqual(config, config_replaced1)

        config = {"d1": {"d2": {"d3": "value1"}}, "key1": "value2"}
        replace_name = "d1::d2::d3"
        new_name = "n1::n2"
        config_replaced_must = {"n1": {"n2": "value1"}, "key1": "value2"}
        with self.assertWarns(DeprecationWarning):
            config_replaced2 = deprecated.replace_deprecated_parameter_in_config(
                deprecated_name=replace_name, real_name=new_name,
                config=config)
        self.assertDictEqual(config_replaced2, config_replaced_must)

    def test_replace_deprecated_parameter_on_method(self):
        @deprecated.replace_deprecated_parameter(
            deprecated_name='deprecated_name',
            real_name='real_name')
        def method(real_name, param2):
            return real_name, param2

        with self.assertWarns(DeprecationWarning):
            res = method(deprecated_name=10, param2=20)
        self.assertTupleEqual(res,
                              (10, 20))
        self.assertTupleEqual(method(real_name=10, param2=20),
                              (10, 20))

    def test_replace_deprecated_parameter_on_class(self):
        class Dummy(object):

            @deprecated.replace_deprecated_parameter(
                deprecated_name='deprecated_name',
                real_name='real_name')
            def __init__(self, real_name, param2):
                self.real_name = real_name
                self.param2 = param2

        with self.assertWarns(DeprecationWarning):
            dummy1 = Dummy(deprecated_name=10, param2=20)
        self.assertEqual(dummy1.real_name,
                         10)
        self.assertEqual(dummy1.param2,
                         20)

        dummy2 = Dummy(real_name=10, param2=20)
        self.assertEqual(dummy2.real_name,
                         10)
        self.assertEqual(dummy2.param2,
                         20)

    @parameterized.parameters(
        {"config": {'a': {'b': {'c': 10}}, 'p1': 100, 'p2': 200},
         "pop_key": 'a::b::c', "result": (10, {'p1': 100, 'p2': 200})},
        {"config": {'a': {'b': {'c': 10}}, 'p1': 100, 'p2': 200},
         "pop_key": 'a', "result": ({'b': {'c': 10}}, {'p1': 100, 'p2': 200})},
        {"config": {'p1': 100, 'p2': 200},
         "pop_key": 'a', "default": 10, "result": (10, {'p1': 100, 'p2': 200})},
        {"config": {'p1': 100, 'p2': 200},
         "pop_key": 'p1', "result": (100, {'p2': 200})},
        {"config": {'p1': 100, 'p2': 200},
         "pop_key": 'a::b', "result": (None, {'p1': 100, 'p2': 200})},
        {"config": {'p1': 100, 'p2': 200},
         "pop_key": 'p1::b', "result": (None, {'p1': 100, 'p2': 200})},
    )
    def test_pop_value_from_config(self, config, pop_key, result, default=None):
        output = deprecated.pop_value_from_config(config, pop_key, default)
        result_value, result_config = result
        if result_value is None:
            self.assertIsNone(result_value)
        elif isinstance(result_value, list):
            self.assertListEqual(result_value, output)
        elif isinstance(result_value, dict):
            self.assertDictEqual(result_value, output)
        else:
            self.assertEqual(result_value, output)
        self.assertDictEqual(result_config,
                             config)

    @parameterized.parameters(
        {"config": {'a': {'b': {'c': 10}}, 'p1': 100, 'p2': 200},
         "get_key": 'a::b::c', "result": 10},
        {"config": {'a': {'b': {'c': 10}}, 'p1': 100, 'p2': 200},
         "get_key": 'a', "result": {'b': {'c': 10}}},
        {"config": {'p1': 100, 'p2': 200},
         "get_key": 'a', "default": 10, "result": 10},
        {"config": {'p1': 100, 'p2': 200},
         "get_key": 'p1', "result": 100},
        {"config": {'p1': 100, 'p2': 200},
         "get_key": 'a::b', "result": None},
        {"config": {'p1': 100, 'p2': 200},
         "get_key": 'p1::b', "result": None},
    )
    def test_get_value_from_config(self, config, get_key, result, default=None):
        output = deprecated.get_value_from_config(config, get_key, default)
        if result is None:
            self.assertIsNone(result)
        elif isinstance(result, list):
            self.assertListEqual(result, output)
        elif isinstance(result, dict):
            self.assertDictEqual(result, output)
        else:
            self.assertEqual(result, output)

    @parameterized.parameters(
        {"config": {'p1': 100, 'p2': 200}, "add_key": 'p1',
         "add_value": 10, "result": {'p1': 10, 'p2': 200}},
        {"config": {'p1': 100, 'p2': 200}, "add_key": 'p3',
         "add_value": 300, "result": {'p1': 100, 'p2': 200, 'p3': 300}},
        {"config": {'p1': 100, 'p2': 200}, "add_key": 'a::b::c',
         "add_value": 100,
         "result": {'a': {'b': {'c': 100}}, 'p1': 100, 'p2': 200}},
        {"config": {'p1': 100, 'p2': 200}, "add_key": 'p1::b::c',
         "add_value": 100,
         "result": {'p1': {'b': {'c': 100}, 'name': 100}, 'p2': 200}}
    )
    def test_add_value_to_config(self, config, add_key, add_value, result):
        self.assertDictEqual(
            result,
            deprecated.add_value_to_config(config, add_key, add_value)
        )

    def test_warning_deprecated_config_param(self):
        with self.assertWarns(DeprecationWarning):
            deprecated.warning_deprecated_config_param(
                "deprecated_name", "real_name", "additional_instructions",
                "value")

    def test_warning_deprecated_config_file(self):
        with self.assertWarns(DeprecationWarning):
            deprecated.warning_deprecated_config_file(
                "deprecated_fname", "real_fname", "additional_instructions",
                "value")
