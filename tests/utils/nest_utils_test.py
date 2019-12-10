# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import copy

import numpy as np
import tensorflow as tf

from nucleus7.utils import nest_utils


class TestNestUtils(tf.test.TestCase):

    def test_combine_nested(self):
        np.random.seed(57454)
        list_of_nested = []
        for i in range(10):
            d = dict()
            d['a'] = np.random.randn(10, 5, 2)
            d['b'] = np.random.randn(100, 2)
            d['c'] = np.random.randn(10)
            d['e'] = np.random.randn(5, 6, 4)
            list_of_nested.append(d)

        combine_funs = [lambda x: np.concatenate(x, 0),
                        lambda x: x[0],
                        lambda x: np.mean(x, 0),
                        {'a': lambda x: np.diff(x, 0),
                         'b': lambda x: np.concatenate(np.argmax(x, 0), 0),
                         'default': lambda x: x[0]}]
        for combine_fun in combine_funs:
            list_of_nested_ = copy.deepcopy(list_of_nested)
            res = nest_utils.combine_nested(list_of_nested_, combine_fun)
            must = {}
            for k in list_of_nested_[0].keys():
                if isinstance(combine_fun, dict):
                    default = combine_fun['default']
                    combine_fun_ = combine_fun.get(k, default)
                else:
                    combine_fun_ = combine_fun
                must[k] = combine_fun_([l[k] for l in list_of_nested_])
            self.assertSetEqual(set(res.keys()),
                                set(list_of_nested_[0].keys()))
            self.assertAllClose(must, res)

    def test_combine_nested_deep(self):
        np.random.seed(57454)
        list_of_nested = []
        for i in range(3):
            d = dict()
            d['a'] = i + 1
            d['b'] = [i + j for j in range(5)]
            d['c'] = {'c1': [j + 10 for j in range(5)],
                      'c2': {'c21': 10, 'c22': 20},
                      'c3': 1}
            list_of_nested.append(d)
        list_of_nested_ = copy.deepcopy(list_of_nested)
        combine_fun = lambda x: np.sum(x)
        res = nest_utils.combine_nested(list_of_nested_, combine_fun)
        must = {'a': 6,
                'b': [3, 6, 9, 12, 15],
                'c': {'c1': [30, 33, 36, 39, 42],
                      'c2': {'c21': 30, 'c22': 60}, 'c3': 3}}
        self.assertDictEqual(res, must)

    def test_unflatten_dict_to_nested(self):
        inputs1 = {'a//b': 1, 'a//c': 2, 'e': 3}
        must1 = {'a': {'b': 1, 'c': 2}, 'e': 3}
        self.assertDictEqual(
            nest_utils.unflatten_dict_to_nested(inputs1), must1)

        inputs2 = {'a//b': 1, 'b': 2, 'e': 3}
        must2 = {'a': {'b': 1}, 'b': 2, 'e': 3}
        self.assertDictEqual(
            nest_utils.unflatten_dict_to_nested(inputs2), must2)

        inputs3 = {'a//b': 1, 'a//c': 2, 'a//d//d1': 2, 'a//d//d2': 4,
                   'e//0': 1, 'e//1': 2, 'e//2//e1': 10}
        must3 = {'a': {'b': 1, 'c': 2, 'd': {'d1': 2, 'd2': 4}},
                 'e': [1, 2, {'e1': 10}]}
        self.assertDictEqual(
            nest_utils.unflatten_dict_to_nested(inputs3), must3)

    def test_flatten_nested_struct(self):
        inputs1 = {'a': {'b': 1, 'c': 2}, 'e': 3}
        must1 = {'a//b': 1, 'a//c': 2, 'e': 3}
        self.assertDictEqual(
            nest_utils.flatten_nested_struct(inputs1), must1)

        inputs2 = {'a': 1, 'e': 3}
        must2 = {'a': 1, 'e': 3}
        self.assertDictEqual(
            nest_utils.flatten_nested_struct(inputs2), must2)

        inputs3 = {'a': {'b': 1, 'c': 2, 'd': {'d1': 2, 'd2': 4}},
                   'e': [1, 2, {'e1': 10}]}
        must3 = {'a//b': 1, 'a//c': 2, 'a//d//d1': 2, 'a//d//d2': 4,
                 'e//0': 1, 'e//1': 2, 'e//2//e1': 10}
        self.assertDictEqual(
            nest_utils.flatten_nested_struct(inputs3), must3)

    def test_flatten_nested_outputs(self):
        class Dummy(object):
            @nest_utils.flatten_nested_outputs
            def process(_self):
                return {"data": {"key1": "value1", "key2": "value2"}}

        res = Dummy().process()
        data_unflatten = {"data//key1": "value1", "data//key2": "value2"}
        self.assertDictEqual(res, data_unflatten)

    def test_unflatten_nested_inputs(self):
        data_flatten = {"data//key1": "value1", "data//key2": "value2"}
        data_unflatten = {"data": {"key1": "value1", "key2": "value2"}}

        class Dummy(object):
            @nest_utils.unflatten_nested_inputs("data")
            def process(_self, data):
                self.assertDictEqual(data, data_unflatten)
                return

        _ = Dummy().process(data=data_flatten)

    def test_flatten_nested_inputs_inside_of_list(self):
        @nest_utils.flatten_nested_inputs_inside_of_list('inputs')
        def method(inputs):
            return inputs

        data = [
            {"data": {"key1": "value1", "key2": "value2"}},
            {"data1": {"key11": "value1", "key12": {'key123': "value2"}}}
        ]
        data_unflatten_must = [
            {"data//key1": "value1", "data//key2": "value2"},
            {"data1//key11": "value1", "data1//key12//key123": "value2"}
        ]

        data_unflatten = method(inputs=data)
        for each_unflatten_result, each_unflatten_must in zip(
                data_unflatten, data_unflatten_must):
            self.assertDictEqual(each_unflatten_must, each_unflatten_result)

    def test_flatten_nested_inputs(self):
        @nest_utils.flatten_nested_inputs('inputs')
        def method(inputs):
            return inputs

        data = {"data": {"key1": "value1", "key2": "value2"}}
        data_unflatten_must = {"data//key1": "value1", "data//key2": "value2"}

        data_unflatten = method(inputs=data)
        self.assertDictEqual(data_unflatten_must, data_unflatten)

    def test_unflatten_nested_outputs(self):
        @nest_utils.unflatten_nested_outputs
        def method(inputs):
            return inputs

        data_flatten = {"data//key1": "value1", "data//key2": "value2"}
        data_unflatten_must = {"data": {"key1": "value1", "key2": "value2"}}
        data_unflatten = method(data_flatten)
        self.assertDictEqual(data_unflatten_must, data_unflatten)
