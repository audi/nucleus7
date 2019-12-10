# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

from unittest.mock import MagicMock

from absl.testing import parameterized

from nucleus7.core.nucleotide import Nucleotide
from nucleus7.test_utils.model_dummies import DummyPluginWithDummyParameter
from nucleus7.test_utils.test_utils import TestCaseWithReset
from nucleus7.test_utils.test_utils import register_new_class
from nucleus7.utils import nucleotide_utils


class TestNucleotideUtils(TestCaseWithReset, parameterized.TestCase):

    def test_get_nucleotide_signature(self):
        node = Nucleotide(
            name='cnn', inbound_nodes=['data'],
            incoming_keys_mapping={'data': {'image': 'inputs_cnn'}})
        node.incoming_keys = ['inputs1', 'inputs2', '_input_optional']
        node.generated_keys = ['predictions', '_predictions_optional']

        node.__doc__ = """
            Attributes
            ----------
            incoming_keys : list
                * inputs1 : inputs1 to node
                * inputs2 : inputs2 to node
                * inputs_wrong : wrong description
                * input_optional : optional inputs
            generated_keys : list
                * predictions : predictions 1
                * predictions_optional : optional predictions
            """
        args_must = {'inputs1': 'inputs1 to node',
                     'inputs2': 'inputs2 to node',
                     'input_optional': 'optional inputs'}
        returns_must = {'predictions': 'predictions 1',
                        'predictions_optional': 'optional predictions'}
        args, returns = nucleotide_utils.get_nucleotide_signature(node)

        self.assertSetEqual({'inputs1', 'inputs2', 'input_optional'},
                            set(args))
        self.assertSetEqual({'predictions', 'predictions_optional'},
                            set(returns))
        self.assertDictEqual(args_must, args)
        self.assertDictEqual(returns_must, returns)

        node.__doc__ = None
        node.process = MagicMock(return_value=0)
        node.process.__doc__ = """
            Parameters
            ----------
            inputs1
                inputs1 to node
            inputs2
                inputs2 to node
            inputs_wrong
                wrong description
            input_optional
                optional inputs

            Returns
            -------
            predictions
                predictions 1
            predictions_optional
                optional predictions
            """
        args, returns = nucleotide_utils.get_nucleotide_signature(node)
        self.assertDictEqual(args_must, args)
        self.assertDictEqual(returns_must, returns)

    @parameterized.parameters(
        {"inputs": [{'a': [1], 'b': 2}, {'a': 3, 'c': 4}],
         "collapsed_must": {'a': [[1], 3], 'b': 2, 'c': 4}},
        {"inputs": [{'a': [1], 'b': 2}, {'c': 3, 'd': 4}],
         "collapsed_must": {'a': [1], 'b': 2, 'c': 3, 'd': 4}},
        {"inputs": [{'a': 1, 'b': 2}, {'a': 3, 'c': 4}],
         "collapsed_must": {'a': [1, 3], 'b': 2, 'c': 4}},
        {"inputs": [{'alist': [[1], 2], 'dict': {'a': 10}},
                    {'alist': [3], 'dict': {'c': 4}}],
         "collapsed_must": {'alist': [[1], 2, 3], 'dict': {'a': 10, 'c': 4}}},
        {"inputs": [{'a': [3, 4], 'b': 4},
                    {'a': {'c': 'd'}, 'b': 6}],
         "collapsed_must": None}
    )
    def test_collapse_inputs(self, inputs, collapsed_must):
        if collapsed_must is not None:
            self.assertDictEqual(collapsed_must,
                                 nucleotide_utils.collapse_inputs(inputs))
        else:
            with self.assertRaises(ValueError):
                nucleotide_utils.collapse_inputs(inputs)

    @parameterized.parameters(
        {"inputs": [{'a': [1], 'b': 2}, {'a': 3, 'c': 4}],
         "mappings": [{}, {'a': 'd'}],
         "outputs_must": {'a': [1], 'b': 2, 'c': 4, 'd': 3}},
        {"inputs": [{'a': [1], 'b': 2}, {'a': 3, 'c': 4}],
         "mappings": [{}, {'a': '_'}],
         "outputs_must": {'a': [1], 'b': 2, 'c': 4}},
        {"inputs": [{'a': [1], 'b': 2, "c": 10}, {'a': 3, 'c': 4}],
         "mappings": [{'a': 'alist:1', 'b': 'alist:2', 'c': 'dict:a'},
                      {'a': 'alist:3', 'c': 'dict:c'}],
         "outputs_must": {'alist': [[1], 2, 3], 'dict': {'a': 10, 'c': 4}}},
        {"inputs": [{'a': [1], 'b': 2}, {'a': 3, 'c': 4}],
         "mappings": [{}, {'a': 'd'}],
         "outputs_must": {'a': [1], 'b': 2, 'c': 4, 'd': 3}},
        {"inputs": [{'a': [1, 2, 3], 'b': 2, "c": {"a": 5}},
                    {'a': 3, 'c': 4}],
         "mappings": [{"a": "_", "a:0": "a", "c:a": "e"}, {'c': 'e'}],
         "outputs_must": {'a': [1, 3], 'b': 2, 'e': [5, 4]}},
        {"inputs": [{'a': [1, 2, 3], 'b': 2, "c": {"a": 5}},
                    {'a': 3, 'c': 4}],
         "mappings": [{"a": "d", "a:0": "a", "c:a": "e"}, {'c': 'e'}],
         "outputs_must": {'a': [1, 3], 'b': 2, 'e': [5, 4], "d": [1, 2, 3]}},
        {"inputs": [{'a': [1, 2, 3], 'ab': 2, "c": {"a": 5}},
                    {'a': 3, 'c': 4}],
         "mappings": [{"a": "d", "a:0": "a", "c:a": "e"}, {'c': 'e'}],
         "outputs_must": {'a': [1, 3], 'ab': 2, 'e': [5, 4], "d": [1, 2, 3]}},
        {"inputs": [{'a': [1, 2, 3], 'ab': 2, "c": {"a": 5}},
                    {'a': 3, 'c': 4}],
         "mappings": [{"ab": "d", "*": "_"}, {'c': 'd', "*": "_"}],
         "outputs_must": {"d": [2, 4]}},
        {"inputs": [{'a': [1, 2, 3], 'ab': 2, "c": {"a": 5}},
                    {'e': 3, 'c': 4}],
         "mappings": [{"ab": "d", "*": "other"}, {'c': 'd', "*": "other"}],
         "outputs_must": {"d": [2, 4],
                          "other": {"a": [1, 2, 3],
                                    "c": {"a": 5},
                                    "e": 3}}},
        {"inputs": [{'a': [1, 2, 3], 'ab': 2, "c": {"a": 5},
                     "d": {"e": {"f": 10, "k": 20}}},
                    {'e': 3, 'c': 4}],
         "mappings": [{"ab": "d", "*": "_", "d:e:*": "other"},
                      {'c': 'd', "*": "_"}],
         "outputs_must": {"d": [2, 4],
                          "other": {"f": 10, "k": 20}}},
    )
    def test_remap_and_collapse_inputs(self, inputs, mappings, outputs_must):
        self.assertDictEqual(
            outputs_must,
            nucleotide_utils.remap_and_collapse_inputs(inputs, mappings))

    def test_get_sample_config_for_nucleotide(self):
        class_name = 'DummyPluginWithDummyParameter'
        register_new_class(class_name, DummyPluginWithDummyParameter)

        inbound_nodes = ['TODO_SET_CORRECT_INBOUND_NODES']
        incoming_keys_mapping = {
            'TODO_SET_INPUT_PLUGIN_NAME_0': {
                'TODO_SET_INPUT_PLUGIN_OUTPUT_KEY_NAME_0': 'inputs_mlp'}}
        correct_dict = {
            'inbound_nodes': inbound_nodes,
            'incoming_keys_mapping': incoming_keys_mapping,
            'dummy_parameter': 'TODO_SET_CORRECT_VALUE',
            'name': class_name,
            'activation': 'elu',
            'initializer': None,
            'dropout': 'normal',
            'trainable': True,
            'stop_gradient_from_inputs': False,
            'load_fname': None,
            'load_var_scope': None,
            'exclude_from_restore': False,
            'optimization_configs': None,
            'data_format': None,
            'allow_mixed_precision': True,
            'class_name': class_name
        }

        out_dict = nucleotide_utils.get_sample_config_for_nucleotide(class_name)
        keys_out = list(out_dict)
        keys_correct = list(correct_dict)
        self.assertSetEqual(set(keys_correct),
                            set(keys_out))
        self.assertDictEqual(correct_dict,
                             out_dict)
