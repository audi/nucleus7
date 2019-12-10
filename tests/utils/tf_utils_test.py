# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import json
import math
import os

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from nucleus7.utils import model_utils
from nucleus7.utils import tf_collections_utils
from nucleus7.utils import tf_utils


class TestTfUtils(tf.test.TestCase, parameterized.TestCase):

    def test_count_trainable_params(self):
        def _get_parameter_number_conv2d(n_inp_fm, n_out_fm, kernel_size):
            return n_inp_fm * n_out_fm * kernel_size * kernel_size + n_out_fm

        tf.reset_default_graph()
        x = tf.placeholder(tf.float32, shape=[None, 10, 10, 1])
        x = tf.layers.conv2d(x, 10, 3)
        x = tf.layers.conv2d(x, 2, 1)

        number_trainable_params_must = (
                _get_parameter_number_conv2d(1, 10, 3)
                + _get_parameter_number_conv2d(10, 2, 1)
        )
        self.assertEqual(number_trainable_params_must,
                         tf_utils.count_trainable_params())

    @parameterized.parameters(
        {'axis': 0, 'pad_to_batch': False, "num_splits": 4},
        {'axis': 0, 'pad_to_batch': False, "num_splits": 3},
        {'axis': 0, 'pad_to_batch': True, "num_splits": 4},
        {'axis': 0, 'pad_to_batch': True, "num_splits": 3},
        {'axis': 1})
    def test_split_dict_to_list_of_dict(self, axis=0, pad_to_batch=False,
                                        num_splits=4):
        tf.reset_default_graph()
        tensor_len = 20

        inputs = {"input1": tf.range(0, tensor_len, 1),
                  "input2": tf.range(100, 100 + tensor_len, 1)}

        inputs_np_must = {'input1': np.arange(0, tensor_len),
                          'input2': np.arange(100, 100 + tensor_len)}
        tensor_len_with_pad = math.ceil(tensor_len / num_splits) * num_splits

        if tensor_len_with_pad > tensor_len:
            pad_len = tensor_len_with_pad - tensor_len
            pad_values = {k: np.repeat(v[-1], pad_len)
                          for k, v in inputs_np_must.items()}
            inputs_np_must = {k: np.concatenate([v, pad_values[k]], 0)
                              for k, v in inputs_np_must.items()}

        if axis == 1:
            inputs = {k: tf.reshape(v, [1, tensor_len])
                      for k, v in inputs.items()}

        if tensor_len % num_splits != 0 and not pad_to_batch:
            with self.assertRaises(ValueError):
                tf_utils.split_dict_to_list_of_dict(
                    inputs, num_splits, axis=axis, pad_to_batch=pad_to_batch)
            return

        result = tf_utils.split_dict_to_list_of_dict(
            inputs, num_splits, axis=axis, pad_to_batch=pad_to_batch)
        with self.test_session() as sess:
            result = sess.run(result)

        split_indices = list(
            range(0, tensor_len, tensor_len_with_pad // num_splits))
        split_indices.append(None)
        data_split_must = [{k: v[split_indices[i]: split_indices[i + 1]]
                            for k, v in inputs_np_must.items()}
                           for i in range(len(split_indices) - 1)]
        if axis == 1:
            data_split_must = [{k: np.expand_dims(v, 0)
                                for k, v in split_dict.items()}
                               for split_dict in data_split_must]
        self.assertAllClose(data_split_must,
                            result)

    def test_get_connected_inputs_to_predictions(self):
        tf.reset_default_graph()
        inputs = {'input1': tf.placeholder(tf.float32),
                  'input2': tf.placeholder(tf.float32),
                  'wrong_input': tf.placeholder(tf.float32)}
        outputs = {'output1': 1 + inputs["input1"],
                   'output2': 1 + inputs["input2"]}
        inputs_connected_must = {k: v for k, v in inputs.items()
                                 if k in ['input1', 'input2']}
        self.assertEqual(inputs_connected_must,
                         tf_utils.get_connected_inputs_to_predictions(
                             inputs, outputs, tf.get_default_graph()))

    def test_remove_tag_from_variable_name(self):
        self.assertEqual(
            'model/variable/name',
            tf_utils.remove_tag_from_variable_name('model/variable/name:1'))
        self.assertEqual(
            'model/variable/name',
            tf_utils.remove_tag_from_variable_name('model/variable/name'))

    @parameterized.parameters({"with_parameter_placeholders": False},
                              {"with_parameter_placeholders": True})
    def test_save_input_output_node_names(self, with_parameter_placeholders):
        tf.reset_default_graph()
        path = os.path.join(self.get_temp_dir(), 'inputs_outputs.json')
        inputs = {'input1': tf.placeholder(tf.float32, shape=[10, 10],
                                           name="input1_op"),
                  'input2': tf.placeholder(tf.float32, shape=[20, 20],
                                           name="input2_op")}
        outputs = {'output1': tf.add(tf.constant(1.0, dtype=tf.float32),
                                     inputs["input1"], name="output1_op"),
                   'output2': tf.add(tf.constant(1.0, dtype=tf.float32),
                                     inputs["input2"], name="output2_op")}
        tf_collections_utils.nested2collection("INPUTS", inputs)
        tf_collections_utils.nested2collection("OUTPUTS", outputs)

        if with_parameter_placeholders:
            parameter_placeholders = [
                model_utils.DefaultPlaceholderInfo(
                    "nucleotide1//param1",
                    tf.constant(20.0, dtype=tf.float32, name="name_param1"),
                    20.0),
                model_utils.DefaultPlaceholderInfo(
                    "nucleotide1//param2",
                    tf.constant([1, 2], dtype=tf.int32, name="name_param2"),
                    [1, 2]),
                model_utils.DefaultPlaceholderInfo(
                    "nucleotide1//param3",
                    tf.placeholder(dtype=tf.int32, name="name_param3",
                                   shape=None),
                    [1, 2])
            ]
        else:
            parameter_placeholders = None
        tf_utils.save_input_output_node_names(
            path, 'INPUTS', 'OUTPUTS',
            parameter_placeholders=parameter_placeholders)

        self.assertTrue(os.path.isfile(path))
        with open(path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        saved_data_must = {
            'inputs': {'input1': {'name': 'input1_op:0',
                                  'shape': [10, 10],
                                  'dtype': "float32"},
                       'input2': {'name': 'input2_op:0',
                                  'shape': [20, 20],
                                  'dtype': "float32"}},
            'outputs': {'output1': {'name': 'output1_op:0',
                                    'shape': [10, 10],
                                    'dtype': "float32"},
                        'output2': {'name': 'output2_op:0',
                                    'shape': [20, 20],
                                    'dtype': "float32"}}
        }
        if with_parameter_placeholders:
            saved_data_must.update({
                "parameter_placeholders": {
                    "nucleotide1//param1": {"name": "name_param1:0",
                                            "shape": [],
                                            "default_value": 20.0,
                                            "dtype": "float32"},
                    "nucleotide1//param2": {"name": "name_param2:0",
                                            "shape": [2],
                                            "default_value": [1, 2],
                                            "dtype": "int32"},
                    "nucleotide1//param3": {"name": "name_param3:0",
                                            "shape": None,
                                            "default_value": [1, 2],
                                            "dtype": "int32"},
                }
            })
        self.assertDictEqual(saved_data_must,
                             saved_data)

    def test_replace_outputs_with_named_identity(self):

        class Dummy(object):

            @tf_utils.replace_outputs_with_named_identity
            def process(self):
                inputs = {'input1': {'input11': tf.constant(1),
                                     'input12': tf.constant(2)},
                          'input2': tf.constant(3),
                          'input3': [tf.constant(4), tf.constant(5)],
                          'input4': 10,
                          'input5': 'temp'}
                return inputs

        tf.reset_default_graph()

        dummy = Dummy()
        with tf.variable_scope('scope1'):
            inputs_with_identity = dummy.process()
        inputs_with_identity_eval_must = {
            'input1': {'input11': 1,
                       'input12': 2},
            'input2': 3,
            'input3': [4, 5],
            'input4': 10,
            'input5': 'temp'
        }

        inputs_with_identity_names_must = {
            'input1': {'input11': 'scope1/input1//input11:0',
                       'input12': 'scope1/input1//input12:0'},
            'input2': 'scope1/input2:0',
            'input3': ['scope1/input3//0:0', 'scope1/input3//1:0']
        }

        inputs_with_identity_names = {
            'input1': {k: v.name
                       for k, v in inputs_with_identity['input1'].items()},
            'input2': inputs_with_identity['input2'].name,
            'input3': [v.name for v in inputs_with_identity['input3']]
        }

        self.assertDictEqual(inputs_with_identity_names_must,
                             inputs_with_identity_names)

        with self.test_session() as sess:
            inputs_with_tensors = {k: inputs_with_identity[k]
                                   for k in ['input1', 'input2', 'input3']}
            inputs_with_identity_eval = sess.run(inputs_with_tensors)
            inputs_with_identity_eval.update({
                k: inputs_with_identity[k]
                for k in ['input4', 'input5']})

        self.assertDictEqual(inputs_with_identity_eval_must,
                             inputs_with_identity_eval)

    def test_filter_variables_by_pattern(self):
        tf.reset_default_graph()
        with tf.variable_scope('var_scope'):
            variables = [tf.Variable(0.0, name="var1"),
                         tf.Variable(0.0, name="bias1"),
                         tf.Variable(0.0, name="var2"),
                         tf.Variable(0.0, name="bias2"),
                         tf.Variable(0.0, name="scope2/bias2")]
        self.assertSetEqual(
            {variables[0], variables[2]},
            set(tf_utils.filter_variables_by_pattern(
                variables, 'var', 'scope')))
        self.assertSetEqual(
            set(variables),
            set(tf_utils.filter_variables_by_pattern(variables, 'scope')))
        with self.assertWarns(Warning):
            self.assertEmpty(tf_utils.filter_variables_by_pattern(
                variables, 'not_existing_pattern'))
        self.assertEmpty(tf_utils.filter_variables_by_pattern(
            variables, 'not_existing_pattern'))
        self.assertSetEqual(
            {variables[1], variables[3], variables[4]},
            set(tf_utils.filter_variables_by_pattern(variables, 'bias')))

    def _test_add_variables_from_graph_without_collection(self):
        # TODO(oleksandr.vorobiov@audi.de): Implement
        pass

    def _test_combine_multiple_graphs_from_meta_and_checkpoint(self):
        # TODO(oleksandr.vorobiov@audi.de): Implement
        pass

    def _test_float32_variable_storage_getter(self):
        # TODO(oleksandr.vorobiov@audi.de): Implement
        pass

    def _test_figure_to_summary(self):
        # TODO(oleksandr.vorobiov@audi.de): Add test
        pass
