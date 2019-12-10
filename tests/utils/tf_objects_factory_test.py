# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

from functools import partial

from absl.testing import parameterized
from tensorflow import keras
import tensorflow as tf

from nucleus7.utils import tf_objects_factory


class TestTfObjectsBuilder(parameterized.TestCase):

    def test_initializer_factory(self):
        initializer = tf_objects_factory.initializer_factory("RandomNormal")
        self.assertIsInstance(initializer, keras.initializers.RandomNormal)

        initializer_in = keras.initializers.RandomUniform()
        initializer = tf_objects_factory.initializer_factory(initializer_in)
        self.assertIs(initializer, initializer_in)

        initializer = tf_objects_factory.initializer_factory(
            {'name': 'Orthogonal', 'gain': 100}
        )
        self.assertIsInstance(initializer, keras.initializers.Orthogonal)
        self.assertEqual(initializer.gain, 100)

    def test_activation_factory(self):
        self.assertIs(tf_objects_factory.activation_factory('elu'), tf.nn.elu)
        self.assertIs(tf_objects_factory.activation_factory('hard_sigmoid'),
                      keras.activations.hard_sigmoid)

        activation_config = {'name': 'relu6', 'first': 100}
        activation = tf_objects_factory.activation_factory(activation_config)
        self.assertIsInstance(activation, partial)
        self.assertIs(activation.func, tf.nn.relu6)
        self.assertDictEqual(activation.keywords, {'first': 100})

        package_name = __name__ + '.'
        correct_name = '_dummy_activation_2'
        partial_name = '_dummy_activation_1'
        wrong_name = '_dummy_dropout_correct_signature'

        self.assertIs(
            tf_objects_factory.activation_factory(package_name+correct_name),
            _dummy_activation_2)

        with self.assertRaises(AssertionError):
            tf_objects_factory.activation_factory(package_name+partial_name)

        activation = tf_objects_factory.activation_factory(
            {'name': package_name+partial_name,
             'other_parameter': 0})
        self.assertIsInstance(activation, partial)
        self.assertIs(activation.func, _dummy_activation_1)
        self.assertDictEqual(activation.keywords, {'other_parameter': 0})

        with self.assertRaises(AssertionError):
            tf_objects_factory.activation_factory(package_name+wrong_name)

    @parameterized.parameters({'name': None, 'rate': None},
                              {'name': 'normal', 'rate': None},
                              {'name': 'alpha', 'rate': None},
                              {'name': 'alpha', 'rate': 0.5},
                              {'name': None, 'rate': 0.5},
                              {'name': 'normal', 'rate': 0.5})
    def test_dropout_factory(self, name, rate):
        dropout_config = {}
        if rate is not None:
            dropout_config['rate'] = rate
        if name is not None:
            dropout_config['name'] = name
        dropout_config = dropout_config or None

        if rate is None:
            self.assertIsNone(
                tf_objects_factory.dropout_factory(dropout_config))
            return

        dropout = tf_objects_factory.dropout_factory(dropout_config)
        self.assertIsInstance(dropout, partial)
        self.assertDictEqual(dropout_config, dropout.keywords)

        training = True
        tf.reset_default_graph()
        _ = dropout(100.0, training)
        graph = tf.get_default_graph()
        all_operation_last_names = [op.name.split('/')[-1]
                                    for op in graph.get_operations()]

        # TODO(johannes.dumler@audi.de): Change to only rate after the minimum
        #  tensorflow version is 1.13
        self.assertTrue(
            'keep_prob' in all_operation_last_names
            or 'rate' in all_operation_last_names)

        training = False
        tf.reset_default_graph()
        _ = dropout(100.0, training)
        graph = tf.get_default_graph()
        all_operation_last_names = [op.name.split('/')[-1]
                                    for op in graph.get_operations()]
        self.assertIn('Identity',
                      all_operation_last_names)

    def test_dropout_factory_custom_dropout(self):
        package_name = __name__ + '.'
        wrong_name = '_dummy_activation_2'
        partial_name = '_dummy_dropout_partial_signature'
        correct_name = '_dummy_dropout_correct_signature'

        self.assertIs(
            tf_objects_factory.dropout_factory(package_name + correct_name),
            _dummy_dropout_correct_signature)

        with self.assertRaises(AssertionError):
            tf_objects_factory.dropout_factory(package_name + partial_name)

        activation = tf_objects_factory.dropout_factory(
            {'name': package_name + partial_name,
             'other_parameter': 0})
        self.assertIsInstance(activation, partial)
        self.assertIs(activation.func,
                      _dummy_dropout_partial_signature)
        self.assertDictEqual(activation.keywords, {'other_parameter': 0})

        with self.assertRaises(AssertionError):
            tf_objects_factory.dropout_factory(package_name + wrong_name)

    def test_optimizer_from_name_and_parameters_factory(self):
        optimizer = tf_objects_factory.optimizer_factory_from_name_and_parameters(
            "AdadeltaOptimizer", learning_rate=100, rho=10)
        self.assertIsInstance(optimizer, tf.train.AdadeltaOptimizer)
        self.assertEqual(optimizer._lr, 100)
        self.assertEqual(optimizer._rho, 10)

        with self.assertRaises(AssertionError):
            tf_objects_factory.optimizer_factory_from_name_and_parameters(
                "NotExistingOptimizer", learning_rate=100)
        with self.assertRaises(AssertionError):
            tf_objects_factory.optimizer_factory_from_name_and_parameters(
                "add_queue_runner", learning_rate=100)


def _dummy_dropout_correct_signature(inputs, training):
    """
    Function with correct signature for dropout
    """
    del training
    return inputs


def _dummy_dropout_partial_signature(inputs, training, other_parameter):
    """
    Function with partially correct signature for dropout
    """
    del training, other_parameter
    return inputs


def _dummy_activation_1(inputs, other_parameter):
    """
    Function with partially correct signature for activations
    """
    del other_parameter
    return inputs


def _dummy_activation_2(x_input):
    """
    Function with correct signature for activations
    """
    return x_input