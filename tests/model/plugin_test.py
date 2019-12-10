# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import os

from absl.testing import parameterized
import tensorflow as tf

from nucleus7.optimization.configs import (
    create_and_validate_optimization_config)
from nucleus7.test_utils.model_dummies import DummyPlugin2Layers
from nucleus7.test_utils.model_dummies import DummyPluginCNN
from nucleus7.utils import tf_utils
from nucleus7.utils import tf_varscopes_utils


class TestModelPlugin(parameterized.TestCase, tf.test.TestCase):

    def setUp(self):
        def side_effect_plugin_cnn(inputs_cnn):
            return {'predictions': tf.layers.conv2d(
                inputs_cnn, 10, 3, padding='same',
                activation=tf.nn.relu)}

        self.side_effects = {'cnn': side_effect_plugin_cnn}

    @parameterized.parameters(
        {'trainable': True},
        {'trainable': False}
    )
    def test_call(self, trainable):
        _, model_plugin, predictions = self._get_plugin_with_predictions(
            trainable=trainable)
        model_plugin.mode = 'train'

        self.assertSetEqual({'predictions'}, set(predictions))
        self.assertEqual(model_plugin._variable_scope, model_plugin.name)
        self.assertEqual(len(model_plugin._variables), 2)
        if trainable:
            self.assertEqual(len(tf.trainable_variables()),
                             len(model_plugin._variables))
        else:
            self.assertEqual(len(tf.trainable_variables()), 0)

        tensor_names = {k: v.name for k, v in predictions.items()}
        tensor_names_must = {k: "/".join([model_plugin.name, k]) + ':0'
                             for k in predictions}
        self.assertDictEqual(tensor_names_must,
                             tensor_names)

    def test_stop_gradients(self):
        inputs, _, predictions = self._get_plugin_with_predictions(
            trainable=True)
        grads_to_inputs = tf.gradients(list(predictions.values()),
                                       list(inputs.values()))
        self.assertLen(grads_to_inputs,
                       len(inputs))
        for each_grad in grads_to_inputs:
            self.assertIsInstance(each_grad, tf.Tensor)

        tf.reset_default_graph()
        inputs, _, predictions = self._get_plugin_with_predictions(
            trainable=True, stop_gradient_from_inputs=True)
        grads_to_inputs = tf.gradients(list(predictions.values()),
                                       list(inputs.values()))
        self.assertEqual([None] * len(inputs),
                         grads_to_inputs)

    @parameterized.parameters({'load_var_scope': None},
                              {'load_var_scope': 'load/temp/var/scope'})
    def test_maybe_initialize_from_checkpoint(self, load_var_scope):
        save_dir = self.get_temp_dir()
        plugin_name = 'cnn_plugin'
        fname = "plugin.chpt"
        load_fname = os.path.join(save_dir, fname)
        _, model_plugin, predictions = self._get_plugin_with_predictions(
            name=plugin_name)
        model_plugin.mode = 'train'
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.variables_ = {
                tf_varscopes_utils.remove_scope_from_name(
                    v.name, model_plugin._variable_scope): v for v
                in model_plugin._variables}
            var_list = self.variables_
            if load_var_scope:
                var_list = {'/'.join([load_var_scope, k]): v
                            for k, v in var_list.items()}
            var_list = {tf_utils.remove_tag_from_variable_name(k): v
                        for k, v in var_list.items()}
            saver = tf.train.Saver(var_list)
            saver.save(sess, load_fname)
            vars_orig = {v.name: v.eval() for v in model_plugin._variables}
        tf.reset_default_graph()
        _, model_plugin, predictions = self._get_plugin_with_predictions(
            name=plugin_name, load_fname=load_fname,
            load_var_scope=load_var_scope)
        model_plugin.maybe_initialize_from_checkpoint()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            vars_after_init = {v.name: v.eval()
                               for v in model_plugin._variables}
        self.assertAllClose(vars_orig, vars_after_init)

    @parameterized.parameters(
        {'use_configs': None, "second_layer_trainable": True},
        {'use_configs': "single", "second_layer_trainable": True},
        {'use_configs': '* bias', "second_layer_trainable": True},
        {'use_configs': "kernel bias", "second_layer_trainable": True},
        {'use_configs': "bias", "second_layer_trainable": True},
        {'use_configs': "*", "second_layer_trainable": True},
        {'use_configs': None, "second_layer_trainable": False},
        {'use_configs': "single", "second_layer_trainable": False},
        {'use_configs': '* bias', "second_layer_trainable": False},
        {'use_configs': "kernel bias", "second_layer_trainable": False},
        {'use_configs': "bias", "second_layer_trainable": False},
        {'use_configs': "*", "second_layer_trainable": False}
    )
    def test_get_optimization_configs_with_variables(
            self, second_layer_trainable, use_configs):
        tf.reset_default_graph()
        inputs = tf.zeros([10, 20], tf.float32)
        config1 = create_and_validate_optimization_config(
            optimizer_name='optimizer', learning_rate_multiplier=0.1)
        config2 = create_and_validate_optimization_config(
            optimizer_name='optimizer', learning_rate_multiplier=0.5)
        if not use_configs:
            optim_configs = None
        elif use_configs == "single":
            optim_configs = config1
        elif use_configs == 'bias':
            optim_configs = {'bias': config1}
        elif use_configs == '*':
            optim_configs = {'*': config1}
        elif use_configs == '* bias':
            optim_configs = {'*': config1,
                             'bias': config2}
        else:
            optim_configs = {'bias': config1,
                             'kernel': config2}

        plugin = DummyPlugin2Layers(
            second_layer_trainable=second_layer_trainable,
            optimization_configs=optim_configs).build()
        plugin.mode = 'train'
        with tf.variable_scope('model_scope'):
            with tf.variable_scope('additional_scope'):
                layer = tf.keras.layers.Dense(10)
                _ = layer(inputs)
            _ = plugin(inputs=inputs)

        configs_with_vars = plugin.get_optimization_configs_with_variables()

        var_names_bias = ['model_scope/DummyPlugin2Layers/dense/bias:0']
        var_names_kernel = ['model_scope/DummyPlugin2Layers/dense/kernel:0']

        if second_layer_trainable:
            var_names_bias += ['model_scope/DummyPlugin2Layers/dense_1/bias:0']
            var_names_kernel += [
                'model_scope/DummyPlugin2Layers/dense_1/kernel:0']

        if not use_configs:
            self.assertIsNone(configs_with_vars)
            return
        if use_configs == "single" or use_configs == '*':
            self.assertEqual(1,
                             len(configs_with_vars))
            var_names_set_must = set(var_names_bias + var_names_kernel)
            config_must = config1
            config_res, vars_res = configs_with_vars[0]
            self.assertIs(config_res, config_must)
            self.assertSetEqual(var_names_set_must,
                                {v.name for v in vars_res})
            return
        if use_configs == "bias":
            self.assertEqual(1,
                             len(configs_with_vars))
            var_names_set_must = set(var_names_bias)
            config_must = config1
            config_res, vars_res = configs_with_vars[0]
            self.assertIs(config_res, config_must)
            self.assertSetEqual(var_names_set_must,
                                {v.name for v in vars_res})
            return
        self.assertEqual(2,
                         len(configs_with_vars))
        if use_configs == '* bias':
            configs_with_var_names_must = [
                (config2, set(var_names_bias)),
                (config1, set(var_names_kernel))]
        else:
            configs_with_var_names_must = [
                (config1, set(var_names_bias)),
                (config2, set(var_names_kernel))]
        for each_must, each_res in zip(configs_with_var_names_must,
                                       configs_with_vars):
            self.assertIs(each_must[0], each_res[0])
            self.assertSetEqual(each_must[1],
                                {v.name for v in each_res[1]})

    def _get_plugin_with_predictions(self, **plugin_kwargs):
        tf.reset_default_graph()
        name = plugin_kwargs.pop('name', 'cnn_plugin')
        model_plugin = DummyPluginCNN(name=name, **plugin_kwargs).build()
        model_plugin.mode = 'train'
        inputs = self._get_inputs()
        predictions = model_plugin(**inputs)
        return inputs, model_plugin, predictions

    @staticmethod
    def _get_inputs():
        return {'inputs_cnn': tf.placeholder(tf.float32,
                                             shape=[None] * 3 + [3])}
