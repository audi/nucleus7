# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import math

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from nucleus7.utils import model_utils


class TestKerasLayersMixin(tf.test.TestCase):
    def setUp(self):
        self.layer1 = tf.keras.layers.Dense(10)
        self.layer2 = tf.keras.layers.Dense(20)
        self.model_inputs = tf.keras.Input([10], dtype=tf.float32)
        self.model_layer1 = tf.keras.layers.Dense(30)
        self.model_layer2 = tf.keras.layers.Dense(40)
        model_outputs = self.model_layer2(self.model_layer1(self.model_inputs))
        self.model = tf.keras.Model(inputs=self.model_inputs,
                                    outputs=model_outputs)
        self.mixin_instance = model_utils.KerasLayersMixin()

    def test_add_keras_layer(self):
        layer1 = self.mixin_instance.add_keras_layer(self.layer1)
        layer2 = self.mixin_instance.add_keras_layer(self.layer2)
        model1 = self.mixin_instance.add_keras_layer(
            lambda: self.model, name="model")
        self.assertIs(self.layer1, layer1)
        self.assertIs(self.layer2, layer2)
        self.assertIs(self.model, model1)
        self.assertSetEqual({self.layer1, self.layer2, self.model},
                            set(self.mixin_instance.keras_layers))

    def test_keras_layers(self):
        _ = self.mixin_instance.add_keras_layer(self.layer1)
        _ = self.mixin_instance.add_keras_layer(self.layer2)
        _ = self.mixin_instance.add_keras_layer(self.model)
        self.assertSetEqual({self.layer1, self.layer2, self.model},
                            set(self.mixin_instance.keras_layers))

    def test_reset_keras_layers(self):
        layer1 = self.mixin_instance.add_keras_layer(self.layer1)
        layer2 = self.mixin_instance.add_keras_layer(self.layer2)
        model1 = self.mixin_instance.add_keras_layer(self.model)

        x = tf.ones([10, 20], tf.float32)
        x = layer1(x)
        x = layer2(x)
        self.assertTrue(self.layer1.built)
        self.assertTrue(self.layer2.built)
        self.assertTrue(self.model.built)
        self.assertTrue(self.model_layer1.built)
        self.assertTrue(self.model_layer2.built)

        self.mixin_instance.reset_keras_layers()
        self.assertFalse(self.layer1.built)
        self.assertFalse(self.layer2.built)
        self.assertFalse(self.model.built)
        self.assertFalse(self.model_layer1.built)
        self.assertFalse(self.model_layer2.built)

    def test_add_keras_layer_with_name(self):
        layer1_added = self.mixin_instance.add_keras_layer(self.layer1)
        layer2_added = self.mixin_instance.add_keras_layer(self.layer2)

        layer1_added_with_name = self.mixin_instance.add_keras_layer(
            self.layer1, "layer1")
        layer2_added_with_name = self.mixin_instance.add_keras_layer(
            self.layer2, "layer1")

        keras_layers_with_names_must = {self.layer1.name: self.layer1,
                                        self.layer2.name: self.layer2,
                                        "layer1": self.layer1}
        self.assertDictEqual(keras_layers_with_names_must,
                             self.mixin_instance.keras_layers_with_names)
        self.assertIs(layer1_added,
                      self.layer1)
        self.assertIs(layer2_added,
                      self.layer2)
        self.assertIs(layer1_added_with_name,
                      self.layer1)
        self.assertIs(layer2_added_with_name,
                      self.layer1)


class TestDefaultPlaceholderMixin(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        self.parameter1 = 10
        self.parameter2 = 20.0
        self.mixin_instance = model_utils.DefaultPlaceholderMixin()
        self.mixin_instance.name = "mixin_dummy"

    def test_empty_default_placeholders(self):
        self.assertEmpty(self.mixin_instance.default_placeholders)

    def test_add_default_placeholder(self):
        placeholder1 = self.mixin_instance.add_default_placeholder(
            self.parameter1, "parameter1")
        placeholder2 = self.mixin_instance.add_default_placeholder(
            self.parameter2, "parameter2")

        default_placeholders_must = {
            "parameter1": model_utils.DefaultPlaceholderInfo(
                "mixin_dummy//parameter1", placeholder1, self.parameter1),
            "parameter2": model_utils.DefaultPlaceholderInfo(
                "mixin_dummy//parameter2", placeholder2, self.parameter2)
        }
        self.assertAllEqual(default_placeholders_must,
                            self.mixin_instance.default_placeholders)
        with self.test_session() as sess:
            self.assertAllClose(10,
                                sess.run(placeholder1))
            self.assertAllClose(
                20,
                sess.run(placeholder1, feed_dict={placeholder1: 20}))
            self.assertAllClose(20.0,
                                sess.run(placeholder2))
            self.assertAllClose(
                50.0,
                sess.run(placeholder2, feed_dict={placeholder2: 50.0}))

    def test_remove_all_placeholders(self):
        _ = self.mixin_instance.add_default_placeholder(
            self.parameter1, "parameter1")
        _ = self.mixin_instance.add_default_placeholder(
            self.parameter2, "parameter2")
        self.mixin_instance.remove_all_placeholders()
        self.assertEmpty(self.mixin_instance.default_placeholders)


class TestModelUtils(tf.test.TestCase, parameterized.TestCase):

    @parameterized.parameters({'predictions_have_variable_shape': True},
                              {'predictions_have_variable_shape': False})
    def test_combine_predictions_from_devices(self,
                                              predictions_have_variable_shape):
        num_devices = 3
        tf.reset_default_graph()
        np.random.seed(45654)
        inputs_np, inputs_tf = [], []
        for i in range(num_devices):
            d, d_tf = {}, {}
            if predictions_have_variable_shape:
                d['a'] = np.random.randn(10, 5 + i, 2)
            else:
                d['a'] = np.random.randn(10, 5, 2)
            if predictions_have_variable_shape:
                d['b'] = np.random.randn(10, 2 + i)
            else:
                d['b'] = np.random.randn(10, 2)
            d['c'] = np.random.randn(10)
            for k, v in d.items():
                d_tf[k] = tf.constant(v)
            inputs_np.append(d)
            inputs_tf.append(d_tf)
        with self.test_session() as sess:
            res = sess.run(model_utils.combine_predictions_from_devices(
                inputs_tf, predictions_have_variable_shape))
        self.assertSetEqual(set(res), set(inputs_np[0]))
        must = {}
        for k in inputs_np[0]:
            if not predictions_have_variable_shape:
                must[k] = np.concatenate(
                    [inp[k] for inp in inputs_np], axis=0)
            else:
                max_shape = np.max([inp[k].shape for inp in inputs_np], 0)
                padded = [np.zeros(max_shape, inp[k].dtype)
                          for inp in inputs_np]
                for padded_, inp_ in zip(padded,
                                         [inp[k] for inp in inputs_np]):
                    padded_[np.where(inp_)] = inp_[np.where(inp_)]
                    must[k] = np.concatenate(padded, 0)

            self.assertAllClose(res[k], must[k])
            batch_shape = inputs_np[0][k].shape[0] * num_devices
            self.assertTrue(res[k].shape[0] == batch_shape)

    def test_combine_losses_from_devices(self):
        num_devices = 3
        tf.reset_default_graph()
        np.random.seed(45654)
        inputs_np, inputs_tf = [], []
        for i in range(num_devices):
            d, d_tf = {}, {}
            d['a'] = np.random.randn(10)
            d['b'] = np.random.randn(10, 5)
            d['c'] = np.random.randn(10, 5, 2)
            for k, v in d.items():
                d_tf[k] = tf.constant(v)
            inputs_np.append(d)
            inputs_tf.append(d_tf)
        with self.test_session() as sess:
            res = sess.run(model_utils.combine_losses_from_devices(inputs_tf))
        self.assertSetEqual(set(res), set(inputs_np[0]))
        must = {}
        for k in inputs_np[0]:
            must[k] = np.mean([inp[k] for inp in inputs_np], axis=0)
            self.assertAllClose(res[k], must[k])
            self.assertTrue(res[k].shape[0] == inputs_np[0][k].shape[0])

    def test_combine_summary_from_devices(self):
        num_devices = 3
        tf.reset_default_graph()
        np.random.seed(45654)
        inputs_np, inputs_tf = [], []
        for i in range(num_devices):
            d, d_tf = {}, {}
            d['scalar_loss'] = np.random.randn(10)
            d['scalar_value'] = np.random.randn(10)
            d['histogram_value'] = np.random.randn(10)

            d['image_temp'] = np.random.randn(10, 50, 10)
            d['text_temp'] = np.random.randn(10, 50)
            d['audio_value'] = np.random.randn(10, 100)

            for k, v in d.items():
                d_tf[k] = tf.constant(v)
            inputs_np.append(d)
            inputs_tf.append(d_tf)
        with self.test_session() as sess:
            res = sess.run(model_utils.combine_summary_from_devices(inputs_tf))
        self.assertSetEqual(set(res), set(inputs_np[0]))
        must = {}
        for k in inputs_np[0]:
            if 'scalar_' in k or 'histogram_' in k:
                must[k] = np.mean([inp[k] for inp in inputs_np], axis=0)
            else:
                must[k] = inputs_np[0][k]
            self.assertAllClose(res[k], must[k])
            self.assertTrue(res[k].shape[0] == inputs_np[0][k].shape[0])

    def test_split_inputs_to_devices(self):
        tf.reset_default_graph()
        np.random.seed(45654)
        devices = ['dev1', 'dev2', 'dev3']
        batch_size = 4
        inputs, inputs_tf = {}, {}
        inputs['a'] = np.random.randn(len(devices) * batch_size)
        inputs['b'] = np.random.randn(len(devices) * batch_size, 5)
        inputs['c'] = np.random.randn(len(devices) * batch_size, 5, 2)
        for k, v in inputs.items():
            inputs_tf[k] = tf.constant(v)

        with self.test_session() as sess:
            inputs_splitted = sess.run(model_utils.split_inputs_to_devices(
                inputs_tf, len(devices)))
        self.assertEqual(len(inputs_splitted), len(devices))
        for i, _ in enumerate(devices):
            for k in inputs:
                inputs_splitted_must = (
                    inputs[k][i * batch_size:(i + 1) * batch_size])
                self.assertAllClose(inputs_splitted[i][k],
                                    inputs_splitted_must)

    def test_select_inputs_by_sample_mask(self):
        batch_size = 5
        sample_mask_np = np.array([0, 1, 1, 0, 1])
        inputs_np = {"input1": np.random.randn(batch_size, 2, 1),
                     "input2": {"input21": np.random.randint(
                         0, 100, size=[batch_size]),
                         "input22": np.random.randn(batch_size, 10)},
                     "input3": [np.random.randn(batch_size, 1)]}
        sample_mask = tf.constant(sample_mask_np, tf.int32)
        inputs = {
            "input1": tf.constant(inputs_np["input1"], tf.float32),
            "input2": {
                "input21": tf.constant(inputs_np["input2"]["input21"],
                                       tf.int32),
                "input22": tf.constant(inputs_np["input2"]["input22"],
                                       tf.float32)},
            "input3": [tf.constant(inputs_np["input3"][0], tf.float32)]
        }
        keys_to_exclude_from_sample_mask = ["input1"]

        inputs_masked = model_utils.select_inputs_by_sample_mask(
            sample_mask, keys_to_exclude_from_sample_mask, **inputs)

        sample_mask_bool = sample_mask_np.astype(bool)
        inputs_masked_must = {
            "input1": inputs_np["input1"],
            "input2": {
                "input21": inputs_np["input2"]["input21"][sample_mask_bool],
                "input22": inputs_np["input2"]["input22"][sample_mask_bool]
            },
            "input3": [inputs_np["input3"][0][sample_mask_bool]],
        }

        inputs_masked_eval = self.evaluate(inputs_masked)
        self.assertAllClose(inputs_masked_must,
                            inputs_masked_eval)

    def test_select_inputs_by_sample_mask_np(self):
        batch_size = 5
        sample_mask = np.array([0, 1, 1, 0, 1])
        inputs = {"input1": np.random.randn(batch_size, 2, 1),
                  "input2": {"input21": np.random.randint(
                      0, 100, size=[batch_size]),
                      "input22": np.random.randn(batch_size, 10)},
                  "input3": [np.random.randn(batch_size, 1)]}
        keys_to_exclude_from_sample_mask = ["input1"]

        inputs_masked = model_utils.select_inputs_by_sample_mask_np(
            sample_mask, keys_to_exclude_from_sample_mask, **inputs)

        sample_mask_bool = sample_mask.astype(bool)
        inputs_masked_must = {
            "input1": inputs["input1"],
            "input2": {
                "input21": inputs["input2"]["input21"][sample_mask_bool],
                "input22": inputs["input2"]["input22"][sample_mask_bool]
            },
            "input3": [inputs["input3"][0][sample_mask_bool]],
        }

        self.assertAllClose(inputs_masked_must,
                            inputs_masked)

    def test_get_epoch_and_iteration_and_summary_step_from_global_train(self):
        number_iterations_per_epoch = 7
        max_number_of_iterations_per_epoch = 7
        global_steps = range(20)
        epoch_numbers_must = (
            math.ceil((each_step + 1) / number_iterations_per_epoch)
            for each_step in global_steps)
        iteration_numbers_must = (
            each_step % number_iterations_per_epoch + 1
            for each_step in global_steps)
        summary_steps_must = (each_step + 1 for each_step in global_steps)
        for global_step, epoch_must, iter_must, summary_step_must in zip(
                global_steps, epoch_numbers_must, iteration_numbers_must,
                summary_steps_must):
            (epoch_number, iteration_number, summary_step
             ) = model_utils.get_iteration_stat_from_global_step(
                mode="train",
                global_step=global_step,
                previous_iteration_number=0,
                number_iterations_per_epoch=number_iterations_per_epoch,
                max_number_of_iterations_per_epoch=
                max_number_of_iterations_per_epoch,
            )
            self.assertEqual(epoch_must,
                             epoch_number)
            self.assertEqual(iter_must,
                             iteration_number)
            self.assertEqual(summary_step_must,
                             summary_step)

    def test_get_epoch_and_iteration_and_summary_step_from_global_eval(self):
        number_iterations_per_epoch = 3
        max_number_of_iterations_per_epoch = 7
        global_steps = [7, 7, 7, 14, 14, 14, 21]
        epoch_numbers_must = [1, 1, 1, 2, 2, 2, 3]
        iteration_numbers_must = [1, 2, 3, 1, 2, 3, 1]
        summary_steps_must = [5, 6, 7, 12, 13, 14, 19]
        previous_iteration_number = 0
        for global_step, epoch_must, iter_must, summary_step_must in zip(
                global_steps, epoch_numbers_must, iteration_numbers_must,
                summary_steps_must):
            (epoch_number, iteration_number, summary_step
             ) = model_utils.get_iteration_stat_from_global_step(
                mode="eval",
                global_step=global_step,
                previous_iteration_number=previous_iteration_number,
                number_iterations_per_epoch=number_iterations_per_epoch,
                max_number_of_iterations_per_epoch=
                max_number_of_iterations_per_epoch,
            )
            self.assertEqual(epoch_must,
                             epoch_number)
            self.assertEqual(iter_must,
                             iteration_number)
            self.assertEqual(summary_step_must,
                             summary_step)
            previous_iteration_number = iteration_number
