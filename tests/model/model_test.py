# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import os
from unittest.mock import MagicMock

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from nucleus7.core.nucleotide import Nucleotide
from nucleus7.model.configs import MixedPrecisionConfig
from nucleus7.model.configs import ModelResults
from nucleus7.model.fields import CollectionNames
from nucleus7.model.fields import ScopeNames
from nucleus7.model.model import Model
from nucleus7.test_utils.model_dummies import DummyMetric
from nucleus7.test_utils.model_dummies import DummyPluginCNN
from nucleus7.test_utils.model_dummies import DummyPluginFlatten
from nucleus7.test_utils.model_dummies import DummyPluginMLP
from nucleus7.test_utils.model_dummies import DummyPluginMLPKeras
from nucleus7.test_utils.model_dummies import DummyPostProcessor
from nucleus7.test_utils.model_dummies import DummySoftmaxLoss
from nucleus7.test_utils.model_dummies import DummySummary
from nucleus7.test_utils.model_dummies import ModelMock
from nucleus7.utils import nest_utils
from nucleus7.utils import tf_collections_utils


class TestModel(parameterized.TestCase, tf.test.TestCase):

    def setUp(self):
        self.batch_size = 2
        self.image_shape = (20, 20, 3)
        self.num_classes = 10
        self.data_dim = 100
        tf.reset_default_graph()
        np.random.seed(4564)

    @parameterized.parameters({'mode': tf.estimator.ModeKeys.TRAIN},
                              {'mode': tf.estimator.ModeKeys.EVAL},
                              {'mode': tf.estimator.ModeKeys.PREDICT})
    def test_call(self, mode):
        model = ModelMock(self.num_classes, regularization_l1=0.5).build()
        _, inputs = model.get_test_inputs(self.batch_size, self.data_dim)
        model.mode = mode

        model_results = model(inputs)
        self.assertIsInstance(model_results, ModelResults)
        self.assertIsInstance(model_results.inputs_preprocessed, dict)

        if mode == tf.estimator.ModeKeys.TRAIN:
            self.assertIsInstance(model_results.predictions_raw, dict)
            self.assertIsNone(model_results.predictions)
            self.assertIsNone(model_results.metrics)
            self.assertIsNone(model_results.summary)
            self.assertIsInstance(model_results.losses, dict)
            self.assertIsInstance(model_results.grads_and_vars, list)
            self.assertIsInstance(
                model_results.regularization_grads_and_vars, list)

        if mode == tf.estimator.ModeKeys.EVAL:
            self.assertIsInstance(model_results.predictions_raw, dict)
            self.assertIsInstance(model_results.predictions, dict)
            self.assertIsInstance(model_results.metrics, dict)
            self.assertIsInstance(model_results.summary, dict)
            self.assertIsInstance(model_results.losses, dict)
            self.assertIsNone(model_results.grads_and_vars)
            self.assertIsNone(model_results.regularization_grads_and_vars)

        if mode == tf.estimator.ModeKeys.PREDICT:
            self.assertIsNone(model_results.predictions_raw)
            self.assertIsInstance(model_results.predictions, dict)
            self.assertIsNone(model_results.losses)
            self.assertIsNone(model_results.metrics)
            self.assertIsNone(model_results.grads_and_vars)
            self.assertIsNone(model_results.summary)
            self.assertIsNone(model_results.regularization_grads_and_vars)

    @parameterized.parameters({'mode': 'train'},
                              {'mode': 'eval'})
    def test_call_with_keras(self, mode):
        inputs = self._get_inputs_for_model()
        model = self._get_model(inputs=inputs, with_keras_layers=True)
        model.mode = mode

        inputs_from_dataset = {'dataset': inputs}
        with tf.variable_scope('model') as scope:
            _ = model.forward_pass(inputs_from_dataset=inputs_from_dataset)
            trainable_variables_after_first_run = tf.trainable_variables()
            for each_nucleotide in model.all_nucleotides.values():
                keras_layers = each_nucleotide.keras_layers
                for each_keras_layer in keras_layers:
                    self.assertTrue(each_keras_layer.built)
            scope.reuse_variables()
            _ = model.forward_pass(inputs_from_dataset=inputs_from_dataset)
            trainable_variables_after_second_run = tf.trainable_variables()

        self.assertSetEqual(set(trainable_variables_after_first_run),
                            set(trainable_variables_after_second_run))

    def test_default_placeholders(self):
        inputs = self._get_inputs_for_model()
        model = self._get_model(with_metric=False, inputs=inputs)
        self.assertEmpty(model.default_placeholders)
        model.all_nucleotides["cnn"].add_default_placeholder(
            10, "parameter1", shape=None)
        model.all_nucleotides["cnn"].add_default_placeholder(
            20, "parameter2")
        model.all_nucleotides["pp_mlp"].add_default_placeholder(
            30, "parameter3")

        default_placeholders = model.default_placeholders
        default_placeholders_names_must = {
            "cnn//parameter1", "cnn//parameter2", "pp_mlp//parameter3"}
        default_placeholders_names = {
            each_item.full_name for each_item in default_placeholders}
        self.assertEqual(3, len(default_placeholders))
        self.assertSetEqual(default_placeholders_names_must,
                            default_placeholders_names)

        model.reset_tf_graph()
        self.assertEmpty(model.default_placeholders)
        self.assertEmpty(model.all_nucleotides["cnn"].default_placeholders)
        self.assertEmpty(model.all_nucleotides["pp_mlp"].default_placeholders)

    def test_model_has_correct_graph(self):
        inputs = self._get_inputs_for_model()
        model = self._get_model(with_metric=False, inputs=inputs)
        model.mode = tf.estimator.ModeKeys.EVAL
        _ = model(inputs)
        graph_def_model = tf.get_default_graph().as_graph_def()

        tf.reset_default_graph()
        inputs = self._get_inputs_for_model()
        self._get_same_as_model_graph_from_scratch(inputs)
        graph_def_must = tf.get_default_graph().as_graph_def()

        tf.test.assert_equal_graph_def(graph_def_model, graph_def_must)

    def test_build_inference_meta_graph(self):
        model = ModelMock(self.num_classes).build()
        model.reset_tf_graph = MagicMock(
            wraps=model.reset_tf_graph)

        _, inputs = model.get_test_inputs(self.batch_size, self.data_dim)
        model_keys = {'predictions_raw': ['predictions_raw'],
                      'losses': ['loss', 'total_loss'],
                      'predictions': ['classes'],
                      'summary': ['scalar_labels', 'scalar_classes'],
                      'metric': ['metric']
                      }
        input_shapes = {k: v.get_shape() for k, v in inputs.items()}

        with self.assertRaises(ValueError):
            # because ModelMock does not have postprocessors set,
            # it should raise error
            _ = model.build_inference_graph(inputs)

        # lets set postprocessord to some mock value
        model.postprocessors = {"postprocessor1": Nucleotide()}
        _ = model.build_inference_graph(inputs)

        self.assertEqual(tf.estimator.ModeKeys.PREDICT,
                         model.mode)
        model.reset_tf_graph.assert_called_once_with()

        inputs_from_coll = tf_collections_utils.collection2nested(
            CollectionNames.INPUTS)
        inputs_connected_names_must = ['data']
        self.assertSetEqual(set(inputs_from_coll),
                            set(inputs_connected_names_must))

        predictions_from_coll = tf_collections_utils.collection2nested(
            CollectionNames.PREDICTIONS)
        self.assertSetEqual(set(predictions_from_coll),
                            set(model_keys['predictions']))

        for k in inputs_connected_names_must:
            shape = input_shapes[k]
            shape_res = inputs_from_coll[k].shape
            shape_must = shape
            self.assertTrue(np.all(
                [a == b for a, b in zip(shape_res.as_list(),
                                        shape_must.as_list())]))

    @parameterized.parameters({"use_mixed_precision": False},
                              {"use_mixed_precision": True})
    def test_forward_pass(self, use_mixed_precision):
        inputs = self._get_inputs_for_model()
        model = self._get_model(inputs=inputs)
        if use_mixed_precision:
            model.mixed_precision_config = MixedPrecisionConfig(True, 100)
        model.mode = 'train'
        inputs_from_dataset = {'dataset': inputs}
        predictions = model.forward_pass(
            inputs_from_dataset=inputs_from_dataset)

        predictions_flatten = nest_utils.flatten_nested_struct(predictions)
        prediction_keys = set(predictions_flatten.keys())
        prediction_keys_must = set(
            ['//'.join([pl.name, 'predictions'])
             for pl in model.plugins.values()])
        self.assertSetEqual(prediction_keys_must, prediction_keys)

        # test predictions dtype
        # since all plugins in model return float predictions, they should be
        # float16 with mixed precision and float32 otherwise
        dtype_must = tf.float16 if use_mixed_precision else tf.float32
        for prediction_name, pred in predictions_flatten.items():
            plugin_name = prediction_name.split("//")[0]
            if not model.plugins[plugin_name].allow_mixed_precision:
                self.assertEqual(pred.dtype, tf.float32)
            else:
                self.assertEqual(pred.dtype, dtype_must)

    @parameterized.parameters({"use_mixed_precision": False},
                              {"use_mixed_precision": True})
    def test_postprocess_predictions(self, use_mixed_precision):
        inputs = self._get_inputs_for_model()
        model = self._get_model(inputs=inputs)
        if use_mixed_precision:
            model.mixed_precision_config = MixedPrecisionConfig(True, 100)
        model.mode = 'train'
        inputs_from_dataset = {'dataset': inputs}
        predictions_raw = model.forward_pass(
            inputs_from_dataset=inputs_from_dataset)
        predictions_postprocessed = model.postprocess_predictions(
            inputs_from_dataset=inputs_from_dataset,
            predictions_raw=predictions_raw)

    @parameterized.parameters({"use_mixed_precision": False},
                              {"use_mixed_precision": True})
    def test_calculate_losses(self, use_mixed_precision=True):
        inputs = self._get_inputs_for_model()
        model = self._get_model(inputs=inputs)
        if use_mixed_precision:
            model.mixed_precision_config = MixedPrecisionConfig(True, 100)
        model.mode = 'train'
        predictions = model.forward_pass(
            inputs_from_dataset={'dataset': inputs})
        losses = model.calculate_losses(inputs_from_dataset={'dataset': inputs},
                                        predictions_raw=predictions)
        losses_flatten = nest_utils.flatten_nested_struct(losses)
        self.assertIn('total_loss', losses)
        # dtype of loss should be always float32 and independent of
        # use_mixed_precision
        for loss in losses_flatten.values():
            self.assertEqual(loss.dtype, tf.float32)

    @parameterized.parameters({"use_mixed_precision": False},
                              {"use_mixed_precision": True})
    def test_get_summaries(self, use_mixed_precision=True):
        inputs = self._get_inputs_for_model()
        model = self._get_model(inputs=inputs)
        if use_mixed_precision:
            model.mixed_precision_config = MixedPrecisionConfig(True, 100)
        model.mode = 'train'
        inputs_from_dataset = {'dataset': inputs}
        predictions_raw = model.forward_pass(
            inputs_from_dataset=inputs_from_dataset)
        predictions_postprocessed = model.postprocess_predictions(
            inputs_from_dataset=inputs_from_dataset,
            predictions_raw=predictions_raw)
        summaries = model.get_summaries(inputs_from_dataset=inputs_from_dataset,
                                        predictions_raw=predictions_raw,
                                        predictions=predictions_postprocessed)

    @parameterized.parameters({"use_mixed_precision": False},
                              {"use_mixed_precision": True})
    def test_get_metrics(self, use_mixed_precision=True):
        inputs = self._get_inputs_for_model()
        model = self._get_model(inputs=inputs)
        if use_mixed_precision:
            model.mixed_precision_config = MixedPrecisionConfig(True, 100)
        model.mode = 'train'
        inputs_from_dataset = {'dataset': inputs}
        predictions_raw = model.forward_pass(
            inputs_from_dataset=inputs_from_dataset)
        predictions_postprocessed = model.postprocess_predictions(
            inputs_from_dataset=inputs_from_dataset,
            predictions_raw=predictions_raw)
        metrics = model.get_metrics(inputs_from_dataset=inputs_from_dataset,
                                    predictions_raw=predictions_raw,
                                    predictions=predictions_postprocessed)

    @parameterized.parameters({'regularization': ""},
                              {'regularization': 'l1'},
                              {'regularization': 'l2'},
                              {'regularization': 'l1l2'})
    def test_calculate_gradients(self, regularization):
        regularization_l1, regularization_l2 = 0, 0
        if 'l1' in regularization:
            regularization_l1 = 0.5
        if 'l2' in regularization:
            regularization_l2 = 0.25
        inputs = {
            'image': tf.constant(np.random.randn(
                self.batch_size, *self.image_shape), tf.float32),
            'labels': tf.constant(np.random.randint(
                0, self.num_classes, size=[self.batch_size, self.num_classes]),
                tf.float32)}

        model = self._get_model(with_metric=False, inputs=inputs,
                                regularization_l1=regularization_l1,
                                regularization_l2=regularization_l2)
        model.mode = tf.estimator.ModeKeys.TRAIN
        model_results = model(inputs)
        trainable_variables = tf.trainable_variables()

        self.assertEqual(len(trainable_variables),
                         len(model_results.grads_and_vars))
        self.assertSetEqual(set(trainable_variables),
                            set(list(zip(*model_results.grads_and_vars))[1]))
        if regularization:
            self.assertEqual(len(trainable_variables),
                             len(model_results.regularization_grads_and_vars))
            self.assertSetEqual(
                set(trainable_variables),
                set(list(zip(*model_results.regularization_grads_and_vars))[1]))
        else:
            self.assertIsNone(model_results.regularization_grads_and_vars)

        loss_total = model_results.losses['total_loss']
        if regularization == 'l1':
            loss_reg = model_results.losses['regularization_loss_l1']
        elif regularization == 'l2':
            loss_reg = model_results.losses['regularization_loss_l2']
        elif regularization == 'l1l2':
            loss_reg = (model_results.losses['regularization_loss_l1']
                        + model_results.losses['regularization_loss_l2'])
        else:
            loss_reg = None

        grads_must = tf.gradients(loss_total, trainable_variables)
        vars_to_grads_must = dict(zip(trainable_variables, grads_must))
        vars_to_grads = {v: k for k, v in model_results.grads_and_vars}

        if loss_reg is not None:
            reg_grads_must = tf.gradients(loss_reg, trainable_variables)
            reg_vars_to_grads_must = dict(
                zip(trainable_variables, reg_grads_must))
            reg_vars_to_grads = {
                v: k for k, v in model_results.regularization_grads_and_vars}
        else:
            reg_vars_to_grads_must = None
            reg_vars_to_grads = None

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            vars_to_grads_must_eval = {k: sess.run(v)
                                       for k, v in vars_to_grads_must.items()}
            vars_to_grads_eval = {k: sess.run(v)
                                  for k, v in vars_to_grads.items()}
            if loss_reg is not None:
                reg_vars_to_grads_must_eval = {
                    k: sess.run(v) for k, v in reg_vars_to_grads_must.items()}
                reg_vars_to_grads_eval = {
                    k: sess.run(v) for k, v in reg_vars_to_grads.items()}
            else:
                reg_vars_to_grads_must_eval = None
                reg_vars_to_grads_eval = None

        self.assertAllClose(vars_to_grads_must_eval,
                            vars_to_grads_eval, rtol=1e-4, atol=1e-5)
        if loss_reg is not None:
            self.assertAllClose(reg_vars_to_grads_must_eval,
                                reg_vars_to_grads_eval)

    @parameterized.parameters({'only_trainable_parameters': False},
                              {'only_trainable_parameters': True})
    def test_maybe_initialize_from_checkpoints(self, only_trainable_parameters):
        # 1. Build the model and save the graph
        temp_dir = self.get_temp_dir()
        model = ModelMock(num_classes=self.num_classes).build()
        _, inputs = model.get_test_inputs(self.batch_size, self.data_dim)
        model.mode = 'train'
        _ = model(inputs)
        saved_fname = os.path.join(temp_dir, 'model.chpt')

        global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.save(sess, saved_fname)
            vars_orig = {v.name: v.eval() for v in global_vars}

        # 2. Build the model again and restore the weights from checkpoint
        tf.reset_default_graph()
        load_config = {'only_trainable_parameters': only_trainable_parameters,
                       'checkpoint': saved_fname}

        model = ModelMock(num_classes=self.num_classes,
                          load_config=load_config).build()
        _, inputs = model.get_test_inputs(self.batch_size, self.data_dim)
        model.mode = 'train'
        model(inputs)

        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            vars_after_init = {v.name: v.eval() for v in global_vars}

        if only_trainable_parameters:
            variables_to_restore_names = [v.name for v in trainable_vars]
        else:
            variables_to_restore_names = [v.name for v in trainable_vars]
        vars_restored_must = {k: v for k, v in vars_orig.items()
                              if k in variables_to_restore_names}
        vars_after_init_restored = {k: v for k, v in vars_after_init.items()
                                    if k in variables_to_restore_names}
        self.assertAllClose(vars_restored_must, vars_after_init_restored)

    def _get_model(self, with_metric=True, inputs=None,
                   with_keras_layers=False,
                   regularization_l1=0, regularization_l2=0):
        plugins = self._get_model_plugins(with_keras_layers=with_keras_layers)
        losses = self._get_loss()
        postprocessors = self._get_postprocessors()
        summary = self._get_summary()
        metric = self._get_metric() if with_metric else None
        model = Model(plugins=plugins,
                      losses=losses,
                      postprocessors=postprocessors,
                      summaries=summary,
                      metrics=metric,
                      regularization_l1=regularization_l1,
                      regularization_l2=regularization_l2).build()
        if inputs is not None:
            dataset_nucleotide = Nucleotide(name='dataset')
            dataset_nucleotide.generated_keys = list(inputs.keys())
            model.build_dna(dataset_nucleotide)
        return model

    def _get_inputs_for_model(self):
        inputs = {'image': tf.placeholder(
            tf.float32, shape=[self.batch_size] + list(self.image_shape)),
            'labels': tf.placeholder(tf.int64)}
        return inputs

    @staticmethod
    def _get_model_plugins(with_keras_layers=False):
        plugin_cnn = DummyPluginCNN(
            name='cnn', inbound_nodes=['dataset'],
            incoming_keys_mapping={'dataset': {'image': 'inputs_cnn'}}).build()
        plugin_flatten = DummyPluginFlatten(
            name='flatten', inbound_nodes=['cnn'],
            incoming_keys_mapping={
                'cnn': {'predictions': 'inputs_flatten'}}).build()
        if with_keras_layers:
            plugin_mlp = DummyPluginMLPKeras(
                name='mlp', inbound_nodes=['flatten'],
                incoming_keys_mapping={
                    'flatten': {'predictions': 'inputs_mlp'}},
                allow_mixed_precision=False).build()
        else:
            plugin_mlp = DummyPluginMLP(
                name='mlp', inbound_nodes=['flatten'],
                incoming_keys_mapping={
                    'flatten': {'predictions': 'inputs_mlp'}},
                allow_mixed_precision=False).build()
        model_plugins = [plugin_cnn, plugin_flatten, plugin_mlp]
        return model_plugins

    @staticmethod
    def _get_loss():
        model_loss = DummySoftmaxLoss(
            inbound_nodes=['dataset', 'mlp'],
            incoming_keys_mapping={'mlp': {'predictions': 'logits'}}).build()
        return model_loss

    @staticmethod
    def _get_postprocessors():
        pp = DummyPostProcessor(
            name='pp_mlp', inbound_nodes=['mlp']).build()
        return pp

    @staticmethod
    def _get_summary():
        summary = DummySummary(
            name='summary', inbound_nodes=['dataset', 'pp_mlp'],
            incoming_keys_mapping={
                'pp_mlp': {'predictions_pp': 'predictions'}}).build()
        return summary

    @staticmethod
    def _get_metric():
        metric = DummyMetric(
            name='metric', inbound_nodes=['dataset', 'pp_mlp'],
            incoming_keys_mapping={
                'pp_mlp': {'predictions_pp': 'predictions'}}).build()
        return metric

    def _get_same_as_model_graph_from_scratch(self, inputs):
        mode = 'train'
        inp = inputs['image']
        labels = inputs['labels']
        model_plugins = self._get_model_plugins()
        for plugin in model_plugins:
            plugin.mode = mode
        outputs = {'dataset/image': inp}
        with tf.variable_scope(ScopeNames.PREPROCESSING):
            inp = tf.identity(inp)
            labels = tf.identity(labels)
        with tf.variable_scope(ScopeNames.MODEL):
            for plugin in model_plugins:
                with tf.variable_scope(plugin.name):
                    out = plugin.predict(inp)['predictions']
                    out = tf.identity(out, name="predictions")
                    outputs['/'.join([plugin.name, 'predictions'])] = out
                    inp = out

        logits = outputs['mlp/predictions']
        model_loss = self._get_loss()
        model_loss.mode = mode

        losses = {'total_loss': 0.0}
        with tf.variable_scope(ScopeNames.LOSSES):
            with tf.variable_scope(model_loss.name):
                losses_ = model_loss.process(
                    logits=logits, labels=labels)
                total_loss = 0.0
                for n, loss in losses_.items():
                    scale_factor = model_loss.loss_weights.get(n, 1.0)
                    total_loss += scale_factor * loss
                    total_loss = tf.identity(total_loss, name='total_loss')
                losses_ = {k: tf.identity(v, name=k)
                           for k, v in losses_.items()}
            losses['total_loss'] += total_loss

        model_pp = self._get_postprocessors()
        model_pp.mode = mode
        with tf.variable_scope(ScopeNames.POSTPROCESSING):
            with tf.variable_scope(model_pp.name):
                pp = model_pp.process(
                    predictions=logits)['predictions_pp']
                pp = tf.identity(pp, name="predictions_pp")

        model_summary = self._get_summary()
        model_summary.mode = mode
        with tf.variable_scope(ScopeNames.SUMMARY):
            with tf.variable_scope(model_summary.name):
                s = model_summary.process(
                    predictions=pp, labels=labels)
                s = {k: tf.identity(v, name=k)
                     for k, v in s.items()}
