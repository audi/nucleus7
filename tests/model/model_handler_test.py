# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

from typing import Callable
from unittest.mock import MagicMock
from unittest.mock import call as mock_call
from unittest.mock import patch

from absl.testing import parameterized
import numpy as np
import pytest
import tensorflow as tf

from nucleus7.model.configs import ModelResults
from nucleus7.model.fields import CollectionNames
from nucleus7.model.model_handler import ModelHandler
from nucleus7.model.plugin import ModelPlugin
from nucleus7.optimization.configs import (
    create_and_validate_optimization_config)
from nucleus7.test_utils.model_dummies import ModelMock
from nucleus7.utils import model_utils
from nucleus7.utils import nest_utils
from nucleus7.utils import tf_collections_utils


class TestModelHandler(tf.test.TestCase,
                       parameterized.TestCase):

    def setUp(self):
        tf.reset_default_graph()
        self.random_seed = 5475
        tf.set_random_seed(self.random_seed)
        np.random.seed(self.random_seed)
        self.learning_rate = 0.1
        self.num_classes = 3
        self.single_cpu_device = ['/cpu:0']
        self.multiple_cpu_gpu_devices = ['/cpu:0', '/gpu:0'] * 2
        self.batch_size = 8
        self.data_dim = 20
        self.model = ModelMock(num_classes=self.num_classes,
                               regularization_l1=0.5).build()
        self.global_optim_config = create_and_validate_optimization_config(
            optimizer_name='GradientDescentOptimizer',
            learning_rate=self.learning_rate, is_global=True)

    def test_build(self):
        model_handler = self._get_model_handler(self.single_cpu_device)
        model_handler.build()
        self.assertTrue(model_handler.optimization_handler.built)

    @pytest.mark.gpu
    @parameterized.parameters(
        {'use_multiple_devices': True, 'mode': tf.estimator.ModeKeys.TRAIN},
        {'use_multiple_devices': True, 'mode': tf.estimator.ModeKeys.EVAL},
        {'use_multiple_devices': True, 'mode': tf.estimator.ModeKeys.PREDICT},
        {'use_multiple_devices': False, 'mode': tf.estimator.ModeKeys.TRAIN},
        {'use_multiple_devices': False, 'mode': tf.estimator.ModeKeys.EVAL},
        {'use_multiple_devices': False, 'mode': tf.estimator.ModeKeys.PREDICT})
    def test_model_fn_integration(self, use_multiple_devices, mode):
        devices = (self.multiple_cpu_gpu_devices if use_multiple_devices
                   else self.single_cpu_device)
        model_handler = self._get_model_handler(devices)
        model_handler.build()
        inputs_np, inputs_tf = self.model.get_test_inputs(
            self.batch_size, self.data_dim)
        estimator_spec = model_handler.model_fn(
            features=inputs_tf, labels=None, mode=mode)
        self.assertIsInstance(estimator_spec, tf.estimator.EstimatorSpec)

    @parameterized.parameters({"mode": tf.estimator.ModeKeys.TRAIN},
                              {"mode": tf.estimator.ModeKeys.EVAL},
                              {"mode": tf.estimator.ModeKeys.PREDICT})
    def test_model_fn(self, mode):

        model_results_combined = ModelResults(
            inputs_preprocessed="inputs_preprocessed_comb",
            predictions="predictions_comb",
            predictions_raw=None,
            losses="losses_comb",
            summary="summary_comb",
            metrics="metrics_comb",
            grads_and_vars="grads_and_vars_comb",
            regularization_grads_and_vars="reg_grads_and_vars_comb")

        def _replicate_model_over_devices(features, mode):
            return ["model_results_{}".format(i) for i in range(5)]

        def _combine_model_results_from_devices(tower_model_results):
            return model_results_combined

        def _add_to_collections(model_results):
            return None

        def _add_summaries(model_results, mode):
            return None

        def _get_estimator_spec(mode, predictions, losses, metrics, train_op):
            return "estimator_spec"

        def _get_train_op(model_results):
            return "train_op"

        model_handler = self._get_model_handler([])
        model_handler.build()
        inputs = {'input1': 10, 'input2': 20}

        model_handler.replicate_model_over_devices = MagicMock(
            side_effect=_replicate_model_over_devices)
        model_handler.combine_model_results_from_devices = MagicMock(
            side_effect=_combine_model_results_from_devices)
        model_handler.add_to_collections = MagicMock(
            side_effect=_add_to_collections)
        model_handler.add_summaries = MagicMock(
            side_effect=_add_summaries)
        model_handler.get_estimator_spec = MagicMock(
            side_effect=_get_estimator_spec)
        model_handler.add_optim_configs_to_handler = MagicMock(
            regurn_value=None)
        model_handler.get_train_op = MagicMock(side_effect=_get_train_op)

        estimator_spec = model_handler.model_fn(
            features=inputs, labels=None, mode=mode)
        model_results_with_inputs = model_results_combined._replace(
            inputs_preprocessed=inputs)

        self.assertEqual(mode,
                         model_handler.model.mode)
        mh = model_handler
        mh.replicate_model_over_devices.assert_called_once_with(
            inputs, mode=mode)
        mh.combine_model_results_from_devices.assert_called_once_with(
            ["model_results_{}".format(i) for i in range(5)])
        if mode == tf.estimator.ModeKeys.TRAIN:
            mh.add_optim_configs_to_handler.assert_called_once_with()
            mh.get_train_op.assert_called_once_with(
                model_results_with_inputs)
        else:
            mh.add_optim_configs_to_handler.assert_not_called()
            mh.get_train_op.assert_not_called()
        if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
            mh.add_summaries.assert_called_once_with(
                model_results_with_inputs, mode)
        else:
            mh.add_summaries.assert_not_called()
        mh.add_to_collections.assert_called_once_with(
            model_results_with_inputs)
        train_op_must = ('train_op' if mode == tf.estimator.ModeKeys.TRAIN
                         else None)
        mh.get_estimator_spec.assert_called_once_with(
            mode=mode,
            predictions=model_results_with_inputs.predictions,
            losses=model_results_with_inputs.losses,
            metrics=model_results_with_inputs.metrics,
            train_op=train_op_must)

        self.assertEqual("estimator_spec",
                         estimator_spec)

    @parameterized.parameters(
        {'use_multiple_devices': True, 'mode': tf.estimator.ModeKeys.TRAIN},
        {'use_multiple_devices': True, 'mode': tf.estimator.ModeKeys.EVAL},
        {'use_multiple_devices': True, 'mode': tf.estimator.ModeKeys.PREDICT},
        {'use_multiple_devices': False, 'mode': tf.estimator.ModeKeys.TRAIN},
        {'use_multiple_devices': False, 'mode': tf.estimator.ModeKeys.EVAL},
        {'use_multiple_devices': False, 'mode': tf.estimator.ModeKeys.PREDICT})
    @patch("nucleus7.model.model.Model.__call__")
    def test_replicate_model_over_devices(self, model_call, mode,
                                          use_multiple_devices):
        def _model_call(inputs_from_dataset):
            _model_call.i = getattr(_model_call, 'i', 0)
            _model_call.i += 1
            return "model_results_{}".format(_model_call.i)

        devices = (self.multiple_cpu_gpu_devices if use_multiple_devices
                   else self.single_cpu_device)
        model_handler = self._get_model_handler(devices)
        model_handler.build()
        model_handler.model.mode = mode
        _, inputs_tf = self.model.get_test_inputs(
            self.batch_size, self.data_dim)
        self.model.reset_tf_graph = MagicMock(return_value=None)

        model_call.side_effect = _model_call
        tower_model_results = model_handler.replicate_model_over_devices(
            inputs_tf, mode)

        num_devices_must = (
            len(devices) if mode != tf.estimator.ModeKeys.PREDICT else 1)
        self.assertEqual(1,
                         self.model.reset_tf_graph.call_count)
        self.assertEqual(num_devices_must,
                         model_call.call_count)
        self.assertEqual(
            ["model_results_{}".format(i + 1) for i in range(num_devices_must)],
            tower_model_results)

        model_call_inputs = [
            each_kwargs['inputs_from_dataset']
            for _, each_kwargs in model_call.call_args_list]
        inputs_must = model_utils.split_inputs_to_devices(
            inputs_tf, num_devices_must)
        with tf.Session() as sess:
            inputs_eval_must = sess.run(inputs_must)
            model_call_inputs_eval = sess.run(model_call_inputs)
        self.assertAllClose(inputs_eval_must,
                            model_call_inputs_eval)

    @parameterized.parameters({'use_multiple_devices': True},
                              {'use_multiple_devices': False})
    @patch("nucleus7.utils.optimization_utils."
           "average_grads_and_vars_from_multiple_devices")
    @patch("nucleus7.utils.model_utils.combine_metrics_from_devices")
    @patch("nucleus7.utils.model_utils.combine_summary_from_devices")
    @patch("nucleus7.utils.model_utils.combine_losses_from_devices")
    @patch("nucleus7.utils.model_utils.combine_predictions_from_devices")
    def test_replicate_model_over_devices(
            self, combine_predictions_from_devices,
            combine_losses_from_devices,
            combine_summary_from_devices,
            combine_metrics_from_devices,
            average_grads_and_vars_from_multiple_devices,
            use_multiple_devices):

        def _combine_predictions_from_devices(**kwargs):
            return "predictions_combined"

        def _combine_summary_from_devices(**kwargs):
            return "summary_combined"

        def _combine_metrics_from_devices(**kwargs):
            return "metrics_combined"

        def _combine_losses_from_devices(**kwargs):
            return "losses_combined"

        def _average_grads_and_vars_from_multiple_devices(**kwargs):
            return "grads_and_vars_averaged"

        combine_predictions_from_devices.side_effect = (
            _combine_predictions_from_devices)
        combine_losses_from_devices.side_effect = (
            _combine_losses_from_devices)
        combine_summary_from_devices.side_effect = (
            _combine_summary_from_devices)
        combine_metrics_from_devices.side_effect = (
            _combine_metrics_from_devices)
        average_grads_and_vars_from_multiple_devices.side_effect = (
            _average_grads_and_vars_from_multiple_devices)

        devices = (self.multiple_cpu_gpu_devices if use_multiple_devices
                   else self.single_cpu_device)
        model_handler = self._get_model_handler(devices)
        model_handler.build()
        tower_model_results = [
            ModelResults(
                predictions_raw="predictions_raw_{}".format(i),
                predictions="predictions_{}".format(i),
                losses="losses_{}".format(i),
                grads_and_vars="grads_and_vars_{}".format(i),
                regularization_grads_and_vars="reg_grads_and_vars_{}".format(i),
                metrics="metrics_{}".format(i),
                summary="summary_{}".format(i),
                inputs_preprocessed=None
            )
            for i in range(len(devices))
        ]

        model_results_combined = (
            model_handler.combine_model_results_from_devices(
                tower_model_results))
        if use_multiple_devices:
            model_results_combined_must = ModelResults(
                inputs_preprocessed=None,
                predictions_raw="predictions_combined",
                predictions="predictions_combined",
                grads_and_vars="grads_and_vars_averaged",
                regularization_grads_and_vars="reg_grads_and_vars_0",
                summary="summary_combined",
                metrics="metrics_combined",
                losses="losses_combined"
            )
        else:
            model_results_combined_must = ModelResults(
                inputs_preprocessed=None,
                predictions_raw="predictions_raw_0",
                predictions="predictions_0",
                grads_and_vars="grads_and_vars_0",
                regularization_grads_and_vars="reg_grads_and_vars_0",
                summary="summary_0",
                metrics="metrics_0",
                losses="losses_0"
            )
        self.assertEqual(model_results_combined_must,
                         model_results_combined)

        if use_multiple_devices:
            combine_predictions_from_devices.assert_called_with(
                predictions_devices=["predictions_{}".format(i)
                                     for i in range(len(devices))],
                predictions_have_variable_shape=
                model_handler.predictions_have_variable_shape
            )
            combine_losses_from_devices.assert_called_with(
                losses_devices=["losses_{}".format(i)
                                for i in range(len(devices))])
            combine_summary_from_devices.assert_called_with(
                summary_devices=["summary_{}".format(i)
                                 for i in range(len(devices))])
            combine_metrics_from_devices.assert_called_with(
                metrics_devices=["metrics_{}".format(i)
                                 for i in range(len(devices))])
            average_grads_and_vars_from_multiple_devices.assert_called_with(
                tower_grads_and_vars=["grads_and_vars_{}".format(i)
                                      for i in range(len(devices))],
                consolidation_device='/cpu:0')
        else:
            combine_predictions_from_devices.assert_not_called()
            combine_losses_from_devices.assert_not_called()
            combine_summary_from_devices.assert_not_called()
            combine_metrics_from_devices.assert_not_called()
            average_grads_and_vars_from_multiple_devices.assert_not_called()

    @pytest.mark.gpu
    @parameterized.parameters(
        {'use_multiple_devices': True, 'mode': tf.estimator.ModeKeys.TRAIN},
        {'use_multiple_devices': True, 'mode': tf.estimator.ModeKeys.EVAL},
        {'use_multiple_devices': True, 'mode': tf.estimator.ModeKeys.PREDICT},
        {'use_multiple_devices': False, 'mode': tf.estimator.ModeKeys.TRAIN},
        {'use_multiple_devices': False, 'mode': tf.estimator.ModeKeys.EVAL},
        {'use_multiple_devices': False, 'mode': tf.estimator.ModeKeys.PREDICT})
    def test_replicate_model_over_devices_integration(
            self, mode, use_multiple_devices):
        devices = (self.multiple_cpu_gpu_devices if use_multiple_devices
                   else self.single_cpu_device)
        model_handler = self._get_model_handler(devices)
        model_handler.build()
        model_handler.model.mode = mode

        inputs_np, inputs_tf = self.model.get_test_inputs(
            self.batch_size, self.data_dim)
        tower_model_results = model_handler.replicate_model_over_devices(
            inputs_tf, mode)
        self.assertEqual(len(devices) if mode != tf.estimator.ModeKeys.PREDICT
                         else 1,
                         len(tower_model_results))
        all_trainable_variables_set = set(tf.trainable_variables())
        self.assertSetEqual(set(self.model.trainable_vars),
                            all_trainable_variables_set)
        model_results_keys = [
            {k for k, v in model_results._asdict().items() if v is not None}
            for model_results in tower_model_results]

        if use_multiple_devices:
            first_tower_keys = model_results_keys[0]
            for each_other_tower_keys in model_results_keys[1:]:
                self.assertSetEqual(first_tower_keys,
                                    each_other_tower_keys)

    @pytest.mark.gpu
    @parameterized.parameters(
        {'use_multiple_devices': True, 'mode': tf.estimator.ModeKeys.TRAIN},
        {'use_multiple_devices': True, 'mode': tf.estimator.ModeKeys.EVAL},
        {'use_multiple_devices': True, 'mode': tf.estimator.ModeKeys.PREDICT},
        {'use_multiple_devices': False, 'mode': tf.estimator.ModeKeys.TRAIN},
        {'use_multiple_devices': False, 'mode': tf.estimator.ModeKeys.EVAL},
        {'use_multiple_devices': False, 'mode': tf.estimator.ModeKeys.PREDICT})
    def test_combine_model_results_from_devices_integration(
            self, mode, use_multiple_devices):
        devices = (self.multiple_cpu_gpu_devices if use_multiple_devices
                   else self.single_cpu_device)
        model_handler = self._get_model_handler(devices)
        model_handler.build()
        model_handler.model.mode = mode
        _, inputs_tf = self.model.get_test_inputs(
            self.batch_size, self.data_dim)
        tower_model_results = model_handler.replicate_model_over_devices(
            inputs_tf, mode)
        model_results = model_handler.combine_model_results_from_devices(
            tower_model_results)
        model_results_must = self._get_model_results_on_all_data()
        for i, (each_result_must, each_result) in enumerate(
                zip(model_results_must, model_results)):
            if i == 0:  # remove inputs, since they must be not combined
                continue
            if each_result_must is None:
                self.assertIsNone(each_result)
            else:
                self.assertIsNotNone(each_result)

        if mode != tf.estimator.ModeKeys.PREDICT:
            self.assertDictEqual(
                _get_nested_shapes(model_results_must.predictions_raw),
                _get_nested_shapes(model_results.predictions_raw))
            self.assertDictEqual(
                _get_nested_shapes(model_results_must.losses),
                _get_nested_shapes(model_results.losses))
        if mode == tf.estimator.ModeKeys.TRAIN:
            self.assertSetEqual(
                set(_get_nested_shapes_for_grads_and_vars(
                    model_results_must.grads_and_vars)),
                set(_get_nested_shapes_for_grads_and_vars(
                    model_results.grads_and_vars)))
            self.assertSetEqual(
                set(_get_nested_shapes_for_grads_and_vars(
                    model_results_must.regularization_grads_and_vars)),
                set(_get_nested_shapes_for_grads_and_vars(
                    model_results.regularization_grads_and_vars)))
        else:
            self.assertDictEqual(
                _get_nested_shapes(model_results_must.predictions),
                _get_nested_shapes(model_results.predictions))
        if mode == tf.estimator.ModeKeys.EVAL:
            self.assertDictEqual(
                _get_nested_shapes(model_results_must.summary),
                _get_nested_shapes(model_results.summary))
            self.assertDictEqual(
                _get_nested_shapes(model_results_must.metrics),
                _get_nested_shapes(model_results.metrics))

    def test_add_optim_configs_to_handler(self):

        def get_optimization_configs_with_variables_config2():
            return [('config1', 'vars1'),
                    ('config2', 'vars2')]

        def get_optimization_configs_with_variables_config3():
            return [('config3', 'vars3')]

        def _add_config_with_variables(config_with_vars, name):
            return None

        model_handler = self._get_model_handler([])
        model_handler.build()
        optimization_handler = model_handler.optimization_handler
        plugin1 = ModelPlugin(name='plugin1')
        plugin2 = ModelPlugin(name='plugin2')
        plugin3 = ModelPlugin(name='plugin3')
        plugin1.get_optimization_configs_with_variables = MagicMock(
            return_value=None)
        plugin2.get_optimization_configs_with_variables = MagicMock(
            side_effect=get_optimization_configs_with_variables_config2)
        plugin3.get_optimization_configs_with_variables = MagicMock(
            side_effect=get_optimization_configs_with_variables_config3)
        optimization_handler.initialize_for_session = MagicMock(
            return_value=None)
        optimization_handler.add_config_with_variables = (
            MagicMock(side_effect=_add_config_with_variables))
        self.model.plugins = {'plugin1': plugin1,
                              'plugin2': plugin2,
                              'plugin3': plugin3}
        model_handler.add_optim_configs_to_handler()
        pl1 = plugin1
        pl2 = plugin2
        pl3 = plugin3
        pl1.get_optimization_configs_with_variables.assert_called_once_with()
        pl2.get_optimization_configs_with_variables.assert_called_once_with()
        pl3.get_optimization_configs_with_variables.assert_called_once_with()
        optimization_handler.initialize_for_session.assert_called_once_with()
        add_config_calls_must = [
            mock_call(('config1', 'vars1'), name="plugin2"),
            mock_call(('config2', 'vars2'), name="plugin2"),
            mock_call(('config3', 'vars3'), name="plugin3")]
        optimization_handler.add_config_with_variables.assert_has_calls(
            add_config_calls_must)

    def test_get_train_op(self):
        model_results = ModelResults(
            inputs_preprocessed=None,
            predictions=None,
            predictions_raw=None,
            losses=None,
            summary=None,
            metrics=None,
            grads_and_vars="grads_and_vars",
            regularization_grads_and_vars="regularization_grads_and_vars"
        )
        model_handler = self._get_model_handler([])
        model_handler.build()
        optimization_handler = model_handler.optimization_handler
        optimization_handler.get_train_op = MagicMock(
            return_value=tf.no_op(name='train_op_opt'))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,
                             tf.no_op(name='update_op'))
        train_op = model_handler.get_train_op(model_results)
        optimization_handler.get_train_op.assert_called_once_with(
            model_results.grads_and_vars,
            model_results.regularization_grads_and_vars,
            trainable_variables=self.model.trainable_variables)
        self.assertIsInstance(train_op, tf.Operation)
        train_op_input_ops_names = [op.name for op in train_op.control_inputs]
        self.assertSetEqual({'train_op_opt', 'update_op'},
                            set(train_op_input_ops_names))

    @parameterized.parameters(
        {'with_loss': True, 'with_predictions': True, 'with_metrics': True},
        {'with_loss': True, 'with_predictions': True, 'with_metrics': False},
        {'with_loss': False, 'with_predictions': True, 'with_metrics': True},
        {'with_loss': False, 'with_predictions': False, 'with_metrics': False})
    def test_get_estimator_spec(self, with_loss, with_predictions,
                                with_metrics):
        mode = 'mode'
        losses = None
        predictions = None
        metrics = None
        train_op = tf.no_op('train_op')
        if with_loss:
            losses = {'total_loss': tf.constant('total_loss_value')}
        if with_predictions:
            predictions = {
                'pp1': {"prediction11": tf.constant("prediction11_value"),
                        "prediction12": tf.constant("prediction12_value")},
                'pp2': {"prediction21": tf.constant("prediction21_value")}}
        if with_metrics:
            metrics = {
                'metric1': {'accuracy11': tf.constant("accuracy11_value"),
                            'accuracy12': tf.constant("accuracy12_value")},
                'metric2': {'accuracy21': tf.constant("accuracy21_value")}}
        model_handler = self._get_model_handler([])
        model_handler.build()
        estimator_spec = model_handler.get_estimator_spec(
            mode, predictions, losses, metrics, train_op)
        self.assertEqual(mode,
                         estimator_spec.mode)
        self.assertEqual(train_op,
                         estimator_spec.train_op)
        if with_loss:
            self.assertEqual(losses["total_loss"],
                             estimator_spec.loss)
        else:
            self.assertIsNone(estimator_spec.loss)
        if with_predictions:
            predictions_must = {
                'pp1//prediction11': predictions["pp1"]["prediction11"],
                'pp1//prediction12': predictions["pp1"]["prediction12"],
                'pp2//prediction21': predictions["pp2"]["prediction21"]}
            self.assertDictEqual(predictions_must,
                                 estimator_spec.predictions)
        else:
            self.assertEmpty(estimator_spec.predictions)
        if with_metrics:
            eval_metric_ops_must = {
                'metric1//accuracy11': (metrics["metric1"]["accuracy11"],
                                        tf.no_op()),
                'metric1//accuracy12': (metrics["metric1"]["accuracy12"],
                                        tf.no_op()),
                'metric2//accuracy21': (metrics["metric2"]["accuracy21"],
                                        tf.no_op())
            }
            self.assertSetEqual(set(eval_metric_ops_must.keys()),
                                set(estimator_spec.eval_metric_ops))
            for each_key in eval_metric_ops_must:
                value_must, _ = eval_metric_ops_must[each_key]
                value, update_op = estimator_spec.eval_metric_ops[each_key]
                self.assertEqual("NoOp",
                                 update_op.type)
                self.assertEqual(value_must,
                                 value)
        else:
            self.assertEmpty(estimator_spec.eval_metric_ops)

    @parameterized.parameters(
        {'with_losses': True, 'with_predictions': True, 'with_metrics': True,
         'with_predictions_raw': True, "with_summaries": True},
        {'with_losses': True, 'with_predictions': True, 'with_metrics': False,
         'with_predictions_raw': False, "with_summaries": False},
        {'with_losses': False, 'with_predictions': True, 'with_metrics': True,
         'with_predictions_raw': False, "with_summaries": True},
        {'with_losses': False, 'with_predictions': False, 'with_metrics': False,
         'with_predictions_raw': False, "with_summaries": False})
    def test_add_to_collections(self, with_losses, with_predictions_raw,
                                with_predictions, with_summaries,
                                with_metrics):
        inputs_preprocessed = {"input1": {"input11": "input11_value",
                                          "input12": "input12_value"},
                               "label1": {"label11": "label11_value",
                                          "label12": "label12_value"}}
        predictions_raw = None
        predictions = None
        losses = None
        summary = None
        metrics = None
        if with_losses:
            losses = {'total_loss': 'total_loss_value',
                      'loss1': {'sub_loss11': 'loss11_value'}}
        if with_predictions:
            predictions = {
                'pp1': {"prediction11": tf.constant("prediction11_value"),
                        "prediction12": tf.constant("prediction12_value")},
                'pp2': {"prediction21": tf.constant("prediction21_value")}}
        if with_predictions_raw:
            predictions_raw = {
                'plugin1': {"raw11": tf.constant("raw11_value"),
                            "raw12": tf.constant("raw12_value")},
                'plugin2': {"raw21": tf.constant("raw21_value")}}
        if with_summaries:
            summary = {
                'summary1': {'subsummary11': tf.constant("summary11_value"),
                             'subsummary12': tf.constant("summary12_value")},
                'summary2': {'subsummary21': tf.constant("summary21_value")}}
        if with_metrics:
            metrics = {
                'metric1': {'accuracy11': tf.constant("accuracy11_value"),
                            'accuracy12': tf.constant("accuracy12_value")},
                'metric2': {'accuracy21': tf.constant("accuracy21_value")}}

        model_results = ModelResults(
            inputs_preprocessed=inputs_preprocessed,
            predictions_raw=predictions_raw,
            predictions=predictions,
            losses=losses,
            summary=summary,
            metrics=metrics,
            grads_and_vars=None,
            regularization_grads_and_vars=None)
        model_handler = self._get_model_handler([])
        model_handler.build()
        model_handler.add_to_collections(model_results)
        self.assertDictEqual(inputs_preprocessed,
                             tf_collections_utils.collection2nested(
                                 CollectionNames.INPUTS_PREPROCESSED))
        if with_predictions:
            self.assertDictEqual(predictions,
                                 tf_collections_utils.collection2nested(
                                     CollectionNames.PREDICTIONS))
        else:
            with self.assertRaises(ValueError):
                tf_collections_utils.collection2nested(
                    CollectionNames.PREDICTIONS)
        if with_predictions_raw:
            self.assertDictEqual(predictions_raw,
                                 tf_collections_utils.collection2nested(
                                     CollectionNames.PREDICTIONS_RAW))
        else:
            with self.assertRaises(ValueError):
                tf_collections_utils.collection2nested(
                    CollectionNames.PREDICTIONS_RAW)
        if with_metrics:
            self.assertDictEqual(metrics,
                                 tf_collections_utils.collection2nested(
                                     CollectionNames.METRIC))
        else:
            with self.assertRaises(ValueError):
                tf_collections_utils.collection2nested(CollectionNames.METRIC)
        if with_summaries:
            self.assertDictEqual(summary,
                                 tf_collections_utils.collection2nested(
                                     CollectionNames.SUMMARY))
        else:
            with self.assertRaises(ValueError):
                tf_collections_utils.collection2nested(CollectionNames.SUMMARY)

    @parameterized.parameters(
        {'with_losses': True, 'with_metrics': True, "with_summaries": True,
         'with_grads_and_vars': True, 'with_reg_grads_and_vars': True},
        {'with_losses': True, 'with_metrics': False, "with_summaries": False,
         'with_grads_and_vars': False, 'with_reg_grads_and_vars': True},
        {'with_losses': False, 'with_metrics': True, "with_summaries": True,
         'with_grads_and_vars': True, 'with_reg_grads_and_vars': False},
        {'with_losses': False, 'with_metrics': False, "with_summaries": False,
         'with_grads_and_vars': False, 'with_reg_grads_and_vars': False})
    @patch("nucleus7.utils.model_utils.add_summary_by_name")
    @patch("nucleus7.utils.model_utils.add_histogram_summary")
    @patch("tensorflow.summary.scalar")
    @patch.object(tf, "norm")
    @patch.object(tf, "global_norm")
    def test_add_summaries(self, tf_global_norm, tf_norm,
                           tf_summary_scalar, add_histogram_summary,
                           add_summary_by_name,
                           with_losses, with_summaries, with_metrics,
                           with_grads_and_vars, with_reg_grads_and_vars):
        def _norm(inp):
            return "_".join(["norm", inp])

        def _global_norm(inputs):
            return "_".join(["norm", *inputs])

        tf_summary_scalar.return_value = None
        add_histogram_summary.return_value = None
        add_summary_by_name.return_value = None
        tf_norm.side_effect = _norm
        tf_global_norm.side_effect = _global_norm

        losses = None
        summary = None
        metrics = None
        grads_and_vars = None
        reg_grads_and_vars = None
        variables = [tf.Variable(0, name="var_{}".format(i)) for i in range(5)]
        if with_losses:
            losses = {'total_loss': 'total_loss_value',
                      'loss1': {'sub_loss11': 'loss11_value'}}
        if with_summaries:
            summary = {
                'summary1': {'subsummary11': "summary11_value",
                             'subsummary12': "summary12_value"},
                'summary2': {'subsummary21': "summary21_value"}}
        if with_metrics:
            metrics = {
                'metric1': {'accuracy11': "accuracy11_value",
                            'accuracy12': "accuracy12_value"},
                'metric2': {'accuracy21': "accuracy21_value"}}
        if with_grads_and_vars:
            grads_and_vars = [('grad_{}'.format(i), variables[i])
                              for i in range(5)]
        if with_reg_grads_and_vars:
            reg_grads_and_vars = [('reg_grad_{}'.format(i), variables[i])
                                  for i in range(3)]

        model_results = ModelResults(
            inputs_preprocessed="not_used",
            predictions_raw="not_used",
            predictions="not_used",
            losses=losses,
            summary=summary,
            metrics=metrics,
            grads_and_vars=grads_and_vars,
            regularization_grads_and_vars=reg_grads_and_vars)

        model_handler = self._get_model_handler([])
        model_handler.build()
        model_handler.optimization_handler._global_learning_rate = (
            "global_learning_rate")
        model_handler.add_summaries(
            model_results, mode=tf.estimator.ModeKeys.TRAIN)

        if with_grads_and_vars:
            for i, var in enumerate(variables):
                call_must = mock_call(
                    "gradient/{}".format(var.name).replace(':', '_'),
                    "grad_{}".format(i))
                add_histogram_summary.assert_has_calls([call_must])
        if with_reg_grads_and_vars:
            for i, var in enumerate(variables[:3]):
                call_must = mock_call(
                    "reg_gradient/{}".format(var.name).replace(':', '_'),
                    "reg_grad_{}".format(i))
                add_histogram_summary.assert_has_calls([call_must])

        if not with_grads_and_vars and not with_reg_grads_and_vars:
            add_histogram_summary.assert_not_called()

        if with_losses:
            tf_summary_scalar.assert_has_calls(
                [mock_call("total_loss", "total_loss_value", family="loss")])
            tf_summary_scalar.assert_has_calls(
                [mock_call("loss1//sub_loss11", "loss11_value", family="loss")])
        tf_summary_scalar.assert_has_calls(
            [mock_call("learning_rate", "global_learning_rate")])

        max_outputs_tb = model_handler.max_outputs_tb
        if with_metrics:
            add_summary_by_name.assert_has_calls(
                [mock_call("metric1//accuracy11", "accuracy11_value",
                           max_outputs_tb)])
            add_summary_by_name.assert_has_calls(
                [mock_call("metric1//accuracy12", "accuracy12_value",
                           max_outputs_tb)])
            add_summary_by_name.assert_has_calls(
                [mock_call("metric2//accuracy21", "accuracy21_value",
                           max_outputs_tb)])
        if with_summaries:
            add_summary_by_name.assert_has_calls(
                [mock_call("summary1//subsummary11", "summary11_value",
                           max_outputs_tb)])
            add_summary_by_name.assert_has_calls(
                [mock_call("summary1//subsummary12", "summary12_value",
                           max_outputs_tb)])
            add_summary_by_name.assert_has_calls(
                [mock_call("summary2//subsummary21", "summary21_value",
                           max_outputs_tb)])
        if not with_metrics and not with_summaries:
            add_summary_by_name.assert_not_called()

    def _get_model_handler(self, devices):
        model_handler = ModelHandler(
            model=self.model, devices=devices,
            global_optimization_config=self.global_optim_config,
            variable_strategy='CPU')
        return model_handler

    def _get_model_results_on_all_data(self):
        tf.reset_default_graph()
        tf.set_random_seed(self.random_seed)
        self.model.reset_tf_graph()
        _, inputs = self.model.get_test_inputs(self.batch_size, self.data_dim)
        model_results = self.model(inputs)
        return model_results


def _get_nested_shapes(nested_dict):
    flatten = nest_utils.flatten_nested_struct(nested_dict)
    flatten_with_shapes = {k: v.get_shape().as_list()
                           for k, v in flatten.items()}
    nested_with_shapes = nest_utils.unflatten_dict_to_nested(
        flatten_with_shapes)
    return nested_with_shapes


def _get_nested_shapes_for_grads_and_vars(grads_and_vars):
    return [(tuple(g.get_shape().as_list()),
             tuple(v.get_shape().as_list()))
            for g, v in grads_and_vars]


def _model_results_replace_none_with(model_results: ModelResults,
                                     value_fn: Callable = tf.no_op
                                     ) -> ModelResults:
    for each_field in model_results._fields:
        if getattr(model_results, each_field) is None:
            model_results = model_results._replace(
                **{each_field: value_fn()})
    return model_results
