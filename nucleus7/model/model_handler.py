# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Class for handling the model, e.g. replicate the model over multiple GPUs,
create train_op, combine the results, add summaries etc.
"""
import logging
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import tensorflow as tf

from nucleus7.core.base import BaseClass
from nucleus7.model.configs import MixedPrecisionConfig
from nucleus7.model.configs import ModelResults
from nucleus7.model.fields import CollectionNames
from nucleus7.model.fields import ScopeNames
from nucleus7.model.model import Model
from nucleus7.optimization.configs import OptimizationConfig
from nucleus7.optimization.optimization_handler import OptimizationHandler
from nucleus7.third_party.cifar10_utils import local_device_setter
from nucleus7.third_party.nvidia_mixed_precision import (
    float32_variable_storage_getter)
from nucleus7.utils import model_utils
from nucleus7.utils import nest_utils
from nucleus7.utils import optimization_utils as opt_utils
from nucleus7.utils import tf_collections_utils

# pylint: disable=invalid-name
# this is a type constant, not a class
_MODEL_FN_SIGNATURE = Callable[[dict, Any, str, Any, tf.estimator.RunConfig],
                               tf.estimator.EstimatorSpec]


# pylint: enable=invalid-name


class ModelHandler(BaseClass):
    """
    Build model_fn that will be used inside of estimator

    Parameters
    ----------
    model
        instance of model
    devices
        list of devices to use for the model; inputs are all always fetched
        and augmented on cpu once for all gpus
    variable_strategy
        CPU to use CPU as the parameter server
    global_optimization_config
        global optimization configuration
    max_outputs_tb
        number of maximum outputs in tensorboard e.g. for images
    predictions_have_variable_shape
        if the predictions from different devices have different shapes
        from batch to batch and so should be concatenated
        with padding during evaluation; useful for object detections

    """

    # pylint: disable=too-many-arguments
    # not possible to have less arguments without more complexity
    def __init__(self,
                 model: Model,
                 devices: list,
                 global_optimization_config: OptimizationConfig,
                 variable_strategy: str = 'GPU',
                 predictions_have_variable_shape: bool = False,
                 max_outputs_tb: int = 1):
        super(ModelHandler, self).__init__()
        logger = logging.getLogger(__name__)
        if 'cpu' in ' '.join(devices) and variable_strategy == 'GPU':
            msg = ("You cannot use GPU variable strategy together with cpu "
                   "device! Strategy will be changed to CPU")
            logger.warning(msg)
        self.model = model
        self.devices = devices
        self.variable_strategy = variable_strategy
        self.predictions_have_variable_shape = predictions_have_variable_shape
        self.max_outputs_tb = max_outputs_tb
        self.global_optimization_config = global_optimization_config
        self.optimization_handler = OptimizationHandler()

    def build(self):
        super(ModelHandler, self).build()
        self.optimization_handler.build()
        return self

    def get_model_fn(self) -> _MODEL_FN_SIGNATURE:
        """
        Returns
        -------
        model_fn
            model_fn with (features, labels, mode, config=None) as attributes;
            inputs and labels are inside of features, and labels attribute is
            there for signature convenience
        """
        return self.model_fn

    # pylint: disable=unused-argument
    # this is API for model_fn
    def model_fn(self, features: dict, labels, mode: str,
                 config: Optional[tf.estimator.RunConfig] = None
                 ) -> tf.estimator.EstimatorSpec:
        """
        Construct the model:
            * replicate over devices
            * combine results
            * get train operation
            * add summaries
            * and collections

        This method must be used inside of the :obj:`tf.estimator.Estimator`.

        Parameters
        ----------
        features
            inputs and labels
        labels
            not used
        mode
            mode of the model
        config
            not used
        params
            not used

        Returns
        -------
        estimator_spec
            estimator spec
        """
        self.model.mode = mode
        train_op = None

        tower_model_results = self.replicate_model_over_devices(
            features, mode=mode)
        model_results = self.combine_model_results_from_devices(
            tower_model_results)
        model_results = model_results._replace(inputs_preprocessed=features)
        if mode == tf.estimator.ModeKeys.TRAIN:
            self.add_optim_configs_to_handler()
            train_op = self.get_train_op(model_results)
        if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
            self.add_summaries(model_results, mode)

        self.add_to_collections(model_results)
        estimator_spec = self.get_estimator_spec(
            mode=mode, predictions=model_results.predictions,
            losses=model_results.losses, metrics=model_results.metrics,
            train_op=train_op)
        return estimator_spec

    # pylint: enable=unused-argument

    def replicate_model_over_devices(self, features: dict, mode: str):
        """
        Replicate the model over all devices. For PREDICT mode, will use only 1
        device

        Parameters
        ----------
        features
            inputs and labels
        mode
            mode of of the model

        Returns
        -------
        tower_model_results
            list of ModelResults from each device
        """
        logger = logging.getLogger(__name__)
        number_of_devices = self._get_number_of_devices_for_mode(mode)
        custom_vars_getter = _get_custom_vars_getter(
            self.model.mixed_precision_config)
        worker_inputs = model_utils.split_inputs_to_devices(
            features, number_of_devices)
        tower_model_results = []

        self.model.reset_tf_graph()
        with tf.variable_scope(tf.get_variable_scope(),
                               custom_getter=custom_vars_getter):
            for worker_device, worker_input in zip(self.devices, worker_inputs):
                logger.info('cloning on device %s', worker_device)
                device_setter = _get_device_setter(
                    worker_device, self.variable_strategy, len(self.devices))
                with tf.device(device_setter):
                    model_result = self.model(inputs_from_dataset=worker_input)
                tower_model_results.append(model_result)
                tf.get_variable_scope().reuse_variables()
        return tower_model_results

    def combine_model_results_from_devices(
            self, tower_model_results: List[ModelResults]) -> ModelResults:
        """
        Combine model results fro all devices

        Parameters
        ----------
        tower_model_results
            list of results for each device

        Returns
        -------
        model_results_combined
            combined model results
        """
        if len(tower_model_results) == 1:
            return tower_model_results[0]

        predictions_raw, predictions = self._combine_predictions(
            tower_model_results)
        losses = self._combine_losses(tower_model_results)
        summary = self._combine_summary(tower_model_results)
        metrics = self._combine_metrics(tower_model_results)

        grads_and_vars = self._combine_grads_and_vars(tower_model_results)
        regularization_grads_and_vars = (
            tower_model_results[0].regularization_grads_and_vars)

        model_results_combined = ModelResults(
            inputs_preprocessed=None,
            predictions_raw=predictions_raw, predictions=predictions,
            losses=losses, summary=summary, metrics=metrics,
            grads_and_vars=grads_and_vars,
            regularization_grads_and_vars=regularization_grads_and_vars)
        return model_results_combined

    def add_optim_configs_to_handler(self):
        """
        Add local optimization configs from plugins to the optimization handler
        """
        self.optimization_handler.global_config = (
            self.global_optimization_config)
        for each_plugin_name, each_plugin in sorted(self.model.plugins.items()):
            plugin_config_with_vars = (
                each_plugin.get_optimization_configs_with_variables())
            if plugin_config_with_vars is not None:
                for each_config, each_vars in plugin_config_with_vars:
                    self.optimization_handler.add_config_with_variables(
                        (each_config, each_vars), name=each_plugin_name)
        self.optimization_handler.initialize_for_session()

    def get_train_op(self, model_results: ModelResults) -> tf.Operation:
        """
        Create train operation using optimization handler. Also will add the
        update operation to it

        Parameters
        ----------
        model_results
            model results

        Returns
        -------
        train_with_update_op
            train operation together with update operation
        """
        with tf.variable_scope(ScopeNames.TRAIN_OP):
            train_op = self.optimization_handler.get_train_op(
                model_results.grads_and_vars,
                model_results.regularization_grads_and_vars,
                trainable_variables=self.model.trainable_variables)
            update_op = self._get_update_op()
            train_with_update_op = tf.group(
                train_op, update_op, name='train_op')
        return train_with_update_op

    @staticmethod
    def get_estimator_spec(mode: str,
                           predictions: Optional[dict],
                           losses: Optional[dict],
                           metrics: Optional[dict],
                           train_op: Optional[tf.Operation]
                           ) -> tf.estimator.EstimatorSpec:
        """
        Construct the estimator spec

        Parameters
        ----------
        mode
            mode of the model
        predictions
            all predictions
        losses
            all losses
        metrics
            metrics of the model
        train_op
            training operation

        Returns
        -------
        estimator_spec
            estimator spec
        """

        loss = None
        eval_metric_ops = None
        if losses is not None and 'total_loss' in losses:
            loss = losses['total_loss']
        if predictions is not None:
            predictions = nest_utils.flatten_nested_struct(predictions)
        if metrics:
            metrics_flatten = nest_utils.flatten_nested_struct(metrics)
            eval_metric_ops = {
                k: (v, tf.no_op()) for k, v in metrics_flatten.items()}
        estimator_spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            predictions=predictions,
            eval_metric_ops=eval_metric_ops)
        return estimator_spec

    def add_summaries(self, model_results: ModelResults, mode: str):
        """
        Add summaries from model_results to the summaries and merge them

        Parameters
        ----------
        model_results
            model results
        mode
            mode of the model
        """
        self._add_loss_to_summaries(model_results)
        self._add_grads_and_vars_to_summaries(model_results)
        self._add_summary_to_summaries(model_results)
        self._add_metrics_to_summaries(model_results)
        self._add_learning_rate_to_summaries(mode)
        tf.summary.merge_all()

    @staticmethod
    def add_to_collections(model_results: ModelResults):
        """
        Add nucleus7 collections from model_results

        Parameters
        ----------
        model_results
            model results
        """
        tf_collections_utils.nested2collection(
            CollectionNames.INPUTS_PREPROCESSED,
            model_results.inputs_preprocessed)
        if model_results.predictions_raw is not None:
            tf_collections_utils.nested2collection(
                CollectionNames.PREDICTIONS_RAW, model_results.predictions_raw)
        if model_results.predictions is not None:
            tf_collections_utils.nested2collection(
                CollectionNames.PREDICTIONS, model_results.predictions)
        if model_results.losses is not None:
            tf_collections_utils.nested2collection(
                CollectionNames.LOSSES, model_results.losses)
        if model_results.summary:
            tf_collections_utils.nested2collection(
                CollectionNames.SUMMARY, model_results.summary)
        if model_results.metrics:
            tf_collections_utils.nested2collection(
                CollectionNames.METRIC, model_results.metrics)

    @staticmethod
    def _get_update_op() -> tf.Operation:
        return tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    def _get_number_of_devices_for_mode(self, mode: str) -> int:
        if mode == tf.estimator.ModeKeys.PREDICT:
            number_of_devices = 1
        else:
            number_of_devices = len(self.devices)
        return number_of_devices

    def _combine_predictions(
            self, tower_model_results: List[ModelResults]
    ) -> Tuple[Optional[tf.Tensor], Optional[tf.Tensor]]:
        predictions_raw = None
        predictions = None

        predictions_raw_workers = [model_results.predictions_raw
                                   for model_results in tower_model_results]
        predictions_workers = [model_results.predictions
                               for model_results in tower_model_results]

        if predictions_raw_workers[0] is not None:
            predictions_raw = model_utils.combine_predictions_from_devices(
                predictions_devices=predictions_raw_workers,
                predictions_have_variable_shape=
                self.predictions_have_variable_shape)
        if predictions_workers[0] is not None:
            predictions = model_utils.combine_predictions_from_devices(
                predictions_devices=predictions_workers,
                predictions_have_variable_shape=
                self.predictions_have_variable_shape)
        return predictions_raw, predictions

    @staticmethod
    def _combine_losses(tower_model_results: List[ModelResults]
                        ) -> Optional[Dict[str, tf.Tensor]]:
        losses_workers = [model_results.losses
                          for model_results in tower_model_results]
        if losses_workers[0] is not None:
            losses = model_utils.combine_losses_from_devices(
                losses_devices=losses_workers)
            return losses
        return None

    @staticmethod
    def _combine_summary(tower_model_results: List[ModelResults]
                         ) -> Optional[Dict[str, tf.Tensor]]:
        summary_workers = [model_results.summary
                           for model_results in tower_model_results]
        if summary_workers[0] is not None:
            summary = model_utils.combine_summary_from_devices(
                summary_devices=summary_workers)
            return summary
        return None

    @staticmethod
    def _combine_metrics(tower_model_results: List[ModelResults]
                         ) -> Optional[Dict[str, tf.Tensor]]:
        metrics_workers = [model_results.metrics
                           for model_results in tower_model_results]
        if metrics_workers[0] is not None:
            metrics = model_utils.combine_metrics_from_devices(
                metrics_devices=metrics_workers)
            return metrics
        return None

    def _combine_grads_and_vars(
            self, tower_model_results: List[ModelResults]
    ) -> Optional[List[Tuple[tf.Tensor, tf.Variable]]]:
        grads_and_vars_workers = [model_results.grads_and_vars
                                  for model_results in tower_model_results]
        if grads_and_vars_workers[0] is not None:
            consolidation_device = (
                '/gpu:0' if self.variable_strategy == 'GPU' else '/cpu:0')
            grads_and_vars = (
                opt_utils.average_grads_and_vars_from_multiple_devices(
                    tower_grads_and_vars=grads_and_vars_workers,
                    consolidation_device=consolidation_device))
            return grads_and_vars
        return None

    def _add_learning_rate_to_summaries(self, mode):
        learning_rate = self.optimization_handler.global_learning_rate
        if mode == tf.estimator.ModeKeys.TRAIN:
            tf.summary.scalar('learning_rate', learning_rate)

    def _add_metrics_to_summaries(self, model_results: ModelResults):
        if model_results.metrics is not None:
            metrics_flatten = nest_utils.flatten_nested_struct(
                model_results.metrics)
            for metric_name, metric_value in metrics_flatten.items():
                model_utils.add_summary_by_name(
                    metric_name, metric_value, self.max_outputs_tb)

    def _add_summary_to_summaries(self, model_results: ModelResults):
        if model_results.summary is not None:
            summary_flatten = nest_utils.flatten_nested_struct(
                model_results.summary)
            for summary_name, summary_value in summary_flatten.items():
                model_utils.add_summary_by_name(
                    summary_name, summary_value, self.max_outputs_tb)

    @staticmethod
    def _add_grads_and_vars_to_summaries(model_results: ModelResults):
        if model_results.grads_and_vars is not None:
            for grad, var in model_results.grads_and_vars:
                grad_name = ('gradient/' + var.name).replace(':', '_')
                model_utils.add_histogram_summary(grad_name, grad)
                grad_norm = tf.norm(grad)
                grad_norm_name = "gradient_l2_norms/scalar_" + grad_name
                model_utils.add_summary_by_name(grad_norm_name, grad_norm)
            all_grads = list(zip(*model_results.grads_and_vars))[0]
            global_grad_norm = tf.global_norm(all_grads)
            global_norm_name = "_".join(["scalar", "global_gradient_l2_norm"])
            model_utils.add_summary_by_name(global_norm_name, global_grad_norm)

        if model_results.regularization_grads_and_vars is not None:
            for grad, var in model_results.regularization_grads_and_vars:
                grad_name = ('reg_gradient/' + var.name).replace(':', '_')
                model_utils.add_histogram_summary(grad_name, grad)

    @staticmethod
    def _add_loss_to_summaries(model_results: ModelResults):
        if model_results.losses is not None:
            losses_flatten = nest_utils.flatten_nested_struct(
                model_results.losses)
            for loss_name, loss in losses_flatten.items():
                tf.summary.scalar(loss_name, loss, family='loss')


def _get_custom_vars_getter(
        mixed_precision_config: Optional[MixedPrecisionConfig] = None
) -> Optional[Callable]:
    if mixed_precision_config is not None and mixed_precision_config.use:
        custom_getter = float32_variable_storage_getter
    else:
        custom_getter = None
    return custom_getter


def _get_device_setter(worker_device, variable_strategy, num_devices):
    if variable_strategy == 'CPU':
        device_setter = local_device_setter(worker_device=worker_device)
    else:
        device_setter = local_device_setter(
            ps_device_type='gpu',
            worker_device=worker_device,
            ps_strategy=
            tf.contrib.training.GreedyLoadBalancingStrategy(
                num_devices,
                tf.contrib.training.byte_size_load_fn))
    return device_setter
