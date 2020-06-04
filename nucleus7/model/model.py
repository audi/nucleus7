# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Interface for model - handler for the collections of tensorflow model related
nucleotides
"""
import logging
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import tensorflow as tf

from nucleus7.core.base import MetaLogAndRegister
from nucleus7.core.gene_handler import GeneHandler
from nucleus7.core.nucleotide import Nucleotide
from nucleus7.core.nucleotide import TfNucleotide
from nucleus7.model.configs import MixedPrecisionConfig
from nucleus7.model.configs import ModelResults
from nucleus7.model.fields import CollectionNames
from nucleus7.model.fields import ScopeNames
from nucleus7.model.loss import ModelLoss
from nucleus7.model.metric import ModelMetric
from nucleus7.model.plugin import ModelPlugin
from nucleus7.model.postprocessor import ModelPostProcessor
from nucleus7.model.summary import ModelSummary
from nucleus7.utils import model_utils
from nucleus7.utils import nest_utils
from nucleus7.utils import object_utils
from nucleus7.utils import tf_collections_utils
from nucleus7.utils import tf_ops
from nucleus7.utils import tf_utils
from nucleus7.utils.model_utils import DefaultPlaceholderInfo

# pylint: disable=invalid-name
# this is a type constant, not a class
_NESTED_TENSORS_DTYPE = Union[Dict[str, tf.Tensor],
                              Dict[str, Dict[str, tf.Tensor]]]
_GRAD_AND_VARS_TYPE = List[Tuple[tf.Tensor, tf.Variable]]


# pylint: enable=invalid-name

# pylint: disable=too-many-instance-attributes
# is needed to make the Model more generic
class Model(GeneHandler, metaclass=MetaLogAndRegister):
    """
    Model to be plugged into ModelHandler and defines the task specific
    architecture like losses, summaries etc.
    this operations will be replicated across all of the devices inside of
    ModelHandler instance
    one per task definition

    Parameters
    ----------
    plugins
        model plugins
    losses
        model losses
    postprocessors
        model postprocessors
    summaries
        model summaries
    metrics
        model metrics
    regularization_l1
        coefficient of the l1 regularization term to use;
        if regularization_l1 == 0, no l1 regularization will be used;
        otherwise, regularization is applied to all trainable variables from
        all nucleotides
    regularization_l2
        coefficient of the l2 regularization term to use;
        if regularization_l2 == 0, no l2 regularization will be used;
        otherwise, regularization is applied to all trainable variables from
        all nucleotides
    aggregation_method
        aggregation method for gradients;
        defaults to tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N
    load_config
        configuration for loading. Can include following keys:

            `checkpoint` - file name of checkpoint with extension .ckpt

            `only_trainable_parameters` - boolean flag to load only
            trainable variables
    mixed_precision_config
        configuration for mixed precision

    Attributes
    ----------
    register_name_scope
        name scope for register and for the config logger
    exclude_args_from_log
        fields that will not be included to the config logger
    """
    register_name_scope = "model"
    exclude_from_register = True
    exclude_from_log = False
    exclude_args_from_log = ["plugins", "losses", "postprocessors",
                             "summaries", "metrics"]

    nucleotide_type_dependency_map = {
        ModelPlugin: [ModelPlugin],
        ModelLoss: [ModelPlugin, ModelLoss],
        ModelPostProcessor: [ModelPlugin, ModelPostProcessor],
        ModelSummary: [ModelPlugin, ModelLoss, ModelPostProcessor,
                       ModelSummary],
        ModelMetric: [ModelPlugin, ModelLoss, ModelPostProcessor]
    }
    gene_name_and_nucleotide_super_cls = {
        'plugins': ModelPlugin,
        'losses': ModelLoss,
        'postprocessors': ModelPostProcessor,
        'summaries': ModelSummary,
        'metrics': ModelMetric,
    }

    def __init__(
            self,
            plugins: List[ModelPlugin],
            losses: List[ModelLoss], *,
            postprocessors: Union[List[ModelPostProcessor], None] = None,
            summaries: Union[List[ModelSummary], None] = None,
            metrics: Union[List[ModelMetric], None] = None,
            regularization_l1: float = 0,
            regularization_l2: float = 0,
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N,
            load_config: Union[dict, None] = None,
            mixed_precision_config: Optional[MixedPrecisionConfig] = None):

        self.plugins = None  # type: Dict[str, ModelPlugin]
        self.losses = None  # type: Dict[str, ModelLoss]
        self.postprocessors = None  # type: Dict[str, ModelPostProcessor]
        self.summaries = None  # type: Dict[str, ModelSummary]
        self.metrics = None  # type: Dict[str, ModelMetric]

        super().__init__(plugins=plugins,
                         losses=losses,
                         postprocessors=postprocessors,
                         summaries=summaries,
                         metrics=metrics)

        self.regularization_l1 = regularization_l1
        self.regularization_l2 = regularization_l2
        self.aggregation_method = aggregation_method
        self.load_config = load_config or {}
        self._mixed_precision_config = mixed_precision_config

    @property
    def mixed_precision_config(self) -> Union[MixedPrecisionConfig, None]:
        """
        Configuration of use of mixed precision

        Returns
        -------
        mixed_precision_config
            config for mixed precision
        """
        return self._mixed_precision_config

    @mixed_precision_config.setter
    def mixed_precision_config(self,
                               mixed_precision_config: MixedPrecisionConfig):
        self._mixed_precision_config = mixed_precision_config

    @property
    def default_placeholders(self) -> List[DefaultPlaceholderInfo]:
        """
        Get all default placeholders of the model

        Returns
        -------
        all_placeholders
            list of default placeholder info from all nucleotides
        """
        if not self.all_nucleotides:
            return []

        all_placeholders = []
        for each_nucleotide in self.all_nucleotides.values():
            if isinstance(each_nucleotide, model_utils.DefaultPlaceholderMixin):
                nucleotide_placeholders = each_nucleotide.default_placeholders
                all_placeholders.extend(list(nucleotide_placeholders.values()))
        return all_placeholders

    @object_utils.assert_is_built
    def reset_tf_graph(self):
        """
        Reset model tensorflow graph
        """
        if not self.all_nucleotides:
            return
        for each_nucleotide in self.all_nucleotides.values():
            if isinstance(each_nucleotide, TfNucleotide):
                each_nucleotide.reset_tf_graph()

    @property
    def trainable_variables(self) -> List[tf.Variable]:
        """
        All trainable variables in all nucleotides inside of the model

        Returns
        -------
        trainable_variables
            trainable variables inside of the model
        """
        if not self.all_nucleotides:
            return []
        trainable_variables = []
        for each_nucleotide in self.all_nucleotides.values():
            trainable_variables.extend(each_nucleotide.trainable_variables)
        return trainable_variables

    @object_utils.assert_is_built
    @object_utils.assert_property_is_defined('mode')
    def __call__(self,
                 inputs_from_dataset: Dict[str, tf.Tensor]) -> ModelResults:
        """
        Build the forward pass graph

        Parameters
        ----------
        inputs_from_dataset
            dict of inputs

        Returns
        -------
        model_results
            model results holding following fields:
                * inputs_preprocessed - inputs after preprocessing
                * predictions_raw - predictions of ModelPlugins
                * predictions - predictions after applying PostProcessors;
                  is None for TRAIN
                * losses=losses - losses, None for PREDICT
                * summary=summary - summaries, only for EVAL
                * metrics=metrics - metrics, only for EVAL
        """
        # pylint: disable=arguments-differ
        # parent __call__ method has more generic signature
        losses = None
        predictions = None
        summary = None
        metrics = None
        grads_and_vars = None
        regularization_grads_and_vars = None

        with tf.variable_scope(ScopeNames.PREPROCESSING):
            inputs_from_dataset = self.preprocess_dataset_inputs(
                inputs_from_dataset)
        with tf.variable_scope(ScopeNames.MODEL):
            predictions_raw = self.forward_pass(
                inputs_from_dataset=inputs_from_dataset)
        if self.mode != tf.estimator.ModeKeys.TRAIN:
            with tf.variable_scope(ScopeNames.POSTPROCESSING):
                predictions = self.postprocess_predictions(
                    inputs_from_dataset=inputs_from_dataset,
                    predictions_raw=predictions_raw)
        if self.mode != tf.estimator.ModeKeys.PREDICT:
            with tf.variable_scope(ScopeNames.LOSSES):
                losses = self.calculate_losses(
                    inputs_from_dataset=inputs_from_dataset,
                    predictions_raw=predictions_raw)

        if self.mode == tf.estimator.ModeKeys.EVAL:
            with tf.variable_scope(ScopeNames.SUMMARY):
                summary = self.get_summaries(
                    inputs_from_dataset=inputs_from_dataset,
                    predictions_raw=predictions_raw,
                    predictions=predictions)
            with tf.variable_scope(ScopeNames.METRIC):
                metrics = self.get_metrics(
                    inputs_from_dataset=inputs_from_dataset,
                    predictions_raw=predictions_raw,
                    predictions=predictions)
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            with tf.variable_scope(ScopeNames.GRADIENTS):
                (grads_and_vars, regularization_grads_and_vars
                 ) = self.calculate_gradients(losses)

        if self.is_training:
            self.maybe_initialize_from_checkpoints()

        if self.mode == tf.estimator.ModeKeys.PREDICT:
            predictions_raw = None

        model_results = ModelResults(
            inputs_preprocessed=inputs_from_dataset,
            predictions_raw=predictions_raw, predictions=predictions,
            losses=losses, summary=summary, metrics=metrics,
            grads_and_vars=grads_and_vars,
            regularization_grads_and_vars=regularization_grads_and_vars)
        return model_results

    @object_utils.assert_is_built
    def build_inference_graph(self, features: Dict[str, tf.Tensor]) -> dict:
        """
        Build graph for inference

        Parameters
        ----------
        features
            dict with mappings feature name to feature tensor

        Returns
        -------
        predictions_flatten
            flatten dict holding predictions

        Raises
        ------
        ValueError
            if no predictions were built
        """
        logger = logging.getLogger(__name__)
        logger.info('Build inference graph')
        self.mode = tf.estimator.ModeKeys.PREDICT
        self._validate_genes_for_inference()
        self.reset_tf_graph()

        model_results = self(features)
        predictions = model_results.predictions
        inputs_connected = tf_utils.get_connected_inputs_to_predictions(
            features, predictions, tf.get_default_graph())
        tf_collections_utils.nested2collection(
            CollectionNames.INPUTS, inputs_connected)
        tf_collections_utils.nested2collection(
            CollectionNames.PREDICTIONS, predictions)
        predictions_flatten = nest_utils.flatten_nested_struct(predictions)
        return predictions_flatten

    @object_utils.assert_is_built
    def maybe_initialize_from_checkpoints(self):
        """
        Initialize variables from checkpoints

        If one variable exists inside of load_config['checkpoint'] and also
        inside of plugin checkpoint, then plugin checkpoint will be used
        """
        load_fname = self.load_config.get('checkpoint')
        if not load_fname:
            self.maybe_initialize_plugins_from_checkpoints()
            return
        logger = logging.getLogger(__name__)
        logger.info("Initialize model from checkpoint %s", load_fname)
        restore_only_trainable = self.load_config.get(
            'only_trainable_parameters')
        vars2restore = set(
            self.trainable_variables if
            restore_only_trainable else
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        # remove variables from plugins if they have flag
        # exclude_from_restore or they have their own checkpoint
        for plugin in self.plugins.values():
            if plugin.exclude_from_restore or plugin.load_fname:
                logger.info("Exclude plugin %s from handler checkpoint %s",
                            plugin.name, load_fname)
                for each_variable in plugin.variables:
                    vars2restore.remove(each_variable)
        assignment_map = {tf_utils.remove_tag_from_variable_name(v.name): v
                          for v in vars2restore}
        tf.train.init_from_checkpoint(load_fname, assignment_map)

    @staticmethod
    def preprocess_dataset_inputs(inputs: Dict[str, tf.Tensor]
                                  ) -> Dict[str, Dict[str, tf.Tensor]]:
        """
        Add preprocessing step as identity nodes on dataset inputs
        and add all the inputs to dataset key

        Parameters
        ----------
        inputs
            inputs from datasets

        Returns
        -------
        inputs_with_identity
            same as inputs, but with added identity ops and add to dataset key
        """
        inputs_flat = nest_utils.flatten_nested_struct(inputs)
        inputs_flat_identity = {
            k: tf.identity(v) for k, v in sorted(inputs_flat.items())}
        inputs_identity = nest_utils.unflatten_dict_to_nested(
            inputs_flat_identity)
        inputs_identity = {"dataset": inputs_identity}
        return inputs_identity

    @object_utils.assert_is_built
    @object_utils.assert_property_is_defined('mode')
    def forward_pass(self, *,
                     inputs_from_dataset: _NESTED_TENSORS_DTYPE
                     ) -> _NESTED_TENSORS_DTYPE:
        """
        Make the forward pass for model

        Parameters
        ----------
        inputs_from_dataset
            dictionary holding the dataset inputs

        Returns
        -------
        predictions
            dict with outputs for each model plugin with keys as plugin names
        """
        return self._process_gene(gene_name='plugins',
                                  gene_inputs=inputs_from_dataset)

    @object_utils.assert_is_built
    @object_utils.assert_property_is_defined('mode')
    def postprocess_predictions(self, *,
                                inputs_from_dataset: _NESTED_TENSORS_DTYPE,
                                predictions_raw: _NESTED_TENSORS_DTYPE
                                ) -> _NESTED_TENSORS_DTYPE:
        """
        Apply the postprocessing on predictions
        e.g. classes or probabilities from logits
        this predictions will be added to collection and further used for the
        summaries / inference

        Parameters
        ----------
        inputs_from_dataset
            dictionary holding the dataset inputs
        predictions_raw
            dict with predictions from model_plugin

        Returns
        -------
        postprocessed_predictions
            dict with outputs of the model for inference
        """
        gene_inputs = {}
        gene_inputs.update(inputs_from_dataset)
        gene_inputs.update(predictions_raw)
        return self._process_gene(gene_name='postprocessors',
                                  gene_inputs=gene_inputs)

    @object_utils.assert_is_built
    @object_utils.assert_property_is_defined('mode')
    def calculate_losses(self, *,
                         inputs_from_dataset: _NESTED_TENSORS_DTYPE,
                         predictions_raw: _NESTED_TENSORS_DTYPE
                         ) -> Dict[str, Union[Dict[str, tf.Tensor], tf.Tensor]]:
        """
        Create loss for model using labels from 'inputs'
        and predictions

        Parameters
        ----------
        inputs_from_dataset
            dictionary holding the dataset inputs
        predictions_raw
            dict with prediction tensors as values

        Returns
        -------
        losses
            dict with values as losses; inside of keys should be 'total_loss'
            with value to optimize; default ragularizations on the all
            training variables will be applied afterwards if specified
        """
        gene_inputs = {}
        gene_inputs.update(inputs_from_dataset)
        gene_inputs.update(predictions_raw)
        losses = self._process_gene(gene_name='losses', gene_inputs=gene_inputs)
        unflatten_losses = nest_utils.unflatten_dict_to_nested(losses)
        total_loss = 0.0
        for loss in unflatten_losses.values():
            if 'total_loss' in loss:
                total_loss += loss['total_loss']
        losses['total_loss'] = total_loss
        losses = self._add_regularization(losses)
        return losses

    @object_utils.assert_is_built
    @object_utils.assert_property_is_defined('mode')
    def get_summaries(self, *,
                      inputs_from_dataset: _NESTED_TENSORS_DTYPE,
                      predictions_raw: _NESTED_TENSORS_DTYPE,
                      predictions: _NESTED_TENSORS_DTYPE
                      ) -> _NESTED_TENSORS_DTYPE:
        """
        Create the summaries for model as dict
        to separate different types of summaries, the prefix is used:

            - `scalar_{}`
            - `image_{}`
            - `histogram_{}`
            - `text_{}`
            - `audio_{}`

        Names without this prefixes will not be stored to tensorboard.

        Parameters
        ----------
        inputs_from_dataset
            dictionary holding the dataset inputs
        predictions_raw
            dict with raw predictions
        predictions
            dict with output tensors as values

        Returns
        -------
        summaries
            combined summaries from all `ModelSummary` instances
        """
        gene_inputs = {}
        gene_inputs.update(inputs_from_dataset)
        gene_inputs.update(predictions_raw)
        gene_inputs.update(predictions)
        summaries = self._process_gene(gene_name='summaries',
                                       gene_inputs=gene_inputs)
        summaries_not_to_store = [n for n, s in self.summaries.items()
                                  if not s.store_inside_tensorboard]
        if summaries_not_to_store:
            logger = logging.getLogger(__name__)
            logger.info('Summaries with names %s will not be stored to '
                        'tensorboard', summaries_not_to_store)
            summaries_unfatten = nest_utils.unflatten_dict_to_nested(summaries)
            for each_summary_name in summaries_not_to_store:
                del summaries_unfatten[each_summary_name]
            summaries = nest_utils.flatten_nested_struct(summaries_unfatten)
        return summaries

    @object_utils.assert_is_built
    @object_utils.assert_property_is_defined('mode')
    def get_metrics(self, *,
                    inputs_from_dataset: _NESTED_TENSORS_DTYPE,
                    predictions_raw: _NESTED_TENSORS_DTYPE,
                    predictions: _NESTED_TENSORS_DTYPE
                    ) -> _NESTED_TENSORS_DTYPE:
        """
        Create metric
        Names without this prefixes will not be plotted in tensorboard.
        Only difference to summary is that metrics are added to its own
        collection (CollectionNames.METRIC) and not to summary collection.
        In that way it is possible to use metrics inside of training e.g. as
        for stopping training

        Parameters
        ----------
        inputs_from_dataset
            dictionary holding the dataset inputs
        predictions_raw
            dict with raw predictions
        predictions
            dict with output tensors as values

        Returns
        -------
        metric
            combined summaries from all `ModelMetric` instances
        """
        gene_inputs = {}
        gene_inputs.update(inputs_from_dataset)
        gene_inputs.update(predictions_raw)
        gene_inputs.update(predictions)
        return self._process_gene(gene_name='metrics',
                                  gene_inputs=gene_inputs)

    @object_utils.assert_is_built
    def calculate_gradients(self, losses: _NESTED_TENSORS_DTYPE
                            ) -> Tuple[_GRAD_AND_VARS_TYPE,
                                       _GRAD_AND_VARS_TYPE]:
        """
        Calculate gradients given losses

        Parameters
        ----------
        losses
            losses

        Returns
        -------
        grads_and_vars
            list of tuples holding pair (gradient, variable)
        regularization_grads_and_vars
            list of tuples holding pair (regularization gradient, variable)
        """
        grads_and_vars = self._calculate_gradients_from_loss(
            losses['total_loss'])
        regularization_loss = self._get_regularization_loss(losses)
        if regularization_loss is not None:
            regularization_grads_and_vars = self._calculate_gradients_from_loss(
                regularization_loss)
        else:
            regularization_grads_and_vars = None
        return grads_and_vars, regularization_grads_and_vars

    @object_utils.assert_is_built
    @object_utils.assert_property_is_defined('mode')
    def maybe_initialize_plugins_from_checkpoints(self):
        """
        Initialize plugins from checkpoints if plugin have them
        """
        for plugin in self.plugins.values():
            plugin.maybe_initialize_from_checkpoint()

    @object_utils.assert_is_built
    @object_utils.assert_property_is_defined('mode')
    def _process_gene(
            self, gene_name: str, *,
            gene_inputs: _NESTED_TENSORS_DTYPE = None
    ) -> _NESTED_TENSORS_DTYPE:
        """
        Wrapper across the process_gene method that uses self.is_training and
        also constructs the function to cast inputs in case of the mixed
        precision and calls the super().process_gene


        Parameters
        ----------
        gene_name
            gene name, e.g. losses, plugins
        gene_inputs
            inputs to the gene in the flatten format

        Returns
        -------
        gene_outputs
            dict with outputs for each gene nucleotide with keys as nucleotide
            names; keys are in flatten form, e.g.
            'nucleotide1//out1//out11' = 'value'
        """
        inputs_cast_fn = _get_inputs_cast_mixed_precision_fn(
            gene_name, self.mixed_precision_config)
        return super().process_gene(
            gene_name, gene_inputs=gene_inputs,
            nucleotide_inputs_preprocessing_fn=inputs_cast_fn)

    def _add_regularization(self, losses: _NESTED_TENSORS_DTYPE
                            ) -> _NESTED_TENSORS_DTYPE:
        """
        Add the regularization to loss if specified

        Parameters
        ----------
        losses
            dict with loss names and losses itself

        Returns
        -------
        losses
            dict with losses added the regularization losses and total_loss
        """
        if self.regularization_l1 == 0 and self.regularization_l2 == 0:
            return losses

        train_variables = tf.concat(
            [tf.reshape(each_var, (-1,))
             for each_var in self.trainable_variables], 0)
        if self.regularization_l1 > 0:
            loss_l1 = tf.reduce_sum(tf.abs(train_variables))
            losses['regularization_loss_l1'] = self.regularization_l1 * loss_l1
        if self.regularization_l2 > 0:
            loss_l2 = tf.reduce_sum(train_variables ** 2) / 2.
            losses['regularization_loss_l2'] = self.regularization_l2 * loss_l2
        return losses

    def _calculate_gradients_from_loss(self, loss: tf.Tensor
                                       ) -> Optional[_GRAD_AND_VARS_TYPE]:
        variables = self.trainable_variables
        use_mixed_precision = (self._mixed_precision_config is not None
                               and self._mixed_precision_config.use)
        if use_mixed_precision:
            loss = (loss * self._mixed_precision_config.loss_scale_factor)
        grads = tf.gradients(loss, variables,
                             aggregation_method=self.aggregation_method)
        if use_mixed_precision:
            grads = [g / self._mixed_precision_config.loss_scale_factor
                     for g in grads]
        grads_and_vars = list(zip(grads, variables))
        return grads_and_vars

    @staticmethod
    def _get_regularization_loss(losses: _NESTED_TENSORS_DTYPE
                                 ) -> Optional[tf.Tensor]:
        regularization_loss = None
        reg_loss_names = ["regularization_loss_l1", "regularization_loss_l2"]
        for each_reg_loss_name in reg_loss_names:
            each_reg_loss = losses.get(each_reg_loss_name)
            if each_reg_loss is not None:
                if regularization_loss is None:
                    regularization_loss = each_reg_loss
                else:
                    regularization_loss += each_reg_loss
        return regularization_loss

    def _validate_genes_for_inference(self):
        if self.mode == tf.estimator.ModeKeys.PREDICT:
            if not self.postprocessors:
                msg = ("Provide postprocessors for inference mode "
                       "since they serve as outputs for inference graph!")
                raise ValueError(msg)


# pylint: enable=too-many-instance-attributes


def _cast_plugin_inputs_for_mixed_precision(plugin: ModelPlugin,
                                            nucleotide_inputs: dict):
    if not plugin.allow_mixed_precision:
        cast_dtypes = {tf.float16: tf.float32}
    else:
        cast_dtypes = {tf.float32: tf.float16}
    nucleotide_inputs_casted = tf_ops.maybe_cast_dtype(
        nucleotide_inputs, cast_dtypes)
    return nucleotide_inputs_casted


# pylint: disable=unused-argument
# nucleotide is needed here for further use as an factory with this signature
def _cast_inputs_back_from_mixed_precision(nucleotide: Nucleotide,
                                           nucleotide_inputs: dict):
    nucleotide_inputs_casted = tf_ops.maybe_cast_dtype(
        nucleotide_inputs, {tf.float16: tf.float32})
    return nucleotide_inputs_casted


# pylint: enable=unused-argument


def _get_inputs_cast_mixed_precision_fn(
        gene_name: str,
        mixed_precision_config: MixedPrecisionConfig = None
) -> Union[Callable[[Nucleotide, dict], dict], None]:
    if (mixed_precision_config is None
            or not mixed_precision_config.use):
        return None

    if gene_name == 'plugins':
        cast_dtype_fn = _cast_plugin_inputs_for_mixed_precision
    else:
        cast_dtype_fn = _cast_inputs_back_from_mixed_precision
    return cast_dtype_fn
