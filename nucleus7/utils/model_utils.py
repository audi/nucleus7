# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Utils for model
"""

from collections import defaultdict
from collections import namedtuple
import math
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
import warnings

import numpy as np
import tensorflow as tf
try:
    from tensorflow.contrib.estimator.python.estimator import early_stopping
except ImportError:
    from tensorflow.python.estimator import early_stopping

from nucleus7.utils import nest_utils
from nucleus7.utils import tf_ops
from nucleus7.utils import tf_utils
from nucleus7.utils import tf_varscopes_utils

DefaultPlaceholderInfo = namedtuple(
    "DefaultPlaceholderInfo",
    ["full_name", "placeholder", "default_value"])


class KerasLayersMixin:
    """
    Mixin to use to add the keras layers to the nucleotide.

    This is needed because of different variable sharing between tensorflow
    layers and its variables and keras layers, since tensorflow uses
    :func:`tf.get_variable` and keras uses :obj:`tf.Variable` and also
    :func:`tf.get_variable_scope().reuse` doesn't work with keras layers.

    """

    @property
    def keras_layers_with_names(self) -> Dict[str, tf.keras.layers.Layer]:
        """
        Get all keras layers as mapping of layer name to the layer

        Returns
        -------
        keras_layer_name_to_layer_map
            dict with mapping {keras_layer_name: layer}
        """
        return getattr(self, '_keras_layers', {})

    @property
    def keras_layers(self) -> List[tf.keras.layers.Layer]:
        """
        Get all keras layers; if no keras layers were added, return empty list

        Returns
        -------
        keras_layers
            list of keras layers
        """
        keras_layers_map = self.keras_layers_with_names
        return list(keras_layers_map.values())

    def add_keras_layer(self, layer: Union[tf.keras.layers.Layer, Callable],
                        name: Optional[str] = None
                        ) -> tf.keras.layers.Layer:
        """
        Add layer or model to self.keras_layers.

        Parameters
        ----------
        layer
            keras layer to add or a function without parameters which will
            return a keras layer / model
        name
            name to store it; if this name already exist, will return layer
            from internal map and not store new one; must be set if the
            callable was provided as layer parameter

        Returns
        -------
        layer
            same layer as input layer

        Raises
        ------
        ValueError
            if provided layer is not an instance of keras layer
        """
        if not hasattr(self, '_keras_layers'):
            self._keras_layers = {}
        if not isinstance(layer, (tf.keras.layers.Layer, Callable)):
            msg = "{} is not a keras layer and not a callable".format(layer)
            raise ValueError(msg)

        if name is None:
            if not isinstance(layer, tf.keras.layers.Layer):
                raise ValueError(
                    "Provide name parameter if you provide callable instead "
                    "of layer")
            name = layer.name

        layer_retrieved = self._keras_layers.get(name)
        if layer_retrieved is None:
            if not isinstance(layer, tf.keras.layers.Layer):
                try:
                    layer = layer()
                except TypeError as exception:
                    msg = "Provided expression for keras layer is ""not valid!"
                    raise TypeError(msg) from exception
            self._keras_layers[name] = layer
            layer_retrieved = layer
        trainable = getattr(self, "trainable", True)
        if not trainable:
            layer_retrieved.trainable = trainable
        return layer_retrieved

    def reset_keras_layers(self):
        """
        Reset keras layers or layers of keras.models by setting its
        self.built = False
        """
        keras_layers = self.keras_layers
        for each_keras_layer in keras_layers:
            _reset_layer_or_model_recursively(each_keras_layer)

    def create_keras_layers(self):
        """
        Initialize and assign the keras layers used in this plugin
        """


class DefaultPlaceholderMixin:
    """
    Mixin to add the possibility to wrap model parameters as default
    placeholders. This allows to modify the parameter value after training,
    e.g. during inference
    """

    @property
    def default_placeholders(self) -> Dict[str, DefaultPlaceholderInfo]:
        """
        All default placeholders as mapping of name to placeholder

        Returns
        -------
        name_to_placeholder_map
            name to placeholder map
        """
        return getattr(self, '_default_placeholders', {})

    def add_default_placeholder(
            self, value, name: str,
            dtype: Optional[tf.DType] = None,
            shape: Optional[Union[int, List[int], Tuple[int]]] = -1,
            broadcast_shape: Optional[List[int]] = None,
    ) -> tf.Tensor:
        """
        Convert value to default placeholder and add it to internal state.

        If placeholder with this name already exists, will return it

        Parameters
        ----------
        value
            value of placeholder
        name
            name of the placeholder; will be accessible for inference through
            the 'nucleotide_name//name' argument.
        dtype
            if specified, cast placeholder to this format
        shape
            shape of placeholder to use
        broadcast_shape
            shape to broadcast the placeholder to if needed

        Returns
        -------
        value_as_default_placeholder
            value as default placeholder
        """
        if not hasattr(self, "_default_placeholders"):
            self._default_placeholders = {}

        if name in self._default_placeholders:
            placeholder = self._default_placeholders[name].placeholder
            if broadcast_shape is None:
                return placeholder
            return tf.broadcast_to(placeholder, broadcast_shape)

        full_name = _get_placeholder_full_name(self.name, name)
        if shape == -1:
            shape = _get_placeholder_shape(value)
        value_original = value

        if dtype:
            value = tf.cast(value, dtype)
        default_placeholder = tf.placeholder_with_default(
            value, shape, full_name)
        self._default_placeholders[name] = DefaultPlaceholderInfo(
            full_name, default_placeholder, value_original)
        if broadcast_shape is None:
            return default_placeholder
        return tf.broadcast_to(default_placeholder, broadcast_shape)

    def remove_all_placeholders(self):
        """
        Remove all placeholders which were added through the
        add_default_placeholder interface
        """
        if hasattr(self, "_default_placeholders"):
            del self._default_placeholders


class CustomSessionHandlerMixin:
    """
    Mixin for custom tensorflow session handling,
    like custom initialization etc.
    """

    # pylint: disable=too-few-public-methods
    # it is a mixin
    def initialize_session(self):
        """
        Method to initialize the custom session variables.
        """
        # pylint: disable=no-self-use
        # is an interface


@nest_utils.flatten_nested_inputs_inside_of_list('predictions_devices')
@nest_utils.unflatten_nested_outputs
def combine_predictions_from_devices(
        predictions_devices: List[Dict[str, tf.Tensor]],
        predictions_have_variable_shape: bool = False) -> Dict[str, tf.Tensor]:
    """
    Combines (concatenates) the predictions from multiple devices

    Parameters
    ----------
    predictions_devices
        list of dicts with same structure from multiple devices
    predictions_have_variable_shape
        if predictions from different devices may have different shapes; if so,
        it will use sparse operations to combine them

    Returns
    -------
    dict with same structure as first element in predictions_devices with
    concatenated over first dimension (batch dimension) values. If inputs
    have variable shape, then concatenation is done using
    :obj:`tf.sparse_concat` instead of :obj:`tf.concat`
    """
    if len(predictions_devices) == 1:
        return _dict_identity(predictions_devices[0])
    if predictions_have_variable_shape:
        combine_fun = lambda x: tf_ops.concat_padded(x, axis=0)
    else:
        combine_fun = lambda x: tf_ops.concat_or_stack(x, axis=0)
    with tf.variable_scope('combine_predictions'):
        predictions = nest_utils.combine_nested(predictions_devices,
                                                combine_fun=combine_fun)
    return predictions


@nest_utils.flatten_nested_inputs_inside_of_list('metrics_devices')
@nest_utils.unflatten_nested_outputs
def combine_metrics_from_devices(metrics_devices: List[Dict[str, tf.Tensor]]
                                 ) -> Dict[str, tf.Tensor]:
    """
    Combine (average) the metrics from multiple devices

    Parameters
    ----------
    metrics_devices
        list of dicts with same structure from multiple devices

    Returns
    -------
    dict with same structure as first element in metric_devices with
    average value of losses for that key
    """
    if len(metrics_devices) == 1:
        return _dict_identity(metrics_devices[0])
    with tf.variable_scope('combine_metrics'):
        losses = nest_utils.combine_nested(
            metrics_devices,
            combine_fun=lambda x: tf.reduce_mean(x, axis=0))
    return losses


@nest_utils.flatten_nested_inputs_inside_of_list('losses_devices')
@nest_utils.unflatten_nested_outputs
def combine_losses_from_devices(losses_devices: List[Dict[str, tf.Tensor]]
                                ) -> Dict[str, tf.Tensor]:
    """
    Combine (average) the losses from multiple devices

    Parameters
    ----------
    losses_devices
        list of dicts with same structure from multiple devices

    Returns
    -------
    dict with same structure as first element in losses_devices with
    average value of losses for that key
    """
    if len(losses_devices) == 1:
        return _dict_identity(losses_devices[0])
    with tf.variable_scope('combine_losses'):
        losses = nest_utils.combine_nested(
            losses_devices,
            combine_fun=lambda x: tf.reduce_mean(x, axis=0))
    return losses


@nest_utils.flatten_nested_inputs_inside_of_list('summary_devices')
@nest_utils.unflatten_nested_outputs
def combine_summary_from_devices(summary_devices: List[Dict[str, tf.Tensor]]
                                 ) -> Dict[str, tf.Tensor]:
    """
    Combine (average or selects first element) the summary
    from multiple devices

    Parameters
    ----------
    summary_devices
        list of dicts with same structure from multiple devices

    Returns
    -------
    dict with same structure as first element in losses_devices with
    combination method depending on type of summary - for scalars and
    histograms values are averaged across devices and for other types
    the summary from first device is taken
    """
    if all(s is None for s in summary_devices):
        return {}
    if len(summary_devices) == 1:
        return _dict_identity(summary_devices[0])
    combine_method = {'scalar': lambda x: tf.reduce_mean(x, axis=0),
                      'histogram': lambda x: tf.reduce_mean(x, axis=0),
                      'image': lambda x: tf.identity(x[0]),
                      'text': lambda x: tf.identity(x[0]),
                      'audio': lambda x: tf.identity(x[0]),
                      'default': lambda x: tf.identity(x[0])}
    combine_method_summary = defaultdict(
        lambda: lambda x: tf.identity(x[0]),
        combine_method)
    summary = nest_utils.combine_nested(summary_devices,
                                        combine_fun=combine_method_summary)
    return summary


def split_inputs_to_devices(inputs: Dict[str, tf.Tensor], num_devices: int
                            ) -> List[Dict[str, tf.Tensor]]:
    """
    Split inputs to devices

    Parameters
    ----------
    inputs
        dict of tensors
    num_devices
        number of devices

    Returns
    -------
    list of length len(self.devices) with same structure as inputs but
    with over first (batch) dimension
    """
    if num_devices == 1:
        return [inputs]
    inputs = tf_utils.split_dict_to_list_of_dict(
        inputs, num_devices, pad_to_batch=False)
    return inputs


@tf_varscopes_utils.with_name_scope("sample_mask")
def select_inputs_by_sample_mask(
        sample_mask: tf.Tensor,
        keys_to_exclude_from_sample_mask: Optional[List[str]] = None,
        **inputs
) -> Dict[str, tf.Tensor]:
    """
    Select inputs by masking out samples with sample_mask == 0

    Parameters
    ----------
    sample_mask
        tensor of shape [batch_size] with 1 indicating that sample should
        be leaved as is and 0 - remove sample
    keys_to_exclude_from_sample_mask
        list of keys that will not be masked using sample_mask
    **inputs
        inputs to mask

    Returns
    -------
    masked_inputs
        masked inputs sample-wise
    """
    inputs_flatten = nest_utils.flatten_nested_struct(inputs)
    inputs_masked_flatten = {}
    keys_to_exclude = keys_to_exclude_from_sample_mask or []
    for each_key, each_value in inputs_flatten.items():
        if each_key in keys_to_exclude:
            inputs_masked_flatten[each_key] = each_value
        else:
            inputs_masked_flatten[each_key] = tf.boolean_mask(
                each_value, sample_mask)
    inputs_masked = nest_utils.unflatten_dict_to_nested(
        inputs_masked_flatten)
    return inputs_masked


def select_inputs_by_sample_mask_np(
        sample_mask: np.ndarray,
        keys_to_exclude_from_sample_mask: Optional[List[str]] = None,
        **inputs
) -> Dict[str, np.ndarray]:
    """
    Select inputs by masking out samples with sample_mask == 0

    Parameters
    ----------
    sample_mask
        tensor of shape [batch_size] with 1 indicating that sample should
        be leaved as is and 0 - remove sample
    keys_to_exclude_from_sample_mask
        list of keys that will not be masked using sample_mask
    **inputs
        inputs to mask

    Returns
    -------
    masked_inputs
        masked inputs sample-wise
    """
    inputs_flatten = nest_utils.flatten_nested_struct(inputs)
    inputs_masked_flatten = {}
    keys_to_exclude = keys_to_exclude_from_sample_mask or []
    sample_mask = sample_mask.astype(bool)
    for each_key, each_value in inputs_flatten.items():
        if each_key in keys_to_exclude:
            inputs_masked_flatten[each_key] = each_value
        else:
            inputs_masked_flatten[each_key] = each_value[sample_mask]
    inputs_masked = nest_utils.unflatten_dict_to_nested(
        inputs_masked_flatten)
    return inputs_masked


def add_histogram_summary(summary_name: str, value: tf.Tensor):
    """
    Add histogram summary and also replace NaN in the value if needed on runtime

    Parameters
    ----------
    summary_name
        name of the summary
    value
        histogram value to add to summary with name

    """
    if isinstance(value, tf.IndexedSlices):
        tf.summary.histogram(
            summary_name, tf_ops.replace_nan_with_zeros(value.values))
    else:
        tf.summary.histogram(summary_name, tf_ops.replace_nan_with_zeros(value))


def add_summary_by_name(summary_name: str, summary_value: tf.Tensor,
                        max_outputs_tb: int = 1):
    """
    Add the summary defining the type of it by name and subtracting
    the prefix from name

    Parameters
    ----------
    summary_name
        name of the summary
    summary_value :
        value of summary
    max_outputs_tb
        number of maximum outputs in tensorboard e.g. for images

    """
    name_splitted = summary_name.split('/')
    if len(name_splitted) > 1:
        family = name_splitted[0]
    else:
        family = None
    if 'scalar_' in summary_name:
        tf.summary.scalar(summary_name.replace('scalar_', ''), summary_value,
                          family=family)
    elif 'image_' in summary_name:
        tf.summary.image(summary_name.replace('image_', ''), summary_value,
                         max_outputs=max_outputs_tb, family=family)
    elif 'histogram_' in summary_name:
        main_name = summary_name.replace('histogram_', '')
        if isinstance(summary_value, dict):
            for each_summary_name, each_summary_value in summary_value.items():
                histogram_name = '_'.join([main_name, each_summary_name])
                add_histogram_summary(histogram_name, each_summary_value)
        else:
            add_histogram_summary(main_name, summary_value)
    elif 'text_' in summary_name:
        tf.summary.text(summary_name.replace('text_', ''), summary_value)
    elif 'audio_' in summary_name:
        # TODO(oleksandr.vorobiov@audi.de) Find a way to set the sample_rate
        tf.summary.audio(summary_name.replace('audio_', ''), summary_value,
                         max_outputs=max_outputs_tb,
                         family=family, sample_rate=16000)
    else:
        msg = ('Warning: summary with name {} will not be added '
               'to tensorboard!'.format(summary_name))
        warnings.warn(msg, RuntimeWarning, stacklevel=2)


def get_iteration_stat_from_global_step(
        mode: str,
        global_step: Union[int, tf.Tensor],
        previous_iteration_number: int,
        number_iterations_per_epoch: int,
        max_number_of_iterations_per_epoch: int) -> Tuple[int, int, int]:
    """
    Calculate iteration statistics like epoch number, iteration number
    and summary step out of global_step for particular mode

    Parameters
    ----------
    mode
        mode of the model
    global_step
        global step
    previous_iteration_number
        previous iteration number for eval mode, since it cannot be inferred
        from global step
    number_iterations_per_epoch
        number of iterations per epoch, which is needed to calculate the summary
        step offset
    max_number_of_iterations_per_epoch
        max number of iterations per epoch

    Returns
    -------
    epoch_number
        epoch number
    iteration_number
        iteration number
    summary_step
        summary step
    """
    if mode == tf.estimator.ModeKeys.TRAIN:
        iteration_number = global_step % number_iterations_per_epoch + 1
        global_step += 1
    else:
        if previous_iteration_number % number_iterations_per_epoch == 0:
            iteration_number = 0
        else:
            iteration_number = previous_iteration_number
        iteration_number += 1

    epoch_number = math.ceil(global_step / max_number_of_iterations_per_epoch)
    iteration_offset = (epoch_number * max_number_of_iterations_per_epoch
                        - number_iterations_per_epoch)
    summary_step = iteration_number + iteration_offset
    return epoch_number, iteration_number, summary_step


# pylint: disable=no-member,protected-access
def get_or_create_early_stop_var() -> tf.Variable:
    """
    Creates a new variable for early stopping or returns existing one

    Wrapper across early_stopping._get_or_create_stop_var

    Returns
    -------
    early_stop_var
        variable for early stopping
    """
    return early_stopping._get_or_create_stop_var()


# pylint: enable=no-member,protected-access


def _dict_identity(data: dict) -> dict:
    return {k: tf.identity(v) for k, v in data.items()}


def _get_placeholder_full_name(object_name: str, placeholder_name: str) -> str:
    placeholder_full_name = "//".join([object_name, placeholder_name])
    return placeholder_full_name


def _get_placeholder_shape(value) -> tf.TensorShape:
    return tf.convert_to_tensor(value).shape


def _reset_layer_or_model_recursively(layer_or_model: tf.keras.layers.Layer):
    layer_or_model.built = False
    if isinstance(layer_or_model, tf.keras.Model):
        for each_layer in layer_or_model.layers:
            _reset_layer_or_model_recursively(each_layer)
