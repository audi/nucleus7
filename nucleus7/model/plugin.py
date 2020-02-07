# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Interface for model plugin
"""
import abc
import logging
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import tensorflow as tf
from tensorflow import keras

from nucleus7.core.nucleotide import TfNucleotide
from nucleus7.optimization.configs import (
    OptimizationConfig)
from nucleus7.utils import nest_utils
from nucleus7.utils import tf_objects_factory
from nucleus7.utils import tf_utils
from nucleus7.utils import tf_varscopes_utils

# pylint: disable=invalid-name
# this is a type constant, not a class
_OPTIM_CONFIG_WITH_VARS = Tuple[OptimizationConfig, List[tf.Variable]]
_OPTIM_CONFIGS_ARG = Union[Dict[str, OptimizationConfig], OptimizationConfig]


# pylint: enable=invalid-name


# pylint: disable=too-many-instance-attributes
# is needed to make the DataExtractor more generic
class ModelPlugin(TfNucleotide):
    """
    Interface of network model to be plugged into Model.
    main goal is to separate the network architecture from the task

    arguments are common for all plugins and may be used or not in the
    implementation

    Parameters
    ----------
    inbound_nodes
        name of the input nodes
    activation
        activation function to use or its configuration in form of name
        or dict {'name': , 'param1': , 'param2': ...}
    dropout
        dropout function to use or its configuration in form of name
        or dict {'name': , 'param1': , 'param2': ...}
    initializer
        initializer to use as keras Initializer or its configuration in form of
        name or dict {'name': , 'param1': , 'param2': ...}
    stop_gradient_from_inputs
        if the gradient from inputs should be stopped, so gradient from this
        plugin will not be backpropagated to its inputs
    load_fname
        checkpoint file name to restore variables
    load_var_scope
        specifies var scope of plugin variables inside of restored checkpoint
    exclude_from_restore
        if variables of this plugin should be not restored from global
        checkpoint
    optimization_configs
        optimization config or a dict with mappings of
        {variable name pattern: optimization}; if you want to specify one
        optimizer for all variables except other patterns, specify it with
        '*' key
    allow_mixed_precision
        defines, if the plugin should be excluded from mixed precision and in
        that way, all inputs with float16 dtype will be casted automatically
        to float32

    Attributes
    ----------
    register_name_scope
        name scope for register and for the config logger
    _variable_scope : str
        name of variable scope for plugin
    _variables : list
        list of plugin global variables (trainable and not trainable)
    """
    register_name_scope = "model_plugin"
    exclude_from_register = True

    _process_method_name = 'predict'

    def __init__(self, *,
                 inbound_nodes: Union[str, list] = 'dataset',
                 activation: Union[str, Dict[str, str], Callable] = 'elu',
                 initializer: Union[str, dict,
                                    keras.initializers.Initializer] = None,
                 dropout: Union[str, Dict[str, str], Callable] = 'normal',
                 stop_gradient_from_inputs: bool = False,
                 load_fname: str = None,
                 load_var_scope: str = None,
                 exclude_from_restore: bool = False,
                 optimization_configs: Optional[_OPTIM_CONFIGS_ARG] = None,
                 allow_mixed_precision: bool = True,
                 **nucleotide_kwargs):
        super(ModelPlugin, self).__init__(inbound_nodes=inbound_nodes,
                                          **nucleotide_kwargs)

        self.stop_gradient_from_inputs = stop_gradient_from_inputs
        self.load_fname = load_fname
        self.load_var_scope = load_var_scope
        self.exclude_from_restore = exclude_from_restore
        self.optimization_configs = optimization_configs
        self.allow_mixed_precision = allow_mixed_precision
        self.activation = activation
        self.dropout = dropout
        self.initializer = initializer

    @property
    def defaults(self):
        defaults = super(ModelPlugin, self).defaults
        defaults.update(
            {'activation': tf_objects_factory.activation_factory,
             'dropout': tf_objects_factory.dropout_factory,
             'initializer': tf_objects_factory.initializer_factory})
        return defaults

    @abc.abstractmethod
    def predict(self, **inputs) -> Dict[str, tf.Tensor]:
        """
        Forward pass given inputs

        Parameters
        ----------
        **inputs
            dict with input tensors

        Returns
        -------
        output
            outputs after this layer
        """

    def _call(self, **inputs) -> Dict[str, tf.Tensor]:
        """
        wrapper on top of `self.predict` with defined initializer
        and variable scope
        """
        # pylint: disable=arguments-differ
        # parent __call__ method has more generic signature
        inputs = self._maybe_stop_gradients_from_inputs(inputs)
        predictions = self.predict(**inputs)
        return predictions

    def maybe_initialize_from_checkpoint(self):
        """
        Initialize the variables from checkpoint if load_fname was provided
        """
        if not self.load_fname:
            return
        logger = logging.getLogger(__name__)
        logger.info("Initialize plugin %s from checkpoint %s",
                    self.name, self.load_fname)
        var_list = {tf_varscopes_utils.remove_scope_from_name(
            v.name, self._variable_scope): v
                    for v in self._variables}
        if isinstance(self.load_var_scope, str):
            if self.load_var_scope[-1] == '/':
                self.load_var_scope = self.load_var_scope[:-1]
            var_list = {'/'.join([self.load_var_scope, k]): v
                        for k, v in var_list.items()}
        var_list = {tf_utils.remove_tag_from_variable_name(k): v
                    for k, v in var_list.items()}
        tf.train.init_from_checkpoint(self.load_fname, var_list)

    def get_optimization_configs_with_variables(
            self) -> Optional[List[_OPTIM_CONFIG_WITH_VARS]]:
        """
        Get the list of (optim_config, variables) to apply on the variables of
        this plugin

        Returns
        -------
        optim_configs_with_vars
            list of optimization configs with variables for it
        """
        trainable_variables = self.trainable_variables
        if (not self.trainable
                or not trainable_variables
                or not self.optimization_configs):
            return None
        if isinstance(self.optimization_configs, OptimizationConfig):
            return [(self.optimization_configs, trainable_variables)]
        optim_configs_with_vars = []
        main_optim_config = self.optimization_configs.get('*', None)
        rest_variables = set(trainable_variables)
        for each_var_name_pattern, each_config in sorted(
                self.optimization_configs.items()):
            if each_var_name_pattern == '*':
                continue
            vars_with_pattern = tf_utils.filter_variables_by_pattern(
                trainable_variables, each_var_name_pattern, self.variable_scope)
            optim_configs_with_vars.append((each_config, vars_with_pattern))
            rest_variables -= set(vars_with_pattern)
        if main_optim_config is not None:
            optim_configs_with_vars.append(
                (main_optim_config, list(rest_variables)))
        return optim_configs_with_vars

    def _maybe_stop_gradients_from_inputs(self, inputs):
        if self.stop_gradient_from_inputs:
            inputs_flat = nest_utils.flatten_nested_struct(inputs)
            inputs_flat_without_grads = {
                k: tf.stop_gradient(v) for k, v in inputs_flat.items()}
            inputs_without_grads = nest_utils.unflatten_dict_to_nested(
                inputs_flat_without_grads)
            return inputs_without_grads
        return inputs
