# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Class for OptimizationHandler
"""
from typing import List
from typing import Optional
from typing import Tuple

import tensorflow as tf

from nucleus7.core.base import BaseClass
from nucleus7.optimization.configs import OptimizationConfig
from nucleus7.optimization.configs import merge_optimization_configs
from nucleus7.utils import object_utils
from nucleus7.utils import optimization_utils
from nucleus7.utils import tf_objects_factory

# pylint: disable=invalid-name
# these are type constants, not a class
_CONFIG_TO_VAR_TYPE = Tuple[OptimizationConfig, List[tf.Variable]]
_GRAD_AND_VARS_TYPE = List[Tuple[tf.Tensor, tf.Variable]]


# pylint: enable=invalid-name


class OptimizationHandler(BaseClass):
    """
    Main class to create train_op for the training, which allows to use multiple
    optimization configurations for different variables
    """

    def __init__(self):
        super(OptimizationHandler, self).__init__()
        self._global_step = None
        self._global_config = None  # type: OptimizationConfig
        self._global_learning_rate = None  # type: tf.Tensor
        self._local_configs_with_vars = []  # type: List[_CONFIG_TO_VAR_TYPE]
        self._vars_with_local_configs = set()

    @property
    def global_step(self) -> Optional[tf.Tensor]:
        """
        Current global step

        Returns
        -------
        global_step
            global step
        """
        return self._global_step

    @property
    def global_learning_rate(self) -> Optional[tf.Tensor]:
        """
        Global learning rate

        Returns
        -------
        global_learning_rate
            global learning rate
        """
        return self._global_learning_rate

    @property
    def global_config(self) -> OptimizationConfig:
        """
        Global optimization config

        Returns
        -------
        global_config
            global optimization config
        """
        return self._global_config

    @global_config.setter
    def global_config(self, config: OptimizationConfig):
        self._check_global_config(config)
        self._global_config = config

    @object_utils.assert_property_is_defined("global_config")
    def add_config_with_variables(self, config_with_vars: _CONFIG_TO_VAR_TYPE,
                                  name: Optional[str] = None):
        """
        Add tuple (config, variables) of local optimization configs to
        variables to use. This variables will be removed from global
        optimization

        Parameters
        ----------
        config_with_vars
            optimization config with list of variables to apply
        name
            needed only to differentiate inside of the warnings / errors

        Raises
        ------
        ValueError
            if local config has learning_rate or optimizer inside
        ValueError
            if variables were already used in other configs
        """
        config, variables = config_with_vars
        self._check_local_config(config, config_name=name)
        self._check_variables_are_unique(variables, config_name=name)
        self._vars_with_local_configs.update(set(variables))
        config_updated = merge_optimization_configs(
            self.global_config, config)
        config_updated_with_vars = (config_updated, variables)
        self._local_configs_with_vars.append(config_updated_with_vars)

    @object_utils.assert_property_is_defined('global_config')
    def initialize_for_session(self):
        """
        Perform tensors initialization, e.g.:
            * create global step tensor
            * create global learning rate
            * create all optimizers
        """
        self._initialize_global_step()
        self._initialize_global_learning_rate()
        self._create_optimizers()

    def get_train_op(self, grads_and_vars: _GRAD_AND_VARS_TYPE,
                     regularization_grads_and_vars: _GRAD_AND_VARS_TYPE,
                     trainable_variables: List[tf.Variable]
                     ) -> tf.Operation:
        """
        Generate training operation using optimization configs

        Parameters
        ----------
        grads_and_vars
            list of (gradient, variable) for all variables in the model
        regularization_grads_and_vars
            list of (gradients, variable) from regularization losses for all
            variables in the model
        trainable_variables
            list of trainable variables

        Returns
        -------
        train_op
            train operation with update
        """
        return self._get_train_op(
            grads_and_vars, regularization_grads_and_vars, trainable_variables)

    def create_configs_with_grads_and_vars(
            self, grads_and_vars: _GRAD_AND_VARS_TYPE,
            regularization_grads_and_vars: _GRAD_AND_VARS_TYPE,
            all_trainable_variables: List[tf.Variable]
    ) -> List[Tuple[OptimizationConfig, _GRAD_AND_VARS_TYPE]]:
        """
        Filter gradients according to variables used in each config and create
        decoupled configs for regularization terms if decouple_regularization
        flag is set inside of some config

        Parameters
        ----------
        grads_and_vars
            gradients and variables
        regularization_grads_and_vars
            regularization gradients and variables
        all_trainable_variables
            all trainable variables

        Returns
        -------
        optim_configs_with_variables
            list of config to variable pairs

        """
        vars_for_global_optimizer = self._get_vars_for_global_optimizer(
            all_trainable_variables)
        if self._local_configs_with_vars:
            local_optim_configs, vars_for_local_configs = zip(
                *self._local_configs_with_vars)
        else:
            local_optim_configs, vars_for_local_configs = [], []

        optim_configs = [self._global_config] + list(local_optim_configs)
        vars_for_configs = ([vars_for_global_optimizer]
                            + list(vars_for_local_configs))

        optim_configs_with_variables = []
        for each_config, each_vars_for_config in zip(optim_configs,
                                                     vars_for_configs):
            optim_configs_for_vars = (
                self.filter_grads_and_vars_with_decouple_for_config(
                    each_config, each_vars_for_config, grads_and_vars,
                    regularization_grads_and_vars))
            optim_configs_with_variables.extend(optim_configs_for_vars)

        return optim_configs_with_variables

    @staticmethod
    def filter_grads_and_vars_with_decouple_for_config(
            optim_config, vars_for_config, grads_and_vars,
            regularization_grads_and_vars
    ) -> List[Tuple[OptimizationConfig, _GRAD_AND_VARS_TYPE]]:
        """
        Filter gradients and regularization gradients according to variables
        of the optimization config and add the decoupled config for the same
        variables in case if decouple_regularization flag was provided inside
        of optim_config

        Parameters
        ----------
        optim_config
            optimization config
        vars_for_config
            variables to use with optimization config
        grads_and_vars
            list of (gradient, variable) for all variables in the model
        regularization_grads_and_vars
            list of (gradients, variable) from regularization losses for all
            variables in the model

        Returns
        -------
        optim_configs_with_variables
            list of the same config with filtered gradients and variables with
            decoupled config (if it was requested)
        """

        optim_configs_with_variables = []
        grads_and_vars_for_config = optimization_utils.filter_grads_for_vars(
            grads_and_vars, vars_for_config)
        if regularization_grads_and_vars:
            reg_grads_and_vars_for_config = (
                optimization_utils.filter_grads_for_vars(
                    regularization_grads_and_vars, vars_for_config))
        else:
            reg_grads_and_vars_for_config = []
        decouple_regularization = (optim_config.decouple_regularization
                                   and len(reg_grads_and_vars_for_config) >= 1)
        if not decouple_regularization:
            grads_and_vars_for_config = (
                optimization_utils.sum_grads_with_vars(
                    [grads_and_vars_for_config, reg_grads_and_vars_for_config]
                ))
        optim_configs_with_variables.append((optim_config,
                                             grads_and_vars_for_config))
        if decouple_regularization:
            regularization_config = (
                OptimizationHandler._get_regularization_config(optim_config))
            optim_configs_with_variables.append(
                (regularization_config, reg_grads_and_vars_for_config))
        return optim_configs_with_variables

    def _get_train_op(self, grads_and_vars: _GRAD_AND_VARS_TYPE,
                      regularization_grads_and_vars: _GRAD_AND_VARS_TYPE,
                      trainable_variables: List[tf.Variable]
                      ) -> tf.Operation:
        """
        Generate training operation using optimization configs

        Parameters
        ----------
        grads_and_vars
            list of (gradient, variable) for all variables in the model
        regularization_grads_and_vars
            list of (gradients, variable) from regularization losses for all
            variables in the model
        trainable_variables
            list of trainable variables

        Returns
        -------
        train_op
            train operation with update
        """
        optim_configs_to_grads_and_vars = (
            self.create_configs_with_grads_and_vars(
                grads_and_vars, regularization_grads_and_vars,
                trainable_variables))

        train_ops_for_all_optimizers = []
        optim_global_step = self._global_step
        gradients = list(zip(*grads_and_vars))[0]
        gradient_l2_norm = optimization_utils.get_gradient_l2_norm(gradients)
        for each_config, each_grads_and_vars in optim_configs_to_grads_and_vars:
            train_op_for_config = (
                optimization_utils.train_op_with_clip_and_noise(
                    optimizer=each_config.optimizer,
                    grads_and_vars=each_grads_and_vars,
                    gradient_clip=each_config.gradient_clip,
                    gradient_noise_std=each_config.gradient_noise_std,
                    gradient_l2_norm=gradient_l2_norm,
                    global_step=optim_global_step))
            optim_global_step = None
            train_ops_for_all_optimizers.append(train_op_for_config)

        train_op = tf.group(*train_ops_for_all_optimizers)
        return train_op

    @staticmethod
    def _get_regularization_config(optim_config: OptimizationConfig
                                   ) -> OptimizationConfig:
        """
        Construct the regularization config

        Parameters
        ----------
        optim_config
            optimization config to construct the regularization config from

        Returns
        -------
        regularization_config
            optimization config for decoupled regularization

        """
        regularization_optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=optim_config.learning_rate,
            use_locking=True, name="RegularizationGradientDescent")
        regularization_config = optim_config._replace(
            optimizer=regularization_optimizer,
            learning_rate=optim_config.learning_rate)
        return regularization_config

    def _check_variables_are_unique(self, new_variables: List[tf.Variable],
                                    config_name: Optional[str] = None):
        new_variables_set = set(new_variables)
        common_vars = new_variables_set.intersection(
            self._vars_with_local_configs)
        if common_vars:
            if config_name is not None:
                msg = ("Variables {} from config {} already used in other "
                       "optimization config!").format(common_vars, config_name)
            else:
                msg = ("Variables {} already used in other optimization config!"
                       ).format(common_vars)
            raise ValueError(msg)

    @staticmethod
    def _check_global_config(config: OptimizationConfig):
        _check_required_parameter_in_config(config, 'optimizer_name', 'global')
        _check_required_parameter_in_config(config, 'learning_rate', 'global')
        _check_not_allowed_parameter_in_config(config, 'optimizer', 'global')
        _check_not_allowed_parameter_in_config(
            config, 'learning_rate_multiplier', 'global')

    @staticmethod
    def _check_local_config(config: OptimizationConfig,
                            config_name: Optional[str] = None):
        config_name = "local {}".format(config_name) if config_name else 'local'
        _check_not_allowed_parameter_in_config(
            config, 'optimizer', config_name)
        _check_not_allowed_parameter_in_config(
            config, 'learning_rate', config_name)
        _check_not_allowed_parameter_in_config(
            config, 'learning_rate_manipulator', config_name)

    def _initialize_global_step(self):
        self._global_step = tf.train.get_or_create_global_step()

    def _initialize_global_learning_rate(self):
        learning_rate = self._global_config.learning_rate
        learning_rate_manipulator = (
            self._global_config.learning_rate_manipulator)
        if learning_rate_manipulator is not None:
            learning_rate = learning_rate_manipulator.get_current_learning_rate(
                learning_rate, self._global_step)
        learning_rate = tf.convert_to_tensor(learning_rate)
        self._global_learning_rate = learning_rate

    def _create_optimizers(self):
        self._global_config = self._add_optimizer_to_config(self.global_config)
        local_configs_with_optimizers = []
        for each_config_to_var in self._local_configs_with_vars:
            each_config, each_vars = each_config_to_var
            each_config = self._add_optimizer_to_config(each_config)
            local_configs_with_optimizers.append((each_config, each_vars))
        self._local_configs_with_vars = local_configs_with_optimizers

    def _add_optimizer_to_config(self, config: OptimizationConfig
                                 ) -> OptimizationConfig:
        learning_rate = self._global_learning_rate
        if config.learning_rate_multiplier is not None:
            learning_rate = learning_rate * config.learning_rate_multiplier
        optimizer_parameters = config.optimizer_parameters or {}
        optimizer = (
            tf_objects_factory.optimizer_factory_from_name_and_parameters(
                config.optimizer_name,
                learning_rate=learning_rate,
                **optimizer_parameters))
        config = config._replace(optimizer=optimizer,
                                 learning_rate=learning_rate)
        return config

    def _get_vars_for_global_optimizer(self, all_trainable_varibles):
        vars_for_global_optimizer = set(all_trainable_varibles).difference(
            self._vars_with_local_configs)
        return vars_for_global_optimizer


def _check_required_parameter_in_config(config: OptimizationConfig,
                                        name: str, config_name: str):
    """
    Check if the required parameter is set

    Parameters
    ----------
    config
        config
    name
        parameter name
    config_name
        name of the config to print inside of the message

    Raises
    ------
    ValueError
        if attribute with name is None in the config
    """
    if getattr(config, name) is None:
        msg = ("{} must be set for {} optimization config! "
               "(got {})".format(name, config_name, config))
        raise ValueError(msg)


def _check_not_allowed_parameter_in_config(config: OptimizationConfig,
                                           name: str, config_name: str):
    """
    Check if the not allowed parameter is None

    Parameters
    ----------
    config
        config
    name
        parameter name
    config_name
        name of the config to print inside of the message

    Raises
    ------
    ValueError
        if attribute with name is not None in the config
    """
    if getattr(config, name) is not None:
        msg = ("{} cannot be set for {} optimization config! "
               "(got {})".format(name, config_name, config))
        raise ValueError(msg)
