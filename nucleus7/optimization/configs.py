# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Configs for optimization
"""

from collections import namedtuple
import logging
from typing import Optional

from nucleus7.optimization.learning_rate_manipulator import (
    LearningRateManipulator)
from nucleus7.utils import deprecated
from nucleus7.utils import utils

OptimizationConfig = namedtuple(
    'OptimizationConfig',
    ['optimizer_parameters', 'learning_rate', 'learning_rate_manipulator',
     'gradient_clip', 'gradient_noise_std', 'optimizer_name',
     'decouple_regularization', 'learning_rate_multiplier',
     'optimizer'])


def create_and_validate_optimization_config(
        optimizer_name: Optional[str] = None,
        optimizer_parameters: Optional[dict] = None,
        learning_rate: Optional[float] = None,
        learning_rate_manipulator: Optional[LearningRateManipulator] = None,
        learning_rate_multiplier: Optional[float] = None,
        gradient_clip: Optional[float] = None,
        gradient_noise_std: Optional[float] = None,
        decouple_regularization: bool = False,
        is_global: bool = False,
        **deprecated_optimizer_parameters
) -> OptimizationConfig:
    """
    Create configuration for optimization

    Parameters
    ----------
    optimizer_name
        name of the optimizer to use
    optimizer_parameters
        parameters to provide for the optimizer constructor
    learning_rate
        learning rate to use; in case of use of learning_rate_manipulator,
        this will be a initial_learning_rate;
        is valid only for global optimization config
    learning_rate_manipulator
        will change the learning rate during the training;
        is valid only for global optimization config
    learning_rate_multiplier
        global learning_rate will be multiplied this value and will be used
        for this optimizer; is valid only for local optimization configs
    gradient_clip
        will rescale the gradients that the l2 norm of the gradients is not more
        than this value; for this method :func:`tf.clip_by_global_norm` is used
    gradient_noise_std
        will add a normal noise this std to all the gradients in this config
    decouple_regularization
        if the regularization gradients, e.g. gradients from regularization
        losses, must be decoupled and use a GradientDescentOptimizer with
        current learning rate; otherwise all the gradients will be summed up
    **deprecated_optimizer_parameters
        optimizer_parameters in deprecated form, e.g. if you not provide it
        as a optimizer_parameters; is there for backward compatibility
    is_global
        specifies whether this is the global optimization configuration

    Returns
    -------
    optimization config
        configuration used for optimization

    Raises
    ------
    AssertionError
        if learning_rate_multiplier is set for global optimization
    AssertionError
        if optimizer_name is not set for global optimization
    AssertionError
        if learning rate is not set for global optimization
    AssertionError
        if learning_rate_decay is set and is not a dict for global optimization
    AssertionError
        if learning rate is set for local optimization (is_global == False)
    ValueError
        if the optimizer parameters were provided inside of the
        optimizer_parameter section and also inside of optimization_parameters
        itself
    """
    # pylint: disable=too-many-arguments
    # all of that arguments are needed for OptimizationConfig
    if is_global:
        assert optimizer_name is not None, (
            "Specify optimizer_name in global optimization config")
        assert learning_rate is not None, (
            "Specify learning_rate in optimization_parameters for global "
            "optimizer")
        assert learning_rate_multiplier is None, (
            "learning_rate_multiplier cannot be set in the global "
            "optimization config")
    else:
        assert learning_rate is None, (
            "learning_rate inside of local config "
            "is not allowed!")
        assert learning_rate_manipulator is None, (
            "learning_rate_manipulator inside of local optimization config "
            "is not allowed!")

    if deprecated_optimizer_parameters and optimizer_parameters:
        msg = ("Do not mix a deprecated style of providing the optimizer "
               "parameters just as parameters inside of "
               "optimization_config with new one by providing it "
               "explicitly inside of optimizer_parameters session "
               "(deprecated provided: {}, "
               "optimizer_parameters section {})"
               ).format(deprecated_optimizer_parameters, optimizer_parameters)
        raise ValueError(msg)
    if deprecated_optimizer_parameters:
        for each_key in deprecated_optimizer_parameters:
            deprecated.warning_deprecated_config_param(
                real_name="optimizer_parameters::{}".format(each_key),
                deprecated_name=each_key,
                additional_instructions=(
                    "Provide optimizer_parameters using optimizer_parameters "
                    "session"))
        optimizer_parameters = deprecated_optimizer_parameters

    if not is_global:
        learning_rate_multiplier = learning_rate_multiplier or 1.0
    optimization_config = OptimizationConfig(
        optimizer=None,
        optimizer_parameters=optimizer_parameters,
        learning_rate=learning_rate,
        learning_rate_manipulator=learning_rate_manipulator,
        gradient_clip=gradient_clip,
        gradient_noise_std=gradient_noise_std,
        optimizer_name=optimizer_name,
        decouple_regularization=decouple_regularization,
        learning_rate_multiplier=learning_rate_multiplier)
    return optimization_config


def merge_optimization_configs(
        global_optimization_config: OptimizationConfig,
        local_optimization_config: OptimizationConfig
) -> OptimizationConfig:
    """
    Create an optimization config out of the global and local optimization
    configs. All of the values, except of learning_rate* parameters will be
    overwritten inside of local_optimization_config from global one if they
    were set to None; if optimizer_name is provided, then optimizer_parameters
    will be not updated, and if not provided, they will be updated with global,
    if parameters are unset

    Parameters
    ----------
    global_optimization_config
        global optimization configuration
    local_optimization_config
        local optimization configuration

    Returns
    -------
    local_optimization_config
        merged optimization configuration

    """
    assert global_optimization_config.learning_rate is not None, (
        "learning_rate inside of global configuration should be set!")
    assert local_optimization_config.learning_rate_multiplier is not None, (
        "learning_rate_multiplier inside of local configuration should be set!")

    logger = logging.getLogger(__name__)
    if not local_optimization_config.optimizer_name:
        if local_optimization_config.optimizer_parameters:
            logger.info("Use global optimizer with local parameters")
            local_optimization_config = local_optimization_config._replace(
                optimizer_name=global_optimization_config.optimizer_name)
        else:
            logger.info("Use global optimizer with global parameters")
            local_optimization_config = local_optimization_config._replace(
                optimizer_name=global_optimization_config.optimizer_name,
                optimizer_parameters=
                global_optimization_config.optimizer_parameters)

    parameters_to_update = [
        'gradient_noise_std', 'decouple_regularization', 'gradient_clip']
    for parameter_name in parameters_to_update:
        global_value = getattr(global_optimization_config, parameter_name)
        local_optimization_config = (
            utils.maybe_update_undefined_parameter_in_config(
                local_optimization_config, parameter_name, global_value))
    return local_optimization_config
