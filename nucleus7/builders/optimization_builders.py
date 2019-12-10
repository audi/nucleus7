# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Builders for optimization objects
"""
import copy

import tensorflow as tf

from nucleus7.optimization import configs as opt_configs
from nucleus7.optimization.learning_rate_manipulator import ConstantLearningRate
from nucleus7.optimization.learning_rate_manipulator import (
    LearningRateManipulator)
from nucleus7.optimization.learning_rate_manipulator import TFLearningRateDecay
from nucleus7.utils import deprecated
from nucleus7.utils.utils import get_member_from_package_and_member_names


@deprecated.replace_deprecated_parameter(
    "learning_rate_decay", "learning_rate_manipulator", required=False)
def build_learning_rate_manipulator(class_name: str = None,
                                    **manipulator_params) -> tf.Tensor:
    """
    Builder for the learning rate manipulator

    Parameters
    ----------
    class_name
        class name of the learning rate manipulator. If it is 'tf',
        TFLearningRateDecay will be used. Default is ConstantLearningRate
    **manipulator_params
        parameters of the learning rate manipulator, which will be passed to
        manipulation function

    Returns
    -------
    learning_rate_manipulator
        built learning rate manipulator
    """
    class_name = class_name or ''
    if not class_name:
        learning_rate_manipulator_cls = ConstantLearningRate
    elif class_name.lower() == 'tf':
        learning_rate_manipulator_cls = TFLearningRateDecay
    else:
        learning_rate_manipulator_cls = (
            get_member_from_package_and_member_names(class_name))
    if not issubclass(learning_rate_manipulator_cls, LearningRateManipulator):
        raise ValueError('The given learning rate manipulator does not inherit '
                         'from LearningRateManipulator')
    learning_rate_manipulator = learning_rate_manipulator_cls(
        **manipulator_params).build()
    return learning_rate_manipulator


def build_optimization_config(
        config: dict, is_global: bool = True) -> opt_configs.OptimizationConfig:
    """
    Build the optimization config object based on the config.

    Will also build the learning rate manipulator if its configuration was
    provided

    Parameters
    ----------
    config
        config of the optimization config
    is_global
        defines if it must be treated as a global optimization or not

    Returns
    -------
    optimization_config
        optimization config
    """
    config = copy.deepcopy(config)
    learning_rate_manipulator_config = config.pop(
        'learning_rate_manipulator', None)
    if learning_rate_manipulator_config:
        learning_rate_manipulator = build_learning_rate_manipulator(
            **learning_rate_manipulator_config)
    else:
        learning_rate_manipulator = None
    optimization_config = opt_configs.create_and_validate_optimization_config(
        learning_rate_manipulator=learning_rate_manipulator,
        is_global=is_global, **config)
    return optimization_config
