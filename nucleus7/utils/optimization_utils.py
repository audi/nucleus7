# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Utils for optimization
"""

import itertools
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import tensorflow as tf

from nucleus7.utils import tf_ops

# pylint: disable=invalid-name
# these are type constants, not a class
_GRAD_AND_VARS_TYPE = List[Tuple[tf.Tensor, tf.Variable]]


# pylint: enable=invalid-name


def average_grads_and_vars_from_multiple_devices(
        tower_grads_and_vars: List[_GRAD_AND_VARS_TYPE],
        consolidation_device: str,
        name_of_name_scope: str = 'gradient_averaging'
) -> _GRAD_AND_VARS_TYPE:
    """
    Average gradients over all devices (towers)

    Parameters
    ----------
    tower_grads_and_vars
        list of lists of gradients with variables for different devices (towers)
    consolidation_device
        device to place the averaged gradients
    name_of_name_scope
        name of name_scope to encapsulate this calculation

    Returns
    -------
    grads_and_vars
        list of (grads, vars) tuples
    """
    grads_and_vars_len = [len(each_grads_and_vars)
                          for each_grads_and_vars in tower_grads_and_vars]
    assert len(set(grads_and_vars_len)) == 1, (
        "Number of grads and vars pairs must be equal to all "
        "devices! (got lengths: {})".format(grads_and_vars_len)
    )
    with tf.name_scope(name_of_name_scope), tf.device(consolidation_device):
        grads_and_vars_averaged = sum_grads_with_vars(
            tower_grads_and_vars, use_mean=True)
    return grads_and_vars_averaged


def train_op_with_clip_and_noise(optimizer: tf.train.Optimizer,
                                 grads_and_vars: _GRAD_AND_VARS_TYPE,
                                 global_step: Optional[tf.Tensor] = None,
                                 gradient_clip: Optional[float] = None,
                                 gradient_noise_std: Optional[float] = None,
                                 gradient_l2_norm: Optional[tf.Tensor] = None
                                 ) -> tf.Operation:
    """
    Produce train op for gradients and variables with
    gradient clip and adding of gradient noise if they were provided inside of
    optim config

    Parameters
    ----------
    optimizer
        optimizer to use
    grads_and_vars
        list of (gradient, variable)
    global_step
        global step to use in the optimizer; Caution: provide global_step only
        once, if you execute this method multiple times in one session
    gradient_clip
        gradient clip value
    gradient_noise_std
        standard deviation of the noise to add to gradients
    gradient_l2_norm
        gradient l2 norm used for the gradient clipping

    Returns
    -------
    train_op
        training operation, which can be used inside of session run
    """
    if gradient_clip is not None:
        grads_and_vars = clip_grads_and_vars(
            grads_and_vars, gradient_clip, gradient_l2_norm)
    if gradient_noise_std is not None:
        grads_and_vars = add_noise_to_grads_and_vars(
            grads_and_vars, gradient_noise_std)
    train_op = optimizer.apply_gradients(
        grads_and_vars, global_step=global_step)
    return train_op


def clip_grads_and_vars(grads_and_vars: _GRAD_AND_VARS_TYPE,
                        gradient_clip: float,
                        gradient_l2_norm: Optional[tf.Tensor] = None
                        ) -> _GRAD_AND_VARS_TYPE:
    """
    Clip all the gradients according to global normal with gradient_clip

    Parameters
    ----------
    grads_and_vars
        list of (gradient, variable)
    gradient_clip
        value to clip
    gradient_l2_norm
        gradient l2 norm used for the gradient clipping

    Returns
    -------
    grads_and_vars
        list of (clipped gradient, variable)
    """
    grads, variables = zip(*grads_and_vars)
    if gradient_l2_norm is None:
        grads_clipped, gradient_l2_norm = tf.clip_by_global_norm(
            grads, gradient_clip)
    else:
        grads_clipped = [each_grad * gradient_clip
                         / tf.maximum(gradient_l2_norm, gradient_clip)
                         for each_grad in grads]
    grads_and_vars_clipped = list(zip(grads_clipped, variables))
    return grads_and_vars_clipped


def add_noise_to_grads_and_vars(grads_and_vars: _GRAD_AND_VARS_TYPE,
                                gradient_noise_std: float
                                ) -> _GRAD_AND_VARS_TYPE:
    """
    Add normal random noise to gradients

    Parameters
    ----------
    grads_and_vars
        list of (gradient, variable)
    gradient_noise_std
        standard deviation of the noise to add to gradients

    Returns
    -------
    grads_and_vars
        list of (gradient with noise, variable)
    """
    grads_and_vars_with_noise = []
    for each_grad, each_var in grads_and_vars:
        each_grad_clipped = tf_ops.add_gradient_noise(
            each_grad, gradient_noise_std)
        grads_and_vars_with_noise.append((each_grad_clipped, each_var))
    return grads_and_vars_with_noise


def sum_grads_with_vars(
        grads_and_vars_to_combine: List[_GRAD_AND_VARS_TYPE],
        use_mean: bool = False) -> _GRAD_AND_VARS_TYPE:
    """
    Sum the gradients for same variables from list of of lists of
    (gradient, variable)

    Parameters
    ----------
    grads_and_vars_to_combine
        list of list of (gradient, variable) to combine
    use_mean
        if the mean istead of sum should be used

    Returns
    -------
    grads_and_vars
        list of (combined gradients, variable)
    """
    grads_and_vars_combined = []
    vars_to_all_grads = {}
    for grad, var in itertools.chain(*grads_and_vars_to_combine):
        if grad is not None:
            vars_to_all_grads.setdefault(var, []).append(grad)
    for var, grads in vars_to_all_grads.items():
        if len(grads) == 1:
            grad_combined = grads[0]
        else:
            grad_combined = tf.add_n(grads)
            if use_mean:
                grad_combined = tf.multiply(grad_combined, 1. / len(grads))
        grads_and_vars_combined.append((grad_combined, var))
    return grads_and_vars_combined


def filter_grads_for_vars(grads_and_vars, vars_to_filter):
    """
    Select only (gradient, variable) tuples, where variable belongs to
    vars_to_filter

    Parameters
    ----------
    grads_and_vars
        all the gradients and variables as list of (grad, var)
    vars_to_filter
        variables to filter
    Returns
    -------
    grads_and_vars
        list of (gradients, variable from vars_to_filter)
    """
    grads_and_vars_filtered = []
    vars_to_filter_set = set(vars_to_filter)
    for each_grad, each_var in grads_and_vars:
        if each_var in vars_to_filter_set:
            grads_and_vars_filtered.append((each_grad, each_var))
    return grads_and_vars_filtered


def get_gradient_l2_norm(gradients: Sequence[tf.Tensor],
                         name_of_name_scope: str = 'grad_l2_norm') -> tf.Tensor:
    """
    Calculate l2 norm for of the gradients

    Parameters
    ----------
    gradients
        list of gradients
    name_of_name_scope
        name of the name scope

    Returns
    -------
    gradient_l2_norm
        l2 norm of the gradients
    """
    with tf.name_scope(name_of_name_scope):
        gradients_flatten = tf.concat([
            tf.reshape(each_grad, [-1]) for each_grad in gradients], 0)
        gradient_l2_norm = tf.reduce_sum(gradients_flatten ** 2) ** 0.5
    return gradient_l2_norm
