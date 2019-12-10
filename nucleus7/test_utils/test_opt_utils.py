# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Test utils for optimization
"""

from typing import List
from typing import Tuple

import numpy as np
import tensorflow as tf


def create_grads_and_vars_np(sizes: List[list]
                             ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create numpy grads and vars of sizes

    Parameters
    ----------
    sizes
        sizes of variables

    Returns
    -------
    grads_and_vars
        gradients and variables
    """
    grads_and_vars = []
    for each_size in sizes:
        grad = np.random.randn(*each_size).astype(np.float32)
        var = np.random.randn(*each_size).astype(np.float32)
        grads_and_vars.append((grad, var))
    return grads_and_vars


def create_tower_grads_and_vars_np(
        num_towers: int, sizes: List[list]
) -> List[List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Create tower numpy grads and vars of sizes

    Parameters
    ----------
    num_towers
        number of towers
    sizes
        sizes of variables

    Returns
    -------
    tower_grads_and_vars
        tower gradients and variables
    """
    first_tower = create_grads_and_vars_np(sizes)
    if num_towers == 1:
        return [first_tower]
    tower_grads_and_vars = [first_tower]
    variables = list(zip(*first_tower))[1]
    for _ in range(num_towers - 1):
        grads_and_vars = create_grads_and_vars_np(sizes)
        grads = list(zip(*grads_and_vars))[0]
        grads_and_vars = list(zip(grads, variables))
        tower_grads_and_vars.append(grads_and_vars)
    return tower_grads_and_vars


def convert_grads_to_tf(grads_np: List[np.ndarray]) -> List[tf.Tensor]:
    """
    Convert numpy gradients to tensorflow

    Parameters
    ----------
    grads_np
        list of numpy gradients

    Returns
    -------
    tensorflow_grads
        list of tensorflow gradients
    """
    return [tf.constant(each_grad) for each_grad in grads_np]


def create_tf_vars_from_np(vars_np: List[np.ndarray]) -> List[tf.Variable]:
    """
    Create tensorflow variables from numpy arrays

    Parameters
    ----------
    vars_np
        list of numpy variables

    Returns
    -------
    tensorflow_variables
        list of tensorflow variables
    """
    vars_tf = [tf.Variable(each_var_np, name="var_{}".format(i))
               for i, each_var_np in enumerate(vars_np)]
    return vars_tf


def convert_grads_and_vars_to_tf(
        grads_and_vars_np: List[Tuple[np.ndarray, np.ndarray]]
) -> List[Tuple[tf.Tensor, tf.Variable]]:
    """
    Convert numpy gradients and variables to tensorlfow

    Parameters
    ----------
    grads_and_vars_np
        numpy grads and vars

    Returns
    -------
    grads_and_vars_tf
        tensorflow grads and vars
    """
    grads_np, vars_np = zip(*grads_and_vars_np)
    grads_tf = convert_grads_to_tf(grads_np)
    vars_tf = create_tf_vars_from_np(vars_np)
    grads_and_vars_tf = list(zip(grads_tf, vars_tf))
    return grads_and_vars_tf


def convert_tower_grads_and_vars_to_tf(
        tower_grads_and_vars_np: List[List[Tuple[np.ndarray, np.ndarray]]]
) -> List[List[Tuple[tf.Tensor, tf.Variable]]]:
    """
    Convert tower numpy grads and vars to tensorflow

    Parameters
    ----------
    tower_grads_and_vars_np
        tower numpy grads and vars

    Returns
    -------
    tower_grads_and_vars_tf
        tensorflow tower grads and vars
    """
    first_tower = tower_grads_and_vars_np[0]
    vars_np = list(zip(*first_tower))[1]
    vars_tf = create_tf_vars_from_np(vars_np)
    tower_grads_and_vars_tf = []
    for each_grads_and_var_np in tower_grads_and_vars_np:
        grads_np = list(zip(*each_grads_and_var_np))[0]
        grads_tf = convert_grads_to_tf(grads_np)
        grads_and_vars_tf = list(zip(grads_tf, vars_tf))
        tower_grads_and_vars_tf.append(grads_and_vars_tf)
    return tower_grads_and_vars_tf
