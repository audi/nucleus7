# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Utils for tensorflow run
"""

from typing import Optional

import tensorflow as tf
# pylint: disable=no-name-in-module
# python is a part of tensorflow
from tensorflow.python.estimator import run_config as run_config_lib


# pylint: enable=no-name-in-module


def get_run_type(run_config: Optional[tf.estimator.RunConfig] = None) -> str:
    """
    Get the estimator run_config type

    Parameters
    ----------
    run_config
        run config

    Returns
    -------
    run_config_type
        type of run config in format {task_type}:{task_id}

    """
    run_config = _maybe_create_config(run_config)
    if is_local_run(run_config):
        return 'chief:0'
    task_type = run_config.task_type
    task_id = run_config.task_id
    run_dir = '{}:{}'.format(task_type, task_id)
    return run_dir


def is_chief(run_config: Optional[tf.estimator.RunConfig] = None) -> bool:
    """
    Check if current run of estimator is a chief

    Parameters
    ----------
    run_config
        run config

    Returns
    -------
    flag
        True if is chief run

    """
    run_config = _maybe_create_config(run_config)
    run_is_chief = False
    if is_local_run(run_config):
        run_is_chief = True
    else:
        if run_config.task_type == run_config_lib.TaskType.CHIEF:
            run_is_chief = True
    return run_is_chief


def is_local_run(run_config: Optional[tf.estimator.RunConfig] = None) -> bool:
    """
    Check if current run of estimator is a local, e.g. not distributed

    Parameters
    ----------
    run_config
        run config

    Returns
    -------
    flag
        True if is local run
    """
    run_config = _maybe_create_config(run_config)
    if (not run_config.cluster_spec and
            run_config.task_type != run_config_lib.TaskType.EVALUATOR):
        return True
    return False


def is_evaluator_run(run_config: Optional[tf.estimator.RunConfig] = None
                     ) -> bool:
    """
    Check if current run of estimator is a evaluator run

    Parameters
    ----------
    run_config
        run config

    Returns
    -------
    flag
        True if is evaluator run
    """
    run_config = _maybe_create_config(run_config)
    return run_config.task_type == run_config_lib.TaskType.EVALUATOR


def get_cluster_spec(run_config: Optional[tf.estimator.RunConfig] = None
                     ) -> dict:
    """
    Returns cluster spec from run_config

    Parameters
    ----------
    run_config
        run config

    Returns
    -------
    cluster_spec
        cluster spec
    """
    run_config = _maybe_create_config(run_config)
    return run_config.cluster_spec.as_dict()


def select_dataset_modes_for_run(
        run_config: Optional[tf.estimator.RunConfig] = None) -> list:
    """
    Get the dataset types for current estimator run

    Parameters
    ----------
    run_config
        run config

    Returns
    -------
    list_of_dataset_modes
        ['train', 'eval'] if is local run or ['train'] for worker and chief
        and ['eval'] for evaluator
    """
    run_config = _maybe_create_config(run_config)
    if is_local_run(run_config):
        return ['train', 'eval']
    task_type = run_config.task_type
    if task_type == run_config_lib.TaskType.EVALUATOR:
        return ['eval']
    return ['train']


def _maybe_create_config(run_config):
    if run_config is None:
        run_config = tf.estimator.RunConfig()
    return run_config
