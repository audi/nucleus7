# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Utils to incorporate mlflow
"""

from functools import partial
from functools import wraps
import logging
import os
import re
import time
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Union
from unittest.mock import patch
import warnings

import mlflow
import mlflow.tensorflow as mlflow_tf
from mlflow.tracking.client import MlflowClient
import tensorflow as tf
# pylint: disable=no-name-in-module
# python is a part of tensorflow
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

from nucleus7.core import project
# pylint: enable=no-name-in-module
from nucleus7.utils import project_utils

PARAMETER_SEPARATOR = "/"
METRIC_SEPARATOR = "--"
_MLFLOW_SUPPORTED_SYMBOLS = "[^A-Za-z0-9._/-]+"
_SOURCE_VERSION_FILE_PREFIX = "SOURCE_VERSION"


def create_mlflow_experiment_and_start_run(run_method: Callable) -> Callable:
    """
    Wrapper about run method to start the mlflow experiment and log
    parameters to it.

    Will also log the configs as artifact

    Parameters
    ----------
    run_method
        run method to decorate

    Returns
    -------
    wrapper
        wrapped method
    """

    def wrapped(*args, **kwargs):
        source_name = "nucleus7"
        active_project = project.get_active_project()
        run_name = active_project.project_type
        entry_point_name = active_project.entry_point
        project_dir = active_project.project_dir
        create_new_or_continue_experiment(project_dir)
        source_versions = project_utils.get_source_versions()
        with mlflow.start_run(run_name=run_name):
            log_config_parameter_to_mlflow("PROJECT_DIR", project_dir)
            _log_source_versions_to_mlflow(source_versions)
            run_method(*args, **kwargs)
            mlflow.set_tag("source", source_name)
            mlflow.set_tag("entry_point", entry_point_name)

    return wrapped


def log_nucleotide_exec_time_to_mlflow(
        method_name: Optional[str] = None,
        prefix: str = "exec_time",
        log_on_train: bool = False,
        log_on_eval: bool = False,
        log_on_infer: bool = True) -> Callable:
    """
    Log the time of the function to mlflow

    Parameter will be logged to metrics with the name
    "performance--{prefix}, [s]--{object_name}--{mode (if was set)}--
    {method_name (if is not None)}"
    and value of time needed to call it

    Parameters
    ----------
    prefix
        prefix to use for parameter name
    method_name
        name of the method
    log_on_train
        if the performance time should be logged during training
    log_on_eval
        if the performance time should be logged during evaluation
    log_on_infer
        if the performance time should be logged during inference

    Returns
    -------
    wrapper
        wrapped method
    """

    def wrapper(function: Callable):
        @wraps(function)
        def wrapped(self, *args, **kwargs):
            if not _decide_if_log_based_on_mode(
                    self, log_on_train, log_on_eval, log_on_infer):
                return function(self, *args, **kwargs)

            name_with_mode = _get_object_name_with_mode(self, METRIC_SEPARATOR)
            method_name_ = method_name or function.__name__
            log_param_name = METRIC_SEPARATOR.join(
                ["performance", prefix, name_with_mode, method_name_])
            start_time = time.time()
            result = function(self, *args, **kwargs)
            execution_time = time.time() - start_time
            log_metric_to_mlflow(log_param_name, execution_time)

            logger = logging.getLogger(__name__)
            logger.debug("%s was executed in %0.3f s",
                         name_with_mode, execution_time)
            return result

        return wrapped

    return wrapper


def create_new_or_continue_experiment(project_dir: str):
    """
    Creates a new experiment or continues already existing one.

    Experiment name is the name of the project_dir

    Parameters
    ----------
    project_dir
        project directory
    """
    mlflow.set_tracking_uri(None)
    experiment_name = project_utils.get_project_name_from_directory(project_dir)
    if "MLFLOW_TRACKING_URI" not in os.environ:
        tracking_uri = os.path.join(os.path.split(project_dir)[0], "mlruns")
        tracking_uri = os.path.realpath(tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_config_parameter_to_mlflow(param_name: str, param_value: Any):
    """
    Log the parameter with its value to mlflow.

    If value is malformed, it will be ignored and logging notification will
    be added

    Parameters
    ----------
    param_name
        parameter name to log
    param_value
        value to log
    """
    logger = logging.getLogger(__name__)
    param_name = _replace_not_allowed_symbols(param_name)
    if isinstance(param_value, str) and os.path.exists(param_value):
        param_value = os.path.realpath(param_value)
    if mlflow.active_run() is None:
        _warn_about_no_run()
        return
    try:
        mlflow.log_param(param_name, param_value)
    # pylint: disable=invalid-name
    # is common practice to call exceptions as e
    # pylint: disable=broad-except
    # this is due to the mlflow itself, they raise this Exception
    except Exception as e:
        logger.warning("Parameter %s has malformed value %s and so will not "
                       "be logged to mlflow!", param_name, param_value)
        logger.warning("Original exception: %s", e)


def log_metric_to_mlflow(metric_name: str, metric_value: Any):
    """
    Log the metric with its value to mlflow.

    If value is malformed, it will be ignored and logging notification will
    be added

    Parameters
    ----------
    metric_name
        metric name to log
    metric_value
        value to log
    """
    logger = logging.getLogger(__name__)
    metric_name = _replace_not_allowed_symbols(metric_name)
    metric_name = _format_metric_name(metric_name)
    if mlflow.active_run() is None:
        _warn_about_no_run()
        return
    try:
        _log_metric(metric_name, metric_value)
    # pylint: disable=invalid-name
    # is common practice to call exceptions as e
    # pylint: disable=broad-except
    # this is due to the mlflow itself, they raise this Exception
    except Exception as e:
        logger.warning("Metric %s has malformed value %s and so will not "
                       "be logged to mlflow!", metric_name, metric_value)
        logger.warning("Original exception: %s", e)


def log_project_artifacts_to_mlflow(function: Callable):
    """
    Log the artifact to mlflow

    Parameters
    ----------
    function
        function to wrap
    """

    @wraps(function)
    def wrapped(*args, **kwargs):
        if mlflow.active_run() is None:
            _warn_about_no_run()
            return function(*args, **kwargs)
        artifacts_path = project.get_active_artifacts_directory()
        artifacts_path_realpath = os.path.realpath(artifacts_path)
        mlflow.log_artifacts(artifacts_path_realpath)
        return function(*args, **kwargs)

    return wrapped


# pylint: disable=invalid-name
# this is method, not a constant, and is used inside of the patch
_load_model_with_clear_devices = partial(
    tf.saved_model.loader.load, clear_devices=True)


# pylint: enable=invalid-name

# this is needed because of the bug in mlflow and tensorflow estimator
# estimator saves models with devices and then mlflow restores them with same
# devices to validate the saved model, but then estimator increases the device
# and so wrong device is used and exception is raised.
# must be removed as it is resolved
# TODO(oleksandr.vorobiov@audi.de): remove patch when it is resolved in
#  dependencies
@patch("tensorflow.saved_model.loader.load",
       wraps=_load_model_with_clear_devices)
def log_saved_model(saved_model_path: Union[bytes, str],
                    global_step: int,
                    saved_model_load_fn: Callable):
    """
    Log all the saved models to mlflow

    Parameters
    ----------
    saved_model_path
        path to saved model
    global_step
        global step for saved model
    """
    # pylint: disable=unused-argument
    # saved_model_load_fn is coming from patch
    if mlflow.active_run() is None:
        _warn_about_no_run()
        return
    if isinstance(saved_model_path, bytes):
        saved_model_path = saved_model_path.decode()
    saved_model_tag = os.path.split(saved_model_path)[-1]
    artifact_path = os.path.join("models", saved_model_tag)
    mlflow_tf.log_model(
        tf_saved_model_dir=saved_model_path,
        tf_meta_graph_tags=[tag_constants.SERVING],
        tf_signature_def_key=
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
        artifact_path=artifact_path)
    mlflow_artifacts_path = mlflow.get_artifact_uri()
    saved_model_artifact_path = os.path.join(
        mlflow_artifacts_path, artifact_path)
    project_utils.log_exported_model_info(
        saved_model_artifact_path, global_step)


def _log_source_versions_to_mlflow(source_versions: Dict[str, str]):
    for each_source_name, each_source_version in source_versions.items():
        parameter_name = "_".join(
            [_SOURCE_VERSION_FILE_PREFIX, each_source_name])
        log_config_parameter_to_mlflow(parameter_name, each_source_version)


def _warn_about_no_run():
    msg = ("No active mlflow run found. No parameters and no metrics will be "
           "logged")
    warnings.warn(msg, stacklevel=2)


def _get_object_name_with_mode(instance: object, separator: str) -> str:
    object_name = getattr(instance, "name", instance.__class__.__name__)
    mode = getattr(instance, "mode", None)
    if mode is None:
        return object_name
    return separator.join([object_name, mode])


def _replace_not_allowed_symbols(name: str, replace_symbol="-") -> str:
    without_duplicated_slashes = re.sub("[/]{2,}", '/', name)
    return re.sub(_MLFLOW_SUPPORTED_SYMBOLS, replace_symbol,
                  without_duplicated_slashes)


def _format_metric_name(name: str):
    name_formatted = re.sub(PARAMETER_SEPARATOR, METRIC_SEPARATOR, name)
    return name_formatted


def _decide_if_log_based_on_mode(instance, log_on_train, log_on_eval,
                                 log_on_infer) -> bool:
    mode = getattr(instance, "mode", None)
    if not mode:
        return True
    if log_on_train and mode == tf.estimator.ModeKeys.TRAIN:
        return True
    if log_on_eval and mode == tf.estimator.ModeKeys.EVAL:
        return True
    if log_on_infer and mode == tf.estimator.ModeKeys.PREDICT:
        return True
    return False


def _log_metric(metric_name: str, metric_value: float):
    # this is a hack to have larger time stamp as original mlflow
    run_id = mlflow.active_run().info.run_uuid
    time_stamp = _get_timestamp()
    MlflowClient().log_metric(run_id, metric_name, metric_value, time_stamp)


def _get_timestamp():
    return int(time.time() * 100000)
