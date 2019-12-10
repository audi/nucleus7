# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Configurations used in coordinator
"""
from collections import namedtuple
import logging
import os
from typing import Optional
from typing import Union

import tensorflow as tf

from nucleus7.core.project_dirs import ProjectDirs
from nucleus7.utils import tf_utils

TrainerRunConfig = namedtuple(
    'TrainerRunConfig',
    ['batch_size', 'devices', 'random_seed', 'num_epochs',
     'iterations_per_epoch', 'variable_strategy',
     'predictions_have_variable_shape', 'continue_training',
     'profile_hook_config'])

TrainerSaveConfig = namedtuple(
    'TrainerSaveConfig',
    ['inference_inputs_have_variable_shape', 'exports_to_keep',
     'save_summary_steps', 'max_outputs_tb'])

InferenceLoadConfig = namedtuple('InferenceLoadConfig',
                                 ['saved_model', 'meta_graph', 'checkpoint'])

RunIterationInfo = namedtuple(
    'RunIterationInfo',
    ['epoch_number', 'iteration_number', 'execution_time',
     'is_last_iteration', 'session_run_context'])
RunIterationInfo.__new__.__defaults__ = (-1, -1, -1, False, None)

TensorrtConfig = namedtuple(
    "TensorrtConfig",
    ["use_tensorrt", "max_batch_size", "max_workspace_size_bytes",
     "precision_mode", "minimum_segment_size", "is_dynamic_op",
     "maximum_cached_engines", "cached_engine_batch_sizes"]
)

_InferenceRunConfig = namedtuple(
    'InferenceRunConfig',
    ['batch_size', 'devices', 'postprocessors_to_use', 'random_seed',
     'prefetch_buffer_size', 'use_multiprocessing', 'continue_last'])

INFERENCE_GRAPH_FILE_NAME = "graph_inference.meta"
INPUT_OUTPUT_NAMES_FILE_NAME = "input_output_names.json"


class InferenceRunConfig(_InferenceRunConfig):
    """
    Configuration for inference run

    Parameters
    ----------
    batch_size
        batch size per device
    postprocessors_to_use
        which postprocessors should be used for predictions; if not provided,
        whole predictions collection will be used
    random_seed
        graph based random seed for tensorflow graph
    prefetch_buffer_size
        number of batches to prefetch; must be >= 1
    use_multiprocessing
        if the multiprocessing must be used for inference, e.g. data prefetching
        in one process, network in other process and callbacks in the third one
    """

    def __new__(cls, batch_size: int,
                postprocessors_to_use: Union[list, None] = None,
                random_seed: int = 512, prefetch_buffer_size: int = 5,
                use_multiprocessing: bool = True,
                continue_last: bool = False):
        assert isinstance(batch_size, int), (
            "Batch size should be of type int! ({})".format(batch_size))
        assert prefetch_buffer_size >= 1, (
            "Prefetch size must be >= 1! ({})".format(prefetch_buffer_size))
        devices = tf_utils.get_available_gpus()
        if devices:
            devices = devices[:1]
        else:
            devices = ['/cpu:0']
        return super(InferenceRunConfig, cls).__new__(
            cls, batch_size, devices, postprocessors_to_use,
            random_seed, prefetch_buffer_size, use_multiprocessing,
            continue_last)


def create_and_validate_trainer_run_config(
        batch_size: Union[int, dict],
        devices: Union[list, None] = None,
        random_seed: Optional[int] = 419,
        num_epochs: int = 100,
        samples_per_epoch: Union[int, dict] = None,
        iterations_per_epoch: Union[int, dict] = None,
        variable_strategy: str = 'GPU',
        predictions_have_variable_shape: bool = False,
        continue_training: Union[bool, None] = False,
        profile_hook_config: dict = None) -> TrainerRunConfig:
    """
    Validate and update trainer run configuration

    Parameters
    ----------
    batch_size
        batch size per device
    devices
        list of devices to use for the model; inputs are all always fetched
        and augmented on cpu once for all gpus
    variable_strategy
        CPU to use CPU as the parameter server
    num_epochs : int, only for training, default: 100
        number of epochs to use
    samples_per_epoch
        defines number of samples per epoch with ['train', 'eval'] as keys
    iterations_per_epoch
        defines number of iterations per epoch with ['train', 'eval'] as keys;
        if defined, will override samples_per_epoch
    predictions_have_variable_shape : optional
        if the predictions from different devices have different shapes
        from batch to batch and so should be concatenated
        with padding during evaluation; useful for object detections
    continue_training
        if the training should be continued in same project folder; if no
        project directories exist, it will have no effect; otherwise it will
        create subfolders continue-{i} in configs and summary folders
    random_seed
        graph based random seed for tensorflow graph
    profile_hook_config
        configuration for profiler hook if profiling should be done;
        will save the profiling information in chrome trace format in the
        project directory; you can provide all the kwargs except output_dir.
        to project directory; this will slow down significantly so use it only
        when you want to profile your models
        for more info see :obj:`tf.train.ProfilerHook`.

    Returns
    -------
    trainer_run_config
        trainer run configuration

    Raises
    ------
    AssertionError
        if same mode is provided in iterations_per_epoch and in
        samples_per_epoch or if provided modes are not 'train' and 'eval'
    AssertionError
        if both iterations_per_epoch and samples_per_epoch are not provided
    AssertionError
        if variable_strategy is not in ['CPU', 'GPU']

    """

    # pylint: disable=too-many-arguments, too-many-locals
    # all the variables are needed inside of the config

    def _select_devices(current_devices):
        if current_devices:
            return current_devices
        available_gpus = tf_utils.get_available_gpus()
        available_devices = available_gpus or ['/cpu:0']
        return available_devices

    def _recalculate_batch_size(batch_size_per_device, num_devices):
        if isinstance(batch_size_per_device, int):
            batch_size_per_device = {mode: batch_size_per_device
                                     for mode in modes}
        batch_size_all_devs = {k: v * num_devices
                               for k, v in batch_size_per_device.items()}
        return batch_size_all_devs

    def _validate_provided_iterations_per_epoch(iterations_per_epoch_,
                                                samples_per_epoch_):
        assert iterations_per_epoch_ or samples_per_epoch_, (
            "Provide samples_per_epoch or iterations_per_epoch or both!")
        defined_keys_iter = (list(samples_per_epoch_.keys()) +
                             list(iterations_per_epoch_.keys()))

        assert_msg = (
            "Provide 'train' and 'eval' keys inside of samples_per_epoch  "
            "or iterations_per_epoch!"
            "(samples_per_epoch: {}, iterations_per_epoch: {})"
        ).format(samples_per_epoch_, iterations_per_epoch_)
        assert (set(modes) == set(defined_keys_iter) and
                len(defined_keys_iter) == 2), assert_msg
        if (not iterations_per_epoch_ or
                tf.estimator.ModeKeys.TRAIN not in iterations_per_epoch_):
            logger.warning("It is better to set iterations_per_epoch for train "
                           "mode, as then number of iterations is independent "
                           "of number of training devices!")

    def _recalculate_iterations_per_epoch(
            iterations_per_epoch_, samples_per_epoch_, batch_size_all_devices):
        for each_mode in samples_per_epoch_:
            batch_size_ = batch_size_all_devices[each_mode]
            iterations_per_epoch_[each_mode] = (samples_per_epoch_[each_mode] //
                                                batch_size_)
        return iterations_per_epoch_

    logger = logging.getLogger(__name__)
    assert variable_strategy in ['GPU', 'CPU'], (
        "variable_strategy should be in ['GPU', 'CPU'], provided {}!".format(
            variable_strategy))
    iterations_per_epoch = iterations_per_epoch or {}
    samples_per_epoch = samples_per_epoch or {}
    modes = [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]
    devices = _select_devices(devices)
    batch_size = _recalculate_batch_size(batch_size, len(devices))
    _validate_provided_iterations_per_epoch(
        iterations_per_epoch, samples_per_epoch)
    iterations_per_epoch = _recalculate_iterations_per_epoch(
        iterations_per_epoch, samples_per_epoch, batch_size)

    return TrainerRunConfig(batch_size, devices, random_seed, num_epochs,
                            iterations_per_epoch, variable_strategy,
                            predictions_have_variable_shape, continue_training,
                            profile_hook_config)


def create_and_validate_trainer_save_config(
        inference_inputs_have_variable_shape: bool = False,
        exports_to_keep: Optional[int] = None,
        save_summary_steps: Union[int, dict] = 100,
        max_outputs_tb: int = 1) -> TrainerSaveConfig:
    """
    Create and validate configuration for exports and summaries during
    training

    Parameters
    ----------
    save_summary_steps
        how often to save summaries; can be also a dict with train and eval keys
    max_outputs_tb
        number of maximum outputs in tensorboard e.g. for images
    inference_inputs_have_variable_shape
        controls if the inputs inside of inference graph should have
        variable shapes; defaults to True
    exports_to_keep
        number of exports to keep

    Returns
    -------
    trainer save configuration
        saving configuration
    """

    def _validate_save_summary_steps(save_summary_steps_):
        modes = [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]
        if isinstance(save_summary_steps_, int):
            save_summary_steps_ = {m: save_summary_steps_ for m in modes}
        else:
            assert (isinstance(save_summary_steps_, dict)
                    and 'train' in save_summary_steps_
                    and 'eval' in save_summary_steps_
                    ), ("If you provide save_summary_steps as a dict, provide "
                        "train and eval keys! ({})").format(save_summary_steps_)
        return save_summary_steps_

    logger = logging.getLogger(__name__)
    if not inference_inputs_have_variable_shape:
        logger.warning("Model will have fixed input dimensions for inference! "
                       "If you want to change it and if your model allows it, "
                       "set inference_inputs_have_variable_shape = True")

    save_summary_steps = _validate_save_summary_steps(save_summary_steps)
    return TrainerSaveConfig(inference_inputs_have_variable_shape,
                             exports_to_keep, save_summary_steps,
                             max_outputs_tb)


def create_and_validate_inference_load_config(
        project_dir: str,
        saved_model_path: Union[str, None] = None,
        checkpoint_path: Union[str, None] = None) -> InferenceLoadConfig:
    """
    Validate paths provided inside of load_config and convert them
    to relative to the project.

    If no path provided, it will search for saved_models directory inside of
    project_dir and use latest export

    saved_model_path must be relative to project_dir/saved_models folder and
    checkpoint_path must be relative to project_dir/checkpoints folder

    Parameters
    ----------
    project_dir
        project directory
    saved_model_path
        path to saved model relative to project_dir/saved_models
    checkpoint_path
        path to .chpt file if saved_model_path was not provided relative to
        project_dir/checkpoints; meta graph will be loadded from the
        project_dir/checkpoints/graph_inference.meta

    Returns
    -------
    load_config
        validated instance of load config

    Raises
    ------
    ValueError
        if both saved model and checkpoint are provided
    FileNotFoundError
        if provided paths not found or no paths provided and there is no
        saved_models directory
    """

    if saved_model_path and checkpoint_path:
        msg = ("Either none or saved_model or meta_graph with checkpoint "
               "should be defined!")
        raise ValueError(msg)

    if all([saved_model_path is None, checkpoint_path is None]):
        recent_saved_model = _get_recent_saved_model(project_dir)
        return InferenceLoadConfig(saved_model=recent_saved_model,
                                   meta_graph=None, checkpoint=None)

    saved_model_full_path = None
    checkpoint_full_path = None
    meta_graph_full_path = None
    if saved_model_path:
        saved_model_full_path = _validate_saved_model(
            project_dir, saved_model_path)
    if checkpoint_path:
        checkpoint_full_path, meta_graph_full_path = _validate_checkpoint(
            project_dir, checkpoint_path)
    load_config = {'saved_model': saved_model_full_path,
                   'meta_graph': meta_graph_full_path,
                   'checkpoint': checkpoint_full_path}
    return InferenceLoadConfig(**load_config)


def create_and_validate_tensorrt_config(
        use_tensorrt: bool = False,
        batch_size: Optional[int] = None,
        max_batch_size: int = 1,
        max_workspace_size_bytes: int = 1 << 30,
        precision_mode: str = "FP32", minimum_segment_size: int = 3,
        is_dynamic_op: bool = False, maximum_cached_engines=1,
        cached_engine_batch_sizes: Optional[list] = None,

):
    """
    Create and validate tensorrt config

    If tensorrt cannot be imported, will warn and set use_tensorrt = False

    Parameters
    ----------
    use_tensorrt
        if tensorrt should be used
    batch_size
        batch size of inference model. If provided and max_batch_size is not
        provided, will use it as max_batch_size. Otherwise,
        max(max_batch_size, batch_size) will be used

    Returns
    -------
    tensorrt_config
        tensorrt config

    Raises
    ------
    ValueError
        if precision_mode not in ["FP16", "FP32"]

    References
    ----------
    parameters
        https://github.com/tensorflow/tensorflow/blob/\
            16d7642c6481b703ab433596af27c2ef5141eb51/tensorflow/python/\
            compiler/tensorrt/trt_convert.py
    """
    # pylint: disable=too-many-arguments
    # not possible to have less arguments since all of them are passed to
    # tensorrt itself
    logger = logging.getLogger(__name__)
    try:
        # pylint: disable=unused-import
        # tensorrt import is needed to understand if tensorrt is installed
        from tensorflow.contrib import tensorrt as trt
    # pylint: disable=invalid-name
    # is common practice to call exceptions as e
    except (ImportError, tf.errors.NotFoundError) as e:
        if use_tensorrt:
            logger.warning(
                "tensorrt will be disabled, since not found. Error: %s", e)
            use_tensorrt = False
    # pylint: enable=invalid-name
    if not use_tensorrt:
        return TensorrtConfig(
            False, *[None] * (len(TensorrtConfig._fields) - 1))

    if precision_mode not in ["FP32", "FP16"]:
        msg = ("Currently only FP32 and FP16 precision modes are supported! "
               "(provided: {})".format(precision_mode))
        raise ValueError(msg)

    batch_size = batch_size or 1
    max_batch_size = max(batch_size, max_batch_size or 1)
    tensorrt_config = TensorrtConfig(
        use_tensorrt=use_tensorrt,
        max_batch_size=max_batch_size,
        max_workspace_size_bytes=max_workspace_size_bytes,
        precision_mode=precision_mode,
        minimum_segment_size=minimum_segment_size,
        is_dynamic_op=is_dynamic_op,
        maximum_cached_engines=maximum_cached_engines,
        cached_engine_batch_sizes=cached_engine_batch_sizes,
    )
    return tensorrt_config


def _validate_saved_model(project_dir: str, saved_model_path: str) -> str:
    saved_mode_full_path = os.path.join(
        project_dir, ProjectDirs.TRAINER.saved_models, saved_model_path)
    if not os.path.isdir(saved_mode_full_path):
        raise FileNotFoundError(
            'Provided saved_model directory does not exist ({})'.format(
                saved_model_path))
    return saved_mode_full_path


def _validate_checkpoint(project_dir: str, checkpoint_path: str) -> (str, str):
    checkpoint_path = os.path.join(
        project_dir, ProjectDirs.TRAINER.checkpoints, checkpoint_path)
    meta_graph_path = os.path.join(
        project_dir, ProjectDirs.TRAINER.checkpoints,
        INFERENCE_GRAPH_FILE_NAME)
    paths_to_validate = (checkpoint_path + '.index', meta_graph_path)
    for each_path in paths_to_validate:
        if not os.path.isfile(each_path):
            raise FileNotFoundError(
                "File {} does not exist!!!".format(each_path))
    return checkpoint_path, meta_graph_path


def _get_recent_saved_model(project_dir: str) -> str:
    saved_models_dir = os.path.join(project_dir,
                                    ProjectDirs.TRAINER.saved_models)
    if not os.path.isdir(saved_models_dir):
        raise FileNotFoundError(
            "No saved_models directory inside of project found! "
            "Provide other options to load in load_config!")
    saved_model_dirs = [
        each_dir for each_dir in os.listdir(saved_models_dir)
        if os.path.isdir(os.path.join(saved_models_dir, each_dir))]
    if not saved_model_dirs:
        raise FileNotFoundError(
            "No saved models in saved_models "
            "directory {} found!".format(saved_models_dir))
    recent_saved_model = sorted(saved_model_dirs)[-1]
    recent_saved_model = os.path.join(saved_models_dir,
                                      recent_saved_model)
    return recent_saved_model
