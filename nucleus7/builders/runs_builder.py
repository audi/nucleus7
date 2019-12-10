# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Builders for executive scripts, e.g. train, infer etc.

"""
import logging
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import tensorflow as tf
# pylint: disable=no-name-in-module
# run_config is more for tensorflow internal use
from tensorflow.python.estimator import run_config as run_config_lib

from nucleus7.builders import callback_builder
from nucleus7.builders import data_feeder_builder
from nucleus7.builders import dataset_builder
from nucleus7.builders import inferer_builder
from nucleus7.builders import kpi_builder
from nucleus7.builders import model_builder
from nucleus7.builders import trainer_builder
from nucleus7.builders.builder_lib import build_config_object
from nucleus7.coordinator.callbacks_handler import CallbacksHandler
from nucleus7.coordinator.inferer import Inferer
from nucleus7.coordinator.trainer import Trainer
from nucleus7.data.dataset import Dataset
from nucleus7.model.configs import MixedPrecisionConfig
from nucleus7.model.loss import ModelLoss
from nucleus7.model.metric import ModelMetric
from nucleus7.model.plugin import ModelPlugin
from nucleus7.model.postprocessor import ModelPostProcessor
from nucleus7.model.summary import ModelSummary
from nucleus7.utils import run_utils


def build_train(project_dir: str, *,
                trainer_config: dict,
                datasets_config: dict,
                model_config: Optional[dict] = None,
                plugins_config: list,
                losses_config: list,
                postprocessors_config: Optional[List[dict]] = None,
                metrics_config: Optional[List[dict]] = None,
                kpi_config: Optional[List[dict]] = None,
                summaries_config: Optional[List[dict]] = None,
                callbacks_config: Optional[List[dict]] = None,
                callbacks_train_config: Optional[List[dict]] = None,
                callbacks_eval_config: Optional[List[dict]] = None,
                continue_training: bool = False) -> Trainer:
    """
    Build all components for training

    Parameters
    ----------
    project_dir
        project directory
    trainer_config
        trainer configuration
    datasets_config
        datasets configuration for each run mode
    model_config
        model configuration
    plugins_config
        configurations of single plugins inside of model
    losses_config
        configurations of single losses inside of model
    postprocessors_config
        configurations of single postprocessors inside of model
    metrics_config
        configurations of single metrics inside of model
    kpi_config
        configurations of single kpi plugins and accumulators inside of model
    summaries_config
        configurations of single summaries inside of model
    callbacks_config
        configurations of single callbacks inside of model that will be shared
        for training and evaluation (instances of callbacks will be still
        different)
    callbacks_train_config
        configurations of single callbacks inside of model for training
    callbacks_eval_config
        configurations of single callbacks inside of model for evaluation
    continue_training
        if the training should be continued in case that the project inside
        of project_dir already exists

    Returns
    -------
    trainer
        trainer ready to run

    """
    # pylint: disable=too-many-arguments,too-many-locals
    # train takes so many arguments, more split will be more confusing
    plugins = model_builder.build_model_nucleotides(
        plugins_config, base_cls=ModelPlugin)
    losses = model_builder.build_model_nucleotides(
        losses_config, base_cls=ModelLoss)
    summaries = model_builder.build_model_nucleotides(
        summaries_config, base_cls=ModelSummary)
    postprocessors = model_builder.build_model_nucleotides(
        postprocessors_config, base_cls=ModelPostProcessor)
    metrics = model_builder.build_model_nucleotides(
        metrics_config, base_cls=ModelMetric)
    mixed_precision_config = _build_mixed_precision_config(model_config)

    model = model_builder.build(model_config,
                                plugins=plugins,
                                losses=losses,
                                postprocessors=postprocessors,
                                metrics=metrics,
                                summaries=summaries,
                                mixed_precision_config=mixed_precision_config)

    datasets_config = _filter_datasets_for_run(datasets_config)
    datasets = _build_datasets_for_each_mode(datasets_config)

    callbacks_train_config = ((callbacks_config or []) +
                              (callbacks_train_config or []))
    callbacks_eval_config = ((callbacks_config or []) +
                             (callbacks_eval_config or []))

    kpi_plugins_and_accumulators = kpi_builder.build_kpi_plugins(kpi_config)
    kpi_evaluator_callback = kpi_builder.build_kpi_evaluator_as_callback(
        kpi_plugins_and_accumulators)

    callbacks_train = callback_builder.build_callbacks_chain(
        callbacks_train_config)
    callbacks_eval = callback_builder.build_callbacks_chain(
        callbacks_eval_config)
    if kpi_evaluator_callback:
        callbacks_eval.append(kpi_evaluator_callback)
    callbacks_handler_train = CallbacksHandler(
        callbacks=callbacks_train).build()
    callbacks_handler_eval = CallbacksHandler(
        callbacks=callbacks_eval).build()

    trainer = trainer_builder.build(
        trainer_config=trainer_config,
        model=model,
        project_dir=project_dir,
        datasets=datasets,
        callbacks_handler_train=callbacks_handler_train,
        callbacks_handler_eval=callbacks_handler_eval,
        continue_training=continue_training)
    return trainer


def build_infer(project_dir: str, *,
                run_name: Optional[str] = None,
                datafeeder_config: dict,
                callbacks_config: list,
                kpi_config: Optional[List[dict]] = None,
                inferer_config: Optional[dict] = None,
                saved_model: Optional[str] = None,
                checkpoint: Optional[str] = None,
                batch_size: Optional[int],
                number_of_shards: int = 1,
                shard_index: int = 0,
                use_single_process: Optional[bool] = None,
                prefetch_buffer_size: Optional[int] = None,
                use_tensorrt: Optional[bool] = None,
                continue_last: bool = False) -> Inferer:
    """
    Build the inferer for inference run based on its single components and
    configs

    Parameters
    ----------
    project_dir
        project directory
    run_name
        optional run name for inference project
    datafeeder_config
        configuration of data feeder
    callbacks_config
        configurations of single callbacks for inference
    kpi_config
        configurations of single kpi plugins and accumulators inside of model
    inferer_config
        configuration of inferer
    checkpoint
        path to checkpoint file to restore the variables relative to
        project_dir/checkpoint; meta graph must be in the same folder
    saved_model
        path to saved_model folder relative to project_dir/saved_models
    batch_size
        batch size to use for inference
    number_of_shards
        number of shards for datafeeder file list, if file list was provided
    shard_index
        shard index for datafeeder file list, if file list was provided
    prefetch_buffer_size
        number of batches to prefetch; must be >= 1
    use_single_process
        if data prefetching, prediction and callbacks must executed in one
        single process
    use_tensorrt
        if the tensorrt should be enabled
    continue_last
        if last project must be continued

    Returns
    -------
    inferer
        inferer ready to run

    """
    # pylint: disable=too-many-locals
    # all the variables are needed for now
    # TODO(oleksnadr.vorobiov@audi.de): refactor and combine arguments
    inferer_config = inferer_config or {}
    if run_name:
        inferer_config["project_additional_kwargs"] = {"run_name": run_name}
    inferer_config = _update_inferer_run_config(
        inferer_config, batch_size, prefetch_buffer_size, use_single_process)
    inferer_config = _update_inferer_load_config(
        inferer_config, saved_model, checkpoint)
    inferer_config = _update_inferer_tensorrt_config(
        inferer_config, use_tensorrt)
    datafeeder_config = _update_datafeeder_shards(
        datafeeder_config, number_of_shards, shard_index)
    data_feeder = data_feeder_builder.build(datafeeder_config)
    callbacks = callback_builder.build_callbacks_chain(callbacks_config)
    kpi_plugins_and_accumulators = kpi_builder.build_kpi_plugins(kpi_config)
    kpi_evaluator_callback = kpi_builder.build_kpi_evaluator_as_callback(
        kpi_plugins_and_accumulators)
    if kpi_evaluator_callback:
        callbacks.append(kpi_evaluator_callback)
    callbacks_handler = CallbacksHandler(callbacks=callbacks).build()
    inferer = inferer_builder.build(project_dir=project_dir,
                                    inferer_config=inferer_config,
                                    data_feeder=data_feeder,
                                    callbacks_handler=callbacks_handler,
                                    continue_last=continue_last)
    return inferer


def build_kpi_evaluate(project_dir: str, *,
                       run_name: Optional[str] = None,
                       datafeeder_config: dict,
                       callbacks_config: list,
                       kpi_config: Optional[List[dict]] = None,
                       batch_size: Optional[int],
                       number_of_shards: int = 1,
                       shard_index: int = 0,
                       use_single_process: Optional[bool] = None,
                       prefetch_buffer_size: Optional[int] = None,
                       continue_last: bool = False) -> Inferer:
    """
    Build the coordinator for kpi evaluation run based on its single components
    and configs

    Parameters
    ----------
    project_dir
        project directory
    run_name
        optional run name for kpi evaluation project
    datafeeder_config
        configuration of data feeder
    callbacks_config
        configurations of single callbacks for inference
    kpi_config
        configurations of single kpi plugins and accumulators inside of model
    batch_size
        batch size to use for inference
    number_of_shards
        number of shards for datafeeder file list, if file list was provided
    shard_index
        shard index for datafeeder file list, if file list was provided
    prefetch_buffer_size
        number of batches to prefetch; must be >= 1
    use_single_process
        if data prefetching, prediction and callbacks must executed in one
        single process
    continue_last
        if last project must be continued

    Returns
    -------
    inferer
        inferer ready to run
    """
    inferer_config = {
        "use_model": False,
        "project_type": "kpi",
    }
    if run_name:
        inferer_config["project_additional_kwargs"] = {"run_name": run_name}
    inferer = build_infer(
        project_dir=project_dir,
        datafeeder_config=datafeeder_config,
        callbacks_config=callbacks_config,
        inferer_config=inferer_config,
        kpi_config=kpi_config,
        batch_size=batch_size,
        number_of_shards=number_of_shards,
        shard_index=shard_index,
        use_single_process=use_single_process,
        prefetch_buffer_size=prefetch_buffer_size,
        continue_last=continue_last
    )
    return inferer


def build_extract_data(project_dir: str,
                       run_name: str, *,
                       datafeeder_config: dict,
                       callbacks_config: list,
                       kpi_config: Optional[List[dict]] = None,
                       batch_size: Optional[int] = 1,
                       number_of_shards: int = 1,
                       shard_index: int = 0,
                       use_single_process: Optional[bool] = None,
                       prefetch_buffer_size: Optional[int] = None,
                       continue_last: bool = False) -> Inferer:
    """
    Build the coordinator for data extraction run based on its single components
    and configs

    Parameters
    ----------
    project_dir
        project directory
    run_name
        run name for data extraction project
    datafeeder_config
        configuration of data feeder
    callbacks_config
        configurations of single callbacks for inference
    kpi_config
        configurations of single kpi plugins and accumulators inside of model
    batch_size
        batch size to use for inference
    number_of_shards
        number of shards for datafeeder file list, if file list was provided
    shard_index
        shard index for datafeeder file list, if file list was provided
    prefetch_buffer_size
        number of batches to prefetch; must be >= 1
    use_single_process
        if data prefetching, prediction and callbacks must executed in one
        single process
    continue_last
        if last project must be continued

    Returns
    -------
    inferer
        inferer ready to run
    """
    inferer_config = {
        "use_model": False,
        "project_type": "data_extraction",
        "project_additional_kwargs": {
            "run_name": run_name
        }
    }
    inferer = build_infer(
        project_dir=project_dir,
        datafeeder_config=datafeeder_config,
        callbacks_config=callbacks_config,
        inferer_config=inferer_config,
        kpi_config=kpi_config,
        batch_size=batch_size,
        number_of_shards=number_of_shards,
        shard_index=shard_index,
        use_single_process=use_single_process,
        prefetch_buffer_size=prefetch_buffer_size,
        continue_last=continue_last
    )
    return inferer


def _update_inferer_tensorrt_config(
        inferer_config: dict, use_tensorrt: Optional[bool]) -> dict:
    if use_tensorrt is None:
        return inferer_config
    tensorrt_config = inferer_config.setdefault("tensorrt_config", {})
    tensorrt_config["use_tensorrt"] = use_tensorrt
    return inferer_config


def _update_inferer_run_config(
        inferer_config: dict, batch_size: Optional[int] = None,
        prefetch_buffer_size: Optional[int] = None,
        use_single_process: Optional[bool] = None) -> dict:
    run_config = inferer_config.setdefault("run_config", {})
    batch_size_from_config = run_config.get("batch_size")
    if all([batch_size is None, batch_size_from_config is None]):
        msg = ("batch size should be set inside of"
               "inferer_config/run_config/batch_size or as --batch_size on "
               "CLI!")
        raise ValueError(msg)
    if batch_size:
        run_config["batch_size"] = batch_size
    if prefetch_buffer_size is not None:
        run_config["prefetch_buffer_size"] = prefetch_buffer_size
    if use_single_process is not None:
        run_config["use_multiprocessing"] = not use_single_process
    return inferer_config


def _update_inferer_load_config(inferer_config: dict,
                                saved_model: Optional[str] = None,
                                checkpoint: Optional[str] = None) -> dict:
    load_config = inferer_config.setdefault("load_config", {})
    if saved_model:
        load_config["checkpoint_path"] = None
        load_config["saved_model_path"] = saved_model
    if checkpoint:
        load_config["saved_model_path"] = None
        load_config["checkpoint_path"] = checkpoint
    return inferer_config


def _update_datafeeder_shards(datafeeder_config: dict,
                              number_of_shards: int = 1,
                              shard_index: int = 0) -> dict:
    file_list_config = datafeeder_config.get("file_list")
    if not file_list_config:
        return datafeeder_config

    if isinstance(file_list_config, list):
        for each_single_config in file_list_config:
            each_single_config['number_of_shards'] = number_of_shards
            each_single_config['shard_index'] = shard_index
    else:
        file_list_config['number_of_shards'] = number_of_shards
        file_list_config['shard_index'] = shard_index
    return datafeeder_config


def _build_mixed_precision_config(model_config: dict):
    mixed_precision_config = build_config_object(
        MixedPrecisionConfig,
        model_config, 'mixed_precision_config', search_in_root=False)
    return mixed_precision_config


def _filter_datasets_for_run(datasets_config):
    dataset_modes_to_use = run_utils.select_dataset_modes_for_run()
    assert set(dataset_modes_to_use).issubset(set(datasets_config.keys())), (
        "Provide datasets for {}".format(dataset_modes_to_use))
    logger = logging.getLogger(__name__)
    logger.info("Use %s datasets", dataset_modes_to_use)
    datasets_config_filtered = {
        mode: datasets_config[mode] for mode in dataset_modes_to_use
    }
    return datasets_config_filtered


def _build_datasets_for_each_mode(datasets_config: dict) -> Dict[str, Dataset]:
    datasets = {}
    number_of_shards, shard_index = _get_dataset_shards_for_training()
    for each_mode, each_dataset_config in datasets_config.items():
        each_dataset_config = _add_shard_config_to_dataset(
            each_dataset_config, number_of_shards=number_of_shards,
            shard_index=shard_index)
        datasets[each_mode] = dataset_builder.build(each_dataset_config)
    return datasets


def _add_shard_config_to_dataset(
        dataset_config: Union[dict, List[dict]],
        number_of_shards: int, shard_index: int
) -> Union[dict, List[dict]]:
    if isinstance(dataset_config, list):
        dataset_config_with_shards = []
        for each_config in dataset_config:
            dataset_config_with_shards.append(
                _add_shard_config(
                    each_config, number_of_shards=number_of_shards,
                    shard_index=shard_index))
    else:
        dataset_config_with_shards = _add_shard_config(
            dataset_config, number_of_shards=number_of_shards,
            shard_index=shard_index)
    return dataset_config_with_shards


def _add_shard_config(config: Union[dict, list, None],
                      number_of_shards: int, shard_index: int) -> dict:
    config['number_of_shards'] = number_of_shards
    config['shard_index'] = shard_index
    return config


def _get_dataset_shards_for_training() -> Tuple[int, int]:
    """
    Get number of dataset shards and shard index according to task and task
    index

    Returns
    -------
    number_of_shards
        number of shards is number of workers inside of cluster;
        if it is a PS, local run or evaluator, number_of_shards=1
    shard_index
        worker shards; if it is a PS, local run or evaluator, shard_index=0

    """
    run_config = tf.estimator.RunConfig()
    if not run_config.cluster_spec:
        return 1, 0

    cluster_config = run_config.cluster_spec
    if run_config.task_type in [run_config_lib.TaskType.EVALUATOR,
                                run_config_lib.TaskType.PS]:
        return 1, 0
    number_of_shards = (len(cluster_config.job_tasks('chief')))
    try:
        number_of_shards += len(cluster_config.job_tasks('worker'))
    except ValueError:
        pass
    if run_config.task_type == run_config_lib.TaskType.CHIEF:
        shard_index = 0
    else:
        shard_index = run_config.task_id + 1
    return number_of_shards, shard_index
