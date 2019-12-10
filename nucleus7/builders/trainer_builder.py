# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Builder for Trainer
"""

import copy
from typing import Dict
from typing import Union

from nucleus7.builders import builder_lib
from nucleus7.builders import optimization_builders
from nucleus7.coordinator import configs as coord_configs
from nucleus7.coordinator.callbacks_handler import CallbacksHandler
from nucleus7.coordinator.trainer import Trainer
from nucleus7.data import Dataset
from nucleus7.model.model import Model
from nucleus7.utils import deprecated
from nucleus7.utils import run_utils


def build(trainer_config: dict,
          model: Model,
          project_dir: str,
          datasets: Union[Dict[str, Dataset], dict],
          callbacks_handler_train: Union[CallbacksHandler, None] = None,
          callbacks_handler_eval: Union[CallbacksHandler, None] = None,
          continue_training: bool = False) -> Trainer:
    """
    Build trainer based on its config and components

    Parameters
    ----------
    model
        model handler to use
    project_dir
        project directory
    datasets
        datasets with ['train', 'eval'] keys in case of local run and 'train'
        if it is a training worker or 'eval' if it is a evaluator worker
    trainer_config
        trainer configuration
    callbacks_handler_train
        callbacks handler with callbacks used during the training
    callbacks_handler_eval
        callbacks handler with callbacks used during the evaluation
    continue_training
        if the training should be continued in case that the project inside
        of project_dir already exists

    Returns
    -------
    trainer
        trainer
    """
    # pylint: disable=too-many-arguments
    # trainer takes so many arguments, more split will be more confusing

    trainer_config = deprecated.replace_deprecated_parameter_in_config(
        'optimization_parameters', 'optimization_config', config=trainer_config,
        required=True)

    trainer_config = copy.deepcopy(trainer_config)
    trainer_config.pop('project_dir', None)
    trainer_config.pop('callbacks', None)
    trainer_config.pop('augmenter', None)
    optimization_config_config = trainer_config.pop("optimization_config")
    optimization_config = optimization_builders.build_optimization_config(
        optimization_config_config, is_global=True)
    run_config = builder_lib.build_config_object(
        coord_configs.create_and_validate_trainer_run_config,
        trainer_config, 'run_config', continue_training=continue_training)
    save_config = builder_lib.build_config_object(
        coord_configs.create_and_validate_trainer_save_config,
        trainer_config, 'save_config')

    if 'mixed_precision_config' in trainer_config:
        deprecated.warning_deprecated_config_param(
            'mixed_precision_config',
            additional_instructions='Use model_config.json to define the '
                                    'mixed_precision_config')
        trainer_config.pop('mixed_precision_config')

    dataset_modes_needed = run_utils.select_dataset_modes_for_run()
    assert set.issubset(set(dataset_modes_needed), set(datasets.keys())), (
        "Provide datasets for {}".format(dataset_modes_needed))

    trainer = builder_lib.build_registered_object(
        default_cls=Trainer,
        base_cls=Trainer,
        run_config=run_config,
        save_config=save_config,
        optimization_config=optimization_config,
        model=model,
        project_dir=project_dir,
        datasets=datasets,
        callbacks_handler_train=callbacks_handler_train,
        callbacks_handler_eval=callbacks_handler_eval,
        **trainer_config
    )
    return trainer
