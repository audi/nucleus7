# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Builder for Inferer
"""
from typing import Union

from nucleus7.builders import builder_lib
from nucleus7.coordinator import configs as coord_configs
from nucleus7.coordinator.callbacks_handler import CallbacksHandler
from nucleus7.coordinator.inferer import Inferer
from nucleus7.data.data_feeder import DataFeeder


def build(inferer_config: dict,
          project_dir: str,
          data_feeder: DataFeeder,
          callbacks_handler: Union[CallbacksHandler, None] = None,
          continue_last: bool = False) -> Inferer:
    """
    Build inferer

    Parameters
    ----------
    inferer_config
        additional configuration of inferer
    project_dir
        project directory
    data_feeder
        configuration for data feeder
    callbacks_handler
        callbacks handler to use
    continue_last
        if last project must be continued

    Returns
    -------
    inferer
        inferer
    """

    inferer_config.pop('callbacks', None)
    inferer_config.pop('project_dir', None)
    use_model = inferer_config.get("use_model", True)

    run_config = builder_lib.build_config_object(
        coord_configs.InferenceRunConfig, inferer_config, 'run_config',
        continue_last=continue_last)
    if use_model:
        load_config = builder_lib.build_config_object(
            coord_configs.create_and_validate_inference_load_config,
            inferer_config, 'load_config', project_dir=project_dir)
    else:
        inferer_config.pop("load_config")
        load_config = None
    tensorrt_config = builder_lib.build_config_object(
        coord_configs.create_and_validate_tensorrt_config, inferer_config,
        'tensorrt_config', search_in_root=False)

    inferer = builder_lib.build_registered_object(
        default_cls=Inferer,
        base_cls=Inferer,
        project_dir=project_dir,
        run_config=run_config,
        load_config=load_config,
        tensorrt_config=tensorrt_config,
        data_feeder=data_feeder,
        callbacks_handler=callbacks_handler,
        **inferer_config)
    return inferer
