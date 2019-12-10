# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Builder methods for data module
"""
from functools import partial
from typing import Callable
from typing import Optional

from nucleus7.builders import builder_lib
from nucleus7.builders import data_filter_builder
from nucleus7.builders import file_list_builder, data_pipe_builder


def build_data_object_from_config(config: dict, base_cls: type,
                                  built_fn: Optional[Callable] = None):
    """
    Build object based on its config from data module with optional file list
    and data filter plugged objects

    Parameters
    ----------
    config
        config to use
    base_cls
        base class of built object
    built_fn
        build function to use to build main data object

    Returns
    -------
    data_object
        built object
    """
    plugged_config_keys_and_builds = [
        builder_lib.PluggedConfigKeyAndBuildFn(
            config_key="data_filter", build_fn=data_filter_builder.build,
            add_to_config=False),
        builder_lib.PluggedConfigKeyAndBuildFn(
            config_key="file_list", build_fn=file_list_builder.build,
            add_to_config=True),
        builder_lib.PluggedConfigKeyAndBuildFn(
            config_key="data_pipe",
            build_fn=data_pipe_builder.build_data_pipe_from_configs,
            add_to_config=True)
    ]
    built_fn = (built_fn
                or partial(builder_lib.build_registered_object_from_config,
                           base_cls=base_cls))
    data_object = builder_lib.build_with_plugged_objects(
        config=config,
        build_fn=built_fn,
        plugged_config_keys_and_builds=plugged_config_keys_and_builds,
        build_callbacks=[data_filter_builder.data_filter_build_callback])
    return data_object
