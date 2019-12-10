# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Builders for FileLists
"""

from functools import partial
from typing import List
from typing import Union

from nucleus7.builders import builder_lib
from nucleus7.builders import data_filter_builder
from nucleus7.data.file_list import FileList
from nucleus7.data.file_list import FileListExtendedMatch


def build(file_list_config: Union[dict, List[dict]]) -> FileList:
    """
    Build the FileList based on its config or if the chain of file lists
    provided, it will sum all the file lists in the order

    Parameters
    ----------
    file_list_config

    Returns
    -------
    file_list
        built file list

    """
    if isinstance(file_list_config, list):
        return _build_file_list_chain(file_list_config)
    return _build_single_file_list(file_list_config)


def _build_file_list_chain(list_of_file_list_configs: List[dict]) -> FileList:
    """
    Build one file list by adding the file lists from each config in the
    list_of_file_list_configs

    Parameters
    ----------
    list_of_file_list_configs
        list where each config is the single config for file_list

    Returns
    -------
    file_lists_sum
        file_list
    """
    file_lists_sum = None
    for each_config in list_of_file_list_configs:
        file_list = _build_single_file_list(each_config)
        if file_lists_sum is None:
            file_lists_sum = file_list
        else:
            file_lists_sum += file_list
    return file_lists_sum


def _build_single_file_list(file_list_config: dict) -> FileList:
    """
    Build file list based on its config

    Parameters
    ----------
    file_list_config
        config of FileList

    Returns
    -------
    file_list
        file_list
    """

    plugged_config_keys_and_builds = [
        builder_lib.PluggedConfigKeyAndBuildFn(
            config_key="data_filter", build_fn=data_filter_builder.build,
            add_to_config=False)]
    file_list = builder_lib.build_with_plugged_objects(
        config=file_list_config,
        build_fn=partial(builder_lib.build_registered_object_from_config,
                         base_cls=FileList, default_cls=FileListExtendedMatch),
        plugged_config_keys_and_builds=plugged_config_keys_and_builds,
        build_callbacks=[data_filter_builder.data_filter_build_callback])
    return file_list
