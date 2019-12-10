# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Builders for DataFilter
"""

from functools import partial
from typing import List
from typing import Optional
from typing import Union

from nucleus7.builders import builder_lib
from nucleus7.data.data_filter import DataFilter
from nucleus7.data.data_filter import DataFilterMixin


def build(data_filter_config: Union[dict, List[dict]]
          ) -> Union[List[DataFilter], DataFilter]:
    """
    Build the FileList based on its config or if the chain of file lists
    provided, it will sum all the file lists in the order

    Parameters
    ----------
    data_filter_config

    Returns
    -------
    file_list
        built file list

    """
    single_data_filter_build_fn = partial(
        builder_lib.build_registered_object_from_config, base_cls=DataFilter)
    if isinstance(data_filter_config, list):
        return builder_lib.build_chain_of_objects(data_filter_config,
                                                  single_data_filter_build_fn)
    return single_data_filter_build_fn(data_filter_config)


def data_filter_build_callback(
        built_object: DataFilterMixin,
        data_filter: Optional[Union[List[DataFilter], DataFilter]] = None,
        **kwargs):
    """
    Build callback to use for objects with data filter

    Parameters
    ----------
    built_object
        built object to add data filter to
    data_filter
        data filter object

    Returns
    -------
    built_object
        built object with added data_filter to it
    """
    # pylint: disable=unused-argument
    # kwargs is needed to handle different not used inputs
    if not data_filter:
        return built_object
    if isinstance(data_filter, list):
        built_object.add_data_filters(data_filter)
    else:
        built_object.add_data_filter(data_filter)
    return built_object
