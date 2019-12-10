# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Builders for DataFeeder
"""

from nucleus7.builders import data_builder_lib
from nucleus7.data.data_feeder import DataFeeder
from nucleus7.utils import deprecated


def build(datafeeder_config: dict) -> DataFeeder:
    """
    Build data feeder based on its config

    Parameters
    ----------
    datafeeder_config
        configuration of data feeder

    Returns
    -------
    data_feeder
        data_feeder
    """
    if 'name' in datafeeder_config:
        msg = "Do not use name inside of data_feeder!"
        deprecated.warning_deprecated_config_param(
            'name', additional_instructions=msg
        )
        raise ValueError(msg)

    if ("data_pipe" in datafeeder_config
            and "class_name" not in datafeeder_config):
        build_fn = lambda x: DataFeeder.from_data_pipe(**x).build()
    else:
        build_fn = None
    data_feeder = data_builder_lib.build_data_object_from_config(
        config=datafeeder_config, base_cls=DataFeeder,
        built_fn=build_fn)
    return data_feeder
