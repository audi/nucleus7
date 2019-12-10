# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Builder for DataPipe objects
"""
from typing import List
from typing import Optional

from nucleus7.builders.builder_lib import build_registered_object
from nucleus7.data.data_pipe import DataPipe
from nucleus7.data.processor import DataProcessor
from nucleus7.data.reader import DataReader


def build(readers: Optional[List[DataReader]] = None,
          processors: Optional[List[DataProcessor]] = None
          ) -> DataPipe:
    """
    Build data pipe from readers and processors

    Parameters
    ----------
    readers
        readers to use
    processors
        processors to use

    Returns
    -------
    data_pipe
        data pipe with readers and processors
    """
    data_pipe = build_registered_object(
        default_cls=DataPipe,
        base_cls=DataPipe,
        readers=readers,
        processors=processors)
    return data_pipe


def build_data_pipe_from_configs(data_pipe_component_configs: List[dict]
                                 ) -> DataPipe:
    """
    Build data pipe object from its component config

    Parameters
    ----------
    data_pipe_component_configs
        list of DataReaders and DataProcessors

    Returns
    -------
    data_pipe
        data pipe built from its components
    """
    readers = []
    processors = []
    for each_component_config in data_pipe_component_configs:
        component = build_registered_object(**each_component_config)
        if isinstance(component, DataReader):
            readers.append(component)
        elif isinstance(component, DataProcessor):
            processors.append(component)
        else:
            msg = ("Component with config {} is not DataReader and not "
                   "DataProcessor and so cannot be used in DataPipe!"
                   ).format(each_component_config)
            raise ValueError(msg)
    data_pipe = build(readers=readers or None,
                      processors=processors or None)
    return data_pipe
