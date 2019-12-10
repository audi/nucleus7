# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
methods for global project configuration

to add a global configuration use:
:func:`nc7.add_global_project_config` or
:func:`add_global_project_config_from_file`

To query it use :func:`query_nucleotide_global_config`,
:func:`query_global_parameter` and to update parameters of nucleotide with
defaults use :func:`add_defaults_to_nucleotide`
"""

import copy
import logging
import os
from typing import Optional
from typing import Union

import dpath

from nucleus7.utils import io_utils
from nucleus7.utils import object_utils


class _ProjectGlobalConfig:
    """
    Class to hold global project configuration

    It allows to specify some global configuration with keys as arguments and
    pass it to all of nucleotides if this parameter exist in its constructor
    and it was not explicitly specified in nucleotide config.

    It is also possible to specify the parameters type-wise and class-wise and
    also as complete global, e.g.
    ```
    {
        "param1": "value1",
            "ModelPlugin": {
                "activation": "elu",
                "param2": "value2"
            },
            "Plugin1": {
                "param1": "value12",
                "activation": "relu"
            }
    }
    ```
    In that case, ModelPlugin will be initialized with activation = elu and
    param1 = value1 and Plugin1 will be initialized with activation = relu and
    param1 = value12 and param2 = value2. E.g. it will create inheritance map
    for each nucleotide and will resolve it in hierarchy order, e.g. lower
    hierarchy (Nucleotide) will be overridden by higher
    (Nucleotide -> ModelPlugin -> Plugin1 -> Plugin1Child -> ...)
    Parameters can be set to all of nucleus7 interfaces.

    Attributes
    ----------
    config
        global configuration
    """
    config = {}

    @staticmethod
    def get(item: Union[list, dict]) -> Optional[Union[list, dict]]:
        """
        Get item from config

        Can be also a multi level item with separator '/'

        Parameters
        ----------
        item
            item to get

        Returns
        -------
        values
            values of item from config or a list of them if list of items was
            specified
        """
        if not item:
            return None

        def _get_single(item_):
            values_ = sorted(list(
                dpath.search(_ProjectGlobalConfig.config, item_,
                             yielded=True)))
            if not values_:
                return None
            if len(values_) > 1:
                values_ = [v[1] for v in values_]
            else:
                values_ = values_[0][1]
            return values_

        if isinstance(item, list):
            values_all = []
            for item_ in item:
                values = _get_single(item_)
                if values:
                    values_all.append(values)
            return values_all if values_all else None
        values = _get_single(item)
        return values

    @staticmethod
    def query_nucleotide_global_config(nucleotide_cls) -> Optional[dict]:
        """
        Query parameters of nucleotide constructor from global config and
        resolve them in hierarchy order

        Parameters
        ----------
        nucleotide_cls
            class of nucleotide

        Returns
        -------
        nucleotide_params_in_project
            dict with keys mapping constructor parameters to parameters found
            in global config; if no parameters were found, then returns None;
            returns only found parameters and if they were specified
        """
        if not _ProjectGlobalConfig.config:
            return None
        init_method = nucleotide_cls.__init__
        nucleotide_args = {
            p for s in object_utils.get_method_signatures(
                nucleotide_cls, init_method)[:-1]
            for p in s.parameters.keys()
        }

        if 'self' in nucleotide_args:
            nucleotide_args.remove('self')
        if 'name' in nucleotide_args:
            nucleotide_args.remove('name')
        nucleotide_params_in_project = {}
        parent_methods_all = object_utils.get_parent_method(nucleotide_cls)
        parent_classes = ([nucleotide_cls]
                          + list(parent_methods_all)[:-1])
        parent_class_names = [cl.__name__ for cl in parent_classes]
        parent_class_names_rev = parent_class_names[::-1]
        for param_name in nucleotide_args:
            search_scopes = [''] + parent_class_names_rev
            value = None
            for scope in search_scopes:
                query = '/'.join([scope, param_name])
                value_ = _ProjectGlobalConfig.get(query)
                if value_ is not None:
                    value = value_
            if value is not None:
                nucleotide_params_in_project[param_name] = value
        return nucleotide_params_in_project or None

    @staticmethod
    def add_defaults_to_nucleotide(nucleotide_cls, nucleotide_params: dict
                                   ) -> dict:
        """
        Add the default nucleotide parameters from global config


        Parameters
        ----------
        nucleotide_cls
            class of nucleotide
        nucleotide_params
            parameters of that nucleotide

        Returns
        -------
        nucleotide_params_with_defaults
            parameters of nucleotide with defaults from global config
        """
        nucleotide_params = nucleotide_params or {}
        logger = logging.getLogger(__name__)

        nucleotide_name = nucleotide_params.get('name', nucleotide_cls.__name__)
        nucleotide_params_project = (
            _ProjectGlobalConfig.query_nucleotide_global_config(nucleotide_cls))
        if not nucleotide_params_project:
            return nucleotide_params

        nucleotide_params = copy.deepcopy(nucleotide_params)
        nucleotide_params_with_defaults = copy.deepcopy(
            nucleotide_params_project)

        for k in nucleotide_params_with_defaults:
            if k in nucleotide_params:
                logger.info('Default value of parameter %s in nucleotide %s '
                            'is overridden', k, nucleotide_name)
                nucleotide_params_with_defaults[k] = nucleotide_params[k]
            else:
                logger.info('Default value of parameter %s in nucleotide %s '
                            'will be used (value: %s)',
                            k, nucleotide_name,
                            nucleotide_params_with_defaults[k])
        for k, each_param in nucleotide_params.items():
            if k not in nucleotide_params_with_defaults:
                nucleotide_params_with_defaults[k] = each_param

        return nucleotide_params_with_defaults

    @staticmethod
    def clear():
        """
        Clear project global config
        """
        _ProjectGlobalConfig.config = {}


def add_global_project_config_from_project_dir(config_dir: str):
    """
    Add project configuration from file 'project_global_config.json' file if
    it exists inside of configs directory directory

    Parameters
    ----------
    config_dir
        directory to search for the config
    """
    project_config_fname = os.path.join(
        config_dir, 'project_global_config.json')
    add_global_project_config_from_file(project_config_fname)


def add_global_project_config_from_file(fname: str):
    """
    Add project configuration from json file

    Parameters
    ----------
    fname
        file name of project configuration .json file

    See Also
    --------
    :func:`add_global_project_config`

    """
    logger = logging.getLogger(__name__)
    if not os.path.isfile(fname):
        logger.info("No project configuration file was found")
        return
    config = io_utils.maybe_load_json(fname)
    logger.info("Add project global config from %s", fname)
    add_global_project_config(config)


def add_global_project_config(config: dict):
    """
    Add global project configuration

    It allows to specify some global configuration with keys as arguments and
    pass it to all of nucleotides if this parameter exist in its constructor
    and it was not explicitly specified in nucleotide config.

    It is also possible to specify the parameters type-wise and class-wise and
    also as complete global, e.g.

    `{
    'param1': 'value1',
    'ModelPlugin': {'activation': 'elu', 'param2': 'value2'}
    'Plugin1': {'param1': 'value12', 'activation': 'relu'}
    }`

    In that case, ModelPlugin will be initialized with activation = elu and
    param1 = value1 and Plugin1 will be initialized with activation = relu and
    param1 = value12 and param2 = value2. E.g. it will create inheritance map
    for each nucleotide and will resolve it in hierarchy order, e.g. lower
    hierarchy (Nucleotide) will be overriden by higher
    (Nucleotide -> ModelPlugin -> Plugin1 -> Plugin1Child -> ...)

    Parameters can be set to all of nucleus7 interfaces.

    Parameters
    ----------
    config
        global configuration

    Raises
    ------
    AssertionError
        when config is not a dict or it has a name or a class_name as keys
    """
    assert isinstance(config, dict) or config is None, (
        "Project global config can be only a dictionary! "
        "({})".format(config))
    if isinstance(config, dict):
        assert 'name' not in config and 'class_name' not in config, (
            "name or class name are not allowed as keys inside of "
            "project global config!!! ({})".format(config))
    config = config or {}
    _ProjectGlobalConfig.config.update(config)


def query_nucleotide_global_config(nucleotide_cls: type) -> dict:
    """
    Query parameters of nucleotide constructor from global project config and
    resolve them in hierarchy order

    Parameters
    ----------
    nucleotide_cls
        class of nucleotide

    Returns
    -------
    nucleotide_params_in_project
        dict with keys mapping constructor parameters to parameters found
        in global config; if no parameters were found, then returns None;
        returns only found parameters and if they were specified
    """
    return _ProjectGlobalConfig.query_nucleotide_global_config(nucleotide_cls)


def add_defaults_to_nucleotide(nucleotide_cls, nucleotide_params: dict
                               ) -> dict:
    """
    Add the default nucleotide parameters from global project config and
    returns its updated configuration


    Parameters
    ----------
    nucleotide_cls
        class of nucleotide
    nucleotide_params
        parameters of that nucleotide

    Returns
    -------
    nucleotide_params_with_defaults
        parameters of nucleotide with defaults from global config
    """
    return _ProjectGlobalConfig.add_defaults_to_nucleotide(
        nucleotide_cls, nucleotide_params)


def query_global_parameter(item: Union[str, list]) -> Union[list, dict]:
    """
    Get item from  global project config

    Can be also a multi level item with separator '/'

    Parameters
    ----------
    item
        item to get

    Returns
    -------
    values
        values of item from config or a list of them if list of items was
        specified
    """
    return _ProjectGlobalConfig.get(item)


def clear_project_global_config():
    """Clear global project configuration"""
    _ProjectGlobalConfig.clear()
