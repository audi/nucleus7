# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Methods to log the used nucleotide configurations, e.g. used constructor
arguments
"""

from nucleus7.utils import object_utils

GET_MODE_CONFIG_NAME = "_get_mode"


class _ConfigLogger:
    """
    Config logger
    """
    _logged_configs = {}

    @staticmethod
    def add_constructor_parameters_to_log(
            instance: object, name_scope: str, parameters_args: tuple,
            parameters_kwargs: dict, exclude_args=None):
        """
        Add the constructor parameters of the instance to the log with name
        scope

        Parameters will be logged in the config_logger under name scope and
        under instance name, e.g. {"name_scope": {"instance.name": [...]}}

        It will also collect the default values from class __init__ method and
        add unused to the config


        Parameters
        ----------
        instance
            instance
        name_scope
            log name scope
        parameters_args
            args passed to the constructor
        parameters_kwargs
            kwargs passed to the constructor
        exclude_args
            arguments from constructor which shouldn't be logged, e.g. some
            other instances which were used in the construction, but were
            logged already
        """
        cls = instance.__class__
        instance_name = getattr(instance, 'name', cls.__name__)
        args_with_defaults = _get_args_with_defaults(
            cls, parameters_args, parameters_kwargs, exclude_args)
        args_with_defaults["class_name"] = (
            object_utils.get_class_name_with_module(instance))
        args_with_defaults = _maybe_add_mode_property_to_config(
            instance, args_with_defaults)
        _ConfigLogger._logged_configs.setdefault(name_scope, dict())
        _ConfigLogger._logged_configs[name_scope].setdefault(instance_name, [])
        _ConfigLogger._logged_configs[name_scope][instance_name].append(
            args_with_defaults)

    @staticmethod
    def reset():
        """
        Reset the config logger
        """
        _ConfigLogger._logged_configs = {}

    @staticmethod
    def get() -> dict:
        """
        Get the logged configs

        Returns
        -------
        logged_configs
            logged configs with name_scopes as first order key
        """
        return _ConfigLogger._logged_configs


def add_constructor_parameters_to_log(
        instance: object, name_scope: str, parameters_args: tuple,
        parameters_kwargs: dict, exclude_args=None):
    """
    Add the constructor parameters of the instance to the log with name
    scope

    Parameters will be logged in the config_logger under name scope and
    under instance name, e.g. {"name_scope": {"instance.name": ...}}

    It will also collect the default values from class __init__ method and
    add unused to the config


    Parameters
    ----------
    instance
        instance
    name_scope
        log name scope
    parameters_args
        args passed to the constructor
    parameters_kwargs
        kwargs passed to the constructor
    exclude_args
        arguments from constructor which shouldn't be logged, e.g. some
        other instances which were used in the construction, but were
        logged already
    """
    _ConfigLogger.add_constructor_parameters_to_log(
        instance, name_scope, parameters_args, parameters_kwargs, exclude_args)


def get_logged_configs() -> dict:
    """
    Get the logged configs

    Returns
    -------
    logged_configs
        logged configs with name_scopes as first order key
    """
    return _ConfigLogger.get()


def reset_logged_configs():
    """
    Reset the config logger
    """
    _ConfigLogger.reset()


def _get_args_with_defaults(
        cls, parameters_args: tuple,
        parameters_kwargs: dict, exclude_args=None
) -> dict:
    exclude_args = exclude_args or []
    cls_args, _, cls_defaults = (
        object_utils.get_full_method_signature(cls))
    args_with_values = dict(zip(cls_args, parameters_args))

    args_with_defaults = cls_defaults
    args_with_defaults.update(args_with_values)
    args_with_defaults.update(parameters_kwargs)
    args_with_defaults = {k: v for k, v in args_with_defaults.items()
                          if k not in exclude_args}
    return args_with_defaults


def _maybe_add_mode_property_to_config(
        instance, args_with_defaults: dict
) -> dict:
    if hasattr(instance, "mode"):
        args_with_defaults[GET_MODE_CONFIG_NAME] = lambda: instance.mode
    return args_with_defaults
