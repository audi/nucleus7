# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Methods to encapsulate deprecations from older versions of nucleus7, primary in
configs
"""

from functools import wraps
import os
import warnings


def warning_deprecated_config_param(deprecated_name: str,
                                    real_name: str = None,
                                    additional_instructions: str = None,
                                    value=None):
    """
    Raise a DeprecationWarning with specified message structure used to point
    the deprecated parameter names

    Parameters
    ----------
    deprecated_name
        deprecated parameter name
    real_name
        real parameter name which will be used instead of deprecated_name
    additional_instructions
        additional information inside of warning message
    value
        value of the parameter to print inside of warning message

    Warnings
    --------
    DeprecationWarning
        deprecation warning

    """

    def _format_names(name):
        if "::" in name:
            num_levels = len(name.split("::")) - 1
            name = name.replace("::", " = {")
            name = "{" + name + ' = VALUE'
            name += "}" * (num_levels + 1)
        return name

    message = ("`{}` configuration parameter IS DEPRECATED. "
               "It will be removed in future versions. "
               "Instructions for updating: "
               ).format(_format_names(deprecated_name))
    if real_name:
        message += "Use `{}` instead in your configs.".format(
            _format_names(real_name))
    if additional_instructions:
        message += additional_instructions
    if value is not None:
        message += " (Provided value is: {})".format(value)
    with warnings.catch_warnings():
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn(message, DeprecationWarning, stacklevel=2)


def warning_deprecated_config_file(deprecated_fname: str,
                                   real_fname: str = None,
                                   additional_instructions: str = None,
                                   value=None):
    """
    Raise a DeprecationWarning with specified message structure used to point
    the deprecated file names

    Parameters
    ----------
    deprecated_fname
        deprecated file name for the config
    real_fname
        real file name for the config
    additional_instructions
        additional information inside of warning message
    value
        value of the parameter to print inside of warning message

    Warnings
    --------
    DeprecationWarning
        deprecation warning

    """
    deprecated_fname = os.path.basename(deprecated_fname)
    message = ("`{}` configuration file IS DEPRECATED. "
               "It will be removed in future versions "
               "and will NOT BE USED or WILL CAUSE ERRORS."
               "Instructions for updating: ").format(deprecated_fname)
    if real_fname:
        message += "Rename this config file to `{}` instead.".format(real_fname)
    if additional_instructions:
        message += additional_instructions
    if value is not None:
        message += " (Full file name: {})".format(value)
    with warnings.catch_warnings():
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn(message, DeprecationWarning, stacklevel=2)


def warning_deprecated_method(deprecated_name: str,
                              real_name: str = None,
                              additional_instructions: str = None):
    """
    Raise a DeprecationWarning with specified message structure used to point
    the deprecated parameter method

    Parameters
    ----------
    deprecated_name
        deprecated method name
    real_name
        new method name
    additional_instructions
        additional information inside of warning message

    Warnings
    --------
    DeprecationWarning
        deprecation warning

    """

    message = ("`{}` method IS DEPRECATED. "
               "It will be removed in future versions. "
               "Instructions for updating: ").format(deprecated_name)
    if real_name:
        message += "Use `{}` instead in your configs.".format(real_name)
    if additional_instructions:
        message += additional_instructions
    with warnings.catch_warnings():
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn(message, DeprecationWarning, stacklevel=2)


def warn_deprecated_method(real_name: str = None,
                           additional_instructions: str = None):
    """
    Warn that the method is deprecated and then execute it

    Parameters
    ----------
    real_name
        new name of the method
    additional_instructions
        additional instruction to use

    Returns
    -------
    wrapped
        decorated method
    """

    def decorator(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            deprecated_name = '.'.join((function.__module__, function.__name__))
            warning_deprecated_method(
                deprecated_name=deprecated_name, real_name=real_name,
                additional_instructions=additional_instructions)
            return function(*args, **kwargs)

        return wrapped

    return decorator


def replace_deprecated_parameter_in_config(deprecated_name: str,
                                           real_name: str,
                                           config: dict,
                                           required=True) -> dict:
    """
    Replace deprecated parameter to real parameter name and raise a warning
    if the deprecated parameter name exists in the config

    Parameters
    ----------
    deprecated_name
        deprecated parameter name
    real_name
        real parameter name which will be used instead of deprecated_name
    config
        config
    required
        if set, then raise a AssertionError if config doesn't have real_name
        parameter or deprecated_name parameter inside

    Returns
    -------
    config_with_replaced_deprecated
        configuration with replaced deprecated name parameter

    Warnings
    --------
    DeprecationWarning
        if parameter with deprecated_name exists

    Raises
    ------
    AssertionError
        if required and value of parameter with deprecated name and real name
        is None
    AssertionError
        if parameter with both deprecated_name and real_name exists inside of
        config

    """
    deprecated_value = pop_value_from_config(config, deprecated_name, None)
    real_value = get_value_from_config(config, real_name, None)

    if required:
        assert deprecated_value is not None or real_value is not None, (
            "Provide {}!".format(real_name))

    if deprecated_value is not None:
        assert real_value is None, (
            ("either {dn} or {rn} (DEPRECATED) should be "
             "provided, not both! ({dn}: {dv}, {rn}: {rv})"
             ).format(dn=deprecated_name, rn=real_name, dv=deprecated_value,
                      rv=real_value))
        warning_deprecated_config_param(deprecated_name, real_name,
                                        value=deprecated_value)
        value = deprecated_value

        add_value_to_config(config, real_name, value)
    return config


def replace_deprecated_parameter(deprecated_name, real_name, required=True):
    """
    Decorator to use to replace the deprecated parameter inside of method
    kwargs and warn about it

    Parameters
    ----------
    deprecated_name
        deprecated parameter name
    real_name
        real parameter name which will be used instead of deprecated_name
    required
        if set, then raise a AssertionError if config doesn't have real_name
        parameter or deprecated_name parameter inside

    Returns
    -------
    decorated
        decorated method
    """

    def decorator(function):
        @wraps(function)
        def wrapped(*args, **config):
            config = replace_deprecated_parameter_in_config(
                deprecated_name, real_name, config, required)
            return function(*args, **config)

        return wrapped

    return decorator


def pop_value_from_config(config: dict, key: str, default=None,
                          level_separator='::'):
    """
    Example
    -------

    >>> config = {'a': {'b': {'c': 10}}, 'p1': 100, 'p2': 200}
    >>> pop_value_from_config(config, 'a::b::c')
    10

    >>> config = {'p1': 100, 'p2': 200}
    >>> pop_value_from_config(config, 'p1')
    100

    """

    def _remove_empty_leave(_config):
        keys = list(_config)
        for k in keys:
            _subconfig = _config[k]
            if (_subconfig is None
                    or (not _subconfig
                        and isinstance(_subconfig, (list, dict, tuple)))):
                del _config[k]
            if isinstance(_subconfig, dict):
                _remove_empty_leave(_subconfig)
                _remove_empty_leave(_config)

    if level_separator not in key:
        return config.pop(key, default)
    keys_nested = key.split(level_separator)
    num_nested_levels = len(keys_nested)

    config_at_level = config
    value = default
    for level, key_at_level in enumerate(keys_nested):
        if not isinstance(config_at_level, dict):
            return default

        if level == num_nested_levels - 1:
            value = config_at_level.pop(key_at_level, default)
            break

        config_at_level = config_at_level.get(key_at_level)
        if config_at_level is None:
            break
    _remove_empty_leave(config)
    return value


def get_value_from_config(config: dict, key: str, default=None,
                          level_separator='::'):
    """
    Example
    -------

    >>> config = {'a': {'b': {'c': 10}}, 'p1': 100, 'p2': 200}
    >>> get_value_from_config(config, 'a::b::c')
    10

    >>> config = {'p1': 100, 'p2': 200}
    >>> get_value_from_config(config, 'p1')
    100

    >>> config = {'a': 100, 'c': 200}
    >>> get_value_from_config(config, 'a::b', 'default_value')
    'default_value'

    """

    if level_separator not in key:
        return config.get(key, default)
    keys_nested = key.split(level_separator)
    num_nested_levels = len(keys_nested)

    config_at_level = config
    value = default
    for level, key_at_level in enumerate(keys_nested):
        if not isinstance(config_at_level, dict):
            return default

        if level == num_nested_levels - 1:
            value = config_at_level.get(key_at_level, default)
            break

        config_at_level = config_at_level.get(key_at_level)
        if config_at_level is None:
            break
    return value


def add_value_to_config(config: dict, key: str, value,
                        level_separator='::', key_if_not_dict='name'):
    """
    Example
    -------

    >>> config = {'p1': 100, 'p2': 200}
    >>> add_value_to_config(config, 'a::b::c', 100)
    {'a': {'b': {'c': 100}}, 'p1': 100, 'p2': 200}

    >>> config = {'p1': 100, 'p2': 200}
    >>> add_value_to_config(config, 'p1::b::c', 100)
    {'p1': {'b': {'c': 100}, 'name': 100}, 'p2': 200}

    """
    if level_separator not in key:
        config[key] = value
        return config

    keys_nested = key.split(level_separator)
    num_nested_levels = len(keys_nested)
    config_at_level = config
    for level, key_at_level in enumerate(keys_nested):
        if level < num_nested_levels - 1:
            existing_value = config_at_level.get(key_at_level)
            if existing_value is None or isinstance(existing_value, dict):
                config_at_level = config_at_level.setdefault(key_at_level, {})
            else:
                existing_value = {key_if_not_dict: existing_value}
                config_at_level[key_at_level] = existing_value
                config_at_level = config_at_level[key_at_level]
        else:
            config_at_level[key_at_level] = value
    return config
