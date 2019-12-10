# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
General utils
"""
# pylint: disable=unused-import
# namedtuple is used inside of examples in the docstring
from collections import namedtuple
# pylint: enable=unused-import
import importlib
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from typing import Union
import zlib

import numpy as np

from nucleus7.utils import nest_utils


def float_or_string(value: float):
    """
    Return float if possible and string otherwise

    Examples
    --------
    >>> float_or_string(10)
    10.0

    >>> float_or_string('a')
    'a'
    """
    try:
        return float(value)
    except ValueError:
        return str(value)


def int_or_string(value: int):
    """
    Return int if possible and string otherwise

    Examples
    --------
    >>> int_or_string(10)
    10

    >>> int_or_string('a')
    'a'
    """
    try:
        return int(value)
    except ValueError:
        return str(value)


def replace_sets_to_lists(nested_dict: dict):
    """
    Recursively replace the sets as values to lists inside of nested_dict

    Examples
    --------
    >>> {'a': {1, 2, 3}, 'b': 1, 'c': {1: {123, 456}}}
    {'a': [1, 2, 3], 'b': 1, 'c': {1: [123, 456]}}

    Parameters
    ----------
    nested_dict
        dict with sets inside as values
    """
    for k, each_value in nested_dict.items():
        if isinstance(each_value, set):
            nested_dict[k] = list(each_value)
        elif isinstance(nested_dict, dict):
            replace_sets_to_lists(each_value)


def get_member_from_package_and_member_names(class_name: str,
                                             package_name: str = None,
                                             default_package_names: list = None,
                                             default_class: type = None):
    """
    Import the class from list of packages (package_name+default_packages)
    by class name; if class_name has '.', it will be splitted to package_name
    and class_name

    Parameters
    ----------
    class_name
        name of the class
    package_name
        name of the package
    default_package_names
        list of packages to look if the class was not found with package_name
    default_class
        if class was not found - this class will be returned

    Returns
    -------
    class
        class from class_name
    """
    if len(class_name.split('.')) > 1:
        names_ = class_name.split('.')
        if package_name is None:
            package_name = '.'.join(names_[:-1])
        else:
            package_name = '.'.join([package_name] + names_[:-1])
        class_name = names_[-1]

    package_names = []
    if package_name is not None:
        if isinstance(package_name, list):
            package_names.extend(package_name)
        else:
            package_names.append(package_name)
    if isinstance(default_package_names, list):
        package_names.extend(default_package_names)
    cls = None
    for each_package_name in package_names:
        try:
            package = importlib.import_module(each_package_name)
            cls = getattr(package, class_name)
            break
        except (ImportError, AttributeError):
            continue
    if cls is None:
        if default_class is not None:
            cls = default_class
        else:
            raise ImportError(
                "Cannot import {} from specified package list {}".format(
                    class_name, package_names))
    return cls


def maybe_update_undefined_parameter_in_config(config: NamedTuple,
                                               param_name: str,
                                               value):
    """
    Update the parameter if it is None with value

    Parameters
    ----------
    config
        config to update
    param_name
        name of the parameter to update
    value
        value to use if the config.param_name is None

    Returns
    -------
    config
        config with updated parameter

    Example
    -------
    >>> Config = namedtuple("T", ['p1', 'p2'])
    >>> config = Config(10, 20)
    >>> maybe_update_undefined_parameter_in_config(config, 'p1', 100)
    Config(10, 20)
    >>> config = Config(None, 10)
    >>> maybe_update_undefined_parameter_in_config(config, 'p1', 100)
    Config(100, 20)
    """
    if getattr(config, param_name) is None:
        config = config._replace(**{param_name: value})
    return config


def split_batch_inputs(inputs: dict,
                       not_batch_keys: Optional[List[str]] = None,
                       ignore_none_values=True,
                       ) -> Tuple[List[dict], dict]:
    """
    Split batch inputs to sample inputs

    Parameters
    ----------
    inputs
        batch inputs to split
    not_batch_keys
        keys to exclude from split
    ignore_none_values
        if the keys with None values should be treated as not batch keys

    Returns
    -------
    batch_inputs_flat_as_list
        list split batch inputs
    not_batch_inputs
        dict with not batch inputs
    """
    not_batch_keys = not_batch_keys or []
    batch_inputs = {
        each_key: each_value for each_key, each_value in inputs.items()
        if each_key not in not_batch_keys}
    not_batch_inputs = {
        each_key: each_value for each_key, each_value in inputs.items()
        if each_key in not_batch_keys}
    batch_inputs_flat = nest_utils.flatten_nested_struct(batch_inputs)
    if ignore_none_values:
        none_keys = [k for k, v in batch_inputs_flat.items() if v is None]
        batch_inputs_flat = {k: v for k, v in batch_inputs_flat.items()
                             if k not in none_keys}
        not_batch_inputs.update(nest_utils.unflatten_dict_to_nested(
            {k: None for k in none_keys}))
    batch_inputs_flat_as_list = (
        nest_utils.dict_of_lists_to_list_of_dicts(batch_inputs_flat))
    batch_inputs_as_list = [
        nest_utils.unflatten_dict_to_nested(each_flat_input)
        for each_flat_input in batch_inputs_flat_as_list]
    return batch_inputs_as_list, not_batch_inputs


def get_is_last_sample_batchwise(
        batch_size: int, is_last_iteration: bool,
        sample_mask: Optional[Union[list, np.ndarray]] = None
) -> list:
    """
    Get the is_last_sample from is_last_iteration and sample_mask for given
    batch_size

    Parameters
    ----------
    sample_mask
        mask of samples
    is_last_iteration
        if it is a last iteration; in that case, last valid item inside of
        is_last_sample_batchwise will be set to 1

    Returns
    -------
    is_last_sample_batchwise
        list with 0 indicating that it is not last sample in iteration and 1
        otherwise
    """
    if not is_last_iteration:
        return [False] * batch_size

    if sample_mask is None:
        sample_mask = [1] * batch_size

    is_last_sample = is_last_iteration
    is_last_sample_batch_rev = []
    for each_sample_mask in sample_mask[::-1]:
        is_last_sample_batch_rev.append(is_last_sample)
        if each_sample_mask:
            is_last_sample = False

    return is_last_sample_batch_rev[::-1]


def string_to_int_code(string: str) -> int:
    """
    Converts the string to an integer code representation.

    Can be used for a random seed generation from a string

    Parameters
    ----------
    string
        string to convert

    Returns
    -------
    int_code
        integer representation of the string
    """
    int_code = zlib.adler32(bytes(string, 'utf-8')) & 0xffffffff
    return int_code
