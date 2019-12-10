# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Utils to work with nested structures
"""

from functools import wraps
from typing import Callable
from typing import Dict
from typing import List
from typing import Union

import numpy as np


def combine_nested(list_of_nested: List[dict],
                   combine_fun: Union[Dict[str, Callable], Callable]) -> dict:
    """
    Combine list of nested structures according to combine_fun

    Parameters
    ----------
    list_of_nested
        list of nested structures (lists, dicts, etc)
    combine_fun
        method of combination, callable taking list of values as input
        or dict with substring to match as keys and corresponding method
        as value together; if there is no matching, the key 'default' will
        be used; callable should take list of inputs together and return
        only one tensor

    Returns
    -------
    nested_combined
        same structure as first element of list_of_nested
    """
    return _combine_nested_r(list_of_nested, combine_fun)


def unflatten_dict_to_nested(d: dict, separator: str = '//') -> dict:
    """
    Unflatten the dictionary with depth of 1 using the separator and forms
    new nested struct

    >>> unflatten_dict_to_nested({'a//b': 1, 'a//c': 2, 'e': 3})
    {'a': {'b': 1, 'c': 2}, 'e': 3}

    >>> unflatten_dict_to_nested({'a//b': 1, 'a//c': 2, 'a//d//d1': 2,
                                  'a//d//d2': 4,
                                  'e//0': 1, 'e//1': 2, 'e//2//e1': 10})
    {'a': {'b': 1, 'c': 2, 'd': {'d1': 2, 'd2': 4}}, 'e': [1, 2, {'e1': 10}]}

    Parameters
    ----------
    d
        1 level depth dictionary
    separator
        separator to use in the name

    Returns
    -------
    nested_d
        2 level dictionary
    """
    # pylint: disable=invalid-name
    # d is self explainable and is part of API
    if not d:
        return d

    result = _unflatten_dict_to_nested_r(d, separator)
    return result


def flatten_nested_struct(d: dict, separator='//',
                          replace_named_tuple_with_dict=False,
                          flatten_lists: bool = True) -> dict:
    """
    Take the nested struct of dicts and concatenates the keys with separator
    to get one level depth dictionary

    >>> flatten_nested_struct({'a': {'b': 1, 'c': 2}, 'e': 3})
    {'a//b': 1, 'a//c': 2, 'e': 3}

    >>> flatten_nested_struct({'a': {'b': 1, 'c': 2, 'd': {'d1': 2, 'd2': 4}},
                               'e': [1, 2, {'e1': 10}]})
    {'a//b': 1, 'a//c': 2, 'a//d//d1': 2, 'a//d//d2': 4,
    'e//0': 1, 'e//1': 2, 'e//2//e1': 10}

    Parameters
    ----------
    d
        2 level dictionary
    separator
        separator to use in the name
    replace_named_tuple_with_dict
        if the named tuples should be treated as a dict
    flatten_lists
        if the lists should be also flatten

    Returns
    -------
    single_d
        1 level dictionary
    """
    # pylint: disable=invalid-name
    # d is self explainable and is part of API

    result = _flatten_nested_struct_r(
        d, separator=separator,
        replace_named_tuple_with_dict=replace_named_tuple_with_dict,
        flatten_lists=flatten_lists)
    return result


def dict_of_lists_to_list_of_dicts(dict_of_lists: Dict[object, list]) -> list:
    """
    Transform the dict of with lists as values to list of dicts with single
    values

    Parameters
    ----------
    dict_of_lists
        dict with values as list

    Returns
    -------
    list_of_dicts
        list with dicts with single items as values

    """
    list_of_dicts = [dict(zip(dict_of_lists, t))
                     for t in zip(*dict_of_lists.values())]
    return list_of_dicts


def flatten_nested_inputs_inside_of_list(*input_names):
    """
    Function decorator to flatten list of nested inputs inside of decorated
    method to list of flatten inputs

    Parameters
    ----------
    input_names
        names of kwargs to flatten inside of decorated method

    Returns
    -------
    decorator
        method decorator

    """

    def wrapper(function):
        def wrapped(*args, **kwargs):
            def _flatten_nested_inside_of_list(list_of_nested):
                flatten_result = [flatten_nested_struct(each_nested)
                                  if each_nested is not None else None
                                  for each_nested in list_of_nested]
                return flatten_result

            kwargs_new = {k: (v if k not in input_names else
                              _flatten_nested_inside_of_list(v))
                          for k, v in kwargs.items()}
            return function(*args, **kwargs_new)

        return wrapped

    return wrapper


def flatten_nested_inputs(*input_names):
    """
    Function decorator to flatten particular inputs inside of decorated method

    Parameters
    ----------
    input_names
        names of kwargs to flatten inside of decorated method

    Returns
    -------
    decorator
        method decorator

    """

    def wrapper(function):
        def wrapped(*args, **kwargs):
            kwargs_new = {k: (v if k not in input_names else
                              flatten_nested_struct(v))
                          for k, v in kwargs.items()}
            return function(*args, **kwargs_new)

        return wrapped

    return wrapper


def unflatten_nested_inputs(*input_names):
    """
    Class decorator to unflatten particular inputs inside of decorated method

    Parameters
    ----------
    input_names
        names of kwargs to unflatten inside of decorated method

    Returns
    -------
    decorator
        class decorator

    """

    def wrapper(function):
        @wraps(function)
        def wrapped(self, *args, **kwargs):
            kwargs_new = {k: (v if k not in input_names else
                              unflatten_dict_to_nested(v))
                          for k, v in kwargs.items()}
            return function(self, *args, **kwargs_new)

        return wrapped

    return wrapper


def flatten_nested_outputs(function: Callable):
    """
    Decorator to flatten the outputs from the decorated function

    Parameters
    ----------
    function
        function to decorate

    Returns
    -------
    decorated_method
        decorated method
    """

    @wraps(function)
    def wrapped(self, *args, **kwargs):
        nested_result = function(self, *args, **kwargs)
        return flatten_nested_struct(nested_result)

    return wrapped


def unflatten_nested_outputs(function: Callable):
    """
    Decorator to unflatten the outputs from the decorated function

    Parameters
    ----------
    function
        function to decorate

    Returns
    -------
    decorated_method
        decorated method
    """

    def wrapped(*args, **kwargs):
        flatten_result = function(*args, **kwargs)
        if not flatten_result:
            return flatten_result
        return unflatten_dict_to_nested(flatten_result)

    return wrapped


def _unflatten_dict_to_nested_r(dict_flatten, separator='//'):
    if not isinstance(dict_flatten, dict):
        return dict_flatten

    keys = sorted(dict_flatten.keys())
    keys_splitted = [k.split(separator) for k in keys]
    keys_this_level = [k[0] for k in keys_splitted]
    keys_next_level = [separator.join(k[1:]) for k in keys_splitted]

    # rearrange for lists
    is_list = False
    if keys_this_level[0].isnumeric():
        is_list = True
        keys_this_level, keys_next_level, keys = zip(
            *sorted(zip(keys_this_level, keys_next_level, keys),
                    key=lambda x: int(x[0])))
    else:
        keys_mapping = {}
        for k_this, k_next in zip(keys_this_level, keys_next_level):
            keys_mapping.setdefault(k_this, [])
            keys_mapping[k_this].append(k_next)
        keys_this_level, keys_next_level = zip(*keys_mapping.items())
        keys_next_level = [[k_ for k_ in k if k_] for k in keys_next_level]

    out = [] if is_list else {}
    for k_this, k_next in zip(keys_this_level, keys_next_level):
        if not k_next:
            d_new = dict_flatten[k_this]
        else:
            if isinstance(k_next, list):
                d_new = {k: dict_flatten[separator.join([k_this, k])]
                         for k in k_next}
            else:
                d_new = {k_next: dict_flatten[separator.join([k_this, k_next])]}
        if is_list:
            out.append(_unflatten_dict_to_nested_r(
                d_new, separator=separator))
        else:
            out[k_this] = _unflatten_dict_to_nested_r(
                d_new, separator=separator)
    return out


def _flatten_nested_struct_r(nested, last_key=None, out=None, separator='//',
                             replace_named_tuple_with_dict=False,
                             flatten_lists: bool = True):
    if out is None:
        out = dict()
    if (replace_named_tuple_with_dict
            and isinstance(nested, tuple) and hasattr(nested, '_asdict')):
        nested = nested._asdict()
    if not isinstance(nested, (list, dict)):
        out[last_key] = nested
    if isinstance(nested, dict):
        for k, each_value in nested.items():
            new_key = separator.join([last_key, k]) if last_key else k
            _flatten_nested_struct_r(
                each_value, new_key, out, separator=separator,
                replace_named_tuple_with_dict=replace_named_tuple_with_dict,
                flatten_lists=flatten_lists)
    elif (flatten_lists
          and isinstance(nested, list)
          and not hasattr(nested, '_fields')):
        for i, each_value in enumerate(nested):
            new_key = separator.join([last_key, str(i)]) if last_key else i
            _flatten_nested_struct_r(
                each_value, new_key, out, separator=separator,
                replace_named_tuple_with_dict=replace_named_tuple_with_dict,
                flatten_lists=flatten_lists)
    else:
        out[last_key] = nested
    return out


def _combine_nested_r(list_of_nested, combine_fun, out=None,
                      combine_fun_last=None):
    """
    Recursively combine the values of list_of_nested using combine_fun
    combined structure will be in the first element of list_nested
    """
    first_item = list_of_nested[0]
    if not isinstance(first_item, (list, dict)):
        if combine_fun_last:
            combine_fn_item = combine_fun
        else:
            combine_fn_item = (combine_fun['default']
                               if isinstance(combine_fun,
                                             dict) else combine_fun)
        out = combine_fn_item(list_of_nested)
        return None

    item_keys = (first_item.keys() if isinstance(first_item, dict) else
                 range(len(first_item)))

    if out is None:
        out = type(first_item)()
    for each_item_key in item_keys:
        values = [l[each_item_key] for l in list_of_nested]
        combine_fn_item = _get_combine_fn(combine_fun, each_item_key)
        if isinstance(first_item[each_item_key], (list, dict)):
            if isinstance(out, list):
                out.append(type(first_item[each_item_key])())
            if isinstance(out, dict):
                out[each_item_key] = type(first_item[each_item_key])()
            _combine_nested_r(values, combine_fun,
                              out=out[each_item_key],
                              combine_fun_last=combine_fn_item)
        else:
            if isinstance(out, list):
                out.append(combine_fn_item(values))
            if isinstance(out, dict):
                out[each_item_key] = combine_fn_item(values)
    return out


def _get_combine_fn(combine_fn: Union[Dict[str, Callable], Callable],
                    item_key: str) -> Callable:
    if isinstance(combine_fn, dict):
        keys = list(combine_fn.keys())
        match_key_inds = np.where(np.greater_equal(
            [item_key.find(k) for k in keys], 0))[0]
        if match_key_inds.size > 0:
            match_key = keys[match_key_inds[0]]
        else:
            match_key = 'default'
        combine_fn = combine_fn[match_key]
        return combine_fn
    return combine_fn
