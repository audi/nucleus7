# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
IO utils
"""

import argparse
import ast
from collections import OrderedDict
import inspect
import json
import logging
import os
from typing import Callable
from typing import Union

from nucleus7.utils import numpydoc_utils
from nucleus7.utils import object_utils


def maybe_load_json(x: Union[str, dict, list], as_ordereddict: bool = False
                    ) -> Union[dict, OrderedDict, list, str]:
    """
    Check whther the x is file and in this case loads as json

    Parameters
    ----------
    x
        path to json file or a string with serialized json or any other object
    as_ordereddict
        controls if the resulting dict in case if x is the path to json
        will be returned as a ordered dict

    Returns
    -------
    loaded_object
        laded json or literally evaluated json string or x itself
    """
    # pylint: disable=invalid-name
    # x is self explainable and is part of API
    if isinstance(x, str) and os.path.isfile(x):
        return load_json(x, as_ordereddict)
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except ValueError:
            pass
    return x


def load_json(fname: str, as_ordereddict: bool = False,
              remove_private: bool = True) -> Union[dict, OrderedDict]:
    """
    Load the json file

    Parameters
    ----------
    fname
        path to json file
    as_ordereddict
        controls if the resulting dict in case if x is the path to json
        will be returned as a ordered dict
    remove_private
        if the private keys inside of json, e.g. starting with '_' should be
        removed

    Returns
    -------
    dict_from_json
        loaded json
    """
    logger = logging.getLogger(__name__)
    object_pairs_hook = OrderedDict if as_ordereddict else None
    try:
        with open(fname, 'r', encoding='utf8') as file:
            data = json.load(file, object_pairs_hook=object_pairs_hook)
            if remove_private and isinstance(data, dict):
                for k in list(data):
                    if k[:1] == '_':
                        del data[k]
                        logger.debug('Tag %s in file %s omitted',
                                     k, fname)
            return data
    # pylint: disable=invalid-name
    # is common practice to call exceptions as e
    except json.decoder.JSONDecodeError as e:
        logger.error("File %s corrupted!!!", fname)
        raise e


def is_jsonserializable(x: object) -> bool:
    """
    Check if the object is json serializable

    Parameters
    ----------
    x
        object to check

    Returns
    -------
    status
        True if object can be serialized as json and False otherwise
    """
    # pylint: disable=invalid-name
    # x is self explainable and is part of API
    try:
        json.dumps(x)
        return True
    except TypeError:
        return False


def add_args_from_method_to_argparse(parser: argparse.ArgumentParser,
                                     cls: Callable,
                                     method: Callable,
                                     method_with_doc: Callable = None,
                                     include_parents: bool = True
                                     ) -> argparse.ArgumentParser:
    """
    Read the args from method and add it to argparse parser

    Parameters
    ----------
    parser
        parser to add the arguments
    cls
        class with method to parse
    method
        callable to get the arguments
    method_with_doc
        method to read the docs; if not specified, then original method will
        be used
    include_parents
        if parent methods should be traced too

    Returns
    -------
    parser_new
        parser with added arguments
    """
    signatures = object_utils.get_method_signatures(
        cls, method, include_parents)
    method_with_doc = method_with_doc or method

    doc = numpydoc_utils.get_docs_of_method(
        cls, method_with_doc, include_parents)
    for i_sign, signature in enumerate(signatures):
        for arg_name in signature.parameters:
            if arg_name == 'self':
                continue
            if "kwargs" in arg_name:
                continue
            if "args" in arg_name:
                continue
            try:
                default = signature.parameters[arg_name].default
                # pylint: disable=protected-access
                # _empty must be used since it is a part of inspect
                arg_type = (type(default)
                            if default not in [inspect._empty, None]
                            else None)
                if default is inspect._empty and i_sign > 0:
                    required = True
                    default = None
                else:
                    required = False
                # pylint: enable=protected-access
                action = 'store'
                if arg_type in (list, tuple):
                    arg_type = type(default[0])
                    action = 'append'
                    default = None
                arg_help = doc.get(arg_name)
                if arg_help is not None:
                    arg_help = arg_help.strip()
                parser.add_argument(
                    '--' + arg_name, type=arg_type, action=action,
                    required=required, default=default,
                    help=arg_help)
            except argparse.ArgumentError:
                continue
    return parser


def maybe_mkdir(path: str):
    """
    Create directory if it not exists

    Parameters
    ----------
    path
        directory path
    """
    logger = logging.getLogger(__name__)
    if not os.path.exists(path):
        logger.debug('Create directory %s', path)
        os.mkdir(path)
    else:
        logger.debug('Directory %s already exists', path)


def print_gflags(args: argparse.Namespace, space_n: int = 10):
    """
    Get flags as a string to print

    Parameters
    ----------
    args
        used flags
    space_n
        number of spaces for tabulation

    Returns
    -------
    flags_string
        string to print the flags

    """
    flags_str = "Used ARGS:\n"
    flags_keys = dir(args)
    for k in flags_keys:
        if 'help' not in k and k[0] != '_':
            if getattr(args, k):
                flags_str += "{}{}: {}\n".format(" " * space_n,
                                                 k, getattr(args, k))
    return flags_str
