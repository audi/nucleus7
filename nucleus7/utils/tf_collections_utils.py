# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Utils to work with tensorflow collections
"""

from functools import wraps
import logging
from typing import Callable
from typing import List
from typing import Optional

import tensorflow as tf

from nucleus7.utils import nest_utils


def get_nucleotide_names_from_collections(graph: tf.Graph,
                                          collection_list: list) -> list:
    """
    Get all nucleotide names from graph collection list

    Parameters
    ----------
    graph
        tensorflow graph
    collection_list
        list of collections to search

    Returns
    -------
    list_of_nucleotide_names
        list of nucleotide names inside of collection_list

    """
    graph = _maybe_get_default_graph(graph)
    node_names = []
    for collection_name in collection_list:
        collection = graph.get_collection(collection_name)
        for each_item in collection:
            if isinstance(each_item, dict):
                node_names.extend(
                    [node.name.split(':')[0] for node in each_item.values()])
            elif isinstance(each_item, (list, tuple)):
                node_names.extend(
                    [node.name.split(':')[0] for node in each_item])
            else:
                node_names.append(each_item.name.split(':')[0])
    return node_names


def get_collections_by_prefix(collection_prefix: str,
                              separator: str = '::',
                              graph: tf.Graph = None,
                              raise_error: bool = True) -> List[str]:
    """
    Get list of collection names if first split value of collection name
    is `collection_prefix`. Collection names are split by `separator`

    Parameters
    ----------
    collection_prefix
        prefix to collection name
    separator
        separator to separate the suffix from key
    graph
        graph to add the collections
    raise_error
        if ValueError should be raised if no collections with defined prefix
        were found

    Returns
    -------
    collection_list
        list of full collection names

    Raises
    ------
    ValueError
        if there are no collections with that prefix (if `raise_error==True`)
    """
    graph = _maybe_get_default_graph(graph)
    collection_keys = graph.get_all_collection_keys()
    collection_list = [ck for ck in collection_keys
                       if ck.split(separator)[0] == collection_prefix]
    if raise_error and not collection_list:
        raise ValueError('No collections with prefix {} found!!!'.format(
            collection_prefix))
    return collection_list


def nested2collection(collection_prefix: str,
                      values: dict,
                      separator: str = '::',
                      graph: Optional[tf.Graph] = None):
    """
    Add values from nested dict / list values to the collection with name
    `{collection_prefix}{separator}{key}`
    If the key starts with `_`, it will be ignored

    Parameters
    ----------
    values
        dict with str as keys and tensors / variables / ops as values
    collection_prefix
        prefix to collection name
    separator
        separator to separate the suffix from key
    graph
        graph to add the collections

    Raises
    ------
    ValueError
        if there are no collections with that prefix (if `raise_error==True`)

    """
    graph = _maybe_get_default_graph(graph)
    unflatten_values = nest_utils.flatten_nested_struct(values)
    for key, value in unflatten_values.items():
        if key[0] == '_':
            continue
        collection_name = separator.join([collection_prefix, key])
        if isinstance(value, list):
            for each_value in value:
                graph.add_to_collection(collection_name, each_value)
        else:
            graph.add_to_collection(collection_name, value)


def collection2nested(collection_prefix: str,
                      separator: str = '::',
                      graph: Optional[tf.Graph] = None,
                      raise_error: bool = True) -> dict:
    """
    Construct the dict with keys as suffixes of collection
    names and values as corresponding tensors / variables / ops
    or list / value depending on the found collection_list

    Parameters
    ----------
    collection_prefix
        prefix to collection name
    separator
        separator to separate the suffix from key
    graph
        graph to add the collections
    raise_error
        if ValueError should be raised if no collections with defined prefix
        were found

    Returns
    -------
    dict_with_values
        dict with values from collections

    """
    graph = _maybe_get_default_graph(graph)
    collection_list = get_collections_by_prefix(
        collection_prefix, separator, graph, raise_error=raise_error)

    if not collection_list:
        return {}
    if len(collection_list) == 1 and collection_list[0] == collection_prefix:
        return graph.get_collection(collection_prefix)[0]

    result = {}
    for collection_name in collection_list:
        if len(graph.get_collection(collection_name)) == 1:
            value = graph.get_collection(collection_name)[0]
        else:
            value = graph.get_collection(collection_name)
        name_splitted = collection_name.split(separator)
        key = name_splitted[1]
        result[key] = value
    if len(result) == 1 and list(result.keys())[0] == "":
        result = list(result.values())[0]

    result = nest_utils.unflatten_dict_to_nested(result)
    return result


def remove_variables_from_collection(collection_name: str,
                                     variable_names_to_remove: list,
                                     graph: Optional[tf.Graph] = None):
    """
    Remove variable from collection. Can be used to freeze some
    variables by removing them from trainable collection

    Parameters
    ----------
    collection_name
        name of collection
    variable_names_to_remove
        list of variable names to remove from collection
    graph
        if not provided, default graph will be used
    """
    graph = _maybe_get_default_graph(graph)
    collection_vars = tf.get_collection(collection_name)
    graph.clear_collection(collection_name)
    collection_vars_after_removing = [
        v for v in collection_vars
        if v.name not in variable_names_to_remove]
    for each_variable in collection_vars_after_removing:
        tf.add_to_collection(collection_name, each_variable)


def remove_not_trainable_from_collection(function: Callable) -> Callable:
    """
    Decorator to remove the variables from trainable collection if
    self.trainable is False
    """

    @wraps(function)
    def wrapped(self, *args, **kwargs):
        res = function(self, *args, **kwargs)
        if not self.trainable:
            remove_variables_from_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                [v.name for v in self.variables])
        return res

    return wrapped


def rename_collection(name_to_rename: str,
                      new_name: str,
                      graph: Optional[tf.Graph] = None):
    """
    Rename collection

    Parameters
    ----------
    name_to_rename
        collection name to rename
    new_name
        new collection name
    graph
        tensorflow graph

    Raises
    ------
    ValueError
        if collection with old_name is not found

    """
    graph = _maybe_get_default_graph(graph)

    existing_collections = graph.get_all_collection_keys()
    assert name_to_rename in existing_collections, \
        "Collection with name {} not found (existing collections: {})".format(
            name_to_rename, existing_collections)

    collection_vars = tf.get_collection(name_to_rename)
    graph.clear_collection(name_to_rename)
    for var in collection_vars:
        graph.add_to_collection(new_name, var)


def rename_collections_with_prefix(prefix_to_rename: str,
                                   new_prefix: str,
                                   graph: tf.Graph = None,
                                   separator_to_rename: str = '::',
                                   new_separator: str = "::"):
    """
    Select collections according to prefix and rename that collections by
    renaming the prefix

    Parameters
    ----------
    prefix_to_rename
        prefix to rename
    new_prefix
        new prefix
    graph
        tensorflow graph
    separator_to_rename
        separator to use to separate the prefix
    new_separator
        new prefix separator

    """
    graph = _maybe_get_default_graph(graph)
    collection_names_with_prefix_to_rename = get_collections_by_prefix(
        prefix_to_rename, separator=separator_to_rename, graph=graph)
    for each_old_collection_name in collection_names_with_prefix_to_rename:
        new_collection_namename = each_old_collection_name.replace(
            separator_to_rename, new_separator)
        new_collection_namename = new_collection_namename.replace(
            prefix_to_rename, new_prefix, 1)
        rename_collection(each_old_collection_name, new_collection_namename,
                          graph=graph)


def add_level_to_collection_with_prefix(
        level_name: str,
        collection_prefix: str,
        graph: tf.Graph = None,
        collection_separator: str = '::',
        level_separator: str = '//',
        collection_names_to_exclude: list = None):
    """
    Add new level before first level of collections under certain prefix

    e.g.: 'prefix::level1//level1' -> 'prefix::level_name//level1//level2'

    Parameters
    ----------
    level_name
        level name to add to prefix
    collection_prefix
        prefix of collections to rename
    graph
        tensorflow graph
    collection_separator
        separator of collections prefix
    level_separator
        separator of the levels
    collection_names_to_exclude
        collections to exclude from renaming

    """
    graph = _maybe_get_default_graph(graph)
    collection_names_with_prefix_to_rename = get_collections_by_prefix(
        collection_prefix, separator=collection_separator, graph=graph,
        raise_error=False)
    if collection_names_with_prefix_to_rename is None:
        logger = logging.getLogger(__name__)
        logger.warning("Collection with prefix %s is not found",
                       collection_prefix)
        return

    for each_collection_name in collection_names_with_prefix_to_rename:
        collection_nested_levels = each_collection_name.split(
            collection_separator)[-1]
        collection_nested_levels = collection_nested_levels.split(
            level_separator)
        if collection_names_to_exclude:
            if collection_nested_levels[0] in collection_names_to_exclude:
                continue
        collection_name_with_new_level = level_separator.join(
            [level_name] + collection_nested_levels)
        collection_name_with_new_level = (collection_prefix
                                          + collection_separator
                                          + collection_name_with_new_level)
        rename_collection(each_collection_name, collection_name_with_new_level,
                          graph=graph)


# TODO(oleksandr.vorobiov@audi.de) make API more clear
def add_prefix_to_collection_with_prefix(
        level_prefix: str,
        collection_prefix: str,
        graph: Optional[tf.Graph] = None,
        collection_separator: str = '::',
        level_separator: str = '//',
        prefix_separator: str = '_',
        collection_names_to_leave: Optional[list] = None):
    """
    Add prefix to collection that already has a prefix

    Parameters
    ----------
    level_prefix
        prefix to add
    collection_prefix
        prefix of the collection to add to
    graph
        tensorflow graph
    collection_separator
        name separator used inside of the collection
    level_separator
        separator to use in new collections
    prefix_separator
        separator to use to separate new prefix
    collection_names_to_leave
        to this collections will not be added the level_prefix
    """
    graph = _maybe_get_default_graph(graph)
    collection_names_with_prefix_to_rename = get_collections_by_prefix(
        collection_prefix, separator=collection_separator, graph=graph,
        raise_error=False)
    if collection_names_with_prefix_to_rename is None:
        logger = logging.getLogger(__name__)
        logger.warning("Collection with prefix %s is not found",
                       collection_prefix)
        return

    for each_collection_name in collection_names_with_prefix_to_rename:
        collection_nested_levels = each_collection_name.split(
            collection_separator)[-1]
        collection_nested_levels = collection_nested_levels.split(
            level_separator)
        skip = False
        if collection_names_to_leave:
            for collection_to_leave in collection_names_to_leave:
                if collection_nested_levels[0].startswith(collection_to_leave):
                    skip = True
                    continue
        if skip:
            continue

        collection_name_with_nested_levels = (
            level_separator.join(collection_nested_levels))
        collection_name_with_new_level = (level_prefix + prefix_separator
                                          + collection_name_with_nested_levels)
        collection_name_with_new_level = (collection_prefix
                                          + collection_separator
                                          + collection_name_with_new_level)
        rename_collection(each_collection_name, collection_name_with_new_level,
                          graph=graph)


def _maybe_get_default_graph(graph):
    if graph is None:
        graph = tf.get_default_graph()
    return graph
