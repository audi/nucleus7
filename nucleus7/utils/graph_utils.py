# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Classes and functions for regular graph manipulation
"""
import logging
import pprint
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import networkx as nx

from nucleus7.core.nucleotide import Nucleotide
from nucleus7.utils import nucleotide_utils


def construct_graph_from_nucleotides(
        nucleotides: List[Nucleotide],
        incoming_nucleotides: Optional[List[Nucleotide]] = None
) -> nx.DiGraph:
    """
    Construct graph from nucleotides

    Parameters
    ----------
    nucleotides
        nucleotides
    incoming_nucleotides
        incoming nucleotides, which will be added to nucleotides to construct
        the graph, but their connections to their inbound nodes will be
        ignored

    Returns
    -------
    graph
        graph with nucleotides as vertices

    Raises
    ------
    ValueError
        if graph contains loops
    """
    graph = nx.DiGraph()
    incoming_nucleotides = incoming_nucleotides or []
    all_nucleotides = nucleotides + incoming_nucleotides
    all_nucleotides_dict = {each_nucleotide.name: each_nucleotide
                            for each_nucleotide in all_nucleotides}
    for each_nucleotide in nucleotides:
        inbound_nodes = each_nucleotide.inbound_nodes
        if not inbound_nodes:
            graph.add_node(each_nucleotide)
        nucleotide_name = each_nucleotide.name
        for each_in_node_name in inbound_nodes:
            u_node = _get_incoming_node(all_nucleotides_dict, each_in_node_name)
            v_node = all_nucleotides_dict[nucleotide_name]
            graph.add_edge(u_node, v_node)

    try:
        cycles = nx.find_cycle(graph)
        raise ValueError("Cycles in the DNA helix graph were found! "
                         "({})".format(cycles))
    except nx.NetworkXNoCycle:
        pass

    return graph


def topological_sort_of_nucleotides(graph: nx.DiGraph) -> List[Nucleotide]:
    """
    Perform topological order of the graph

    Parameters
    ----------
    graph

    Returns
    -------
    sorted_nucleotides
        list of nucleotides sorted in topological order
    """
    nucleotides_without_inputs = [
        each_nucleotide for each_nucleotide in graph
        if not list(graph.predecessors(each_nucleotide))]
    nucleotides_without_inputs_sorted = sorted(
        nucleotides_without_inputs, key=lambda x: x.name)
    topological_order = list(nx.topological_sort(graph))
    topological_order_only_with_inputs = [
        each_nucleotide for each_nucleotide in topological_order
        if each_nucleotide not in nucleotides_without_inputs]
    topological_order_sorted = (nucleotides_without_inputs_sorted
                                + topological_order_only_with_inputs)
    return topological_order_sorted


def check_class_topology(
        graph: nx.DiGraph,
        dependency_map: dict,
        incoming_nucleotides: Optional[List[Nucleotide]] = None,
        exclude_incoming_nucleotides: bool = True
) -> bool:
    """
    Check class dependency in the model graph

    Parameters
    ----------
    graph
        graph with nucleotides as vertices
    dependency_map
        map describing allowed inbound node classes respect to the node, e.g.
        {:obj:`ModelPlugin`: [:obj:`ModelPlugin`]}
    incoming_nucleotides
        list of incoming nucleotides
    exclude_incoming_nucleotides
        if the incoming nucleotides will be excluded from topologicyl check,
        e.g. they are allowed to be of different classes

    Returns
    -------
    status
        True if check passed else raises :obj:`ValueError`

    Raises
    ------
    ValueError
        if some node has not allowed dependency
    """
    logger = logging.getLogger(__name__)
    logger.info('Check model graph class dependency')
    incoming_nucleotides = incoming_nucleotides or []
    for each_nucleotide in graph:
        if each_nucleotide in incoming_nucleotides:
            continue

        inbound_nucleotides = list(graph.predecessors(each_nucleotide))
        for each_class, allowed_classes in dependency_map.items():
            if isinstance(each_nucleotide, each_class):
                _check_node_dependency_topology(
                    each_nucleotide, inbound_nucleotides, incoming_nucleotides,
                    allowed_classes, exclude_incoming_nucleotides)
    return True


def check_graph_connections(
        graph: nx.DiGraph,
        incoming_nucleotides: Optional[List[Nucleotide]] = None) -> bool:
    """
    Check the graph connectivity according to the input and output names

    Parameters
    ----------
    graph
        graph with nucleotides as vertices
    incoming_nucleotides
        list of incoming nucleotides; will be excluded from check to its inbound
        nodes

    Returns
    -------
    status
        True if check passed else raises :obj:`ValueError`

    Raises
    ------
    ValueError
        if some node needs particular inputs that do not exist in the
        input nodes

    """
    logger = logging.getLogger(__name__)
    logger.info('Check model graph connectivity')
    incoming_nucleotides = incoming_nucleotides or []
    for each_nucleotide in graph:
        if each_nucleotide in incoming_nucleotides:
            continue
        inbound_nucleotides = list(graph.predecessors(each_nucleotide))
        check_node2node_connection(each_nucleotide, inbound_nucleotides)
    return True


def check_node2node_connection(nucleotide: Nucleotide,
                               inbound_nucleotides: List[Nucleotide]) -> bool:
    """
    Check the connectivity of node to all inbound nodes

    Parameters
    ----------
    nucleotide
        nucleotide
    inbound_nucleotides
        set of incoming nucleotides

    Returns
    -------
    status
        True if all arguments to `process` or `predict` method of `node` are
        taken from `inbound_nodes` else raises :obj:`ValueError`

    Raises
    ------
    ValueError
        if the inputs to node cannot be taken from inbound_nodes
    """
    logger = logging.getLogger(__name__)
    logger.debug('Check %s nucleotide connectivity', nucleotide.name)

    incoming_keys_with_description = (
        nucleotide_utils.get_nucleotide_signature(nucleotide)[0])
    (required_keys_missing, optional_keys_missing,
     inbound_nodes_with_description
     ) = _get_missing_keys(nucleotide, inbound_nucleotides)

    inbound_nodes_have_dynamic_keys = any(
        [each_nucleotide.dynamic_generated_keys
         for each_nucleotide in inbound_nucleotides])

    if required_keys_missing:
        incoming_keys_with_description_required = {
            k: incoming_keys_with_description[k]
            for k in required_keys_missing}
        msg = "Missing required incoming keys to {} found: \n\t{}!!!\n".format(
            nucleotide.name, incoming_keys_with_description_required)
        msg += "This method has following input nodes:\n\t{}\n".format(
            pprint.pformat(inbound_nodes_with_description))
        if not inbound_nodes_have_dynamic_keys:
            raise ValueError(msg)
        msg += "Be sure, that they are specified in dynamic keys!"
        logger.info(msg)

    optional_keys_without_input = {}
    for key_optional in nucleotide.incoming_keys_optional:
        if key_optional in optional_keys_missing:
            optional_keys_without_input[key_optional] = (
                incoming_keys_with_description[key_optional])
    if optional_keys_without_input:
        logger.warning(
            "Following optional incoming keys to nucleotide %s have no "
            "input: \n\t%s\n", nucleotide.name, optional_keys_without_input)
    return True


def get_repeated_node_names(nucleotides: List[Nucleotide]) -> Dict[str, list]:
    """
    Return the non unique names of the nucleotides inside of each gene

    Parameters
    ----------
    nucleotides
        list of nucleotides

    Returns
    -------
    repeated_nucleotide_names_to_class_names
        nucleotide names as keys and list of class names as values for repeated
        nucleotides
    """
    nucleotide_names_names_to_class_names = {}
    for each_nucleotide in nucleotides:
        nucleotide_names_names_to_class_names.setdefault(
            each_nucleotide.name, [])
        nucleotide_names_names_to_class_names[each_nucleotide.name].append(
            each_nucleotide.__class__.__name__)

    repeated_nucleotide_names_to_class_names = {
        k: v for k, v in nucleotide_names_names_to_class_names.items()
        if len(v) > 1}
    return repeated_nucleotide_names_to_class_names


def _insert_list_to_index(initial_list, list_to_insert, value_of_index):
    index = initial_list.index(value_of_index)
    new_list = initial_list[:index] + list_to_insert + initial_list[index + 1:]
    return new_list


def _warn_about_nucleotide_without_inputs(nucleotides: Dict[str, Nucleotide]):
    nucleotide_names_without_inputs = (
        _get_nucleotide_names_without_inbound_nodes(nucleotides)
    )
    logger = logging.getLogger(__name__)
    for nucleotide_name in nucleotide_names_without_inputs:
        nucleotide = nucleotides[nucleotide_name]
        required_incoming_keys = nucleotide.incoming_keys_required
        if required_incoming_keys:
            logger.warning(
                "Nucleotide %s doesn't have any inbound_nodes, "
                "but have incoming_keys in its implementation",
                nucleotide_name)
        else:
            logger.info("Nucleotide %s doesn't have any inbound_nodes",
                        nucleotide_name)


def _get_missing_keys(
        nucleotide: Nucleotide, inbound_nucleotides: List[Nucleotide]
) -> Tuple[set, set, dict]:
    logger = logging.getLogger(__name__)
    incoming_keys_required = set(nucleotide.incoming_keys_required)
    incoming_keys_optional = set(nucleotide.incoming_keys_optional)
    inbound_nodes_with_description = {}
    for inbound_node in inbound_nucleotides:
        incoming_keys_map = nucleotide.incoming_keys_mapping.get(
            inbound_node.name, {})
        incoming_keys_map = {k.split(':')[0]: v.split(':')[0]
                             for k, v in incoming_keys_map.items()}
        inbound_node_returns = (
            nucleotide_utils.get_nucleotide_signature(inbound_node)[1])
        mapped_returns = {incoming_keys_map.get(k, k): v
                          for k, v in inbound_node_returns.items()}

        inbound_nodes_with_description[inbound_node.name] = {
            'returns': inbound_node_returns,
            'mapping': incoming_keys_map
        }

        for inbound_node_return in mapped_returns:
            if inbound_node_return in incoming_keys_required:
                incoming_keys_required.remove(inbound_node_return)
                logger.debug('...Incoming key %s found',
                             inbound_node_return)

            if inbound_node_return in incoming_keys_optional:
                incoming_keys_optional.remove(inbound_node_return)
                logger.debug('...Optional incoming key %s found',
                             inbound_node_return)
    return (incoming_keys_required, incoming_keys_optional,
            inbound_nodes_with_description)


def _get_nucleotide_names_without_inbound_nodes(
        nucleotides: Dict[str, Nucleotide]) -> List[str]:
    nucleotides_without_input = [each.name for each in nucleotides.values()
                                 if not each.inbound_nodes]
    return nucleotides_without_input


def _check_node_dependency_topology(
        nucleotide: Nucleotide,
        inbound_nucleotides: List[Nucleotide],
        graph_incoming_nucleotides: List[Nucleotide],
        allowed_dependency_classes: List[type],
        exclude_incoming_nucleotides: bool = True):
    logger = logging.getLogger(__name__)
    for inbound_nucleotide in inbound_nucleotides:
        if (exclude_incoming_nucleotides
                and inbound_nucleotide in graph_incoming_nucleotides):
            continue
        class_found = False
        for allowed_class in allowed_dependency_classes:
            if isinstance(inbound_nucleotide, allowed_class):
                logger.debug(
                    "Nucleotide %s of type %s has inbound nucleotide %s "
                    "of type %s, which is OK",
                    nucleotide.name, nucleotide.__class__.__name__,
                    inbound_nucleotide.name,
                    inbound_nucleotide.__class__.__name__)
                class_found = True
                break
        if not class_found:
            msg = ("Nucleotide {} of type {} has inbound nucleotide {} "
                   "of type {}, which is not allowed!!!"
                   ).format(nucleotide.name, nucleotide.__class__.__name__,
                            inbound_nucleotide.name,
                            inbound_nucleotide.__class__.__name__)
            raise ValueError(msg)


def _get_incoming_node(all_nucleotides: dict, incoming_node_name: str):
    logger = logging.getLogger(__name__)
    try:
        return all_nucleotides[incoming_node_name]
    except KeyError as e:  # pylint: disable=invalid-name
        for each_nucleotide in all_nucleotides.values():
            if hasattr(each_nucleotide, "build_dna"):
                if incoming_node_name in each_nucleotide.all_nucleotides:
                    logger.info("Found reference to nucleotide %s as part of "
                                "%s", incoming_node_name, each_nucleotide.name)
                    return each_nucleotide
        raise e
