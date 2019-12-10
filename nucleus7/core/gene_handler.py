# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Interface for gene handler
"""
from abc import abstractmethod
from functools import partial
import logging
import pprint
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import tensorflow as tf

from nucleus7.core.base import BaseClass
from nucleus7.core.dna_helix import DNAHelix
from nucleus7.core.nucleotide import Nucleotide
from nucleus7.utils import graph_utils
from nucleus7.utils import nucleotide_utils
from nucleus7.utils import object_utils

# pylint: disable=invalid-name
# these are type constants, not a class
_TENSOR_OR_NPARRAY = Union[tf.Tensor, np.ndarray]

_NESTED_TENSOR_OR_NPARRAY = Union[_TENSOR_OR_NPARRAY,
                                  Dict[str, _TENSOR_OR_NPARRAY]]

_NUCLEOTIDE_INPUT_PREPROCESSING_FN_TYPE = Callable[
    [Tuple[Nucleotide, Dict[str, _NESTED_TENSOR_OR_NPARRAY]]],
    Dict[str, _NESTED_TENSOR_OR_NPARRAY]]

_INPUT_NUCLEOTIDES_TYPE = Union[Sequence[Nucleotide], Dict[str, Nucleotide],
                                Nucleotide]


# pylint: enable=invalid-name

class GeneHandler(BaseClass):
    """
    This class is needed to handle the execution of multiple genes (collections
    of nucleotides)

    Parameters
    ----------
    **genes
        dict with mapping of gene name, e.g. callbacks, plugins etc. to the
        collection of nucleotides inside of that gene (dict, list or single
        nucleotide)

    Attributes
    ----------
    nucleotide_type_dependency_map
        describes possible node class dependency inside of model graph, e.g.
        nucleotide_class_1 can have only input nodes of type
        nucleotide_class_2 and nucleotide_class_3
    gene_name_and_nucleotide_super_cls
    _sorted_nucleotide_names : list
        dict with keys as model part, like 'plugin', 'loss' and values is list
        of names in topological sorted order
    _all_nucleotides : dict
        mapping {name : nucleotide} for all nucleotides in model

    """
    nucleotide_type_dependency_map = {
        Nucleotide: [Nucleotide]
    }  # type: Dict[Nucleotide, List[Nucleotide]]
    gene_name_and_nucleotide_super_cls = {
        'gene': Nucleotide
    }  # type: Dict[str, Nucleotide]

    def __init__(self, **genes: _INPUT_NUCLEOTIDES_TYPE):
        super().__init__()
        for gene_name in self.gene_name_and_nucleotide_super_cls:
            gene_of_type = genes.get(gene_name)
            setattr(self, gene_name, gene_of_type)

        self._dna_helix = None  # type: Optional[DNAHelix]
        self._sorted_nucleotide_names = {}  # type: Dict[str, List[str]]
        self._all_nucleotides = None  # type: Optional[Dict[str, Nucleotide]]
        self._mode = None  # type: Optional[str]

    @property
    def inbound_nodes(self) -> List[str]:
        """
        Get all inbound nodes for all genes

        Returns
        -------
        inbound_nodes_for_handler
            inbound nodes
        """
        inbound_nodes_from_single_nucleotides = (
            self.get_property_from_all_genes('inbound_nodes'))
        inbound_nodes_from_single_nucleotides = {
            each_node for each_node_list
            in inbound_nodes_from_single_nucleotides.values()
            for each_node in each_node_list
        }
        all_nucleotide_names = set(
            self.get_property_from_all_genes('name').values())
        all_nucleotide_nested_names = {
            each_sub_name for each_nucleotide in self.all_nucleotides.values()
            for each_sub_name in getattr(each_nucleotide, "all_nucleotides", [])
        }
        all_nucleotide_names = set.union(all_nucleotide_names,
                                         all_nucleotide_nested_names)
        inbound_nodes_for_handler = [
            nucleotide_name
            for nucleotide_name in inbound_nodes_from_single_nucleotides
            if nucleotide_name not in all_nucleotide_names]
        return inbound_nodes_for_handler

    @property
    def dna_helix(self) -> Optional[DNAHelix]:
        """
        Get DNA helix in format
        {node: {'in': set(input nodes), 'out': set(output nodes)}}

        Returns
        -------
        dna_helix
            dna helix
        """
        return self._dna_helix

    @property
    def all_nucleotides(self) -> Dict[str, Nucleotide]:
        """
        All nucleotides without belonging gene

        Returns
        -------
        all_nucleotides
            all nucleotides inside of the handler
        """
        return self._all_nucleotides

    @property
    def sorted_nucleotide_names(self) -> Dict[str, Sequence[str]]:
        """
        Nucleotide names in the topological sorted order

        Returns
        -------
        sorted_nucleotide_names
            sorted nucleotide names
        """
        return self._sorted_nucleotide_names

    @property
    def mode(self) -> Optional[str]:
        """
        Mode of the handler, e.g. train, eval etc

        Returns
        -------
        mode
            mode
        """
        return self._mode

    @mode.setter
    def mode(self, mode: str):
        """
        Set the mode to model and also to all nucleotides in all genes

        Parameters
        ----------
        mode
            mode to set
        """
        self._mode = mode
        self._set_property_for_all_genes('mode', self._mode)

    @property
    def is_training(self) -> bool:
        """
        If mode == train, e.g. training mode

        Returns
        -------
        is_training_flag
            True if mode == train, else otherwise
        """
        return self._mode == tf.estimator.ModeKeys.TRAIN

    @property
    def defaults(self):
        defaults_dict = {
            gene_name: partial(_create_named_dict,
                               super_cls=nucleotide_super_cls)
            for gene_name, nucleotide_super_cls in
            self.gene_name_and_nucleotide_super_cls.items()
        }
        return defaults_dict

    def build(self):
        """
        Build the gene handler, e.g. make the genes to have form of
        {gene_name}: [list of nucleotides for that gene]

        Raises
        ------
        AssertionError
            if there are nucleotides with repeated names inside of all genes
        """
        super().build()
        self._check_repeated_nodes()
        all_nucleotides = {}
        for gene_name in self.gene_name_and_nucleotide_super_cls:
            all_nucleotides.update(getattr(self, gene_name))
        self._all_nucleotides = all_nucleotides
        return self

    @object_utils.assert_is_built
    def build_dna(
            self,
            incoming_nucleotides: Optional[_INPUT_NUCLEOTIDES_TYPE] = None):
        """
        Construct graph of the model genes and perform the topological sort of
        all nucleotides.

        Also multiple checks are made:

            1. Class dependency of nucleotides, e.g. nucleotide_class_1 node can
                depend only on `nucleotide_class_2 node or on incoming genes;
                this dependency map is declared in
                :obj:`GeneHandler._nucleotide_type_dependency_map`.
                If the check failed, :obj:`ValueError` will be raised
            2. Node connectivity according to node inputs and inbounded node
                outputs according to provided mapping inside of
                :obj:`Nucleotide.incoming_keys_mapping`, e.g. if node has
                inputs `['input', 'input2']` and inbound node has outputs
                `['output', 'output2']` then to pass this check, mapping should
                be {'output': 'input', 'output2': 'input2'}. If check is
                failed, :obj:`ValueError` will be raised

        Parameters
        ----------
        incoming_nucleotides
            incoming genes to this handler, e.g. collections of nucleotides

        Raises
        ------
        ValueError
            if not all nucleotides are connected
        ValueError
            if some nucleotide has not allowed dependency

        """
        logger = logging.getLogger(__name__)
        logger.info('Build dna for %s', self.__class__.__name__)
        incoming_nucleotides = _create_named_dict(
            incoming_nucleotides, Nucleotide)
        all_nucleotides = self._all_nucleotides

        self._dna_helix = DNAHelix(
            list(all_nucleotides.values()),
            list(incoming_nucleotides.values())).build()
        self._dna_helix.check_class_topology(
            self.nucleotide_type_dependency_map)

        for gene_name in self.gene_name_and_nucleotide_super_cls:
            self._sorted_nucleotide_names[gene_name] = [
                each_sorted_name
                for each_sorted_name in self._dna_helix.names_topological_order
                if each_sorted_name in getattr(self, gene_name)]
        self._maybe_build_nucleotides_dna(incoming_nucleotides)

    @object_utils.assert_is_built
    @object_utils.assert_property_is_defined('dna_helix')
    def process_gene(
            self,
            gene_name: str, *,
            gene_inputs: Dict[str, _NESTED_TENSOR_OR_NPARRAY] = None,
            nucleotide_inputs_preprocessing_fn:
            _NUCLEOTIDE_INPUT_PREPROCESSING_FN_TYPE = None,
            **additional_call_kwargs
    ) -> Union[Dict[str, _NESTED_TENSOR_OR_NPARRAY]]:
        """
        Make forward pass for different types of genes (sequence of
        nucleotides), like plugins, loss, metric, callbacks etc.

        Parameters
        ----------
        gene_name
            name of process, like 'plugin', 'loss', 'callback' etc.
        gene_inputs
            inputs to these gene from other genes, e.g. nucleotides;
            must have a flatten for, e.g. 'nucleotide1//out1//out11' = 'value'
        nucleotide_inputs_preprocessing_fn
            function that will be called on the selected and filtered nucleotide
            inputs before nucleotide call
        additional_call_kwargs
            kwargs that will be additionally passed to the nucleotide call

        Returns
        -------
        gene_outputs
            dict with outputs for each gene nucleotide with keys as nucleotide
            names

        """
        logger = logging.getLogger(__name__)
        logger.info('Process %s gene:', gene_name)

        gene_inputs = gene_inputs or {}
        gene_outputs = {}

        sorted_nucleotide_names = self._sorted_nucleotide_names[gene_name]

        for nucleotide_name in sorted_nucleotide_names:
            logger.info("\tprocess nucleotide %s", nucleotide_name)
            nucleotide = self._all_nucleotides[nucleotide_name]
            nucleotide_inputs = {}
            nucleotide_inputs.update(gene_inputs)
            nucleotide_inputs.update(gene_outputs)
            if not _is_composed_nucleotide(nucleotide):
                nucleotide_inputs_filtered = nucleotide.filter_inputs(
                    nucleotide_inputs)
                if nucleotide_inputs_filtered is None:
                    gene_outputs[nucleotide_name] = None
                    logger.info(
                        "Nucleotide %s is ignored due to not complete inputs",
                        nucleotide_name)
                    continue

                if nucleotide_inputs_preprocessing_fn is not None:
                    nucleotide_inputs_filtered = (
                        nucleotide_inputs_preprocessing_fn(
                            nucleotide, nucleotide_inputs_filtered)
                    )
            else:
                nucleotide_inputs_filtered = nucleotide_inputs
            nucleotide_output = nucleotide(**nucleotide_inputs_filtered,
                                           **additional_call_kwargs)

            if _is_composed_nucleotide(nucleotide):
                gene_outputs.update(nucleotide_output)
            else:
                gene_outputs[nucleotide_name] = nucleotide_output
        return gene_outputs

    def assert_all_nucleotides_have_same_mode(self):
        """
        Assert that all nucleotides in all genes have same mode as self._mode

        Raises
        ------
        ValueError
            if any nucleotide inside of any gene has different mode compared
            to self.mode

        """
        self._assert_all_genes_have_same_property('mode')

    def get_property_from_all_genes(self,
                                    property_name: str) -> Dict[str, List[str]]:
        """
        Get the value of property with name for all nucleotides

        Parameters
        ----------
        property_name
            name of the property

        Returns
        -------
        property_from_all_genes
            mapping of nucleotide name to its property value


        Raises
        ------
        ValueError
            if any nucleotide inside of any gene has different property value
            compared to self

        """
        all_gene_properties = {}
        for gene_name in self.gene_name_and_nucleotide_super_cls:
            gene = getattr(self, gene_name)
            for each_nucleotide in gene.values():
                all_gene_properties[each_nucleotide.name] = (
                    getattr(each_nucleotide, property_name))
        return all_gene_properties

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @staticmethod
    def flatten_results(nested_results: dict,
                        keys_mapping: Optional[Dict] = None) -> dict:
        """
        Remap results got from gene handler according to mapping

        Parameters
        ----------
        nested_results
            results
        keys_mapping
            mapping of type {nucleotide_name: {result_key: remapped:key}}

        Returns
        -------
        flatten_results
            flattened results from nucleotides
        """
        sorted_nucleotide_names = sorted(nested_results.keys())
        remapped_and_collapsed_results = (
            nucleotide_utils.remap_and_collapse_nucleotide_inputs(
                nested_results, sorted_nucleotide_names, keys_mapping))
        return remapped_and_collapsed_results

    @object_utils.assert_is_built
    def get_flat_generated_keys_required(
            self,
            keys_mapping: Optional[Dict] = None,
    ) -> List[str]:
        """
        Flatten required generated keys from all nucleotides and remap
        them according to keys_mapping

        Parameters
        ----------
        keys_mapping
            mapping of type {nucleotide_name: {result_key: remapped:key}}

        Returns
        -------
        flat_generated_keys_required
            flat required generated keys
        """
        generated_keys_required = {
            each_name: dict(zip(*[each_nucleotide.generated_keys_required] * 2))
            for each_name, each_nucleotide in self.all_nucleotides.items()}
        generated_keys_required_flat = self._flatten_keys(
            generated_keys_required, keys_mapping)
        return generated_keys_required_flat

    @object_utils.assert_is_built
    def get_flat_generated_keys_optional(
            self,
            keys_mapping: Optional[Dict] = None,
    ) -> List[str]:
        """
        Flatten optional generated keys from all nucleotides and remap
        them according to keys_mapping

        Parameters
        ----------
        keys_mapping
            mapping of type {nucleotide_name: {result_key: remapped:key}}

        Returns
        -------
        flat_generated_keys_optional
            flat optional generated keys
        """
        generated_keys_optional = {
            each_name: dict(zip(*[each_nucleotide.generated_keys_optional] * 2))
            for each_name, each_nucleotide in self.all_nucleotides.items()}
        generated_keys_optional_flat = self._flatten_keys(
            generated_keys_optional, keys_mapping)
        return generated_keys_optional_flat

    def _flatten_keys(self, keys, keys_mapping):
        keys = {k: v for k, v in keys.items() if v}
        keys_combined = sorted(self.flatten_results(keys, keys_mapping))
        return keys_combined

    def _check_repeated_nodes(self):
        all_nucleotides = [
            nucleotide
            for gene_name in self.gene_name_and_nucleotide_super_cls
            for nucleotide in getattr(self, gene_name).values()
        ]
        repeated_nucleotide_names = graph_utils.get_repeated_node_names(
            all_nucleotides)
        repeated_nucleotide_names_len = len(repeated_nucleotide_names)
        assert repeated_nucleotide_names_len == 0, (
            "Repeated nucleotide names found!!! {}".format(
                pprint.pformat(repeated_nucleotide_names))
        )

    def _set_property_for_all_genes(self, property_name: str,
                                    property_value):
        for gene_name in self.gene_name_and_nucleotide_super_cls:
            gene = getattr(self, gene_name)
            for each_nucleotide in gene.values():
                setattr(each_nucleotide, property_name, property_value)

    def _assert_all_genes_have_same_property(self, property_name: str):
        """
        Assert that all nucleotides have same property value as handler

        Raises
        ------
        ValueError
            if any nucleotide inside of any gene has different property value
            compared to self

        """
        for gene_name in self.gene_name_and_nucleotide_super_cls:
            gene = getattr(self, gene_name)
            for each_nucleotide in gene.values():
                if (getattr(each_nucleotide, property_name)
                        != getattr(self, '_' + property_name)):
                    raise ValueError(
                        "Nucleotide {} has different {} as handler {}".format(
                            each_nucleotide.name, property_name,
                            self.__class__.__name__
                        )
                    )

    def _maybe_build_nucleotides_dna(self, incoming_nucleotides):
        for each_nucleotide in self._all_nucleotides.values():
            if hasattr(each_nucleotide, "build_dna"):
                incoming_nucleotides = [
                    self.all_nucleotides.get(
                        each_inbound_node_name,
                        incoming_nucleotides[each_inbound_node_name])
                    for each_inbound_node_name in
                    each_nucleotide.inbound_nodes]
                each_nucleotide.build_dna(incoming_nucleotides)


def _create_named_dict(nodes: Union[Nucleotide,
                                    Sequence[Nucleotide],
                                    Dict[str, Nucleotide],
                                    None],
                       super_cls: type) -> dict:
    if nodes is None:
        return {}
    if isinstance(nodes, dict):
        nodes = list(nodes.values())
    elif not isinstance(nodes, (list, tuple)):
        assert isinstance(nodes, super_cls), (
            "Node with name {} is not instance of {}!!!".format(
                nodes.name, super_cls.__name__))
        nodes = [nodes]
    return {n.name: n for n in nodes}


def _is_composed_nucleotide(nucleotide: Union[Nucleotide, GeneHandler]):
    if isinstance(nucleotide, GeneHandler):
        return True
    return nucleotide.use_genes_as_inputs
