# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Main interface for DNA Helix
"""
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Set

import networkx as nx

from nucleus7.core.base import BaseClass
from nucleus7.core.nucleotide import Nucleotide
from nucleus7.utils import graph_utils
from nucleus7.utils import object_utils

DNAConnection = NamedTuple(
    "DNAConnection", [('incoming', Set[str]), ('outgoing', Set[str])])


class DNAHelix(BaseClass):
    """
    Main class for DNA Helix, which creates the graph and sorts it in
    topological order out of nucleotides. It also performs the graph checks
    so that generated nucleotide graph has no cycles and all the required
    incoming keys for all the nucleotides have inputs,

    Parameters
    ----------
    nucleotides
        list of nucleotides to build the helix
    incoming_nucleotides
        list nucleotides which serve as inputs to generated dna
    """

    def __init__(
            self,
            nucleotides: List[Nucleotide],
            incoming_nucleotides: Optional[List[Nucleotide]] = None):
        super(DNAHelix, self).__init__()
        self._nucleotides = nucleotides
        self._incoming_nucleotides = incoming_nucleotides or []
        self._topological_order = None  # type: Optional[List[Nucleotide]]
        self._graph = nx.DiGraph()

    @property
    def nucleotides(self) -> List[Nucleotide]:
        """
        Returns
        -------
        nucleotides
            nucleotides of the dna
        """
        return self._nucleotides

    @property
    def incoming_nucleotides(self) -> Optional[List[Nucleotide]]:
        """
        Returns
        -------
        incoming_nucleotides
            incoming nucleotide to DNA
        """
        return self._incoming_nucleotides

    @property
    def names_topological_order(self) -> Optional[List[str]]:
        """
        Nucleotide names in the topological sorted order

        Returns
        -------
        sorted_names
            sorted nucleotide names
        """
        if self._topological_order is None:
            return None
        return [nucleotide.name for nucleotide in self._topological_order]

    @property
    def topological_order(self) -> Optional[List[Nucleotide]]:
        """
        Nucleotides in the topological sorted order

        Returns
        -------
        sorted_nucleotides
            sorted nucleotides
        """
        return self._topological_order

    def get(self, with_incoming_nucleotides: bool = True) -> nx.DiGraph:
        """
        Parameters
        ----------
        with_incoming_nucleotides
            if the incoming_nucleotides should be included in the resulted graph

        Returns
        -------
        dna_graph
            graph representation of dna_helix
        """
        if with_incoming_nucleotides:
            return self._graph
        return self._graph.subgraph(self._nucleotides)

    def build(self):
        """
        Build the DNA

        Returns
        -------
        self
            self

        Raises
        ------
        ValueError
            if not all nucleotides are connected
        ValueError
            if nucleotides cannot be sorted in topological order, e.g. cycles
            exist
        """
        super(DNAHelix, self).build()
        self._graph = graph_utils.construct_graph_from_nucleotides(
            self._nucleotides, self._incoming_nucleotides)
        topological_order = graph_utils.topological_sort_of_nucleotides(
            self._graph)
        self._topological_order = [
            each_nucleotide for each_nucleotide in topological_order
            if each_nucleotide not in self._incoming_nucleotides]
        graph_utils.check_graph_connections(
            self._graph, self.incoming_nucleotides)
        return self

    def check_class_topology(
            self,
            nucleotide_type_dependency_map: Dict[Nucleotide, List[Nucleotide]]
    ) -> bool:
        """
        Check class dependency of nucleotides

        Parameters
        ----------
        nucleotide_type_dependency_map
            map describing allowed inbound node classes respect to the node,
            e.g. {:obj:`ModelPlugin`: [:obj:`ModelPlugin`]}

        Returns
        -------
        status
            True if check passed else raises :obj:`ValueError`

        Raises
        ------
        ValueError
            if some node has not allowed dependency
        """
        return graph_utils.check_class_topology(
            self._graph,
            dependency_map=nucleotide_type_dependency_map,
            incoming_nucleotides=self._incoming_nucleotides)

    @object_utils.assert_is_built
    def as_dict(self, with_incoming_nucleotides=True
                ) -> Dict[str, DNAConnection]:
        """
        Parameters
        ----------
        with_incoming_nucleotides
            if incoming nucleotides must be included

        Returns
        -------
        dna_as_dict
            dna helix as dict in structure
            `{nucleotide_name: (set(incoming nucleotide names),
            set(outgoing nucleotide names))}`
        """
        dna_as_dict = {}
        for each_nucleotide in self._graph:
            if (not with_incoming_nucleotides
                    and each_nucleotide in self._incoming_nucleotides):
                continue
            incoming_nucleotides = self._graph.predecessors(each_nucleotide)
            outgoing_nucleotides = self._graph.successors(each_nucleotide)
            incoming_names = {each_incoming.name
                              for each_incoming in incoming_nucleotides}
            outgoing_names = {each_outgoing.name
                              for each_outgoing in outgoing_nucleotides}
            dna_as_dict[each_nucleotide.name] = DNAConnection(
                incoming=incoming_names, outgoing=outgoing_names)
        return dna_as_dict

    @object_utils.assert_is_built
    def __add__(self, other: 'DNAHelix') -> 'DNAHelix':
        object_utils.assert_object_is_built(other)
        all_nucleotides = self.nucleotides + other.nucleotides
        all_incoming_nucleotides = (self.incoming_nucleotides
                                    + other.incoming_nucleotides)
        all_incoming_nucleotides_filtered = [
            each_nucleotide for each_nucleotide in all_incoming_nucleotides
            if each_nucleotide not in all_nucleotides]

        merged_dna = DNAHelix(
            nucleotides=all_nucleotides,
            incoming_nucleotides=all_incoming_nucleotides_filtered).build()
        return merged_dna

    @object_utils.assert_is_built
    def visualize(self, *, verbosity=0,
                  save_path: Optional[str] = None,
                  **draw_kwargs):
        """
        Visualize the DNA

        Draw interactive dna helix

        All nucleotides bodies are clickable; also the font is automatic
        rescalable to fit it to the circles / wedges.

        By single click on nucleotide, the nucleotide description pops up on the
        right side of image.

        By double click on nucleotide, new figure is opened and subgraph with
        all connections of this nucleotide is drawn in same fashion with
        verbosity=2.

        If save_path was provided, then figure will be saved to that path.

        Parameters
        ----------
        verbosity
            how verbose should be visualization
        save_path
            path to save the image
        **draw_kwargs
            additional kwargs to be passed to
            :obj:`nucleus7.utils.vis_utilsdraw_dna_helix`

        """
        # is needed here, otherwise will cause circular imports
        # TODO(oleksandr.vorobiov@audi.de): refactor to solve it
        from nucleus7.utils import vis_utils  # pylint: disable=cyclic-import

        subplot = vis_utils.draw_dna_helix(
            self.get(), verbosity=verbosity, **draw_kwargs)
        if save_path:
            figure = subplot.figure
            figure.savefig(save_path)

    def _get_incoming_nucleotide_names(self):
        incoming_nucleotides_names = [
            each_nucleotide.name
            for each_nucleotide in self._incoming_nucleotides]
        return incoming_nucleotides_names
