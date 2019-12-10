# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import copy
import unittest

from nucleus7.core.nucleotide import Nucleotide
from nucleus7.utils import graph_utils


class TestGraphUtils(unittest.TestCase):

    def test_construct_graph_from_nucleotides(self):
        nucleotides = [
            Nucleotide(name="input_node1"),
            Nucleotide(name="input_node2"),
            Nucleotide(name='first', inbound_nodes=['input_node1']),
            Nucleotide(name='second', inbound_nodes=['first', 'input_node2']),
            Nucleotide(name='fourth', inbound_nodes=['first', 'third']),
            Nucleotide(name='fifth', inbound_nodes=['fourth']),
            Nucleotide(name='third', inbound_nodes=['first', 'second']),
            Nucleotide(name='sixth')
        ]
        graph = graph_utils.construct_graph_from_nucleotides(nucleotides)

        edges_must = {(nucleotides[0], nucleotides[2]),
                      (nucleotides[1], nucleotides[3]),
                      (nucleotides[2], nucleotides[3]),
                      (nucleotides[2], nucleotides[4]),
                      (nucleotides[2], nucleotides[6]),
                      (nucleotides[3], nucleotides[6]),
                      (nucleotides[4], nucleotides[5]),
                      (nucleotides[6], nucleotides[4])}

        self.assertSetEqual(set(nucleotides),
                            set(graph.nodes))
        self.assertSetEqual(edges_must,
                            set(graph.edges))

    def test_construct_graph_from_nucleotides_with_cycle(self):
        nucleotides = [
            Nucleotide(name='first'),
            Nucleotide(name='second', inbound_nodes=['first']),
            Nucleotide(name='third',
                       inbound_nodes=['first', 'second', 'third']),
            Nucleotide(name='fourth', inbound_nodes=['third', 'second']),
        ]
        with self.assertRaises(ValueError):
            _ = graph_utils.construct_graph_from_nucleotides(nucleotides)

    def test_topological_sort_of_nucleotides(self):
        nucleotides = [
            Nucleotide(name='input_node'),
            Nucleotide(name='first', inbound_nodes=['input_node']),
            Nucleotide(name='second', inbound_nodes=['first']),
            Nucleotide(name='fourth', inbound_nodes=['first', 'third']),
            Nucleotide(name='fifth', inbound_nodes=['fourth']),
            Nucleotide(name='third', inbound_nodes=['first', 'second'])
        ]
        graph = graph_utils.construct_graph_from_nucleotides(nucleotides)
        nodes_sorted = graph_utils.topological_sort_of_nucleotides(graph)
        node_names_sorted_must = ['input_node', 'first', 'second', 'third',
                                  'fourth', 'fifth']
        node_names_sorted = [each_nucleotide.name
                             for each_nucleotide in nodes_sorted]
        self.assertListEqual(node_names_sorted_must,
                             node_names_sorted)

    def test_topological_sort_of_nucleotides_multiple_inputs(self):
        nucleotides = [
            Nucleotide(name='input_node'),
            Nucleotide(name='first', inbound_nodes=['input_node']),
            Nucleotide(name='second', inbound_nodes=[]),
            Nucleotide(name='fourth', inbound_nodes=['first', 'third']),
            Nucleotide(name='fifth', inbound_nodes=['fourth']),
            Nucleotide(name='third', inbound_nodes=[])
        ]
        graph = graph_utils.construct_graph_from_nucleotides(nucleotides)
        nodes_sorted = graph_utils.topological_sort_of_nucleotides(graph)

        self.assertGreater(nodes_sorted.index(nucleotides[1]),
                           nodes_sorted.index(nucleotides[0]))
        self.assertGreater(nodes_sorted.index(nucleotides[3]),
                           nodes_sorted.index(nucleotides[1]))
        self.assertGreater(nodes_sorted.index(nucleotides[3]),
                           nodes_sorted.index(nucleotides[5]))
        self.assertGreater(nodes_sorted.index(nucleotides[4]),
                           nodes_sorted.index(nucleotides[3]))
        node_names_sorted_must = ['input_node', 'second', 'third',
                                  'first', 'fourth', 'fifth']
        node_names_sorted = [each_nucleotide.name
                             for each_nucleotide in nodes_sorted]
        self.assertListEqual(node_names_sorted_must,
                             node_names_sorted)

    def test_check_class_topology(self):
        class Node1(Nucleotide):
            def __init__(self, **kwargs):
                super(Node1, self).__init__(**kwargs)

        class Node2(Nucleotide):
            def __init__(self, **kwargs):
                super(Node2, self).__init__(**kwargs)

        class Node3(Nucleotide):
            def __init__(self, **kwargs):
                super(Node3, self).__init__(**kwargs)

        dependency_map = {Node1: [Node1],
                          Node2: [Node1, Node2],
                          Node3: [Node2, Node3]}

        nucleotides_right = [Nucleotide(name='input_node', inbound_nodes=[]),
                             Node1(name='n1', inbound_nodes=['input_node']),
                             Node1(name='n2', inbound_nodes=['n1']),
                             Node1(name='n3', inbound_nodes=['n2']),
                             Node2(name='n4', inbound_nodes=['n1']),
                             Node2(name='n5', inbound_nodes=['n2']),
                             Node2(name='n6', inbound_nodes=['n4', 'n5']),
                             Node2(name='n7', inbound_nodes=['n6'])]
        incoming_nucleotides = [nucleotides_right[0]]
        nucleotides_wrong = copy.deepcopy(nucleotides_right)
        nucleotides_wrong[2] = Node2(name='n2', inbound_nodes=['n1'])
        nucleotides_all = {'right': nucleotides_right,
                           'wrong': nucleotides_wrong}

        for t, nodes in nucleotides_all.items():
            graph = graph_utils.construct_graph_from_nucleotides(nodes)
            if t == 'right':
                self.assertTrue(graph_utils.check_class_topology(
                    graph, dependency_map, incoming_nucleotides))
            else:
                with self.assertRaises(ValueError):
                    graph_utils.check_class_topology(
                        graph, dependency_map, incoming_nucleotides)

    def test_check_node2node_connection(self):
        dataset_node = Nucleotide(name='data', inbound_nodes=[])
        dataset_node.generated_keys = ['image', 'labels', 'temp']

        node_cnn1 = Nucleotide(
            name='cnn', inbound_nodes=['data'],
            incoming_keys_mapping={'data': {'image': 'inputs_cnn',
                                            'temp': 'inputs_optional'}})
        node_cnn1.incoming_keys = ['inputs_cnn', '_inputs_optional',
                                   '_inputs_optional_2']
        node_cnn1.generated_keys = ['predictions']

        node_loss = Nucleotide(
            name='loss', inbound_nodes=['data', 'cnn'],
            incoming_keys_mapping={'cnn': {'predictions:0': 'logits:0'}})
        node_loss.incoming_keys = ['logits', 'labels']

        node_loss_wrong = Nucleotide(
            name='loss', inbound_nodes=['data', 'cnn'],
            incoming_keys_mapping={'cnn': {'predictions': 'logits2'}})
        node_loss_wrong.incoming_keys = ['logits', 'labels']

        nodes_all = {'right': [dataset_node, node_cnn1, node_loss],
                     'wrong': [dataset_node, node_cnn1, node_loss_wrong]}
        for t, nodes in nodes_all.items():
            nodes = {n.name: n for n in nodes}
            if t == 'right':
                self.assertTrue(graph_utils.check_node2node_connection(
                    nodes['loss'], [nodes[k] for k in ['data', 'cnn']]))
            else:
                with self.assertRaises(ValueError):
                    graph_utils.check_node2node_connection(
                        nodes['loss'], [nodes[k] for k in ['data', 'cnn']])

    def test_check_graph_connections(self):
        dataset_node = Nucleotide(name='data', inbound_nodes=[])
        dataset_node.generated_keys = ['image', 'labels', 'temp']

        node_cnn = Nucleotide(
            name='cnn', inbound_nodes=['data'],
            incoming_keys_mapping={'data': {'image': 'inputs_cnn'}})
        node_cnn.incoming_keys = ['inputs_cnn']
        node_cnn.generated_keys = ['predictions']

        node_flatten = Nucleotide(
            name='flatten', inbound_nodes=['cnn'],
            incoming_keys_mapping={'cnn': {'predictions': 'inputs_flatten'}})
        node_flatten.incoming_keys = ['inputs_flatten']
        node_flatten.generated_keys = ['predictions']

        node_mlp = Nucleotide(
            name='mlp', inbound_nodes=['flatten'],
            incoming_keys_mapping={'flatten': {'predictions': 'inputs_mlp'}})
        node_mlp.incoming_keys = ['inputs_mlp']
        node_mlp.generated_keys = ['predictions']

        node_mlp_wrong = Nucleotide(
            name='mlp', inbound_nodes=['flatten'],
            incoming_keys_mapping={'flatten': {'predictions': 'inputs'}})
        node_mlp_wrong.incoming_keys = ['inputs_mlp']
        node_mlp_wrong.generated_keys = ['predictions']

        node_loss = Nucleotide(
            name='loss', inbound_nodes=['data', 'mlp'],
            incoming_keys_mapping={'mlp': {'predictions': 'logits'}})
        node_loss.incoming_keys = ['labels', 'logits']

        node_loss_wrong = Nucleotide(
            name='loss', inbound_nodes=['data', 'mlp'])
        node_loss_wrong.incoming_keys = ['labels', 'logits']

        node_pp = Nucleotide(
            name='pp', inbound_nodes=['mlp'],
            incoming_keys_mapping={'mlp': {'predictions': 'inputs_pp'}})
        node_pp.incoming_keys = ['inputs_pp']

        nodes_all = {'right': [[dataset_node, node_cnn, node_flatten, node_mlp,
                                node_loss, node_pp]],
                     'wrong': [[dataset_node, node_cnn, node_flatten,
                                node_mlp_wrong, node_loss, node_pp],
                               [dataset_node, node_cnn, node_flatten,
                                node_mlp_wrong, node_loss_wrong, node_pp]]}

        for test_mode, nodes_case in nodes_all.items():
            for nodes in nodes_case:
                graph = graph_utils.construct_graph_from_nucleotides(nodes)
                if test_mode == 'right':
                    self.assertTrue(graph_utils.check_graph_connections(
                        graph))
                else:
                    with self.assertRaises(ValueError):
                        graph_utils.check_graph_connections(graph)

    def test_get_repeated_node_names(self):
        unique_node_names = ['first', 'second', 'third']
        repeated_node_names = ['first', 'second', 'third', 'second']
        nodes_unique_names = [Nucleotide([], name=name)
                              for name in unique_node_names]
        nodes_repeated_names = [Nucleotide([], name=name)
                                for name in repeated_node_names]
        self.assertDictEqual(graph_utils.get_repeated_node_names(
            nodes_unique_names), {})
        self.assertDictEqual(graph_utils.get_repeated_node_names(
            nodes_repeated_names), {'second': [Nucleotide.__name__] * 2})
