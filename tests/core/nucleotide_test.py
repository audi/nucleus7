# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

from unittest import TestCase

from nucleus7.core.nucleotide import Nucleotide


class TestNucleotide(TestCase):

    def test_keys(self):
        class NucleotideDummy(Nucleotide):
            incoming_keys = ['_in_optional1',
                             '_in_optional2',
                             'in_required1',
                             'in_required2']

            generated_keys = ['_gen_optional1',
                              '_gen_optional2',
                              'gen_required1',
                              'gen_required2']

        self.assertListEqual(NucleotideDummy.incoming_keys_optional,
                             ['in_optional1', 'in_optional2'])
        self.assertListEqual(NucleotideDummy.incoming_keys_required,
                             ['in_required1', 'in_required2'])
        self.assertListEqual(NucleotideDummy.incoming_keys_all,
                             ['in_optional1', 'in_optional2',
                              'in_required1', 'in_required2'])
        self.assertListEqual(NucleotideDummy.generated_keys_optional,
                             ['gen_optional1', 'gen_optional2'])
        self.assertListEqual(NucleotideDummy.generated_keys_required,
                             ['gen_required1', 'gen_required2'])
        self.assertListEqual(NucleotideDummy.generated_keys_all,
                             ['gen_optional1', 'gen_optional2',
                              'gen_required1', 'gen_required2'])

    def test_constructor(self):
        inbound_nodes = ['node1', 'node2']
        incoming_keys_mapping = {'node1': {"out1": "in1"}}
        _ = Nucleotide(inbound_nodes=inbound_nodes,
                       incoming_keys_mapping=incoming_keys_mapping)
        incoming_keys_mapping_wrong = {'node1': {"out1": "in1"},
                                       'node3': {"out1": "in2"}}
        with self.assertRaises(AttributeError):
            _ = Nucleotide(inbound_nodes=inbound_nodes,
                           incoming_keys_mapping=incoming_keys_mapping_wrong)

        inbound_nodes_with_mapping = {'node1': {"out1": "in1"},
                                      'node2': {"out2": "in2"}}
        nucleotide = Nucleotide(
            inbound_nodes=inbound_nodes_with_mapping)
        self.assertSetEqual(set(nucleotide.inbound_nodes),
                            set(inbound_nodes_with_mapping.keys()))
        self.assertDictEqual(nucleotide.incoming_keys_mapping,
                             inbound_nodes_with_mapping)
        self.assertEqual(nucleotide.name, nucleotide.__class__.__name__)

        nucleotide_name = "nucleotide_name"
        nucleotide_with_name = Nucleotide(inbound_nodes=inbound_nodes,
                                          name=nucleotide_name)
        self.assertEqual(nucleotide_with_name.name, nucleotide_name)

    def test_filter_inputs(self):
        mapping = {'node1': {'out11': 'inp11',
                             'out12': 'inp12'},
                   'node2': {'out22': 'inp22'}}
        nucleotide = Nucleotide(
            inbound_nodes=['node1', 'node2', 'node3'],
            incoming_keys_mapping=mapping)
        nucleotide.incoming_keys = ['inp11', 'inp12', 'inp22', 'out31']
        data = {'node1': {'out11': 'value11',
                          'out12': 'value12'},
                'node2': {'out21': 'value21',
                          'out22': 'value22'},
                'node3': {'out31': 'value31'},
                'node4': {'out41': 'value41'}}
        data_remapped_must = {'inp11': 'value11',
                              'inp12': 'value12',
                              'inp22': 'value22',
                              'out31': 'value31'}
        data_remapped = nucleotide.filter_inputs(data)
        self.assertDictEqual(data_remapped_must, data_remapped)

    def test_filter_inputs_nested_input_mapping(self):
        mapping = {'node1': {'out11:out111': 'inp11',
                             'out12': 'inp12'},
                   'node2': {'out22:2': 'inp22'}}
        nucleotide = Nucleotide(
            inbound_nodes=['node1', 'node2', 'node3'],
            incoming_keys_mapping=mapping)
        nucleotide.incoming_keys = ['inp11', 'inp12', 'inp22', 'out31']
        data = {'node1': {'out11': {'out111': 'value11',
                                    'out112': 'value112'},
                          'out12': 'value12'},
                'node2': {'out21': 'value21',
                          'out22': ['value220', 'value221',
                                    'value22', 'value223']},
                'node3': {'out31': 'value31'},
                'node4': {'out41': 'value41'}}
        data_remapped_must = {'inp11': 'value11',
                              'inp12': 'value12',
                              'inp22': 'value22',
                              'out31': 'value31'}
        data_remapped = nucleotide.filter_inputs(data)
        self.assertDictEqual(data_remapped_must, data_remapped)

    def test_filter_inputs_empty(self):
        mapping = {'node1': {'out11': 'inp11',
                             'out12': 'inp12'},
                   'node2': {'out22': 'inp22'}}
        nucleotide = Nucleotide(
            inbound_nodes=['node1', 'node2', 'node3'],
            incoming_keys_mapping=mapping)
        nucleotide.incoming_keys = ['inp11', 'inp12', 'inp22', 'out31']
        data = {'node1': {'out11': 'value11',
                          'out12': 'value12'},
                'node2': {'out21': 'value21',
                          'out22': 'value22'},
                'node3': None,
                'node4': {'out41': 'value41'}}
        self.assertIsNone(nucleotide.filter_inputs(data))

    def test_filter_inputs_dynamic_keys(self):
        mapping = {'node1': {'out11': 'inp11',
                             'out12': 'inp12'},
                   'node2': {'out22': 'inp22',
                             'out21': '_'}}
        nucleotide = Nucleotide(
            inbound_nodes=['node1', 'node2', 'node3'],
            incoming_keys_mapping=mapping)
        nucleotide.incoming_keys = ['inp11']
        nucleotide.dynamic_incoming_keys = True
        data = {'node1': {'out11': 'value11',
                          'out12': 'value12'},
                'node2': {'out21': 'value21',
                          'out22': 'value22',
                          'out': 'value23'},
                'node3': {'out31': 'value31',
                          'out32': 'value32',
                          'out': 'value33'},
                'node4': {'out41': 'value41'}}
        data_remapped_must = {'inp11': 'value11',
                              'inp12': 'value12',
                              'inp22': 'value22',
                              'out31': 'value31',
                              'out32': 'value32',
                              'out': ['value23', 'value33']}
        data_remapped = nucleotide.filter_inputs(data)
        self.assertDictEqual(data_remapped_must, data_remapped)
