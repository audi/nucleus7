# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

from absl.testing import parameterized

from nucleus7.core.dna_helix import DNAConnection
from nucleus7.core.dna_helix import DNAHelix
from nucleus7.core.nucleotide import Nucleotide


class _NucleotideType1(Nucleotide):
    pass


class _NucleotideType2(Nucleotide):
    pass


class _NucleotideType3(Nucleotide):
    pass


class _InputNucleotide(Nucleotide):
    pass


def _get_dna():
    nucleotide1 = Nucleotide(name="nucleotide1")
    nucleotide1.generated_keys = ["y"]

    nucleotide2 = Nucleotide(name="nucleotide2",
                             inbound_nodes=["nucleotide1"])
    nucleotide2.incoming_keys = ["y"]
    nucleotide2.generated_keys = ["z"]

    nucleotide3 = Nucleotide(name="nucleotide3",
                             inbound_nodes=["nucleotide1", "nucleotide2"])
    nucleotide3.dynamic_incoming_keys = True
    nucleotide3.generated_keys = ["w"]
    nucleotide3.dynamic_generated_keys = True

    nucleotide4 = Nucleotide(name="nucleotide4",
                             inbound_nodes=["nucleotide3"])
    nucleotide4.incoming_keys = ["y", "z", "w"]

    nucleotides_list = [nucleotide2, nucleotide3, nucleotide4]
    incoming_nucleotides_list = [nucleotide1]

    dna_must = {
        'nucleotide2': DNAConnection(incoming={'nucleotide1'},
                                     outgoing={'nucleotide3'}),
        'nucleotide3': DNAConnection(incoming={'nucleotide1', 'nucleotide2'},
                                     outgoing={'nucleotide4'}),
        'nucleotide4': DNAConnection(incoming={'nucleotide3'},
                                     outgoing=set()),
    }
    topological_sort_must = ["nucleotide2", "nucleotide3", "nucleotide4"]

    raise_build_error = False
    yield (nucleotides_list, incoming_nucleotides_list, dna_must,
           topological_sort_must, raise_build_error)

    nucleotide3 = Nucleotide(name="nucleotide3",
                             inbound_nodes=["nucleotide1", "nucleotide2"])
    nucleotide3.incoming_keys = []
    nucleotide3.generated_keys = ["w"]

    nucleotides_list = [nucleotide2, nucleotide3, nucleotide4]
    raise_build_error = True
    yield (nucleotides_list, incoming_nucleotides_list, dna_must,
           topological_sort_must, raise_build_error)

    nucleotides_list = [
        Nucleotide(name='first', inbound_nodes=['input_node1']),
        Nucleotide(name='second', inbound_nodes=['first', 'input_node2']),
        Nucleotide(name='fourth', inbound_nodes=['first', 'third']),
        Nucleotide(name='fifth', inbound_nodes=['fourth']),
        Nucleotide(name='third', inbound_nodes=['first', 'second'])
    ]
    incoming_nucleotides_list = [
        Nucleotide(name='input_node1'), Nucleotide(name='input_node2')]
    dna_must = {
        'first': DNAConnection(incoming={'input_node1'},
                               outgoing={'second', 'fourth', 'third'}),
        'second': DNAConnection(incoming={'first', 'input_node2'},
                                outgoing={'third'}),
        'third': DNAConnection(incoming={'first', 'second'},
                               outgoing={'fourth'}),
        'fourth': DNAConnection(incoming={'first', 'third'},
                                outgoing={'fifth'}),
        'fifth': DNAConnection(incoming={'fourth'}, outgoing=set())
    }
    topological_sort_must = ["first", "second", "third", "fourth", "fifth"]
    raise_build_error = False
    yield (nucleotides_list, incoming_nucleotides_list, dna_must,
           topological_sort_must, raise_build_error)

    nucleotides_list = [
        Nucleotide(name='first', inbound_nodes=['input_node1']),
        Nucleotide(name='second', inbound_nodes=[]),
        Nucleotide(name='fourth', inbound_nodes=['first', 'third']),
        Nucleotide(name='fifth', inbound_nodes=['fourth']),
        Nucleotide(name='third', inbound_nodes=['first', 'second'])
    ]

    incoming_nucleotides_list = [Nucleotide(name='input_node1')]
    dna_must = {
        'first': DNAConnection(incoming={'input_node1'},
                               outgoing={'fourth', 'third'}),
        'second': DNAConnection(incoming=set(),
                                outgoing={'third'}),
        'third': DNAConnection(incoming={'first', 'second'},
                               outgoing={'fourth'}),
        'fourth': DNAConnection(incoming={'first', 'third'},
                                outgoing={'fifth'}),
        'fifth': DNAConnection(incoming={'fourth'}, outgoing=set())
    }
    topological_sort_must = ["second", "first", "third", "fourth", "fifth"]
    raise_build_error = False
    yield (nucleotides_list, incoming_nucleotides_list, dna_must,
           topological_sort_must, raise_build_error)

    nucleotides_list = [
        Nucleotide(name='first', inbound_nodes=[]),
        Nucleotide(name='second', inbound_nodes=[]),
        Nucleotide(name='fourth', inbound_nodes=['first', 'third']),
        Nucleotide(name='fifth', inbound_nodes=['fourth']),
        Nucleotide(name='third', inbound_nodes=['first', 'second'])
    ]

    incoming_nucleotides_list = []
    dna_must = {
        'first': DNAConnection(incoming=set(),
                               outgoing={'fourth', 'third'}),
        'second': DNAConnection(incoming=set(),
                                outgoing={'third'}),
        'third': DNAConnection(incoming={'first', 'second'},
                               outgoing={'fourth'}),
        'fourth': DNAConnection(incoming={'first', 'third'},
                                outgoing={'fifth'}),
        'fifth': DNAConnection(incoming={'fourth'}, outgoing=set())
    }
    topological_sort_must = ["first", "second", "third", "fourth", "fifth"]
    raise_build_error = False
    yield (nucleotides_list, incoming_nucleotides_list, dna_must,
           topological_sort_must, raise_build_error)


class TestDNAHelix(parameterized.TestCase):

    def setUp(self):
        self.incoming_nucleotides = [_InputNucleotide(name='input')]

        self.nucleotide_type_dependency_map = {
            _NucleotideType1: [_NucleotideType1],
            _NucleotideType2: [_NucleotideType1, _NucleotideType2],
            _NucleotideType3: [_NucleotideType2, _NucleotideType3]}

    @parameterized.parameters(_get_dna())
    def test_build(self, nucleotides, incoming_nucleotides, dna_must,
                   topological_order_must, raise_build_error):
        dna_helix = DNAHelix(nucleotides, incoming_nucleotides)
        if raise_build_error:
            with self.assertRaises(ValueError):
                dna_helix.build()
            return

        dna_helix.build()
        self.assertDictEqual(dna_must,
                             dna_helix.as_dict(False))
        self.assertListEqual(topological_order_must,
                             dna_helix.names_topological_order)

    def test_check_class_topology_right(self):
        nucleotides = [
            _NucleotideType1(name='gene1_nucleotide1',
                             inbound_nodes=['input']),
            _NucleotideType1(name='gene1_nucleotide2',
                             inbound_nodes=[]),
            _NucleotideType1(name='gene1_nucleotide3',
                             inbound_nodes=[]),
            _NucleotideType2(name='gene2_nucleotide1',
                             inbound_nodes=['input',
                                            'gene1_nucleotide1',
                                            'gene1_nucleotide2']),
            _NucleotideType2(name='gene2_nucleotide2',
                             inbound_nodes=['gene2_nucleotide1',
                                            'gene1_nucleotide3']),
            Nucleotide(name='gene3_nucleotide1',
                       inbound_nodes=['gene2_nucleotide1']),
            Nucleotide(name='gene3_nucleotide2',
                       inbound_nodes=['gene2_nucleotide2',
                                      'gene3_nucleotide1'])
        ]
        dna_helix = DNAHelix(nucleotides, self.incoming_nucleotides)
        dna_helix.build()
        self.assertTrue(dna_helix.check_class_topology(
            self.nucleotide_type_dependency_map))

    def test_check_class_topology_wrong(self):
        nucleotides_wrong_type_dependency = [
            _NucleotideType1(name='gene1_nucleotide1',
                             inbound_nodes=['input']),
            _NucleotideType1(name='gene1_nucleotide2',
                             inbound_nodes=['input']),
            _NucleotideType1(name='gene1_nucleotide3',
                             inbound_nodes=['input']),
            _NucleotideType2(name='gene2_nucleotide1',
                             inbound_nodes=['gene1_nucleotide1',
                                            'gene1_nucleotide2']),
            _NucleotideType2(name='gene2_nucleotide2',
                             inbound_nodes=['gene2_nucleotide1',
                                            'gene1_nucleotide3']),
            _NucleotideType3(name='gene3_nucleotide1',
                             inbound_nodes=['gene2_nucleotide1']),
            _NucleotideType3(name='gene3_nucleotide2',
                             inbound_nodes=['gene1_nucleotide2',
                                            # not allowed
                                            'gene3_nucleotide1'])]

        dna_helix = DNAHelix(nucleotides_wrong_type_dependency,
                             self.incoming_nucleotides)
        dna_helix.build()
        with self.assertRaises(ValueError):
            self.assertTrue(dna_helix.check_class_topology(
                self.nucleotide_type_dependency_map))

    def test_add(self):
        nucleotide1 = Nucleotide(name="nucleotide1")
        nucleotide2 = Nucleotide(name="nucleotide2",
                                 inbound_nodes=["nucleotide1"])
        nucleotide3 = Nucleotide(name="nucleotide3")
        nucleotide4 = Nucleotide(name="nucleotide4",
                                 inbound_nodes=["nucleotide2", "nucleotide3"])

        nucleotide5 = Nucleotide(name="nucleotide5")
        nucleotide6 = Nucleotide(name="nucleotide6",
                                 inbound_nodes=["nucleotide4"])
        nucleotide7 = Nucleotide(
            name="nucleotide7",
            inbound_nodes=["nucleotide5", "nucleotide3", "nucleotide6"])

        incoming_nucleotides1 = [nucleotide1]
        incoming_nucleotides2 = [nucleotide3, nucleotide4, nucleotide5]
        nucleotides1 = [nucleotide2, nucleotide3, nucleotide4]
        nucleotides2 = [nucleotide6, nucleotide7]

        incoming_nucleotides12 = [nucleotide1, nucleotide5]
        nucleotides12 = [nucleotide2, nucleotide3, nucleotide4,
                         nucleotide6, nucleotide7]

        dna_helix1 = DNAHelix(nucleotides1, incoming_nucleotides1).build()
        dna_helix2 = DNAHelix(nucleotides2, incoming_nucleotides2).build()
        dna_helix12 = dna_helix1 + dna_helix2

        dna_helix12_must = DNAHelix(
            nucleotides12, incoming_nucleotides12).build()

        self.assertEqual(7,
                         len(dna_helix12_must.get()))
        self.assertEqual(5,
                         len(dna_helix12_must.get(False)))

        self.assertTrue(dna_helix12.built)
        self.assertDictEqual(dna_helix12_must.as_dict(),
                             dna_helix12.as_dict())
        self.assertSetEqual(set(dna_helix12_must.incoming_nucleotides),
                            set(dna_helix12.incoming_nucleotides))
        self.assertSetEqual(set(dna_helix12_must.nucleotides),
                            set(dna_helix12.nucleotides))
        self.assertListEqual(dna_helix12_must.topological_order,
                             dna_helix12.topological_order)

    def test_visualize(self):
        # TODO(oleksandr.vorobiov@audi.de): implement
        pass
